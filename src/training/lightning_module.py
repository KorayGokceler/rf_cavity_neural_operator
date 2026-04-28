import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from src.models.gnot import GNOTModel

class GNOTLightning(pl.LightningModule):
    def __init__(self, val_dim=6, grid_dim=2, theta_dim=1, hidden_dim=256, 
                 n_shared_layers=2, n_mode_layers=2, n_field_head_layers=2,
                 n_heads=4, num_experts=4, num_field_modes=3,
                 lr=1e-3, freq_weight=0.5, smoothness_weight=0.1,
                 mode_loss_weights=None,
                 lr_mode_specific=None, lr_freq_heads=None,
                 scheduler='onecycle', weight_decay=1e-4, use_checkpoint=False,
                 reducelr_patience=10, reducelr_factor=0.5,
                 rff_scale=1.0, use_rff=True, predict_frequency=True,
                 onecycle_pct_start=0.3, onecycle_div_factor=25, onecycle_final_div_factor=1e4,
                 cosine_eta_min=1e-6,
                 gradient_clip_val=None):
        super().__init__()
        # Suppress harmless DDP + gradient checkpointing stream mismatch warning
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        self.lr_mode_specific = lr_mode_specific
        self.lr_freq_heads = lr_freq_heads
        self.gradient_clip_val = gradient_clip_val
        
        self.save_hyperparameters()
        
        # --- GRADNORM INITIALIZATION ---
        self.automatic_optimization = False  # GradNorm requires manual control
        self.gradnorm_alpha = 1.5             # Dengeleme sertliği (1.0-2.0)
        self.num_tasks = num_field_modes + 1  # Field (per mode) + Frequency
        self.register_buffer('loss_weights', torch.ones(self.num_tasks))
        self.register_buffer('initial_losses', torch.zeros(self.num_tasks))
        self.has_initial_losses = False
        # -------------------------------
        
        self.model = GNOTModel(
            val_dim=val_dim,
            grid_dim=grid_dim,
            theta_dim=theta_dim,
            embed_dim=hidden_dim,
            n_shared_layers=n_shared_layers,
            n_mode_layers=n_mode_layers,
            n_field_head_layers=n_field_head_layers,
            n_heads=n_heads,
            num_experts=num_experts,
            num_field_modes=num_field_modes,
            rff_scale=rff_scale,
            use_rff=use_rff,
            use_checkpoint=use_checkpoint,
            predict_frequency=predict_frequency
        )
        self.freq_weight = freq_weight
        self.smoothness_weight = smoothness_weight
        self.predict_frequency = predict_frequency
        # Per-mode loss weights: [w0, w1, w2] — zayıf modlara daha yüksek ağırlık verilebilir
        if mode_loss_weights is None:
            self.mode_loss_weights = [1.0] * num_field_modes
        else:
            self.mode_loss_weights = list(mode_loss_weights)
        self.freq_stats = None

        # Optional: Metrics to evaluate and measure model's success
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()
        
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

    def forward(self, batch):
        return self.model(batch)

    def _compute_loss(self, batch, prefix):
        outputs = self.model(batch)
        mask = batch.get('Mask', None)  # [B, N] boolean

        pred_field = outputs['field']      # [B, N, 1]
        true_field = batch['Y_field']      # [B, N, 1]
        B = pred_field.shape[0]

        # --- PHYSICS-INFORMED NEURAL NETWORK (PINN) BOUNDARY CONSTRAINT ---
        dist_bnd = batch['Input_funcs'][:, :, 2]  # [B, N]
        bnd_mask = (dist_bnd < 1e-4).unsqueeze(-1).expand_as(pred_field) # [B, N, 1]
        
        if bnd_mask.any():
            loss_bnd = (pred_field[bnd_mask] ** 2).mean()
        else:
            loss_bnd = torch.tensor(0.0, device=pred_field.device)
            
        self.log(f'{prefix}/loss_bnd', loss_bnd, prog_bar=True, batch_size=B, sync_dist=True)
        
        # Hard Constraint
        pred_field = pred_field * (~bnd_mask).float()
        # ------------------------------------------------------------------

        # --- PHASE (SIGN) REALIGNMENT ---
        with torch.no_grad():
            if mask is not None:
                m_f = mask.unsqueeze(-1).expand_as(pred_field).float()
                diff_pos = ((pred_field - true_field) ** 2 * m_f).sum(dim=1)  # [B, 1]
                diff_neg = ((pred_field + true_field) ** 2 * m_f).sum(dim=1)  # [B, 1]
            else:
                diff_pos = ((pred_field - true_field) ** 2).mean(dim=1)  # [B, 1]
                diff_neg = ((pred_field + true_field) ** 2).mean(dim=1)  # [B, 1]
            
            signs = torch.where(diff_pos <= diff_neg, 1.0, -1.0).unsqueeze(1) # [B, 1, 1]
        
        aligned_true_field = true_field * signs
        # ------------------------------------------------------------------

        # --- FIELD LOSS (Peak-Weighted Hybrid MSE + L1) ---
        if mask is not None:
            m = mask.float()
            n_v = m.sum(dim=1).clamp(min=1.0)  # [B]
            w = 1.0 + 5.0 * aligned_true_field.abs().squeeze(-1)  # [B, N]
            diff = (pred_field.squeeze(-1) - aligned_true_field.squeeze(-1))  # [B, N]
            mse = ((diff ** 2) * w * m).sum(dim=1) / n_v  # [B]
            l1 = (diff.abs() * m).sum(dim=1) / n_v  # [B]
            loss_field = (mse + 0.1 * l1).mean()
        else:
            w = 1.0 + 5.0 * aligned_true_field.abs().squeeze(-1)
            diff = (pred_field.squeeze(-1) - aligned_true_field.squeeze(-1))
            mse = ((diff ** 2) * w).mean(dim=1)
            l1 = diff.abs().mean(dim=1)
            loss_field = (mse + 0.1 * l1).mean()

        # Per-mode loss logging
        theta_in = batch['Theta_in'][:, 0]  # [B]
        for mode_val in range(3):
            mode_mask_sel = (theta_in == mode_val)
            if mode_mask_sel.any():
                if mask is not None:
                    m_sel = mask[mode_mask_sel].float()
                    n_v_sel = m_sel.sum(dim=1).clamp(min=1.0)
                    d = (pred_field[mode_mask_sel].squeeze(-1) - aligned_true_field[mode_mask_sel].squeeze(-1))
                    mode_loss = ((d ** 2) * m_sel).sum(dim=1) / n_v_sel
                else:
                    d = (pred_field[mode_mask_sel].squeeze(-1) - aligned_true_field[mode_mask_sel].squeeze(-1))
                    mode_loss = (d ** 2).mean(dim=1)
                self.log(f'{prefix}/mode_{mode_val}_loss', mode_loss.mean(), on_step=False, on_epoch=True, prog_bar=False, batch_size=B)

        # Frequency loss
        if self.predict_frequency and outputs.get('freq') is not None:
            freq_pred = outputs['freq']          # [B, 1]
            freq_true = batch['Y_freq']          # [B, 1]
            loss_freq = F.mse_loss(freq_pred, freq_true)
            self.log(f'{prefix}/freq_loss', loss_freq, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)
            
            if self.freq_stats:
                freq_pred_ghz = freq_pred * self.freq_stats['std'] + self.freq_stats['mean']
                freq_true_ghz = freq_true * self.freq_stats['std'] + self.freq_stats['mean']
                mae_ghz = F.l1_loss(freq_pred_ghz, freq_true_ghz)
                self.log(f'{prefix}/freq_mae_ghz', mae_ghz, on_step=False, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        else:
            loss_freq = torch.tensor(0.0, device=pred_field.device)

        # Mesh-aware gradient matching smoothness loss
        if self.smoothness_weight > 0:
            elements_list = batch.get('elements', None)
            if elements_list is not None and len(elements_list) > 0:
                loss_smooth = self._compute_smoothness_loss(pred_field, aligned_true_field, elements_list)
            else:
                loss_smooth = torch.tensor(0.0, device=pred_field.device)
        else:
            loss_smooth = torch.tensor(0.0, device=pred_field.device)
        self.log(f'{prefix}/smooth_loss', loss_smooth, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)

        # Total Loss
        total_loss = loss_field + (self.freq_weight * loss_freq) + (1.0 * loss_bnd) + (self.smoothness_weight * loss_smooth)

        self.log(f'{prefix}/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self.log(f'{prefix}/field_loss', loss_field, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)

        # Relative L2 Error
        if mask is not None:
            m_f = mask.unsqueeze(-1).float()
            diff_sq = ((pred_field - aligned_true_field) ** 2 * m_f).sum(dim=1)  # [B, 1]
            true_sq = ((aligned_true_field) ** 2 * m_f).sum(dim=1)  # [B, 1]
        else:
            diff_sq = ((pred_field - aligned_true_field) ** 2).sum(dim=1)
            true_sq = ((aligned_true_field) ** 2).sum(dim=1)
            
        rel_l2_all = torch.sqrt(diff_sq) / (torch.sqrt(true_sq) + 1e-8)  # [B, 1]
        rel_l2 = rel_l2_all.mean()
        self.log(f'{prefix}/field_rel_l2', rel_l2, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)

        # Per-mode Relative L2
        for mode_val in range(3):
            mode_mask_sel = (theta_in == mode_val)
            if mode_mask_sel.any():
                mode_rel = rel_l2_all[mode_mask_sel].mean()
                self.log(f'{prefix}/mode_{mode_val}_rel_l2', mode_rel, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)

        # Tensors for torchmetrics (R2, MAE)
        if mask is not None:
            mask_flat = mask.unsqueeze(-1).expand_as(pred_field)
            preds_valid = pred_field[mask_flat].contiguous()
            targets_valid = aligned_true_field[mask_flat].contiguous()
        else:
            preds_valid = pred_field.contiguous().view(-1)
            targets_valid = aligned_true_field.contiguous().view(-1)

        return total_loss, preds_valid, targets_valid

    def _compute_smoothness_loss(self, pred_field, aligned_true_field, elements_list):
        """Mesh-aware gradient matching loss using triangle edge connectivity.
        
        Penalizes the difference between predicted and true spatial gradients
        across mesh edges. Forces spatially coherent outputs without suppressing
        legitimate field variations. Cost: ~5-10% overhead (no autograd needed).
        """
        total_loss = 0.0
        count = 0
        B = pred_field.shape[0]
        
        for b in range(B):
            elems = elements_list[b].to(pred_field.device)  # [num_elements, 3]
            if elems.numel() == 0:
                continue
            
            pred_b = pred_field[b, :, 0]   # [N]
            true_b = aligned_true_field[b, :, 0]  # [N]
            
            # 3 edges per triangle: (v0,v1), (v1,v2), (v0,v2)
            e0, e1, e2 = elems[:, 0], elems[:, 1], elems[:, 2]
            
            # Predicted gradients across edges
            dp_01 = pred_b[e0] - pred_b[e1]
            dp_12 = pred_b[e1] - pred_b[e2]
            dp_02 = pred_b[e0] - pred_b[e2]
            
            # True gradients across edges
            dt_01 = true_b[e0] - true_b[e1]
            dt_12 = true_b[e1] - true_b[e2]
            dt_02 = true_b[e0] - true_b[e2]
            
            # Gradient matching: penalize excess/missing spatial gradients
            grad_loss = ((dp_01 - dt_01)**2 + (dp_12 - dt_12)**2 + (dp_02 - dt_02)**2).mean()
            
            total_loss += grad_loss
            count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=pred_field.device)
        return total_loss / count

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        
        # 1. Forward Pass
        outputs = self.model(batch)
        mask = batch.get('Mask', None)
        theta_in = batch['Theta_in'][:, 0]
        B = theta_in.shape[0]

        # 2. Individual Task Losses Calculation
        # We split field loss per mode to balance them individually via GradNorm
        task_losses = []
        
        pred_field = outputs['field']
        true_field = batch['Y_field']
        
        # Phase (Sign) Alignment
        with torch.no_grad():
            if mask is not None:
                m_f = mask.unsqueeze(-1).expand_as(pred_field).float()
                diff_pos = ((pred_field - true_field) ** 2 * m_f).sum(dim=1)
                diff_neg = ((pred_field + true_field) ** 2 * m_f).sum(dim=1)
            else:
                diff_pos = ((pred_field - true_field) ** 2).mean(dim=1)
                diff_neg = ((pred_field + true_field) ** 2).mean(dim=1)
            signs = torch.where(diff_pos <= diff_neg, 1.0, -1.0).unsqueeze(1)
        aligned_true_field = true_field * signs

        # Per-mode field losses (Tasks 0 to num_modes-1)
        for mode_val in range(self.hparams.num_field_modes):
            mode_mask_sel = (theta_in == mode_val)
            if mode_mask_sel.any():
                m_sel = mask[mode_mask_sel].float() if mask is not None else None
                p_sel = pred_field[mode_mask_sel].squeeze(-1)
                t_sel = aligned_true_field[mode_mask_sel].squeeze(-1)
                
                if m_sel is not None:
                    n_v_sel = m_sel.sum(dim=1).clamp(min=1.0)
                    # Use standard MSE for gradnorm stability (unweighted by peak)
                    loss_m = (((p_sel - t_sel) ** 2) * m_sel).sum(dim=1) / n_v_sel
                else:
                    loss_m = ((p_sel - t_sel) ** 2).mean(dim=1)
                task_losses.append(loss_m.mean())
            else:
                # Handle missing modes in batch with a zero that still tracks grads
                task_losses.append(torch.tensor(0.0, device=self.device, requires_grad=True))

        # Frequency Loss (Task N)
        if self.predict_frequency and outputs.get('freq') is not None:
            task_losses.append(F.mse_loss(outputs['freq'], batch['Y_freq']))
        else:
            task_losses.append(torch.tensor(0.0, device=self.device, requires_grad=True))

        task_losses = torch.stack(task_losses) # [num_tasks]

        # 3. Save Initial Losses (L0) on first valid batch
        if not self.has_initial_losses:
            if (task_losses.detach() > 0).all():
                self.initial_losses.copy_(task_losses.detach())
                self.has_initial_losses = True

        # 4. Weighted Total Loss
        weighted_loss = (task_losses * self.loss_weights).sum()
        
        # Add Boundary and Smoothness (fixed weights, not part of gradnorm)
        dist_bnd = batch['Input_funcs'][:, :, 2]
        bnd_mask = (dist_bnd < 1e-4).unsqueeze(-1).expand_as(pred_field)
        loss_bnd = (pred_field[bnd_mask] ** 2).mean() if bnd_mask.any() else torch.tensor(0.0, device=self.device)
        
        elements_list = batch.get('elements', None)
        loss_smooth = self._compute_smoothness_loss(pred_field, aligned_true_field, elements_list) if (self.smoothness_weight > 0 and elements_list) else torch.tensor(0.0, device=self.device)
        
        total_loss = weighted_loss + (1.0 * loss_bnd) + (self.smoothness_weight * loss_smooth)

        # 5. Optimization Step (Manual)
        optimizer.zero_grad()
        
        # 6. GRADNORM UPDATE: Ana backward'dan önce yapılmalı ki grafik silinmesin.
        if self.has_initial_losses:
            self._update_loss_weights(task_losses)
            
        self.manual_backward(total_loss)
            
        # 6.5 Manual Gradient Clipping
        if self.gradient_clip_val is not None:
            self.clip_gradients(
                optimizer, 
                gradient_clip_val=self.gradient_clip_val, 
                gradient_clip_algorithm="norm"
            )
            
        optimizer.step()
        
        # 7. Scheduler Step (Required for manual optimization)
        sch = self.lr_schedulers()
        if sch is not None:
            # Step every iteration if using OneCycle or step-based schedulers
            if self.trainer.lr_scheduler_configs[0].interval == 'step':
                sch.step()
            elif self.trainer.is_last_batch and self.trainer.lr_scheduler_configs[0].interval == 'epoch':
                sch.step()
        
        # Logging
        self.log('train/loss', total_loss, prog_bar=True, batch_size=B)
        self.log('train/weighted_loss', weighted_loss, batch_size=B)
        for i, w in enumerate(self.loss_weights):
            self.log(f'gradnorm/weight_{i}', w, batch_size=B)
            self.log(f'train/task_{i}_loss', task_losses[i], batch_size=B)

        return total_loss

    def _get_shared_layer(self):
        """Reference layer for GradNorm: Last linear layer of the shared trunk."""
        return self.model.shared_blocks[-1].ffn.experts[0][3]

    def _update_loss_weights(self, task_losses):
        """Dynamic Loss Weighting via GradNorm."""
        W = self._get_shared_layer().weight
        
        norms = []
        for i in range(self.num_tasks):
            # Gradient of L_i w.r.t. W
            grad = torch.autograd.grad(task_losses[i], W, retain_graph=True, allow_unused=True)[0]
            if grad is not None:
                # G_i = || w_i * grad(L_i) ||
                norms.append(torch.norm(self.loss_weights[i] * grad, p=2))
            else:
                norms.append(torch.tensor(0.0, device=self.device))
        
        norms = torch.stack(norms)
        
        # Average gradient norm
        avg_norm = norms.mean().detach()
        
        # Relative loss (inverse training rate)
        # rel_loss = (L_i / L_i_0) / avg(L_j / L_j_0)
        rel_loss = (task_losses.detach() / (self.initial_losses + 1e-8))
        avg_rel_loss = rel_loss.mean()
        inv_train_rate = rel_loss / (avg_rel_loss + 1e-8)
        
        # Target norm: E[G] * (inv_rate ^ alpha)
        target_norms = (avg_norm * (inv_train_rate ** self.gradnorm_alpha)).detach()
        
        # GradNorm Loss: L1 distance between actual and target norms
        gradnorm_loss = F.l1_loss(norms, target_norms)
        
        # Update weights (Separate from main optimizer)
        self.loss_weights.requires_grad = True
        if self.loss_weights.grad is not None:
            self.loss_weights.grad.zero_()
            
        gradnorm_loss.backward()
        
        with torch.no_grad():
            # Update step (lr=0.025 is standard for GradNorm)
            self.loss_weights -= 0.025 * self.loss_weights.grad
            
            # Constraints: clamp and re-normalize to sum to num_tasks
            self.loss_weights.clamp_(min=0.01)
            self.loss_weights *= (self.num_tasks / self.loss_weights.sum())
            
        self.loss_weights.requires_grad = False
        self.loss_weights.grad = None

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch, "val")
        self.val_r2(preds, targets)
        self.val_mae(preds, targets)
        self.log('val/r2', self.val_r2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/mae', self.val_mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        if hasattr(self.model, 'reset_expert_calls'):
            self.model.reset_expert_calls()

    def on_validation_epoch_end(self):
        # Only print on rank 0 to avoid duplicate output in DDP
        if self.global_rank != 0:
            return

        # Log Expert Load Balancing per block
        if hasattr(self.model, 'get_expert_calls_per_block'):
            calls_dict = self.model.get_expert_calls_per_block()
            for block_name, calls in calls_dict.items():
                calls = calls.float()
                total_calls = calls.sum()
                if total_calls > 0:
                    percentages = (calls / total_calls) * 100
                    for i, p in enumerate(percentages):
                        self.logger.experiment.add_scalar(f"Experts_{block_name}/E{i}", p, self.current_epoch)
        
        metrics = self.trainer.logged_metrics
        epoch = self.trainer.current_epoch

        # Collect mode-specific errors for both Train and Val
        # We use a dict of {mode_idx: value}
        train_errors = {
            k.split('mode_')[1].split('_')[0]: v.item() if torch.is_tensor(v) else v
            for k, v in metrics.items() if k.startswith('train/mode_') and k.endswith('_rel_l2')
        }
        val_errors = {
            k.split('mode_')[1].split('_')[0]: v.item() if torch.is_tensor(v) else v
            for k, v in metrics.items() if k.startswith('val/mode_') and k.endswith('_rel_l2')
        }

        mode_indices = sorted(list(set(train_errors.keys()) | set(val_errors.keys())))
        if not mode_indices:
            return

        # Header
        print(f"\n{'━'*64}")
        print(f"  Epoch {epoch:>3d} │ Mode-Specific Relative L2 Field Error")
        print(f"{'─'*64}")
        print(f"  Mode      │   Train Rel   │    Val Rel    │ Progress")
        print(f"{'─'*64}")

        for m_idx in mode_indices:
            t_val = train_errors.get(m_idx, 0.0)
            v_val = val_errors.get(m_idx, 0.0)
            
            # Status indicators
            status = '✅' if v_val < 0.1 else ('⚠️ ' if v_val < 0.3 else '❌')
            if v_val == 0: status = '??'

            # Mini bar chart for Val error
            bar_len = int(min(v_val * 15, 15))
            bar = '█' * bar_len + '░' * (15 - bar_len)
            
            t_str = f"{t_val:10.4f}" if m_idx in train_errors else "   -      "
            v_str = f"{v_val:10.4f}" if m_idx in val_errors else "   -      "
            
            print(f"  Mode {m_idx:2s}   │  {t_str}   │  {v_str}   │ [{bar}] {status}")

        # Global Summary
        print(f"{'─'*64}")
        # Try both direct name and _epoch suffix (Lightning adds _epoch when on_step=True)
        global_val_l2 = metrics.get('val/field_rel_l2_epoch', metrics.get('val/field_rel_l2', 0.0))
        global_val_r2 = metrics.get('val/r2_epoch', metrics.get('val/r2', 0.0))
        
        print(f"  GLOBAL VAL  │  Rel L2: {global_val_l2:.4f}  │  R²: {global_val_r2:.4f}")
        print(f"{'━'*64}\n")

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch, "test")
        self.test_r2(preds, targets)
        self.test_mae(preds, targets)
        self.log('test/r2', self.test_r2, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/mae', self.test_mae, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        base_lr = self.hparams.lr
        param_groups = []
        handled_param_ids = set()

        # Mode Specific LRs
        if self.lr_mode_specific is not None:
            for mode_idx, mode_lr in enumerate(self.lr_mode_specific):
                if mode_lr is not None and mode_idx < len(self.model.mode_field_blocks):
                    # Gather parameters for this mode's blocks and field head
                    mode_params = []
                    # Mode Field Blocks
                    for p in self.model.mode_field_blocks[mode_idx].parameters():
                        mode_params.append(p)
                        handled_param_ids.add(id(p))
                    # Mode Field Heads
                    for p in self.model.field_heads[mode_idx].parameters():
                        mode_params.append(p)
                        handled_param_ids.add(id(p))
                    
                    if len(mode_params) > 0:
                        param_groups.append({"params": mode_params, "lr": mode_lr})
                        print(f"Optimizer: Mode {mode_idx} branch trained with lr={mode_lr}")

        # Freq Heads LR
        if self.lr_freq_heads is not None and self.predict_frequency:
            freq_params = []
            for p in self.model.freq_heads.parameters():
                if id(p) not in handled_param_ids:
                    freq_params.append(p)
                    handled_param_ids.add(id(p))
            if len(freq_params) > 0:
                param_groups.append({"params": freq_params, "lr": self.lr_freq_heads})
                print(f"Optimizer: Frequency heads trained with lr={self.lr_freq_heads}")

        # Base param group (General trunk, embeddings, RFF, unhandled parts)
        base_params = []
        for p in self.parameters():
            if id(p) not in handled_param_ids:
                base_params.append(p)
        
        if len(base_params) > 0:
            param_groups.append({"params": base_params, "lr": base_lr})
            print(f"Optimizer: Shared trunk & remaining params trained with lr={base_lr}")

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        
        if self.hparams.scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=self.hparams.lr, 
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.hparams.onecycle_pct_start,
                div_factor=self.hparams.onecycle_div_factor,
                final_div_factor=self.hparams.onecycle_final_div_factor
            )
            return {
                "optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler, 
                    "interval": "step"
                }
            }
        elif self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs, 
                eta_min=self.hparams.cosine_eta_min
            )
            return {
                "optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler, 
                    "interval": "epoch"
                }
            }
        elif self.hparams.scheduler == 'reducelr':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.hparams.reducelr_factor,
                patience=self.hparams.reducelr_patience,
                min_lr=1e-7
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/field_rel_l2",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            return optimizer
