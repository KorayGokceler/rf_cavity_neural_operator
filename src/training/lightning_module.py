import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from src.models.gnot import GNOTModel

class GNOTLightning(pl.LightningModule):
    def __init__(self, val_dim=6, grid_dim=2, theta_dim=1, hidden_dim=256, 
                 n_shared_layers=2, n_mode_layers=2,
                 n_heads=4, num_experts=4, num_field_modes=3,
                 lr=1e-3, freq_weight=0.5, ortho_weight=0.01,
                 mode_loss_weights=None,
                 lr_mode_specific=None, lr_freq_heads=None,
                 scheduler='onecycle', weight_decay=1e-4, use_checkpoint=False,
                 reducelr_patience=10, reducelr_factor=0.5,
                 rff_scale=1.0, use_rff=True, predict_frequency=True,
                 onecycle_pct_start=0.3, onecycle_div_factor=25, onecycle_final_div_factor=1e4,
                 cosine_eta_min=1e-6):
        super().__init__()
        # Suppress harmless DDP + gradient checkpointing stream mismatch warning
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        self.lr_mode_specific = lr_mode_specific
        self.lr_freq_heads = lr_freq_heads
        
        self.save_hyperparameters()
        self.model = GNOTModel(
            val_dim=val_dim,
            grid_dim=grid_dim,
            theta_dim=theta_dim,
            embed_dim=hidden_dim,
            n_shared_layers=n_shared_layers,
            n_mode_layers=n_mode_layers,
            n_heads=n_heads,
            num_experts=num_experts,
            num_field_modes=num_field_modes,
            rff_scale=rff_scale,
            use_rff=use_rff,
            use_checkpoint=use_checkpoint,
            predict_frequency=predict_frequency
        )
        self.freq_weight = freq_weight
        self.ortho_weight = ortho_weight
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

    def _compute_orthogonality_loss(self, pred_field, batch):
        """
        Fiziksel ortogonalite kısıtı: aynı geometrinin farklı modları birbirine dik olmalı.
        ∫ E_m(x) · E_n(x) dA = 0  (m ≠ n)

        node_area (Input_funcs[:, :, 5]) ile ağırlıklandırılmış iç çarpım kullanılır.
        Bu, alan integralinin doğru bir ayrık yaklaşımıdır.
        """
        geom_ids = batch['geom_id'].squeeze(-1)   # [B]
        mode_ids = batch['Theta_in'].squeeze(-1)   # [B]
        mask = batch.get('Mask', None)
        B = pred_field.shape[0]

        # node_area feature index = 5 (bkz: docs/02_FEATURE_ENGINEERING.md)
        node_area = batch['Input_funcs'][:, :, 5]  # [B, N]

        loss_ortho = torch.tensor(0.0, device=pred_field.device)
        n_pairs = 0

        # Eşsiz geometrileri bul
        unique_geoms = geom_ids.unique()

        for g_id in unique_geoms:
            g_mask = (geom_ids == g_id)
            g_indices = g_mask.nonzero(as_tuple=True)[0]

            if len(g_indices) < 2:
                continue  # Tek mod varsa dik olacak bir şey yok

            # Bu geometriye ait tüm mod tahminlerini topla
            g_modes = mode_ids[g_indices]          # [n_modes_in_batch]
            g_preds = pred_field[g_indices]          # [n_modes, N, 1]
            g_area = node_area[g_indices[0]]         # [N] — aynı geometri, aynı area

            if mask is not None:
                g_valid = mask[g_indices[0]].float()  # [N]
            else:
                g_valid = torch.ones(g_area.shape[0], device=g_area.device)

            # Area * valid mask = integral ağırlığı
            w = (g_area * g_valid).unsqueeze(-1)  # [N, 1]

            # Her (m, n) çifti için cosine similarity hesapla
            for i in range(len(g_indices)):
                for j in range(i + 1, len(g_indices)):
                    if g_modes[i] == g_modes[j]:
                        continue  # Aynı modun tekrarı — skip

                    u_i = g_preds[i]  # [N, 1]
                    u_j = g_preds[j]  # [N, 1]

                    # Alan-ağırlıklı iç çarpım
                    dot = (u_i * u_j * w).sum()
                    norm_i = (u_i ** 2 * w).sum().sqrt().clamp(min=1e-8)
                    norm_j = (u_j ** 2 * w).sum().sqrt().clamp(min=1e-8)

                    cos_sim = dot / (norm_i * norm_j)
                    loss_ortho = loss_ortho + cos_sim ** 2
                    n_pairs += 1

        if n_pairs > 0:
            loss_ortho = loss_ortho / n_pairs

        return loss_ortho

    def _compute_loss(self, batch, prefix):
        outputs = self.model(batch)
        mask = batch.get('Mask', None)  # [B, N] boolean

        pred_field = outputs['field']      # [B, N, 1]
        true_field = batch['Y_field']      # [B, N, 1]
        mode_ids = batch['Theta_in'].squeeze(-1)  # [B]
        B, N, _ = pred_field.shape

        # --- PHYSICS-INFORMED NEURAL NETWORK (PINN) BOUNDARY CONSTRAINT ---
        # Input_funcs[:, :, 2] is the 'dist_boundary' feature
        dist_bnd = batch['Input_funcs'][:, :, 2]  # [B, N]
        bnd_mask = (dist_bnd < 1e-4).unsqueeze(-1) # [B, N, 1] boolean mask
        
        # 1. PINN Boundary Penalty: Raw ağ çıktısının duvarda (0'a gitmesi gerekirken) yaptığı hata
        # (Bu, ağın sınırları fiziksel olarak öğrenmesini sağlar)
        if bnd_mask.any():
            loss_bnd = (pred_field[bnd_mask] ** 2).mean()
        else:
            loss_bnd = torch.tensor(0.0, device=pred_field.device)
            
        self.log(f'{prefix}/loss_bnd', loss_bnd, prog_bar=True, batch_size=B, sync_dist=True)
        
        # 2. Hard Constraint: Tahminleri duvar düğümlerinde tartışılamaz şekilde 0.0'a ez.
        # Bu işlem gradyanı keser, bu yüzden ağın duvarda gradyan alabilmesi için üstte loss_bnd hesapladık.
        pred_field = pred_field * (~bnd_mask).float()
        # ------------------------------------------------------------------

        # --- SIGN REALIGNMENT DURING TRAINING ---
        # Her örnek için pred ve true arasındaki faza (işarete) bak.
        # Eğer ters işaret daha yakınsa pred'i ters çevir.
        with torch.no_grad():
            # [B] boyutunda işaret belirle: MSE(p, t) < MSE(p, -t) ise 1, değilse -1
            # Maske varsa sadece maskeli bölgede karara var.
            if mask is not None:
                m_f = mask.unsqueeze(-1).float()
                diff_pos = ((pred_field - true_field) ** 2 * m_f).sum(dim=(1, 2))
                diff_neg = ((pred_field + true_field) ** 2 * m_f).sum(dim=(1, 2))
            else:
                diff_pos = ((pred_field - true_field) ** 2).mean(dim=(1, 2))
                diff_neg = ((pred_field + true_field) ** 2).mean(dim=(1, 2))
            
            signs = torch.where(diff_pos <= diff_neg, 1.0, -1.0).view(B, 1, 1)
        
        # Gradyan akışını bozmadan işareti düzelt (differentiable sign selection)
        pred_field = pred_field * signs
        # ----------------------------------------

        # Per-mode weighted field loss
        loss_field = 0.0
        for mode_val in range(len(self.mode_loss_weights)):
            mode_mask = (mode_ids == mode_val)
            if not mode_mask.any():
                continue
            p = pred_field[mode_mask]   # [n_mode, N, 1]
            t = true_field[mode_mask]
            w = float(self.mode_loss_weights[mode_val])

            if mask is not None:
                m = mask[mode_mask].unsqueeze(-1).float()
                n_valid = m.sum().clamp(min=1.0)
                
                # Peak-Weighted Loss: Tepe noktalarında (amplitude yüksek olan veriler) cezayı artır
                # Hata = (p-t)^2 * (1 + 5 * |t|)
                weight = 1.0 + 5.0 * t.abs()
                mse_weighted = (((p - t) ** 2) * weight * m).sum() / n_valid
                
                # L1 Loss: Ufak tefek genlik sapmalarını silmek için
                l1_loss = ((p - t).abs() * m).sum() / n_valid
                
                mode_loss = mse_weighted + 0.1 * l1_loss
            else:
                weight = 1.0 + 5.0 * t.abs()
                mse_weighted = (((p - t) ** 2) * weight).mean()
                l1_loss = F.l1_loss(p, t)
                mode_loss = mse_weighted + 0.1 * l1_loss

            loss_field = loss_field + w * mode_loss
            # Log mode-specific loss on epoch level for both train and val
            self.log(f'{prefix}/mode_{mode_val}_loss', mode_loss,
                     on_step=False, on_epoch=True, prog_bar=False, 
                     batch_size=int(mode_mask.sum()), sync_dist=True)
        
        # Normalize by sum of weights
        loss_field = loss_field / sum(self.mode_loss_weights)

        # Frequency loss: skip if predict_frequency is disabled
        if self.predict_frequency and outputs.get('freq') is not None:
            loss_freq = F.mse_loss(outputs['freq'], batch['Y_freq'])
            self.log(f'{prefix}/freq_loss', loss_freq, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)
            
            if self.freq_stats:
                freq_pred_ghz = outputs['freq'] * self.freq_stats['std'] + self.freq_stats['mean']
                freq_true_ghz = batch['Y_freq'] * self.freq_stats['std'] + self.freq_stats['mean']
                mae_ghz = F.l1_loss(freq_pred_ghz, freq_true_ghz)
                self.log(f'{prefix}/freq_mae_ghz', mae_ghz, on_step=False, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        else:
            loss_freq = torch.tensor(0.0, device=pred_field.device)

        # --- ORTHOGONALITY LOSS ---
        # Aynı geometrinin farklı modlarının tahminleri birbirine dik olmalı
        if self.ortho_weight > 0:
            loss_ortho = self._compute_orthogonality_loss(pred_field, batch)
            self.log(f'{prefix}/loss_ortho', loss_ortho, on_step=False, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        else:
            loss_ortho = torch.tensor(0.0, device=pred_field.device)

        # Toplam Loss = Alan + Frekans + Duvar(PINN) + Ortogonalite
        total_loss = loss_field + (self.freq_weight * loss_freq) + (1.0 * loss_bnd) + (self.ortho_weight * loss_ortho)

        self.log(f'{prefix}/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self.log(f'{prefix}/field_loss', loss_field, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)

        # Per-sample Relative L2 Error — VECTORIZED (no Python for-loop)
        if mask is not None:
            m_f = mask.unsqueeze(-1).float()  # [B, N, 1]
            diff_sq = ((pred_field - true_field) ** 2 * m_f).sum(dim=(1, 2))  # [B]
            true_sq = ((true_field) ** 2 * m_f).sum(dim=(1, 2))  # [B]
        else:
            diff_sq = ((pred_field - true_field) ** 2).sum(dim=(1, 2))  # [B]
            true_sq = ((true_field) ** 2).sum(dim=(1, 2))  # [B]
        rel_l2 = (torch.sqrt(diff_sq) / (torch.sqrt(true_sq) + 1e-8)).mean()
        self.log(f'{prefix}/field_rel_l2', rel_l2, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)

        # Per-mode relative L2 error — vectorized
        mode_ids = batch['Theta_in'].squeeze(-1)  # [B]
        for mode_val in range(len(self.mode_loss_weights)):
            mode_mask_b = (mode_ids == mode_val)  # [B]
            if not mode_mask_b.any():
                continue
            mode_diff = diff_sq[mode_mask_b]  # [n_mode]
            mode_true = true_sq[mode_mask_b]  # [n_mode]
            mode_rel = (torch.sqrt(mode_diff) / (torch.sqrt(mode_true) + 1e-8)).mean()
            self.log(f'{prefix}/mode_{mode_val}_rel_l2', mode_rel,
                     on_step=False, on_epoch=True, prog_bar=False, 
                     batch_size=int(mode_mask_b.sum()), sync_dist=True)

        # Extract valid-only flat tensors for torchmetrics (R2, MAE)
        if mask is not None:
            mask_flat = mask.unsqueeze(-1).expand_as(pred_field)  # [B, N, 1]
            preds_valid = pred_field[mask_flat].contiguous()
            targets_valid = true_field[mask_flat].contiguous()
        else:
            preds_valid = pred_field.contiguous().view(-1)
            targets_valid = true_field.contiguous().view(-1)

        return total_loss, preds_valid, targets_valid

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch, "train")
        self.train_r2(preds, targets)
        self.log('train/r2', self.train_r2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Öğrenme oranını (learning rate) progress bar'a yansıt
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True, on_step=True, on_epoch=False)
        
        return loss

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
        global_val_l2 = metrics.get('val/field_rel_l2', 0.0)
        global_val_r2 = metrics.get('val/r2', 0.0)
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
