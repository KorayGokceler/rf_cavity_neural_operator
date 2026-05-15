import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import math
import itertools
import numpy as np
from src.models.gnot import GNOTModel


# ════════════════════════════════════════════════════════════════════════════
#  Set-prediction loss helpers (frequency matching + Grassmannian subspace)
# ════════════════════════════════════════════════════════════════════════════

def match_frequencies(f_pred, f_true):
    """Enumerate K! permutations and return the one minimising the squared
    frequency assignment cost.

    Args:
        f_pred: [K] predicted (sorted) frequencies for one sample.
        f_true: [K] ground-truth frequencies for one sample.
    Returns:
        list[int] best permutation p such that f_pred[p[i]] matches f_true[i].
    """
    K = f_true.shape[0]
    fp = f_pred.detach().float().tolist()
    ft = f_true.detach().float().tolist()
    best_perm = min(
        itertools.permutations(range(K)),
        key=lambda p: sum((fp[p[i]] - ft[i]) ** 2 for i in range(K)),
    )
    return list(best_perm)


def detect_clusters(f_true_matched, threshold):
    """Group mode indices by relative frequency proximity (for 'hard' mode).

    Two consecutive (sorted) modes belong to the same cluster when
        |f_i - f_j| / max(|f_mean|, eps) < threshold.

    Args:
        f_true_matched: [K] target frequencies aligned to prediction order.
        threshold: relative gap threshold.
    Returns:
        list[list[int]] e.g. [[0], [1, 2]] or [[0], [1], [2]].
    """
    f = f_true_matched.detach().float().tolist()
    K = len(f)
    f_mean = sum(abs(v) for v in f) / max(K, 1)
    denom = max(abs(f_mean), 1e-6)
    clusters = [[0]]
    for i in range(1, K):
        if abs(f[i] - f[i - 1]) / denom < threshold:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def _masked_orthonormalize(E, mask=None):
    """Return an orthonormal basis (over the masked rows) for the column span
    of E using a numerically stable reduced QR.

    Args:
        E: [N, n] matrix whose columns span the subspace.
        mask: [N] boolean (True = valid node) or None.
    Returns:
        Q: [N, n] with masked rows zeroed and Q^T Q = I over valid rows.
    """
    if mask is not None:
        m = mask.to(E.dtype).unsqueeze(-1)  # [N, 1]
        E = E * m
    # Reduced QR; add tiny ridge for stability when columns are near-collinear
    # (degenerate/near-degenerate eigenvectors).
    Q, R = torch.linalg.qr(E, mode='reduced')
    # Fix sign ambiguity of QR (not required for Grassmannian but keeps R* sane)
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    sign = torch.sign(diag)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    Q = Q * sign.unsqueeze(-2)
    if mask is not None:
        Q = Q * mask.to(Q.dtype).unsqueeze(-1)
    return Q


def grassmannian_loss(E_hat, E_tgt, mask=None):
    """Grassmannian subspace distance:  n - ||Q_hat^T Q_tgt||_F^2.

    Equals 0 when the two subspaces coincide and n when fully orthogonal.
    For n == 1 this reduces exactly to 1 - cos^2(theta) between the two
    vectors, i.e. the standard sign-invariant field similarity.

    Args:
        E_hat, E_tgt: [N, n] predicted / target subspace bases.
        mask: [N] boolean or None.
    """
    n = E_hat.shape[-1]
    Qh = _masked_orthonormalize(E_hat, mask)
    Qt = _masked_orthonormalize(E_tgt, mask)
    M = Qh.transpose(-2, -1) @ Qt          # [n, n]
    return n - (M ** 2).sum()


def soft_procrustes_loss(E_hat, E_tgt, f_matched, sigma, mask=None):
    """Frequency-weighted soft Procrustes alignment loss.

    W[i,j] = exp(-(f_i - f_j)^2 / sigma^2) softly couples only the
    near-degenerate modes; well-separated modes keep an (almost) identity
    weighting so this gracefully reduces to a per-mode field loss.

        M_w = (E_hat^T E_tgt) * W      (elementwise)
        SVD: M_w = U S V^T  ->  R* = V U^T   (orthogonal alignment)
        loss = relative_L2(E_hat, E_tgt @ R*)

    Args:
        E_hat, E_tgt: [N, K] columns are the (mask-applied) mode fields.
        f_matched: [K] target frequencies (prediction-aligned order).
        sigma: scalar coupling bandwidth.
        mask: [N] boolean or None.
    """
    if mask is not None:
        m = mask.to(E_hat.dtype).unsqueeze(-1)  # [N, 1]
        E_hat = E_hat * m
        E_tgt = E_tgt * m

    fi = f_matched.view(-1, 1)
    fj = f_matched.view(1, -1)
    sigma2 = (sigma ** 2) + 1e-12
    W = torch.exp(-((fi - fj) ** 2) / sigma2)  # [K, K]

    M = (E_hat.transpose(-2, -1) @ E_tgt) * W  # [K, K]
    # Orthogonal Procrustes: best R aligning E_tgt onto E_hat.
    U, S, Vh = torch.linalg.svd(M)
    R = Vh.transpose(-2, -1) @ U.transpose(-2, -1)  # [K, K], R* = V U^T

    E_tgt_rot = E_tgt @ R                       # [N, K]
    num = ((E_hat - E_tgt_rot) ** 2).sum()
    den = (E_tgt ** 2).sum() + 1e-8
    return num / den

class GNOTLightning(pl.LightningModule):
    def __init__(self, val_dim=6, grid_dim=2, hidden_dim=256,
                 n_shared_layers=2, n_mode_layers=2, n_field_head_layers=2,
                 n_heads=4, num_experts=4, num_field_modes=3,
                 lr=1e-3, freq_weight=0.5, smoothness_weight=0.0,
                 mode_loss_weights=None,
                 lr_mode_specific=None, lr_freq_heads=None,
                 scheduler='onecycle', weight_decay=1e-4, use_checkpoint=False,
                 reducelr_patience=10, reducelr_factor=0.5,
                 predict_frequency=True,
                 onecycle_pct_start=0.3, onecycle_div_factor=25, onecycle_final_div_factor=1e4,
                 cosine_eta_min=1e-6,
                 gradient_clip_val=None,
                 dropout=0.0,
                 rff_dim=64, rff_length_scale=0.1,
                 degeneracy_mode='soft',
                 near_deg_threshold=0.05,
                 deg_sigma_rel=0.5,
                 freq_match_weight=0.5):
        super().__init__()
        # Suppress harmless DDP + gradient checkpointing stream mismatch warning
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        self.lr_mode_specific = lr_mode_specific
        self.lr_freq_heads = lr_freq_heads
        self.gradient_clip_val = gradient_clip_val
        
        self.save_hyperparameters()
        
        self.model = GNOTModel(
            val_dim=val_dim,
            grid_dim=grid_dim,
            embed_dim=hidden_dim,
            n_shared_layers=n_shared_layers,
            n_mode_layers=n_mode_layers,
            n_field_head_layers=n_field_head_layers,
            n_heads=n_heads,
            num_experts=num_experts,
            num_field_modes=num_field_modes,
            use_checkpoint=use_checkpoint,
            predict_frequency=predict_frequency,
            dropout=dropout,
            rff_dim=rff_dim,
            rff_length_scale=rff_length_scale,
        )
        self.freq_weight = freq_weight
        self.smoothness_weight = smoothness_weight
        self.predict_frequency = predict_frequency
        self.num_field_modes = num_field_modes
        self.freq_stats = None
        # Set-prediction / degeneracy handling
        assert degeneracy_mode in ('soft', 'hard'), \
            f"degeneracy_mode must be 'soft' or 'hard', got {degeneracy_mode}"
        self.degeneracy_mode = degeneracy_mode
        self.near_deg_threshold = near_deg_threshold
        self.deg_sigma_rel = deg_sigma_rel
        self.freq_match_weight = freq_match_weight

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
        mask = batch.get('Mask', None)               # [B, N] boolean

        pred_field = outputs['field']                # [B, N, K]
        true_field = batch['Y_field']                # [B, N, K]
        f_pred = outputs['freq']                     # [B, K] (model-sorted)
        f_true = batch['Y_freq']                     # [B, K] (dataset-sorted)
        B, N, K = pred_field.shape
        eps = 1e-8

        # --- PINN BOUNDARY CONSTRAINT (applied to every mode column) -------
        dist_bnd = batch['Input_funcs'][:, :, 2]     # [B, N]
        valid_mask = mask if mask is not None else torch.ones(B, N, dtype=torch.bool, device=pred_field.device)
        bnd_mask = (dist_bnd < 1e-4) & valid_mask    # [B, N]
        bnd_mask_f = bnd_mask.unsqueeze(-1).float()  # [B, N, 1]

        if bnd_mask.any():
            sq_diff = (pred_field ** 2) * bnd_mask_f                  # [B, N, K]
            loss_bnd = sq_diff.sum() / (bnd_mask_f.sum() * K).clamp(min=1.0)
        else:
            loss_bnd = torch.tensor(0.0, device=pred_field.device)
        self.log(f'{prefix}/loss_bnd', loss_bnd, prog_bar=True, batch_size=B, sync_dist=True)

        # Hard constraint: zero predictions at boundary nodes (all modes)
        pred_field = pred_field * (1.0 - bnd_mask_f)

        # --- SET-PREDICTION ALIGNMENT (per batch item) --------------------
        # f_pred / f_true are both individually sorted ascending, but the model
        # may still mis-order which physical mode each slot captured.  Resolve
        # by enumerating K! frequency permutations and reordering predictions
        # to match the target order.
        device = pred_field.device
        loss_freq = torch.zeros((), device=device)
        loss_field = torch.zeros((), device=device)
        rel_l2_per_mode = torch.zeros(K, device=device)
        rel_l2_count = torch.zeros(K, device=device)
        aligned_pred_field = torch.zeros_like(pred_field)   # [B, N, K] — for R2/MAE

        # Spectral gap for soft sigma (relative to mean target gap in batch)
        for b in range(B):
            fp_b = f_pred[b]                          # [K]
            ft_b = f_true[b]                          # [K]
            perm = match_frequencies(fp_b, ft_b)
            perm_t = torch.tensor(perm, device=device, dtype=torch.long)

            fp_aligned = fp_b[perm_t]                 # [K] reordered to target order
            E_hat = pred_field[b][:, perm_t]          # [N, K] reorder mode columns
            aligned_pred_field[b] = E_hat             # store for metrics
            E_tgt = true_field[b]                     # [N, K]
            m_b = valid_mask[b]                       # [N]

            # Frequency regression loss (aligned)
            loss_freq = loss_freq + F.mse_loss(fp_aligned, ft_b)

            if self.degeneracy_mode == 'hard':
                clusters = detect_clusters(ft_b, self.near_deg_threshold)
                fl_b = torch.zeros((), device=device)
                for cl in clusters:
                    cl_t = torch.tensor(cl, device=device, dtype=torch.long)
                    g = grassmannian_loss(E_hat[:, cl_t], E_tgt[:, cl_t], m_b)
                    fl_b = fl_b + g
                loss_field = loss_field + fl_b
            else:  # 'soft'
                # sigma = deg_sigma_rel * mean spectral gap of this sample
                if K > 1:
                    gaps = (ft_b[1:] - ft_b[:-1]).abs()
                    mean_gap = gaps.mean().clamp(min=1e-4)
                else:
                    mean_gap = torch.tensor(1.0, device=device)
                sigma = self.deg_sigma_rel * mean_gap
                loss_field = loss_field + soft_procrustes_loss(
                    E_hat, E_tgt, ft_b, sigma, m_b)

            # --- per-mode sign-agnostic relative L2 (reporting only) ------
            with torch.no_grad():
                m_f = m_b.float().unsqueeze(-1)       # [N, 1]
                for k in range(K):
                    eh = E_hat[:, k:k + 1]
                    et = E_tgt[:, k:k + 1]
                    num_p = (((eh - et) ** 2) * m_f).sum()
                    num_n = (((eh + et) ** 2) * m_f).sum()
                    den = ((et ** 2) * m_f).sum() + eps
                    rl = torch.sqrt(torch.minimum(num_p, num_n) / den)
                    rel_l2_per_mode[k] = rel_l2_per_mode[k] + rl
                    rel_l2_count[k] = rel_l2_count[k] + 1.0

        loss_freq = loss_freq / max(B, 1)
        loss_field = loss_field / max(B, 1)

        if not (self.predict_frequency and outputs.get('freq') is not None):
            loss_freq = torch.zeros((), device=device)

        total_loss = loss_field + (self.freq_weight * loss_freq) + (self.smoothness_weight * loss_bnd)

        # --- logging -----------------------------------------------------
        self.log(f'{prefix}/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self.log(f'{prefix}/field_loss', loss_field, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)
        self.log(f'{prefix}/freq_loss', loss_freq, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)

        rel_l2_count = rel_l2_count.clamp(min=1.0)
        rel_l2_mean_per_mode = rel_l2_per_mode / rel_l2_count
        rel_l2 = rel_l2_mean_per_mode.mean()
        self.log(f'{prefix}/field_rel_l2', rel_l2, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        for k in range(K):
            self.log(f'{prefix}/mode_{k}_rel_l2', rel_l2_mean_per_mode[k],
                     on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)

        if self.predict_frequency and outputs.get('freq') is not None and self.freq_stats:
            with torch.no_grad():
                fp_ghz = f_pred * self.freq_stats['std'] + self.freq_stats['mean']
                ft_ghz = f_true * self.freq_stats['std'] + self.freq_stats['mean']
                mae_ghz = F.l1_loss(fp_ghz, ft_ghz)
                self.log(f'{prefix}/freq_mae_ghz', mae_ghz, on_step=False, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)

        # Tensors for torchmetrics (R2, MAE) — use Hungarian-aligned predictions.
        with torch.no_grad():
            mask_km = valid_mask.unsqueeze(-1).expand_as(aligned_pred_field)  # [B, N, K]
            preds_valid = aligned_pred_field[mask_km].contiguous()
            targets_valid = true_field[mask_km].contiguous()

        return total_loss, preds_valid, targets_valid

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch, "train")
        self.train_r2(preds, targets)
        self.log('train/r2', self.train_r2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Öğrenme oranını (learning rate) progress bar'a yansıt
        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
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

            # Mini bar chart for Val error (Safe for NaN)
            if not np.isfinite(v_val):
                bar_len = 0
                bar = '?' * 15
            else:
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

        # Freq Head LR (single global head now)
        if self.lr_freq_heads is not None and self.predict_frequency:
            freq_params = []
            for p in self.model.freq_head_global.parameters():
                if id(p) not in handled_param_ids:
                    freq_params.append(p)
                    handled_param_ids.add(id(p))
            if len(freq_params) > 0:
                param_groups.append({"params": freq_params, "lr": self.lr_freq_heads})
                print(f"Optimizer: Frequency head trained with lr={self.lr_freq_heads}")

        # Base param group (General trunk, embeddings, unhandled parts)
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
        elif self.hparams.scheduler == 'custom_cosine':
            # Phase 1 (epochs 0-9):  constant at base_lr (warm-up / stable start)
            # Phase 2 (epoch 10+):   cosine decay from base_lr down to cosine_eta_min
            def lr_lambda(epoch):
                base_lr = self.hparams.lr
                eta_min = self.hparams.cosine_eta_min
                warmup_epochs = 10
                if epoch < warmup_epochs:
                    return 1.0
                total_cos_epochs = self.trainer.max_epochs - warmup_epochs
                if total_cos_epochs <= 0:
                    return eta_min / base_lr
                progress = min(1.0, (epoch - warmup_epochs) / total_cos_epochs)
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                target_lr = eta_min + (base_lr - eta_min) * cosine_factor
                return target_lr / base_lr

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        else:
            return optimizer
