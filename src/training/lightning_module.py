import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from src.models.gnot import GNOTModel

class GNOTLightning(pl.LightningModule):
    def __init__(self, val_dim=6, grid_dim=2, theta_dim=1, hidden_dim=256, 
                 n_shared_layers=2, n_mode_layers=2, n_freq_layers=2,
                 n_heads=4, num_experts=4, num_field_modes=3,
                 lr=1e-3, freq_weight=0.5, mode_loss_weights=None,
                 scheduler='onecycle', weight_decay=1e-4, use_checkpoint=False,
                 rff_scale=1.0, use_rff=True,
                 onecycle_pct_start=0.3, onecycle_div_factor=25, onecycle_final_div_factor=1e4,
                 cosine_eta_min=1e-6):
        super().__init__()
        self.save_hyperparameters()
        self.model = GNOTModel(
            val_dim=val_dim,
            grid_dim=grid_dim,
            theta_dim=theta_dim,
            embed_dim=hidden_dim,
            n_shared_layers=n_shared_layers,
            n_mode_layers=n_mode_layers,
            n_freq_layers=n_freq_layers,
            n_heads=n_heads,
            num_experts=num_experts,
            num_field_modes=num_field_modes,
            rff_scale=rff_scale,
            use_rff=use_rff,
            use_checkpoint=use_checkpoint
        )
        self.freq_weight = freq_weight
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
        mode_ids = batch['Theta_in'].squeeze(-1)  # [B]
        B, N, _ = pred_field.shape

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
                mode_loss = ((p - t) ** 2 * m).sum() / n_valid
            else:
                mode_loss = F.mse_loss(p, t)

            loss_field = loss_field + w * mode_loss
            self.log(f'{prefix}/mode_{mode_val}_loss', mode_loss,
                     prog_bar=False, batch_size=int(mode_mask.sum()))
        
        # Normalize by sum of weights
        loss_field = loss_field / sum(self.mode_loss_weights)

        loss_freq = F.mse_loss(outputs['freq'], batch['Y_freq'])
        total_loss = loss_field + (self.freq_weight * loss_freq)

        self.log(f'{prefix}/loss', total_loss, prog_bar=True, batch_size=B)
        self.log(f'{prefix}/field_loss', loss_field, prog_bar=False, batch_size=B)
        self.log(f'{prefix}/freq_loss', loss_freq, prog_bar=False, batch_size=B)
        
        if self.freq_stats:
            freq_pred_ghz = outputs['freq'] * self.freq_stats['std'] + self.freq_stats['mean']
            freq_true_ghz = batch['Y_freq'] * self.freq_stats['std'] + self.freq_stats['mean']
            mae_ghz = F.l1_loss(freq_pred_ghz, freq_true_ghz)
            self.log(f'{prefix}/freq_mae_ghz', mae_ghz, prog_bar=True, batch_size=B)

        # Per-sample Relative L2 Error (Artık hizalı olduğu için doğrudan norm yeterli)
        rel_l2_list = []
        for i in range(B):
            m = mask[i] if mask is not None else torch.ones_like(pred_field[i, :, 0], dtype=torch.bool)
            p = pred_field[i, m, :]
            t = true_field[i, m, :]
            rel_l2_list.append(torch.norm(p - t) / (torch.norm(t) + 1e-8))
            
        rel_l2 = torch.stack(rel_l2_list).mean()
        self.log(f'{prefix}/field_rel_l2', rel_l2, prog_bar=True, batch_size=B)

        # Per-mode relative L2 error — hangi modun hatalı olduğunu görmek için
        mode_ids = batch['Theta_in'].squeeze(-1)  # [B]
        for mode_val in mode_ids.unique():
            idx = (mode_ids == mode_val).nonzero(as_tuple=True)[0]  # bu modun sample indeksleri
            if mask is not None:
                mode_rels = []
                for i in idx:
                    m = mask[i]
                    p = pred_field[i, m, :]
                    t = true_field[i, m, :]
                    mode_rels.append(torch.norm(p - t) / (torch.norm(t) + 1e-8))
                mode_rel = torch.stack(mode_rels).mean()
            else:
                p = pred_field[idx].view(len(idx), -1)
                t = true_field[idx].view(len(idx), -1)
                mode_rel = (torch.norm(p - t, dim=1) / (torch.norm(t, dim=1) + 1e-8)).mean()
            self.log(f'{prefix}/mode_{mode_val.item()}_rel_l2', mode_rel,
                     prog_bar=False, batch_size=len(idx))

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
        self.log('train/r2', self.train_r2, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch, "val")
        self.val_r2(preds, targets)
        self.val_mae(preds, targets)
        self.log('val/r2', self.val_r2, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mae', self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        metrics = self.trainer.logged_metrics
        epoch = self.trainer.current_epoch

        # Tüm mode_X_rel_l2 metriklerini topla
        mode_errors = {
            k: v for k, v in metrics.items()
            if k.startswith('val/mode_') and k.endswith('_rel_l2')
        }

        if not mode_errors:
            return

        # Başlık
        print(f"\n{'─'*50}")
        print(f"  Epoch {epoch:>3d} │ Val Per-Mode Field Error")
        print(f"{'─'*50}")

        # Her modu sıralı bas
        for key in sorted(mode_errors.keys()):
            mode_num = key.split('mode_')[1].split('_')[0]
            val = mode_errors[key]
            # Basit durum çubuğu (0.0 = iyi, 1.0+ = kötü)
            bar_len = int(min(val * 20, 20))
            bar = '█' * bar_len + '░' * (20 - bar_len)
            status = '✅' if val < 0.1 else ('⚠️ ' if val < 0.3 else '❌')
            print(f"  Mode {mode_num}  [{bar}]  {val:.4f}  {status}")

        # Global özet
        val_loss = metrics.get('val/loss', None)
        val_r2   = metrics.get('val/r2',   None)
        print(f"{'─'*50}")
        if val_loss is not None:
            print(f"  Total Loss: {val_loss:.4f}   R²: {val_r2:.4f}" if val_r2 is not None else f"  Total Loss: {val_loss:.4f}")
        print(f"{'─'*50}\n")

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch, "test")
        self.test_r2(preds, targets)
        self.test_mae(preds, targets)
        self.log('test/r2', self.test_r2, on_step=False, on_epoch=True)
        self.log('test/mae', self.test_mae, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Differential learning rates kaldırıldı, tüm model aynı lr ile eğitilir.
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        print(f"Optimizer: All params trained with lr={self.hparams.lr}")
        
        if self.hparams.scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=self.hparams.lr, 
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.hparams.onecycle_pct_start,
                div_factor=self.hparams.onecycle_div_factor,
                final_div_factor=self.hparams.onecycle_final_div_factor
            )
            interval = 'step'
        elif self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs, 
                eta_min=self.hparams.cosine_eta_min
            )
            interval = 'epoch'
        else:
            return optimizer

        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": interval
            }
        }
