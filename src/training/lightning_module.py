import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from src.models.gnot import GNOTModel

class GNOTLightning(pl.LightningModule):
    def __init__(self, val_dim=6, grid_dim=2, theta_dim=1, hidden_dim=128, n_layers=4, lr=1e-3, freq_weight=0.1, 
                 scheduler='onecycle', weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = GNOTModel(
            val_dim=val_dim,
            grid_dim=grid_dim,
            theta_dim=theta_dim,
            embed_dim=hidden_dim,
            n_layers=n_layers
        )
        self.freq_weight = freq_weight
        self.freq_stats = None # Will be set by the dataloader or manually

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
        loss_field = F.mse_loss(outputs['field'], batch['Y_field'])
        loss_freq = F.mse_loss(outputs['freq'], batch['Y_freq'])
        total_loss = loss_field + (self.freq_weight * loss_freq)

        self.log(f'{prefix}/loss', total_loss, prog_bar=True)
        self.log(f'{prefix}/field_loss', loss_field, prog_bar=False)
        self.log(f'{prefix}/freq_loss', loss_freq, prog_bar=False)
        
        # Denormalized Freq MAE for better tracking
        if self.freq_stats:
            freq_pred_ghz = outputs['freq'] * self.freq_stats['std'] + self.freq_stats['mean']
            freq_true_ghz = batch['Y_freq'] * self.freq_stats['std'] + self.freq_stats['mean']
            mae_ghz = F.l1_loss(freq_pred_ghz, freq_true_ghz)
            self.log(f'{prefix}/freq_mae_ghz', mae_ghz, prog_bar=True)

        # Calculate Relative L2 Error for field
        rel_l2 = torch.norm(outputs['field'] - batch['Y_field'], p=2) / (torch.norm(batch['Y_field'], p=2) + 1e-8)
        self.log(f'{prefix}/field_rel_l2', rel_l2, prog_bar=True)

        return total_loss, outputs['field'], batch['Y_field']

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch, "train")
        self.train_r2(preds.contiguous().view(-1), targets.contiguous().view(-1))
        self.log('train/r2', self.train_r2, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch, "val")
        self.val_r2(preds.contiguous().view(-1), targets.contiguous().view(-1))
        self.val_mae(preds.contiguous().view(-1), targets.contiguous().view(-1))
        self.log('val/r2', self.val_r2, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mae', self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch, "test")
        self.test_r2(preds.contiguous().view(-1), targets.contiguous().view(-1))
        self.test_mae(preds.contiguous().view(-1), targets.contiguous().view(-1))
        self.log('test/r2', self.test_r2, on_step=False, on_epoch=True)
        self.log('test/mae', self.test_mae, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        if self.hparams.scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=self.hparams.lr, 
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                div_factor=25,
                final_div_factor=1e4
            )
            interval = 'step'
        elif self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs, 
                eta_min=1e-6
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
