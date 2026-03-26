import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from src.models.gnot import GNOTModel

class GNOTLightning(pl.LightningModule):
    def __init__(self, val_dim=6, grid_dim=2, theta_dim=1, hidden_dim=128, n_layers=4, lr=1e-3, freq_weight=0.1):
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
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
