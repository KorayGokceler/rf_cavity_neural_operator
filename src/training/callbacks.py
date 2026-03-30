import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
from matplotlib.tri import Triangulation

class FieldVisualizationCallback(pl.Callback):
    def __init__(self, num_samples=3, log_every_n_epochs=5):
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        # Get a batch from validation dataloader
        val_loaders = trainer.val_dataloaders
        if val_loaders is None:
            return
            
        if isinstance(val_loaders, list):
            val_loader = val_loaders[0]
        else:
            val_loader = val_loaders
        
        # In modern Lightning, if trainer.datamodule exists, use it
        if hasattr(trainer, 'datamodule') and trainer.datamodule is not None:
            db_loader = trainer.datamodule.val_dataloader()
            if db_loader: val_loader = db_loader

        batch = next(iter(val_loader))
        
        # Move to device
        batch = {k: v.to(pl_module.device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = pl_module(batch)
            preds = outputs['field']
            targets = batch['Y_field']
            coords = batch['X']

        for i in range(min(self.num_samples, len(preds))):
            fig = self._plot_comparison(
                coords[i].cpu().numpy(),
                targets[i].cpu().numpy(),
                preds[i].cpu().numpy(),
                title=f"Epoch {trainer.current_epoch} - Sample {i}"
            )
            
            # Log to TensorBoard
            trainer.logger.experiment.add_figure(
                f"Validation/Field_Comparison_{i}", fig, global_step=trainer.global_step
            )
            plt.close(fig)

    def _plot_comparison(self, coords, target, pred, title=""):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Simple triangulation for point cloud visualization
        # Note: This assumes points are somewhat regularly distributed
        tri = Triangulation(coords[:, 0], coords[:, 1])
        
        im1 = axes[0].tripcolor(tri, target.flatten(), cmap='RdBu_r', shading='gouraud')
        axes[0].set_title("Ground Truth")
        fig.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].tripcolor(tri, pred.flatten(), cmap='RdBu_r', shading='gouraud')
        axes[1].set_title("Prediction")
        fig.colorbar(im2, ax=axes[1])
        
        for ax in axes:
            ax.set_aspect('equal')
            ax.axis('off')
            
        fig.suptitle(title)
        return fig
