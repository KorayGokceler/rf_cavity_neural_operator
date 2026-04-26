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
        # Only run visualization on rank 0 (avoids duplicate logging in multi-GPU DDP)
        if trainer.global_rank != 0:
            return
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        # Get a batch from validation dataloader
        try:
            val_loaders = trainer.val_dataloaders
            if val_loaders is None:
                return
            
            # If it's a list or sequence, take first one
            if isinstance(val_loaders, (list, tuple)):
                val_loader = val_loaders[0]
            else:
                val_loader = val_loaders
                
            # If we have a datamodule, it's often more reliable to get it from there
            if hasattr(trainer, 'datamodule') and trainer.datamodule:
                val_loader = trainer.datamodule.val_dataloader()
                
            batch = next(iter(val_loader))
        except Exception as e:
            print(f"Warning: Could not get validation batch for visualization: {e}")
            return
        
        # Move to device
        batch = {k: (v.to(pl_module.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = pl_module(batch)
            preds = outputs['field']      # [B, N, 3]
            targets = batch['Y_fields']   # [B, N, 3]
            coords = batch['X']
            mask = batch.get('Mask', None)

        for i in range(min(self.num_samples, len(preds))):
            m = mask[i] if mask is not None else slice(None)
            
            # Extract only valid nodes for visualization
            valid_coords = coords[i, m].cpu().numpy()
            valid_targets = targets[i, m].cpu().numpy() # [N_valid, 3]
            valid_preds = preds[i, m].cpu().numpy()     # [N_valid, 3]
            
            fig = self._plot_comparison(
                valid_coords,
                valid_targets,
                valid_preds,
                title=f"Epoch {trainer.current_epoch} - Geometry {i} (Multi-Mode)"
            )
            
            # Log to TensorBoard (guard against missing logger)
            if trainer.logger and hasattr(trainer.logger, 'experiment'):
                trainer.logger.experiment.add_figure(
                    f"Validation/Field_Comparison_{i}", fig, global_step=trainer.global_step
                )
            plt.close(fig)

    def _plot_comparison(self, coords, target, pred, title=""):
        num_modes = target.shape[1] # usually 3
        fig, axes = plt.subplots(num_modes, 2, figsize=(12, 4 * num_modes))
        
        # Simple triangulation for point cloud visualization
        tri = Triangulation(coords[:, 0], coords[:, 1])
        
        for m_idx in range(num_modes):
            im1 = axes[m_idx, 0].tripcolor(tri, target[:, m_idx], cmap='RdBu_r', shading='gouraud')
            axes[m_idx, 0].set_title(f"Ground Truth - Mode {m_idx}")
            fig.colorbar(im1, ax=axes[m_idx, 0])
            
            im2 = axes[m_idx, 1].tripcolor(tri, pred[:, m_idx], cmap='RdBu_r', shading='gouraud')
            axes[m_idx, 1].set_title(f"Prediction - Mode {m_idx}")
            fig.colorbar(im2, ax=axes[m_idx, 1])
            
            for ax in axes[m_idx]:
                ax.set_aspect('equal')
                ax.axis('off')
            
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
