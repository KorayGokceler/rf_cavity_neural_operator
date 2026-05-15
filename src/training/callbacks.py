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
        
        # Move batch tensors to device
        def to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            if isinstance(obj, list):
                return [to_device(i, device) for i in obj]
            if isinstance(obj, dict):
                return {k: to_device(v, device) for k, v in obj.items()}
            return obj
            
        batch = to_device(batch, pl_module.device)
        
        with torch.no_grad():
            outputs = pl_module(batch)
            preds = outputs['field']            # [B, N, K]
            targets = batch['Y_field']          # [B, N, K]
            coords = batch['X']
            mask = batch.get('Mask', None)

        K = preds.shape[-1]
        for i in range(min(self.num_samples, len(preds))):
            m = mask[i] if mask is not None else slice(None)
            valid_coords = coords[i, m].cpu().numpy()

            for k in range(K):  # one figure per mode (ascending freq order)
                valid_targets = targets[i, m, k].cpu().numpy()
                valid_preds = preds[i, m, k].cpu().numpy()

                # Sign-agnostic alignment for visualization (eigenmode sign)
                err_pos = np.linalg.norm(valid_preds - valid_targets)
                err_neg = np.linalg.norm(valid_preds + valid_targets)
                if err_neg < err_pos:
                    valid_preds = -valid_preds

                fig = self._plot_comparison(
                    valid_coords,
                    valid_targets,
                    valid_preds,
                    title=f"Epoch {trainer.current_epoch} - Mode {k} (asc. freq) - Sample {i}"
                )

                if trainer.logger and hasattr(trainer.logger, 'experiment'):
                    trainer.logger.experiment.add_figure(
                        f"Validation/Field_Comparison_s{i}_m{k}", fig,
                        global_step=trainer.global_step
                    )
                plt.close(fig)

    def _plot_comparison(self, coords, target, pred, title=""):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Simple triangulation for point cloud visualization
        # Note: This assumes points are somewhat regularly distributed
        tri = Triangulation(coords[:, 0], coords[:, 1])
        target_flat = target.flatten()
        pred_flat = pred.flatten()
        error_flat = pred_flat - target_flat

        # Symmetric color scale for ground truth and prediction
        vmax = max(np.abs(target_flat).max(), np.abs(pred_flat).max(), 1e-8)

        im1 = axes[0].tripcolor(tri, target_flat, cmap='RdBu_r', shading='gouraud', vmin=-vmax, vmax=vmax)
        axes[0].set_title("Ground Truth")
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].tripcolor(tri, pred_flat, cmap='RdBu_r', shading='gouraud', vmin=-vmax, vmax=vmax)
        axes[1].set_title("Prediction")
        fig.colorbar(im2, ax=axes[1])

        # Error plot — symmetric colormap centred at 0
        err_max = max(np.abs(error_flat).max(), 1e-8)
        rel_l2 = np.linalg.norm(error_flat) / (np.linalg.norm(target_flat) + 1e-8)
        im3 = axes[2].tripcolor(tri, error_flat, cmap='RdBu_r', shading='gouraud', vmin=-err_max, vmax=err_max)
        axes[2].set_title(f"Error (pred - truth) | rel L2: {rel_l2:.4f}")
        fig.colorbar(im3, ax=axes[2])

        for ax in axes:
            ax.set_aspect('equal')
            ax.axis('off')

        fig.suptitle(title)
        return fig
