"""Lightning smoke test for SpectralNO: one train, validation and test step.

Lightning runs trainer.validate/test under torch.inference_mode() by default
(fit's own validation loop uses no_grad), which
the SpectralNO stiffness assembly (autograd w.r.t. coordinates) must survive.
Tiny synthetic disk geometries, CPU, a few seconds.
"""
import math

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import gnot_collate_fn
from src.training.lightning_module import GNOTLightning


class _DiskDataset(Dataset):
    """Variable-size point clouds in the unit disk with consistent features and
    max-normalized, frequency-sorted targets (same layout as GNOTDataset)."""

    def __init__(self, n=4, K=3, val_dim=12):
        g = torch.Generator().manual_seed(0)
        self.items = []
        for i in range(n):
            N = 30 + 5 * i
            r = torch.rand(N, generator=g).sqrt() * 0.98
            t = torch.rand(N, generator=g) * 2 * math.pi
            X = torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)
            Y = torch.rand(N, val_dim, generator=g)
            Y[:, 0:2] = X
            Y[:, 2] = 1.0 - r
            Y[:, 3:5] = X / r.unsqueeze(-1).clamp(min=1e-6)
            Y[:, 5] = 0.5 + torch.rand(N, generator=g)
            field = torch.stack([torch.cos(math.pi / 2 * r) * torch.cos(k * t) for k in range(K)], -1)
            field = field / field.abs().amax(dim=0, keepdim=True)
            self.items.append({
                'X': X, 'Input_funcs': Y, 'Y_field': field,
                'Y_freq': torch.sort(torch.randn(K, generator=g)).values,
                'geom_id': torch.tensor([i]),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def test_spectral_no_fit_and_validate_under_inference_mode(tmp_path):
    torch.manual_seed(0)
    model = GNOTLightning(
        val_dim=12, grid_dim=2, hidden_dim=16, n_heads=2, num_field_modes=3,
        lr=1e-3, scheduler='none', model_type='spectral_no', n_basis=6,
        rff_dim=8, dropout=0.1, degeneracy_mode='hard', rayleigh_weight=0.1,
    )
    loader = DataLoader(_DiskDataset(), batch_size=2, collate_fn=gnot_collate_fn)
    # A real logger: on_validation_epoch_end dereferences self.logger.experiment.
    logger = pl.loggers.TensorBoardLogger(save_dir=str(tmp_path))
    trainer = pl.Trainer(fast_dev_run=True, accelerator='cpu', logger=logger,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, inference_mode=True)
    trainer.fit(model, loader, loader)      # fit's val loop runs under no_grad
    metrics = trainer.callback_metrics
    assert torch.isfinite(metrics['train/loss']).all()
    assert torch.isfinite(metrics['val/loss']).all()
    # trainer.test / trainer.validate run under torch.inference_mode (train.py
    # calls trainer.test after fit) -> used to raise "does not require grad".
    res = trainer.test(model, dataloaders=loader, verbose=False)
    assert math.isfinite(res[0]['test/r2'])
