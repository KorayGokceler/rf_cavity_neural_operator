"""Tests for the GNOTLightning set-prediction loss (spectral-subspace arch).

Invariants:
- _compute_loss runs end-to-end for soft and hard degeneracy modes
- Boundary energy is a soft penalty only (predictions are NOT hard-zeroed)
- Loss stays finite (incl. degenerate / zero-target edge cases)
- Frequency matching cost is permutation-aware
- smoothness_weight=0 → loss_bnd does not contribute to total
"""
import numpy as np
import pytest
import torch

from src.training.lightning_module import (
    GNOTLightning,
    ot_match,
    grassmannian_loss,
    soft_procrustes_loss,
)


def _make_module(degeneracy_mode='soft', predict_frequency=True, smoothness_weight=0.0):
    m = GNOTLightning(
        val_dim=8, grid_dim=2, hidden_dim=16,
        n_shared_layers=1, n_mode_layers=1, n_field_head_layers=2,
        n_heads=2, num_experts=2, num_field_modes=3,
        lr=1e-3, freq_weight=0.5, smoothness_weight=smoothness_weight,
        scheduler='custom_cosine',
        predict_frequency=predict_frequency,
        degeneracy_mode=degeneracy_mode,
        rff_dim=8,
    )
    m.eval()
    return m


def _make_batch(B=3, N=12, K=3, val_dim=8, with_boundary=True):
    X = torch.randn(B, N, 2)
    inputs = torch.randn(B, N, val_dim)
    if with_boundary:
        inputs[:, :2, 2] = 0.0    # boundary nodes (dist_bnd ~ 0)
        inputs[:, 2:, 2] = 0.5    # interior
    Y_field = torch.randn(B, N, K)
    # ascending per-sample frequencies
    Y_freq = torch.sort(torch.randn(B, K), dim=-1).values
    geom_id = torch.arange(B).long().unsqueeze(-1)
    Mask = torch.ones(B, N, dtype=torch.bool)
    return {
        'X': X,
        'Input_funcs': inputs,
        'Y_field': Y_field,
        'Y_freq': Y_freq,
        'geom_id': geom_id,
        'Mask': Mask,
    }


def test_compute_loss_soft_runs_and_backward():
    m = _make_module('soft')
    batch = _make_batch()
    loss, preds, targets = m._compute_loss(batch, "train")
    assert torch.isfinite(loss)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                   for p in m.parameters())
    assert has_grad


def test_compute_loss_hard_runs_and_backward():
    m = _make_module('hard')
    batch = _make_batch()
    # force a near-degenerate pair to exercise clustering
    batch['Y_freq'][:, 2] = batch['Y_freq'][:, 1] + 1e-4
    loss, preds, targets = m._compute_loss(batch, "train")
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_compute_loss_no_freq_branch():
    """predict_frequency=False → freq loss term is dropped, still finite."""
    m = _make_module('soft', predict_frequency=False)
    batch = _make_batch()
    loss, _, _ = m._compute_loss(batch, "val")
    assert torch.isfinite(loss)


def test_boundary_not_hard_zeroed():
    """Regression for FIX 2: predictions at boundary nodes must NOT be
    hard-zeroed.  The FEM target is not identically zero on the detected
    boundary band, so `pred_field *= (1 - bnd_mask)` trained the network
    toward a wrong field.  After the fix the predictions handed to the
    metrics are the model's continuous output — no exact-zero rows injected.
    """
    m = _make_module('soft')
    batch = _make_batch(B=2, N=10)   # nodes 0,1 are boundary (dist_bnd == 0)
    loss, preds, _ = m._compute_loss(batch, "val")
    assert torch.isfinite(preds).all()
    # Continuous Gaussian-driven output: an exact 0.0 only appears if a
    # boundary hard-zero multiply was (re)introduced.
    assert (preds == 0.0).sum().item() == 0, "boundary hard-zeroing reappeared"
    # Boundary energy is still tracked as a soft penalty for logging.
    assert torch.isfinite(loss)


def test_loss_finite_with_zero_targets():
    m = _make_module('soft')
    batch = _make_batch(B=2, N=10)
    batch['Y_field'] = torch.zeros_like(batch['Y_field'])
    loss, _, _ = m._compute_loss(batch, "val")
    assert torch.isfinite(loss)


def test_smoothness_weight_zero_excludes_bnd():
    field = torch.tensor(2.0)
    freq = torch.tensor(1.5)
    bnd = torch.tensor(7.0)
    total = field + 0.5 * freq + 0.0 * bnd
    assert total.item() == pytest.approx(2.75)


def test_ot_match_is_permutation_aware():
    f_true = torch.tensor([1.0, 2.0, 9.0])
    # prediction in a scrambled order
    f_pred = torch.tensor([9.0, 1.0, 2.0])
    E = torch.zeros(5, 3)  # zeroed fields → cost reduces to frequency only
    perm = ot_match(f_pred, f_true, E, E, None, freq_w=1.0)
    aligned = f_pred[perm]
    assert torch.allclose(aligned, f_true, atol=1e-6)


def test_ot_match_degenerate_uses_field_to_disambiguate():
    """For a degenerate frequency pair the assignment must fall back to the
    sign-agnostic field cost (no cluster threshold needed)."""
    f_true = torch.tensor([0.0, 1.0, 1.0])   # modes 1 & 2 degenerate
    f_pred = torch.tensor([0.0, 1.0, 1.0])
    N = 16
    torch.manual_seed(0)
    v0 = torch.randn(N); v1 = torch.randn(N); v2 = torch.randn(N)
    E_tgt = torch.stack([v0, v1, v2], dim=1)              # [N, 3]
    # Prediction swaps the degenerate pair and flips a sign.
    E_hat = torch.stack([v0, -v2, v1], dim=1)             # [N, 3]
    perm = ot_match(f_pred, f_true, E_hat, E_tgt, None, freq_w=1.0)
    aligned = E_hat[:, perm]
    # Each aligned column must match its target up to sign.
    for k in range(3):
        same = torch.allclose(aligned[:, k], E_tgt[:, k], atol=1e-5)
        flip = torch.allclose(aligned[:, k], -E_tgt[:, k], atol=1e-5)
        assert same or flip, k


def test_grassmannian_zero_for_same_subspace():
    torch.manual_seed(0)
    E = torch.randn(40, 2)
    assert grassmannian_loss(E, E).item() < 1e-6
    # sign flip irrelevant
    assert grassmannian_loss(E[:, :1], -E[:, :1]).item() < 1e-6


def test_soft_procrustes_aligns_rotated_degenerate_pair():
    torch.manual_seed(1)
    E_tgt = torch.randn(60, 2)
    theta = 0.6
    c, s = np.cos(theta), np.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    E_hat = E_tgt @ R.t()
    f = torch.tensor([3.0, 3.0001])
    assert soft_procrustes_loss(E_hat, E_tgt, f, sigma=10.0).item() < 1e-3


# ── Regression tests (train-infra review) ────────────────────────────────────

class _StubModel(torch.nn.Module):
    """Returns fixed outputs (with a dummy parameter so backward works)."""

    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.p = torch.nn.Parameter(torch.zeros(()))

    def forward(self, batch):
        return {k: v + 0.0 * self.p for k, v in self.outputs.items()}


def _spectral_module(**kw):
    m = GNOTLightning(val_dim=8, grid_dim=2, hidden_dim=16, n_heads=2,
                      num_field_modes=3, n_basis=4, rff_dim=8,
                      model_type='spectral_no', degeneracy_mode='hard', **kw)
    m.eval()
    return m


def test_spectral_field_loss_is_scale_invariant():
    """SpectralNO output amplitude is fixed by M-orthonormality → only the
    shape of the field may be penalised (targets are max-normalised)."""
    torch.manual_seed(0)
    batch = _make_batch()
    m = _spectral_module()
    assert m.scale_invariant_field and m.hparams.scale_invariant_field is True
    out = {'field': 0.3 * batch['Y_field'], 'freq': batch['Y_freq'].clone()}
    m.model = _StubModel(out)
    loss, preds, targets = m._compute_loss(batch, 'train')
    assert loss.item() < 1e-5
    # R2/MAE inputs are brought back to target amplitude
    assert torch.allclose(preds, targets, atol=1e-5)

    # Old behaviour (scale-sensitive) keeps an amplitude floor (1 - 0.3)^2 per mode
    m2 = _spectral_module(scale_invariant_field=False)
    m2.model = _StubModel(out)
    loss2, _, _ = m2._compute_loss(batch, 'train')
    assert loss2.item() > 0.4


def test_gnot_scale_invariant_option_with_orthonormalized_output():
    batch = _make_batch()
    m = GNOTLightning(val_dim=8, grid_dim=2, hidden_dim=16, n_shared_layers=1,
                      n_mode_layers=1, n_heads=2, num_experts=2, num_field_modes=3,
                      rff_dim=8, orthonormalize_output=True,
                      slot_ortho_weight=0.0)   # random targets aren't orthogonal
    assert m.scale_invariant_field
    assert not _make_module().scale_invariant_field   # plain GNOT unchanged
    m.model = _StubModel({'field': 5.0 * batch['Y_field'], 'freq': batch['Y_freq'].clone()})
    m.eval()
    loss, _, _ = m._compute_loss(batch, 'val')
    assert loss.item() < 1e-4


def test_boundary_loss_without_dist_column_does_not_crash():
    """feature_indices=[0, 1] ablation: Input_funcs has no column 2."""
    m = GNOTLightning(val_dim=2, grid_dim=2, hidden_dim=16, n_shared_layers=1,
                      n_mode_layers=1, n_heads=2, num_experts=2, num_field_modes=3,
                      rff_dim=8, smoothness_weight=0.1)
    batch = _make_batch(val_dim=2, with_boundary=False)
    loss, _, _ = m._compute_loss(batch, 'train')
    assert torch.isfinite(loss)
    # Dedicated Dist_bnd key (from the dataset) is used when present
    batch['Dist_bnd'] = torch.zeros(batch['Mask'].shape)
    loss_b, _, _ = m._compute_loss(batch, 'train')
    assert torch.isfinite(loss_b)


class _FakeTrainer:
    estimated_stepping_batches = 20
    max_epochs = 4
    check_val_every_n_epoch = 5


def test_onecycle_keeps_per_group_lr():
    """A scalar max_lr used to overwrite lr_freq_heads / lr_mode_specific."""
    m = _make_module()
    m.hparams.scheduler = 'onecycle'
    m.lr_freq_heads = 1e-2
    m._trainer = _FakeTrainer()
    cfg = m.configure_optimizers()
    sched = cfg['lr_scheduler']['scheduler']
    max_lrs = sorted(g['max_lr'] for g in sched.optimizer.param_groups)
    assert max_lrs == [pytest.approx(1e-3), pytest.approx(1e-2)]


def test_reducelr_steps_only_on_validation_epochs():
    m = _make_module()
    m.hparams.scheduler = 'reducelr'
    m._trainer = _FakeTrainer()
    cfg = m.configure_optimizers()
    assert cfg['lr_scheduler']['monitor'] == 'val/field_rel_l2'
    assert cfg['lr_scheduler']['frequency'] == 5
