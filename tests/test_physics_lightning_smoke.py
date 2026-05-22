"""Smoke test: GNOTLightning with physics losses enabled runs end-to-end.

Covers the autograd-Rayleigh path, hard Gram-Schmidt forward, and the
curriculum scheduler.  This is the integration-level proof that the
plan's hybrid loss survives backprop through the model.
"""
from __future__ import annotations

import pytest
import torch

from src.training.lightning_module import GNOTLightning


def _make_physics_module():
    m = GNOTLightning(
        val_dim=8, grid_dim=2, hidden_dim=16,
        n_shared_layers=0, n_mode_layers=1, n_field_head_layers=2,
        n_heads=2, num_experts=2, num_field_modes=3,
        lr=1e-3, freq_weight=0.5, smoothness_weight=0.0,
        scheduler='custom_cosine',
        predict_frequency=True,
        degeneracy_mode='soft',
        rff_dim=8,
        # Physics-informed
        enable_physics_loss=True,
        enable_curriculum=True,
        curriculum_e1=2, curriculum_e2=5,
        rayleigh_weight=0.05,
        rayleigh_mode='autograd',
        param_rayleigh_weight=0.1,
        use_gram_schmidt=True,
        order_weight=0.01,
        order_margin=0.01,
        length_scale=0.1,
    )
    m.freq_stats = {'mean': 5.0, 'std': 1.5}  # GHz-scale
    m.eval()
    return m


def _make_batch(B=2, N=24, K=3):
    X = torch.rand(B, N, 2)
    inputs = torch.rand(B, N, 8)
    # Ensure node_area column (5) is sensible
    inputs[..., 5] = 1.0 / float(N)
    Y_field = torch.randn(B, N, K)
    Y_freq = torch.sort(torch.randn(B, K), dim=-1).values
    geom_id = torch.arange(B).long().unsqueeze(-1)
    Mask = torch.ones(B, N, dtype=torch.bool)
    return {
        'X': X, 'Input_funcs': inputs,
        'Y_field': Y_field, 'Y_freq': Y_freq,
        'geom_id': geom_id, 'Mask': Mask,
    }


def _force_epoch(m, epoch):
    """Override curriculum to evaluate at a chosen epoch without a Trainer."""
    real_weights = m.curriculum.weights
    m.curriculum.weights = lambda _e: real_weights(epoch)


def test_physics_module_phase_a_loss_finite():
    m = _make_physics_module()
    _force_epoch(m, 0)  # Phase A: physics only
    batch = _make_batch()
    loss, _, _ = m._compute_loss(batch, "train")
    assert torch.isfinite(loss), f"Phase-A loss not finite: {loss}"
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0
               for p in m.model.parameters())


def test_physics_module_phase_c_loss_finite():
    m = _make_physics_module()
    _force_epoch(m, 10)  # Phase C: supervised dominant + small physics anchor
    batch = _make_batch()
    loss, _, _ = m._compute_loss(batch, "train")
    assert torch.isfinite(loss)
    loss.backward()


def test_gram_schmidt_forward_produces_orthogonal_columns():
    """After GS in forward, the K predicted mode columns are area-orthogonal."""
    m = _make_physics_module()
    batch = _make_batch(B=1, N=64, K=3)
    out = m.model({k: v for k, v in batch.items()})
    field = out['field']                             # [1, 64, 3]
    area = batch['Input_funcs'][..., 5]              # [1, 64]
    # Pairwise area-weighted inner products
    G = torch.einsum('bnk,bnl,bn->bkl', field, field, area)
    eye = torch.eye(3).unsqueeze(0)
    off = (G * (1 - eye)).abs().max().item()
    # GS is applied → off-diagonals should be at numerical-precision zero
    assert off < 1e-4, f"off-diag inner products too large after GS: {off}"
