"""Tests for the changes from the math reviews (docs/14–17).

- P1 element assembly of SpectralNO equals scikit-fem's P1 mass/stiffness
- assembly='p1' gives true Rayleigh–Ritz upper bounds (ψ ∈ H¹₀)
- physics_freq: f = c·√λ / (2π·scale), exact 1/scale scaling
- GNOTLightning.freq_stats reaches the wrapped model
- area-weighted field loss down-weights errors on small-area nodes
"""
import math

import numpy as np
import pytest
import torch

from src.models.spectral_no import SpectralNO, _p1_galerkin
from src.training.lightning_module import GNOTLightning, _sqrt_area_weights

skfem = pytest.importorskip("skfem")

J01_SQ = 2.404825557695773 ** 2   # λ₁ of the unit disk


def _disk_mesh(refine=3):
    m = skfem.MeshTri.init_circle(refine)
    X = torch.tensor(m.p.T, dtype=torch.float64)            # [N, 2]
    elems = torch.tensor(m.t.T, dtype=torch.long)           # [T, 3]
    bnd = np.zeros(m.p.shape[1], dtype=bool)
    bnd[m.boundary_nodes()] = True
    return m, X, elems, torch.tensor(bnd)


def _disk_batch(refine=3, val_dim=12):
    _, X, elems, bnd = _disk_mesh(refine)
    N = X.shape[0]
    r = X.norm(dim=-1)
    Y = torch.zeros(1, N, val_dim)
    Y[0, :, 0:2] = X.float()
    Y[0, :, 2] = torch.where(bnd, torch.zeros_like(r), (1.0 - r).clamp(min=1e-3)).float()
    Y[0, :, 3:5] = (X / r.clamp(min=1e-6).unsqueeze(-1)).float()
    Y[0, :, 5] = 1.0
    return {'X': X.float().unsqueeze(0), 'Input_funcs': Y,
            'Mask': torch.ones(1, N, dtype=torch.bool), 'Elements': elems.unsqueeze(0)}


def test_p1_galerkin_matches_scikit_fem():
    from skfem.models.poisson import laplace, mass
    m, X, elems, _ = _disk_mesh(2)
    basis = skfem.Basis(m, skfem.ElementTriP1())
    K = skfem.asm(laplace, basis).toarray()
    Mm = skfem.asm(mass, basis).toarray()
    psi = torch.randn(X.shape[0], 4, dtype=torch.float64)
    M_mat, L_mat = _p1_galerkin(psi.unsqueeze(0), X.unsqueeze(0), elems.unsqueeze(0))
    P = psi.numpy()
    np.testing.assert_allclose(L_mat[0].numpy(), P.T @ K @ P, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(M_mat[0].numpy(), P.T @ Mm @ P, rtol=1e-10, atol=1e-10)


def test_p1_galerkin_padded_triangles_add_nothing():
    _, X, elems, _ = _disk_mesh(1)
    psi = torch.randn(1, X.shape[0], 3, dtype=torch.float64)
    padded = torch.cat([elems, torch.zeros(5, 3, dtype=torch.long)]).unsqueeze(0)
    a = _p1_galerkin(psi, X.unsqueeze(0), elems.unsqueeze(0))
    b = _p1_galerkin(psi, X.unsqueeze(0), padded)
    for u, v in zip(a, b):
        torch.testing.assert_close(u, v)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_p1_assembly_gives_upper_bounds(seed):
    """ψ = 0 on boundary nodes + exact P1 integrals ⇒ Ritz values ≥ λ_true."""
    torch.manual_seed(seed)
    model = SpectralNO(val_dim=12, embed_dim=16, n_basis=8, num_field_modes=3,
                       rff_dim=8, rff_length_scale=0.5, assembly='p1').eval()
    batch = _disk_batch(3)
    with torch.no_grad():
        out = model(batch)
    lam = out['eigenvalues'][0]
    assert torch.isfinite(lam).all()
    assert lam[0].item() >= J01_SQ * (1 - 1e-6)
    assert (lam[1:] >= lam[:-1] - 1e-6).all()


def test_p1_assembly_backward_and_missing_elements_error():
    torch.manual_seed(0)
    model = SpectralNO(val_dim=12, embed_dim=16, n_basis=6, num_field_modes=3,
                       rff_dim=8, assembly='p1')
    batch = _disk_batch(2)
    out = model(batch)
    (out['eigenvalues'].sum() + out['field'].pow(2).mean()).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    batch.pop('Elements')
    with pytest.raises(ValueError, match="Elements"):
        model(batch)


def test_physics_freq_formula_and_scale():
    torch.manual_seed(0)
    model = SpectralNO(val_dim=12, embed_dim=16, n_basis=6, num_field_modes=3,
                       rff_dim=8, assembly='p1', physics_freq=True).eval()
    assert model.freq_transform is None
    model.freq_stats = {'mean': 5.0, 'std': 2.0}
    batch = _disk_batch(2)
    batch['Scale'] = torch.tensor([0.04])
    with torch.no_grad():
        out = model(batch)
    lam = out['eigenvalues'][0].double()
    f_ghz = 299792458.0 * lam.sqrt() / (2 * math.pi * 0.04) / 1e9
    torch.testing.assert_close(out['freq'][0].double(), (f_ghz - 5.0) / 2.0,
                               rtol=1e-5, atol=1e-5)
    # Halving the size doubles every frequency (λ(sΩ) = λ(Ω)/s²).
    batch['Scale'] = torch.tensor([0.02])
    with torch.no_grad():
        out2 = model(batch)
    torch.testing.assert_close(out2['freq'][0] * 2.0 + 5.0,
                               2.0 * (out['freq'][0] * 2.0 + 5.0), rtol=1e-5, atol=1e-5)
    batch.pop('Scale')
    with pytest.raises(ValueError, match="Scale"):
        model(batch)


def test_freq_stats_reach_wrapped_model():
    m = GNOTLightning(val_dim=12, hidden_dim=16, n_heads=2, num_field_modes=3,
                      n_basis=4, rff_dim=8, model_type='spectral_no',
                      spectral_kwargs={'physics_freq': True})
    stats = {'mean': 5.0, 'std': 1.0}
    m.freq_stats = stats
    assert m.freq_stats is stats and m.model.freq_stats is stats


def test_area_weighted_field_loss_downweights_small_nodes():
    torch.manual_seed(0)
    B, N, K = 2, 30, 3
    mask = torch.ones(B, N, dtype=torch.bool)
    area = torch.ones(B, N)
    area[:, :10] = 1e-3                      # tiny wall-band nodes
    w = _sqrt_area_weights({'Area': area}, mask)
    assert w.shape == (B, N, 1)
    torch.testing.assert_close((w.squeeze(-1) ** 2).mean(dim=1), torch.ones(B))

    true = torch.randn(B, N, K)
    pred = true.clone()
    pred[:, :10] += 1.0                       # error only on tiny-area nodes
    batch = {'Area': area, 'Mask': mask}

    def field_err(weighted):
        m = GNOTLightning(val_dim=12, hidden_dim=16, n_heads=2, num_field_modes=K,
                          n_basis=4, rff_dim=8, model_type='spectral_no',
                          area_weighted_field=weighted)
        lp, lt, _ = m._field_loss_inputs(batch, pred, true, mask)
        return ((lp - lt) ** 2).sum().item()

    assert field_err(True) < 0.05 * field_err(False)


def test_gnot_physics_freq_scales_with_size():
    from src.models.gnot import GNOTModel
    torch.manual_seed(0)
    m = GNOTModel(val_dim=12, embed_dim=16, n_heads=2, n_mode_layers=1, num_experts=2,
                  num_field_modes=3, rff_dim=8, physics_freq=True).eval()
    m.freq_stats = {'mean': 5.0, 'std': 1.0}
    b = _disk_batch(1)
    b['Scale'] = torch.tensor([0.04])
    with torch.no_grad():
        f1 = m(b)['freq'] + 5.0
        b['Scale'] = torch.tensor([0.02])
        f2 = m(b)['freq'] + 5.0
    torch.testing.assert_close(f2, 2.0 * f1, rtol=1e-5, atol=1e-5)
    assert (f1 > 1.0).all() and (f1 < 20.0).all()   # untrained head starts in a GHz-sane range


def test_spectral_feature_columns_follow_feature_indices():
    from types import SimpleNamespace
    from train import _spectral_kwargs
    mc = SimpleNamespace(spectral={'assembly': 'p1'})
    assert _spectral_kwargs(mc, None) == {'assembly': 'p1'}
    kw = _spectral_kwargs(mc, [0, 1, 2, 5, 8])
    assert kw == {'assembly': 'p1', 'coord_feature_idx': [0, 1], 'dist_feature_idx': 2,
                  'dir_feature_idx': None, 'area_feature_idx': 3}
    kw = _spectral_kwargs(SimpleNamespace(), [2, 0, 1])
    assert kw['coord_feature_idx'] == [1, 2] and kw['dist_feature_idx'] == 0
    assert kw['area_feature_idx'] is None


def test_gnot_torsion_prior_starts_at_disk_spectrum():
    """Unit disk (max w = 1/4): the untrained head predicts the disk's own
    frequencies f_k = c·j_k / (2π·scale) from the torsion prior."""
    from src.models.gnot import GNOTModel
    torch.manual_seed(0)
    m = GNOTModel(val_dim=13, embed_dim=16, n_heads=2, n_mode_layers=1, num_experts=2,
                  num_field_modes=3, rff_dim=8, physics_freq=True).eval()
    m.freq_stats = {'mean': 0.0, 'std': 1.0}
    b = _disk_batch(2, val_dim=13)
    b['Scale'] = torch.tensor([0.04])
    b['TorsionMax'] = torch.tensor([0.25])
    with torch.no_grad():
        f = m(b)['freq'][0]
    j = torch.tensor([2.404826, 3.831706, 3.831706])
    torch.testing.assert_close(f, 299792458.0 * j / (2 * math.pi * 0.04) / 1e9, rtol=2e-2, atol=0.0)


def test_torsion_gate_keeps_upper_bound():
    torch.manual_seed(0)
    b = _disk_batch(3, val_dim=13)
    b['Input_funcs'][0, :, 12] = (1 - b['X'][0].pow(2).sum(-1)).clamp(min=0)
    m = SpectralNO(val_dim=13, embed_dim=16, n_basis=8, num_field_modes=3, rff_dim=8,
                   rff_length_scale=0.5, assembly='p1', torsion_feature_idx=12).eval()
    with torch.no_grad():
        lam = m(b)['eigenvalues'][0]
    assert lam[0].item() >= J01_SQ * (1 - 1e-6)
