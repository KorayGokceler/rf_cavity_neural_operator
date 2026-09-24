"""Tests for SpectralNO (physical-Galerkin eigendecomposition neural operator).

Invariants:
- Shapes, finiteness, forward/backward on CPU with a scaled-down
  configs/spectral_no.yaml
- Padding invariance and node-permutation equivariance
- Works under torch.inference_mode (Lightning validation/test default)
- Stiffness L is assembled from the *total* spatial gradient of the basis
  (coordinate feature columns + Dirichlet gate included)
- Generalized eigensolve: M-orthonormal, residual ~ 0, finite gradients for
  degenerate spectra, exact gradients for separated spectra
- Frequencies are monotone in the eigenvalue (sorted like the targets)
- Output fields follow the target amplitude convention (signed peak = +1)
- Eval-mode backward sees the same gradient as train mode (dropout=0)
- AMP (bf16 autocast) does not break the eigensolve
"""
import math
from pathlib import Path

import pytest
import torch

from src.config import load_config
from src.models.spectral_no import SpectralNO, _BroadenedEigh

REPO_ROOT = Path(__file__).resolve().parent.parent


def _small_model(val_dim=8, n_basis=6, K=3, dropout=0.0, seed=0, **kw):
    torch.manual_seed(seed)
    return SpectralNO(val_dim=val_dim, grid_dim=2, embed_dim=16, n_basis=n_basis,
                      num_field_modes=K, rff_dim=8, rff_length_scale=0.5,
                      dropout=dropout, **kw)


def _disk_batch(B=2, N=40, val_dim=8, seed=0):
    """Random points in the unit disk with geometrically consistent features:
    cols 0-1 = X, col 2 = dist to the unit circle, cols 3-4 = unit direction
    to the nearest boundary point, col 5 = area (> 0), rest random."""
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(B, N, generator=g).sqrt() * 0.98
    t = torch.rand(B, N, generator=g) * 2 * math.pi
    X = torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)
    Y = torch.rand(B, N, val_dim, generator=g)
    Y[..., 0:2] = X
    Y[..., 2] = 1.0 - r
    Y[..., 3:5] = X / r.unsqueeze(-1).clamp(min=1e-6)
    Y[..., 5] = 0.5 + torch.rand(B, N, generator=g)
    return {'X': X, 'Input_funcs': Y, 'Mask': torch.ones(B, N, dtype=torch.bool)}


def _pad(batch, n_extra):
    """Append n_extra padded (masked-out, garbage-valued) nodes."""
    B = batch['X'].shape[0]
    out = {}
    out['X'] = torch.cat([batch['X'], torch.randn(B, n_extra, 2)], dim=1)
    out['Input_funcs'] = torch.cat(
        [batch['Input_funcs'], torch.randn(B, n_extra, batch['Input_funcs'].shape[-1])], dim=1)
    out['Mask'] = torch.cat([batch['Mask'], torch.zeros(B, n_extra, dtype=torch.bool)], dim=1)
    return out


# ─── Shapes / config smoke ───────────────────────────────────────────────────

def test_forward_shapes_and_finite():
    model = _small_model()
    out = model(_disk_batch(B=3, N=30))
    assert out['field'].shape == (3, 30, 3)
    assert out['freq'].shape == (3, 3)
    assert out['eigenvalues'].shape == (3, 3)
    assert out['L_mat'].shape == (3, 6, 6)
    assert out['M_mat'].shape == (3, 6, 6)
    assert out['u_K'].shape == (3, 6, 3)
    for v in out.values():
        assert v.dtype == torch.float32
        assert torch.isfinite(v).all()


def test_spectral_config_forward_backward_cpu():
    """configs/spectral_no.yaml (scaled down), train-mode forward + backward."""
    mc = load_config(str(REPO_ROOT / 'configs' / 'spectral_no.yaml')).model
    torch.manual_seed(0)
    model = SpectralNO(
        val_dim=mc.val_dim, grid_dim=mc.grid_dim, embed_dim=16,
        n_basis=mc.n_basis, num_field_modes=mc.num_field_modes,
        rff_dim=8, rff_length_scale=mc.rff_length_scale, n_heads=mc.n_heads,
        dropout=mc.dropout, use_checkpoint=mc.use_checkpoint, bc_scale=mc.bc_scale,
    ).train()
    batch = _disk_batch(B=2, N=48, val_dim=mc.val_dim)
    batch['Mask'][1, 30:] = False
    out = model(batch)
    loss = out['field'].pow(2).mean() + out['freq'].pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


def test_val_dim_mismatch_raises_clear_error():
    """The converter writes 12 features; a config saying val_dim=8 must fail
    with an actionable message, not an opaque matmul shape error."""
    model = _small_model(val_dim=8)
    with pytest.raises(ValueError, match="val_dim=8"):
        model(_disk_batch(val_dim=12))


def test_twelve_feature_input():
    model = _small_model(val_dim=12)
    out = model(_disk_batch(val_dim=12))
    assert torch.isfinite(out['field']).all()


# ─── Masking / symmetry ──────────────────────────────────────────────────────

@pytest.mark.parametrize("train", [False, True])
def test_padding_invariance(train):
    model = _small_model().train(train)
    batch = _disk_batch(B=2, N=30)
    with torch.no_grad():
        o1 = model(batch)
        o2 = model(_pad(batch, 9))
    assert torch.allclose(o1['field'], o2['field'][:, :30], atol=1e-5)
    assert torch.allclose(o1['freq'], o2['freq'], atol=1e-5)
    assert torch.allclose(o1['eigenvalues'], o2['eigenvalues'], rtol=1e-5)
    assert o2['field'][:, 30:].abs().max() == 0


def test_permutation_equivariance():
    model = _small_model().eval()
    batch = _disk_batch(B=2, N=30)
    perm = torch.randperm(30)
    with torch.no_grad():
        o1 = model(batch)
        o2 = model({k: v[:, perm] for k, v in batch.items()})
    assert torch.allclose(o1['field'][:, perm], o2['field'], atol=1e-5)
    assert torch.allclose(o1['freq'], o2['freq'], atol=1e-5)


def test_eval_mode_deterministic_with_dropout():
    model = _small_model(dropout=0.3).eval()
    batch = _disk_batch()
    with torch.no_grad():
        o1, o2 = model(batch), model(batch)
    assert torch.equal(o1['field'], o2['field'])
    assert torch.equal(o1['freq'], o2['freq'])


# ─── Inference mode (Lightning validate/test) ────────────────────────────────

def test_forward_under_inference_mode():
    """Lightning runs validation/test under torch.inference_mode(), where
    torch.enable_grad() alone does not record a graph -> autograd.grad for
    the stiffness used to raise 'does not require grad'."""
    model = _small_model().eval()
    batch = _disk_batch()
    with torch.no_grad():
        ref = model(batch)
    with torch.inference_mode():
        inf_batch = {k: v.clone() for k, v in batch.items()}  # inference tensors
        out = model(inf_batch)
    assert torch.allclose(out['field'], ref['field'], atol=1e-6)
    assert torch.allclose(out['freq'], ref['freq'], atol=1e-6)


# ─── Physics of the Galerkin assembly ────────────────────────────────────────

def test_stiffness_uses_total_spatial_gradient():
    """Half-plane y > -1: dist = y + 1, dir = (0, -1) (constant), all other
    features constant, so the basis is an explicit function of position.
    L must equal Σ w ∇ψ·∇ψ with the *total* gradient — including the (x, y)
    feature columns and the Dirichlet gate 2σ(d/s) − 1, whose slope at ∂Ω
    is 1/(2s) = 25.  The old code differentiated only the RFF path."""
    torch.manual_seed(0)
    model = _small_model(n_basis=5, bc_scale=0.02).eval()
    B, N, V = 1, 50, 8
    X = torch.rand(B, N, 2) * torch.tensor([1.0, 0.1]) + torch.tensor([-0.5, -1.0])
    X[:, :8, 1] = -1.0            # boundary nodes (d = 0), where the gate slope peaks

    def features(X):
        Y = torch.zeros(B, N, V)
        Y = Y.index_copy(-1, torch.tensor([0, 1]), X)
        Y[..., 2] = X[..., 1] + 1.0
        Y[..., 4] = -1.0          # dir to nearest boundary point (0, -1) ...
        Y[:, :8, 4] = 0.0         # ... but zero ON the boundary, as the converter writes it
        Y[..., 5] = 1.0           # uniform area
        return Y

    with torch.no_grad():
        out = model({'X': X, 'Input_funcs': features(X)})

    # Reference: ψ(x) as an explicit function of position, autograd through all.
    Xr = X.clone().requires_grad_(True)
    Yr = torch.cat([Xr, Xr[..., 1:2] + 1.0, features(X)[..., 3:]], dim=-1)
    h = model.local_encoder(torch.cat([model.spatial_encoder(Xr), Yr], dim=-1))
    psi = model.basis_net(h) * (2 * torch.sigmoid((Xr[..., 1:2] + 1.0) / model.bc_scale) - 1)
    grads = torch.stack([torch.autograd.grad(psi[..., m].sum(), Xr, retain_graph=True)[0]
                         for m in range(model.M)], dim=2)            # [B, N, M, 2]
    w = torch.full((B, N), 1.0 / N)
    M_ref = torch.einsum('bnm,bnl,bn->bml', psi, psi, w)
    ridge = M_ref.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * torch.eye(model.M)
    L_ref = torch.einsum('bnmd,bnld,bn->bml', grads, grads, w) + model.stiff_ridge * ridge
    M_ref = M_ref + model.mass_ridge * ridge

    assert torch.allclose(out['M_mat'], M_ref.detach(), rtol=1e-4, atol=1e-6)
    assert torch.allclose(out['L_mat'], L_ref.detach(), rtol=1e-3, atol=1e-5), \
        (out['L_mat'].diagonal(dim1=-2, dim2=-1), L_ref.diagonal(dim1=-2, dim2=-1))


def test_generalized_eigenpairs_residual_and_m_orthonormal():
    model = _small_model(n_basis=6, K=3)
    with torch.no_grad():
        out = model(_disk_batch())
        vals, vecs = model._generalized_eigh_safe(out['L_mat'].double(), out['M_mat'].double())
    L, M = out['L_mat'].double(), out['M_mat'].double()
    gram = vecs.transpose(-1, -2) @ M @ vecs
    assert torch.allclose(gram, torch.eye(6, dtype=torch.float64).expand_as(gram), atol=1e-6)
    resid = L @ vecs - M @ vecs * vals.unsqueeze(-2)
    assert resid.abs().max() < 1e-6 * vals.abs().max()
    assert (vals[:, 1:] >= vals[:, :-1]).all()


# ─── Eigensolver gradients ───────────────────────────────────────────────────

def test_broadened_eigh_matches_exact_gradient_for_separated_spectrum():
    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(5, 5, dtype=torch.float64))
    base = Q @ torch.diag(torch.tensor([1., 2., 4., 7., 11.], dtype=torch.float64)) @ Q.T
    S = torch.randn(5, 5, dtype=torch.float64) * 0.01
    g_vec = torch.randn(5, 3, dtype=torch.float64)
    g_val = torch.randn(5, dtype=torch.float64)

    def loss(fn, S):
        A = base + S + S.T
        vals, vecs = fn(A)
        v = vecs[:, :3] * torch.sign(vecs[:1, :3])  # gauge-fix the sign
        return (v * g_vec).sum() + (vals * g_val).sum()

    S1 = S.clone().requires_grad_(True)
    S2 = S.clone().requires_grad_(True)
    loss(torch.linalg.eigh, S1).backward()
    loss(lambda A: _BroadenedEigh.apply(A, 1e-12), S2).backward()
    assert torch.allclose(S1.grad, S2.grad, atol=1e-8)


def test_degenerate_spectrum_gives_finite_gradients():
    """Exactly degenerate L (e.g. a dipole pair): plain eigh backward has
    1/(λ_i − λ_j) = inf terms -> NaN/inf gradients."""
    model = _small_model(n_basis=4, K=3)
    S = torch.zeros(2, 4, 4, dtype=torch.float64, requires_grad=True)
    L = torch.diag_embed(torch.tensor([[1., 2., 2., 5.], [3., 3., 3., 4.]],
                                      dtype=torch.float64)) + S + S.transpose(-1, -2)
    M = torch.eye(4, dtype=torch.float64).expand(2, 4, 4)
    vals, vecs = model._generalized_eigh_safe(L, M)
    (vecs[:, :, :3].sum() + vals[:, :3].sum()).backward()
    assert torch.isfinite(S.grad).all()


def test_near_degenerate_model_backward_finite():
    """End-to-end: an L with a near-degenerate pair must not blow up grads."""
    model = _small_model(n_basis=4, K=3)
    S = torch.zeros(1, 4, 4, dtype=torch.float64, requires_grad=True)
    L = torch.diag_embed(torch.tensor([[1., 2., 2. + 1e-12, 5.]], dtype=torch.float64)) + S
    vals, vecs = model._generalized_eigh_safe(L, torch.eye(4, dtype=torch.float64)[None])
    vecs[:, 0, :3].pow(2).sum().backward()
    assert torch.isfinite(S.grad).all()
    assert S.grad.abs().max() < 1e6


def test_eval_mode_backward_matches_train_mode():
    """create_graph used to be tied to self.training: in eval mode the
    stiffness path was silently detached from the parameters."""
    grads = {}
    for train in (True, False):
        model = _small_model(dropout=0.0).train(train)
        out = model(_disk_batch())
        out['eigenvalues'].sum().backward()
        grads[train] = torch.cat([p.grad.flatten() for p in model.basis_net.parameters()])
    assert torch.allclose(grads[True], grads[False], rtol=1e-4, atol=1e-7)


# ─── Frequency head / field normalization ────────────────────────────────────

@pytest.mark.parametrize("seed", range(8))
def test_freq_transform_monotone(seed):
    """Frequencies must be ordered like the eigenvalues (and the sorted
    targets).  A GELU MLP on raw λ is not monotone for random weights."""
    model = _small_model(seed=seed)
    lam = torch.logspace(-1, 4, 400).unsqueeze(-1)
    with torch.no_grad():
        f = model.freq_transform(lam).squeeze(-1)
    assert (f[1:] > f[:-1]).all()


def test_forward_freq_sorted_ascending():
    model = _small_model()
    with torch.no_grad():
        out = model(_disk_batch(B=4))
    assert (out['freq'][:, 1:] >= out['freq'][:, :-1]).all()


def test_fields_follow_target_peak_convention():
    """Targets are divided by max|Y| (dataset_converter).  M-orthonormal fields
    have Σ w φ² = 1 regardless of the parameters, so without rescaling the
    sign-agnostic rel-L2 has an irreducible floor (≈0.9 for a disk TM010)."""
    model = _small_model()
    batch = _disk_batch(B=2, N=40)
    batch['Mask'][1, 25:] = False
    with torch.no_grad():
        f = model(batch)['field']
    peak = f.abs().amax(dim=1)                                    # [B, K]
    assert torch.allclose(peak, torch.ones_like(peak), atol=1e-5)
    # sign is canonical: the peak value itself is +1
    assert torch.allclose(f.amax(dim=1), torch.ones_like(peak), atol=1e-5)


def test_eigenvalues_invariant_to_basis_scale():
    """Rayleigh–Ritz is invariant to rescaling the basis; with absolute
    ridges λ used to depend on the (non-physical) basis amplitude."""
    model = _small_model()
    batch = _disk_batch()
    with torch.no_grad():
        lam1 = model(batch)['eigenvalues']
        model.basis_net[-1].weight.mul_(10.0)
        model.basis_net[-1].bias.mul_(10.0)
        lam2 = model(batch)['eigenvalues']
    assert torch.allclose(lam1, lam2, rtol=1e-4)


# ─── Mixed precision ─────────────────────────────────────────────────────────

def test_bf16_autocast_forward_backward():
    """Under AMP the einsums ran in bf16 and linalg.cholesky has no bf16
    kernel; the Galerkin solve must run in full precision."""
    model = _small_model().train()
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        out = model(_disk_batch())
    assert torch.isfinite(out['field']).all()
    (out['field'].float().pow(2).mean() + out['freq'].float().pow(2).mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
