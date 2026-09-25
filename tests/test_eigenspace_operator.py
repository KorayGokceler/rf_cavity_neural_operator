"""Tests for the NEO-style EigenspaceOperator (src/models/eigenspace_operator.py)."""
import math

import pytest
import torch

from src.models.eigenspace_operator import EigenspaceOperator, mass_weights
from src.models.gnot import LinearAttention

skfem = pytest.importorskip("skfem")
from tests.test_math_findings import J01_SQ, _disk_mesh  # noqa: E402

VAL_DIM, TORSION = 13, 12
KEYS = {'basis', 'field', 'eigenvalues', 'freq', 'M_mat', 'L_mat'}


def _disk_batch(refine=2, scale=0.05):
    _, X, elems, bnd = _disk_mesh(refine)
    N = X.shape[0]
    r = X.norm(dim=-1)
    dist = torch.where(bnd, torch.zeros_like(r), (1.0 - r).clamp(min=1e-3))
    e1, e2 = X[elems[:, 1]] - X[elems[:, 0]], X[elems[:, 2]] - X[elems[:, 0]]
    tri = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).abs()
    area = torch.zeros(N, dtype=torch.float64).index_add_(0, elems.reshape(-1), tri.repeat_interleave(3) / 3)
    Y = torch.zeros(N, VAL_DIM)
    Y[:, 0:2] = X.float()
    Y[:, 2] = dist.float()
    Y[:, 5] = area.float()
    Y[:, TORSION] = torch.where(bnd, torch.zeros_like(r), 1.0 - r * r).float()   # disk torsion w/max w
    return {'X': X.float()[None], 'Input_funcs': Y[None], 'Mask': torch.ones(1, N, dtype=torch.bool),
            'Elements': elems[None], 'Dist_bnd': dist.float()[None], 'Area': area.float()[None],
            'Scale': torch.tensor([scale])}


def _pad(batch, n_extra, t_extra):
    """Append zero-padded nodes and (0, 0, 0) triangles, as gnot_collate_fn does."""
    out = dict(batch)
    for k in ('X', 'Input_funcs', 'Mask', 'Dist_bnd', 'Area'):
        v = batch[k]
        out[k] = torch.cat([v, torch.zeros(v.shape[0], n_extra, *v.shape[2:], dtype=v.dtype)], dim=1)
    out['Elements'] = torch.cat([batch['Elements'], torch.zeros(1, t_extra, 3, dtype=torch.long)], dim=1)
    return out


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    m = EigenspaceOperator(val_dim=VAL_DIM, embed_dim=32, n_layers=2, n_heads=4, n_basis=8,
                           num_field_modes=3, rff_dim=16, **kw).eval()
    m.freq_stats = {'mean': 5.0, 'std': 2.0}
    return m


def test_shapes_and_contract_keys():
    model, batch = _model(), _disk_batch()
    N = batch['X'].shape[1]
    with torch.no_grad():
        out = model(batch)
    assert set(out) == KEYS
    assert out['basis'].shape == (1, N, 8) and out['field'].shape == (1, N, 3)
    assert out['eigenvalues'].shape == out['freq'].shape == (1, 3)
    assert out['M_mat'].shape == out['L_mat'].shape == (1, 8, 8)
    assert all(v.dtype == torch.float32 for v in out.values())
    lam = out['eigenvalues'][0].double()
    f_ghz = 299792458.0 * lam.sqrt() / (2 * math.pi * 0.05) / 1e9
    torch.testing.assert_close(out['freq'][0].double(), (f_ghz - 5.0) / 2.0, rtol=1e-5, atol=1e-5)
    # Peak-normalised: signed max over nodes is +1.
    torch.testing.assert_close(out['field'][0].max(0).values, torch.ones(3))


def test_padding_invariance():
    model, batch = _model(), _disk_batch()
    N = batch['X'].shape[1]
    with torch.no_grad():
        a, b = model(batch), model(_pad(batch, 9, 5))
    torch.testing.assert_close(b['eigenvalues'], a['eigenvalues'], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(b['field'][:, :N], a['field'], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(b['basis'][:, :N], a['basis'], rtol=1e-5, atol=1e-5)
    assert (b['field'][:, N:] == 0).all() and (b['basis'][:, N:] == 0).all()


def test_permutation_equivariance():
    model, batch = _model(), _disk_batch()
    N = batch['X'].shape[1]
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
    inv = torch.argsort(perm)
    pb = {k: (v[:, perm] if k in ('X', 'Input_funcs', 'Mask', 'Dist_bnd', 'Area') else v)
          for k, v in batch.items()}
    pb['Elements'] = inv[batch['Elements']]       # node i of the old mesh is inv[i] now
    with torch.no_grad():
        a, b = model(batch), model(pb)
    torch.testing.assert_close(b['eigenvalues'], a['eigenvalues'], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(b['field'], a['field'][:, perm], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("torsion", [None, TORSION])
def test_ritz_values_upper_bound_disk(seed, torsion):
    model, batch = _model(seed, torsion_feature_idx=torsion), _disk_batch(3)
    with torch.no_grad():
        lam = model(batch)['eigenvalues'][0]
    assert torch.isfinite(lam).all()
    assert lam[0].item() >= J01_SQ * (1 - 1e-6)
    assert (lam[1:] >= lam[:-1] - 1e-6).all()


def test_mass_aware_attention_split_node_invariance():
    """Splitting a node into two half-area copies leaves the quadrature unchanged."""
    torch.manual_seed(0)
    attn = LinearAttention(16, 4).eval()
    x, area = torch.randn(1, 10, 16), torch.rand(1, 10) + 0.1
    mask = torch.ones(1, 10, dtype=torch.bool)
    j = 3
    x2 = torch.cat([x, x[:, j:j + 1]], dim=1)
    area2 = torch.cat([area, area[:, j:j + 1] / 2], dim=1)
    area2[:, j] /= 2
    mask2 = torch.ones(1, 11, dtype=torch.bool)
    w = mass_weights({'X': x, 'Area': area}, mask)
    w2 = mass_weights({'X': x2, 'Area': area2}, mask2)
    with torch.no_grad():
        y, y2 = attn(x, x, x, mask, w), attn(x2, x2, x2, mask2, w2)
        y_plain = attn(x2, x2, x2, mask2)
    torch.testing.assert_close(y2[:, :10], y, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(y2[:, 10], y[:, j], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(y_plain[:, :10], y, atol=1e-4)   # unweighted sum is not invariant


def test_trunk_split_node_invariance_and_uniform_fallback():
    model, batch = _model(), _disk_batch()
    N, j = batch['X'].shape[1], 7
    split = {k: torch.cat([batch[k], batch[k][:, j:j + 1]], dim=1)
             for k in ('X', 'Input_funcs', 'Mask', 'Dist_bnd', 'Area')}
    split['Area'][:, [j, N]] = batch['Area'][:, j:j + 1] / 2
    with torch.no_grad():
        h, h2 = model.embed(batch), model.embed(split)
        torch.testing.assert_close(h2[:, :N], h, rtol=1e-4, atol=1e-5)
        uniform = dict(batch, Area=torch.ones_like(batch['Area']))
        no_area = {k: v for k, v in batch.items() if k != 'Area'}
        torch.testing.assert_close(model.embed(no_area), model.embed(uniform))


def test_finite_gradients():
    model, batch = _model(), _disk_batch()
    model.train()
    out = model(batch)
    loss = (out['eigenvalues'].log().sum() + out['field'].pow(2).mean()
            + out['freq'].pow(2).mean() + out['basis'].pow(2).mean())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert model.head.weight.grad.abs().sum() > 0
    assert model.blocks[0].attn.q_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("torsion", [None, TORSION])
def test_basis_exactly_zero_on_boundary(torsion):
    model, batch = _model(torsion_feature_idx=torsion), _disk_batch()
    bnd = batch['Dist_bnd'][0] <= 1e-9
    assert bnd.any()
    with torch.no_grad():
        out = model(batch)
    assert (out['basis'][0, bnd] == 0).all()
    assert (out['field'][0, bnd] == 0).all()
    assert (out['basis'][0, ~bnd] != 0).any()


@pytest.mark.parametrize("key", ['Elements', 'Dist_bnd', 'Scale'])
def test_missing_keys_raise(key):
    model, batch = _model(), _disk_batch()
    batch.pop(key)
    with pytest.raises(ValueError, match=key):
        model(batch)


def test_constructor_validation():
    with pytest.raises(ValueError, match="physics_freq"):
        EigenspaceOperator(val_dim=VAL_DIM, physics_freq=False)
    with pytest.raises(ValueError, match="n_basis"):
        EigenspaceOperator(val_dim=VAL_DIM, n_basis=2, num_field_modes=3)
