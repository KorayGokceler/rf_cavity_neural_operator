"""NEO-style eigenspace losses (span / compliance / logdet) and the
model_type='eigenspace' training integration (data modes > K).

Reference problems: P1 Dirichlet Laplacian on skfem disk / ellipse meshes,
eigenpairs from scipy eigsh.  CPU, a few seconds.
"""
import math

import numpy as np
import pytest
import torch

skfem = pytest.importorskip("skfem")
from scipy.sparse.linalg import eigsh                        # noqa: E402
from skfem.models.poisson import laplace, mass               # noqa: E402

from src.data.dataset import gnot_collate_fn                 # noqa: E402
from src.training.lightning_module import (                  # noqa: E402
    GNOTLightning, eigenspace_grams, ritz_compliance, ritz_logdet, span_residual)

_C0 = 299792458.0
_SCALE = 0.05          # physical length of one normalised unit [m]


def _p1_problem(refine=3, a=1.0, b=1.0, k=12):
    """Ellipse (a, b) P1 mesh: X, elements, dense M/A, interior-only eigenpairs
    (λ ascending, U [N, k] zero on the boundary, M-orthonormal)."""
    m0 = skfem.MeshTri.init_circle(refine)
    mesh = skfem.MeshTri(m0.p * np.array([[a], [b]]), m0.t)
    basis = skfem.Basis(mesh, skfem.ElementTriP1())
    A = laplace.assemble(basis)
    M = mass.assemble(basis)
    bnd = mesh.boundary_nodes()
    I = np.setdiff1d(np.arange(mesh.p.shape[1]), bnd)
    lam, UI = eigsh(A[I][:, I], k=k, M=M[I][:, I], sigma=0.0)
    order = np.argsort(lam)
    U = np.zeros((mesh.p.shape[1], k))
    U[I] = UI[:, order]
    return dict(X=mesh.p.T.copy(), E=mesh.t.T.copy(), M=M.toarray(), A=A.toarray(),
                lam=lam[order], U=U, bnd=bnd, I=I)


@pytest.fixture(scope="module")
def disk():
    return _p1_problem()


def _t(x):
    return torch.as_tensor(np.asarray(x))[None]


def _grams(P, V, T):
    return eigenspace_grams(_t(V), _t(T), _t(P['X']), _t(P['E']))


def _interior_noise(P, n, seed):
    """Smooth random fields that vanish on ∂Ω (random mixes of modes 6-11)."""
    rng = np.random.default_rng(seed)
    return P['U'][:, 6:12] @ rng.standard_normal((6, n))


# ── exact P1 Grams ────────────────────────────────────────────────────────────

def test_grams_equal_skfem_p1_matrices(disk):
    rng = np.random.default_rng(0)
    V = rng.standard_normal((disk['X'].shape[0], 5))     # not zero on ∂Ω: full matrices
    T = disk['U'][:, :3]
    G_M, G_A = _grams(disk, V, T)
    W = np.concatenate([V, T], axis=1)
    np.testing.assert_allclose(G_M[0].numpy(), W.T @ disk['M'] @ W, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(G_A[0].numpy(), W.T @ disk['A'] @ W, rtol=1e-10, atol=1e-10)


# ── span / projection loss ────────────────────────────────────────────────────

def test_span_zero_when_targets_in_span_any_sign_rotation_extra_columns(disk):
    rng = np.random.default_rng(1)
    T = disk['U'][:, :3] / np.abs(disk['U'][:, :3]).max(0)      # max-normalised like the data
    R = rng.standard_normal((3, 3))                              # arbitrary mixing (rotation, signs)
    extra = _interior_noise(disk, 4, 2)
    V = np.concatenate([T @ R, extra], axis=1)[:, rng.permutation(7)]
    for flip in (np.ones(3), np.array([-1.0, 1.0, -1.0])):
        G_M, G_A = _grams(disk, V, T * flip)
        for G in (G_M, G_A):
            assert float(span_residual(G, 7, ridge=1e-12).max()) < 1e-9
            assert float(span_residual(G, 7).max()) < 1e-6          # default ridge
    # basis-gauge invariance: V → V S (rotation + column scales) gives the same residual
    Tn = disk['U'][:, 2:8] + 0.1 * _interior_noise(disk, 6, 3)
    S = np.linalg.qr(rng.standard_normal((7, 7)))[0] @ np.diag(np.logspace(-1, 1, 7))
    r1 = span_residual(_grams(disk, V, Tn)[0], 7, ridge=1e-12)
    r2 = span_residual(_grams(disk, V @ S, Tn)[0], 7, ridge=1e-12)
    np.testing.assert_allclose(r1.numpy(), r2.numpy(), atol=1e-8)


def test_span_mean_invariant_to_rotating_orthonormal_targets(disk):
    rng = np.random.default_rng(3)
    T = disk['U'][:, :4]                                         # M-orthonormal targets
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    V = disk['U'][:, :3] + 0.1 * _interior_noise(disk, 3, 4)
    r1 = span_residual(_grams(disk, V, T)[0], 3, ridge=1e-12).mean()
    r2 = span_residual(_grams(disk, V, T @ Q)[0], 3, ridge=1e-12).mean()
    assert abs(float(r1) - float(r2)) < 1e-10


def test_span_positive_and_decreasing_as_span_improves(disk):
    T = disk['U'][:, :6]
    N = _interior_noise(disk, 6, 5)
    prev_M, prev_A = np.inf, np.inf
    for eps in (1.0, 0.3, 0.1, 0.03, 0.01):
        G_M, G_A = _grams(disk, T + eps * N, T)
        rM = float(span_residual(G_M, 6).mean())
        rA = float(span_residual(G_A, 6).mean())
        assert 0.0 < rM < prev_M and 0.0 < rA < prev_A
        prev_M, prev_A = rM, rA
    # nested spans of the true eigenvectors: capture grows with the dimension
    rs = [float(span_residual(_grams(disk, disk['U'][:, :j], T)[0], j).mean())
          for j in (1, 3, 5, 6)]
    assert all(x > y for x, y in zip(rs, rs[1:])) and rs[-1] < 1e-5


def test_span_gradient_finite_on_degenerate_targets(disk):
    # Disk modes 1-2 are (near-)degenerate: no cluster threshold, no eigh → finite grads.
    V = torch.tensor(disk['U'][:, :6] + 0.05 * _interior_noise(disk, 6, 6))[None].requires_grad_(True)
    G_M, G_A = eigenspace_grams(V, _t(disk['U'][:, :6]), _t(disk['X']), _t(disk['E']))
    (span_residual(G_M, 6).mean() + span_residual(G_A, 6).mean()).backward()
    assert torch.isfinite(V.grad).all() and V.grad.abs().sum() > 0


# ── label-free compliance / logdet ────────────────────────────────────────────

def test_compliance_is_maximised_by_lowest_eigenvectors(disk):
    m = 5                                                        # 1 + 2 + 2: complete clusters
    lam = disk['lam']
    rng = np.random.default_rng(7)
    cands = {
        'lowest': disk['U'][:, :m],
        'higher': disk['U'][:, m:2 * m],
        'random': disk['U'][:, :12] @ rng.standard_normal((12, m)),
        'rotated_lowest_plus_noise': disk['U'][:, :m] @ rng.standard_normal((m, m))
        + 0.05 * _interior_noise(disk, m, 8),
    }
    comp, logd = {}, {}
    for name, V in cands.items():
        G_M, G_A = _grams(disk, V, V[:, :1])
        comp[name] = float(ritz_compliance(G_M[:, :m, :m], G_A[:, :m, :m], ridge=1e-12))
        logd[name] = float(ritz_logdet(G_M[:, :m, :m], G_A[:, :m, :m], ridge=1e-12))
    assert comp['lowest'] == pytest.approx(float((1.0 / lam[:m]).sum()), rel=1e-8)
    assert logd['lowest'] == pytest.approx(float(np.log(lam[:m]).mean()), rel=1e-8)
    for name in ('higher', 'random', 'rotated_lowest_plus_noise'):
        assert comp['lowest'] > comp[name]
        assert logd['lowest'] < logd[name]


def test_compliance_closed_form_equals_monte_carlo(disk):
    """E_b[(Mb)ᵀ V (VᵀAV)⁻¹ VᵀMb] with b ~ N(0, M⁻¹) == tr((VᵀAV)⁻¹ VᵀMV)."""
    m = 4
    rng = np.random.default_rng(9)
    V = disk['U'][:, :8] @ rng.standard_normal((8, m)) + 0.2 * _interior_noise(disk, m, 10)
    G_M, G_A = _grams(disk, V, V[:, :1])
    closed = float(ritz_compliance(G_M[:, :m, :m], G_A[:, :m, :m], ridge=0.0))

    M, A = disk['M'], disk['A']
    Lc = np.linalg.cholesky(M)
    z = rng.standard_normal((M.shape[0], 20000))
    b = np.linalg.solve(Lc.T, z)                                 # Cov(b) = L⁻ᵀL⁻¹ = M⁻¹
    Mb = M @ b
    x = V @ np.linalg.solve(V.T @ A @ V, V.T @ Mb)               # Galerkin solution in span(V)
    mc = (Mb * x).sum(0)                                         # compliance per load
    se = mc.std() / math.sqrt(mc.size)
    assert abs(mc.mean() - closed) < 4.0 * se
    assert abs(mc.mean() - closed) / closed < 0.03


# ── Lightning integration: model_type='eigenspace', data modes (6) > K (3) ─────

def _features(P):
    """13-column Input_funcs (current converter layout) for an ellipse mesh."""
    X = P['X']
    N = X.shape[0]
    rng = np.random.default_rng(N)
    F = rng.random((N, 13)).astype(np.float32)
    F[:, 0:2] = X
    bxy = X[P['bnd']]
    d = np.sqrt(((X[:, None, :] - bxy[None]) ** 2).sum(-1))
    F[:, 2] = d.min(1)
    F[P['bnd'], 2] = 0.0
    F[:, 3:5] = (bxy[d.argmin(1)] - X) / np.maximum(d.min(1), 1e-9)[:, None]
    F[:, 5] = P['M'].sum(1)                                      # lumped mass = node area
    a, b = np.abs(X).max(0)
    w = np.clip(1.0 - (X[:, 0] / a) ** 2 - (X[:, 1] / b) ** 2, 0.0, None)
    w[P['bnd']] = 0.0
    F[:, 12] = w / w.max()
    return F


class _EllipseDataset(torch.utils.data.Dataset):
    """Items in the GNOTDataset layout with n_modes stored target modes."""

    def __init__(self, n_modes=6, shapes=((1.0, 1.0), (1.2, 0.8), (0.9, 1.1), (1.1, 0.7))):
        probs = [_p1_problem(3, a, b, k=n_modes + 2) for a, b in shapes]
        f = [_C0 * np.sqrt(P['lam'][:n_modes]) / (2 * math.pi * _SCALE) / 1e9 for P in probs]
        allf = np.concatenate(f)
        self.stats = {'mean': float(allf.mean()), 'std': float(allf.std())}
        self.items = []
        for i, (P, fi) in enumerate(zip(probs, f)):
            U = P['U'][:, :n_modes]
            U = U / np.abs(U).max(0)
            dist = _features(P)[:, 2]
            self.items.append({
                'X': torch.tensor(P['X'], dtype=torch.float32),
                'Input_funcs': torch.tensor(_features(P)),
                'Y_field': torch.tensor(U, dtype=torch.float32),
                'Y_freq': torch.tensor((fi - self.stats['mean']) / self.stats['std'],
                                       dtype=torch.float32),
                'Dist_bnd': torch.tensor(dist, dtype=torch.float32),
                'Area': torch.tensor(P['M'].sum(1), dtype=torch.float32),
                'Scale': torch.tensor(_SCALE, dtype=torch.float32),
                'Elements': torch.tensor(P['E'], dtype=torch.int64),
                'geom_id': torch.tensor([i]),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


@pytest.fixture(scope="module")
def ellipses():
    return _EllipseDataset()


def _model(stats, **kw):
    torch.manual_seed(0)
    model = GNOTLightning(val_dim=13, hidden_dim=16, n_heads=2, num_field_modes=3, n_basis=8,
                          rff_dim=8, lr=1e-3, scheduler='none', model_type='eigenspace',
                          physics_freq=True, degeneracy_mode='hard',
                          near_deg_rel_threshold=0.05,
                          eigenspace_kwargs=dict(n_layers=1, torsion_feature_idx=12), **kw)
    model.freq_stats = stats
    return model


def _logged_loss(model, batch):
    logs = {}
    model.log = lambda name, val, **kw: logs.__setitem__(name, float(torch.as_tensor(val).detach()))
    loss, preds, targets = model._compute_loss(batch, 'train')
    return loss, preds, targets, logs


def test_data_modes_greater_than_K_are_sliced(ellipses):
    b6 = gnot_collate_fn([ellipses[i] for i in range(3)])
    b3 = dict(b6, Y_field=b6['Y_field'][..., :3], Y_freq=b6['Y_freq'][:, :3])
    # Post-Ritz terms only (span/selfsup/ortho off): the 3 extra stored
    # modes must not change anything that compares against the K=3 outputs.
    kw = dict(span_weight=0.0, selfsup_weight=0.0, ortho_weight=0.0,
              ritz_field_weight=1.0, freq_weight=0.5)
    l6, p6, t6, g6 = _logged_loss(_model(ellipses.stats, **kw), b6)
    l3, p3, t3, g3 = _logged_loss(_model(ellipses.stats, **kw), b3)
    assert torch.allclose(l6, l3) and torch.equal(t6, t3) and torch.allclose(p6, p3)
    for key in ('train/field_rel_l2', 'train/freq_mae_ghz', 'train/mode_2_rel_l2'):
        assert g6[key] == pytest.approx(g3[key])
    assert 'train/mode_3_rel_l2' not in g6
    # ...while the span loss sees every stored target mode.
    assert 'train/span_mode_5_rel_l2' in g6 and 'train/span_mode_3_rel_l2' not in g3

    loss, _, _, logs = _logged_loss(_model(ellipses.stats), b6)   # default weights
    loss.backward()
    assert torch.isfinite(loss) and 0.0 < logs['train/span_loss'] <= 1.0


def test_eigenspace_fit_validate_and_full_split_eval(ellipses, tmp_path):
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from scripts.infer_val_all import evaluate_split

    model = _model(ellipses.stats)
    loader = DataLoader(ellipses, batch_size=2, collate_fn=gnot_collate_fn)
    trainer = pl.Trainer(fast_dev_run=True, accelerator='cpu', enable_progress_bar=False,
                         enable_model_summary=False, logger=pl.loggers.CSVLogger(str(tmp_path)))
    trainer.fit(model, loader, loader)
    trainer.validate(model, loader, verbose=False)               # inference_mode=True
    assert torch.isfinite(trainer.callback_metrics['val/field_rel_l2'])
    assert 'val/span_mode_5_rel_l2' in trainer.callback_metrics
    assert 'val/mode_3_rel_l2' not in trainer.callback_metrics

    model.eval()
    agg, rows = evaluate_split(model, ellipses, 'cpu', 2, 0.05, None)
    assert agg['K'] == 3 and len(agg['per_mode_relL2']) == 3 and len(rows) == len(ellipses)
    assert all(np.isfinite(r['freq_mae_ghz']) for r in rows)


def test_train_eigenspace_kwargs_block_top_level_and_remap():
    from src.config import ConfigDict
    from train import _eigenspace_kwargs
    mc = ConfigDict({'n_layers': 2, 'eigenspace': {'n_layers': 3, 'torsion_feature_idx': 12}})
    assert _eigenspace_kwargs(mc, None) == {'n_layers': 3, 'torsion_feature_idx': 12}
    assert _eigenspace_kwargs(mc, [0, 1, 2, 5, 12]) == {'n_layers': 3, 'torsion_feature_idx': 4}
    assert _eigenspace_kwargs(ConfigDict({'n_layers': 2}), None) == {'n_layers': 2}
    assert _eigenspace_kwargs(ConfigDict({}), None) is None
