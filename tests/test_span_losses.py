"""Span residual and label-free compliance / logdet of the eigenspace losses (src/training/
lightning_module.py), checked on a reference eigenproblem with known eigenvectors: the P1 Dirichlet
Laplacian on a disk (skfem). The losses only see Grams, so any SPD pencil serves as reference."""
import math

import numpy as np
import pytest
import torch

skfem = pytest.importorskip("skfem")
from scipy.sparse.linalg import eigsh                        # noqa: E402
from skfem.models.poisson import laplace, mass               # noqa: E402

from src.training.lightning_module import ritz_compliance, ritz_logdet, span_residual  # noqa: E402


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
    """Mass / stiffness Grams of [V | T] with the problem's dense matrices."""
    W = np.concatenate([V, T], axis=1)
    return _t(W.T @ P['M'] @ W), _t(W.T @ P['A'] @ W)


def _interior_noise(P, n, seed):
    """Smooth random fields that vanish on ∂Ω (random mixes of modes 6-11)."""
    rng = np.random.default_rng(seed)
    return P['U'][:, 6:12] @ rng.standard_normal((6, n))


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
    assert all(x > y for x, y in zip(rs, rs[1:], strict=False)) and rs[-1] < 1e-5


def test_span_gradient_finite_on_degenerate_targets(disk):
    # Disk modes 1-2 are (near-)degenerate: no cluster threshold, no eigh → finite grads.
    V = torch.tensor(disk['U'][:, :6] + 0.05 * _interior_noise(disk, 6, 6))[None].requires_grad_(True)
    W = torch.cat([V, _t(disk['U'][:, :6])], dim=-1)
    G_M, G_A = W.transpose(-1, -2) @ _t(disk['M']) @ W, W.transpose(-1, -2) @ _t(disk['A']) @ W
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
