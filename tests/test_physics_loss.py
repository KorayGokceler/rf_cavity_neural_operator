"""Unit tests for src.training.physics_losses.

Anchored on the analytical 1D and 2D Laplace eigenproblem from
Rowan et al. (arXiv:2506.04375) Section 4: u_i(x) = √2·sin(iπx) gives
R[u_i] = i²π². We verify Rayleigh, Gram-Schmidt, ordering hinge, and
the curriculum scheduler against these closed-form references.
"""
from __future__ import annotations

import math
import pytest
import torch

from src.training.physics_losses import (
    rayleigh_quotient,
    gram_schmidt_modes,
    eigenvalue_ordering_loss,
    parametric_expected_rayleigh,
    PhysicsCurriculum,
)


# ── helpers ──

def _grid_1d(N: int = 256, x_max: float = 1.0):
    x = torch.linspace(0.0, x_max, N).view(1, N, 1).requires_grad_(True)
    return x


def _analytic_modes(x, K=3):
    # u_i(x) = sqrt(2) sin(i π x), Dirichlet on [0,1] → λ_i = (iπ)²
    cols = []
    for i in range(1, K + 1):
        cols.append(math.sqrt(2.0) * torch.sin(i * math.pi * x.squeeze(-1)))
    return torch.stack(cols, dim=-1)               # [1, N, K]


# ── Rayleigh ──

def test_rayleigh_autograd_matches_1d_eigenvalues():
    """R[sin(iπx)] = (iπ)² for i=1..3 within 1% on a 256-pt grid."""
    N, K = 256, 3
    x = _grid_1d(N)                                # [1, N, 1] grad-enabled
    u = _analytic_modes(x, K=K)                    # [1, N, K]
    # Trapezoidal area weights for [0,1]
    area = torch.ones(1, N) / (N - 1)
    area[..., 0] *= 0.5
    area[..., -1] *= 0.5

    R = rayleigh_quotient(u, coords=x, area=area, mode="autograd")
    expected = torch.tensor([[(math.pi) ** 2, (2 * math.pi) ** 2,
                              (3 * math.pi) ** 2]])
    rel = ((R - expected).abs() / expected)
    assert rel.max().item() < 0.01, f"R={R.tolist()} vs {expected.tolist()}"


def test_rayleigh_zero_field_safe():
    """A zero field must not crash and should return zero / eps-bounded."""
    N = 64
    x = torch.linspace(0, 1, N).view(1, N, 1).requires_grad_(True)
    u = torch.zeros(1, N, 2, requires_grad=True)
    # Force a dummy dependence so autograd works
    u = u + 0.0 * x
    R = rayleigh_quotient(u, coords=x, mode="autograd")
    assert torch.isfinite(R).all()


def test_rayleigh_fem_matches_autograd():
    """Sparse FEM path agrees with autograd path on a 1D toy stiffness/mass."""
    # 1D linear-element K, M for [0,1] split into n_el = N-1 equal segments.
    N = 64
    h = 1.0 / (N - 1)
    # Tridiagonal stiffness: K_ii = 2/h (interior), K_{i,i+1}=K_{i+1,i}=-1/h
    K = torch.zeros(N, N)
    M = torch.zeros(N, N)
    for i in range(N - 1):
        K[i, i] += 1.0 / h
        K[i + 1, i + 1] += 1.0 / h
        K[i, i + 1] -= 1.0 / h
        K[i + 1, i] -= 1.0 / h
        # Consistent mass (linear elements):
        # M_local = h/6 * [[2,1],[1,2]]
        M[i, i] += 2.0 * h / 6.0
        M[i + 1, i + 1] += 2.0 * h / 6.0
        M[i, i + 1] += h / 6.0
        M[i + 1, i] += h / 6.0

    K_sp = K.to_sparse_csr()
    M_sp = M.to_sparse_csr()

    x_lin = torch.linspace(0, 1, N)
    K_modes = 3
    cols = [math.sqrt(2.0) * torch.sin(i * math.pi * x_lin) for i in range(1, K_modes + 1)]
    field = torch.stack(cols, dim=-1).unsqueeze(0)  # [1, N, K]

    R_fem = rayleigh_quotient(field, mode="fem", K_sparse=[K_sp], M_sparse=[M_sp])
    expected = torch.tensor([[(math.pi) ** 2, (2 * math.pi) ** 2,
                              (3 * math.pi) ** 2]])
    rel = ((R_fem - expected).abs() / expected)
    # FEM linear elements give slight discretization error; allow 5%
    assert rel.max().item() < 0.05, f"R_fem={R_fem.tolist()} vs {expected.tolist()}"


# ── Gram-Schmidt ──

def test_gram_schmidt_produces_orthogonal_columns():
    """After GS, ⟨u_i, u_j⟩_area ≈ 0 for i ≠ j."""
    torch.manual_seed(0)
    B, N, K = 2, 128, 3
    raw = torch.randn(B, N, K)
    area = torch.ones(B, N) / N
    ortho = gram_schmidt_modes(raw, area=area)
    # Inner products
    G = torch.einsum('bnk,bnl,bn->bkl', ortho, ortho, area)
    eye = torch.eye(K).expand(B, -1, -1)
    off_diag = (G * (1 - eye)).abs().max().item()
    diag = G.diagonal(dim1=-2, dim2=-1).abs().min().item()
    assert off_diag < 1e-5, f"off-diagonal {off_diag} too large"
    assert diag > 1e-3, f"diag norms collapsed: {diag}"


def test_gram_schmidt_first_mode_unchanged():
    raw = torch.randn(1, 32, 3)
    ortho = gram_schmidt_modes(raw)
    assert torch.allclose(ortho[..., 0], raw[..., 0])


def test_gram_schmidt_is_differentiable():
    raw = torch.randn(1, 16, 3, requires_grad=True)
    ortho = gram_schmidt_modes(raw)
    ortho.sum().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


# ── Ordering hinge ──

def test_ordering_loss_zero_when_sorted():
    f = torch.tensor([[0.1, 0.2, 0.3]])
    assert eigenvalue_ordering_loss(f).item() == 0.0


def test_ordering_loss_positive_when_unsorted():
    f = torch.tensor([[0.3, 0.1, 0.2]])
    assert eigenvalue_ordering_loss(f).item() > 0.0


# ── Parametric / expected ──

def test_parametric_expected_rayleigh_uniform():
    N, K = 64, 2
    x = torch.linspace(0, 1, N).view(1, N, 1).expand(3, -1, -1).contiguous()
    x = x.requires_grad_(True)
    u = _analytic_modes(x, K=K)                    # [3, N, K]
    area = torch.ones(3, N) / (N - 1)
    R = parametric_expected_rayleigh(u, coords=x, area=area, mode="autograd")
    # All 3 batch samples are identical → expected ≈ mean(R)
    R_full = rayleigh_quotient(u, coords=x, area=area)
    assert torch.allclose(R, R_full.mean(), atol=1e-4)


# ── Curriculum ──

def test_curriculum_phases():
    sched = PhysicsCurriculum(e1=10, e2=30, w_field=1.0, w_freq=0.5,
                              w_phys_rayleigh=1.0, w_rayleigh_anchor=0.05)
    a = sched.weights(0)                            # Phase A
    b = sched.weights(20)                           # mid Phase B (t=0.5)
    c = sched.weights(40)                           # Phase C

    assert a['field'] == 0.0
    assert a['freq'] == 0.0
    assert a['rayleigh'] == pytest.approx(1.0)

    assert b['field'] == pytest.approx(0.5)
    assert b['freq'] == pytest.approx(0.25)
    # 0.5 * 1.0 + 0.5 * 0.05 = 0.525
    assert b['rayleigh'] == pytest.approx(0.525)

    assert c['field'] == pytest.approx(1.0)
    assert c['freq'] == pytest.approx(0.5)
    assert c['rayleigh'] == pytest.approx(0.05)


def test_curriculum_disabled_returns_unity():
    sched = PhysicsCurriculum(enabled=False)
    w = sched.weights(0)
    assert all(v == 1.0 for v in w.values())
