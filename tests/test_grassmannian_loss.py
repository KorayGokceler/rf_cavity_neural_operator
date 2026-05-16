"""Unit tests for the spectral-subspace set-prediction loss.

Covers:
- Grassmannian(1D) reduces to 1 - cos^2(theta)  (sign-invariant field loss)
- Frequency matching is invariant to target permutation
- For a near-degenerate pair, an SO(2) rotation of the target subspace does
  not change the Grassmannian loss
- soft / hard modes agree in the well-separated and tightly-coupled limits
"""
import itertools
import math

import torch

from src.training.lightning_module import (
    match_frequencies,
    detect_clusters,
    grassmannian_loss,
    soft_procrustes_loss,
)


def test_n1_reduces_to_standard():
    """Grassmannian on a single vector == 1 - cos^2(theta)."""
    torch.manual_seed(0)
    N = 64
    for _ in range(5):
        a = torch.randn(N, 1)
        b = torch.randn(N, 1)
        loss = grassmannian_loss(a, b).item()

        ca = a / a.norm()
        cb = b / b.norm()
        cos2 = float((ca.t() @ cb) ** 2)
        expected = 1.0 - cos2
        assert abs(loss - expected) < 1e-4, (loss, expected)

    # Identical direction (any positive scale) → loss 0
    v = torch.randn(N, 1)
    assert grassmannian_loss(v, 3.0 * v).item() < 1e-6
    # Sign flip is irrelevant for a subspace
    assert grassmannian_loss(v, -2.0 * v).item() < 1e-6
    # Orthogonal vectors → loss 1 (== n)
    e1 = torch.zeros(N, 1); e1[0] = 1.0
    e2 = torch.zeros(N, 1); e2[1] = 1.0
    assert abs(grassmannian_loss(e1, e2).item() - 1.0) < 1e-5


def test_permutation_invariance():
    """match_frequencies recovers any permutation of the targets, and the
    resulting aligned frequency MSE is ~0 regardless of prediction order."""
    f_true = torch.tensor([1.0, 2.5, 5.0])
    for p in itertools.permutations(range(3)):
        f_pred = f_true[list(p)].clone()
        perm = match_frequencies(f_pred, f_true)
        aligned = f_pred[torch.tensor(perm)]
        assert torch.allclose(aligned, f_true, atol=1e-6), (p, perm, aligned)

    # Grassmannian field loss must also be invariant to a target column perm
    torch.manual_seed(1)
    N = 80
    E = torch.randn(N, 3)
    base = grassmannian_loss(E, E).item()
    for p in itertools.permutations(range(3)):
        Ep = E[:, list(p)]
        # Whole 3-D subspace is identical regardless of column order
        assert abs(grassmannian_loss(E, Ep).item() - base) < 1e-4


def test_rotation_invariance_degenerate():
    """For a (near-)degenerate pair the Grassmannian distance is invariant to
    an SO(2) rotation applied within the degenerate subspace of the target."""
    torch.manual_seed(2)
    N = 100
    E_hat = torch.randn(N, 2)
    E_tgt = torch.randn(N, 2)

    base = grassmannian_loss(E_hat, E_tgt).item()
    for theta in [0.1, 0.7, 1.5, 2.9, math.pi / 3]:
        c, s = math.cos(theta), math.sin(theta)
        R = torch.tensor([[c, -s], [s, c]])
        E_rot = E_tgt @ R.t()
        rotated = grassmannian_loss(E_hat, E_rot).item()
        assert abs(rotated - base) < 1e-4, (theta, base, rotated)


def test_soft_hard_consistency():
    """In the tightly-coupled limit (huge sigma / threshold) the soft and hard
    treatments of a near-degenerate pair both collapse to the same subspace
    distance: ~0 when the predicted span equals the target span."""
    torch.manual_seed(3)
    N = 120
    # Predicted == arbitrary rotation of target within its 2-D span
    E_tgt = torch.randn(N, 2)
    theta = 0.9
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]])
    E_hat = E_tgt @ R.t()

    f = torch.tensor([3.0, 3.0001])  # essentially degenerate

    # hard: a single cluster → Grassmannian over the pair ≈ 0
    clusters = detect_clusters(f, threshold=0.05)
    assert clusters == [[0, 1]]
    g = grassmannian_loss(E_hat, E_tgt).item()
    assert g < 1e-4

    # soft: large sigma fully couples the pair → Procrustes finds the rotation
    sp = soft_procrustes_loss(E_hat, E_tgt, f, sigma=10.0).item()
    assert sp < 1e-3, sp

    # Well-separated frequencies → hard makes singletons, and a per-mode (n=1)
    # Grassmannian still equals the sign-invariant field loss.
    f_sep = torch.tensor([1.0, 9.0])
    assert detect_clusters(f_sep, threshold=0.05) == [[0], [1]]
    v = torch.randn(N, 1)
    assert grassmannian_loss(v, -v).item() < 1e-6


def test_grassmannian_mask_aware():
    """Padded (masked-out) rows must not affect the Grassmannian distance."""
    torch.manual_seed(4)
    N = 50
    E_hat = torch.randn(N, 2)
    E_tgt = torch.randn(N, 2)
    mask = torch.ones(N, dtype=torch.bool)
    mask[40:] = False

    ref = grassmannian_loss(E_hat[:40], E_tgt[:40]).item()
    # Corrupt padded region arbitrarily; masked loss must be unchanged
    E_hat_pad = E_hat.clone(); E_hat_pad[40:] = 999.0
    E_tgt_pad = E_tgt.clone(); E_tgt_pad[40:] = -123.0
    masked = grassmannian_loss(E_hat_pad, E_tgt_pad, mask).item()
    assert abs(masked - ref) < 1e-4, (ref, masked)


def test_soft_procrustes_reduces_to_field_loss_when_separated():
    """When frequencies are well separated, W ≈ I, so soft-Procrustes acts as
    an (orthogonal/sign) aligned per-mode relative L2 — near zero for a sign
    flip of an otherwise perfect prediction."""
    torch.manual_seed(5)
    N = 90
    E_tgt = torch.randn(N, 3)
    E_hat = E_tgt.clone()
    E_hat[:, 1] *= -1.0  # global sign flip on one mode
    f = torch.tensor([1.0, 5.0, 12.0])
    loss = soft_procrustes_loss(E_hat, E_tgt, f, sigma=0.01).item()
    assert loss < 1e-3, loss


def test_soft_procrustes_three_way_degeneracy():
    """Regression test for the 3-way-degeneracy fix.

    The old loss built M = (E_hat^T E_tgt) * W and took its SVD; for a triple
    degeneracy that element-wise mask drops valid off-block couplings, so the
    recovered rotation is NOT orthogonal-optimal and an arbitrary SO(3)
    rotation of the target subspace produces a spuriously large loss.

    With the detached un-weighted Procrustes + per-mode soft blend, an SO(3)
    rotation of a fully degenerate triple must leave the loss ~0.
    """
    torch.manual_seed(7)
    N = 128
    E_tgt = torch.randn(N, 3)

    # Random proper rotation in SO(3) via QR of a Gaussian matrix.
    A = torch.randn(3, 3)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    E_hat = E_tgt @ Q.t()                 # prediction = rotated target subspace

    f_deg = torch.tensor([4.0, 4.0001, 4.0002])   # fully degenerate triple
    loss_deg = soft_procrustes_loss(E_hat, E_tgt, f_deg, sigma=0.3).item()
    assert loss_deg < 1e-3, ("3-way degenerate SO(3) not invariant", loss_deg)

    # Well-separated frequencies: the same rotation now MIXES physically
    # distinct modes, so the loss must NOT be ~0 (rotation freedom is gated off
    # per-mode by alpha -> the model is held to the per-mode field).
    f_sep = torch.tensor([1.0, 6.0, 13.0])
    loss_sep = soft_procrustes_loss(E_hat, E_tgt, f_sep, sigma=0.3).item()
    assert loss_sep > 0.1, ("separated modes wrongly allowed to rotate", loss_sep)
