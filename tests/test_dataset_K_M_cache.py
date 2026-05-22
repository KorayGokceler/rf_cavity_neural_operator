"""Integration test for the P1 FEM K, M cache used by the Rayleigh fem path.

Generates a tiny synthetic unit-square mesh, assembles K, M via the
converter helper, and checks that the discrete Rayleigh quotient
E^T K E / E^T M E matches the analytical Laplace eigenvalue for the
fundamental Dirichlet eigenfunction u(x,y) = sin(πx)·sin(πy), λ=2π².
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from src.data.dataset_converter import _assemble_p1_K_M


def _unit_square_mesh(n: int):
    """Tensor-product triangular mesh on [0,1]² with (n+1)² nodes."""
    xs = np.linspace(0, 1, n + 1)
    ys = np.linspace(0, 1, n + 1)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    nodes = np.stack([XX.ravel(), YY.ravel()], axis=-1)         # [(n+1)², 2]
    elems = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b = a + 1
            c = a + (n + 1)
            d = c + 1
            elems.append([a, b, d])
            elems.append([a, d, c])
    return nodes.astype(np.float32), np.array(elems, dtype=np.int32)


def test_p1_assembly_recovers_first_eigenvalue():
    n = 40
    nodes, elems = _unit_square_mesh(n)
    K, M = _assemble_p1_K_M(nodes, elems)

    # Analytical Dirichlet eigenfunction on [0,1]²
    u = np.sin(math.pi * nodes[:, 0]) * np.sin(math.pi * nodes[:, 1])
    num = float(u @ (K @ u))
    den = float(u @ (M @ u))
    R = num / den
    expected = 2.0 * (math.pi ** 2)
    rel = abs(R - expected) / expected
    # P1 linear elements on a 40×40 grid: ~1% accuracy is plenty.
    assert rel < 0.02, f"R={R} vs expected {expected} (rel={rel:.3%})"


def test_p1_K_M_symmetric_positive():
    n = 8
    nodes, elems = _unit_square_mesh(n)
    K, M = _assemble_p1_K_M(nodes, elems)
    K_dense = K.toarray()
    M_dense = M.toarray()
    # Symmetry
    assert np.allclose(K_dense, K_dense.T, atol=1e-6)
    assert np.allclose(M_dense, M_dense.T, atol=1e-6)
    # M is SPD (diagonal blocks of consistent mass are positive)
    eig_M = np.linalg.eigvalsh(M_dense)
    assert eig_M.min() > 0, "Mass matrix must be SPD"
    # K is SPSD (Neumann null space contains constants), so smallest eig ~ 0
    eig_K = np.linalg.eigvalsh(K_dense)
    assert eig_K.min() > -1e-6, f"K should be PSD, min eig {eig_K.min()}"


def test_fem_rayleigh_path_with_sparse_csr_tensor():
    """Cross-check torch sparse path matches numpy reference."""
    n = 20
    nodes, elems = _unit_square_mesh(n)
    K_csr, M_csr = _assemble_p1_K_M(nodes, elems)
    u = np.sin(math.pi * nodes[:, 0]) * np.sin(math.pi * nodes[:, 1])

    R_np = (u @ (K_csr @ u)) / (u @ (M_csr @ u))

    # Torch sparse CSR tensor path
    K_t = torch.sparse_csr_tensor(
        crow_indices=torch.from_numpy(K_csr.indptr.astype(np.int64)),
        col_indices=torch.from_numpy(K_csr.indices.astype(np.int64)),
        values=torch.from_numpy(K_csr.data.astype(np.float32)),
        size=K_csr.shape,
    )
    M_t = torch.sparse_csr_tensor(
        crow_indices=torch.from_numpy(M_csr.indptr.astype(np.int64)),
        col_indices=torch.from_numpy(M_csr.indices.astype(np.int64)),
        values=torch.from_numpy(M_csr.data.astype(np.float32)),
        size=M_csr.shape,
    )
    e = torch.from_numpy(u.astype(np.float32)).reshape(-1, 1)
    R_t = float((e * torch.sparse.mm(K_t, e)).sum() / (e * torch.sparse.mm(M_t, e)).sum())
    assert abs(R_np - R_t) / R_np < 1e-4
