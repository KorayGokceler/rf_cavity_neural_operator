"""FEM refinement (src/fem_refine.py): a 5%-perturbed prediction is driven to
the P2 eigenpairs of the same mesh by inverse iteration + Rayleigh–Ritz."""
import numpy as np
import pytest

skfem = pytest.importorskip("skfem")
import scipy.sparse.linalg as spla                     # noqa: E402
from skfem.models.poisson import laplace, mass         # noqa: E402

from src.fem_refine import refine_modes                # noqa: E402


def _disk_reference(R=0.04):
    m = skfem.MeshTri.init_circle(4).scaled(R)
    b = skfem.Basis(m, skfem.ElementTriP2())
    I = b.complement_dofs(b.get_dofs())
    lam, v = spla.eigsh(skfem.asm(laplace, b)[I][:, I], k=3,
                        M=skfem.asm(mass, b)[I][:, I], sigma=0)
    full = np.zeros((b.N, 3))
    full[I] = v
    return m, lam, full[b.nodal_dofs[0]]


def test_refinement_recovers_p2_eigenpairs():
    m, lam, U = _disk_reference()
    xy = m.p.T
    r = np.linalg.norm(xy, axis=1) / 0.04
    E = np.stack([(1 - r ** 2) * (xy[:, 0] / 0.04) ** j for j in range(1, 4)], 1)  # smooth error
    E *= 0.05 * np.linalg.norm(U, axis=0) / np.linalg.norm(E, axis=0)
    err = []
    for n in (1, 2):
        lr, ur, f = refine_modes(xy, m.t.T, U + E, n_iter=n)
        err.append(np.abs(lr / lam - 1).max())
    assert err[0] < 2e-3 and err[1] < 1e-4 and err[1] < err[0]
    np.testing.assert_allclose(f, 299792458.0 * np.sqrt(lr) / (2 * np.pi) / 1e9)
    assert np.allclose(np.abs(ur).max(axis=0), 1.0)
    # fundamental (non-degenerate) field matches up to sign
    u0 = U[:, 0] / U[np.abs(U[:, 0]).argmax(), 0]
    assert np.abs(ur[:, 0] - u0).max() < 1e-2
