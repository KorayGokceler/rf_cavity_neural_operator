"""Few-step FEM refinement of predicted eigenmodes (docs/17, hybrid inference).

The network's vertex fields span a trial subspace; block inverse iteration +
Rayleigh–Ritz with the mesh's own P2 matrices (the discretisation the labels
were solved with) turns a few-% field error into a ~1e-5 eigenvalue error in
tens of ms.  Rayleigh–Ritz on the vertex fields alone stalls at the P1 level,
the inverse-iteration step adds the missing P2 content.
"""
import numpy as np
import scipy.linalg
import scipy.sparse.linalg as spla
import skfem
from skfem.models.poisson import laplace, mass

C0 = 299792458.0  # speed of light [m/s]


def refine_modes(xy, elements, fields, n_iter=1):
    """Refine K predicted Dirichlet eigenmodes on a triangle mesh.

    Args:
        xy: [N, 2] physical node coordinates [m].
        elements: [T, 3] triangle node indices.
        fields: [N, K] predicted vertex values (any sign / amplitude).
        n_iter: block inverse-iteration + Rayleigh–Ritz steps.
    Returns:
        lam [K] ascending eigenvalues [1/m²], vertex fields [N, K] (peak +1),
        freq_ghz [K].
    """
    mesh = skfem.MeshTri(np.ascontiguousarray(np.asarray(xy, dtype=np.float64).T),
                         np.ascontiguousarray(np.asarray(elements, dtype=np.int64).T))
    basis = skfem.Basis(mesh, skfem.ElementTriP2())
    I = basis.complement_dofs(basis.get_dofs())            # interior (Dirichlet) dofs
    A = skfem.asm(laplace, basis)[I][:, I].tocsc()
    M = skfem.asm(mass, basis)[I][:, I].tocsc()

    k = fields.shape[1]
    X = np.zeros((basis.N, k))                             # P1 → P2 prolongation
    X[basis.nodal_dofs[0]] = fields
    X[basis.facet_dofs[0]] = 0.5 * (fields[mesh.facets[0]] + fields[mesh.facets[1]])
    V = X[I]

    lu = spla.splu(A)
    for _ in range(n_iter):
        S, _ = np.linalg.qr(np.hstack([V, lu.solve(M @ V)]))   # span{V, A⁻¹MV}
        lam, c = scipy.linalg.eigh(S.T @ (A @ S), S.T @ (M @ S))
        V = S @ c[:, :k]
    lam = lam[:k] if n_iter else np.full(k, np.nan)

    out = np.zeros((basis.N, k))
    out[I] = V
    u = out[basis.nodal_dofs[0]]                            # vertex values
    peak = u[np.abs(u).argmax(axis=0), np.arange(k)]
    u = u / np.where(np.abs(peak) > 0, peak, 1.0)
    return lam, u, C0 * np.sqrt(np.abs(lam)) / (2 * np.pi) / 1e9
