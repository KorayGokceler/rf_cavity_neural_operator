"""Tiny synthetic 3D Maxwell (N0, E field) geometries: PEC boxes on skfem tensor tet meshes (interior
vertices optionally jittered) as converter-style geometry dicts (src/data/dataset_converter_3d.py), modes
from a dense generalized eigensolve with the curl kernel dropped.  Used by tests/test_qoi.py,
tests/test_viz_nedelec.py and tests/test_predict_geometry.py (box_geometry).

PEC wall edges (edges of boundary faces) are essential zeros, G = gradients of the interior-vertex
potentials plus one column per boundary component except the first (zero columns up to Nv),
Kp = GᵀMG (SPD on its first n_pot rows/cols), Y rows of wall edges exactly 0.
"""
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp


def n0_operators(mesh):
    """(K, M, G_full, G_fullᵀ M G_full, edges) of the full N0 space: all edges, all vertices."""
    from skfem import Basis, BilinearForm, ElementTetN0
    from skfem.helpers import curl, dot

    @BilinearForm
    def curlcurl(u, v, w):
        return dot(curl(u), curl(v))

    @BilinearForm
    def mass(u, v, w):
        return dot(u, v)

    basis = Basis(mesh, ElementTetN0())
    K, M = curlcurl.assemble(basis).tocsr(), mass.assemble(basis).tocsr()
    e = mesh.edges                                          # skfem: e[0] < e[1], DOF = ∫_e v·(p1 − p0)
    ne, nv = e.shape[1], mesh.p.shape[1]
    ar = np.arange(ne)
    G = sp.csr_matrix((np.r_[-np.ones(ne), np.ones(ne)], (np.r_[ar, ar], np.r_[e[0], e[1]])), shape=(ne, nv))
    return K, M, G, (G.T @ M @ G).tocsr(), e.T.astype(np.int64)


def wall_edges(mesh, edges):
    """bool [Ne]: edge of a boundary face (the PEC wall)."""
    f = np.sort(mesh.facets[:, mesh.boundary_facets()].T, axis=1)          # [Nb,3]
    nv = mesh.p.shape[1]
    fk = np.unique(np.concatenate([f[:, [0, 1]], f[:, [0, 2]], f[:, [1, 2]]]) @ np.array([nv, 1]))
    return np.isin(edges @ np.array([nv, 1]), fk)


def boundary_components(mesh):
    """List of vertex-index arrays, one per connected component of ∂Ω (face adjacency)."""
    from scipy.sparse.csgraph import connected_components
    f = mesh.facets[:, mesh.boundary_facets()].T
    nv = mesh.p.shape[1]
    r = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    c = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(nv, nv))
    _, lab = connected_components(A, directed=False)
    bv = np.unique(f)
    return [bv[lab[bv] == k] for k in np.unique(lab[bv])]


def e_operators(mesh):
    """E formulation on all edges: (K, M, G, Kp, edges, bnd_edge, n_pot, n_comp).
    G columns: interior vertices (increasing index), then G_full·1_C for every
    boundary component C but the first, then zero columns up to Nv."""
    K, M, G_full, _, edges = n0_operators(mesh)
    nv = mesh.p.shape[1]
    comps = boundary_components(mesh)
    inner = np.setdiff1d(np.arange(nv), np.concatenate(comps))
    ind = sp.csr_matrix((np.ones(sum(len(c) for c in comps[1:])),
                         (np.concatenate([np.zeros(0, int)] + comps[1:]),
                          np.repeat(np.arange(len(comps) - 1), [len(c) for c in comps[1:]]))),
                        shape=(nv, len(comps) - 1))
    n_pot = len(inner) + len(comps) - 1
    G = sp.hstack([G_full[:, inner], G_full @ ind, sp.csr_matrix((len(edges), nv - n_pot))]).tocsr()
    bnd = wall_edges(mesh, edges)
    assert abs(G[bnd]).sum() == 0
    return K, M, G, (G.T @ M @ G).tocsr(), edges, bnd, n_pot, len(comps)


def box_geometry(dims=(1.0, 0.8, 0.6), n=5, jitter=0.0, scale=0.1, n_modes=6, seed=0):
    """(geometry dict, eigenvalues [n_modes], E-field DOFs [Ne, n_modes], wall rows 0) of a PEC box
    whose normalised extent is `dims` (centred at 0); physical size = dims·scale."""
    from skfem import MeshTet
    dims = np.asarray(dims, dtype=float)
    h = dims.max() / n
    ax = [np.linspace(-L / 2, L / 2, max(2, int(round(L / h))) + 1) for L in dims]
    mesh = MeshTet.init_tensor(*ax)
    p = mesh.p.copy()
    if jitter > 0:
        inner = np.setdiff1d(np.arange(p.shape[1]), mesh.boundary_nodes())
        p[:, inner] += jitter * h * np.random.default_rng(seed).uniform(-1, 1, (3, len(inner)))
        mesh = MeshTet(p, mesh.t)
    K, M, G, Kp, edges, bnd, n_pot, n_comp = e_operators(mesh)
    free = ~bnd
    lam, sub = sla.eigh(K[free][:, free].toarray(), M[free][:, free].toarray())
    keep = lam > 1e-8 * lam.max()                           # drop the kernel (n_pot zeros)
    assert (~keep).sum() == n_pot, ((~keep).sum(), n_pot)
    vec = np.zeros((len(edges), int(keep.sum())))
    vec[free] = sub[:, keep]
    lam, vec = lam[keep][:n_modes], vec[:, :n_modes]
    X = p.T.astype(np.float64)
    gap = dims / 2 - np.abs(X)                              # distance to each face pair
    j = gap.argmin(1)
    dist = gap[np.arange(len(X)), j]
    direc = np.zeros_like(X)
    direc[np.arange(len(X)), j] = np.sign(X[np.arange(len(X)), j]) + (X[np.arange(len(X)), j] == 0)
    vol = np.zeros(len(X))
    P = X[mesh.t.T]
    tv = np.abs(np.linalg.det(P[:, 1:] - P[:, :1])) / 6.0
    np.add.at(vol, mesh.t.T.reshape(-1), np.repeat(tv / 4.0, 4))
    tors = np.prod(1.0 - (2.0 * X / dims) ** 2, axis=1)
    feats = np.column_stack([X, dist, direc, vol / vol.mean(), tors / tors.max()]).astype(np.float32)
    csr = lambda A: (A.indptr, A.indices, A.data)            # noqa: E731
    geom = {'X': X.astype(np.float32), 'Input_funcs': feats, 'edges': edges, 'tets': mesh.t.T.astype(np.int64),
            'M': csr(M), 'K': csr(K), 'G': csr(G), 'Kp': csr(Kp), 'scale': float(scale),
            'center': np.zeros(3), 'shape_type': 'box' if jitter == 0 else 'box_jitter',
            'torsion_max': float(tors.max())}
    chi = len(X) - len(edges) + mesh.facets.shape[1] - mesh.t.shape[1]
    geom.update(bnd_edge=bnd, n_pot=int(n_pot), n_bnd_components=int(n_comp),
                betti1=int(1 + (n_comp - 1) - chi), field='E')
    return geom, lam, vec
