"""Tiny synthetic 3D Maxwell (N0) data in the convert_3d.py PKL format: PEC
boxes on skfem tensor tet meshes (interior vertices optionally jittered), modes
from a dense generalized eigensolve with the curl kernel dropped.  Used by
tests/test_eigenspace_3d.py, tests/test_viz_*.py and the synthetic smoke run.

field='H' (docs/20 §1–4): all edges free, kernel ∇P1 (G = full gradient).
field='E' (e_contract.md): PEC wall edges (edges of boundary faces) are
essential zeros, G = gradients of the interior-vertex potentials plus one
column per boundary component except the first (zero columns up to Nv),
Kp = GᵀMG (SPD on its first n_pot rows/cols), Y rows of wall edges exactly 0.

    python -m tests.maxwell3d_synth out.pkl [n_geoms] [n_cells] [n_modes] [H|E]
"""
import pickle
import sys

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

C0 = 299_792_458.0


def n0_operators(mesh):
    """(K, M, G, Kp, edges) of the H formulation: all edges, all vertices."""
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


def edge_dofs(X, edges, field):
    """Canonical N0 interpolant ∫_e v·t by 3-point Gauss–Legendre per edge."""
    xa, xb = X[edges[:, 0]], X[edges[:, 1]]
    s, w = np.polynomial.legendre.leggauss(3)
    s, w = 0.5 * (s + 1), 0.5 * w
    return sum(wi * (field(xa + si * (xb - xa)) * (xb - xa)).sum(-1) for si, wi in zip(s, w, strict=True))


def box_geometry(dims=(1.0, 0.8, 0.6), n=5, jitter=0.0, scale=0.1, n_modes=6, seed=0, field='H'):
    """(geometry dict, eigenvalues [n_modes], field DOFs [Ne, n_modes]) of a PEC box
    whose normalised extent is `dims` (centred at 0); physical size = dims·scale.
    field='H': H-field DOFs on all edges; field='E': E-field DOFs, wall rows 0."""
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
    if field == 'E':
        K, M, G, Kp, edges, bnd, n_pot, n_comp = e_operators(mesh)
        free = ~bnd
        lam, sub = sla.eigh(K[free][:, free].toarray(), M[free][:, free].toarray())
        keep = lam > 1e-8 * lam.max()                       # drop the kernel (n_pot zeros)
        assert (~keep).sum() == n_pot, ((~keep).sum(), n_pot)
        vec = np.zeros((len(edges), int(keep.sum())))
        vec[free] = sub[:, keep]
    elif field == 'H':
        K, M, G, Kp, edges = n0_operators(mesh)
        lam, vec = sla.eigh(K.toarray(), M.toarray())
        keep = lam > 1e-8 * lam.max()                       # drop ∇P1 (Nv − 1 zeros)
        vec = vec[:, keep]
    else:
        raise ValueError(f"field must be 'H' or 'E', got {field!r}")
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
    if field == 'E':
        chi = len(X) - len(edges) + mesh.facets.shape[1] - mesh.t.shape[1]
        geom.update(bnd_edge=bnd, n_pot=int(n_pot), n_bnd_components=int(n_comp),
                    betti1=int(1 + (n_comp - 1) - chi), field='E')
    return geom, lam, vec


def make_dataset(n_geoms=6, n=5, n_modes=6, seed=0, field='H'):
    rng = np.random.default_rng(seed)
    pool, samples = {}, []
    for g in range(n_geoms):
        dims = np.sort(rng.uniform(0.5, 1.0, 3))[::-1] / 1.0
        geom, lam, vec = box_geometry(dims, n, jitter=0.15 * (g % 2), scale=float(rng.uniform(0.05, 0.2)),
                                      n_modes=n_modes, seed=g, field=field)
        pool[g] = geom
        f = C0 * np.sqrt(lam) / (2 * np.pi * geom['scale']) / 1e9
        sgn = rng.choice([-1.0, 1.0], len(lam))
        for k in range(len(lam)):
            samples.append({'geom_id': g, 'Y': (sgn[k] * vec[:, k]).astype(np.float32),
                            'Theta': np.array([k, f[k], len(samples)], dtype=np.float64)})
    freqs = np.array([s['Theta'][1] for s in samples])
    meta = {'freq_stats': {'mean': float(freqs.mean()), 'std': float(freqs.std() + 1e-9)},
            'n_modes': n_modes, 'field': field, 'element': 'N0',
            'feature_names': ['x', 'y', 'z', 'dist_to_boundary', 'dir_bnd_x', 'dir_bnd_y', 'dir_bnd_z',
                              'node_volume', 'torsion']}
    return {'geometry_pool': pool, 'samples': samples, 'metadata': meta}


if __name__ == '__main__':
    out = sys.argv[1]
    args = [int(a) for a in sys.argv[2:5]]
    fld = sys.argv[5] if len(sys.argv) > 5 else 'H'
    data = make_dataset(*(args[:1] or [6]), **dict(zip(("n", "n_modes"), args[1:], strict=False)), field=fld)
    with open(out, 'wb') as f:
        pickle.dump(data, f)
    ne = [len(g['edges']) for g in data['geometry_pool'].values()]
    print(f"wrote {out}: {len(ne)} geometries, edges {min(ne)}–{max(ne)}")
