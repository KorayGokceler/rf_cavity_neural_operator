"""3D H5 (dataset_generator_3d) → training PKL for the 3D Maxwell (H-field, N0) model.

Output contract (docs/19_3D_DATA_PIPELINE.md):

    {'geometry_pool': {g_id: {
        'X'          float32 [Nv,3]   vertices, centred at the volume centroid, divided by 'scale'
        'Input_funcs' float32 [Nv,F]  vertex features, FEATURE_NAMES_3D
        'edges'      int64 [Ne,2]     vertex pairs, low → high index, rows sorted lexicographically
                                      (= np.unique of the sorted tet edges); this row order is the
                                      DOF order of Y, M, K, G everywhere
        'tets'       int64 [Nt,4]
        'M','K'      CSR (indptr, indices, data) [Ne×Ne]  N0 mass ∫w_i·w_j / curl-curl ∫curl w_i·curl w_j
                                      on the NORMALISED mesh (Whitney w_ab = λ_a∇λ_b − λ_b∇λ_a, a < b)
        'G'          CSR [Ne×Nv]      G[e, high] = +1, G[e, low] = −1  (N0 DOFs of ∇φ for P1 φ)
        'Kp'         CSR [Nv×Nv]      Gᵀ M G (P1 Neumann stiffness; singular on constants)
        'scale','center','shape_type','n_nodes','n_edges', 'torsion_max'}},
     'samples': [{'geom_id', 'Y' float32 [Ne] (unit M-norm on the normalised mesh, sign arbitrary),
                  'Theta' float32 [3] = [mode_idx, freq_GHz, sample_id]}],
     'metadata': {...}}

Scaling: X = (x − center)/scale ⇒ λ_norm = λ_phys·scale², f = c·√λ_norm / (2π·scale).
"""
import json
import pickle
from collections import defaultdict

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from tqdm import tqdm

C0 = 299792458.0

FEATURE_NAMES_3D = [
    "x", "y", "z",                     # normalised coordinates
    "dist_to_boundary",                # distance to the nearest point of ∂Ω (normalised units)
    "dir_bnd_x", "dir_bnd_y", "dir_bnd_z",  # unit vector to that point (outward normal on ∂Ω)
    "node_volume",                     # lumped vertex volume Σ|T|/4, divided by its max
    "torsion",                         # w / max w,  −Δw = 1 in Ω, w = 0 on ∂Ω (P1)
]

# Local edges of a tet (vertex pairs) — N0 has one DOF per edge.
_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
_TET_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])  # face i is opposite vertex i


# ─────────────────────────── mesh topology ─────────────────────

def edges_from_tets(tets):
    """Unique edges [Ne,2] (low, high), rows sorted lexicographically."""
    t = np.asarray(tets, dtype=np.int64)
    e = np.sort(t[:, _TET_EDGES].reshape(-1, 2), axis=1)
    return np.unique(e, axis=0)


def _edge_index(edges, n_nodes, pairs):
    """Row of each (a, b) pair (any orientation) in the sorted `edges` array."""
    keys = edges[:, 0] * n_nodes + edges[:, 1]
    p = np.sort(np.asarray(pairs, dtype=np.int64), axis=-1)
    q = p[..., 0] * n_nodes + p[..., 1]
    idx = np.searchsorted(keys, q)
    if np.any(idx >= len(keys)) or np.any(keys[np.minimum(idx, len(keys) - 1)] != q):
        raise ValueError("edge not found in edge list")
    return idx


def boundary_faces(nodes, tets):
    """Boundary triangles [Nb,3], each ordered so that (p1−p0)×(p2−p0) points OUTWARD."""
    t = np.asarray(tets, dtype=np.int64)
    f = t[:, _TET_FACES].reshape(-1, 3)
    opp = np.repeat(np.arange(len(t)), 4), np.tile(np.arange(4), len(t))
    _, inv, cnt = np.unique(np.sort(f, axis=1), axis=0, return_inverse=True, return_counts=True)
    b = cnt[inv.ravel()] == 1
    f, opp_v = f[b], t[opp[0][b], opp[1][b]]
    p0, p1, p2 = nodes[f[:, 0]], nodes[f[:, 1]], nodes[f[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    inward = (n * (nodes[opp_v] - p0)).sum(1) > 0
    f[inward] = f[inward][:, [0, 2, 1]]
    return f


def tet_geometry(nodes, tets):
    """Signed-volume-free geometry: |T| [Nt] and barycentric gradients ∇λ [Nt,4,3]."""
    P = nodes[np.asarray(tets)]                       # [Nt,4,3]
    J = (P[:, 1:] - P[:, :1]).transpose(0, 2, 1)       # columns p_i − p_0
    Jinv = np.linalg.inv(J)                            # rows = ∇λ_1..3
    grads = np.concatenate([-Jinv.sum(1, keepdims=True), Jinv], axis=1)
    vol = np.abs(np.linalg.det(J)) / 6.0
    return vol, grads


# ─────────────────────────── N0 assembly (closed form) ─────────

def assemble_n0(nodes, tets, edges=None):
    """Whitney (lowest-order Nédélec) mass and curl-curl matrices, closed form.

    DOF e = (a, b), a < b: basis w_e = λ_a∇λ_b − λ_b∇λ_a (∫_e w_e·dl = 1 from a to b),
    curl w_e = 2∇λ_a × ∇λ_b, ∫_T λ_iλ_j = |T|(1+δ_ij)/20. Identical to skfem
    ElementTetN0 (tested). Returns (M, K, edges) with M, K CSR [Ne×Ne] in `edges` order.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    if edges is None:
        edges = edges_from_tets(tets)
    vol, gr = tet_geometry(nodes, tets)
    loc = _TET_EDGES[None].repeat(len(tets), 0)        # [Nt,6,2] local vertex ids
    gv = tets[:, _TET_EDGES]                           # [Nt,6,2] global vertex ids
    swap = gv[..., 0] > gv[..., 1]                     # orient low → high global index
    loc = np.where(swap[..., None], loc[..., ::-1], loc)
    ia, ib = loc[..., 0], loc[..., 1]                  # [Nt,6]
    ga = np.take_along_axis(gr, ia[..., None], 1)      # [Nt,6,3]
    gb = np.take_along_axis(gr, ib[..., None], 1)
    curls = 2.0 * np.cross(ga, gb)
    Kl = vol[:, None, None] * np.einsum("tik,tjk->tij", curls, curls)
    gg = np.einsum("tik,tjk->tij", gr, gr)             # ∇λ_i·∇λ_j [Nt,4,4]

    def g(i, j):                                       # gather gg[t, i[t,e], j[t,f]] → [Nt,6,6]
        return np.take_along_axis(np.take_along_axis(gg, i[:, :, None].repeat(4, 2), 1),
                                  j[:, None, :].repeat(6, 1), 2)

    def m(i, j):                                       # ∫λ_iλ_j / |T| = (1+δ)/20
        return (1.0 + (i[:, :, None] == j[:, None, :])) / 20.0

    Ml = vol[:, None, None] * (m(ia, ia) * g(ib, ib) - m(ia, ib) * g(ib, ia)
                               - m(ib, ia) * g(ia, ib) + m(ib, ib) * g(ia, ia))
    ge = _edge_index(edges, len(nodes), gv)            # [Nt,6] global DOF
    rows = np.repeat(ge, 6, axis=1).ravel()
    cols = np.tile(ge, (1, 6)).ravel()
    n = len(edges)
    M = sp.csr_matrix((Ml.ravel(), (rows, cols)), shape=(n, n))
    K = sp.csr_matrix((Kl.ravel(), (rows, cols)), shape=(n, n))
    M.sum_duplicates()
    K.sum_duplicates()
    return M, K, edges


def discrete_gradient(edges, n_nodes):
    """G [Ne×Nv]: G[e, high] = +1, G[e, low] = −1 (N0 DOFs of the gradient of P1 φ)."""
    ne = len(edges)
    r = np.r_[np.arange(ne), np.arange(ne)]
    return sp.csr_matrix((np.r_[np.ones(ne), -np.ones(ne)], (r, np.r_[edges[:, 1], edges[:, 0]])),
                         shape=(ne, n_nodes))


def to_csr_tuple(A):
    A = sp.csr_matrix(A)
    A.sort_indices()
    return (A.indptr.astype(np.int64), A.indices.astype(np.int64), A.data.astype(np.float64))


def from_csr_tuple(t, shape):
    return sp.csr_matrix((t[2], t[1], t[0]), shape=shape)


# ─────────────────────────── features ──────────────────────────

def _closest_point_on_triangles(p, a, b, c):
    """Closest points on triangles (a, b, c) to points p (all [..., 3]); Ericson, RTCD §5.1.5."""
    p = np.broadcast_to(p, a.shape)
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    eps = 1e-300
    denom = va + vb + vc
    v = vb / np.where(np.abs(denom) > eps, denom, eps)
    w = vc / np.where(np.abs(denom) > eps, denom, eps)
    res = a + ab * v[..., None] + ac * w[..., None]                     # interior
    # edges
    t_ab = d1 / np.where(np.abs(d1 - d3) > eps, d1 - d3, eps)
    t_ac = d2 / np.where(np.abs(d2 - d6) > eps, d2 - d6, eps)
    t_bc = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) > eps, (d4 - d3) + (d5 - d6), eps)
    conds = [
        ((d1 <= 0) & (d2 <= 0), a),
        ((d3 >= 0) & (d4 <= d3), b),
        ((d6 >= 0) & (d5 <= d6), c),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * t_ab[..., None]),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * t_ac[..., None]),
        ((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + (c - b) * t_bc[..., None]),
    ]
    done = np.zeros(p.shape[:-1], dtype=bool)
    out = res.copy()
    for cnd, val in conds:
        sel = cnd & ~done
        out[sel] = val[sel]
        done |= sel
    return out


def boundary_distance(X, bfaces, n_cand=16):
    """Distance and unit direction from each vertex to the nearest point of ∂Ω.

    Candidates: the n_cand boundary triangles with the nearest centroids (exact
    point–triangle distance on each), and the nearest boundary vertex. Boundary
    vertices get distance 0 and the (area-weighted) outward vertex normal.
    """
    tri = X[bfaces]                                                    # [Nb,3,3]
    cent = tri.mean(1)
    k = min(n_cand, len(bfaces))
    _, ci = cKDTree(cent).query(X, k=k)
    ci = ci.reshape(len(X), k)
    t = tri[ci]                                                        # [Nv,k,3,3]
    cp = _closest_point_on_triangles(X[:, None, :], t[..., 0, :], t[..., 1, :], t[..., 2, :])
    d = np.linalg.norm(cp - X[:, None, :], axis=-1)
    j = d.argmin(1)
    near = cp[np.arange(len(X)), j]
    dist = d[np.arange(len(X)), j]
    bv = np.unique(bfaces)
    dv, iv = cKDTree(X[bv]).query(X)
    use_v = dv < dist
    near[use_v], dist[use_v] = X[bv][iv[use_v]], dv[use_v]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])        # outward, |fn| = 2·area
    vn = np.zeros_like(X)
    for i in range(3):
        np.add.at(vn, bfaces[:, i], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-300)
    direc = near - X
    on_bnd = np.zeros(len(X), dtype=bool)
    on_bnd[bv] = True
    small = dist <= 1e-12 * np.abs(X).max()
    direc[on_bnd | small] = vn[on_bnd | small]
    dist[on_bnd] = 0.0
    direc /= np.linalg.norm(direc, axis=1, keepdims=True).clip(1e-300)
    return dist, direc


def torsion_function(X, tets, bnd_vertices):
    """P1 solution of −Δw = 1 in Ω, w = 0 on ∂Ω at the vertices (normalised mesh)."""
    import skfem
    from skfem.models.poisson import laplace, unit_load
    mesh = skfem.MeshTet(np.ascontiguousarray(X.T, dtype=np.float64), np.ascontiguousarray(np.asarray(tets).T))
    basis = skfem.Basis(mesh, skfem.ElementTetP1())
    w = skfem.solve(*skfem.condense(skfem.asm(laplace, basis), skfem.asm(unit_load, basis), D=bnd_vertices))
    return np.clip(w, 0.0, None)


def extract_geometry_3d(nodes, tets):
    """Normalised mesh, features and operators of one geometry (see module docstring)."""
    nodes = np.asarray(nodes, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    vol, _ = tet_geometry(nodes, tets)
    center = (vol[:, None] * nodes[tets].mean(1)).sum(0) / vol.sum()
    scale = float(np.max(np.abs(nodes - center)))
    X = (nodes - center) / scale

    bf = boundary_faces(X, tets)
    bv = np.unique(bf)
    dist, direc = boundary_distance(X, bf)
    vol_n = vol / scale ** 3
    node_vol = np.zeros(len(X))
    for i in range(4):
        np.add.at(node_vol, tets[:, i], vol_n / 4.0)
    w = torsion_function(X, tets, bv)
    torsion_max = float(w.max())
    feats = np.column_stack([X, dist, direc, node_vol / node_vol.max(), w / torsion_max]).astype(np.float32)

    ops, M = geometry_operators(X, tets)
    return {
        "X": X.astype(np.float32), "Input_funcs": feats, "tets": tets, **ops,
        "scale": scale, "center": center.astype(np.float64),
        "n_nodes": int(len(X)), "n_edges": int(len(ops["edges"])), "torsion_max": torsion_max,
        "volume": float(vol_n.sum()),
    }, M


def geometry_operators(X, tets):
    """Contract operators of one (normalised) mesh: {'edges', 'M', 'K', 'G', 'Kp'} (CSR tuples)
    and the scipy M. Deterministic from (X, tets) alone, so a loader can rebuild them when the
    PKL was written with store_operators=False (~1.5 s for 50k edges on one CPU core)."""
    M, K, edges = assemble_n0(X, tets)
    G = discrete_gradient(edges, len(X))
    Kp = (G.T @ M @ G).tocsr()
    return {"edges": edges.astype(np.int64), "M": to_csr_tuple(M), "K": to_csr_tuple(K),
            "G": to_csr_tuple(G), "Kp": to_csr_tuple(Kp)}, M


def h5_edges_to_canonical(edges_h5, n_nodes, edges):
    """Map H5 DOF i (edge edges_h5[i] = (tail, head)) to the canonical row and the sign
    s_i = +1 if tail < head else −1:  Y_canonical[row_i] = s_i · h_edges[i]."""
    edges_h5 = np.asarray(edges_h5, dtype=np.int64)
    rows = _edge_index(edges, n_nodes, edges_h5)
    if len(np.unique(rows)) != len(edges) or len(rows) != len(edges):
        raise ValueError("H5 edge list is not a permutation of the tet edges")
    sign = np.where(edges_h5[:, 0] < edges_h5[:, 1], 1.0, -1.0)
    return rows, sign


# ─────────────────────────── conversion ────────────────────────

class RFCavity3DConverter:
    def __init__(self, h5_filepath: str):
        self.h5_filepath = h5_filepath
        self.geometry_pool = {}
        self.samples = []
        self.freq_by_mode = defaultdict(list)

    def convert_dataset(self, output_filepath, mode_indices=None, max_samples=None,
                        freq_mean=None, freq_std=None, check_rayleigh=True, store_operators=True):
        """store_operators=False drops M/K/G/Kp from the PKL (they dominate its size, ~85 %);
        rebuild them with geometry_operators(X, tets). Everything else is unchanged."""
        with h5py.File(self.h5_filepath, "r") as f:
            file_meta = json.loads(f.attrs.get("metadata", "{}"))
            keys = sorted(k for k in f.keys() if isinstance(f[k], h5py.Group) and "h_edges" in f[k])
            if not keys:
                raise ValueError(f"No 3D sample groups (sample_XXXX/h_edges) in {self.h5_filepath}")
            if max_samples:
                keys = keys[:max_samples]
            n_modes = None
            max_rel = 0.0
            for key in tqdm(keys, desc="Converting 3D"):
                grp = f[key]
                sid = int(key.split("_")[-1])
                freqs = np.asarray(grp["freqs"][:], dtype=np.float64)
                H = np.asarray(grp["h_edges"][:], dtype=np.float64)
                order = np.argsort(freqs, kind="stable")
                freqs, H = freqs[order], H[:, order]
                n_modes = len(freqs) if n_modes is None else min(n_modes, len(freqs))
                geom, M = extract_geometry_3d(grp["nodes"][:], grp["tets"][:])
                geom["shape_type"] = str(grp.attrs.get("shape_type", "unknown"))
                # freq of mode K+1 [GHz]: f_next ≈ f_K means the K-mode cut splits a degenerate cluster
                geom["freq_next"] = float(grp.attrs.get("freq_next", np.nan))
                rows, sign = h5_edges_to_canonical(grp["edges"][:], geom["n_nodes"], geom["edges"])
                Y = np.zeros_like(H)
                Y[rows] = sign[:, None] * H
                Y /= np.sqrt(np.einsum("ij,ij->j", Y, M @ Y))[None]
                if check_rayleigh:  # λ_norm = λ_phys·scale² ⇒ f from the stored normalised operators
                    Kc = from_csr_tuple(geom["K"], M.shape)
                    lam = np.einsum("ij,ij->j", Y, Kc @ Y)
                    f_rq = C0 * np.sqrt(lam) / (2 * np.pi * geom["scale"]) / 1e9
                    rel = np.abs(f_rq / freqs - 1).max()
                    max_rel = max(max_rel, rel)
                    if rel > 1e-6:
                        raise RuntimeError(f"{key}: Rayleigh frequency mismatch {rel:.2e}")
                if not store_operators:
                    for k_ in ("M", "K", "G", "Kp"):
                        geom.pop(k_)
                self.geometry_pool[sid] = geom
                modes = range(len(freqs)) if mode_indices is None else mode_indices
                for i, m_idx in enumerate(modes):
                    if m_idx >= len(freqs):
                        continue
                    self.samples.append({
                        "geom_id": sid, "Y": Y[:, m_idx].astype(np.float32),
                        "Theta": np.array([float(i), freqs[m_idx], float(sid)], dtype=np.float32),
                    })
                    self.freq_by_mode[i].append(float(freqs[m_idx]))
        allf = np.concatenate([np.asarray(v) for v in self.freq_by_mode.values()])
        metadata = {
            "freq_stats": {
                "mean": float(freq_mean) if freq_mean is not None else float(allf.mean()),
                "std": float(freq_std) if freq_std is not None else float(allf.std() + 1e-10),
                "mode_stats": {str(m): {"mean": float(np.mean(v)), "std": float(np.std(v) + 1e-10)}
                               for m, v in self.freq_by_mode.items()},
            },
            "n_modes": int(len(self.freq_by_mode)),
            "mode_indices": list(range(n_modes)) if mode_indices is None else list(mode_indices),
            "field": "H", "element": "N0",
            "feature_names": list(FEATURE_NAMES_3D),
            "n_samples": len(self.samples), "n_geometries": len(self.geometry_pool),
            "edge_convention": "edges = np.unique(sorted tet edges), low->high; DOF = line integral low->high",
            "scaling": "lambda_norm = lambda_phys * scale^2; f_GHz = c*sqrt(lambda_norm)/(2*pi*scale)/1e9",
            "rayleigh_max_rel_err": float(max_rel),
            "operators_stored": bool(store_operators),
            "source_h5": self.h5_filepath, "generator_metadata": file_meta,
        }
        with open(output_filepath, "wb") as fo:
            pickle.dump({"geometry_pool": self.geometry_pool, "samples": self.samples, "metadata": metadata}, fo,
                        protocol=pickle.HIGHEST_PROTOCOL)
        nv = [g["n_nodes"] for g in self.geometry_pool.values()]
        ne = [g["n_edges"] for g in self.geometry_pool.values()]
        print(f"Saved {output_filepath}: {len(self.geometry_pool)} geometries, {len(self.samples)} samples, "
              f"Nv {min(nv)}-{max(nv)}, Ne {min(ne)}-{max(ne)}, f {allf.min():.3f}-{allf.max():.3f} GHz, "
              f"Rayleigh max rel err {max_rel:.1e}")
        return output_filepath
