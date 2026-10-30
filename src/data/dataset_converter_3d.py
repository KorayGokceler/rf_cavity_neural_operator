"""Whitney N0 geometry and operators of a linear tetrahedral mesh (E field): the display / QoI / viewer
layer of the web service (the generator's former N0 label path is removed; src.qoi.analytic.fe_modes is the
N0 reference solve).

Geometry dict (extract_geometry_3d; the former PKL contract of docs/19, now in memory only):

    'X'           float32 [Nv,3]  vertices, centred at the volume centroid, divided by 'scale'
    'Input_funcs' float32 [Nv,F]  vertex features, FEATURE_NAMES_3D
    'edges'       int64 [Ne,2]    vertex pairs, low → high index, rows sorted lexicographically
                                  (= np.unique of the sorted tet edges); this row order is the
                                  DOF order of M, K, G everywhere
    'tets'        int64 [Nt,4]
    'M','K'       CSR (indptr, indices, data) [Ne×Ne]  N0 mass ∫w_i·w_j / curl-curl ∫curl w_i·curl w_j
                                  on the NORMALISED mesh (Whitney w_ab = λ_a∇λ_b − λ_b∇λ_a, a < b)
    'G'           CSR [Ne×Nv]     E potential matrix (e_potential_matrix): column j < n_pot is
                                  potential j — interior vertices (increasing index), then one
                                  column per boundary component except the first
                                  (= G_full @ indicator); columns ≥ n_pot all zero; wall rows 0
    'Kp'          CSR [Nv×Nv]     Gᵀ M G (SPD on the first n_pot rows/cols, zero elsewhere)
    'bnd_edge'    bool [Ne]       PEC wall edge (edge of a boundary face); 'n_pot',
                                  'n_bnd_components', 'betti1' (= 1 + b2 − χ, informative)
    'scale','center','n_nodes','n_edges','torsion_max','volume' (+ 'shape_type', 'params' set by callers)

Scaling: X = (x − center)/scale ⇒ λ_norm = λ_phys·scale², f = c·√λ_norm / (2π·scale).
"""
import importlib
import json

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


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


def boundary_topology(tets, n_nodes):
    """Boundary structure of a tet mesh from its connectivity alone (deterministic from tets).

    Returns dict:
      faces        [Nb,3] boundary triangles (sorted vertex triples; faces of exactly one tet)
      edge_keys    sorted int64 keys a·Nv + b (a < b) of the edges of the boundary faces (PEC wall edges)
      vert_comp    int64 [Nv]: −1 for interior vertices, else the boundary component of the vertex.
                   Components = connected components of the boundary faces (sharing an edge or a
                   vertex), ordered by vertex count DESCENDING, ties by smallest vertex index —
                   component 0 is the outer wall (the reference potential φ = 0 in the E formulation)
      n_comp       number of boundary components (b2 = n_comp − 1 for a connected domain in R³)
      euler        V − E + F − T of the mesh (χ(Ω) = 1 − b1 + b2)
      euler_boundary  χ(∂Ω) = V_b − E_b + F_b
      manifold     every boundary edge lies in exactly 2 boundary faces and χ(∂Ω) = 2χ(Ω)
                   (identity of compact 3-manifolds with boundary; fails on pinched vertices / edges)
      betti1       1 + b2 − χ(Ω)   (number of handles; informative)
    Raises RuntimeError if a face is shared by more than two tets."""
    from scipy.sparse.csgraph import connected_components
    t = np.sort(np.asarray(tets, dtype=np.int64), axis=1)
    n = int(n_nodes)
    faces, cnt = np.unique(t[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3), axis=0,
                           return_counts=True)
    if cnt.max() > 2:
        raise RuntimeError("non-manifold mesh (face shared by > 2 tets)")
    n_edges = len(np.unique(t[:, _TET_EDGES].reshape(-1, 2), axis=0))
    euler = n - n_edges + len(faces) - len(t)
    bf = faces[cnt == 1]
    be_all = bf[:, [[0, 1], [1, 2], [0, 2]]].reshape(-1, 2)            # sorted pairs (bf rows sorted)
    ekeys, ecnt = np.unique(be_all[:, 0] * n + be_all[:, 1], return_counts=True)
    bv = np.unique(bf)
    vert_comp = np.full(n, -1, dtype=np.int64)
    n_comp = 0
    if len(bv):
        a, b = ekeys // n, ekeys % n
        loc_a, loc_b = np.searchsorted(bv, a), np.searchsorted(bv, b)
        Ab = sp.coo_matrix((np.ones(len(ekeys)), (loc_a, loc_b)), shape=(len(bv),) * 2)
        n_comp, lab = connected_components(Ab, directed=False)
        size = np.bincount(lab, minlength=n_comp)
        vmin = np.full(n_comp, np.iinfo(np.int64).max)
        np.minimum.at(vmin, lab, bv)
        order = np.lexsort((vmin, -size))                              # size desc, then smallest vertex
        rank = np.empty(n_comp, dtype=np.int64)
        rank[order] = np.arange(n_comp)
        vert_comp[bv] = rank[lab]
    euler_b = int(len(bv) - len(ekeys) + len(bf))
    manifold = bool(np.all(ecnt == 2) and euler_b == 2 * euler)
    return dict(faces=bf, edge_keys=ekeys, vert_comp=vert_comp, n_comp=int(n_comp), euler=int(euler),
                euler_boundary=euler_b, manifold=manifold, betti1=int(1 + (n_comp - 1) - euler))


def e_potential_matrix(edges, n_nodes, tets, component_potentials=True, topo=None):
    """E-formulation (n × E = 0 on ∂Ω) kernel basis in the order of `edges` (any row order;
    G[i, edges[i,1]] = +1, G[i, edges[i,0]] = −1 as in discrete_gradient).

    The discrete kernel of curl on the N0 space with the wall edges removed is exactly
    {Gφ : φ P1, φ = 0 on the first boundary component, constant on each other one}: the
    interior-vertex gradients (∇H0¹) plus one potential per extra boundary component
    (G @ indicator(component) — the b2 harmonic Dirichlet fields of isolated conductors).
    Handles (b1) add nothing.  Columns: interior vertices in increasing index, then the
    components 1 … n_comp−1; the matrix is padded with ALL-ZERO columns to [Ne × Nv].
    component_potentials=False drops the component columns (validation only: the b2
    fields then survive the projection as spurious λ = 0 modes).

    Returns (G CSR [Ne×Nv], bnd_edge bool [Ne], n_pot, topo)."""
    edges = np.asarray(edges, dtype=np.int64)
    n = int(n_nodes)
    topo = boundary_topology(tets, n) if topo is None else topo
    lo, hi = edges.min(1), edges.max(1)
    bnd_edge = np.isin(lo * n + hi, topo["edge_keys"])
    Gf = discrete_gradient(edges, n).tocsc()
    vc = topo["vert_comp"]
    interior = np.flatnonzero(vc < 0)
    blocks = [Gf[:, interior]]
    if component_potentials and topo["n_comp"] > 1:
        ind = sp.csr_matrix((np.ones(int((vc > 0).sum())), (np.flatnonzero(vc > 0), vc[vc > 0] - 1)),
                            shape=(n, topo["n_comp"] - 1))
        blocks.append((Gf @ ind).tocsc())
    Gp = sp.hstack(blocks, format="csc")
    n_pot = Gp.shape[1]
    G = sp.hstack([Gp, sp.csc_matrix((len(edges), n - n_pot))], format="csr")
    G.eliminate_zeros()
    if abs(G[bnd_edge]).sum() != 0:                                      # φ constant on each wall component
        raise RuntimeError("E potential matrix has non-zero wall-edge rows")
    return G, bnd_edge, int(n_pot), topo


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


def normalise(nodes, tets):
    """(X = (nodes − center)/scale, center, scale): center = volume centroid, scale = max |x − center|."""
    nodes = np.asarray(nodes, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    vol, _ = tet_geometry(nodes, tets)
    center = (vol[:, None] * nodes[tets].mean(1)).sum(0) / vol.sum()
    scale = float(np.max(np.abs(nodes - center)))
    return (nodes - center) / scale, center, scale


def node_features(X, tets):
    """Vertex features of a normalised mesh (FEATURE_NAMES_3D order) → (feats [Nv, 9] float32,
    torsion_max, volume)."""
    X = np.asarray(X, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    bf = boundary_faces(X, tets)
    bv = np.unique(bf)
    dist, direc = boundary_distance(X, bf)
    vol_n, _ = tet_geometry(X, tets)
    node_vol = np.zeros(len(X))
    for i in range(4):
        np.add.at(node_vol, tets[:, i], vol_n / 4.0)
    w = torsion_function(X, tets, bv)
    torsion_max = float(w.max())
    feats = np.column_stack([X, dist, direc, node_vol / node_vol.max(), w / torsion_max]).astype(np.float32)
    return feats, torsion_max, float(vol_n.sum())


def extract_geometry_3d(nodes, tets):
    """Normalised mesh, features and operators of one geometry (see module docstring)."""
    tets = np.asarray(tets, dtype=np.int64)
    X, center, scale = normalise(nodes, tets)
    feats, torsion_max, volume = node_features(X, tets)
    ops, M = geometry_operators(X, tets)
    return {
        "X": X.astype(np.float32), "Input_funcs": feats, "tets": tets, **ops,
        "scale": scale, "center": center.astype(np.float64),
        "n_nodes": int(len(X)), "n_edges": int(len(ops["edges"])), "torsion_max": torsion_max,
        "volume": volume,
    }, M


def geometry_operators(X, tets):
    """Contract operators of one (normalised) mesh and the scipy M: {'edges', 'M', 'K', 'G' (the E
    potential matrix, e_potential_matrix), 'Kp' = GᵀMG, 'bnd_edge', 'n_pot', 'n_bnd_components',
    'betti1', 'field'}. Deterministic from (X, tets) alone, so a loader can rebuild them when the PKL
    was written with store_operators=False (~1.5 s for 50k edges on one CPU core)."""
    M, K, edges = assemble_n0(X, tets)
    G, bnd_edge, n_pot, topo = e_potential_matrix(edges, len(X), tets)
    Kp = (G.T @ M @ G).tocsr()
    return {"edges": edges.astype(np.int64), "M": to_csr_tuple(M), "K": to_csr_tuple(K),
            "G": to_csr_tuple(G), "Kp": to_csr_tuple(Kp), "bnd_edge": bnd_edge, "n_pot": int(n_pot),
            "n_bnd_components": int(topo["n_comp"]), "betti1": int(topo["betti1"]), "field": "E"}, M


def h5_edges_to_canonical(edges_h5, n_nodes, edges):
    """Map H5 DOF i (edge edges_h5[i] = (tail, head)) to the canonical row and the sign
    s_i = +1 if tail < head else −1:  Y_canonical[row_i] = s_i · e_edges[i]."""
    edges_h5 = np.asarray(edges_h5, dtype=np.int64)
    rows = _edge_index(edges, n_nodes, edges_h5)
    if len(np.unique(rows)) != len(edges) or len(rows) != len(edges):
        raise ValueError("H5 edge list is not a permutation of the tet edges")
    sign = np.where(edges_h5[:, 0] < edges_h5[:, 1], 1.0, -1.0)
    return rows, sign


# ─────────────────────────── cavity QoI labels ─────────────────

# Settings of the stored QoI labels (docs/24_CAVITY_QOI.md §0.4): copper wall, β = 1, linac R/Q.
QOI_LABEL_SETTINGS = {"sigma": 5.8e7, "beta": 1.0, "convention": "linac"}


def qoi_api():
    """src.qoi, imported lazily (the converter / dataset import without it)."""
    return importlib.import_module("src.qoi")


def geom_params_of(attrs):
    """The family's geom_params [m] of an H5 sample group (attrs['geom_params'] JSON; {} when absent)."""
    return json.loads(attrs.get("geom_params", "{}"))


def qoi_operators_of(geom, M=None, **kw):
    """src.qoi.build_qoi_operators for one geometry_pool entry (X, tets, edges, scale, center);
    M: scipy mass matrix (assembled by build_qoi_operators when None).  The beam axis and the
    accelerating length follow the geometry's family and geom['params'] (src.qoi.operators.beam_axis,
    acceleration_length) unless kw sets an axis."""
    op = importlib.import_module("src.qoi.operators")
    center = geom.get("center", None)
    center = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64).reshape(3)
    st, prm = geom.get("shape_type", ""), geom.get("params")
    if not any(k in kw for k in ("axis_xy", "axis_dir", "axis_point")):
        kw.update(op.beam_axis(st, geom["X"], float(geom["scale"]), center, prm))
    ops = qoi_api().build_qoi_operators(np.asarray(geom["X"], dtype=np.float64),
                                        np.asarray(geom["tets"], dtype=np.int64),
                                        np.asarray(geom["edges"], dtype=np.int64),
                                        float(geom["scale"]), center, M=M, **kw)
    L = op.acceleration_length(st, prm)
    ops["L_acc"] = ops["L_axis"] if L is None else L / ops["scale"]     # normalised, like L_axis
    return ops


# ─────────────────────────── data checks ───────────────────────

def require_e_field(metadata):
    """Raise for data of the removed H formulation (PKL / H5 metadata 'field' = 'H')."""
    field = str((metadata or {}).get("field", None) or "E").upper()
    if field != "E":
        raise ValueError(f"data of the removed {field} formulation (only E-field data is supported; "
                         "regenerate with the current generator)")
