"""Mesh helpers and model input features of a linear tetrahedral mesh (the display geometry of the web service, the
generator's boundary topology and the per-vertex features of the field model).

    boundary_faces, tet_geometry     outward boundary triangles, |T| and barycentric gradients
    boundary_topology                connected boundary components, genus / b1 of the cavity (generator)
    normalise                        X = (nodes - center) / scale, center = volume centroid, scale = max |x - center|
    node_features, FEATURE_NAMES_3D  the 9 per-vertex features (position, wall distance and direction, node volume,
                                     torsion): the UI's view of the model's inputs (the model itself evaluates them
                                     at the quadrature points, highorder.point_features)
    display_geometry                 normalised mesh + features of one geometry (feature cut planes of the UI)
    geom_params_of                   the family's geom_params of an H5 sample group

Scaling: X = (x - center)/scale, so a normalised eigenvalue is lambda_norm = lambda_phys * scale^2 and
f = c * sqrt(lambda_norm) / (2 pi scale).  The Whitney-N0 edge operators this module used to assemble (display,
QoI and viewer layer) were removed with docs/31 §6 item 15: fields are drawn and the figures of merit are
computed directly on the HCurl(p2) space of the curved mesh (src/viz/hcurl_display.py, src/data_gen/highorder.py).
"""
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

_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
_TET_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])  # face i is opposite vertex i


# ─────────────────────────── mesh topology ─────────────────────

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
    torsion_max, volume).  torsion_max = 0 (and a zero torsion column) when the mesh has no interior vertex."""
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
    if not torsion_max > 0.0:                     # no interior vertex (very coarse mesh): w = 0 everywhere, not 0/0
        torsion_max, w = 0.0, np.zeros_like(w)
    feats = np.column_stack([X, dist, direc, node_vol / node_vol.max(),
                             w / (torsion_max or 1.0)]).astype(np.float32)
    return feats, torsion_max, float(vol_n.sum())


def display_geometry(nodes, tets):
    """Normalised mesh and vertex features of one linear mesh [m]: {'X' float32 [Nv,3], 'Input_funcs' float32
    [Nv,9] (FEATURE_NAMES_3D), 'tets', 'scale', 'center', 'n_nodes', 'torsion_max', 'volume'}."""
    tets = np.asarray(tets, dtype=np.int64)
    X, center, scale = normalise(nodes, tets)
    feats, torsion_max, volume = node_features(X, tets)
    return {"X": X.astype(np.float32), "Input_funcs": feats, "tets": tets, "scale": scale,
            "center": center.astype(np.float64), "n_nodes": int(len(X)), "torsion_max": torsion_max,
            "volume": volume}


def geom_params_of(attrs):
    """The family's geom_params [m] of an H5 sample group (attrs['geom_params'] JSON; {} when absent)."""
    return json.loads(attrs.get("geom_params", "{}"))
