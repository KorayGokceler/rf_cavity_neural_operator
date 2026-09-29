"""Whitney N0 field reconstruction and point/plane sampling for 3D visualisation.

Contract: scratchpad/viz3d_contract.md §A1. Conventions follow
src/data/dataset_converter_3d.py: DOF e = (a, b), a < b (global vertex index),
w_e = λ_a∇λ_b − λ_b∇λ_a, F = Σ_e u_e w_e.  F is whatever field the DOFs are
(H for H-formulation PKLs, E for E ones, pred['field']); the functions are
field-agnostic and keep the historical 'H' names / dict keys. `edges` here is only assumed to
contain the (low, high) pair of every tet edge exactly once — NOT to be sorted
(skfem's mesh.edges, used by tests/maxwell3d_synth.py, is not lexicographically
sorted) — so DOF lookup sorts its own key copy rather than assuming order.
"""
import numpy as np
from scipy.spatial import cKDTree

from src.data.dataset_converter_3d import tet_geometry

_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
_AXES = {"x": 0, "y": 1, "z": 2}


def tet_dofs(tets, edges, n_nodes):
    """Per-tet edge DOFs, global row into `edges` and local low->high endpoints.

    Does not assume `edges` is sorted: builds a sorted copy of its (a*n_nodes+b)
    keys and maps back through the sort permutation (vectorised, no tet loop).

    Parameters
    ----------
    tets : [Nt,4] int
    edges : [Ne,2] int, each row a tet edge's (low, high) global vertex pair
    n_nodes : int, vertex count (packing key base)

    Returns
    -------
    dof : [Nt,6] int64   row of each local tet edge in `edges`
    loc : [Nt,6,2] int64 local vertex ids of that edge, ordered low->high global index
    """
    tets = np.asarray(tets, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64)
    loc = _TET_EDGES[None].repeat(len(tets), 0)        # [Nt,6,2] local vertex ids
    gv = tets[:, _TET_EDGES]                           # [Nt,6,2] global vertex ids
    swap = gv[..., 0] > gv[..., 1]                     # orient low -> high global index
    loc = np.where(swap[..., None], loc[..., ::-1], loc)
    gv = np.where(swap[..., None], gv[..., ::-1], gv)

    def key(ab):
        return ab[..., 0] * n_nodes + ab[..., 1]

    order = np.argsort(key(edges), kind="stable")
    skey = key(edges)[order]
    pos = np.searchsorted(skey, key(gv))
    dof = order[np.clip(pos, 0, len(order) - 1)]
    return dof, loc


def _whitney_vectors(grads, loc, bary):
    """Whitney basis vectors λ_a∇λ_b − λ_b∇λ_a at barycentric coords `bary`.

    grads [N,4,3] (∇λ_i), loc [N,6,2] local (a,b) ids, bary [N,4] -> [N,6,3].
    """
    ia, ib = loc[..., 0], loc[..., 1]
    ga = np.take_along_axis(grads, ia[:, :, None].repeat(3, 2), axis=1)
    gb = np.take_along_axis(grads, ib[:, :, None].repeat(3, 2), axis=1)
    la = np.take_along_axis(bary, ia, axis=1)
    lb = np.take_along_axis(bary, ib, axis=1)
    return la[..., None] * gb - lb[..., None] * ga


def cell_field(X, tets, edges, u):
    """Whitney N0 field value at every tet centroid.

    H(x) = Σ_e u_e (λ_a∇λ_b − λ_b∇λ_a); at the centroid λ = (1/4,1/4,1/4,1/4).

    Parameters
    ----------
    X : [Nv,3] float, tets : [Nt,4] int, edges : [Ne,2] int (see module docstring)
    u : [Ne] or [Ne,K] float  edge DOFs (one or several modes)

    Returns
    -------
    H : [Nt,3,K] float, or [Nt,3] if `u` was 1-D.
    """
    X = np.asarray(X, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    u = np.asarray(u, dtype=np.float64)
    was_1d = u.ndim == 1
    if was_1d:
        u = u[:, None]
    dof, loc = tet_dofs(tets, edges, len(X))
    _, grads = tet_geometry(X, tets)
    bary = np.full((len(tets), 4), 0.25)
    W = _whitney_vectors(grads, loc, bary)             # [Nt,6,3]
    ue = u[dof]                                        # [Nt,6,K]
    H = np.einsum("tec,tek->tck", W, ue)
    return H[..., 0] if was_1d else H


def _nodal_average(tets, n_nodes, vol, vals):
    """Volume-weighted average onto vertices of per-(tet, local vertex) values [Nt,4,...]."""
    out = np.zeros((n_nodes,) + vals.shape[2:])
    w = np.zeros(n_nodes)
    for i in range(4):
        np.add.at(out, tets[:, i], vol.reshape((-1,) + (1,) * (vals.ndim - 2)) * vals[:, i])
        np.add.at(w, tets[:, i], vol)
    return out / np.maximum(w, 1e-300).reshape((-1,) + (1,) * (vals.ndim - 2))


def vertex_field(X, tets, edges, u, curl=False):
    """Continuous P1 recovery of the N0 field (or of its curl) for display.

    Each tet's field is evaluated at its own 4 vertices and averaged over the
    tets sharing a vertex, volume-weighted (nodal averaging, as field viewers
    such as CST do).  The raw N0 field is linear per tet with a normal jump
    across faces, which shows up as facets on a coarse mesh.  curl=True: the
    per-tet constant curl F = Σ u_e 2∇λ_a×∇λ_b instead (∝ E of the mode for
    H DOFs, ∝ H for E DOFs).

    Returns [Nv,3,K], or [Nv,3] if `u` was 1-D.
    """
    X = np.asarray(X, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    u = np.asarray(u, dtype=np.float64)
    was_1d = u.ndim == 1
    if was_1d:
        u = u[:, None]
    dof, loc = tet_dofs(tets, edges, len(X))
    vol, grads = tet_geometry(X, tets)
    ue = u[dof]                                                     # [Nt,6,K]
    if curl:
        ga = np.take_along_axis(grads, loc[..., 0][:, :, None].repeat(3, 2), axis=1)
        gb = np.take_along_axis(grads, loc[..., 1][:, :, None].repeat(3, 2), axis=1)
        c = np.einsum("tec,tek->tck", 2.0 * np.cross(ga, gb), ue)   # [Nt,3,K]
        vals = np.repeat(c[:, None], 4, axis=1)
    else:
        eye = np.eye(4)
        vals = np.stack([np.einsum("tec,tek->tck", _whitney_vectors(grads, loc, np.tile(eye[i], (len(tets), 1))), ue)
                         for i in range(4)], axis=1)                # [Nt,4,3,K]
    out = _nodal_average(tets, len(X), vol, vals)
    return out[..., 0] if was_1d else out


def locate(X, tets, points, tol=1e-9):
    """Vectorised point-in-tet-mesh location (centroid k-d tree + barycentric test).

    Candidates come from a k-nearest-centroid query (k grows 16 -> 64 -> 256 for
    points not yet resolved, each stage only re-querying what's left unresolved);
    a candidate is accepted when min(bary) >= -tol. A point still unresolved past
    k=256 is treated as outside — for a reasonably shaped mesh the true containing
    tet's centroid is essentially always among the 256 nearest, and growing k to
    "all tets" (checked and rejected: ~50x slower on a 17k-tet mesh, since cKDTree
    sorts the full candidate set) buys robustness on pathological meshes only.

    Parameters
    ----------
    X : [Nv,3] float, tets : [Nt,4] int, points : [P,3] float, tol : float

    Returns
    -------
    tet_id : [P] int64  containing tet, -1 if outside the mesh
    bary : [P,4] float  barycentric coordinates (sum to 1), NaN if outside
    """
    X = np.asarray(X, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    n, nt = len(pts), len(tets)
    tet_id = np.full(n, -1, dtype=np.int64)
    bary = np.full((n, 4), np.nan)
    if n == 0 or nt == 0:
        return tet_id, bary
    _, grads = tet_geometry(X, tets)
    p0 = X[tets[:, 0]]                                 # [Nt,3]
    tree = cKDTree(X[tets].mean(1))
    unresolved = np.arange(n)
    for k in (16, 64, 256):
        k = min(k, nt)
        if len(unresolved) == 0 or k == 0:
            break
        _, cand = tree.query(pts[unresolved], k=k)
        cand = cand.reshape(len(unresolved), k)
        d = pts[unresolved][:, None, :] - p0[cand]                  # [m,k,3]
        b = np.einsum("mkic,mkc->mki", grads[cand], d)              # [m,k,4]
        b[..., 0] += 1.0
        valid = b.min(-1) >= -tol
        found = valid.any(-1)
        first = np.argmax(valid, axis=1)
        sel = unresolved[found]
        tet_id[sel] = cand[found, first[found]]
        bary[sel] = b[found, first[found]]
        unresolved = unresolved[~found]
    return tet_id, bary


def eval_field(X, tets, edges, u, points):
    """Whitney N0 field value at arbitrary points (NaN outside the mesh).

    Parameters
    ----------
    X, tets, edges : mesh (see module docstring)
    u : [Ne] or [Ne,K] float
    points : [P,3] float

    Returns
    -------
    H : [P,3,K] float, or [P,3] if `u` was 1-D. Rows outside the mesh are NaN.
    """
    X = np.asarray(X, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    u = np.asarray(u, dtype=np.float64)
    was_1d = u.ndim == 1
    if was_1d:
        u = u[:, None]
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    tet_id, bary = locate(X, tets, pts)
    K = u.shape[1]
    H = np.full((len(pts), 3, K), np.nan)
    inside = tet_id >= 0
    if inside.any():
        ti = tet_id[inside]
        sub_tets = tets[ti]                             # [m,4]
        dof, loc = tet_dofs(sub_tets, edges, len(X))
        _, grads = tet_geometry(X, sub_tets)
        W = _whitney_vectors(grads, loc, bary[inside])   # [m,6,3]
        ue = u[dof]                                      # [m,6,K]
        H[inside] = np.einsum("mec,mek->mck", W, ue)
    return H[..., 0] if was_1d else H


def plane_grid(X, axis="y", offset=0.0, res=121, pad=0.02):
    """Regular grid on the plane {coord[axis] == offset} over the mesh bbox of the
    two other axes (padded by `pad`): (points [res²,3], U, V, (iu, iv))."""
    X = np.asarray(X, dtype=np.float64)
    ia = _AXES[axis]
    iu, iv = [c for c in range(3) if c != ia]
    lo, hi = X.min(0), X.max(0)
    p = pad * (hi - lo)
    U, V = np.meshgrid(np.linspace(lo[iu] - p[iu], hi[iu] + p[iu], res),
                       np.linspace(lo[iv] - p[iv], hi[iv] + p[iv], res), indexing="xy")
    pts = np.empty((U.size, 3))
    pts[:, ia], pts[:, iu], pts[:, iv] = offset, U.ravel(), V.ravel()
    return pts, U, V, (iu, iv)


def interp_vertex(tets, tet_id, bary, nodal):
    """Barycentric interpolation of vertex values nodal [Nv,...] at located points
    (tet_id, bary from locate); NaN outside."""
    out = np.full((len(tet_id),) + nodal.shape[1:], np.nan)
    m = tet_id >= 0
    out[m] = np.einsum("pi,pi...->p...", bary[m], nodal[tets[tet_id[m]]])
    return out


def plane_sample(X, tets, edges, u, axis="y", offset=0.0, res=121, pad=0.02, smooth=False):
    """Sample the field on an axis-aligned plane, on a regular grid.
    smooth=True samples the nodal-averaged field (vertex_field) instead of the raw N0 one.

    plane = {coord[axis] == offset}; the two remaining axes, in (x,y,z) order,
    become (u, v). Grid bounds are the mesh bbox of those two axes, padded by
    `pad` (fraction of each axis's extent).

    Returns
    -------
    dict with keys 'axis', 'offset', 'u_label', 'v_label',
    'U', 'V' ([res,res] meshgrid, indexing='xy'), 'inside' ([res,res] bool),
    'H' ([res,res,3,K] or [res,res,3] if `u` is 1-D, NaN outside), 'in_plane' (iu, iv).
    """
    X = np.asarray(X, dtype=np.float64)
    labels = {0: "x", 1: "y", 2: "z"}
    pts, U, V, (iu, iv) = plane_grid(X, axis, offset, res, pad)
    if smooth:
        tet_id, bary = locate(X, tets, pts)
        H = interp_vertex(np.asarray(tets, dtype=np.int64), tet_id, bary, vertex_field(X, tets, edges, u))
    else:
        H = eval_field(X, tets, edges, u, pts)
    was_1d = H.ndim == 2
    if was_1d:
        H = H[:, :, None]
    K = H.shape[-1]
    Hgrid = H.reshape(res, res, 3, K)
    inside = ~np.isnan(Hgrid[..., 0, 0])
    if was_1d:
        Hgrid = Hgrid[..., 0]
    return {"axis": axis, "offset": float(offset), "u_label": labels[iu], "v_label": labels[iv],
            "U": U, "V": V, "inside": inside, "H": Hgrid, "in_plane": (iu, iv)}
