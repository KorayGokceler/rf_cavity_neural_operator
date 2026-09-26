"""Whitney N0 field reconstruction and point/plane sampling for 3D visualisation.

Contract: scratchpad/viz3d_contract.md §A1. Conventions follow
src/data/dataset_converter_3d.py: DOF e = (a, b), a < b (global vertex index),
w_e = λ_a∇λ_b − λ_b∇λ_a, H = Σ_e u_e w_e. `edges` here is only assumed to
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


def plane_sample(X, tets, edges, u, axis="y", offset=0.0, res=121, pad=0.02):
    """Sample the field on an axis-aligned plane, on a regular grid.

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
    ia = _AXES[axis]
    iu, iv = [c for c in range(3) if c != ia]
    labels = {0: "x", 1: "y", 2: "z"}

    def bounds(col):
        lo, hi = X[:, col].min(), X[:, col].max()
        p = pad * (hi - lo)
        return lo - p, hi + p

    lo_u, hi_u = bounds(iu)
    lo_v, hi_v = bounds(iv)
    U, V = np.meshgrid(np.linspace(lo_u, hi_u, res), np.linspace(lo_v, hi_v, res), indexing="xy")

    pts = np.empty((U.size, 3))
    pts[:, ia] = offset
    pts[:, iu] = U.ravel()
    pts[:, iv] = V.ravel()
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
