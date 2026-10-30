"""Point location in a linear tetrahedral mesh and barycentric sampling of vertex values (cut planes of the model's
input features, src/service/features.py; the oracle of tests/test_hcurl_display.py).  The field display itself
evaluates HCurl(p2) directly on the curved mesh: src/viz/hcurl_display.py.
"""
import numpy as np
from scipy.spatial import cKDTree

from src.data.mesh_features import tet_geometry

_AXES = {"x": 0, "y": 1, "z": 2}


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


def locate_grid(X, tets, lo, d, dims, tol=1e-9, chunk=2_000_000):
    """Point location for a regular grid lo + d·(i, j, k), i < dims[0] … (x fastest; d a scalar or one
    spacing per axis; a dimension of 1 makes a plane): every tet visits the grid points of its bounding
    box (barycentric test), so the cost is linear in the points covered — unlike locate(), whose points
    outside the mesh each search 16 → 256 candidate tets (an off-centre cut plane took seconds).
    Returns (tet_id [N] int64, -1 outside; bary [N,4], NaN outside), N = prod(dims)."""
    X = np.asarray(X, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    dims = np.asarray(dims, dtype=np.int64)
    lo = np.asarray(lo, dtype=np.float64)
    d = np.broadcast_to(np.asarray(d, dtype=np.float64), (3,))
    P = X[tets]                                                     # [T,4,3]
    a = np.ceil((P.min(1) - lo) / d - 1e-9).astype(np.int64)        # first / last grid index in the box
    b = np.floor((P.max(1) - lo) / d + 1e-9).astype(np.int64)
    hit = ((a <= b) & (b >= 0) & (a <= dims - 1)).all(1)            # the box holds grid points
    imin, imax = np.clip(a, 0, dims - 1), np.clip(b, 0, dims - 1)
    span = np.where(hit[:, None], imax - imin + 1, 0)
    count = span.prod(1)
    J = np.linalg.inv(np.transpose(P[:, 1:] - P[:, :1], (0, 2, 1)))  # [T,3,3]: x − p0 → (λ1, λ2, λ3)
    n = int(np.prod(dims))
    tet_id = np.full(n, -1, dtype=np.int64)
    bary = np.full((n, 4), np.nan)
    ends = np.cumsum(count)
    t0 = 0
    while t0 < len(tets):                                           # chunks of ≤ `chunk` (tet, point) pairs
        t1 = int(np.searchsorted(ends, (ends[t0 - 1] if t0 else 0) + chunk, side="right"))
        t1 = max(t1, t0 + 1)
        c = count[t0:t1]
        tid = np.repeat(np.arange(t0, t1), c)
        loc = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        sx, sy = span[tid, 0], span[tid, 1]
        ijk = imin[tid] + np.stack([loc % sx, (loc // sx) % sy, loc // (sx * sy)], 1)
        x = lo + d * ijk
        lam = np.einsum("pij,pj->pi", J[tid], x - P[tid, 0])
        lam = np.concatenate([1.0 - lam.sum(1, keepdims=True), lam], 1)
        ok = lam.min(1) >= -tol
        g = ijk[ok, 0] + dims[0] * (ijk[ok, 1] + dims[1] * ijk[ok, 2])
        tet_id[g], bary[g] = tid[ok], lam[ok]
        t0 = t1
    return tet_id, bary


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
