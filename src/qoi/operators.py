"""CST-style cavity figures of merit from N0 (Whitney) eigenmode DOFs — numpy / scipy reference.

Contract: docs/24_CAVITY_QOI.md §0 (keys, units, conventions); physics and discretisation: §1–§2.

Every figure of merit is a ratio of quadratic / linear functionals of the DOF vector u, evaluated
with fixed sparse per-geometry operators on the NORMALISED mesh (ξ = (x − center)/scale):

    U    = ½ ε0 s³ · uᵀMu
    P_c  = ½ R_s/(ωμ0)² · uᵀSu
    V    = |Σ_p q_p (A_z u)_p e^{jω s ζ_p/(βc)}| · s
    E_pk, B_pk = max_i ‖(E_surf u)_i‖, ‖(H_surf u)_i‖ (row triplets) · unit factors

and every reported value is rescaled to a stored energy U = 1 J.

Conventions (same as src/data/dataset_converter_3d.py and src/viz/nedelec.py): DOF e = (a, b) with
a < b (global vertex index), w_e = λ_a∇λ_b − λ_b∇λ_a, curl w_e = 2∇λ_a×∇λ_b.  `edges` may be in
any row order (the PKL's sorted order or skfem's) — DOF lookup goes through nedelec.tet_dofs.

Surface evaluation (docs §2.4): one point per boundary face (its centroid).  E is the value of the
face's own tet there ('centroid'); H is the nodal average onto the boundary vertices
(volume-weighted over all tets sharing the vertex, as CST-like field viewers do) interpolated
linearly to the centroid ('recovered').  Both are reduced to the components the exact wall field
has: the NORMAL part of E and the TANGENTIAL part of H (n × E = 0, n · H = 0 on a PEC wall).
`surface_operators(..., method=...)` exposes the other combinations (scripts/qoi_validation.py).

Known limitation (§2.2, §3): the wall loss uses the per-tet constant curl Ẽ of the boundary tets;
on CURVED walls approximated by flat facets this converges only to first order (pillbox: P_c
+6.5 % → +2.9 % for Ne 3k → 54k), flat walls are fine.
"""
import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as EPS0, mu_0 as MU0

from src.data.dataset_converter_3d import assemble_n0, tet_geometry
from src.viz.nedelec import _whitney_vectors, locate, tet_dofs

C0 = 299792458.0
SIGMA_CU = 5.8e7

QOI_LABELS = ('Q0', 'G_ohm', 'R_over_Q_ohm', 'R_sh_ohm', 'T_transit', 'Epk_Eacc',
              'Bpk_Eacc_mT_per_MVm')

QOI_KEYS = ('f_Hz', 'Rs_ohm', 'U_J', 'P_c_W', 'Q0', 'G_ohm', 'V_acc_V', 'T_transit', 'R_over_Q_ohm',
            'R_sh_ohm', 'L_acc_m', 'E_acc_Vm', 'E_pk_Vm', 'B_pk_T', 'Epk_Eacc', 'Bpk_Eacc_mT_per_MVm')

_TET_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])   # face i is opposite vertex i


# ─────────────────────────── helpers ───────────────────────────

def surface_resistance(f_hz, sigma=SIGMA_CU):
    """Normal-conductor surface resistance R_s = √(π f μ0 / σ) = √(ωμ0 / 2σ) [Ω]."""
    return np.sqrt(np.pi * np.asarray(f_hz, dtype=np.float64) * MU0 / sigma)


def _as_csr(A, n):
    """scipy sparse matrix or PKL CSR tuple (indptr, indices, data) → CSR float64 [n×n]."""
    if A is None:
        return None
    if isinstance(A, (tuple, list)):
        return sp.csr_matrix((np.asarray(A[2], dtype=np.float64), np.asarray(A[1]), np.asarray(A[0])),
                             shape=(n, n))
    return sp.csr_matrix(A, dtype=np.float64)


def _local_basis(X, tets, edges):
    """Per-tet DOF rows, local (a, b) ids, volumes, ∇λ and the 6 constant basis curls [Nt,6,3]."""
    dof, loc = tet_dofs(tets, edges, len(X))
    vol, grads = tet_geometry(X, tets)
    ga = np.take_along_axis(grads, loc[..., 0][:, :, None].repeat(3, 2), axis=1)
    gb = np.take_along_axis(grads, loc[..., 1][:, :, None].repeat(3, 2), axis=1)
    return dof, loc, vol, grads, 2.0 * np.cross(ga, gb)


def boundary_face_data(X, tets):
    """Boundary faces with their adjacent tet (vectorised, no loops).

    Returns dict:
      faces  [Nf,3] vertex triples ordered so (p1−p0)×(p2−p0) points OUTWARD — same rows and order
             as dataset_converter_3d.boundary_faces
      tet    [Nf]   the (unique) tet the face belongs to
      opp    [Nf]   local index (0..3) in that tet of the vertex opposite the face
      normal [Nf,3] unit outward normal;  area [Nf]  (normalised units)
    """
    t = np.asarray(tets, dtype=np.int64)
    f = t[:, _TET_FACES].reshape(-1, 3)
    tid, opp = np.repeat(np.arange(len(t)), 4), np.tile(np.arange(4), len(t))
    _, inv, cnt = np.unique(np.sort(f, axis=1), axis=0, return_inverse=True, return_counts=True)
    b = cnt[inv.ravel()] == 1
    f, tid, opp = f[b], tid[b], opp[b]
    p0, p1, p2 = X[f[:, 0]], X[f[:, 1]], X[f[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    inward = (n * (X[t[tid, opp]] - p0)).sum(1) > 0
    f[inward] = f[inward][:, [0, 2, 1]]
    n[inward] = -n[inward]
    nn = np.linalg.norm(n, axis=1)
    return {"faces": f, "tet": tid, "opp": opp, "normal": n / nn[:, None], "area": 0.5 * nn}


def _coo(rows, cols, vals, shape):
    A = sp.csr_matrix((np.ravel(vals), (np.ravel(rows), np.ravel(cols))), shape=shape)
    A.sum_duplicates()
    A.eliminate_zeros()
    return A


# ─────────────────────────── wall-loss form S ──────────────────

def wall_loss_matrix(X, tets, edges, bf=None, basis=None):
    """S [Ne×Ne] CSR: S_ij = ∮_∂Ω̃ (n × curl w_i)·(n × curl w_j) dS̃ on the normalised boundary (the
    wall H is ∝ curl E, constant per boundary tet), exact: area · (c_i·P c_j), P = I − nnᵀ."""
    X = np.asarray(X, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    ne = len(edges)
    bf = boundary_face_data(X, tets) if bf is None else bf
    dof, loc, vol, grads, curls = _local_basis(X, tets, edges) if basis is None else basis
    t, n, A = bf["tet"], bf["normal"], bf["area"]
    P = np.eye(3)[None] - n[:, :, None] * n[:, None, :]                  # [Nf,3,3]
    c = curls[t]                                                         # [Nf,6,3]
    Sl = A[:, None, None] * np.einsum("fic,fcd,fjd->fij", c, P, c)
    d = dof[t]
    return _coo(np.repeat(d, 6, axis=1), np.tile(d, (1, 6)), Sl, (ne, ne))


# ─────────────────────────── beam axis ─────────────────────────

def _axis_line(axis_xy=(0.0, 0.0), axis_dir=None, axis_point=None):
    """(unit direction d, physical point p) of the beam axis: default the line (x, y) = axis_xy
    along z; axis_dir 'x' | 'y' | 'z' or a 3-vector, axis_point a physical point on it [m]."""
    if axis_dir is None:
        axis_dir = "z"
    if isinstance(axis_dir, str):
        d = np.eye(3)["xyz".index(axis_dir.lower())]
    else:
        d = np.asarray(axis_dir, dtype=np.float64).reshape(3)
        d = d / np.linalg.norm(d)
    if axis_point is None:
        axis_point = (float(axis_xy[0]), float(axis_xy[1]), 0.0)
    return d, np.asarray(axis_point, dtype=np.float64).reshape(3)


def axis_operator(X, tets, edges, scale, center, n_axis=401, axis_xy=(0.0, 0.0), basis=None,
                  axis_dir=None, axis_point=None):
    """Axis sampling: midpoints ζ_p of n_axis equal cells over the extent of the mesh along the beam
    axis (default: the physical line (x, y) = axis_xy along z; or axis_dir / axis_point, see
    _axis_line), located in the mesh; q_p = Δζ inside, 0 outside.
    Returns (Az CSR [P×Ne], zeta [P], q [P], L_axis).  Az rows: the axial component of the Whitney
    field at the point."""
    X = np.asarray(X, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    dof, loc, vol, grads, curls = _local_basis(X, tets, edges) if basis is None else basis
    d, p = _axis_line(axis_xy, axis_dir, axis_point)
    pn = (p - np.asarray(center, dtype=np.float64).reshape(3)) / float(scale)
    perp = pn - (pn @ d) * d                                  # foot of the axis through the origin
    t = X @ d
    z0, z1 = t.min(), t.max()
    dz = (z1 - z0) / n_axis
    zeta = z0 + (np.arange(n_axis) + 0.5) * dz
    pts = perp[None, :] + zeta[:, None] * d[None, :]
    tid, bary = locate(X, tets, pts)
    inside = tid >= 0
    q = np.where(inside, dz, 0.0)
    p_in, ti = np.flatnonzero(inside), tid[inside]
    vals = _whitney_vectors(grads[ti], loc[ti], bary[inside]) @ d       # [m,6]
    Az = _coo(np.repeat(p_in[:, None], 6, 1), dof[ti], vals, (n_axis, len(edges)))
    return Az, zeta, q, float(q.sum())


# Beam axis per generator family when it is not the default line x = y = 0 along z (the families are
# built with z = beam axis, docs/24 §0.2) — hwr: the coaxial line runs along z, the beam crosses it
# transversally along x at mid-length (cavity_shapes.build_hwr).
def beam_axis(shape_type, X, scale, center):
    """{'axis_dir', 'axis_point'} kwargs of build_qoi_operators for a generator family ({} = default)."""
    if str(shape_type) == "hwr":
        X = np.asarray(X, dtype=np.float64)
        zmid = 0.5 * (X[:, 2].min() + X[:, 2].max()) * float(scale) + float(np.asarray(center).reshape(3)[2])
        return {"axis_dir": "x", "axis_point": (0.0, 0.0, zmid)}
    return {}


# ─────────────────────────── surface fields ────────────────────

SURFACE_METHOD = ("centroid", "recovered")   # (physical E, physical H) — docs/24 §2.4


def surface_operators(X, tets, edges, method=SURFACE_METHOD, project=True, bf=None, basis=None):
    """(Esurf, Hsurf) CSR [3Nf×Ne]: Cartesian wall-field components at the boundary-face centroids
    (rows 3i..3i+2 ↔ face i of boundary_face_data): Esurf ↔ Ẽ, Hsurf ↔ curl Ẽ (unit factors are
    applied in qoi_from_dofs).

    method: one name for both, or (method for the physical E, method for the physical H):
      'centroid'  the value of the face's own tet at the centroid (Whitney value with λ_opp = 0 and
                  1/3 elsewhere, or the tet's constant curl);
      'recovered' nodal average onto the boundary vertices (volume-weighted over EVERY tet sharing the
                  vertex; Whitney value at the vertex, or the tet's constant curl), then linear
                  interpolation to the centroid (= mean of the 3 vertex values).
    Default ('centroid', 'recovered'): the most accurate pair on the analytic pillbox / box (§2.4, §3).
    project=True keeps only the physical wall components: n nᵀ E and (I − n nᵀ) H (exact PEC wall
    fields have nothing else).  For 'centroid' both projections are identities in exact
    arithmetic (tangential Ẽ and normal curl Ẽ on a wall face only involve the zero wall DOFs).
    """
    m_E, m_H = (method, method) if isinstance(method, str) else tuple(method)
    for m in (m_E, m_H):
        if m not in ("centroid", "recovered"):
            raise ValueError(f"surface method must be 'recovered' or 'centroid', got {m!r}")
    X = np.asarray(X, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    ne = len(edges)
    bf = boundary_face_data(X, tets) if bf is None else bf
    dof, loc, vol, grads, curls = _local_basis(X, tets, edges) if basis is None else basis
    nf = len(bf["faces"])
    n = bf["normal"]
    Pn = n[:, :, None] * n[:, None, :]
    I3 = np.broadcast_to(np.eye(3), (nf, 3, 3))
    P_E, P_H = (Pn, I3 - Pn) if project else (I3, I3)

    def centroid_op(kind, Pm):
        t = bf["tet"]
        if kind == "val":
            bary = np.full((nf, 4), 1.0 / 3.0)
            bary[np.arange(nf), bf["opp"]] = 0.0
            V = _whitney_vectors(grads[t], loc[t], bary)                   # [Nf,6,3]
        else:
            V = curls[t]
        rows = (3 * np.arange(nf)[:, None, None] + np.arange(3)[None, :, None]).repeat(6, 2)  # [Nf,3,6]
        cols = dof[t][:, None, :].repeat(3, 1)
        return _coo(rows, cols, np.einsum("fcd,fed->fce", Pm, V), (3 * nf, ne))

    def recovered_op(kind, Pm):
        bv = np.unique(bf["faces"])
        vmap = np.full(len(X), -1, dtype=np.int64)
        vmap[bv] = np.arange(len(bv))
        touch = np.flatnonzero((vmap[tets] >= 0).any(1))                   # tets with a wall vertex
        tt = tets[touch]
        isb = vmap[tt] >= 0                                                # [m,4]
        wsum = np.zeros(len(bv))
        np.add.at(wsum, vmap[tt][isb], np.repeat(vol[touch][:, None], 4, 1)[isb])
        wt = np.where(isb, vol[touch][:, None] / wsum[np.maximum(vmap[tt], 0)], 0.0)   # [m,4]
        if kind == "val":
            eye = np.eye(4)
            V = np.stack([_whitney_vectors(grads[touch], loc[touch], np.tile(eye[i], (len(touch), 1)))
                          for i in range(4)], axis=1)                      # [m,4,6,3]
        else:
            V = np.repeat(curls[touch][:, None], 4, axis=1)
        vrow = np.broadcast_to((3 * np.maximum(vmap[tt], 0))[:, :, None, None]
                               + np.arange(3)[None, None, None, :], V.shape)
        vcol = np.broadcast_to(dof[touch][:, None, :, None], V.shape)
        keep = np.broadcast_to(isb[:, :, None, None], V.shape)
        Rv = _coo(vrow[keep], vcol[keep], (wt[:, :, None, None] * V)[keep], (3 * len(bv), ne))
        shp = (nf, 3, 3, 3)                                                # [face, vertex k, c, c']
        r = np.broadcast_to(3 * np.arange(nf)[:, None, None, None] + np.arange(3)[None, None, :, None], shp)
        c = np.broadcast_to(3 * vmap[bf["faces"]][:, :, None, None] + np.arange(3)[None, None, None, :], shp)
        v = np.broadcast_to(Pm[:, None, :, :] / 3.0, shp)
        return (_coo(r, c, v, (3 * nf, 3 * len(bv))) @ Rv).tocsr()

    ops = {"centroid": centroid_op, "recovered": recovered_op}
    return ops[m_E]("val", P_E), ops[m_H]("curl", P_H)


# ─────────────────────────── public API (§0.3) ─────────────────

def build_qoi_operators(X, tets, edges, scale, center, M=None, n_axis=401, axis_xy=(0.0, 0.0),
                        axis_dir=None, axis_point=None):
    """Per-geometry QoI operators on the NORMALISED mesh (docs/24 §0.3).

    X [Nv,3] normalised vertices, tets [Nt,4], edges [Ne,2] (DOF order; any row order), scale s [m],
    center [m] (3,), M the N0 mass (CSR or PKL CSR tuple; assembled if None),
    n_axis axis sample count, axis_xy the physical (x, y) of the beam axis [m] (along z); or a general
    axis: axis_dir 'x' | 'y' | 'z' | 3-vector and axis_point [m] (beam_axis(shape_type, …) per family).
    Returns dict with keys 'scale', 'M', 'S', 'Az', 'zeta', 'q', 'L_axis', 'Esurf', 'Hsurf',
    'face_area' (see §0.3).  Surface rows: boundary-face centroids; E from the face's own tet, H
    nodally recovered (SURFACE_METHOD, §2.4); n nᵀE and (I − n nᵀ)H kept.
    """
    X = np.asarray(X, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64)
    ne = len(edges)
    if M is None:
        if np.all(np.diff(edges[:, 0] * len(X) + edges[:, 1]) > 0):
            M = assemble_n0(X, tets, edges)[0]
        else:                                  # assemble_n0 needs sorted edges: assemble, then permute
            from src.data.dataset_converter_3d import _edge_index, edges_from_tets
            es = edges_from_tets(tets)
            Ms = assemble_n0(X, tets, es)[0]
            p = _edge_index(es, len(X), edges)
            M = Ms[p][:, p]
    M = _as_csr(M, ne)
    basis = _local_basis(X, tets, edges)
    bf = boundary_face_data(X, tets)
    S = wall_loss_matrix(X, tets, edges, bf=bf, basis=basis)
    Az, zeta, q, L_axis = axis_operator(X, tets, edges, scale, center, n_axis, axis_xy, basis=basis,
                                        axis_dir=axis_dir, axis_point=axis_point)
    Esurf, Hsurf = surface_operators(X, tets, edges, SURFACE_METHOD, True, bf=bf, basis=basis)
    return {"scale": float(scale), "M": M, "S": S, "Az": Az, "zeta": zeta, "q": q,
            "L_axis": L_axis, "Esurf": Esurf, "Hsurf": Hsurf, "face_area": bf["area"]}


def _peak(A, U):
    """max over surface points of the Euclidean norm of the row triplets of A @ U → [K]."""
    v = np.asarray(A @ U).reshape(-1, 3, U.shape[1])
    return np.sqrt((v ** 2).sum(1)).max(0) if len(v) else np.zeros(U.shape[1])


def qoi_from_dofs(ops, U, f_hz, Rs=None, sigma=SIGMA_CU, beta=1.0, L_acc=None, convention="linac"):
    """Figures of merit of the mode(s) U [Ne] or [Ne,K] at frequency f_hz [Hz] (scalar or [K]),
    normalised to a stored energy of 1 J (docs/24 §0.2).  Amplitude and sign of U are irrelevant.

    Rs [Ω] scalar or [K] overrides the copper model R_s = √(π f μ0/σ) (σ = sigma) at each mode's own
    f; beta the particle velocity (transit phase ω z/(βc)); L_acc [m] the accelerating length
    (default: the axis chord inside Ω, ops['L_axis']·s); convention 'linac' (R/Q = V²/(ωU),
    R_sh = V²/P_c) or 'circuit' (both halved).
    Returns dict {key: float64 ndarray [K]} for QOI_KEYS (K = 1 for a 1-D U).
    """
    if convention not in ("linac", "circuit"):
        raise ValueError(f"convention must be 'linac' or 'circuit', got {convention!r}")
    U = np.asarray(U, dtype=np.float64)
    if U.ndim == 1:
        U = U[:, None]
    K = U.shape[1]
    f = np.broadcast_to(np.asarray(f_hz, dtype=np.float64).ravel(), (K,)).copy()
    w = 2.0 * np.pi * f
    s = ops["scale"]
    Rs = surface_resistance(f, sigma) if Rs is None else np.broadcast_to(
        np.asarray(Rs, dtype=np.float64).ravel(), (K,)).copy()
    uMu = np.einsum("ik,ik->k", U, ops["M"] @ U)
    uSu = np.einsum("ik,ik->k", U, ops["S"] @ U)
    Az = np.asarray(ops["Az"] @ U)                                         # [P,K]
    q = np.asarray(ops["q"], dtype=np.float64)
    phase = np.exp(1j * np.outer(np.asarray(ops["zeta"]) * s, w / (beta * C0)))   # [P,K]
    Vc = np.abs((q[:, None] * Az * phase).sum(0))
    V0 = (q[:, None] * np.abs(Az)).sum(0)
    ep, hp = _peak(ops["Esurf"], U), _peak(ops["Hsurf"], U)
    U_raw = 0.5 * EPS0 * s ** 3 * uMu
    Pc_raw = 0.5 * Rs / (w * MU0) ** 2 * uSu
    V_raw, V0_raw = s * Vc, s * V0
    Epk_raw, Bpk_raw = ep, hp / (w * s)
    a = 1.0 / np.sqrt(U_raw)                                               # amplitude → U = 1 J
    fac = 1.0 if convention == "linac" else 0.5
    P_c = a ** 2 * Pc_raw
    V = a * V_raw
    L = np.broadcast_to(np.asarray(ops["L_axis"] * s if L_acc is None else L_acc, dtype=np.float64),
                        (K,)).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        Q0 = w / P_c
        E_acc = V / L
        E_pk, B_pk = a * Epk_raw, a * Bpk_raw
        return {
            "f_Hz": f, "Rs_ohm": Rs, "U_J": np.ones(K), "P_c_W": P_c, "Q0": Q0, "G_ohm": Q0 * Rs,
            "V_acc_V": V, "T_transit": V_raw / V0_raw, "R_over_Q_ohm": fac * V ** 2 / w,
            "R_sh_ohm": fac * V ** 2 / P_c, "L_acc_m": L, "E_acc_Vm": E_acc, "E_pk_Vm": E_pk,
            "B_pk_T": B_pk, "Epk_Eacc": E_pk / E_acc, "Bpk_Eacc_mT_per_MVm": 1e9 * B_pk / E_acc,
        }


_BUILD_KW = ("n_axis", "axis_xy", "axis_dir", "axis_point")


def cavity_qoi(geom, U, f_ghz, **kw):
    """QoI of PKL geometry `geom` (dict with X, tets, edges, scale, center[, M]) for E DOFs U [Ne] or
    [Ne,K] at f_ghz [GHz].  kw: n_axis / axis_xy go to
    build_qoi_operators, the rest to qoi_from_dofs.  Builds the operators on every call — cache
    build_qoi_operators(...) yourself when a geometry is reused."""
    bkw = {k: kw.pop(k) for k in _BUILD_KW if k in kw}
    ops = build_qoi_operators(geom["X"], geom["tets"], geom["edges"], geom["scale"], geom["center"],
                              M=geom.get("M"), **bkw)
    return qoi_from_dofs(ops, U, np.asarray(f_ghz, dtype=np.float64) * 1e9, **kw)
