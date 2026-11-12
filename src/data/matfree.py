"""Matrix-free HCurl(p = 2) operators of the learned-field model (docs/40).

The model needs, per geometry, K, M (HCurl p2 curl-curl / mass), the DOF map u = C·vec(Pψ) and the token
points / weights.  `highorder.field_operators` assembles them as global sparse matrices (285-1300 MB per
geometry at 20-50k tets).  They are all functions of ONE small record of the curved mesh (~640 B per tet):

  * the HCurl p2 hierarchical basis of an element depends on the element only through the RANK PERMUTATION
    of its four global vertex numbers (24 classes): reference shapes / curls and the dual functionals of
    the ConvertOperator are 24 small tables, built from NGSolve alone (`reference_tables`, cached on disk);
  * for geometry order <= 3 (with or without the generator's smooth deformation) the element map x(ξ) is
    an exact cubic: 20 monomial coefficients x 3 per element (`build_record`), hence J at any point;
  * the global operators are element sums:  K = Σ_e scatter(K_e)·s,  M = Σ_e scatter(M_e)/s  (s = the
    ξ-scaling `ops['scale']`), K_e / M_e from the degree-8 TET rule (46 points = volume_measure(2)).

Everything below is float64 torch, device agnostic, no Python loop over elements (`chunk` bounds the
transient memory).  The record carries no gradient; V and ψ do.  Nothing assembles a global matrix.

Public API (shapes: nt tets, n = n_dof, m basis vectors, 30 local DOFs, 14 token points):
    reference_tables(order=2)                  numpy tables (disk-cached, keyed by ngsolve version + FORMAT)
    build_record(mesh, order=2, scale=None, center=None)    numpy record of a curved NGSolve mesh
    to_torch(rec, device=None)                 the record as torch tensors (what the ops below use)
    jacobians(rec, rule)                       J [nt, nq, 3, 3], rule "mass" (46 pts) or "token" (14 pts)
    element_matrices(rec)                      (Me, Ke) [nt, 30, 30], the model's M and K element parts
    apply(Ae, dofs, V)                         A·V  [n, m]
    gram(Ae, dofs, V, T=None)                  VᵀAV or the [V|T] block Gram
    grams_plain(rec, V, T)                     what hcurl.hcurl_grams_plain returns, for one geometry
    psi_map(rec) / psi_to_dofs(rec, psi)       ψ [nt, 14, m, 3] → u = C·vec(Pψ) [n, m]  (before DOF masking)
    token_points(rec) / token_weights(rec)     ops['qx'] / ops['qw'] [nt, 14, 3] / [nt, 14]
    check_against_ngsolve(mesh)                debug: the errors against highorder.field_operators
"""
import itertools
import os
import re
import tempfile
import time

import numpy as np
import torch

FORMAT = 1                      # bump when the table layout or the way they are built changes
NLOC = 30                       # HCurl(2) DOFs per tet
NL2 = 10                        # L2(2) functions per tet
N_MASS, N_TOKEN = 46, 14        # points of the degree-8 mass rule and of the degree-4 token rule (2·order)
RULE_DEGREE = {"mass": 8, "token": 4}
PERMS = list(itertools.permutations(range(4)))
MONO3 = [(a, b, c) for a in range(4) for b in range(4 - a) for c in range(4 - a - b)]     # 20 monomials, degree ≤ 3
_REF_VERTICES = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)]       # NGSolve's reference tet
_MEMO = {}                      # table path → tables dict
_DEVICE_TABLES = {}             # (path, device) → torch tables
_FIT_TOL = 1e-9                 # x(ξ) cubic fit residual (relative to the element size) above which the record is refused


# ─────────────────────────── element classes, rules ───────────────────────────

def tets_of(mesh):
    """Vertex numbers (0-based) of the tets in the order of `ngmesh.Elements3D()`: int64 [nt, 4]."""
    return np.array([[v.nr - 1 for v in el.vertices] for el in mesh.ngmesh.Elements3D()], dtype=np.int64)


_CLASS_LUT = np.zeros(256, dtype=np.int64)
for _i, _p in enumerate(PERMS):
    _CLASS_LUT[_p[0] * 64 + _p[1] * 16 + _p[2] * 4 + _p[3]] = _i


def element_classes(tets):
    """Orientation class 0..23 of every tet = index (itertools.permutations(range(4)) order) of the rank
    permutation rk = argsort(argsort(vertex numbers)): the only thing the hierarchical HCurl / L2 bases of an
    element depend on.  tets [nt, 4] → int64 [nt]."""
    rk = np.argsort(np.argsort(tets, axis=1, kind="stable"), axis=1, kind="stable")
    return _CLASS_LUT[rk[:, 0] * 64 + rk[:, 1] * 16 + rk[:, 2] * 4 + rk[:, 3]]


def _rule(degree):
    """(points [nq, 3], weights [nq]) of NGSolve's TET rule of that degree."""
    from ngsolve import TET, IntegrationRule
    ir = IntegrationRule(TET, int(degree))
    return np.array([tuple(p) for p in ir.points])[:, :3], np.array(list(ir.weights))


def _monomials(ref):
    """Values [nq, 20] and reference gradients [nq, 20, 3] of the monomials ξ^a η^b ζ^c (a + b + c ≤ 3) at ref [nq, 3]."""
    V = np.stack([ref[:, 0] ** a * ref[:, 1] ** b * ref[:, 2] ** c for a, b, c in MONO3], 1)
    D = np.zeros((len(ref), len(MONO3), 3))
    for m, e in enumerate(MONO3):
        for j in range(3):
            if e[j]:
                ex = list(e)
                ex[j] -= 1
                D[:, m, j] = e[j] * ref[:, 0] ** ex[0] * ref[:, 1] ** ex[1] * ref[:, 2] ** ex[2]
    return V, D


# ─────────────────────────── reference tables (NGSolve only) ───────────────────────────

def _ngsolve_version():
    import ngsolve
    return str(ngsolve.__version__)


def cache_dir():
    """$RFCAV_CACHE or ~/.cache/rfcav."""
    return os.environ.get("RFCAV_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "rfcav")


def table_path():
    """Cache file of the reference tables: keyed by FORMAT and the ngsolve version (the hierarchical basis, the
    ConvertOperator and the point rules belong to the NGSolve build that produced them)."""
    ver = re.sub(r"[^0-9A-Za-z.]+", "_", _ngsolve_version())
    return os.path.join(cache_dir(), f"matfree_tables_f{FORMAT}_ngsolve_{ver}.npz")


def _synthetic_mesh(reps, seed, amp):
    """Mesh of 24·reps DISJOINT tets, `reps` of each orientation class (element = reference tet with its global
    vertex numbers the class' rank permutation).  amp > 0: curved (order 3) and deformed by a random cubic
    vector field of that amplitude (J varies per element and point), else the undeformed reference tets (J = I).
    Every face is the wall.  Returns the NGSolve mesh."""
    import netgen.meshing as nm
    from netgen.meshing import Element2D, Element3D, FaceDescriptor, MeshPoint, Pnt
    from ngsolve import GridFunction, Mesh, VectorH1
    ng = nm.Mesh()
    ng.dim = 3
    ng.Add(FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))
    ng.SetBCName(0, "wall")
    ng.SetMaterial(1, "mat")
    for p in PERMS:
        for _ in range(reps):
            pid = {}
            for j in np.argsort(p):             # points added in rank order: global number of vertex j ~ p[j]
                pid[int(j)] = ng.Add(MeshPoint(Pnt(*_REF_VERTICES[j])))
            ng.Add(Element3D(1, [pid[0], pid[1], pid[2], pid[3]]))
            for tri in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]):
                ng.Add(Element2D(1, [pid[t] for t in tri]))
    mesh = Mesh(ng)
    if amp > 0:
        mesh.Curve(3)
        gd = GridFunction(VectorH1(mesh, order=3))
        gd.vec.FV().NumPy()[:] = amp * np.random.default_rng(seed).standard_normal(gd.space.ndof)
        mesh.SetDeformation(gd)
    return mesh


def _jacobians_ngsolve(mesh, ir):
    from ngsolve import VOL, specialcf
    mips = mesh.MapToAllElements(ir, VOL)
    return np.asarray(specialcf.JacobianMatrix(3)(mips)).reshape(mesh.ne, -1, 3, 3), mips


def _local_conversion(mesh, ref46):
    """Per element of a synthetic (disjoint tets) mesh, everything the T fit needs: element dofs [nt, 30], J at the
    46 points [nt, 46, 3, 3], the L2 basis there (GridFunction of every local function) [nt, 46, 10] and the
    ConvertOperator block C_e [nt, 30 (dofs), 30 (c, l)] of the element's own HCurl DOFs from its VectorL2 ones."""
    from ngsolve import L2, VOL, ConvertOperator, ElementId, GridFunction, HCurl, IntegrationRule, TET, VectorL2
    from src.data_gen.highorder import _tosp
    nt = mesh.ne
    fes = HCurl(mesh, order=2, dirichlet="wall")
    sl, vl = L2(mesh, order=2), VectorL2(mesh, order=2)
    dofs = np.array([fes.GetDofNrs(ElementId(VOL, e)) for e in range(nt)])
    sd = np.array([sl.GetDofNrs(ElementId(VOL, e)) for e in range(nt)])
    colmap = (np.arange(3)[None, :, None] * sl.ndof + sd[:, None, :]).reshape(-1)         # (t, c, l) columns
    Cs = _tosp(ConvertOperator(vl, fes))[:, colmap].tocsr()
    if np.bincount(dofs.reshape(-1)).max() != 1:
        raise RuntimeError("synthetic mesh elements share DOFs")
    Cblk = np.stack([Cs[dofs[e]][:, e * NLOC:(e + 1) * NLOC].toarray() for e in range(nt)])
    J, mips = _jacobians_ngsolve(mesh, IntegrationRule(TET, 8))
    g = GridFunction(sl)
    Phi = np.zeros((nt, len(ref46), NL2))
    for l in range(NL2):
        g.vec[:] = 0
        g.vec.FV().NumPy()[sd[:, l]] = 1.0
        Phi[:, :, l] = np.asarray(g(mips)).reshape(nt, -1)
    return fes, sl, J, Phi, Cblk


def _build_tables(verbose=False):
    """The 24-class tables from NGSolve alone (no data, no assembled operator of any real mesh).  Stage 1 (undeformed
    reference tets, J = I): hierarchical HCurl shapes and their curls at the 46 mass points, L2 shapes at the 46
    and 14 points, per class.  Stage 2 (deformed copies): the dual functionals T[class, 30, 138] of the
    ConvertOperator acting on the pulled-back field Jᵀψ at the 46 points, from a least-squares fit over 14
    random cubic deformations per class, then checked on an independent mesh (another seed): the table is
    not returned if the relative error exceeds 1e-10.  The curls are computed at J = I, so the table is
    free of the 1e-14 noise a deformed element would put into detJ·J⁻¹·curl (kept as is, not rationalised)."""
    from ngsolve import VOL, ElementId
    t0 = time.perf_counter()
    ref46, w46 = _rule(8)
    ref14, w14 = _rule(4)
    if len(w46) != N_MASS or len(w14) != N_TOKEN or w46.min() <= 0:
        raise RuntimeError(f"unexpected NGSolve TET rules: {len(w46)} / {len(w14)} points, min weight {w46.min()}")
    # stage 1: reference shapes, curls, L2 tables
    mesh0 = _synthetic_mesh(1, 0, 0.0)
    from ngsolve import HCurl, L2
    fes0, sl0 = HCurl(mesh0, order=2, dirichlet="wall"), L2(mesh0, order=2)
    cls0 = element_classes(tets_of(mesh0))
    if sorted(cls0) != list(range(24)):
        raise RuntimeError("synthetic mesh does not hold the 24 classes once each")
    phi, crl = np.zeros((24, N_MASS, NLOC, 3)), np.zeros((24, N_MASS, NLOC, 3))
    L46, L14 = np.zeros((24, N_MASS, NL2)), np.zeros((24, N_TOKEN, NL2))
    for e in range(mesh0.ne):
        k = int(cls0[e])
        eid = ElementId(VOL, e)
        fel, fl2, tr = fes0.GetFE(eid), sl0.GetFE(eid), mesh0.GetTrafo(eid)
        for q, (x, y, z) in enumerate(ref46):
            mip = tr(x, y, z)
            if not np.allclose(np.array(mip.jacobi).reshape(3, 3), np.eye(3), atol=1e-14):
                raise RuntimeError("synthetic reference element is not the identity map")
            phi[k, q] = np.array(fel.CalcShape(x, y, z)).reshape(NLOC, 3)
            crl[k, q] = np.array(fel.CalcCurlShape(mip)).reshape(NLOC, 3)       # J = I: the reference curl
            L46[k, q] = np.array(fl2.CalcShape(x, y, z)).reshape(-1)
        for q, (x, y, z) in enumerate(ref14):
            L14[k, q] = np.array(fl2.CalcShape(x, y, z)).reshape(-1)
    # stage 2: dual functionals
    reps, T = 14, np.zeros((24, NLOC, N_MASS * 3))
    mesh = _synthetic_mesh(reps, 7, 0.06)
    _, sl, J, Phi, Cblk = _local_conversion(mesh, ref46)
    cls = element_classes(tets_of(mesh))
    if np.linalg.det(J).min() <= 0.05:
        raise RuntimeError("synthetic deformation folded an element")
    for k in range(24):
        idx = np.flatnonzero(cls == k)
        Gf = np.einsum("sqck,sql->sclqk", J[idx], Phi[idx]).reshape(len(idx) * NLOC, N_MASS * 3)
        Y = Cblk[idx].transpose(0, 2, 1).reshape(len(idx) * NLOC, NLOC)
        A = np.linalg.lstsq(Gf, Y, rcond=1e-11)[0]
        T[k] = A.T
        res = np.abs(Gf @ A - Y).max() / np.abs(Y).max()
        l2err = max(np.abs(Phi[idx[0]] - L46[k]).max(), 0.0)
        if res > 1e-10 or l2err > 1e-12:
            raise RuntimeError(f"class {k}: T fit residual {res:.1e}, L2 table vs GridFunction {l2err:.1e}")
    # independent check of T, the shapes and the curls on a mesh with another deformation
    err = _check_tables(dict(phi=phi, crl=crl, T=T, ref46=ref46, w46=w46), seed=123)
    if err > 1e-10:
        raise RuntimeError(f"reference tables fail the independent check: relative error {err:.1e}")
    if verbose:
        print(f"matfree tables built in {time.perf_counter() - t0:.1f} s (independent check {err:.1e})")
    return dict(phi=phi, crl=crl, L46=L46, L14=L14, T=T, ref46=ref46, w46=w46, ref14=ref14, w14=w14)


def _check_tables(tab, seed):
    """max relative error of the tables against NGSolve on a fresh deformed synthetic mesh (2 elements per class):
    the local conversion block C_e = T·(Jᵀ ψ) and the mapped element mass / curl-curl matrices."""
    from ngsolve import HCurl, TET, BilinearForm, IntegrationRule, curl, dx
    mesh = _synthetic_mesh(2, seed, 0.06)
    ref46, w46 = tab["ref46"], tab["w46"]
    fes, _, J, Phi, Cblk = _local_conversion(mesh, ref46)
    cls = element_classes(tets_of(mesh))
    err = 0.0
    G = np.einsum("tqck,tql->tclqk", J, Phi).reshape(mesh.ne, NLOC, N_MASS * 3)
    for e in range(mesh.ne):
        ref = Cblk[e]                                                 # [30 dofs, 30 (c, l)]
        mine = tab["T"][cls[e]] @ G[e].T
        err = max(err, np.abs(mine - ref).max() / np.abs(ref).max())
    # element matrices from the tables vs the assembled (explicit degree-8 rule) global matrices of the disjoint mesh
    fes = HCurl(mesh, order=2)
    u, v = fes.TnT()
    rule = dx(intrules={TET: IntegrationRule(TET, 8)})
    Ks = BilinearForm(curl(u) * curl(v) * rule).Assemble()
    Ms = BilinearForm(u * v * rule).Assemble()
    from ngsolve import VOL, ElementId
    from src.data_gen.highorder import _tosp
    Km, Mm = _tosp(Ks.mat), _tosp(Ms.mat)
    dofs = np.array([fes.GetDofNrs(ElementId(VOL, e)) for e in range(mesh.ne)])
    dJ = np.linalg.det(J)
    Ji = np.linalg.inv(J)
    for e in range(mesh.ne):
        ph, cr = tab["phi"][cls[e]], tab["crl"][cls[e]]
        um = np.einsum("qki,qlk->qli", Ji[e], ph)
        cm = np.einsum("qik,qlk->qli", J[e], cr) / dJ[e][:, None, None]
        w = np.abs(dJ[e]) * w46
        Me = np.einsum("q,qli,qmi->lm", w, um, um)
        Ke = np.einsum("q,qli,qmi->lm", w, cm, cm)
        d = dofs[e]
        for mine, glob in ((Me, Mm), (Ke, Km)):
            ref = glob[d][:, d].toarray()
            err = max(err, np.abs(mine - ref).max() / np.abs(ref).max())
    return err


def _derive(tab):
    """Add the monomial tables (functions of the rule points only) to a tables dict."""
    out = dict(tab)
    out["mono46"], out["dmono46"] = _monomials(tab["ref46"])
    out["mono14"], out["dmono14"] = _monomials(tab["ref14"])
    return out


_TABLE_KEYS = ("phi", "crl", "L46", "L14", "T", "ref46", "w46", "ref14", "w14")


def reference_tables(order=2, rebuild=False, verbose=False):
    """The 24-class reference tables (numpy dict): phi, crl [24, 46, 30, 3] (reference shapes / curls at the mass
    rule), L46 [24, 46, 10] / L14 [24, 14, 10] (L2(2) shapes at the mass / token points), T [24, 30, 138] (dual
    functionals of the ConvertOperator on the pulled-back field Jᵀψ at the 46 points, columns (point, component)),
    ref46 / w46, ref14 / w14 (the TET rules of degree 8 and 4), plus the monomial tables mono46, dmono46, mono14,
    dmono14.  Built from NGSolve alone on first use (a few seconds) and cached in `cache_dir()` under a name
    keyed by FORMAT and the ngsolve version (atomic write); rebuild=True ignores the cache."""
    if int(order) != 2:
        raise NotImplementedError("matfree: HCurl order 2 only")
    path = table_path()
    if not rebuild and path in _MEMO:
        return _MEMO[path]
    tab = None
    if not rebuild and os.path.exists(path):
        try:
            with np.load(path) as z:
                if int(z["format"]) == FORMAT and str(z["ngsolve_version"]) == _ngsolve_version():
                    tab = {k: np.array(z[k]) for k in _TABLE_KEYS}
        except (OSError, ValueError, KeyError):
            tab = None                                  # unreadable / truncated: rebuilt below
    if tab is None:
        tab = _build_tables(verbose)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp.npz")
        os.close(fd)
        try:
            np.savez(tmp, format=FORMAT, ngsolve_version=_ngsolve_version(), **tab)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    _MEMO[path] = _derive(tab)
    return _MEMO[path]


def _tables(device):
    """Torch (float64) copy of the tables on `device`."""
    device = torch.device("cpu" if device is None else device)
    np_tab = reference_tables()
    key = (table_path(), str(device))
    if key not in _DEVICE_TABLES:
        _DEVICE_TABLES[key] = {k: torch.as_tensor(v, device=device) for k, v in np_tab.items()
                               if k not in ("ref46", "ref14")} | {"ref46": np_tab["ref46"], "ref14": np_tab["ref14"]}
    return _DEVICE_TABLES[key]


# ─────────────────────────── the record ───────────────────────────

def build_record(mesh, order=2, scale=None, center=None):
    """Compact record of one curved (and deformed) NGSolve mesh, numpy; assembles no global matrix.
    Keys:
        dofs [nt, 30] int32     HCurl(2) DOFs of every element (fes = HCurl(mesh, order=2, dirichlet="wall"))
        cls  [nt] uint8         orientation class 0..23 (element_classes)
        xc   [nt, 20, 3] f64    x(ξ) − x0 = Σ_m xc[m]·mono_m(ξ): the element map, an exact cubic for geometry
                                order ≤ 3 (fitted at the 46 mass points; a larger residual raises ValueError).  float64:
                                float32 loses the digits the Jacobians need
        x0   [nt, 3] f64        element offset (the mean of x over the 46 points; J does not see it)
        bnd  [n_dof] bool       wall DOFs
        n_dof, n_tets, scale, center, order, format
    center / scale as in highorder.field_operators: None = vertex mean and max |x − center| of the mesh; the dataset
    passes field_labels.model_normalisation(mesh).  ≈ 640 B per tet.  Meshes that went through
    highorder.single_level and bisected ones without it (unused DOFs: n_d = 0 there, handled) both work."""
    from ngsolve import CF, VOL, ElementId, HCurl, x, y, z
    if int(order) != 2:
        raise NotImplementedError("matfree: HCurl order 2 only")
    tab = reference_tables()
    nt = mesh.ne
    fes = HCurl(mesh, order=2, dirichlet="wall")
    Xv = np.array([tuple(p.p) for p in mesh.ngmesh.Points()], dtype=np.float64)
    tets = tets_of(mesh)
    if tets.size == 0 or tets.max() >= len(Xv) or tets.min() < 0:
        raise ValueError("matfree: tets refer to vertices outside the mesh")
    cen = Xv.mean(0) if center is None else np.asarray(center, dtype=np.float64)
    scl = float(np.max(np.abs(Xv - cen))) if scale is None else float(scale)
    from ngsolve import IntegrationRule, TET
    ir = IntegrationRule(TET, RULE_DEGREE["mass"])
    mips = mesh.MapToAllElements(ir, VOL)
    qx = np.asarray(CF((x, y, z))(mips)).reshape(nt, N_MASS, 3)
    x0 = qx.mean(axis=1)
    rhs = qx - x0[:, None, :]
    xc = np.einsum("mq,tqk->tmk", np.linalg.pinv(tab["mono46"]), rhs)
    fit = np.abs(np.einsum("qm,tmk->tqk", tab["mono46"], xc) - rhs).max(axis=(1, 2))
    size = np.abs(rhs).max(axis=(1, 2))
    if (fit > _FIT_TOL * size).any():
        raise ValueError(f"matfree: the element map is not a cubic polynomial (max relative fit residual "
                         f"{(fit / size).max():.1e}): geometry order > 3?")
    dofs = np.array([fes.GetDofNrs(ElementId(VOL, e)) for e in range(nt)], dtype=np.int32).reshape(nt, NLOC)
    bnd = ~np.asarray(list(fes.FreeDofs()), dtype=bool)
    return {"dofs": dofs, "cls": element_classes(tets).astype(np.uint8), "xc": xc, "x0": x0, "bnd": bnd,
            "n_dof": int(fes.ndof), "n_tets": int(nt), "scale": scl, "center": cen, "order": 2, "format": FORMAT}


def record_nbytes(rec):
    """Bytes of the array part of a numpy record."""
    return sum(v.nbytes for v in rec.values() if isinstance(v, np.ndarray))


def to_torch(rec, device=None):
    """The record as torch tensors on `device` (default: where it is; cpu for a numpy record): dofs int64, cls int64,
    xc / x0 float64, bnd bool, n_d [n_dof] float64 (how many elements share each DOF; 0 for unused DOFs) and the
    scalars.  An already converted record is returned as it is, or moved."""
    if rec.get("_torch"):
        if device is None or rec["xc"].device == torch.device(device):
            return rec
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in rec.items()}
    device = torch.device("cpu" if device is None else device)
    cv = lambda a, dt: torch.as_tensor(np.asarray(a), dtype=dt).to(device)   # noqa: E731
    dofs = cv(rec["dofs"], torch.int64)
    return {"dofs": dofs, "cls": cv(rec["cls"], torch.int64), "xc": cv(rec["xc"], torch.float64),
            "x0": cv(rec["x0"], torch.float64), "bnd": cv(rec["bnd"], torch.bool),
            "n_d": torch.bincount(dofs.reshape(-1), minlength=int(rec["n_dof"])).to(torch.float64),
            "n_dof": int(rec["n_dof"]), "n_tets": int(rec["n_tets"]), "scale": float(rec["scale"]),
            "center": np.asarray(rec["center"], dtype=np.float64), "_torch": True}


def _chunks(n, chunk):
    step = n if not chunk else int(chunk)
    return [slice(s, min(n, s + step)) for s in range(0, max(n, 1), max(step, 1))]


# ─────────────────────────── geometry ───────────────────────────

def _det_inv(J):
    """det and inverse of J [..., 3, 3] (cofactors; no LU, no sync on a GPU)."""
    a, b, c = J[..., 0, 0], J[..., 0, 1], J[..., 0, 2]
    d, e, f = J[..., 1, 0], J[..., 1, 1], J[..., 1, 2]
    g, h, i = J[..., 2, 0], J[..., 2, 1], J[..., 2, 2]
    c00, c01, c02 = e * i - f * h, f * g - d * i, d * h - e * g
    det = a * c00 + b * c01 + c * c02
    adj = torch.stack([torch.stack([c00, c * h - b * i, b * f - c * e], -1),
                       torch.stack([c01, a * i - c * g, c * d - a * f], -1),
                       torch.stack([c02, b * g - a * h, a * e - b * d], -1)], -2)
    return det, adj / det[..., None, None]


def jacobians(rec, rule="mass"):
    """J [nt, nq, 3, 3], J[t, q, i, j] = ∂x_i/∂ξ_j (physical units, NOT normalised) at the 46 points of the degree-8
    mass rule (rule "mass") or the 14 token points (rule "token"), from the cubic element map of the record."""
    rec = to_torch(rec)
    D = _tables(rec["xc"].device)["dmono46" if rule == "mass" else "dmono14"]
    return torch.einsum("tmi,qmj->tqij", rec["xc"], D)


def token_points(rec):
    """Token (quadrature) points [nt, 14, 3] of the normalised geometry ξ = (x − center)/scale: ops['qx']."""
    rec = to_torch(rec)
    mono = _tables(rec["xc"].device)["mono14"]
    x = rec["x0"][:, None, :] + torch.einsum("qm,tmk->tqk", mono, rec["xc"])
    return (x - torch.as_tensor(rec["center"], device=x.device)) / rec["scale"]


def _token_weights_raw(rec, J=None):
    tab = _tables(rec["xc"].device)
    J = jacobians(rec, "token") if J is None else J
    return torch.linalg.det(J).abs() * tab["w14"][None, :]


def token_weights(rec):
    """Token weights [nt, 14] = |det J|·w / scale³: ops['qw'] (the attention's mass weights)."""
    rec = to_torch(rec)
    return _token_weights_raw(rec) / rec["scale"] ** 3


def _token_P(cls, J14):
    """P [b, 10, 14]: point values at the 14 token points → L2(2) coefficients, the |J|-weighted least-squares
    projection (ΦᵀWΦ)⁻¹ΦᵀW (what field_operators stores as ops['P']); Φ = L14[class]."""
    tab = _tables(cls.device)
    W = torch.linalg.det(J14).abs() * tab["w14"][None, :]
    Phi = tab["L14"][cls]
    A = torch.einsum("tqi,tq,tqj->tij", Phi, W, Phi)
    return torch.linalg.solve(A, torch.einsum("tqi,tq->tiq", Phi, W))


# ─────────────────────────── element operators ───────────────────────────

def _element_mats_chunk(xc, cls, tab, scale):
    b = xc.shape[0]
    J = torch.einsum("tmi,qmj->tqij", xc, tab["dmono46"])
    det, Ji = _det_inv(J)
    phi, crl = tab["phi"][cls], tab["crl"][cls]                                  # [b, 46, 30, 3]
    w = det.abs() * tab["w46"][None, :]
    sw = w.sqrt()[..., None, None]
    um = torch.einsum("tqki,tqlk->tqli", Ji, phi) * sw                           # J⁻ᵀ φ_ref
    cm = torch.einsum("tqik,tqlk->tqli", J, crl) / det[..., None, None] * sw     # J curl_ref / det J
    um = um.permute(0, 2, 1, 3).reshape(b, NLOC, -1)
    cm = cm.permute(0, 2, 1, 3).reshape(b, NLOC, -1)
    Me, Ke = torch.bmm(um, um.transpose(1, 2)), torch.bmm(cm, cm.transpose(1, 2))
    return 0.5 * (Me + Me.transpose(1, 2)) / scale, 0.5 * (Ke + Ke.transpose(1, 2)) * scale


def element_matrices(rec, chunk=1024):
    """(Me, Ke) [nt, 30, 30] float64: element mass / curl-curl matrices of the model's operators, including the
    ξ-scaling, so that  M = Σ_e scatter(Me),  K = Σ_e scatter(Ke)  are `field_operators`' M and K:
    Me = (1/s)·Σ_q w_q |detJ| u_l·u_m with u = J⁻ᵀ φ_ref, Ke = s·Σ_q w_q |detJ| c_l·c_m with c = J·curl_ref/detJ,
    on the degree-8 rule.  chunk: elements per step (transient memory ≈ chunk·0.12 MB)."""
    rec = to_torch(rec)
    tab = _tables(rec["xc"].device)
    nt, s = rec["n_tets"], rec["scale"]
    Me = torch.empty(nt, NLOC, NLOC, dtype=torch.float64, device=rec["xc"].device)
    Ke = torch.empty_like(Me)
    for sl in _chunks(nt, chunk):
        Me[sl], Ke[sl] = _element_mats_chunk(rec["xc"][sl], rec["cls"][sl], tab, s)
    return Me, Ke


def apply(Ae, dofs, V, chunk=None):
    """A·V [n, m] for A = Σ_e scatter(Ae): gather V on the element DOFs, batched product, index_add.  Ae [nt, 30, 30],
    dofs [nt, 30] (int64), V [n, m].  Differentiable w.r.t. V (and Ae)."""
    V = V.to(torch.float64)
    out = torch.zeros_like(V)
    for sl in _chunks(dofs.shape[0], chunk):
        Y = torch.bmm(Ae[sl], V[dofs[sl]])
        out = out.index_add(0, dofs[sl].reshape(-1), Y.reshape(-1, V.shape[1]))
    return out


def gram(Ae, dofs, V, T=None, chunk=None):
    """VᵀAV [m, m] (symmetrised), or with T [n, k] the block Gram of [V | T]: [[VᵀAV, VᵀAT], [TᵀAV, TᵀAT]]
    [m + k, m + k] (symmetrised) — element sums Σ_e V_eᵀ A_e V_e, no scatter.  Differentiable w.r.t. V."""
    W = V.to(torch.float64) if T is None else torch.cat([V.to(torch.float64), T.to(torch.float64)], dim=1)
    G = None
    for sl in _chunks(dofs.shape[0], chunk):
        We = W[dofs[sl]]
        g = We.reshape(-1, W.shape[1]).T @ torch.bmm(Ae[sl], We).reshape(-1, W.shape[1])
        G = g if G is None else G + g
    return 0.5 * (G + G.T)


def grams_plain(rec, V, T, mats=None, chunk=None):
    """Mass / curl-curl Grams of [V | T] without the kernel projection, for ONE geometry: what
    hcurl.hcurl_grams_plain(V, T, batch) returns, minus the batch axis.  rec: the record (numpy or torch);
    V [Nd, m], T [Nd, n] (rows ≥ n_dof are padding and ignored; T's padded rows give zero rows of MT);
    mats: (Me, Ke) from element_matrices (computed if None).  Returns
        G_M, G_A  [m + n, m + n]  float64: blocks [[VᵀAV, VᵀAT], [TᵀAV, TᵀAT]] for A = M and A = K (the VV blocks
                  symmetrised, as hcurl_grams_plain does)
        MT        [Nd, n]         M·T
    A batch is the list of records with V [B, Nd, m], T [B, Nd, n]: the results are stacked on a new first axis."""
    if isinstance(rec, (list, tuple)):
        mats = mats or [None] * len(rec)
        res = [grams_plain(r, V[b], T[b], mats[b], chunk) for b, r in enumerate(rec)]
        return tuple(torch.stack(x) for x in zip(*res, strict=True))
    rec = to_torch(rec)
    Me, Ke = element_matrices(rec) if mats is None else mats
    n = rec["n_dof"]
    Nd = T.shape[0]
    V, T = V.to(torch.float64)[:n], T.to(torch.float64)[:n]
    MT, KT = apply(Me, rec["dofs"], T, chunk), apply(Ke, rec["dofs"], T, chunk)
    M_V, A_V = gram(Me, rec["dofs"], V, chunk=chunk), gram(Ke, rec["dofs"], V, chunk=chunk)

    def block(VV, A_T):
        return torch.cat([torch.cat([VV, V.T @ A_T], -1), torch.cat([A_T.T @ V, T.T @ A_T], -1)], -2)
    pad = lambda X: torch.nn.functional.pad(X, (0, 0, 0, Nd - n))   # noqa: E731
    return block(M_V, MT), block(A_V, KT), pad(MT)


# ─────────────────────────── ψ → DOFs ───────────────────────────

def _psi_map_chunk(xc, cls, tab, n_d_e):
    J = torch.einsum("tmi,qmj->tqij", xc, tab["dmono46"])
    J14 = torch.einsum("tmi,qmj->tqij", xc, tab["dmono14"])
    P = _token_P(cls, J14)                                           # [b, 10, 14]
    Tt = tab["T"][cls].reshape(xc.shape[0], NLOC, N_MASS, 3)
    R = torch.einsum("tdqk,tqck->tdqc", Tt, J)                                   # functionals applied to Jᵀ ψ46
    S = torch.einsum("tdqc,tql->tdcl", R, tab["L46"][cls])                       # ψ46 = L46 · a (L2 coefficients a)
    return torch.einsum("tdcl,tlq->tdqc", S, P) / n_d_e[:, :, None, None]        # a = P ψ14, 1/n_d averaging


def psi_map(rec, chunk=1024):
    """Ce [nt, 30, 14, 3] float64: the whole DOF map of one element, including the 1/n_d averaging of shared DOFs:
        u = Σ_e scatter(Ce · ψ_e),  u_d = Σ_{(t,q,c)} Ce[t, d, q, c] ψ[t, q, c]
    i.e. C·vec_{t,c,l}(Pψ) of field_operators.  Built from the cubic element map: P (|J|-weighted least squares at
    the 14 token points), the L2 basis at the 46 points, the pull-back Jᵀψ and the class' dual functionals T."""
    rec = to_torch(rec)
    tab = _tables(rec["xc"].device)
    nt = rec["n_tets"]
    Ce = torch.empty(nt, NLOC, N_TOKEN, 3, dtype=torch.float64, device=rec["xc"].device)
    for sl in _chunks(nt, chunk):
        Ce[sl] = _psi_map_chunk(rec["xc"][sl], rec["cls"][sl], tab, rec["n_d"][rec["dofs"][sl]])
    return Ce


def psi_to_dofs(rec, psi, Ce=None, chunk=1024):
    """u = C·vec(Pψ) [n_dof, m] float64 for ψ [nt, 14, m, 3] (the model's psi of one geometry, layout of
    EigenspaceOperatorField.field_basis: element, token point, basis vector, component).  This is the model's
    `spmm(C, coef)` BEFORE the DofMask / BndDof masking (wall DOFs are not zeroed here).  Ce: psi_map(rec) when
    reused (a constant of the geometry); computed per chunk otherwise.  Differentiable w.r.t. ψ."""
    rec = to_torch(rec)
    psi = psi.to(torch.float64)
    tab = _tables(rec["xc"].device)
    m = psi.shape[2]
    out = torch.zeros(rec["n_dof"], m, dtype=torch.float64, device=rec["xc"].device)
    for sl in _chunks(rec["n_tets"], chunk):
        C = Ce[sl] if Ce is not None else _psi_map_chunk(rec["xc"][sl], rec["cls"][sl], tab,
                                                         rec["n_d"][rec["dofs"][sl]])
        u = torch.einsum("tdqc,tqmc->tdm", C, psi[sl])
        out = out.index_add(0, rec["dofs"][sl].reshape(-1), u.reshape(-1, m))
    return out


# ─────────────────────────── verification helper ───────────────────────────

def check_against_ngsolve(mesh, ops=None, m=8, seed=0, scale=None, center=None):
    """Errors of this module against highorder.field_operators(mesh, 2, potential=False) on random V, ψ (debug /
    CI helper; needs ngsolve): max-norm relative errors of K·V, M·V, VᵀKV, VᵀMV, u = C·vec(Pψ), qx, qw, P and the
    grams_plain blocks.  Returns a dict of floats and the class count."""
    from src.data_gen.highorder import field_operators
    from src.models.hcurl import hcurl_grams_plain
    rec = build_record(mesh, scale=scale, center=center)
    if ops is None:
        ops = field_operators(mesh, 2, scale=rec["scale"], center=rec["center"], potential=False)
    tr = to_torch(rec)
    rng = np.random.default_rng(seed)
    nt, n = rec["n_tets"], rec["n_dof"]
    V = rng.standard_normal((n, m))
    Tt = rng.standard_normal((n, 3))
    psi = rng.standard_normal((nt, N_TOKEN, m, 3))
    rel = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(b).max(), 1e-300))   # noqa: E731
    Me, Ke = element_matrices(tr)
    Vt = torch.from_numpy(V)
    out = {"classes": int(len(np.unique(rec["cls"])))}
    out["K_apply"] = rel(apply(Ke, tr["dofs"], Vt).numpy(), ops["K"] @ V)
    out["M_apply"] = rel(apply(Me, tr["dofs"], Vt).numpy(), ops["M"] @ V)
    out["K_gram"] = rel(gram(Ke, tr["dofs"], Vt).numpy(), V.T @ (ops["K"] @ V))
    out["M_gram"] = rel(gram(Me, tr["dofs"], Vt).numpy(), V.T @ (ops["M"] @ V))
    coef = np.einsum("tlq,tqmc->tclm", ops["P"], psi)
    out["psi_to_dofs"] = rel(psi_to_dofs(tr, torch.from_numpy(psi)).numpy(), ops["C"] @ coef.reshape(-1, m))
    out["qx"] = rel(token_points(tr).numpy(), ops["qx"])
    out["qw"] = rel(token_weights(tr).numpy(), ops["qw"])
    batch = {"M": _to_sparse(ops["M"]), "K": _to_sparse(ops["K"]), "SymmetricMK": True}
    GM, GA, MT = grams_plain(tr, Vt, torch.from_numpy(Tt), (Me, Ke))
    rGM, rGA, rMT = hcurl_grams_plain(Vt[None], torch.from_numpy(Tt)[None], batch)
    out["grams_plain"] = max(rel(GM.numpy(), rGM[0].numpy()), rel(GA.numpy(), rGA[0].numpy()), rel(MT.numpy(), rMT[0].numpy()))
    return out


def _to_sparse(A):
    import warnings
    A = A.tocsr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # torch: "sparse CSR support is in beta"
        return torch.sparse_csr_tensor(torch.from_numpy(A.indptr.astype(np.int64)),
                                       torch.from_numpy(A.indices.astype(np.int64)), torch.from_numpy(A.data), A.shape)
