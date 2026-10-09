"""High-order reference eigenmodes of a PEC cavity with NGSolve: curved tets + Nédélec order p.

    curl curl E = k² E in Ω,  n × E = 0 on ∂Ω (all of ∂Ω is the face set "wall")

Recipe (docs/18 E7, measured on the pillbox: p = 3, curve 3, 3.4k DOF → TM010 −9e-6):

- geometry: the gmsh OCC model of a generator family, written as BREP and read by netgen
  (`netgen_mesh`).  Revolved faces with a seam (every axisymmetric family) can make netgen's
  surface mesher fail ("boundary mesh is overlapping"); cutting the solid into four quarters about
  the z axis (in gmsh: netgen's own OCC booleans crash on some of these solids) fixes it without
  changing the geometry — the cut faces become an internal "interface".  mesh.Curve(curve) maps the
  tets onto the exact CAD surfaces.
- space: the FULL HCurl space of order p with its own discrete gradient (fes.CreateGradient, H1 of
  order p + 1).  nograds=True with a P1-gradient projection gives ~2 % errors for p ≥ 2.
- eigensolver (static condensation of the element-interior DOFs): shift-invert with σ < 0 (K − σM SPD) and the M-orthogonal projection
  P = I − G Kp⁻¹ GᵀM, Kp = ∇·∇ on the H1 space, after every solve (same as the N0 generator).
  Both factorisations are NGSolve's multithreaded sparse Cholesky; ARPACK runs on numpy views.
  Isolated conductors (more than one wall shell) leave one spurious λ ≈ 0 mode per extra shell
  (H1 with p = 0 on every shell misses their potentials); those are filtered out.

Wall quantities are evaluated on the boundary elements from the adjacent volume element.
"""
import time

import numpy as np
from scipy.constants import epsilon_0 as EPS0, mu_0 as MU0
from scipy.sparse.linalg import LinearOperator, eigsh

C0 = 299792458.0


# ─────────────────────────── geometry ──────────────────────────

def quarter_cad(src, dst):
    """Write `src`'s solid cut into its four quarters about the z axis (gmsh OCC fragment: shared,
    conforming cut faces) to `dst`.  Same geometry; the revolved faces lose their seam."""
    import gmsh
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("highorder_quarter")
    try:
        occ = gmsh.model.occ
        v = occ.importShapes(str(src))
        occ.synchronize()
        bb = gmsh.model.getBoundingBox(-1, -1)
        big = 2.0 * max(abs(c) for c in bb) + 1e-3
        z0, dz = bb[2] - big, bb[5] - bb[2] + 2 * big
        boxes = [occ.addBox(0.0 if sx > 0 else -big, 0.0 if sy > 0 else -big, z0, big, big, dz)
                 for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        out, omap = occ.fragment(v, [(3, b) for b in boxes])
        occ.remove([d for d in out if d not in omap[0]], recursive=True)
        occ.synchronize()
        gmsh.write(str(dst))
    finally:
        gmsh.model.remove()
        if own:
            gmsh.finalize()


def _name_faces(shape):
    """Faces shared by two solids → "interface", every other face → "wall".  Faces are matched
    geometrically (centre of mass + area): OCC's face hash ignores the location, so the rotated
    copies the quartering produces (e.g. the four quarter end caps) would collide."""
    lo, hi = (np.asarray(tuple(v)) for v in shape.bounding_box)
    size = float(np.max(hi - lo))
    count, faces = {}, []
    for s in shape.solids:
        for f in s.faces:
            c = f.center
            key = tuple(np.round(np.r_[c.x, c.y, c.z] / size, 7)) + (round(f.mass / size ** 2, 7),)
            count[key] = count.get(key, 0) + 1
            faces.append((key, f))
    for key, f in faces:
        f.name = "interface" if count[key] > 1 else "wall"


def netgen_mesh(cad_path, maxh, curve=3, deform=None, split="auto", retries=2, min_jacobian_ratio=0.25,
                **meshing):
    """NGSolve mesh of the solid in `cad_path` (BREP / STEP, metres), curved to order `curve`; the
    cavity surface is the boundary "wall" (internal cut faces: "interface").  deform: optional
    (om [n,3], a [n,3], ph [n]) of the generator's smooth map x → x + Σ_m a_m sin(om_m·x + ph_m)
    (dataset_generator_3d.deform_map), applied exactly at geometry order `curve`.
    split: True / False / "auto" (plain first, quartered with quarter_cad on failure).
    meshing: netgen GenerateMesh options (curvaturesafety, default 2: element size on curved faces
    ~ radius / curvaturesafety, whatever maxh is).  Curving coarse tets onto small fillets can fold
    them (det J < 0) or nearly so: surface fields and wall loss are then garbage (pillbox with pipes:
    Q0 off 10× at a ratio of 0.05, peaks 3× at 0.07, fine at ≥ 0.4) although the frequencies hardly
    change.  Every curved mesh is checked (jacobian_ratio > min_jacobian_ratio); a failing one is
    regenerated with curvaturesafety doubled, up to `retries` times."""
    import os
    import tempfile

    from netgen.occ import OCCGeometry
    from ngsolve import Mesh

    tries = {"auto": (False, True), True: (True,), False: (False,)}[split]
    err, mesh = None, None
    with tempfile.TemporaryDirectory() as tmp:
        for quarter in tries:
            path = cad_path
            if quarter:
                path = os.path.join(tmp, "quartered.brep")
                quarter_cad(cad_path, path)
            shape = OCCGeometry(path).shape
            if not quarter:
                if len(shape.solids) != 1:
                    raise ValueError(f"{cad_path}: expected one solid, got {len(shape.solids)}")
                shape = shape.solids[0]           # gmsh also writes free faces (the revolved profile)
            _name_faces(shape)
            try:
                mesh = Mesh(OCCGeometry(shape).GenerateMesh(maxh=maxh, **meshing))
                break
            except Exception as e:                # netgen raises NgException on a failed surface mesh
                err = e
    if mesh is None:
        raise RuntimeError(f"netgen could not mesh {cad_path}: {err}")
    if not set(mesh.GetBoundaries()) <= {"wall", "interface"}:
        raise RuntimeError(f"unexpected boundary names {set(mesh.GetBoundaries())}")
    mesh.Curve(int(curve))
    if deform is not None:
        set_deformation(mesh, *deform, order=int(curve))
    q = jacobian_ratio(mesh)
    if q <= min_jacobian_ratio:
        cs = float(meshing.get("curvaturesafety", 2.0))
        if retries > 0:                           # coarse tets bent onto small fillets fold: refine there
            return netgen_mesh(cad_path, maxh, curve, deform, split, retries=retries - 1,
                               min_jacobian_ratio=min_jacobian_ratio, **{**meshing, "curvaturesafety": 2 * cs})
        raise RuntimeError(f"curved mesh of {cad_path} has folded elements (min det J ratio {q:.2f})")
    return mesh


def jacobian_ratio(mesh, order=6):
    """min over elements of min/max det J at the integration points (≤ 0: a folded curved element)."""
    from ngsolve import TET, VOL, Det, IntegrationRule, specialcf
    mips = mesh.MapToAllElements(IntegrationRule(TET, order), VOL)
    J = np.asarray(Det(specialcf.JacobianMatrix(3))(mips)).reshape(mesh.ne, -1)
    return float((J.min(1) / J.max(1)).min())


def deformation_cf(om, a, ph):
    """CoefficientFunction δ(x) = Σ_m a_m sin(om_m·x + ph_m) (generator smooth_deform)."""
    from ngsolve import CF, sin, x, y, z
    om, a, ph = (np.asarray(v, dtype=np.float64) for v in (om, a, ph))
    s = [sin(float(o[0]) * x + float(o[1]) * y + float(o[2]) * z + float(p)) for o, p in zip(om, ph)]
    return CF(tuple(sum(float(a[m, c]) * s[m] for m in range(len(s))) for c in range(3)))


def set_deformation(mesh, om, a, ph, order=3):
    from ngsolve import GridFunction, VectorH1
    gfd = GridFunction(VectorH1(mesh, order=order))
    gfd.Set(deformation_cf(om, a, ph))
    mesh.SetDeformation(gfd)
    return gfd


# ─────────────────────────── eigensolver ───────────────────────

def _diag_mean(mat, free):
    r, c, v = mat.COO()
    r, c, v = np.asarray(r), np.asarray(c), np.asarray(v)
    d = np.zeros(mat.height)
    on = r == c
    np.add.at(d, r[on], v[on])
    return float(d[free].mean())


def _condensed_inverse(form, fes):
    """Exact inverse of a statically condensed SPD form: the element-interior DOFs are eliminated
    element by element and only the coupling DOFs are factorised (sparse Cholesky) — for p ≥ 3
    most DOFs are interior, so fill and time drop by an order of magnitude."""
    from ngsolve import IdentityMatrix
    inv = form.mat.Inverse(fes.FreeDofs(coupling=True), inverse="sparsecholesky")
    ext = IdentityMatrix() + form.harmonic_extension
    ext_t = IdentityMatrix() + form.harmonic_extension_trans
    return ext @ inv @ ext_t + form.inner_solve


def solve_modes(mesh, k, order=3, tol=1e-10, sigma=None, n_extra=4, max_ndof=None):
    """k lowest physical eigenpairs.  Returns dict(lam [k] (k² in 1/m²), f_hz [k], gfs (one
    GridFunction per mode, unit L2 norm), fes, info).  max_ndof: raise before assembling a larger
    space (memory guard: ~1 GB per 70k DOF at p = 4)."""
    from ngsolve import BilinearForm, GridFunction, HCurl, TaskManager, curl, dx, grad

    t0 = time.perf_counter()
    fes = HCurl(mesh, order=order, dirichlet="wall")
    if max_ndof is not None and fes.ndof > max_ndof:
        raise RuntimeError(f"HCurl p={order} on {mesh.ne} curved tets has {fes.ndof} DOF > max_ndof={max_ndof}")
    u, v = fes.TnT()
    free = np.asarray(list(fes.FreeDofs()), dtype=bool)
    with TaskManager():
        K = BilinearForm(curl(u) * curl(v) * dx, symmetric=True).Assemble()
        M = BilinearForm(u * v * dx, symmetric=True).Assemble()
        G, fesh1 = fes.CreateGradient()
        p, q = fesh1.TnT()
        Kp = BilinearForm(grad(p) * grad(q) * dx, symmetric=True, condense=True).Assemble()
    lam_scale = _diag_mean(K.mat, free) / _diag_mean(M.mat, free)
    if sigma is None:
        sigma = -1e-2 * lam_scale
    with TaskManager():
        A = BilinearForm((curl(u) * curl(v) - sigma * u * v) * dx, symmetric=True, condense=True).Assemble()
    t1 = time.perf_counter()
    with TaskManager():
        Ainv = _condensed_inverse(A, fes)
        Kpinv = _condensed_inverse(Kp, fesh1)
    t2 = time.perf_counter()

    xv, yv = K.mat.CreateColVector(), K.mat.CreateColVector()
    hv, hw = Kp.mat.CreateColVector(), Kp.mat.CreateColVector()
    Gt = G.T
    xn, yn = xv.FV().NumPy(), yv.FV().NumPy()

    def project(vec):                                  # vec ← (I − G Kp⁻¹ GᵀM) vec, in place
        yv.data = M.mat * vec
        hv.data = Gt * yv
        hw.data = Kpinv * hv
        vec.data -= G * hw

    def opinv(b):
        xn[:] = b
        with TaskManager():
            yv.data = Ainv * xv
            xv.data = yv
            project(xv)
        return xn.copy()

    def mmul(b):
        xn[:] = b
        yv.data = M.mat * xv
        return yn.copy()

    def kmul(b):
        xn[:] = b
        yv.data = K.mat * xv
        return yn.copy()

    n = fes.ndof
    rng = np.random.default_rng(0)
    xn[:] = rng.standard_normal(n) * free
    project(xv)
    v0 = xn.copy()
    nev = k + n_extra
    vals, vecs = eigsh(LinearOperator((n, n), matvec=kmul, dtype=float), k=nev,
                       M=LinearOperator((n, n), matvec=mmul, dtype=float), sigma=sigma,
                       OPinv=LinearOperator((n, n), matvec=opinv, dtype=float), which="LM", v0=v0,
                       tol=tol, ncv=max(2 * nev + 1, 30))
    t3 = time.perf_counter()
    order_ = np.argsort(vals)
    vals, vecs = vals[order_], vecs[:, order_]
    phys = vals > 1e-6 * lam_scale                     # drop isolated-conductor zero modes
    n_zero = int((~phys).sum())
    vals, vecs = vals[phys][:k], vecs[:, phys][:, :k]
    if len(vals) < k:
        raise RuntimeError(f"only {len(vals)} physical modes among {nev} (zeros {n_zero})")
    gfs = []
    for j in range(k):
        g = GridFunction(fes)
        g.vec.FV().NumPy()[:] = vecs[:, j] / np.sqrt(vecs[:, j] @ mmul(vecs[:, j]))
        gfs.append(g)
    info = dict(ndof=int(n), n_free=int(free.sum()), n_h1=int(fesh1.ndof), n_zero=n_zero, sigma=float(sigma),
                t_assemble=t1 - t0, t_factor=t2 - t1, t_eig=t3 - t2, n_el=int(mesh.ne))
    return dict(lam=vals, f_hz=C0 * np.sqrt(vals) / (2 * np.pi), gfs=gfs, fes=fes, info=info)


# ─────────────────────────── evaluation ────────────────────────

def locate(mesh, pts):
    """Find physical points pts [N,3] in the (curved) mesh → (mesh points of the inside ones,
    inside [N] bool); points outside the mesh are flagged False."""
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    mips = mesh(pts[:, 0], pts[:, 1], pts[:, 2])
    inside = mips["nr"] >= 0
    return mips[inside], inside


def evaluate(cf, located, dim=3):
    """cf at located points (locate(...)) → [N, dim], 0 outside."""
    mips, inside = located
    vals = np.zeros((len(inside), dim))
    if inside.any():
        vals[inside] = np.asarray(cf(mips)).reshape(-1, dim)
    return vals


def eval_field(mesh, cf, pts):
    """cf (3-vector CoefficientFunction) at physical points pts [N,3] → (values [N,3], inside [N]);
    points outside the curved mesh get 0."""
    loc = locate(mesh, pts)
    return evaluate(cf, loc), loc[1]


def axis_samples(mesh, axis_dir, axis_point, n_axis=2001):
    """Axis line through axis_point along axis_dir over the mesh extent: (t [P], dt, points [P,3])."""
    d = np.asarray(axis_dir, dtype=np.float64) / np.linalg.norm(axis_dir)
    p0 = np.asarray(axis_point, dtype=np.float64)
    pts_v = mesh.ngmesh.Coordinates()
    t = pts_v @ d
    t0, t1 = float(t.min()), float(t.max())
    dt = (t1 - t0) / n_axis
    tt = t0 + (np.arange(n_axis) + 0.5) * dt
    perp = p0 - (p0 @ d) * d
    return tt, dt, perp[None, :] + tt[:, None] * d[None, :]


def mode_qoi(mesh, gfs, f_hz, axis_dir=(0, 0, 1), axis_point=(0, 0, 0), sigma=5.8e7, beta=1.0,
             convention="linac", n_axis=2001, wall_order=8):
    """src.qoi QOI_KEYS of the NGSolve modes (copper walls, U = 1 J), directly from the high-order
    fields: U = ½ε0∫|E|², P_c = ½R_s/(ωμ0)² ∫_wall |n × curl E|², V = |∫ E·d e^{jωt/(βc)} dt| on the
    axis, E_pk / B_pk = max over the wall integration points (order wall_order) of |E| / |curl E|/ω.
    Wall values come from the adjacent volume element (BoundaryFromVolumeCF: a plain boundary
    evaluation of an HCurl field is its tangential trace, ≈ 0 on PEC).  L_acc = axis chord inside
    the mesh."""
    from ngsolve import (BND, TRIG, BoundaryFromVolumeCF, Cross, InnerProduct, IntegrationRule, Integrate,
                         curl, ds, specialcf)
    from src.qoi.operators import figures_of_merit, surface_resistance

    f = np.asarray(f_hz, dtype=np.float64).reshape(-1)
    w = 2 * np.pi * f
    Rs = surface_resistance(f, sigma)
    order = 2 * gfs[0].space.globalorder + 2
    nrm = specialcf.normal(3)
    mips = mesh.MapToAllElements(IntegrationRule(TRIG, wall_order), BND)
    on_wall = np.asarray(mesh.BoundaryCF({"wall": 1.0}, default=0.0)(mips)).reshape(-1) > 0.5
    tt, dt, axp = axis_samples(mesh, axis_dir, axis_point, n_axis)
    d = np.asarray(axis_dir, dtype=np.float64) / np.linalg.norm(axis_dir)
    raw = {k: np.zeros(len(gfs)) for k in ("U", "P", "V", "V0", "E", "B")}
    loc_ax = locate(mesh, axp)
    q = np.where(loc_ax[1], dt, 0.0)
    L = float(q.sum())
    for j, g in enumerate(gfs):
        Eb, Cb = BoundaryFromVolumeCF(g), BoundaryFromVolumeCF(curl(g))
        ct = Cross(nrm, Cb)
        raw["U"][j] = 0.5 * EPS0 * Integrate(InnerProduct(g, g), mesh, order=order)
        sw = Integrate(InnerProduct(ct, ct) * ds("wall", bonus_intorder=order), mesh)
        raw["P"][j] = 0.5 * Rs[j] / (w[j] * MU0) ** 2 * sw
        Ez = evaluate(g, loc_ax) @ d
        raw["V"][j] = np.abs((q * Ez * np.exp(1j * w[j] * tt / (beta * C0))).sum())
        raw["V0"][j] = (q * np.abs(Ez)).sum()
        raw["E"][j] = np.linalg.norm(np.asarray(Eb(mips)).reshape(-1, 3)[on_wall], axis=1).max()
        raw["B"][j] = np.linalg.norm(np.asarray(Cb(mips)).reshape(-1, 3)[on_wall], axis=1).max() / w[j]
    return figures_of_merit(f, Rs, raw["U"], raw["P"], raw["V"], raw["V0"], raw["E"], raw["B"], L, convention)


def edge_dofs(mesh, gfs, X_phys, edges, n_gauss=4):
    """N0 interpolant of the modes on a straight-sided mesh: u_e = ∫_e E·t dl (t from edges[:,0]
    to edges[:,1]) by n_gauss-point Gauss–Legendre.  Returns (U [Ne, K], frac_inside [Ne]) —
    quadrature points outside the curved mesh (chords across a concave wall) contribute 0."""
    s, wq = np.polynomial.legendre.leggauss(n_gauss)
    s, wq = 0.5 * (s + 1), 0.5 * wq
    xa, xb = X_phys[edges[:, 0]], X_phys[edges[:, 1]]
    tvec = xb - xa
    pts = (xa[:, None, :] + s[None, :, None] * tvec[:, None, :]).reshape(-1, 3)
    U = np.zeros((len(edges), len(gfs)))
    loc = locate(mesh, pts)
    for j, g in enumerate(gfs):
        Ev = evaluate(g, loc).reshape(len(edges), n_gauss, 3)
        U[:, j] = np.einsum("q,eqc,ec->e", wq, Ev, tvec)
    frac = loc[1].reshape(len(edges), n_gauss).mean(1)
    return U, frac
