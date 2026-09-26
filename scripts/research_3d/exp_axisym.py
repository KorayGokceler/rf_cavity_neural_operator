"""E3: axisymmetric (2.5D) cavity eigenmodes on the meridian plane (x0 = z, x1 = r).

(a) m = 0 TM (monopole, accelerating) modes, SUPERFISH-style scalar H_φ formulation
    ∫ [∂zH ∂zv + (1/r)∂r(rH)(1/r)∂r(rv)] r dA = k² ∫ H v r dA,
    H = 0 on the axis (and on magnetic walls), natural (n×E = 0) on PEC walls.
    Lagrange P2 — NO spurious modes (curl(H_φ φ̂) = 0 ⇒ H = 0).
(b) m ≥ 1 (dipole, HOM) modes: E = (E_r cos mφ, E_φ sin mφ, E_z cos mφ),
    E_mer = (E_z, E_r) ∈ Nédélec N1 (meridian), u = r·E_φ ∈ P1:
    ∫ r rot(E)² + (1/r)|m E + ∇u|²  =  k² ∫ r|E|² + u²/r.
    Kernel: E = ∇p, u = −m p (3D gradients of p(r,z) cos mφ) → projected away.
Validated on a pillbox and applied to a TESLA mid-cell (π- and 0-mode, k_cc).
"""
import sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import fsolve
from scipy.special import jn_zeros, jnp_zeros
from skfem import (Basis, BilinearForm, ElementComposite, ElementTriN1, ElementTriP1,
                   ElementTriP2, MeshTri, condense)
from skfem.helpers import curl, dot, grad

C0 = 299792458.0


def gmsh_meridian(build, h, h_min=None):
    """2D gmsh triangulation; build(occ) returns the surface tag. Physical groups
    are not needed: boundary pieces are identified geometrically afterwards."""
    import gmsh
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("mer")
    build(gmsh.model.occ)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h_min or 0.2 * h)
    gmsh.model.mesh.generate(2)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    et, _, conn = gmsh.model.mesh.getElements(2)
    tri = np.asarray(conn[[int(t) for t in et].index(2)], dtype=np.int64).reshape(-1, 3)
    xyz = np.asarray(xyz).reshape(-1, 3)[:, :2]
    tags = np.asarray(tags, dtype=np.int64)
    t2r = np.full(tags.max() + 1, -1); t2r[tags] = np.arange(len(tags)); tri = t2r[tri]
    used = np.unique(tri); rm = np.full(len(tags), -1); rm[used] = np.arange(len(used))
    gmsh.finalize()
    return MeshTri(np.ascontiguousarray(xyz[used].T), np.ascontiguousarray(rm[tri].T))


# ───────────────────────── (a) m = 0, H_φ, P2 ─────────────────────────

@BilinearForm
def a_hphi(u, v, w):
    r = w.x[1]
    # (1/r)∂r(r u) = ∂r u + u/r
    return r * u.grad[0] * v.grad[0] + r * (u.grad[1] + u / r) * (v.grad[1] + v / r)


@BilinearForm
def m_hphi(u, v, w):
    return w.x[1] * u * v


def solve_m0(mesh, dirichlet_facets, k, elem=ElementTriP2()):
    """TM0 monopole modes. dirichlet_facets: facet indices with H_φ = 0 (axis, magnetic walls)."""
    t0 = time.perf_counter()
    b = Basis(mesh, elem, intorder=6)
    A, M = a_hphi.assemble(b), m_hphi.assemble(b)
    D = b.get_dofs(dirichlet_facets).all()
    Ac, Mc, _, _ = condense(A, M, D=D)
    vals = spla.eigsh(Ac.tocsc(), k=k, M=Mc.tocsc(), sigma=0.0, which='LM')[0]
    return np.sort(vals), Ac.shape[0], time.perf_counter() - t0


# ───────────────────── (b) m ≥ 1, (N1 × P1) mixed ─────────────────────

def solve_m(mesh, m_az, k):
    """Azimuthal index m ≥ 1 modes of a closed body of revolution (PEC walls, axis).
    All boundary DOFs are essential: n×E_mer = 0 and u = 0 on PEC walls; on the
    axis E_z = 0 (tangential N1 DOF) and u = r E_φ = 0."""
    t0 = time.perf_counter()
    e = ElementComposite(ElementTriN1(), ElementTriP1())
    b = Basis(mesh, e, intorder=6)

    @BilinearForm
    def a(E, u, F, v, w):
        r = w.x[1]
        mE_gu = m_az * E.value + grad(u)
        mF_gv = m_az * F.value + grad(v)
        return r * curl(E) * curl(F) + dot(mE_gu, mF_gv) / r

    @BilinearForm
    def mm(E, u, F, v, w):
        r = w.x[1]
        return r * dot(E.value, F.value) + u.value * v.value / r

    A, M = a.assemble(b), mm.assemble(b)
    D = b.get_dofs(mesh.boundary_facets()).all()
    I = np.setdiff1d(np.arange(b.N), D)
    # kernel generator: p ∈ P1_0 ↦ (E = ∇p, u = −m p)
    iN, iP = b.split_indices()
    f = mesh.facets
    nv = mesh.p.shape[1]
    nf = f.shape[1]
    Gcand = sp.csr_matrix((np.r_[-np.ones(nf), np.ones(nf)], (np.r_[iN, iN], np.r_[f[0], f[1]])), shape=(b.N, nv))
    Gu = sp.csr_matrix((-m_az * np.ones(nv), (iP, np.arange(nv))), shape=(b.N, nv))
    Iv = np.setdiff1d(np.arange(nv), mesh.boundary_nodes())
    G = None
    for sgn in (1, -1):
        Gt = (sgn * Gcand + Gu)[:, Iv]
        if abs(A @ Gt).max() < 1e-8 * abs(A).max():
            G = Gt[I]; break
    if G is None:
        raise RuntimeError("could not identify N1 edge orientation")
    Ai, Mi = A[I][:, I].tocsc(), M[I][:, I].tocsr()
    sigma = -1e-3 * Ai.diagonal().mean() / Mi.diagonal().mean()
    lu = spla.splu((Ai - sigma * Mi).tocsc())
    Kp = (G.T @ Mi @ G).tocsc(); lup = spla.splu(Kp)
    MG = (Mi @ G).tocsr()
    P = lambda x: x - G @ lup.solve(MG.T @ x)
    OP = spla.LinearOperator(Ai.shape, matvec=lambda x: P(lu.solve(x)), dtype=float)
    v0 = P(np.random.default_rng(0).standard_normal(Ai.shape[0]))
    vals = spla.eigsh(Ai, k=k, M=Mi, sigma=sigma, OPinv=OP, which='LM', v0=v0)[0]
    return np.sort(vals), Ai.shape[0], G.shape[1], time.perf_counter() - t0


# ───────────────────────── geometries ─────────────────────────

def pillbox_build(R, L):
    def build(occ):
        occ.addRectangle(0, 0, 0, L, R)   # (z, r) ∈ [0,L]×[0,R]
    return build


def tesla_midcell_profile(L=57.7e-3, R_iris=35e-3, R_eq=103.3e-3, A=42e-3, B=42e-3, a=12e-3, b=19e-3, n=60):
    """Half cell (z from iris plane 0 to equator plane L) of the TESLA mid-cup
    (Aune et al., PRST-AB 3, 092001 (2000), Table 1): iris ellipse (a, b) centred
    at (0, R_iris + b), equator ellipse (A, B) centred at (L, R_eq − B), joined by
    their common tangent.  Returns the wall polyline from (0, R_iris) to (L, R_eq)."""
    c1 = np.array([0.0, R_iris + b]); c2 = np.array([L, R_eq - B])
    P1 = lambda t: c1 + np.array([a * np.sin(t), -b * np.cos(t)])
    d1 = lambda t: np.array([a * np.cos(t), b * np.sin(t)])
    P2 = lambda s: c2 + np.array([-A * np.sin(s), B * np.cos(s)])
    d2 = lambda s: np.array([A * np.cos(s), B * np.sin(s)])
    cr = lambda u, v: u[0] * v[1] - u[1] * v[0]
    t, s = fsolve(lambda x: [cr(P2(x[1]) - P1(x[0]), d1(x[0])), cr(P2(x[1]) - P1(x[0]), d2(x[1]))], [1.3, 1.3])
    iris = [P1(tt) for tt in np.linspace(0, t, n)]
    equ = [P2(ss) for ss in np.linspace(s, 0, n)]
    slope = np.degrees(np.arctan2(*(d1(t)[::-1])))
    return np.array(iris), np.array(equ), (t, s, slope)


def tesla_build(iris, equ, L):
    def build(occ):
        pts_i = [occ.addPoint(z, r, 0) for z, r in iris]
        pts_e = [occ.addPoint(z, r, 0) for z, r in equ]
        p0 = occ.addPoint(0, 0, 0); pL = occ.addPoint(L, 0, 0)
        c = [occ.addLine(p0, pL), occ.addLine(pL, pts_e[-1])]
        c += [occ.addSpline(pts_e[::-1])]              # equator top → tangent point P2
        c += [occ.addLine(pts_e[0], pts_i[-1])]         # P2 → P1 (straight wall)
        c += [occ.addSpline(pts_i[::-1])]               # P1 → iris bottom (0, R_iris)
        c += [occ.addLine(pts_i[0], p0)]                # iris plane down to the axis
        occ.addPlaneSurface([occ.addCurveLoop(c)])
    return build


def facets_where(mesh, pred):
    mid = mesh.p[:, mesh.facets].mean(axis=1)
    bf = mesh.boundary_facets()
    return bf[pred(mid[:, bf])]


if __name__ == "__main__":
    R, L = 1.0, 1.0
    # ---- pillbox, m = 0 (TM0np) ----
    j0 = jn_zeros(0, 3)
    ref0 = np.sort([(j0[n] / R) ** 2 + (p * np.pi / L) ** 2 for n in range(3) for p in range(3)])[:5]
    print("pillbox R=L=1, m=0 TM0np analytic k:", np.round(np.sqrt(ref0), 4))
    for h in (0.2, 0.1, 0.05, 0.025):
        mesh = gmsh_meridian(pillbox_build(R, L), h)
        axis = facets_where(mesh, lambda x: x[1] < 1e-9)
        for el, nm in ((ElementTriP1(), 'P1'), (ElementTriP2(), 'P2')):
            vals, ndof, t = solve_m0(mesh, axis, 5, el)
            print(f"  m=0 {nm} h={h:.3f} dof={ndof:6d} t={t*1e3:7.1f} ms  rel λ err:",
                  np.array2string(vals / ref0 - 1, precision=1, formatter={'float': lambda x: f'{x:+.1e}'}))
    # ---- pillbox, m = 1 ----
    j1, jp1 = jn_zeros(1, 3), jnp_zeros(1, 3)
    ref1 = np.sort([(j1[n] / R) ** 2 + (p * np.pi / L) ** 2 for n in range(3) for p in range(3)] +
                   [(jp1[n] / R) ** 2 + (p * np.pi / L) ** 2 for n in range(3) for p in range(1, 3)])[:6]
    print("pillbox m=1 analytic k (TE111, TM110, TM111, TE121, TE112, TM120):", np.round(np.sqrt(ref1), 4))
    for h in (0.2, 0.1, 0.05, 0.025):
        mesh = gmsh_meridian(pillbox_build(R, L), h)
        vals, ndof, nker, t = solve_m(mesh, 1, 6)
        print(f"  m=1 N1×P1 h={h:.3f} dof={ndof:6d} kernel={nker:5d} t={t*1e3:7.1f} ms  rel λ err:",
              np.array2string(vals / ref1 - 1, precision=1, formatter={'float': lambda x: f'{x:+.1e}'}))
    sys.stdout.flush()
    # ---- TESLA mid-cell (half cell), m = 0: π-mode and 0-mode ----
    Lc = 57.7e-3
    iris, equ, (t, s, slope) = tesla_midcell_profile(L=Lc)
    print(f"TESLA mid half-cell: tangent at iris-ellipse t={t:.3f}, equator s={s:.3f}, wall angle {slope:.2f}°")
    for h in (4e-3, 2e-3, 1e-3, 0.5e-3):
        mesh = gmsh_meridian(tesla_build(iris, equ, Lc), h)
        axis = facets_where(mesh, lambda x: x[1] < 1e-9)
        iris_plane = facets_where(mesh, lambda x: (x[0] < 1e-9) & (x[1] > 1e-9))
        out = {}
        for nm, Df in (("pi", np.r_[axis, iris_plane]), ("zero", axis)):
            vals, ndof, tt = solve_m0(mesh, Df, 2)
            out[nm] = (C0 * np.sqrt(vals) / (2 * np.pi) / 1e9, tt, ndof)
        fpi, f0 = out['pi'][0][0], out['zero'][0][0]
        kcc = 2 * (fpi - f0) / (fpi + f0)
        print(f"  h={h*1e3:.1f} mm dof={out['pi'][2]:6d}  f_pi={fpi:.5f} GHz  f_0={f0:.5f} GHz  k_cc={kcc*100:.3f}%  "
              f"t={out['pi'][1]*1e3:.0f}+{out['zero'][1]*1e3:.0f} ms  (next TM0 band: {out['pi'][0][1]:.4f} GHz)")
