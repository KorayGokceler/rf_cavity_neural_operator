"""E2: PEC pillbox (R=1, L=1) with Whitney N0 tets from gmsh OCC; analytic comparison.
Also: 3D scalar Dirichlet Laplacian (P1/P2) on the same meshes (Phase-1 cost)."""
import sys, time
import numpy as np
import scipy.sparse.linalg as spla
from scipy.special import j0
from skfem import Basis, ElementTetP1, ElementTetP2, condense
from skfem.models.poisson import laplace, mass
from n0lib import gmsh_mesh, assemble_n0, solve_projected, pillbox_spectrum, factor

R, L = 1.0, 1.0
K_MODES = 10
ref = pillbox_spectrum(R, L)[:K_MODES]
refk2 = np.array([r[0] for r in ref])
print("analytic k:", [f"{np.sqrt(k):.4f}({lab})" for k, lab in ref])
f_scale = 2.405 / 0.0883  # (for R = 88.3 mm → TM010 = 1.300 GHz)


def build(occ):
    occ.addCylinder(0, 0, 0, 0, 0, L, R)


def edge_dofs_from_nodal(mesh, basis, E_nodal):
    """Whitney interpolation of a nodal vector field [3, Nv]: DOF_e = ½(E(a)+E(b))·(b−a)."""
    e = mesh.edges
    t = mesh.p[:, e[1]] - mesh.p[:, e[0]]
    dofs = np.zeros(basis.N)
    dofs[basis.edge_dofs[0]] = 0.5 * ((E_nodal[:, e[0]] + E_nodal[:, e[1]]) * t).sum(0)
    return dofs


def edge_dofs_exact(mesh, basis, Efun, nq=4):
    """Canonical N0 interpolant: DOF_e = ∫_e E·t ds (Gauss–Legendre)."""
    e = mesh.edges
    a, b = mesh.p[:, e[0]], mesh.p[:, e[1]]
    xq, wq = np.polynomial.legendre.leggauss(nq)
    xq, wq = 0.5 * (xq + 1), 0.5 * wq
    val = 0
    for s, w in zip(xq, wq):
        val = val + w * (Efun(a + s * (b - a)) * (b - a)).sum(0)
    dofs = np.zeros(basis.N); dofs[basis.edge_dofs[0]] = val
    return dofs


def tm010(x):
    r = np.hypot(x[0], x[1])
    return np.stack([0 * r, 0 * r, j0(2.404825557695773 * r / R)])


for h in (0.35, 0.25, 0.18, 0.13, 0.1):
    t0 = time.perf_counter(); m = gmsh_mesh(build, h); tm = time.perf_counter() - t0
    t0 = time.perf_counter(); A = assemble_n0(m); ta = time.perf_counter() - t0
    vals, vecs, info = solve_projected(A, K_MODES)
    rel = vals / refk2 - 1
    # TM010 Rayleigh quotient of interpolated analytic field (nodal→edge vs exact edge integral)
    I = A['I']
    rq = {}
    for name, u in (('nodal', edge_dofs_from_nodal(m, A['basis'], tm010(m.p))),
                    ('edgeint', edge_dofs_exact(m, A['basis'], tm010))):
        ui = u[I]
        rq[name] = (ui @ (A['K'] @ ui)) / (ui @ (A['M'] @ ui)) / refk2[0] - 1
    # scalar 3D Dirichlet Laplacian (P1, P2) for cost comparison
    sc = {}
    for nm, E in (('P1', ElementTetP1()), ('P2', ElementTetP2())):
        t0 = time.perf_counter()
        bs = Basis(m, E); Kp, Mp = laplace.assemble(bs), mass.assemble(bs)
        Kc, Mc, _, _ = condense(Kp, Mp, D=bs.get_dofs())
        sol, _ = factor(Kc.tocsr())
        OP = spla.LinearOperator(Kc.shape, matvec=sol, dtype=float)
        lam = np.sort(spla.eigsh(Kc, k=6, M=Mc, sigma=0.0, OPinv=OP, which='LM')[0])
        sc[nm] = (Kc.shape[0], time.perf_counter() - t0, lam[0] / ((2.404825557695773 / R) ** 2 + (np.pi / L) ** 2) - 1)
    print(f"h={h:.2f} tets={m.t.shape[1]:7d} verts={m.p.shape[1]:6d} N0dof={A['K'].shape[0]:7d} "
          f"mesh={tm:.2f}s asm={ta:.2f}s fac={info['t_factor']:.2f}s eig={info['t_eig']:.2f}s | "
          f"rel λ: TM010 {rel[0]:+.2e} max|.| {np.abs(rel).max():.2e} | RQ(TM010 interp) nodal {rq['nodal']:+.2e} edge {rq['edgeint']:+.2e} | "
          f"scalar P1 dof={sc['P1'][0]} {sc['P1'][1]:.2f}s err {sc['P1'][2]:+.1e}; P2 dof={sc['P2'][0]} {sc['P2'][1]:.2f}s err {sc['P2'][2]:+.1e}")
    print("     k_h:", np.round(np.sqrt(vals), 4))
    sys.stdout.flush()
