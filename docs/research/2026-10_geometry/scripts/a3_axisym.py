"""Axisymmetric TM0 eigen-solve (H_phi formulation) of the generator's reentrant / pillbox_pipes profiles.

Read-only use of the repository: imports src.data_gen.cavity_shapes, replaces revolve_segments by a
capture so the builders return the (z, r) profile instead of calling gmsh.  Research script of report a3
(docs/33); `python a3_axisym.py` runs the analytic pillbox check.

Weak form (u = H_phi, u = 0 on the axis, natural BC = PEC wall):
  int [ r u_z w_z + (1/r) d(r u)/dr d(r w)/dr ] dA = k^2 int r u w dA
E_z = (u/r + u_r)/(j w eps0), E_r = -u_z/(j w eps0).  U = pi mu0 int u^2 r dA (peak amplitudes),
P = pi Rs int_wall u^2 r ds, V = |int E_z(0,z) exp(j k z) dz| (beta = 1), R/Q = V^2/(w U) (linac).
"""
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.data_gen import cavity_shapes as cs  # noqa: E402

from netgen.occ import Pnt, Segment, ArcOfCircle, Wire, Face, OCCGeometry  # noqa: E402
import ngsolve as ngs  # noqa: E402

C0, MU0, EPS0, SIGMA = 299792458.0, 4e-7 * np.pi, 8.8541878128e-12, 5.8e7

_captured = []


def _capture(occ, segs):
    _captured.append(segs)
    return 1


cs.revolve_segments = _capture
cs.MIN_FILLET = 0.05


def draw(family, rng):
    _captured.clear()
    if family == "reentrant":
        _, p = cs.build_reentrant(None, rng)
    elif family == "pillbox_pipes":
        _, p = cs.build_pillbox_pipes(None, rng)
    else:
        raise ValueError(family)
    return p, _captured[-1]


def pillbox_segments(R, d):
    return [("line", (0, 0), (0, R)), ("line", (0, R), (d, R)), ("line", (d, R), (d, 0)), ("line", (d, 0), (0, 0))]


def build_face(segs):
    edges = []
    for s in segs:
        if s[0] == "line":
            p, q = np.asarray(s[1], float), np.asarray(s[2], float)
            if np.linalg.norm(p - q) < 1e-12:
                continue
            e = Segment(Pnt(p[0], p[1], 0), Pnt(q[0], q[1], 0))
            e.name = "axis" if (abs(p[1]) < 1e-12 and abs(q[1]) < 1e-12) else "wall"
        else:
            p, c, q = (np.asarray(x, float) for x in s[1:])
            rad = np.linalg.norm(p - c)
            mvec = 0.5 * (p + q) - c
            m = c + rad * mvec / np.linalg.norm(mvec)
            e = ArcOfCircle(Pnt(p[0], p[1], 0), Pnt(m[0], m[1], 0), Pnt(q[0], q[1], 0))
            e.name = "wall"
        edges.append(e)
    face = Face(Wire(edges))
    scale = max(abs(c) for e in face.edges for c in (e.start.x, e.start.y, e.end.x, e.end.y))
    for e in face.edges:                      # names set before Face() are not kept: name by geometry
        on_axis = max(abs(e.start.y), abs(e.end.y), abs(e.center.y)) < 1e-9 * scale
        e.name = "axis" if on_axis else "wall"
    return face


def solve(segs, R_char, order=3, nev=3, hfac=12.0):
    face = build_face(segs)
    geo = OCCGeometry(face, dim=2)
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=R_char / hfac))
    mesh.Curve(order)
    fes = ngs.H1(mesh, order=order, dirichlet="axis")
    u, w = fes.TnT()
    r = ngs.y
    a = ngs.BilinearForm(fes)
    a += (r * ngs.grad(u)[0] * ngs.grad(w)[0] + (u + r * ngs.grad(u)[1]) * (w + r * ngs.grad(w)[1]) / r) * ngs.dx
    m = ngs.BilinearForm(fes)
    m += r * u * w * ngs.dx
    a.Assemble()
    m.Assemble()
    free = np.array(list(fes.FreeDofs()), bool)
    A = _csr(a.mat)[free][:, free]
    M = _csr(m.mat)[free][:, free]
    lam, vec = eigsh(A, k=nev, M=M, sigma=0.0, which="LM")
    idx = np.argsort(lam)
    lam, vec = lam[idx], vec[:, idx]
    out = []
    for j in range(nev):
        gf = ngs.GridFunction(fes)
        full = np.zeros(fes.ndof)
        full[free] = vec[:, j]
        gf.vec.FV().NumPy()[:] = full
        out.append(qoi(mesh, gf, lam[j], segs))
    return out, mesh.ne


def _csr(mat):
    val, col, ind = mat.CSR()
    n = mat.height
    return sp.csr_matrix((np.array(val), np.array(col), np.array(ind)), shape=(n, mat.width))


def qoi(mesh, gf, lam, segs):
    k = np.sqrt(lam)
    om = C0 * k
    r = ngs.y
    U = np.pi * MU0 * ngs.Integrate(gf * gf * r, mesh)
    wallint = ngs.Integrate(gf * gf * r, mesh, ngs.BND, definedon=mesh.Boundaries("wall"))
    Rs = np.sqrt(om * MU0 / (2 * SIGMA))
    P = np.pi * Rs * wallint
    # axis line: z range of the profile
    pts = np.array([s[1] for s in segs] + [s[-1] for s in segs], float)
    z0, z1 = pts[:, 0].min(), pts[:, 0].max()
    Rmax = pts[:, 1].max()
    zz = np.linspace(z0 + 1e-6 * Rmax, z1 - 1e-6 * Rmax, 3001)
    rr = 2e-4 * Rmax
    gr = ngs.grad(gf)
    Ez = np.zeros_like(zz)
    for i, z in enumerate(zz):
        try:
            mp = mesh(z, rr)
            Ez[i] = (gf(mp) / rr + gr(mp)[1]) / (om * EPS0)
        except Exception:
            Ez[i] = 0.0
    dz = zz[1] - zz[0]
    V = abs(np.sum(Ez * np.exp(1j * k * zz)) * dz)
    Vabs = np.sum(np.abs(Ez)) * dz
    Q0 = om * U / P
    return {"f": om / (2 * np.pi), "k": k, "RQ": V ** 2 / (om * U), "T": V / Vabs if Vabs > 0 else np.nan,
            "Q0": Q0, "G": Q0 * Rs, "V": V, "U": U}


if __name__ == "__main__":
    # analytic check: pillbox R = 5 cm, d = 5 cm (no pipes)
    R, d = 0.05, 0.05
    res, ne = solve(pillbox_segments(R, d), R)
    x01 = 2.404825557695773
    J1 = 0.5191474972894669
    f_an = x01 * C0 / (2 * np.pi * R)
    kk = x01 / R
    T_an = np.sin(kk * d / 2) / (kk * d / 2)
    RQ_an = 2 * (1 / (C0 * EPS0)) / (np.pi * x01 * J1 ** 2) * (d / R) * T_an ** 2
    G_an = (1 / (C0 * EPS0)) * x01 / (2 * (1 + R / d))     # eta x01 / (2 (1 + R/d))
    r0 = res[0]
    print(f"pillbox check ne={ne}: f {r0['f']:.6e} vs {f_an:.6e} ({r0['f']/f_an-1:+.2e}); "
          f"R/Q {r0['RQ']:.3f} vs {RQ_an:.3f} ({r0['RQ']/RQ_an-1:+.2e}); T {r0['T']:.4f} vs {T_an:.4f}; "
          f"G {r0['G']:.2f} vs {G_an:.2f} ({r0['G']/G_an-1:+.2e})")
