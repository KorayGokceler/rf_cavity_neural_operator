"""E6: cost per 3D sample (one process, fresh). Usage: python exp_timing.py <h> [pillbox|tesla]
pillbox R = L = 1 (normalised); 'tesla' = 3D TESLA mid-cell (full cell, PEC iris planes = 0-mode BC),
built by revolving the meridian profile of exp_axisym.py (lengths in metres)."""
import resource, sys, time
import numpy as np
from n0lib import gmsh_mesh, assemble_n0, solve_projected, pillbox_spectrum

h = float(sys.argv[1]); geo = sys.argv[2] if len(sys.argv) > 2 else "pillbox"
K = 10
t0 = time.perf_counter()
if geo == "pillbox":
    mesh = gmsh_mesh(lambda occ: occ.addCylinder(0, 0, 0, 0, 0, 1.0, 1.0), h)
    ref = np.array([k for k, _ in pillbox_spectrum(1.0, 1.0)[:K]])
else:
    from exp_axisym import tesla_midcell_profile
    Lc = 57.7e-3
    iris, equ, _ = tesla_midcell_profile(L=Lc)

    def build(occ):
        # meridian profile in the (x = r, z) plane; full cell = half cell mirrored about z = Lc
        half = [(r, z) for z, r in iris] + [(r, z) for z, r in equ]        # iris bottom → equator top
        n = len(iris)
        prof = half + [(r, 2 * Lc - z) for r, z in half[::-1][1:]]       # … → other iris bottom
        pts = [occ.addPoint(r, 0, z) for r, z in prof]
        p0, p1 = occ.addPoint(0, 0, 0), occ.addPoint(0, 0, 2 * Lc)
        c = [occ.addLine(p0, pts[0]), occ.addSpline(pts[:n]), occ.addLine(pts[n - 1], pts[n]),
             occ.addSpline(pts[n:3 * n - 1]), occ.addLine(pts[3 * n - 2], pts[3 * n - 1]),
             occ.addSpline(pts[3 * n - 1:]), occ.addLine(pts[-1], p1), occ.addLine(p1, p0)]
        s_ = occ.addPlaneSurface([occ.addCurveLoop(c)])
        occ.revolve([(2, s_)], 0, 0, 0, 0, 0, 1, 2 * np.pi)
    mesh = gmsh_mesh(build, h)
    ref = None
tm = time.perf_counter() - t0
t0 = time.perf_counter(); A = assemble_n0(mesh); ta = time.perf_counter() - t0
vals, vecs, info = solve_projected(A, K, tol=1e-8)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
line = (f"{geo} h={h:g} tets={mesh.t.shape[1]} verts={mesh.p.shape[1]} N0dof={A['K'].shape[0]} "
        f"mesh={tm:.2f}s asm={ta:.2f}s fac={info['t_factor']:.2f}s eig={info['t_eig']:.2f}s peakRSS={rss:.0f}MB ")
if ref is not None:
    line += f"relerr TM010={vals[0]/ref[0]-1:+.2e} max={np.abs(vals/ref-1).max():.2e}"
else:
    f = 299792458 * np.sqrt(vals) / (2 * np.pi) / 1e9
    line += "f[GHz]=" + np.array2string(f, precision=4)
print(line)
