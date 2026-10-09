"""E7 (docs/18): NGSolve high-order Nédélec with curved elements on the PEC pillbox R = L = 1.

The full HCurl space is used together with its full discrete-gradient space
(fes.CreateGradient()). The matrices are exported to SciPy and solved with the same
projected shift-invert as n0lib. NOTE: nograds=True combined with a P1-gradient
projection gave wrong results (~2 %) for p ≥ 2, so only pass "nograds" for p = 1.

Usage: python exp_ngsolve.py <maxh> <order> [<curve_order>] [full|nograds]
Needs: pip install ngsolve  (optional: pypardiso)
"""
import resource
import sys
import time

import numpy as np
import scipy.sparse as sp
from ngsolve import HCurl, BilinearForm, Mesh, curl, dx, TaskManager, SetNumThreads
from netgen.occ import Cylinder, Axes, OCCGeometry, Z

from n0lib import pillbox_spectrum, solve_projected

maxh, order = float(sys.argv[1]), int(sys.argv[2])
curve = int(sys.argv[3]) if len(sys.argv) > 3 else max(order, 1)
nograds = (sys.argv[4] == "nograds") if len(sys.argv) > 4 else False
ref = np.array([k for k, _ in pillbox_spectrum(1.0, 1.0)[:10]])


def tosp(mat):
    r, c, v = mat.COO()
    return sp.csr_matrix((np.asarray(v), (np.asarray(r), np.asarray(c))), shape=(mat.height, mat.width))


SetNumThreads(4)
with TaskManager():
    t0 = time.perf_counter()
    mesh = Mesh(OCCGeometry(Cylinder(Axes((0, 0, 0), Z), r=1.0, h=1.0)).GenerateMesh(maxh=maxh))
    mesh.Curve(curve)
    tm = time.perf_counter() - t0
    t0 = time.perf_counter()
    fes = HCurl(mesh, order=order, dirichlet=".*", nograds=nograds)
    u, v = fes.TnT()
    a = BilinearForm(curl(u) * curl(v) * dx).Assemble()
    m = BilinearForm(u * v * dx).Assemble()
    gradmat, fesh1 = fes.CreateGradient()
    ta = time.perf_counter() - t0

fr = np.array(list(fes.FreeDofs()), dtype=bool)
frh = np.array(list(fesh1.FreeDofs()), dtype=bool)
K, M, G = tosp(a.mat), tosp(m.mat), tosp(gradmat)
A = dict(K=K[fr][:, fr].tocsr(), M=M[fr][:, fr].tocsr(), G=G[fr][:, frh].tocsr())
lam, _, info = solve_projected(A, 10, tol=1e-8)
lam = np.sort(lam)[:10]
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"NGSolve p={order} maxh={maxh} ndof={fes.ndof} (free {int(fr.sum())}) mesh={tm:.2f}s assemble={ta:.2f}s "
      f"shift-invert+proj={info['t_factor'] + info['t_eig']:.2f}s peakRSS={rss:.0f}MB  "
      f"rel λ err: TM010={lam[0]/ref[0]-1:+.1e} max={np.abs(lam/ref-1).max():.1e}")
