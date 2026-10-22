"""1.6-cell BNL/SLAC/UCLA gun rebuilt from the SLAC TN 04-4 table (report a3, docs/33).
Published: full cell L 3.25 cm, D 8.35 cm; half cell L 2.28 cm, D 8.30 cm; iris D 2.50 cm with a 0.95 cm
radius of curvature.  Assumed (not in the extract): iris thickness 2 x 0.95 cm (full round), exit iris =
same shape, exit pipe radius = iris radius, length 3 x radius, sharp cell corners, flat cathode wall."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_axisym as ax

def gun(Lh=22.8e-3, Rh=41.5e-3, Lf=32.5e-3, Rf=41.75e-3, a=12.5e-3, rho=9.5e-3, pipe=3.0):
    t = 2 * rho
    z1 = Lh                  # half-cell end / iris start
    zc1 = z1 + rho           # iris 1 centre plane
    z2 = z1 + t              # full cell start
    z3 = z2 + Lf             # full cell end
    zc2 = z3 + rho
    zp = zc2 + pipe * a
    S = []
    L = lambda p, q: S.append(("line", p, q))
    A = lambda p, c, q: S.append(("arc", p, c, q))
    L((0, 0), (0, Rh)); L((0, Rh), (z1, Rh)); L((z1, Rh), (z1, a + rho))
    A((z1, a + rho), (zc1, a + rho), (zc1, a)); A((zc1, a), (zc1, a + rho), (z2, a + rho))
    L((z2, a + rho), (z2, Rf)); L((z2, Rf), (z3, Rf)); L((z3, Rf), (z3, a + rho))
    A((z3, a + rho), (zc2, a + rho), (zc2, a))
    L((zc2, a), (zp, a)); L((zp, a), (zp, 0)); L((zp, 0), (0, 0))
    return S

for hf in (12, 25):
    res, ne = ax.solve(gun(), 41.75e-3, nev=2, hfac=hf)
    f = [r["f"] / 1e6 for r in res]
    print(f"hfac {hf}: ne {ne}  TM0 modes {f[0]:.1f} MHz, {f[1]:.1f} MHz, separation {f[1]-f[0]:.2f} MHz")
# sensitivity to the assumed iris thickness
for t_scale in (0.8, 1.2):
    res, ne = ax.solve(gun(rho=9.5e-3 * t_scale), 41.75e-3, nev=2, hfac=12)
    f = [r["f"] / 1e6 for r in res]
    print(f"iris round x{t_scale}: {f[0]:.1f}, {f[1]:.1f} MHz, sep {f[1]-f[0]:.2f}")
