"""SLAC 2pi/3 cell (two-mile structure, 'cavity 79' of SLAC/AP-86): 3 periods closed by electric walls at the
cell mid-planes -> TM0 passband modes phi = n pi/3.  Published: 2a = 1.99 cm, 2b = 8.192 cm, t = 0.584 cm,
d = 3.499 cm.  Assumed: iris edge = full round of radius t/2 (not given in the extract)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_axisym as ax

def segment(a=9.95e-3, b=40.96e-3, t=5.84e-3, d=34.99e-3, N=3):
    rho = t / 2
    S = [("line", (0, 0), (0, b))]
    z = 0.0
    cur = (0, b)
    for i in range(N):
        zi = (i + 0.5) * d            # iris centre plane
        z1, z2 = zi - rho, zi + rho
        S.append(("line", cur, (z1, b))); S.append(("line", (z1, b), (z1, a + rho)))
        S.append(("arc", (z1, a + rho), (zi, a + rho), (zi, a)))
        S.append(("arc", (zi, a), (zi, a + rho), (z2, a + rho)))
        S.append(("line", (z2, a + rho), (z2, b)))
        cur = (z2, b)
    S.append(("line", cur, (N * d, b))); S.append(("line", (N * d, b), (N * d, 0))); S.append(("line", (N * d, 0), (0, 0)))
    return S

res, ne = ax.solve(segment(), 40.96e-3, nev=4, hfac=15)
for n, r in enumerate(res):
    print(f"phi = {n}pi/3: f = {r['f']/1e6:.1f} MHz")
print("published operating frequency 2856 MHz at 2pi/3")
