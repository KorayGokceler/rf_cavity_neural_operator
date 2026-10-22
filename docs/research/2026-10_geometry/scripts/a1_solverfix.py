"""Does the single start [pi/2, 0] with tight tolerances recover the valid walls the code misses?
Also: fraction of random beta=1 draws lost to solver failure."""
import sys, numpy as np
from pathlib import Path
from scipy.optimize import least_squares
sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_halfcell import tangent
from a1_notangent3 import solve_all
def tight(Req, Riris, L, A, B, a, b):
    ci, ce = np.array([0.0, Riris + b]), np.array([L, Req - B])
    P1 = lambda t: ci + np.array([a*np.sin(t), -b*np.cos(t)])
    P2 = lambda s: ce + np.array([-A*np.cos(s), B*np.sin(s)])
    T1 = lambda t: np.array([a*np.cos(t), b*np.sin(t)])
    T2 = lambda s: np.array([A*np.sin(s), B*np.cos(s)])
    cross = lambda u, v: u[0]*v[1]-u[1]*v[0]
    def res(x):
        t, s = x; d = P2(s) - P1(t)
        return [cross(T1(t), T2(s))/(a*A), cross(T1(t), d)/(a*L)]
    sol = least_squares(res, [np.pi/2, 0.0], bounds=([1e-3, -np.pi/2+1e-3], [np.pi-1e-3, np.pi/2]), xtol=1e-15, ftol=1e-15, gtol=1e-15)
    t, s = sol.x
    return np.max(np.abs(sol.fun)) <= 1e-9 and (P2(s)-P1(t)) @ T1(t) > 0
rng = np.random.default_rng(2)
tot = miss = rec_tight = 0; th_missed = []
for _ in range(3000):
    Req = 1.0
    L = Req*rng.uniform(0.45, 0.65); A = L*rng.uniform(0.6, 0.85); B = A*rng.uniform(0.85, 1.25)
    a = L*rng.uniform(0.15, 0.30); b = a*rng.uniform(1.0, 1.9); Ri = Req*rng.uniform(0.25, 0.42)
    h = dict(Req=Req, Riris=Ri, L=L, A=A, B=B, a=a, b=b)
    tot += 1
    if Ri + b >= Req - B or tangent(**h) is not None:
        continue
    f = solve_all(**h)
    good = [th for o, m, th in f if o and m]
    if good:
        miss += 1; th_missed.append(good[0]); rec_tight += tight(**h)
print("random beta=1 draws:", tot, " valid walls missed by the solver:", miss, f"({100*miss/tot:.1f} %)",
      " recovered by tight tolerances alone:", rec_tight)
if th_missed:
    print(" wall angle of missed walls (deg) p5/50/95:", np.round(np.percentile(th_missed, [5, 50, 95]), 1))
