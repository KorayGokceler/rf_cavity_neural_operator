"""Acceptance / wall-angle spread of beta=1 random half-cells: current ranges+rules vs widened ranges with
proposed rules (tight tolerance multi-start; re-entrant allowed up to theta_max with a web guard L-A >= 0.05 Req
when the wall turns back; Ri+b<Req-B replaced by monotone r)."""
import sys, numpy as np
from pathlib import Path
from scipy.optimize import least_squares
sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_halfcell import tangent
def walls(Req, Riris, L, A, B, a, b):
    ci, ce = np.array([0.0, Riris + b]), np.array([L, Req - B])
    P1 = lambda t: ci + np.array([a*np.sin(t), -b*np.cos(t)])
    P2 = lambda s: ce + np.array([-A*np.cos(s), B*np.sin(s)])
    T1 = lambda t: np.array([a*np.cos(t), b*np.sin(t)])
    T2 = lambda s: np.array([A*np.sin(s), B*np.cos(s)])
    cross = lambda u, v: u[0]*v[1]-u[1]*v[0]
    def res(x):
        t, s = x; d = P2(s) - P1(t)
        return [cross(T1(t), T2(s))/(a*A), cross(T1(t), d)/(a*L)]
    for x0 in ([np.pi/2, 0.0], [1.2, 0.3], [2.0, -0.5], [0.8, 0.8], [2.5, -1.0]):
        sol = least_squares(res, x0, bounds=([1e-3, -np.pi/2+1e-3], [np.pi-1e-3, np.pi/2]), xtol=1e-15, ftol=1e-15, gtol=1e-15)
        t, s = sol.x
        if np.max(np.abs(sol.fun)) > 1e-9 or (P2(s)-P1(t)) @ T1(t) <= 0: continue
        d = P2(s)-P1(t)
        if np.sign(cross(d, ci-P1(t))) == np.sign(cross(d, ce-P1(t))): continue    # wrong tangent type
        n = 120
        w = np.vstack([np.array([P1(x) for x in np.linspace(0, t, n)]), np.array([P2(x) for x in np.linspace(s, np.pi/2, n)])])
        if np.any(np.diff(w[:, 1]) < -1e-12): continue                              # r must be monotone
        return np.degrees(np.arctan2(d[1], d[0])), w[:, 0][n:].min()                # theta, min z after iris arc
    return None
def run(widened, proposed, n=2500, theta_max=105.0, seed=3):
    rng = np.random.default_rng(seed); acc, th = 0, []
    for _ in range(n):
        Req = 1.0; L = rng.uniform(0.45, 0.65)
        A = L*rng.uniform(0.6, 0.9 if widened else 0.85); B = A*rng.uniform(0.6 if widened else 0.85, 1.3 if widened else 1.25)
        a = L*rng.uniform(0.12 if widened else 0.15, 0.35 if widened else 0.30); b = a*rng.uniform(0.65 if widened else 1.0, 1.9)
        Ri = rng.uniform(0.25, 0.42)
        h = dict(Req=Req, Riris=Ri, L=L, A=A, B=B, a=a, b=b)
        if not proposed:
            if Ri + b >= Req - B: continue
            r = tangent(**h)
            if r is None or r["reentrant"]: continue
            acc += 1; th.append(r["theta"]); continue
        r = walls(**h)
        if r is None: continue
        theta, zmin = r
        if theta > theta_max or (theta > 90 and zmin < 0.05*Req): continue
        acc += 1; th.append(theta)
    th = np.array(th)
    return acc/n, np.round(np.percentile(th, [1, 10, 50, 90, 99]), 1), round(float(np.mean(th > 90)), 3)
for w, p in [(False, False), (False, True), (True, False), (True, True)]:
    a, pc, fre = run(w, p)
    print(f"ranges={'widened' if w else 'current'} rules={'proposed' if p else 'current'}: accepted {a:.3f}; theta p1/p10/p50/p90/p99 {pc}; fraction re-entrant {fre}")
