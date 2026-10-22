"""Re-check the 'no tangent' rejections: a valid wall tangent has the iris- and equator-ellipse centres on
OPPOSITE sides of the line (verified on TESLA). Count failures for which a multi-start solve finds such a
tangent with a monotone (non-re-entrant) wall, i.e. failures of the single-start solver."""
import sys, numpy as np
from pathlib import Path
from scipy.optimize import least_squares
sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_halfcell import tangent
def solve_all(Req, Riris, L, A, B, a, b):
    ci, ce = np.array([0.0, Riris + b]), np.array([L, Req - B])
    P1 = lambda t: ci + np.array([a*np.sin(t), -b*np.cos(t)])
    P2 = lambda s: ce + np.array([-A*np.cos(s), B*np.sin(s)])
    T1 = lambda t: np.array([a*np.cos(t), b*np.sin(t)])
    T2 = lambda s: np.array([A*np.sin(s), B*np.cos(s)])
    cross = lambda u, v: u[0]*v[1]-u[1]*v[0]
    def res(x):
        t, s = x; d = P2(s) - P1(t)
        return [cross(T1(t), T2(s))/(a*A), cross(T1(t), d)/(a*L)]
    found = []
    for t0 in np.linspace(0.2, 2.9, 7):
        for s0 in np.linspace(-1.3, 1.3, 7):
            sol = least_squares(res, [t0, s0], bounds=([1e-3, -np.pi/2+1e-3], [np.pi-1e-3, np.pi/2]), xtol=1e-15, ftol=1e-15, gtol=1e-15)
            t, s = sol.x
            if np.max(np.abs(sol.fun)) > 1e-9 or (P2(s)-P1(t)) @ T1(t) <= 0:
                continue
            d = P2(s)-P1(t)
            opposite = np.sign(cross(d, ci-P1(t))) != np.sign(cross(d, ce-P1(t)))
            n = 200
            wall = np.vstack([np.array([P1(x) for x in np.linspace(0, t, n)]), np.array([P2(x) for x in np.linspace(s, np.pi/2, n)])])
            mono = not (np.any(np.diff(wall[:, 0]) < -1e-12) or np.any(np.diff(wall[:, 1]) < -1e-12))
            found.append((opposite, mono, np.degrees(np.arctan2(d[1], d[0]))))
    return found
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    anchors = {"tesla": dict(Req=0.1033, Riris=0.035, L=0.0577, A=0.042, B=0.042, a=0.012, b=0.019),
               "ilc_ll": dict(Req=0.0986, Riris=0.030, L=0.0577, A=0.0501, B=0.0342, a=0.0076, b=0.0100)}
    for name, an in anchors.items():
        n_fail, kinds = 0, {}
        for _ in range(1500):
            mid = {k: v*rng.uniform(0.9, 1.1) for k, v in an.items()}
            if mid["Riris"] + mid["b"] >= mid["Req"] - mid["B"]:
                continue
            if tangent(**mid) is None:
                n_fail += 1
                f = solve_all(**mid)
                if any(o and m for o, m, _ in f): key = "valid wall missed by solver"
                elif any(o for o, m, _ in f): key = "valid-type tangent but re-entrant wall"
                elif f: key = "only wrong-type tangents"
                else: key = "no tangent at all"
                kinds[key] = kinds.get(key, 0) + 1
        print(name, "no-tangent rejections:", n_fail, kinds)
