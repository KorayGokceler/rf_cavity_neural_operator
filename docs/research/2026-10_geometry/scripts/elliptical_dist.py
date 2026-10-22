"""Distribution of the elliptical generator's mid-cell shape (no meshing): ratios, wall angle, rejection rate."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
from scipy.optimize import least_squares
from src.data_gen import cavity_shapes as cs

def wall_angle(h):
    """Angle [deg] of the common tangent to the beam axis (90 = vertical wall), or None."""
    Req, Riris, L, A, B, a, b = (h[k] for k in cs.HALF_KEYS)
    ci, ce = np.array([0.0, Riris + b]), np.array([L, Req - B])
    P1 = lambda t: ci + np.array([a * np.sin(t), -b * np.cos(t)])
    P2 = lambda s: ce + np.array([-A * np.cos(s), B * np.sin(s)])
    T1 = lambda t: np.array([a * np.cos(t), b * np.sin(t)])
    T2 = lambda s: np.array([A * np.sin(s), B * np.cos(s)])
    cr = lambda u, v: u[0] * v[1] - u[1] * v[0]
    res = lambda x: [cr(T1(x[0]), T2(x[1])) / (a * A), cr(T1(x[0]), P2(x[1]) - P1(x[0])) / (a * L)]
    sol = least_squares(res, [np.pi / 2, 0.0], bounds=([1e-3, -np.pi / 2 + 1e-3], [np.pi - 1e-3, np.pi / 2]))
    if np.max(np.abs(sol.fun)) > 1e-9:
        return None
    T = T1(sol.x[0])
    return float(np.degrees(np.arctan2(T[1], T[0])))

rng = np.random.default_rng(0)
rows, rej = [], 0
for i in range(3000):
    p, halves, walls = cs.draw_elliptical(rng)
    m = halves["mid"]
    rows.append((m["Riris"] / m["Req"], m["L"] / m["Req"], m["A"] / m["L"], m["B"] / m["A"], m["a"] / m["L"], m["b"] / m["a"],
                 wall_angle(m) or np.nan, p["beta"], p["anchor"], p["n_cells"]))
R = np.array(rows, float)
names = ["Riris/Req", "L/Req", "A/L", "B/A", "a/L", "b/a", "wall angle deg", "beta"]
for j, n in enumerate(names):
    q = np.nanpercentile(R[:, j], [1, 10, 50, 90, 99])
    print(f"{n:15s} p1 {q[0]:.3f}  p10 {q[1]:.3f}  median {q[2]:.3f}  p90 {q[3]:.3f}  p99 {q[4]:.3f}")
print("anchor share", np.mean(R[:, 8] > 0), "low-beta share", np.mean(R[:, 7] < 1))
print("wall angle > 85 deg:", np.mean(R[:, 6] > 85), " > 89:", np.mean(R[:, 6] > 89))
# rejection rate of the random (non-anchor) branch
ok = tot = 0
r2 = np.random.default_rng(1)
for i in range(3000):
    beta = 1.0
    Req = r2.uniform(0.04, 0.11)
    h = cs._draw_half(r2, Req, Req * r2.uniform(0.25, 0.42), beta)
    tot += 1
    ok += cs._valid_half(h) is not None
print(f"random beta=1 half-cells accepted: {ok}/{tot} = {ok/tot:.2f}")
for name, a in cs.ANCHORS.items():
    print(name, "wall angle", wall_angle(a), "Riris/Req", a["Riris"] / a["Req"], "L/Req", a["L"] / a["Req"])
