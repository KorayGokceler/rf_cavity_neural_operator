"""Monte Carlo of the elliptical mid half-cell sampler of cavity_shapes.draw_elliptical (replicated),
reporting acceptance and the distribution of wall angles and ratios."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_halfcell import tangent
rng = np.random.default_rng(0)
ANCH = {"tesla": dict(Req=0.1033, Riris=0.035, L=0.0577, A=0.042, B=0.042, a=0.012, b=0.019),
        "ilc_ll": dict(Req=0.0986, Riris=0.030, L=0.0577, A=0.0501, B=0.0342, a=0.0076, b=0.0100)}
def draw_half(Req, Riris, beta):
    h = {"Req": Req, "Riris": Riris, "L": Req*rng.uniform(0.45, 0.65)*beta}
    h["A"] = h["L"]*rng.uniform(0.6, 0.85); h["B"] = h["A"]*rng.uniform(0.85, 1.25)
    h["a"] = h["L"]*rng.uniform(0.15, 0.30); h["b"] = h["a"]*rng.uniform(1.0, 1.9)
    return h
def valid(h):
    if h["Riris"] + h["b"] >= h["Req"] - h["B"]:
        return None, "overlap"
    r = tangent(**h)
    if r is None: return None, "no tangent"
    if r["reentrant"]: return None, "reentrant"
    return r, "ok"
# 1) anchors
for name, an in ANCH.items():
    stats = {}
    for _ in range(4000):
        mid = {k: v*rng.uniform(0.9, 1.1) for k, v in an.items()}
        r, why = valid(mid); stats[why] = stats.get(why, 0) + 1
    print(name, {k: round(v/4000, 3) for k, v in stats.items()})
# 2) random family
for label, betafun in [("beta=1", lambda: 1.0), ("beta 0.5-0.95", lambda: rng.uniform(0.5, 0.95))]:
    stats, th, rat = {}, [], []
    for _ in range(20000):
        Req = 1.0; beta = betafun()
        h = draw_half(Req, Req*rng.uniform(0.25, 0.42), beta)
        r, why = valid(h); stats[why] = stats.get(why, 0) + 1
        if r is not None:
            th.append(r["theta"]); rat.append([h["Riris"]/Req, h["L"]/Req, h["A"]/h["L"], h["B"]/h["A"], h["a"]/h["L"], h["b"]/h["a"], beta])
    th = np.array(th); rat = np.array(rat)
    print(label, {k: round(v/20000, 3) for k, v in stats.items()})
    print("  wall angle theta (deg, 90 = vertical) percentiles 0/5/50/95/100:", np.round(np.percentile(th, [0, 5, 50, 95, 100]), 1))
    names = ["Ri/Req", "L/Req", "A/L", "B/A", "a/L", "b/a", "beta"]
    for i, n in enumerate(names):
        print(f"  accepted {n:7s} min/median/max: {rat[:, i].min():.3f} {np.median(rat[:, i]):.3f} {rat[:, i].max():.3f}")
