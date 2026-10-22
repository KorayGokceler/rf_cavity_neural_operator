"""Distribution of shape ratios and TM010-like figures of merit for the generator's reentrant and
pillbox_pipes families (axisymmetric TM0 solve, a3_axisym).  Report a3, docs/33.

Usage: python a3_family_stats.py <reentrant|pillbox_pipes> <n draws> <seed>"""
import sys
from pathlib import Path
import time
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_axisym as ax  # noqa: E402

C0 = ax.C0


def one(family, rng):
    p, segs = ax.draw(family, rng)
    R = p["R"]
    res, ne = ax.solve(segs, R, nev=2)
    q = res[0]
    lam = C0 / q["f"]
    row = dict(p)
    row.update({"f": q["f"], "f2_TM0": res[1]["f"], "R_over_lambda": R / lam, "RQ": q["RQ"], "T": q["T"],
                "G": q["G"], "Q0": q["Q0"], "ne2d": ne})
    row["Lc_over_R"] = p["Lc"] / R
    row["rp_over_R"] = p["rp"] / R
    row["Lc_over_lambda"] = p["Lc"] / lam
    if family == "reentrant":
        row["g_over_Lc"] = p["gap"] / p["Lc"]
        row["g_over_R"] = p["gap"] / R
        row["g_over_lambda"] = p["gap"] / lam
        row["rt_over_R"] = (p["rp"] + p["t_tip"]) / R
        row["rn_over_R"] = p["rn"] / R
        row["nose_len_over_Lc"] = (p["Lc"] - p["gap"]) / 2 / p["Lc"]
        row["rho_tip_over_ttip"] = p["rho_tip"] / p["t_tip"]
        row["rho_tip_at_floor"] = float(abs(p["rho_tip"] - 0.05 * R) < 1e-12 * R)
    else:
        row["rho_iris_over_rp"] = p["rho_iris"] / p["rp"]
        row["rho_iris_at_floor"] = float(abs(p["rho_iris"] - 0.05 * R) < 1e-12 * R)
    return {k: float(v) for k, v in row.items() if not k.startswith("_")}


def main(family, n, seed):
    rng = np.random.default_rng(seed)
    rows, fails = [], 0
    t0 = time.time()
    for i in range(n):
        try:
            rows.append(one(family, rng))
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("fail", i, repr(e)[:120], flush=True)
    out = f"a3_{family}_stats.json"                       # current directory (data: do not commit)
    with open(out, "w") as fh:
        json.dump(rows, fh)
    print(f"{family}: {len(rows)} ok, {fails} failed, {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
