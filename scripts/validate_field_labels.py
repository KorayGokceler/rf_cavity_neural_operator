"""Analytic check of the adaptive field labels (docs/30 §1–2): the generator's label path — remeshing
adaptive refinement, HCurl p3, mode_qoi — on a copper pillbox (R = 40 mm, L = 50 mm), compared level by
level with the closed forms:
- the six lowest frequencies: TM010, TE111 ×2, TM011, TM110 ×2;
- the TM010 Q0, G, R/Q, E_pk/E_acc and B_pk/E_acc.

Exits non-zero when the final level misses a target:
- frequency 1e-5;
- Q0, G and R/Q 0.2 %;
- the peaks 1 %.

    python scripts/validate_field_labels.py [--max_ndof 300000] [--threads 2] [--json out.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.special import jn_zeros, jnp_zeros

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_gen import adaptive as ad  # noqa: E402
from src.qoi.analytic import pillbox_tm010  # noqa: E402
from src.qoi.figures import C0  # noqa: E402

R, L = 0.040, 0.050
TARGETS = {"f": 1e-5, "Q0": 2e-3, "G": 2e-3, "R/Q": 2e-3, "Epk/Eacc": 1e-2, "Bpk/Eacc": 1e-2}
KEYS = {"Q0": "Q0", "G": "G_ohm", "R/Q": "R_over_Q_ohm", "Epk/Eacc": "Epk_Eacc", "Bpk/Eacc": "Bpk_Eacc_mT_per_MVm"}


def analytic_freqs():
    """The six lowest PEC pillbox frequencies [Hz], ascending."""
    x01, x11, xp11 = jn_zeros(0, 1)[0], jn_zeros(1, 1)[0], jnp_zeros(1, 1)[0]
    k = [x01 / R, np.hypot(xp11 / R, np.pi / L), np.hypot(xp11 / R, np.pi / L), np.hypot(x01 / R, np.pi / L),
         x11 / R, x11 / R]
    return np.sort(k) * C0 / (2 * np.pi)


def errors(f_hz, q, f_ref, ref):
    """Relative errors: max over the six frequencies, then the TM010 (mode 0) figures of merit."""
    e = {"f": float(np.abs(np.asarray(f_hz[:6]) / f_ref - 1).max())}
    for n, key in KEYS.items():
        e[n] = float(q[key][0] / ref[key] - 1)
    return e


def run(max_ndof=300_000, tol_f=1e-6, tol_q=1e-3, maxh=0.012, max_iter=8, log=print):
    from netgen.occ import Axes, Cylinder, Z
    cyl = Cylinder(Axes((0, 0, -L / 2), Z), r=R, h=L)
    cyl.faces.name = "wall"
    f_ref, ref = analytic_freqs(), pillbox_tm010(R, L)
    remesh = ad.mesh_remesher(cyl, maxh, curve=3, curvaturesafety=1.5)
    rows, t0 = [], time.perf_counter()

    def on_iter(mesh, lab, q, rec):
        rows.append({"ne": rec["ne"], "ndof": rec["ndof"], "est_f": rec["est_rel_f"], "t": time.perf_counter() - t0,
                     **errors(lab["f_hz"], q, f_ref, ref)})
        r = rows[-1]
        log(f"{r['ne']:>7}{r['ndof']:>9}{r['est_f']:>9.1e}" + "".join(f"{r[n]:>10.1e}" for n in TARGETS)
            + f"{r['t']:>7.0f}")

    log(f"{'tets':>7}{'DOF':>9}{'est f':>9}" + "".join(f"{n:>10}" for n in TARGETS) + f"{'t [s]':>7}")
    res = ad.adaptive_modes(remesh(np.zeros((0, 3)), np.zeros(0)), 6, order=3, axis=((0, 0, 1), (0, 0, 0)),
                            remesh=remesh, on_iter=on_iter,
                            settings={"tol_f": tol_f, "tol_q": tol_q, "max_ndof": max_ndof, "max_iter": max_iter})
    last = rows[-1]
    fails = [n for n, tol in TARGETS.items() if not abs(last[n]) <= tol]
    return {"rows": rows, "stop": res["stop"], "fails": fails, "targets": TARGETS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max_ndof", type=int, default=300_000, help="label-space DOF budget (~4 GB at 300k)")
    ap.add_argument("--tol_f", type=float, default=1e-6)
    ap.add_argument("--tol_q", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--json", default=None, help="also write the per-level errors here")
    a = ap.parse_args(argv)
    import ngsolve
    ngsolve.SetNumThreads(max(1, a.threads))
    out = run(a.max_ndof, a.tol_f, a.tol_q)
    print(f"stop: {out['stop']}; targets " + ", ".join(f"{n} {t:.0e}" for n, t in TARGETS.items()))
    print("PASS" if not out["fails"] else "FAIL: " + ", ".join(out["fails"]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 1 if out["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
