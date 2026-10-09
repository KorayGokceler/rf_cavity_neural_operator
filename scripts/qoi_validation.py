"""Mesh-convergence validation of the cavity QoI (docs/24_CAVITY_QOI.md §3).

Reference cavities (src/qoi/analytic.py): copper pillbox TM010 (R = L = 0.1 m) and the E ∥ z
fundamental ('TE101') of the box 0.10 × 0.08 × 0.06 m, both with the beam axis x = y = 0.  Each is
meshed with gmsh at several sizes h, solved with the generator's E eigensolver, and run
through src.qoi.cavity_qoi; the table lists the relative error vs the closed form.  A second table
compares the surface-peak evaluation methods (src.qoi.operators.surface_operators).

    python scripts/qoi_validation.py [--h 0.02 0.01] [--shapes pillbox box] [--md out.md]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.qoi.analytic import reference_case  # noqa: E402
from src.qoi.operators import build_qoi_operators, qoi_from_dofs, surface_operators  # noqa: E402

CASES = {"pillbox": (0.10, 0.10), "box": (0.10, 0.08, 0.06)}
# gmsh saturates the box mesh for h ≳ 0.014 (same mesh at 0.014–0.02), so the box uses finer sizes
H_DEFAULT = {"pillbox": (0.02, 0.014, 0.01, 0.007), "box": (0.01, 0.007, 0.005, 0.0038)}
COLS = ("f_Hz", "Q0", "G_ohm", "R_over_Q_ohm", "T_transit", "Epk_Eacc", "Bpk_Eacc_mT_per_MVm")
SHORT = {"f_Hz": "f", "Q0": "Q0", "G_ohm": "G", "R_over_Q_ohm": "R/Q", "T_transit": "T",
         "Epk_Eacc": "Epk/Eacc", "Bpk_Eacc_mT_per_MVm": "Bpk/Eacc"}
# surface-peak variants: (label, method (E, H), project)
METHODS = (("E centroid", ("centroid", "recovered"), True), ("E recovered", ("recovered", "recovered"), True),
           ("B centroid", ("centroid", "centroid"), True), ("B recovered", ("centroid", "recovered"), True),
           ("B recovered, no proj.", ("centroid", "recovered"), False))


def run(hs, shapes=("pillbox", "box")):
    rows, peak_rows = [], []
    for shape in shapes:
        for h in (hs or H_DEFAULT[shape]):
            t0 = time.perf_counter()
            geom, Y, f, ref = reference_case(shape, CASES[shape], h)
            t1 = time.perf_counter()
            ops = build_qoi_operators(geom["X"], geom["tets"], geom["edges"], geom["scale"], geom["center"],
                                      M=geom["M"])
            q = qoi_from_dofs(ops, Y[:, 0], f[0])
            t2 = time.perf_counter()
            rows.append(dict(shape=shape, h=h, ne=len(geom["edges"]), nt=len(geom["tets"]),
                             err={k: float(q[k][0] / ref[k] - 1) for k in COLS},
                             val={k: float(q[k][0]) for k in COLS}, ref=ref, t_solve=t1 - t0, t_qoi=t2 - t1))
            pk = {}
            for label, method, proj in METHODS:
                Es, Hs = surface_operators(geom["X"], geom["tets"], geom["edges"], method, proj)
                qq = qoi_from_dofs({**ops, "Esurf": Es, "Hsurf": Hs}, Y[:, 0], f[0])
                key = "Epk_Eacc" if label.startswith("E") else "Bpk_Eacc_mT_per_MVm"
                pk[label] = float(qq[key][0] / ref[key] - 1)
            peak_rows.append(dict(shape=shape, h=h, ne=len(geom["edges"]), pk=pk))
            print(f"{shape:8s} h={h:.4f} Ne={len(geom['edges']):6d} solve {t1 - t0:5.1f}s "
                  f"qoi {t2 - t1:4.2f}s  " + "  ".join(f"{SHORT[k]} {rows[-1]['err'][k]:+.2%}" for k in COLS),
                  flush=True)
    return rows, peak_rows


def orders(rows):
    """Observed order p of |err| ~ Ne^(−p/3) between the two finest meshes of each shape."""
    out = {}
    for key in {r["shape"] for r in rows}:
        rr = sorted([r for r in rows if r["shape"] == key], key=lambda r: r["ne"])
        if len(rr) < 2:
            continue
        a, b = rr[-2], rr[-1]
        hr = (b["ne"] / a["ne"]) ** (1 / 3)
        out[key] = {k: (np.log(a["err"][k] / b["err"][k]) / np.log(hr)       # NaN on a sign change
                        if a["err"][k] * b["err"][k] > 0 else np.nan) for k in COLS}
    return out


def markdown(rows, peak_rows):
    lines = []
    for shape in dict.fromkeys(r["shape"] for r in rows):
        ref = next(r["ref"] for r in rows if r["shape"] == shape)
        dims = CASES[shape]
        lines.append(f"\n**{shape}** {dims} m, copper, analytic: f = {ref['f_Hz'] / 1e9:.5f} GHz, "
                     f"Q0 = {ref['Q0']:.0f}, G = {ref['G_ohm']:.2f} Ω, R/Q = {ref['R_over_Q_ohm']:.2f} Ω, "
                     f"T = {ref['T_transit']:.4f}, Epk/Eacc = {ref['Epk_Eacc']:.4f}, "
                     f"Bpk/Eacc = {ref['Bpk_Eacc_mT_per_MVm']:.4f} mT/(MV/m)\n")
        lines.append("| h [m] | Ne | " + " | ".join(SHORT[k] for k in COLS) + " |")
        lines.append("|---|---|" + "---|" * len(COLS))
        for r in rows:
            if r["shape"] == shape:
                lines.append(f"| {r['h']:.4f} | {r['ne']} | " + " | ".join(f"{r['err'][k]:+.2%}" for k in COLS) + " |")
    od = orders(rows)
    if od:
        lines.append("\nObserved order p (|err| ∝ h^p between the two finest meshes, h ∝ Ne^(−1/3); – = sign change; indicative only):\n")
        lines.append("| case | " + " | ".join(SHORT[k] for k in COLS) + " |")
        lines.append("|---|" + "---|" * len(COLS))
        for key in sorted(od):
            lines.append(f"| {key} | " + " | ".join("–" if np.isnan(od[key][k]) else f"{od[key][k]:.1f}"
                                                                 for k in COLS) + " |")
    lines.append("\nSurface-peak methods (relative error of Epk/Eacc for 'E …', of Bpk/Eacc for 'B …'; "
                 "default = E centroid + B recovered, projected):\n")
    lines.append("| case | h [m] | Ne | " + " | ".join(m[0] for m in METHODS) + " |")
    lines.append("|---|---|---|" + "---|" * len(METHODS))
    for r in peak_rows:
        lines.append(f"| {r['shape']} | {r['h']:.4f} | {r['ne']} | "
                     + " | ".join(f"{r['pk'][m[0]]:+.2%}" for m in METHODS) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h", type=float, nargs="+", default=None,
                    help="mesh sizes [m] for every shape (default: per shape, H_DEFAULT)")
    ap.add_argument("--shapes", nargs="+", default=["pillbox", "box"])
    ap.add_argument("--md", default=None, help="also write the markdown tables here")
    a = ap.parse_args(argv)
    rows, peak_rows = run(a.h, a.shapes)
    md = markdown(rows, peak_rows)
    print(md)
    if a.md:
        with open(a.md, "w") as fo:
            fo.write(md + "\n")


if __name__ == "__main__":
    main()
