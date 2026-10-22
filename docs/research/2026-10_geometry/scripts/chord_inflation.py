"""Ratio of the stored acceleration length (axis chord inside the mesh, beam pipes included) to the
standard one (cell length; elliptical: sum of the cell lengths), drawn with the generator's own
parameter samplers.  docs/33 §2.2.  Run from the repository root:

    python docs/research/2026-10_geometry/scripts/chord_inflation.py [--n 200000] [--n_ell 5000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.data_gen import cavity_shapes as cs  # noqa: E402


class _NoOcc:
    """Stands in for gmsh's OCC kernel: the builders only need their parameter draws here."""

    def revolve(self, *args, **kwargs):
        return [(3, 1)]

    def __getattr__(self, name):
        return lambda *args, **kwargs: 1


def axisym(builder, n, rng):
    occ, out = _NoOcc(), np.empty(n)
    for i in range(n):
        _, p = builder(occ, rng)
        out[i] = (p["Lc"] + 2 * p["Lpipe"]) / p["Lc"]
    return out


def elliptical(n, n_cells, rng):
    out = np.empty(n)
    for i in range(n):
        p, h, _ = cs.draw_elliptical(rng, n_cells)
        k = int(p["n_cells"])
        cells = h["e1"]["L"] + h["e2"]["L"] + 2 * (k - 1) * h["mid"]["L"]
        out[i] = (cells + p["Lpipe1"] + p["Lpipe2"]) / cells
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200000, help="draws for pillbox_pipes and reentrant")
    ap.add_argument("--n_ell", type=int, default=5000, help="draws per elliptical cell count")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = [("pillbox_pipes", axisym(cs.build_pillbox_pipes, args.n, rng)),
            ("reentrant", axisym(cs.build_reentrant, args.n, rng))]
    rows += [(f"elliptical, {k} cell(s)", elliptical(args.n_ell, k, rng)) for k in (1, 2, 5)]
    rows.append(("elliptical, training mix", elliptical(args.n_ell, None, rng)))
    print(f"{'family':26s} {'median':>7s} {'p5':>6s} {'p95':>6s}")
    for name, r in rows:
        p5, p50, p95 = np.percentile(r, [5, 50, 95])
        print(f"{name:26s} {p50:7.2f} {p5:6.2f} {p95:6.2f}")


if __name__ == "__main__":
    main()
