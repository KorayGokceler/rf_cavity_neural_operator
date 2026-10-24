"""Figures of the preprint (docs/35 check 4): PDF, 6.5 in text width, 8-9 pt text, Okabe-Ito colours, no titles.

    field_modes            |E| on the y = 0 plane for modes 0 and 1 of an HF test geometry (stored p2 field,
                           evaluated with NGSolve on the rebuilt curved mesh)
    equal_cost_per_geometry  grouped bars: mean |f error| per test geometry for the equal-cost arms A, B2, C2
                           (numbers of docs/34 section 8.2, typed in below)
    cost_accuracy          time vs frequency error (log-log) of direct FE solves (check_baseline_coarse_fe.py) and of
                           the model (arms A / C2)

    python scripts/paper/make_figures.py field_modes --data hf_test.h5 --key sample_0024 --out paper/figures
    python scripts/paper/make_figures.py equal_cost_per_geometry --out paper/figures
    python scripts/paper/make_figures.py cost_accuracy --fe coarse_fe_all.json --out paper/figures
"""
import argparse
import json
import os
import sys

import torch  # noqa: F401  (before ngsolve: tests/conftest.py)

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Okabe-Ito (colour-blind safe)
BLUE, ORANGE, GREEN, VERMILION, SKY, PURPLE, YELLOW, BLACK = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#F0E442", "#000000")
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7})
WIDTH = 6.5

# docs/34 section 8.2: mean |f error| [%] per test geometry (ids 24-29, modes 0-3)
EQUAL_COST = {"A": [0.74, 1.18, 1.24, 4.44, 0.92, 1.46],
              "B2": [0.76, 1.33, 1.34, 3.59, 1.04, 1.24],
              "C2": [0.66, 1.23, 1.25, 2.93, 0.98, 1.17]}
EQUAL_COST_LABEL = {"A": "A: HF 8 (1242 s)", "B2": "B2: LF 24 (997 s)", "C2": "C2: LF 40 (1704 s)"}
EQUAL_COST_MEAN = {"A": 1.66, "B2": 1.55, "C2": 1.37}          # mean |f error| over 24 values [%]
MODEL_TIME_S = 8.3                                              # operators 5 s + forward 3.3 s (docs/34 section 8.2)


def fig_field_modes(args):
    from ngsolve import CF, GridFunction, HCurl
    from ngsolve import x as X, y as Y, z as Z  # noqa: F401
    from src.data_gen import highorder as ho
    from src.data_gen.field_labels import read_field_group
    with h5py.File(args.data, "r") as f:
        mesh, rec = read_field_group(f[args.key])
    pts_v = mesh.ngmesh.Coordinates()
    lo, hi = pts_v.min(0), pts_v.max(0)
    nz, nx = 700, 360
    zz = np.linspace(lo[2], hi[2], nz)
    xx = np.linspace(lo[0], hi[0], nx)
    ZZ, XX = np.meshgrid(zz, xx)
    P = np.column_stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel()])
    fes = HCurl(mesh, order=rec["model_order"], dirichlet="wall")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True, constrained_layout=True)
    for ax, mode in zip(axes, args.modes):
        gf = GridFunction(fes)
        gf.vec.FV().NumPy()[:] = rec["u_model"][:, mode]
        val, inside = ho.eval_field(mesh, CF(gf), P)
        mag = np.linalg.norm(val, axis=1)
        mag = np.where(inside, mag / mag[inside].max(), np.nan).reshape(XX.shape)
        im = ax.pcolormesh(ZZ * 1e3, XX * 1e3, mag, cmap="viridis", vmin=0, vmax=1, shading="auto", rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlabel(f"$z$ [mm]\nmode {mode}, $f$ = {rec['freqs'][mode]:.3f} GHz")
    axes[0].set_ylabel("$x$ [mm]")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label(r"$|\mathbf{E}|\,/\,\max|\mathbf{E}|$")
    return fig


def fig_equal_cost(args):
    ids = [24, 25, 26, 27, 28, 29]
    style = {"A": (ORANGE, ""), "B2": (SKY, "//"), "C2": (BLUE, "..")}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.5), constrained_layout=True)
    w = 0.26
    for i, (arm, vals) in enumerate(EQUAL_COST.items()):
        col, hatch = style[arm]
        ax.bar(np.arange(6) + (i - 1) * w, vals, w * 0.92, color=col, hatch=hatch, edgecolor="white" if not hatch else BLACK,
               linewidth=0.5, label=EQUAL_COST_LABEL[arm])
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels([str(i) for i in ids])
    ax.set_xlabel("test geometry id")
    ax.set_ylabel(r"mean $|f/f_{\mathrm{HF}}-1|$ [%]")
    ax.yaxis.grid(True, linewidth=0.4, color="#cccccc")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    return fig


def fig_cost_accuracy(args):
    d = json.load(open(args.fe))
    n_all = max(s["n_geo"] for s in d["summary"])          # rungs that exist for every geometry only
    S = [s for s in d["summary"] if s["settings"] == args.settings and s["n_geo"] == n_all]
    fig, ax = plt.subplots(figsize=(WIDTH * 0.8, 3.0), constrained_layout=True)
    series = {}
    for s in S:
        series.setdefault((s["order"], "model" if s["level"] == "model" else "coarse"), []).append(s)
    sty = {(1, "model"): (VERMILION, "o", "p1, model mesh"), (2, "model"): (BLUE, "s", "p2, model mesh (LF label)"),
           (1, "coarse"): (VERMILION, "^", "p1, coarser mesh"), (2, "coarse"): (BLUE, "v", "p2, coarser mesh")}
    for key, rows in series.items():
        col, mk, lab = sty[key]
        ax.plot([r["t_total"] for r in rows], [100 * r["err_mean"] for r in rows], mk, color=col, ms=6.5,
                mfc=col if key[1] == "model" else "white", mew=1.2, ls="none", label=lab)
    ax.plot([MODEL_TIME_S] * 2, [EQUAL_COST_MEAN["A"], EQUAL_COST_MEAN["C2"]], "D", color=BLACK, ms=6, ls="none",
            label="model (arms A, C2)")
    ax.annotate("A", (MODEL_TIME_S, EQUAL_COST_MEAN["A"]), xytext=(5, 2), textcoords="offset points", fontsize=8)
    ax.annotate("C2", (MODEL_TIME_S, EQUAL_COST_MEAN["C2"]), xytext=(5, -9), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time per geometry, 1 thread [s]")
    ax.set_ylabel(r"mean $|f/f_{\mathrm{HF}}-1|$, modes 0--3 [%]")
    ax.grid(True, which="major", linewidth=0.4, color="#cccccc")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure", choices=["field_modes", "equal_cost_per_geometry", "cost_accuracy"])
    ap.add_argument("--data", help="field_modes: HF field-label H5")
    ap.add_argument("--key", default="sample_0024")
    ap.add_argument("--modes", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--fe", help="cost_accuracy: JSON of check_baseline_coarse_fe.py")
    ap.add_argument("--settings", default="gen", choices=["gen", "lean"])
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig = {"field_modes": fig_field_modes, "equal_cost_per_geometry": fig_equal_cost,
           "cost_accuracy": fig_cost_accuracy}[args.figure](args)
    path = os.path.join(args.out, f"{args.figure}.pdf")
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".pdf", ".png"), dpi=130)         # quick look only (not for the paper)
    print("wrote", path)


if __name__ == "__main__":
    main()
