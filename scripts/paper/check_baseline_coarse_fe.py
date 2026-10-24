"""Coarse-FE cost-accuracy baseline (referee items E7 / E8, docs/35 check 3b).

For the geometries of an HF test file, a direct eigensolve (highorder.solve_modes: the generator's solver,
shift-invert + kernel projection, 1 NGSolve thread) in cheaper spaces, timed in three parts and compared with the
stored HF frequencies:

    model mesh p1, model mesh p2 (= the LF label), and coarser curved meshes of the same BREP (netgen
    curvaturesafety / maxh ladder; rungs that fold elements are skipped) with p1 and p2.

Per solve: mesh generation (coarse rungs only), assembly (spaces, K, M, Kp, the shifted operator), factorisation +
ARPACK ("solve"), tets, DOFs, and the frequency error |f / f_HF - 1| for the lowest K modes.  Each solve is run with
the generator's settings (tol 1e-10, 4 extra Ritz vectors) and with lean settings (tol 1e-6, 1 extra).

    python scripts/paper/check_baseline_coarse_fe.py --data hf_test.h5 [--ids 0 1] [--json part.json]
    python scripts/paper/check_baseline_coarse_fe.py --merge part1.json part2.json [--json all.json]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402,F401  (before ngsolve: tests/conftest.py)

import h5py  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data_gen import highorder as ho  # noqa: E402
from src.data_gen.field_labels import DEFAULTS, read_field_group  # noqa: E402

# coarse rungs (curvaturesafety, geometry order), coarsest first; maxh is the stored model mesh's (the curvature
# sets the element size on these CAD faces: maxh x 1 and x 2 gave the same meshes).  Measured on test geometry 0:
# curvaturesafety <= 1.0 folds the curved tets for every maxh; 1.25 gives 30 % (curve 3) / 42 % (curve 2) fewer tets.
LADDER = [(1.1, 2), (1.1, 3), (1.25, 2), (1.25, 3)]
SETTINGS = {"gen": {"tol": 1e-10, "n_extra": 4}, "lean": {"tol": 1e-6, "n_extra": 1}}


def solve_row(mesh, p, K, f_hf, tag, level, t_mesh=0.0, **extra):
    rows = []
    for sname, kw in SETTINGS.items():
        t0 = time.perf_counter()
        r = ho.solve_modes(mesh, K, order=p, **kw)
        t_tot = time.perf_counter() - t0
        info = r["info"]
        f = r["f_hz"][:K] / 1e9
        err = f / f_hf - 1.0
        rows.append({"level": level, "order": p, "settings": sname, "n_tets": int(mesh.ne), "n_dof": info["ndof"],
                     "t_mesh": t_mesh, "t_assemble": info["t_assemble"], "t_solve": info["t_factor"] + info["t_eig"],
                     "t_factor": info["t_factor"], "t_eig": info["t_eig"], "t_total": t_tot,
                     "f": f.tolist(), "err": err.tolist(), **extra})
        print(f"{tag} {level:<10} p{p} {sname:<4} tets {mesh.ne:>6} dof {info['ndof']:>7} | mesh {t_mesh:5.1f}s "
              f"assemble {info['t_assemble']:6.1f}s solve {info['t_factor'] + info['t_eig']:6.1f}s | err "
              f"{' '.join(f'{100 * e:+7.3f}' for e in err)} %", flush=True)
    return rows


def summarise(rows):
    """Per (level, order, settings): mean over geometries of the times, tets, and mean / max of |err| over all values."""
    keys = sorted({(r["level"], r["order"], r["settings"]) for r in rows},
                  key=lambda k: (k[2], k[1], -np.mean([r["n_tets"] for r in rows if (r["level"], r["order"], r["settings"]) == k])))
    out = []
    print(f"\n{'level':<10} {'p':>2} {'set':<5} {'n_geo':>5} {'tets':>7} {'dof':>8} | {'mesh':>6} {'assemb':>7} {'solve':>7} "
          f"{'total':>7} [s] | {'mean|err|':>9} {'max|err|':>9} [%]")
    for k in keys:
        R = [r for r in rows if (r["level"], r["order"], r["settings"]) == k]
        e = np.abs(np.array([r["err"] for r in R]))
        s = {"level": k[0], "order": k[1], "settings": k[2], "n_geo": len(R),
             "n_tets": float(np.mean([r["n_tets"] for r in R])), "n_dof": float(np.mean([r["n_dof"] for r in R])),
             "t_mesh": float(np.mean([r["t_mesh"] for r in R])), "t_assemble": float(np.mean([r["t_assemble"] for r in R])),
             "t_solve": float(np.mean([r["t_solve"] for r in R])), "t_total": float(np.mean([r["t_total"] for r in R])),
             "err_mean": float(e.mean()), "err_median": float(np.median(e)), "err_max": float(e.max())}
        out.append(s)
        print(f"{s['level']:<10} {s['order']:>2} {s['settings']:<5} {s['n_geo']:>5} {s['n_tets']:>7.0f} {s['n_dof']:>8.0f} | "
              f"{s['t_mesh']:6.2f} {s['t_assemble']:7.2f} {s['t_solve']:7.2f} {s['t_total']:7.2f}     | "
              f"{100 * s['err_mean']:9.4f} {100 * s['err_max']:9.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="HF field-label H5 (frequency reference + BREP + model mesh)")
    ap.add_argument("--ids", type=int, nargs="*", default=None, help="sample indices in the file (default: all)")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--no_coarse", action="store_true")
    ap.add_argument("--merge", nargs="+", default=None, help="merge --json outputs and print the summary")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.merge:
        rows = [r for p in args.merge for r in json.load(open(p))["rows"]]
        summ = summarise(rows)
        if args.json:
            json.dump({"rows": rows, "summary": summ}, open(args.json, "w"), indent=1)
        return
    import ngsolve
    ngsolve.SetNumThreads(1)
    rows = []
    with h5py.File(args.data, "r") as f:
        keys = sorted(k for k in f if k.startswith("sample_"))
        for i, key in enumerate(keys):
            if args.ids is not None and i not in args.ids:
                continue
            g = f[key]
            mesh, rec = read_field_group(g)
            f_hf = rec["freqs"][:args.K]
            maxh = float(g.attrs["maxh"])
            tag = key
            for p in args.orders:
                rows += solve_row(mesh, p, args.K, f_hf, tag, "model", key=key)
            if args.no_coarse:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                brep = os.path.join(tmp, "s.brep")
                with open(brep, "wb") as fh:
                    fh.write(np.asarray(g["brep"]).tobytes())
                for cs, cv in LADDER:
                    t0 = time.perf_counter()
                    try:
                        cm = ho.netgen_mesh(brep, maxh, curve=cv, retries=0,
                                            min_jacobian_ratio=DEFAULTS["min_jacobian_ratio"],
                                            fix_passes=DEFAULTS["fix_passes"], fix_below=DEFAULTS["fix_below"],
                                            curvaturesafety=cs, grading=DEFAULTS["grading"])
                    except (RuntimeError, ho.TooManyElements) as e:
                        print(f"{tag} coarse cs={cs} curve={cv}: skipped ({str(e)[:60]})", flush=True)
                        continue
                    t_mesh = time.perf_counter() - t0
                    if cm.ne >= 0.9 * mesh.ne:
                        print(f"{tag} coarse cs={cs} curve={cv}: {cm.ne} tets, not coarser than the model mesh, skipped",
                              flush=True)
                        continue
                    for p in args.orders:
                        rows += solve_row(cm, p, args.K, f_hf, tag, f"cs{cs}/c{cv}", t_mesh=t_mesh, key=key)
    summ = summarise(rows)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "summary": summ}, fh, indent=1)


if __name__ == "__main__":
    main()
