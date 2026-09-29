"""Label-free active sampling: pick the geometries the model is worst on, BEFORE solving them.

Candidates (generator ids, same families / sampling / deformation flags as the training data) are
only meshed (no eigen-solve, ~5× cheaper), the model predicts their eigenpairs, and each is scored
by the relative residual of the predicted pairs on the candidate's own N0 operators:

    η_k = ‖K u_k − λ_k M u_k‖_{D⁻¹} / λ_k ,   ‖u_k‖_M = 1,  D = diag(M)

(η = 0 for an exact discrete eigenpair; eigenvector error ≲ η / relative spectral gap).  The top
--n_select ids are written to --out; label them with
    python src/data_gen/dataset_generator_3d.py --ids_file ids.txt ... --h5_filename active.h5

    python scripts/active_sampling.py --checkpoint DIR --n_candidates 2000 --n_select 300 \
        --start_id 100000 --seed 0 --mesh_size 0.10 [--sampling sobol --deform_prob 0.5] --out ids.txt
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.data_gen.dataset_generator_3d as gen                      # noqa: E402
from src.data.dataset_3d import item_from_geometry, maxwell3d_collate  # noqa: E402
from src.data.dataset_converter_3d import extract_geometry_3d           # noqa: E402


def residual_indicator(K, M, U, lam):
    """η_k for eigenpairs (λ_k, u_k), U [N, k] with unit M-norm columns, scipy K and M."""
    R = K @ U - (M @ U) * lam[None, :]
    return np.sqrt((R ** 2 / M.diagonal()[:, None]).sum(0)) / np.abs(lam)


def _mesh(s_id):
    try:
        g = gen.mesh_sample(s_id)
        return s_id, g["nodes"], g["tets"], g["shape_type"]
    except Exception as e:           # a candidate that cannot be meshed is simply skipped
        print(f"candidate {s_id}: {e}")
        return s_id, None, None, None


def score(lm, s_id, nodes, tets, shape_type, device="cpu", feature_indices=None):
    geom, _ = extract_geometry_3d(nodes, tets)
    geom["shape_type"] = shape_type
    item = item_from_geometry(geom, feature_indices=feature_indices, g_id=s_id)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in maxwell3d_collate([item]).items()}
    with torch.no_grad():
        out = lm.model(batch)
    U = out["field"][0].double().cpu().numpy()
    lam = out["eigenvalues"][0].double().cpu().numpy()
    eta = residual_indicator(item["K"], item["M"], U, lam)
    return {"id": s_id, "shape_type": shape_type, "n_edges": len(geom["edges"]), "eta_mean": float(eta.mean()),
            "eta_max": float(eta.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_candidates", type=int, default=1000)
    ap.add_argument("--n_select", type=int, default=200)
    ap.add_argument("--start_id", type=int, default=100000, help="candidate ids: start_id … (keep clear of training ids)")
    ap.add_argument("--out", default="active_ids.txt")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--n_workers", type=int, default=os.cpu_count())
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args, gen_argv = ap.parse_known_args()   # remaining flags go to the generator (--families, --mesh_size, ...)
    gen.ARGS = gen.parse_args(gen_argv + ["--start_id", str(args.start_id), "--n_total", str(args.n_candidates)])

    from infer import resolve_checkpoint
    from src.training.lightning_module import GNOTLightning
    lm = GNOTLightning.load_from_checkpoint(resolve_checkpoint(args.checkpoint), map_location=args.device).eval()
    lm.freq_stats = {"mean": 0.0, "std": 1.0}          # GHz scaling is irrelevant for the residual
    fi = (lm.hparams.get("data_cfg") or {}).get("feature_indices")
    ids = range(args.start_id, args.start_id + args.n_candidates)
    rows = []
    with Pool(args.n_workers, initializer=gen._init_worker, initargs=(gen.ARGS,), maxtasksperchild=50) as pool:
        for s_id, nodes, tets, st in pool.imap_unordered(_mesh, ids):
            if nodes is not None:
                rows.append(score(lm, s_id, nodes, tets, st, args.device, fi))
    rows.sort(key=lambda r: -r["eta_mean"])
    with open(args.out, "w") as f:
        f.write("\n".join(str(r["id"]) for r in rows[:args.n_select]) + "\n")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    sel = rows[:args.n_select]
    fams = {}
    for r in sel:
        fams[r["shape_type"]] = fams.get(r["shape_type"], 0) + 1
    print(f"{len(rows)} candidates scored; selected {len(sel)} → {args.out}")
    print(f"η_mean: all {np.mean([r['eta_mean'] for r in rows]):.3g}, selected {np.mean([r['eta_mean'] for r in sel]):.3g}")
    print("selected per family:", fams)


if __name__ == "__main__":
    main()
