"""Per-family evaluation of a field3d checkpoint (also projection: false checkpoints): is a bad prediction a
SUBSPACE problem (span(V) does not contain the modes) or a SELECTION problem (the modes are in span(V), the Ritz
step picks / orders / values them wrongly)?

For every geometry (read from field-label H5 files; the operators are rebuilt like the dataset does, from the stored
mesh and wall features) the model runs its projected forward (trunk -> V -> kernel projection -> Rayleigh-Ritz with
the K = num_field_modes lowest Ritz pairs), then

  span_M / span_A  sqrt(r_k): best-approximation rel-L2 error of each of the n stored target modes (all of them, not
                   only K) from the PROJECTED span PV, in the mass and in the energy norm (what the loss's span term
                   measures, but kernel-projected: no gradient content).  ~0 = the mode is in the learned subspace.
  span_plain       the same in the unprojected mass norm (= the training / validation quantity val/span_rel_l2).
  f_err            Ritz frequency / stored label frequency - 1 for the K Ritz outputs (percent in the table).  The label
                   is the FE eigenvalue of the model space itself (p2 labels), so f_err >= 0 up to the CG tolerance
                   (Ritz margin); negative values mean a label error (p3 labels) or a bug.
  field_l2         Ritz field rel-L2 per mode, degenerate clusters as a subspace (as test/mode_k_rel_l2); NaN = a
                   cluster cut by output K.
  assign           for each Ritz output the stored mode with the largest overlap |f_i^T M t_j|^2 / ||t_j||^2.  The
                   identity 0 1 2 ... (up to permutations inside degenerate clusters) means the right modes come out
                   in the right order; a gap (0 1 3 4 ...) means a mode was skipped, a repeat means two outputs found
                   the same mode.
  f_any            min_j |f_i / f_label_j - 1| over ALL stored labels: small while f_err is large = the output is a
                   true mode of the cavity but the wrong one (ordering / skipping), not an arbitrary frequency.
  grad_frac        gradient mass fraction of V, 1 - diag(M_div)/diag(V^T M V) (mean over columns).

Reading the table per family:
  span_A large                            -> the learned subspace misses the modes (generalisation of the trunk).
  span_A small, f_err / field_l2 large    -> the modes are in span(V) but Ritz selects wrongly (ordering, an extra
                                             low spurious combination, cluster split): look at assign / f_any.
  span_plain >> span_M, grad_frac large   -> V carries gradient content (wasted capacity, harmless after projection).

Examples:
  # in-distribution: the test split of the training run (same data list as in training; the split rule is rebuilt from
  # the checkpoint's seed / ratios, require_label_ok as in the config)
  python scripts/eval_by_family.py --ckpt best.ckpt --data /content/h5/{elliptical,reentrant,...} --split test
  # OOD families: every geometry of the files
  python scripts/eval_by_family.py --ckpt best.ckpt --data /content/h5/box /content/h5/coax_qw ... --split all --max_per_family 20
Add --json out.json for all per-geometry numbers, --per_geometry to print one line per geometry, --device cuda on a GPU.
"""
import argparse
import json
import os
import sys
import time

import torch  # noqa: F401  (before ngsolve: tests/conftest.py)

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import field_dataset as fd  # noqa: E402
from src.data_gen import highorder as ho  # noqa: E402
from src.data_gen.field_labels import model_normalisation, read_field_group  # noqa: E402
from src.models.hcurl import hcurl_grams, hcurl_grams_plain, mode_rel_l2  # noqa: E402
from src.training.lightning_module import CavityLightning, span_residual  # noqa: E402

C0 = 299792458.0


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def sample_list(paths, split, ckpt_cfg, require_label_ok, max_tets):
    """[(path, key)] of the geometries to evaluate: every sample of the files (split 'all'), or the geometries of one
    split of FieldDataset (same seeded permutation as in training when `paths` is the training data list)."""
    if split == "all":
        files = fd._expand(paths)
        out = []
        for p in files:
            with h5py.File(p, "r") as f:
                for k in sorted(k for k in f.keys() if k.startswith("sample_")):
                    if require_label_ok and int(f[k].attrs.get("label_ok", 1)) == 0:
                        continue
                    if max_tets is not None and int(f[k].attrs.get("n_tets", 0)) > max_tets:
                        continue
                    out.append((p, k))
        return out
    ds = fd.FieldDataset(paths, split=split, train_ratio=ckpt_cfg.get("train_ratio", 0.8),
                         val_ratio=ckpt_cfg.get("val_ratio", 0.1), random_seed=ckpt_cfg.get("random_seed", 42),
                         require_label_ok=require_label_ok, max_tets=max_tets, build_potential=False)
    return [ds.index[g] for g in ds.active_geoms]


FEATURES = ("x", "y", "z", "dist_to_boundary", "dir_bnd_x", "dir_bnd_y", "dir_bnd_z", "point_volume", "torsion")


def ablate_features(item, names, feature_indices):
    """Replace the named token features by their mass-weighted mean over the geometry's tokens (a feature-reliance
    probe at inference: if the numbers of the ID test set collapse, the model reads that feature)."""
    cols = list(range(len(FEATURES))) if feature_indices is None else list(feature_indices)
    w = item["Area"].double()
    for n in names:
        j = cols.index(FEATURES.index(n))
        f = item["Input_funcs"][:, j].double()
        item["Input_funcs"][:, j] = float((f * w).sum() / w.sum())


def evaluate(model, g, device, feature_indices=None, ablate=()):
    mesh, rec = read_field_group(g)
    center, scale = model_normalisation(mesh)
    ops = ho.field_operators(mesh, order=rec["model_order"], scale=scale, center=center, wall_feat=rec.pop("wall_feat"))
    ops = {k: v for k, v in ops.items() if k not in ("bary", "X", "tets")}
    K = model.num_field_modes
    n = rec["u_model"].shape[1]
    f_lab = rec["freqs"][:n]
    item = fd.field_item(ops, rec["u_model"][:, :n], f_lab, feature_indices, str(g.attrs.get("shape_type", "")), check=False)
    item["FreqNext"] = torch.tensor(float(rec["freq_next"]), dtype=torch.float32)
    if ablate:
        ablate_features(item, ablate, feature_indices)
    batch = to_device(fd.field_collate([item]), device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.model(batch)
        V, T = out["basis"], batch["Y_field"]
        m = V.shape[-1]
        G_M, G_A, MT = hcurl_grams(V, T, out, batch)                      # projected Grams of [PV | T]
        span_M = span_residual(G_M, m, model.span_ridge)[0].clamp(min=0).sqrt().cpu().numpy()
        span_A = span_residual(G_A, m, model.span_ridge)[0].clamp(min=0).sqrt().cpu().numpy()
        P_M, P_A, _ = hcurl_grams_plain(V, T, batch)                      # unprojected Grams (training quantity)
        P_M, P_A, _ = CavityLightning._normalise_columns(P_M, P_A, m)
        span_plain = span_residual(P_M, m, model.span_ridge)[0].clamp(min=0).sqrt().cpu().numpy()
        grad_frac = (1.0 - torch.diagonal(out["M_div"], dim1=-2, dim2=-1)
                     / torch.diagonal(out["M_V"], dim1=-2, dim2=-1).clamp(min=1e-300))[0]
        lam = out["eigenvalues"][0].double().cpu().numpy()
        f_pred = C0 * np.sqrt(np.maximum(lam, 0)) / (2 * np.pi * scale) / 1e9
        Fd, Tk, MTk = out["field"][0].double(), T[0].double(), MT[0].double()
        inside, split = model._clusters_3d(batch, 0, K)
        rl = mode_rel_l2(Fd, Tk[:, :K], MTk[:, :K], inside).cpu().numpy()
        rl[split] = np.nan
        Cm = (Fd.T @ MTk).cpu().numpy()                                    # [K, n]: f_i^T M t_j
        tt = (Tk * MTk).sum(0).clamp(min=1e-300).cpu().numpy()
        cap = Cm ** 2 / tt[None, :]
    t_fwd = time.perf_counter() - t0
    Kd = min(K, len(f_pred))
    fp, fl = f_pred[:Kd], f_lab[:Kd]
    f_any = np.abs(fp[:, None] / f_lab[None, :] - 1.0).min(1)
    return {"shape_type": str(g.attrs.get("shape_type", "")), "n_tets": int(g.attrs.get("n_tets", ops["qx"].shape[0])),
            "betti1": int(g.attrs.get("betti1", -1)), "n_bnd": int(g.attrs.get("n_bnd_components", -1)),
            "label_ok": int(g.attrs.get("label_ok", 1)), "scale": float(scale),
            "f_label": f_lab.tolist(), "f_pred": fp.tolist(), "f_err": (fp / fl - 1.0).tolist(),
            "f_any": f_any.tolist(), "field_l2": rl.tolist(),
            "span_M": span_M.tolist(), "span_A": span_A.tolist(), "span_plain": span_plain.tolist(),
            "assign": cap.argmax(1).tolist(), "cap_max": cap.max(1).tolist(),
            "grad_frac": float(grad_frac.mean()), "grad_frac_max": float(grad_frac.max()),
            "t_forward": t_fwd}


def _ms(a):
    a = np.asarray(a, float)
    return f"{np.nanmean(a):6.3f}/{np.nanmedian(a):6.3f}/{np.nanmax(a):6.3f}" if np.isfinite(a).any() else "   nan"


def summarize(rows, K):
    """Print the per-family table: mean over geometries of each per-mode quantity, plus summaries."""
    fams = sorted({r["shape_type"] for r in rows})
    n_modes = max(len(r["span_A"]) for r in rows)

    def mean_modes(key, scale=1.0, n=None):
        n = n or len(rows[0][key])
        return [np.nanmean([r[key][k] for r in rows_f if k < len(r[key])]) * scale for k in range(n)]

    for fam in fams:
        rows_f = [r for r in rows if r["shape_type"] == fam]
        f_lab = np.array([r["f_label"] for r in rows_f])
        allf = np.array([r["f_label"][0] for r in rows_f]), np.array([max(r["f_label"]) for r in rows_f])
        print(f"\n=== {fam}: {len(rows_f)} geometries | f_label[0] {allf[0].min():.2f}-{allf[0].max():.2f} GHz, "
              f"f_label[K-1] {f_lab[:, K - 1].min():.2f}-{f_lab[:, K - 1].max():.2f} GHz | tets {min(r['n_tets'] for r in rows_f)}"
              f"-{max(r['n_tets'] for r in rows_f)} | betti1 {sorted({r['betti1'] for r in rows_f})} "
              f"bnd comps {sorted({r['n_bnd'] for r in rows_f})}")
        hdr = "mode k:           " + " ".join(f"{k:>7d}" for k in range(n_modes))
        print(hdr)
        fmt = lambda name, v: print(f"{name:<18s} " + " ".join(f"{x:7.3f}" for x in v))   # noqa: E731
        fmt("span_M (proj)", mean_modes("span_M", n=n_modes))
        fmt("span_A (proj)", mean_modes("span_A", n=n_modes))
        fmt("span_plain", mean_modes("span_plain", n=n_modes))
        print("-- Ritz outputs (first K modes) --")
        fmt("f_err mean [%]", mean_modes("f_err", 100, K))
        fmt("|f_err| max [%]", [np.nanmax([abs(r["f_err"][k]) for r in rows_f if k < len(r["f_err"])]) * 100 for k in range(K)])
        fmt("f_any mean [%]", mean_modes("f_any", 100, K))
        fmt("field_l2", mean_modes("field_l2", n=K))
        # overall summaries (mean/median/max over all (geometry, mode) pairs of the K outputs)
        fe = np.abs(np.array([r["f_err"][:K] for r in rows_f])) * 100
        print(f"|f_err| [%] over geometries x K modes: mean {fe.mean():.3f} median {np.median(fe):.3f} max {fe.max():.3f} | "
              f"Ritz margin min(f_err) {np.min([min(r['f_err'][:K]) for r in rows_f]) * 100:+.3f} % | "
              f"field_l2 mean/median/max {_ms(np.array([r['field_l2'][:K] for r in rows_f]))} | "
              f"grad_frac mean {np.mean([r['grad_frac'] for r in rows_f]):.3f} max {np.max([r['grad_frac_max'] for r in rows_f]):.3f}")
        nan_share = np.mean(~np.isfinite(np.array([r["field_l2"][:K] for r in rows_f], float)))
        print(f"field_l2 undefined (NaN) for {100 * nan_share:.0f} % of the (geometry, output) pairs: the near-degenerate cluster "
              f"(2 % relative gap) continues past output K, so the K-th Ritz vector is an arbitrary mix of the cluster")
        ident = np.mean([len(set(r["assign"][:K])) == K and r["assign"][:K] == sorted(r["assign"][:K]) and r["assign"][0] == 0
                         for r in rows_f])
        skipped = np.mean([max(r["assign"][:K]) >= K for r in rows_f])
        dup = np.mean([len(set(r["assign"][:K])) < K for r in rows_f])
        print(f"mode assignment: ascending without repeats from mode 0 in {100 * ident:.0f} % of geometries; "
              f"a stored mode >= K is returned in {100 * skipped:.0f} % (modes skipped); a stored mode returned twice in {100 * dup:.0f} %")
        s_lo = np.mean([np.mean(np.array(r["span_A"][:K]) < 0.05) for r in rows_f])
        f_hi = np.mean([np.mean(np.abs(np.array(r["f_err"][:K])) > 0.01) for r in rows_f])
        both = np.mean([np.mean((np.array(r["span_A"][:K]) < 0.05) & (np.abs(np.array(r["f_err"][:K])) > 0.01)) for r in rows_f])
        print(f"modes with span_A < 5 % (in the subspace): {100 * s_lo:.0f} % | modes with |f_err| > 1 %: {100 * f_hi:.0f} % | "
              f"in the subspace AND |f_err| > 1 % (selection problem): {100 * both:.0f} %")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="field-label H5 files / globs / directories")
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"],
                    help="all: every geometry of the files (OOD sets); test/val/train: the dataset's seeded split of --data "
                         "(pass the SAME list as in training)")
    ap.add_argument("--max_per_family", type=int, default=0, help="at most this many geometries per family (0 = all)")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--keep_label_not_ok", action="store_true", help="also evaluate label_ok = 0 samples")
    ap.add_argument("--max_tets", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ablate", nargs="*", default=(), choices=FEATURES,
                    help="replace these token features by their per-geometry mean (inference-time reliance probe; e.g. "
                         "point_volume, which is mesh grading, not geometry)")
    ap.add_argument("--per_geometry", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    import ngsolve
    ngsolve.SetNumThreads(args.threads)
    model = CavityLightning.load_from_checkpoint(args.ckpt, map_location=args.device, strict=False).eval().to(args.device)
    cfg = dict(getattr(model.hparams, "data_cfg", None) or {})
    model.freq_stats = {"mean": 0.0, "std": 1.0}              # Y_freq / FreqNext are raw GHz here (cluster rule: relative gap)
    fi = cfg.get("feature_indices")
    print(f"checkpoint {args.ckpt}: projection={getattr(model, 'projection', None)} K={model.num_field_modes} "
          f"n_basis={model.model.n_basis} val_dim={model.model.val_dim} params={sum(p.numel() for p in model.parameters()):,} "
          f"| torch threads {torch.get_num_threads()}, ngsolve threads {ngsolve.GetNumThreads()}, device {args.device}")
    items = sample_list(args.data, args.split, cfg, not args.keep_label_not_ok, args.max_tets)
    # family of each sample from the attrs
    fam_of = {}
    for p, k in items:
        with h5py.File(p, "r") as f:
            fam_of[(p, k)] = str(f[k].attrs.get("shape_type", ""))
    seen, todo = {}, []
    for p, k in items:
        fam = fam_of[(p, k)]
        if args.families and fam not in args.families:
            continue
        seen[fam] = seen.get(fam, 0) + 1
        if args.max_per_family and seen[fam] > args.max_per_family:
            continue
        todo.append((p, k))
    print(f"{len(todo)} geometries in {len(set(fam_of[x] for x in todo))} families, split={args.split}")
    rows = []
    for i, (p, k) in enumerate(todo):
        with h5py.File(p, "r") as f:
            r = evaluate(model, f[k], args.device, fi, tuple(args.ablate))
        r.update(file=p, key=k)
        rows.append(r)
        if args.per_geometry:
            print(f"[{i + 1}/{len(todo)}] {r['shape_type']:<15s} {k} {r['n_tets']:>6d} tets | f_lab {r['f_label'][0]:.2f}.."
                  f"{r['f_label'][len(r['f_pred']) - 1]:.2f} GHz | f_err % " + " ".join(f"{100 * v:+6.2f}" for v in r["f_err"])
                  + " | span_A " + " ".join(f"{v:.2f}" for v in r["span_A"]) + " | assign " + "".join(str(a) if a < 10 else "+" for a in r["assign"]),
                  flush=True)
        elif (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(todo)}", flush=True)
    summarize(rows, model.num_field_modes)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
