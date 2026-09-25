"""Run inference over the ENTIRE validation split (no sample cap).

Unlike infer.py (which stops after --num_samples geometries and only makes
plots), this evaluates every geometry in the split and reports aggregate
metrics, a per-geometry CSV, and an optional .npz dump of all
predictions/targets/coords for offline inspection.

Metrics (per geometry, then aggregated):
  - per-mode sign-agnostic relative-L2          (matches the training metric)
  - the same re-weighted by node_area           (M-norm proxy, mesh-invariant)
  - near-degenerate subspace projection error   (rotation-invariant pair error)
  - frequency MAE in GHz (per mode + overall)
  - field R^2 over all valid nodes (sign-aligned)

Usage:
    python scripts/infer_val_all.py --checkpoint path/to.ckpt \
        [--data_path data/gnot_dataset_5k.pkl] [--split val|train|test|all] \
        [--csv val_metrics.csv] [--dump_npz val_preds.npz] [--json out.json]
"""
import os
import sys
import csv
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.dataset import gnot_collate_fn
from infer import (_near_degenerate_clusters, plot_geometry_comparison,
                   _near_degenerate_note, _ot_match_np, build_dataset,
                   load_model, rescale_to_target)
from scripts.diagnose_data_floor import _sign_agnostic_rel_l2, _subspace_rel_l2_w


def _load_elements_pool(data_path):
    """geom_id -> triangle connectivity, for FEM-correct triangulation."""
    pool = {}
    p = str(data_path)
    try:
        if p.endswith('.pkl'):
            import pickle
            with open(data_path, 'rb') as f:
                raw = pickle.load(f)
            for gid, g in raw['geometry_pool'].items():
                if 'elements' in g:
                    pool[int(gid)] = np.asarray(g['elements'])
        elif p.endswith('.h5'):
            import h5py
            with h5py.File(data_path, 'r') as f:
                for gid in f['geometry_pool']:
                    grp = f['geometry_pool'][gid]
                    if 'elements' in grp:
                        pool[int(gid)] = grp['elements'][:]
    except Exception as e:
        print(f"  (warning: could not load mesh elements: {e})")
    return pool


def evaluate_split(model, dataset, device, batch_size, deg_threshold,
                   dump_rows, plot_dir=None, elements_pool=None,
                   split_name='', geom_pool=None):
    """Run the model over every geometry in `dataset`; return (agg, per_geom).

    If `plot_dir` is set, also save a GT | Prediction | Error figure for
    EVERY geometry (no cap), one PNG per geometry.
    """
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    collate_fn=gnot_collate_fn)
    fs = getattr(model, 'freq_stats', None)
    seen = set()
    per_geom = []
    K = None
    # global accumulators for R^2 (sign-aligned residuals over all nodes)
    sse = 0.0
    sst = 0.0

    with torch.no_grad():
        for batch in dl:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            out = model(batch)
            P = out['field']                       # [B,N,K]
            k_out = P.shape[-1]
            T = batch['Y_field'][..., :k_out]      # [B,N,K] (eigenspace data may store > K)
            Mk = batch['Mask']                     # [B,N]
            IF = batch['Input_funcs']              # [B,N,val_dim] (12 = current converter; idx5 = node_area)
            gids = batch['geom_id'].squeeze(-1).cpu().numpy()
            ftn = batch['Y_freq'][:, :k_out]       # [B,K] normalised asc
            fpn = out.get('freq')                  # [B,K] normalised asc or None
            if fs:
                ft_ph = (ftn * fs['std'] + fs['mean']).cpu().numpy()
                fp_ph = ((fpn * fs['std'] + fs['mean']).cpu().numpy()
                         if fpn is not None else None)
            else:
                ft_ph = ftn.cpu().numpy()
                fp_ph = fpn.cpu().numpy() if fpn is not None else None

            B, N, k = P.shape
            if K is None:
                K = k
            for i in range(B):
                gid = int(gids[i])
                if gid in seen:
                    continue
                seen.add(gid)
                m = Mk[i].cpu().numpy().astype(bool)
                Eh = P[i][m].cpu().numpy()                    # [Nv,K]
                Et = T[i][m].cpu().numpy()
                w = np.clip(IF[i][m, 5].cpu().numpy().astype(np.float64),
                            1e-8, None)
                nv = int(m.sum())

                # ── Slot-to-mode alignment ──────────────────────────────────
                # GNOT: Hungarian OT matching (slot ordering not guaranteed).
                # SpectralNO / eigenspace: eigh returns sorted eigenvalues → identity perm.
                _model_type = getattr(model, 'model_type', 'gnot')
                if (_model_type in ('spectral_no', 'eigenspace')
                        or getattr(model.model, 'ritz_basis', 0)):
                    perm = np.arange(K, dtype=np.int64)  # identity (eigh order)
                    fp_ph_i = fp_ph[i] if fp_ph is not None else None
                else:
                    fp_norm_i = (fpn[i].cpu().numpy()
                                 if fpn is not None else ftn[i].cpu().numpy())
                    ft_norm_i = ftn[i].cpu().numpy()
                    freq_w = getattr(model, 'freq_match_weight', 0.5) if fpn is not None else 0.0
                    perm = _ot_match_np(fp_norm_i, ft_norm_i, Eh, Et, freq_w=freq_w)
                    fp_ph_i = (fp_ph[i][perm] if fp_ph is not None else None)
                Eh = Eh[:, perm]                              # aligned to target mode order
                if getattr(model, 'scale_invariant_field', False):
                    Eh = rescale_to_target(Eh, Et)            # amplitude is a gauge

                shape_type = 'unknown'
                if geom_pool and gid in geom_pool:
                    shape_type = geom_pool[gid].get('shape_type', 'unknown')
                row = {'geom_id': gid, 'n_nodes': nv, 'shape_type': shape_type,
                       'topology': 'holed' if '_hole' in shape_type or shape_type == 'annulus' else 'simple',
                       # smallest relative gap (f_{k+1}-f_k)/f_k among the K true modes
                       'min_rel_gap': float(np.min(np.diff(ft_ph[i]) / ft_ph[i][:-1])) if K > 1 else float('nan')}
                rl_uni, rl_w = [], []
                signs = np.ones(K)
                for kk in range(K):
                    ru = _sign_agnostic_rel_l2(Eh[:, kk], Et[:, kk])
                    rw = _sign_agnostic_rel_l2(Eh[:, kk], Et[:, kk], w)
                    rl_uni.append(ru)
                    rl_w.append(rw)
                    row[f'mode{kk}_relL2'] = ru
                    row[f'mode{kk}_relL2_areaw'] = rw
                    # sign-aligned R^2 contribution
                    s = -1.0 if (np.sum((Eh[:, kk] + Et[:, kk]) ** 2)
                                 < np.sum((Eh[:, kk] - Et[:, kk]) ** 2)) else 1.0
                    signs[kk] = s
                    sse += float(np.sum((s * Eh[:, kk] - Et[:, kk]) ** 2))
                    sst += float(np.sum((Et[:, kk] - Et[:, kk].mean()) ** 2))

                clusters = _near_degenerate_clusters(ft_ph[i], deg_threshold)
                is_deg = any(len(c) > 1 for c in clusters)
                deg_err = []
                for cl in clusters:
                    if len(cl) > 1:
                        for kk in cl:
                            deg_err.append(
                                _subspace_rel_l2_w(Eh[:, cl], Et[:, kk]))
                row['is_near_degenerate'] = int(is_deg)
                row['near_deg_subspace_relL2'] = (
                    float(np.mean(deg_err)) if deg_err else float('nan'))

                if fp_ph_i is not None:
                    fmae = np.abs(fp_ph_i - ft_ph[i])
                    for kk in range(K):
                        row[f'mode{kk}_freq_true_ghz'] = float(ft_ph[i][kk])
                        row[f'mode{kk}_freq_pred_ghz'] = float(fp_ph_i[kk])
                        row[f'mode{kk}_freq_abserr_ghz'] = float(fmae[kk])
                    row['freq_mae_ghz'] = float(fmae.mean())
                else:
                    row['freq_mae_ghz'] = float('nan')

                row['mean_relL2'] = float(np.mean(rl_uni))
                row['mean_relL2_areaw'] = float(np.mean(rl_w))
                per_geom.append(row)

                coords = batch['X'][i][m].cpu().numpy()
                if dump_rows is not None:
                    dump_rows[gid] = {
                        'coords': coords,
                        'pred': Eh, 'target': Et,
                        'freq_true_ghz': ft_ph[i],
                        'freq_pred_ghz': (fp_ph_i if fp_ph_i is not None
                                          else np.full(K, np.nan)),
                    }

                if plot_dir is not None:
                    modes = {}
                    for kk in range(K):
                        fp_v = float(fp_ph_i[kk]) if fp_ph_i is not None \
                            else float('nan')
                        flip = ' (sign-flipped)' if signs[kk] < 0 else ''
                        modes[kk] = {
                            'coords': coords,
                            'target': Et[:, kk],
                            'pred': signs[kk] * Eh[:, kk],
                            'f_true': float(ft_ph[i][kk]),
                            'f_pred': fp_v,
                            'err_val': rl_uni[kk],
                            'err_label': 'rel-L2',
                            'mode_label': f'Mode {kk}{flip}',
                        }
                    el = elements_pool.get(gid) if elements_pool else None
                    try:
                        if el is not None and int(np.max(el)) >= nv:
                            el = None      # indices don't match kept nodes
                        save_path = os.path.join(
                            plot_dir, f"{split_name}_geom_{gid:04d}.png")
                        plot_geometry_comparison(
                            gid, modes, save_path, el,
                            _near_degenerate_note(ft_ph[i]))
                    except Exception as e:
                        print(f"  (warning: plot failed for geom {gid}: {e})")

    def ms(key, rows):
        v = [r[key] for r in rows if not np.isnan(r.get(key, np.nan))]
        return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (
            float('nan'), 0.0, 0)

    sep = [r for r in per_geom if not r['is_near_degenerate']]
    deg = [r for r in per_geom if r['is_near_degenerate']]
    agg = {
        'n_geoms': len(per_geom),
        'K': K,
        'overall_mean_relL2': ms('mean_relL2', per_geom),
        'overall_mean_relL2_areaw': ms('mean_relL2_areaw', per_geom),
        'per_mode_relL2': [ms(f'mode{k}_relL2', per_geom) for k in range(K or 0)],
        'per_mode_relL2_areaw': [ms(f'mode{k}_relL2_areaw', per_geom)
                                 for k in range(K or 0)],
        'well_separated_mean_relL2': ms('mean_relL2', sep),
        'near_degenerate_subspace_relL2': ms('near_deg_subspace_relL2', deg),
        'freq_mae_ghz': ms('freq_mae_ghz', per_geom),
        'field_R2': float(1.0 - sse / sst) if sst > 0 else float('nan'),
        # topology × smallest relative eigen-gap → (mean rel-L2, freq MAE)
        'breakdown': {f'{topo} gap {lo:g}-{hi:g}%': (ms('mean_relL2', b), ms('freq_mae_ghz', b))
                      for topo in ('simple', 'holed')
                      for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 1000))
                      for b in [[r for r in per_geom if r['topology'] == topo
                                 and lo <= 100 * r['min_rel_gap'] < hi]] if b},
    }
    return agg, per_geom


def _fmt(t):
    return f"{t[0]:.4f} ± {t[1]:.4f}  (n={t[2]})"


def print_agg(split, a):
    print("\n" + "=" * 68)
    print(f"  FULL-SPLIT INFERENCE   split={split}   geometries={a['n_geoms']}")
    print("=" * 68)
    print(f"  field R^2 (sign-aligned, all nodes) : {a['field_R2']:.4f}")
    print(f"  mean rel-L2  uniform                : "
          f"{_fmt(a['overall_mean_relL2'])}")
    print(f"  mean rel-L2  area-weighted (M-proxy): "
          f"{_fmt(a['overall_mean_relL2_areaw'])}")
    for k, (u, w) in enumerate(zip(a['per_mode_relL2'],
                                   a['per_mode_relL2_areaw'])):
        print(f"  mode {k}: uniform {_fmt(u)}   area-w {_fmt(w)}")
    print(f"  well-separated geoms  mean rel-L2   : "
          f"{_fmt(a['well_separated_mean_relL2'])}")
    print(f"  near-degenerate  subspace rel-L2    : "
          f"{_fmt(a['near_degenerate_subspace_relL2'])}")
    print(f"  frequency MAE (GHz)                 : "
          f"{_fmt(a['freq_mae_ghz'])}")
    print("  --- by topology × smallest relative eigen-gap (rel-L2 | freq MAE GHz) ---")
    for key, (rl, fm) in a.get('breakdown', {}).items():
        print(f"  {key:22s} rel-L2 {rl[0]:.4f} | freq {fm[0]:.4f}  (n={rl[2]})")
    print("=" * 68 + "\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data_path', default='data/gnot_dataset_5k.pkl')
    ap.add_argument('--split', default='val',
                    choices=['train', 'val', 'test', 'all'])
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--deg_threshold', type=float, default=0.05)
    ap.add_argument('--csv', default=None, help='per-geometry metrics CSV')
    ap.add_argument('--json', default=None, help='aggregate metrics JSON')
    ap.add_argument('--dump_npz', default=None,
                    help='dump all preds/targets/coords for offline plotting')
    ap.add_argument('--plot_dir', default=None,
                    help='save a GT|Pred|Error figure for EVERY geometry here')
    args = ap.parse_args()

    model, device = load_model(args.checkpoint)

    elements_pool = {}
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        elements_pool = _load_elements_pool(args.data_path)
        print(f"Plotting enabled -> {args.plot_dir}  "
              f"(mesh elements for {len(elements_pool)} geometries)")

    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    report = {}
    all_rows = []
    dump = {} if args.dump_npz else None
    for sp in splits:
        ds = build_dataset(model, args.data_path, sp)   # same split/features as training
        if getattr(ds, 'stats', None):
            model.freq_stats = ds.stats
        geom_pool = getattr(ds, 'geometry_pool', None)
        agg, rows = evaluate_split(model, ds, device, args.batch_size,
                                   args.deg_threshold, dump,
                                   plot_dir=args.plot_dir,
                                   elements_pool=elements_pool,
                                   split_name=sp,
                                   geom_pool=geom_pool)
        print_agg(sp, agg)
        report[sp] = agg
        for r in rows:
            r['split'] = sp
        all_rows.extend(rows)

    if args.csv and all_rows:
        keys = sorted({k for r in all_rows for k in r})
        with open(args.csv, 'w', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=keys)
            wtr.writeheader()
            wtr.writerows(all_rows)
        print(f"Per-geometry metrics -> {args.csv}  ({len(all_rows)} rows)")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Aggregate metrics -> {args.json}")

    if args.dump_npz and dump:
        np.savez_compressed(
            args.dump_npz,
            **{f'g{gid}_{k}': v for gid, d in dump.items()
               for k, v in d.items()})
        print(f"Predictions dump -> {args.dump_npz}  "
              f"({len(dump)} geometries)")


if __name__ == '__main__':
    main()
