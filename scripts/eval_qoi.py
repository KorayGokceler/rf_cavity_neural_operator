"""Cavity figures of merit of an eigenspace3d checkpoint vs FE (docs/24_CAVITY_QOI.md).

Per geometry of the split: src/viz/predict.py predict() (aligned Ritz fields, predicted and FE
frequencies, near-degenerate clusters / split pairs), then with ONE set of QoI operators
(src.qoi.build_qoi_operators of the stored mesh):

    pred  = qoi_from_dofs(ops, predicted field, f_pred)     (what the model delivers)
    true  = qoi_from_dofs(ops, FE field,        f_true)     (CST-style FE post-processing)

for Q0, G, R/Q, R_sh, T, Epk/Eacc, Bpk/Eacc (copper unless --Rs/--sigma, β = 1, 'linac').
Modes inside a near-degenerate cluster, or of a pair split by the last output, get NaN errors:
their individual value depends on the arbitrary rotation inside the eigenspace.  The
"accelerating mode" of a geometry = the isolated output with the largest TRUE R/Q.
In the summary the voltage-based QoI (R/Q, R_sh, T, Epk/Eacc, Bpk/Eacc) of the 'modes' selection
only count modes whose true R/Q ≥ --rq_floor × the geometry's max (non-accelerating modes have
V ≈ 0 and meaningless ratios); Q0, G and f count every isolated mode.
Stored PKL labels (samples[i]['qoi']), when present, are checked against the recomputed truth.
--deg_rel r replaces the model's cluster rule (near_deg_rel_threshold of the checkpoint, used by
eval_3d / predict) by "relative gap of the FE frequencies < r" (all stored modes + freq_next, so a
pair cut by the last output is still flagged); modes the model clusters but r does not are then
evaluated with their RAW Ritz column (no oracle rotation) — useful for multicell passbands, whose
modes (incl. the π mode) are a few 0.1 % apart and fall inside the training threshold (2 %).

    python scripts/eval_qoi.py --checkpoint ckpt_or_dir --data_path data/maxwell3d.pkl \\
        [--split test|val|train|all] [--csv qoi.csv] [--summary_csv qoi_summary.csv] [--max_geoms N]

CSV: one row per (geometry, output mode) — geom_id, shape_type, n_edges, mode, f_true_GHz,
f_pred_GHz, f_rel_err, rel_l2, degenerate, accel, accel_all (argmax true R/Q over ALL outputs,
degenerate or not), rq_frac, for each QoI q: q_true, q_pred, q_rel
(signed (pred − true)/|true|, NaN if degenerate) and q_label (stored, NaN if none), label_rel_diff,
t_ops_s (operator build, once per mesh), t_qoi_s (QoI of the K predicted modes).
"""
import argparse
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.dataset_converter_3d import QOI_LABEL_SETTINGS, qoi_operators_of   # noqa: E402

# QoI whose value involves the on-axis voltage (ill-defined for non-accelerating modes)
V_BASED = ('R_over_Q_ohm', 'R_sh_ohm', 'T_transit', 'Epk_Eacc', 'Bpk_Eacc_mT_per_MVm')
LABEL_TOL = 1e-4


def qoi_api():
    return importlib.import_module('src.qoi')


def degenerate_mask(K, clusters, split):
    """bool [K]: output mode inside a near-degenerate cluster (size > 1) or split by the last output."""
    deg = np.zeros(K, dtype=bool)
    sp_ = np.asarray(split).reshape(-1)
    if sp_.dtype == bool:
        deg[:len(sp_)] = sp_[:K]
    else:                                                   # index list
        deg[[int(k) for k in sp_ if int(k) < K]] = True
    for cl in clusters:
        if len(cl) > 1:
            deg[[k for k in cl if k < K]] = True
    return deg


def geometry_rows(ops, out, labels=None, settings=None, ops_time=float('nan')):
    """Long rows (one per output mode) of one predict() result with operators `ops`.
    labels: float [K_data, n_qoi] stored labels (columns = QOI_LABELS) or None."""
    q = qoi_api()
    names = tuple(q.QOI_LABELS)
    kw = dict(QOI_LABEL_SETTINGS if settings is None else settings)
    K = len(out['f_pred'])
    t0 = time.perf_counter()
    qp = q.qoi_from_dofs(ops, np.asarray(out['pred'], np.float64), np.asarray(out['f_pred']) * 1e9, **kw)
    t_qoi = time.perf_counter() - t0
    qt = q.qoi_from_dofs(ops, np.asarray(out['true'], np.float64), np.asarray(out['f_true']) * 1e9, **kw)
    qp = {n: np.asarray(qp[n], np.float64).reshape(-1) for n in names}
    qt = {n: np.asarray(qt[n], np.float64).reshape(-1) for n in names}
    deg = degenerate_mask(K, out['clusters'], out['split'])
    rq = qt['R_over_Q_ohm'] if 'R_over_Q_ohm' in qt else np.full(K, np.nan)
    iso_rq = np.where(~deg & np.isfinite(rq), rq, -np.inf)
    accel = int(np.argmax(iso_rq)) if np.isfinite(iso_rq).any() else -1
    all_rq = np.where(np.isfinite(rq), rq, -np.inf)
    accel_all = int(np.argmax(all_rq)) if np.isfinite(all_rq).any() else -1
    rq_max = iso_rq[accel] if accel >= 0 else np.nan
    ft, fp = np.asarray(out['f_true'], np.float64), np.asarray(out['f_pred'], np.float64)
    rows = []
    for k in range(K):
        r = {'geom_id': int(out['geom_id']), 'shape_type': str(out['shape_type']),
             'n_edges': int(len(out['edges'])), 'mode': k, 'f_true_GHz': float(ft[k]),
             'f_pred_GHz': float(fp[k]), 'f_rel_err': float((fp[k] - ft[k]) / abs(ft[k])),
             'rel_l2': float(out['rel_l2'][k]), 'degenerate': bool(deg[k]), 'accel': k == accel,
             'accel_all': k == accel_all,
             'rq_frac': float(rq[k] / rq_max) if np.isfinite(rq_max) and rq_max > 0 else float('nan')}
        diffs = []
        for j, n in enumerate(names):
            t_, p_ = float(qt[n][k]), float(qp[n][k])
            r[f'{n}_true'], r[f'{n}_pred'] = t_, p_
            r[f'{n}_rel'] = float('nan') if deg[k] or t_ == 0 else (p_ - t_) / abs(t_)
            lab = float(labels[k, j]) if labels is not None and k < len(labels) else float('nan')
            r[f'{n}_label'] = lab
            if np.isfinite(lab) and np.isfinite(t_):
                diffs.append(abs(lab - t_) / max(abs(t_), 1e-300))
        r['label_rel_diff'] = float(max(diffs)) if diffs else float('nan')
        r['t_ops_s'], r['t_qoi_s'] = float(ops_time), float(t_qoi)
        rows.append(r)
    return rows


def stored_labels(ds, g_id, names):
    """float [K_data, n_qoi] labels of geometry g_id in `names` order (dataset mode order) or None."""
    s_idx = ds.geom_to_samples[g_id]
    if not any('qoi' in ds.samples_metadata[j] for j in s_idx):
        return None
    nan = float('nan')
    return np.array([[float((ds.samples_metadata[j].get('qoi') or {}).get(n, nan)) for n in names]
                     for j in s_idx], dtype=np.float64)


def rel_gap_clusters(f_all, K, rel):
    """(clusters inside the K outputs, split indices) of ascending FE frequencies f_all (all stored
    modes, + freq_next if known): consecutive modes with gap < rel·f join a cluster."""
    f = [float(v) for v in f_all if np.isfinite(v)]
    clusters = [[0]] if f else []
    for i in range(1, len(f)):
        if abs(f[i] - f[i - 1]) < rel * abs(f[i - 1]):
            clusters[-1].append(i)
        else:
            clusters.append([i])
    inside = [c for c in clusters if c[-1] < K]
    split = [k for c in clusters if c[0] < K <= c[-1] for k in c if k < K]
    return inside, split


def regroup(lm, ds, idx, out, rel, device='cpu'):
    """predict() output re-clustered with the FE relative-gap rule (rel_gap_clusters); modes no
    longer degenerate get their raw Ritz column instead of the cluster-projected target."""
    import torch
    from src.data.dataset_3d import maxwell3d_collate
    from src.viz.predict import _to
    K = len(out['f_pred'])
    s_idx = ds.geom_to_samples[out['geom_id']]
    f_all = [float(ds.samples_metadata[j]['Theta'][1]) for j in s_idx]
    g = ds.geometry_pool[ds.samples_metadata[s_idx[0]]['geom_id']]
    f_next = g.get('freq_next', None)
    if f_next is not None and np.isfinite(f_next):
        f_all.append(float(f_next))
    inside, split = rel_gap_clusters(f_all, K, rel)
    old = degenerate_mask(K, out['clusters'], out['split'])
    new = degenerate_mask(K, inside, split)
    pred = np.array(out['pred'], dtype=np.float64, copy=True)
    redo = np.flatnonzero(old & ~new)
    if len(redo):
        with torch.no_grad():
            o = lm(_to(maxwell3d_collate([ds[idx]]), device))
        F = o['field'][0, :pred.shape[0]].double().cpu().numpy()
        pred[:, redo] = F[:, redo]                          # QoI are amplitude / sign invariant
    return dict(out, pred=pred, clusters=inside, split=new & np.isin(np.arange(K), split))


def evaluate_qoi(lm, ds, device='cpu', max_geoms=None, settings=None, n_axis=None, verbose=False,
                 deg_rel=None):
    """Long table (list of row dicts, see module docstring) over the dataset's geometries."""
    from src.viz.predict import _geom_of, predict
    names = tuple(qoi_api().QOI_LABELS)
    settings = dict(QOI_LABEL_SETTINGS if settings is None else settings)
    same = all(settings.get(k) == v for k, v in QOI_LABEL_SETTINGS.items()) and 'Rs' not in settings
    ext = {} if n_axis is None else {'n_axis': int(n_axis)}
    rows = []
    n = len(ds) if max_geoms is None else min(len(ds), int(max_geoms))
    for idx in range(n):
        out = predict(lm, ds, idx, device=device)
        if deg_rel is not None:
            out = regroup(lm, ds, idx, out, float(deg_rel), device)
        g_id = out['geom_id']
        geom = _geom_of(ds, g_id)
        g_key = ds.samples_metadata[ds.geom_to_samples[g_id][0]]['geom_id']
        t0 = time.perf_counter()
        ops = qoi_operators_of(geom, M=ds.operators(g_key)[0], **ext)
        t_ops = time.perf_counter() - t0
        labels = stored_labels(ds, g_id, names) if same and not ext else None
        r = geometry_rows(ops, out, labels, settings, t_ops)
        rows += r
        if verbose:
            print(f"  [{idx + 1}/{n}] geom {g_id} ({out['shape_type']}): ops {t_ops:.2f} s, "
                  f"QoI {r[0]['t_qoi_s'] * 1e3:.1f} ms")
    return rows


def summarize_qoi(rows, rq_floor=0.01):
    """DataFrame indexed by (group, qoi): n / median / mean |rel err| over 'modes' (isolated
    outputs; V-based QoI only with rq_frac ≥ rq_floor) and 'accel' (accelerating mode)."""
    df = pd.DataFrame(rows)
    names = [c[:-4] for c in df.columns if c.endswith('_rel') and c != 'f_rel']
    df['f_rel'] = df['f_rel_err'].where(~df['degenerate'].astype(bool))
    out = []
    for grp, d in [('all', df)] + sorted(df.groupby('shape_type'), key=lambda kv: str(kv[0])):
        iso = d[~d['degenerate'].astype(bool)]
        acc = d[d['accel'].astype(bool)]
        for n in ['f'] + names:
            sel = iso[iso['rq_frac'] >= rq_floor] if n in V_BASED else iso
            a, b = sel[f'{n}_rel'].abs().dropna(), acc[f'{n}_rel'].abs().dropna()
            out.append({'group': grp, 'qoi': n, 'n_modes': len(a),
                        'modes_median': a.median() if len(a) else np.nan,
                        'modes_mean': a.mean() if len(a) else np.nan, 'n_accel': len(b),
                        'accel_median': b.median() if len(b) else np.nan,
                        'accel_mean': b.mean() if len(b) else np.nan})
    return pd.DataFrame(out).set_index(['group', 'qoi'])


def print_summary(rows, summary):
    df = pd.DataFrame(rows)
    g = df.groupby('geom_id').first()
    print(f"{len(g)} geometries, {len(df)} output modes ({int(df['degenerate'].sum())} degenerate / split → NaN)")
    print(f"QoI post-processing per geometry: operators {g['t_ops_s'].median():.3f} s (median, once per mesh), "
          f"QoI of K={df['mode'].max() + 1} predicted modes {g['t_qoi_s'].median() * 1e3:.2f} ms")
    lab = df['label_rel_diff'].dropna()
    if len(lab):
        flag = 'OK' if lab.max() <= LABEL_TOL else f'MISMATCH (> {LABEL_TOL:g})'
        print(f"stored labels vs recomputed FE truth: max rel diff {lab.max():.2e} over {len(lab)} modes — {flag}")
    else:
        print("stored labels: none in this PKL (or other settings) — truth recomputed from the FE field")
    acc = df[df['accel'].astype(bool)]
    if len(acc):
        print(f"accelerating mode (argmax true R/Q of the isolated outputs): {len(acc)}/{len(g)} geometries, "
              f"mode index counts {acc['mode'].value_counts().to_dict()}, "
              f"true R/Q median {acc['R_over_Q_ohm_true'].median():.4g} Ω, Q0 {acc['Q0_true'].median():.4g}")
    hid = df[df['accel_all'].astype(bool) & df['degenerate'].astype(bool)]
    if len(hid):
        print(f"  {len(hid)} geometries: the largest-R/Q output is inside a near-degenerate cluster (excluded; "
              f"see --deg_rel)")
    print("|rel err| = |pred/true − 1| — modes: isolated outputs; accel: accelerating mode")
    with pd.option_context('display.float_format', '{:.3e}'.format, 'display.width', 160,
                           'display.max_rows', 500):
        print(summary.to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', required=True, help='.ckpt file or a training dir')
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test', 'all'])
    ap.add_argument('--csv', default=None, help='per (geometry, mode) rows')
    ap.add_argument('--summary_csv', default=None)
    ap.add_argument('--max_geoms', type=int, default=None)
    ap.add_argument('--rq_floor', type=float, default=0.01,
                    help="summary 'modes': V-based QoI only for true R/Q ≥ rq_floor × max of the geometry")
    ap.add_argument('--sigma', type=float, default=QOI_LABEL_SETTINGS['sigma'], help='wall conductivity [S/m]')
    ap.add_argument('--Rs', type=float, default=None, help='fixed surface resistance [Ω] (overrides --sigma)')
    ap.add_argument('--beta', type=float, default=QOI_LABEL_SETTINGS['beta'])
    ap.add_argument('--convention', default=QOI_LABEL_SETTINGS['convention'], choices=['linac', 'circuit'])
    ap.add_argument('--n_axis', type=int, default=None, help='axis points (default: build_qoi_operators)')
    ap.add_argument('--deg_rel', type=float, default=None,
                    help="degenerate = FE relative frequency gap < deg_rel (default: the model's clusters)")
    ap.add_argument('--device', default=None)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args(argv)
    import torch
    from src.viz.predict import load
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    lm, ds = load(args.checkpoint, args.data_path, split=args.split, device=device)
    settings = {'sigma': args.sigma, 'beta': args.beta, 'convention': args.convention}
    if args.Rs is not None:
        settings['Rs'] = args.Rs
    rows = evaluate_qoi(lm, ds, device=device, max_geoms=args.max_geoms, settings=settings,
                        n_axis=args.n_axis, verbose=args.verbose, deg_rel=args.deg_rel)
    if not rows:
        print("no geometries in this split")
        return [], None
    summary = summarize_qoi(rows, args.rq_floor)
    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"wrote {args.csv} ({len(rows)} rows)")
    if args.summary_csv:
        summary.to_csv(args.summary_csv)
        print(f"wrote {args.summary_csv}")
    print_summary(rows, summary)
    return rows, summary


if __name__ == '__main__':
    main()
