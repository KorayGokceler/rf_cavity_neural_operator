"""Evaluate an eigenspace3d checkpoint (EigenspaceOperator3D) on a 3D Maxwell PKL.

Per geometry: per-mode M-norm rel-L2 of the projected Ritz fields (sign-agnostic,
subspace error inside near-degenerate clusters, NaN for a cluster split by the
last output), predicted / true frequency [GHz] and relative error, span rel-L2 of
every stored target from span(PV) (mass and curl–curl norms), the basis'
gradient mass fraction.  Writes a CSV, prints the aggregate and a shape_type
breakdown.

    python scripts/eval_3d.py --checkpoint ckpt --data_path data/maxwell3d.pkl \
        [--split val|test|train|all] [--csv out.csv] [--batch_size 2]
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.dataset_3d import Maxwell3DDataset, maxwell3d_collate   # noqa: E402
from src.models.hcurl import hcurl_grams, mode_rel_l2                  # noqa: E402
from src.training.lightning_module import GNOTLightning, span_residual  # noqa: E402
from infer import resolve_checkpoint                                      # noqa: E402


def _to(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(lm, dataset, device='cpu', batch_size=2):
    """Per-geometry metric rows (dicts) and seconds per forward batch."""
    lm.eval().to(device)
    fs = lm.freq_stats
    ghz = (lambda z: z * fs['std'] + fs['mean']) if fs else (lambda z: z)   # noqa: E731
    rows, t_fwd = [], []
    for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=maxwell3d_collate):
        batch = _to(batch, device)
        t0 = time.perf_counter()
        out = lm(batch)
        t_fwd.append(time.perf_counter() - t0)
        V, T = out['basis'], batch['Y_field']
        m, K = V.shape[-1], out['field'].shape[-1]
        G_M, G_A, MT = hcurl_grams(V, T, out, batch)
        sM, sA = span_residual(G_M, m).sqrt(), span_residual(G_A, m).sqrt()
        gfrac = 1 - torch.diagonal(out['M_div'], dim1=-2, dim2=-1) / torch.diagonal(out['M_V'], dim1=-2, dim2=-1)
        fp, ft = ghz(out['freq']), ghz(batch['Y_freq'][:, :K])
        for b in range(V.shape[0]):
            inside, split = lm._clusters_3d(batch, b, K)
            rl = mode_rel_l2(out['field'][b].double(), T[b, :, :K].double(), MT[b, :, :K], inside)
            rl[split] = float('nan')
            row = {'geom_id': int(batch['geom_id'][b]), 'shape_type': batch['shape_type'][b],
                   'n_vertices': int(batch['Mask'][b].sum()), 'n_edges': int(batch['EdgeMask'][b].sum()),
                   'grad_frac': float(gfrac[b].mean())}
            for k in range(K):
                row.update({f'rel_l2_{k}': float(rl[k]), f'f_pred_{k}': float(fp[b, k]),
                            f'f_true_{k}': float(ft[b, k]),
                            f'f_rel_err_{k}': float((fp[b, k] - ft[b, k]).abs() / ft[b, k].abs())})
            for k in range(T.shape[-1]):
                row.update({f'span_M_{k}': float(sM[b, k]), f'span_A_{k}': float(sA[b, k])})
            row['rel_l2'] = float(np.nanmean([row[f'rel_l2_{k}'] for k in range(K)]))
            row['f_rel_err'] = float(np.mean([row[f'f_rel_err_{k}'] for k in range(K)]))
            row['f_mae_ghz'] = float(np.mean([abs(row[f'f_pred_{k}'] - row[f'f_true_{k}']) for k in range(K)]))
            row['span_M'] = float(sM[b].mean())
            row['span_A'] = float(sA[b].mean())
            rows.append(row)
    return rows, float(np.mean(t_fwd)) if t_fwd else float('nan')


SUMMARY = ('rel_l2', 'f_rel_err', 'f_mae_ghz', 'span_M', 'span_A', 'grad_frac')


def summarize(rows):
    """{group: {metric: mean}} for 'all' and every shape_type."""
    groups = defaultdict(list)
    for r in rows:
        groups['all'].append(r)
        groups[r['shape_type']].append(r)
    return {g: {'n': len(rs), **{k: float(np.nanmean([r[k] for r in rs])) for k in SUMMARY}}
            for g, rs in groups.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', required=True, help='.ckpt file or a training dir')
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test', 'all'])
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    ckpt = resolve_checkpoint(args.checkpoint)   # .ckpt or a training dir (best / last)
    print(f"Checkpoint: {ckpt}")
    lm = GNOTLightning.load_from_checkpoint(ckpt, map_location=args.device)
    dc = dict(lm.hparams.get('data_cfg') or {})
    kw = dict(random_seed=dc.get('random_seed', 42), feature_indices=dc.get('feature_indices'))
    if args.split == 'all':
        ds = Maxwell3DDataset(args.data_path, split='test', train_ratio=0.0, val_ratio=0.0, **kw)
    else:
        ds = Maxwell3DDataset(args.data_path, split=args.split, train_ratio=dc.get('train_ratio', 0.8),
                              val_ratio=dc.get('val_ratio', 0.1), **kw)
    lm.freq_stats = ds.stats
    rows, t = evaluate(lm, ds, args.device, args.batch_size)
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv} ({len(rows)} geometries)")
    print(f"forward: {t:.3f} s / batch of {args.batch_size} on {args.device}")
    print(f"{'group':<14}{'n':>4}" + ''.join(f"{k:>12}" for k in SUMMARY))
    for g, s in sorted(summarize(rows).items(), key=lambda kv: kv[0] != 'all'):
        print(f"{g:<14}{s['n']:>4}" + ''.join(f"{s[k]:>12.4g}" for k in SUMMARY))


if __name__ == '__main__':
    main()
