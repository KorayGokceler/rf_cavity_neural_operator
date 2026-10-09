"""Cross-resolution test: one eigenspace3d checkpoint (trained at one mesh size) on the
SAME geometries meshed at several sizes (dataset_generator_3d with the same --seed /
--start_id and different --mesh_size → identical shapes, finer meshes).

Per resolution, over the geometries present at every resolution:
  field rel-L2    model vs the FE solution on that mesh (M-norm, as eval_3d)
  f err (own)     model vs FE on that mesh
  f err (ref)     model vs FE on the finest mesh (≈ converged reference)
  FE f err (ref)  FE on that mesh vs FE on the finest mesh: the discretisation error a
                  plain solve at that resolution would have — the bar to compare with
All frequency errors: mean over the K output modes of |f/f_ref − 1|.

    python scripts/resolution_study.py --checkpoint DIR --pkls p010.pkl p007.pkl p005.pkl \
        --labels 0.10 0.07 0.05 [--csv res.csv] [--plot res.png] [--by_shape]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.training.checkpoint import resolve_checkpoint                                   # noqa: E402
from scripts.eval_3d import evaluate                                   # noqa: E402
from src.data.dataset_3d import Maxwell3DDataset                       # noqa: E402
from src.training.lightning_module import CavityLightning                # noqa: E402

MODEL_C, FE_C = '#2a78d6', '#eb6834'     # categorical slots 1 / 2 (dataviz reference palette)


def study(lm, pkls, labels, device='cpu'):
    """Long table: one row per (mesh label, geometry) with the metrics of the docstring."""
    rows = []
    for pkl, lab in zip(pkls, labels, strict=True):
        ds = Maxwell3DDataset(pkl, split='test', train_ratio=0.0, val_ratio=0.0)
        lm.freq_stats = ds.stats
        r, _ = evaluate(lm, ds, device, batch_size=1)
        for row in r:
            row['mesh'] = float(lab)
        rows += r
    df = pd.DataFrame(rows)
    K = sum(c.startswith('f_pred_') for c in df.columns)
    ref_mesh = df['mesh'].min()
    common = set.intersection(*(set(g['geom_id']) for _, g in df.groupby('mesh')))
    df = df[df['geom_id'].isin(common)].copy()
    ref = df[df['mesh'] == ref_mesh].set_index('geom_id')
    fp = df[[f'f_pred_{k}' for k in range(K)]].to_numpy()
    ft = df[[f'f_true_{k}' for k in range(K)]].to_numpy()
    fr = ref.loc[df['geom_id'], [f'f_true_{k}' for k in range(K)]].to_numpy()
    df['f_err_own'] = np.abs(fp / ft - 1).mean(1)
    df['f_err_ref'] = np.abs(fp / fr - 1).mean(1)
    df['fe_f_err_ref'] = np.abs(ft / fr - 1).mean(1)
    return df


COLS = ['n_edges', 'rel_l2', 'f_err_own', 'f_err_ref', 'fe_f_err_ref']
PCT = ['f_err_own', 'f_err_ref', 'fe_f_err_ref']


def summarize(df, by_shape=False):
    """Means per resolution (and shape type); frequency errors in %."""
    keys = ['mesh', 'shape_type'] if by_shape else ['mesh']
    out = df.groupby(keys)[COLS].mean()
    out[PCT] *= 100
    out = out.rename(columns={c: c + ' [%]' for c in PCT})
    out.insert(0, 'n', df.groupby(keys).size())
    return out.sort_index(level=0, ascending=False)


def plot(summary, path):
    """Two single-axis charts: frequency error vs edges (model and plain FE, against the
    finest mesh) and model field error vs edges."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = summary.reset_index()
    ref_mesh = s['mesh'].min()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for ax in (a, b):
        ax.grid(True, which='both', color='#d8d8d8', linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.set_xscale('log')
        ax.set_xlabel('mean N0 edges per geometry')
    m = s['mesh'] > ref_mesh                                     # FE vs itself is 0 at the reference
    a.plot(s['n_edges'], s['f_err_ref [%]'], '-o', color=MODEL_C, lw=2, ms=8, label='model')
    a.plot(s['n_edges'][m], s['fe_f_err_ref [%]'][m], '-s', color=FE_C, lw=2, ms=8,
           label='FE at this mesh')
    a.set_yscale('log')
    a.set_ylabel(f'frequency error vs FE @ mesh {ref_mesh:g}  [%]')
    a.legend(frameon=False)
    for _, r in s.iterrows():
        a.annotate(f"mesh {r['mesh']:g}", (r['n_edges'], r['f_err_ref [%]']), textcoords='offset points',
                   xytext=(0, 8), ha='center', fontsize=8, color='#555555')
    b.plot(s['n_edges'], s['rel_l2'], '-o', color=MODEL_C, lw=2, ms=8)
    b.set_ylim(0, 1.1 * s['rel_l2'].max())
    b.set_ylabel('model field rel-L2 vs FE on the same mesh')
    b.set_title('field error', fontsize=10, loc='left')
    a.set_title('frequency error', fontsize=10, loc='left')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', required=True, help='.ckpt file or a training dir')
    ap.add_argument('--pkls', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', required=True, help='mesh size of each PKL (finest = reference)')
    ap.add_argument('--csv', default=None)
    ap.add_argument('--plot', default=None)
    ap.add_argument('--by_shape', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    lm = CavityLightning.load_from_checkpoint(resolve_checkpoint(args.checkpoint), map_location=args.device)
    df = study(lm, args.pkls, args.labels, args.device)
    if args.csv:
        df.to_csv(args.csv, index=False)
    pd.set_option('display.width', 160)
    s = summarize(df)
    print(f"{df['geom_id'].nunique()} geometries at every resolution; reference = mesh {df['mesh'].min():g}")
    print(s.round(4).to_string())
    if args.by_shape:
        print(summarize(df, True).round(4).to_string())
    if args.plot:
        print(plot(s, args.plot))


if __name__ == '__main__':
    main()
