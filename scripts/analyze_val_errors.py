#!/usr/bin/env python3
"""Diagnostic error analysis for a trained RF-cavity neural operator.

Takes output from scripts/infer_val_all.py (CSV + optional NPZ) and produces
matplotlib PNG figures that reveal *where* and *why* the model makes the most
errors on the validation split.

Figures produced in --out_dir:
  fig1_mode_errors.png        per-mode rel-L2 boxplots (degen vs non-degen)
  fig2_worst_geoms.png        field plots for the top-N worst geometries
  fig3_error_vs_features.png  rel-L2 vs geometric feature scatter (Spearman r)
  fig4_spatial_error.png      error binned by boundary distance
  fig5_degen_comparison.png   degenerate vs well-separated error comparison
  fig6_ranking_table.png      matplotlib table of top-20 worst geometries
  val_metrics_ranked.csv      full CSV sorted by mean rel-L2 (worst first)

Usage:
    # If infer_val_all.py output already exists:
    python scripts/analyze_val_errors.py \\
        --csv val_metrics.csv \\
        --npz val_preds.npz \\
        --out_dir error_analysis/ \\
        --top_n 10

    # Auto-run inference first (needs checkpoint + data):
    python scripts/analyze_val_errors.py \\
        --checkpoint checkpoints/best.ckpt \\
        --data_path data/gnot_dataset_5k.pkl \\
        --out_dir error_analysis/ \\
        --top_n 10
"""
import os
import re
import sys
import argparse
import subprocess
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── helpers ──────────────────────────────────────────────────────────────────

def _infer_if_needed(args):
    """Run infer_val_all.py when CSV/NPZ are missing."""
    need_run = (args.csv and not os.path.exists(args.csv)) or \
               (args.npz and not os.path.exists(args.npz))
    if not need_run:
        return
    if not args.checkpoint:
        raise RuntimeError(
            "CSV/NPZ not found and --checkpoint not given. "
            "Run scripts/infer_val_all.py first or provide --checkpoint.")
    print("Running inference (this may take a while) …")
    cmd = [
        sys.executable, os.path.join(_ROOT, 'scripts', 'infer_val_all.py'),
        '--checkpoint', args.checkpoint,
        '--data_path', args.data_path,
        '--split', 'val',
    ]
    if args.csv:
        cmd += ['--csv', args.csv]
    if args.npz:
        cmd += ['--dump_npz', args.npz]
    subprocess.run(cmd, check=True)


def _load_npz(npz_path):
    """Return dict: gid (int) -> {field_name: array}."""
    if not npz_path or not os.path.exists(npz_path):
        return {}
    data = defaultdict(dict)
    pattern = re.compile(r'^g(\d+)_(.+)$')
    npz = np.load(npz_path, allow_pickle=True)
    for key in npz.files:
        m = pattern.match(key)
        if m:
            data[int(m.group(1))][m.group(2)] = npz[key]
    return dict(data)


def _detect_K(df):
    """Infer number of modes from column names."""
    ks = [int(m.group(1))
          for col in df.columns
          for m in [re.match(r'^mode(\d+)_relL2$', col)]
          if m]
    return max(ks) + 1 if ks else 0


def _boundary_dist(coords):
    """Normalized distance from convex hull boundary (0=boundary, 1=deep interior)."""
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(coords)
        boundary_mask = np.zeros(len(coords), dtype=bool)
        boundary_mask[hull.vertices] = True
        hull_pts = coords[hull.vertices]

        from scipy.spatial.distance import cdist
        d = cdist(coords, hull_pts).min(axis=1)
        d[boundary_mask] = 0.0
        d_max = d.max()
        return d / (d_max + 1e-12) if d_max > 0 else d
    except Exception:
        # Fallback: radial distance from centroid
        c = coords.mean(axis=0)
        r = np.linalg.norm(coords - c, axis=1)
        r_max = r.max()
        return (r_max - r) / (r_max + 1e-12)  # 0=boundary, 1=center


def _sign_agnostic_rel_l2(pred, target):
    """min(||p-t||, ||p+t||) / ||t||"""
    den = np.linalg.norm(target) + 1e-12
    return min(np.linalg.norm(pred - target), np.linalg.norm(pred + target)) / den


def _savefig(fig, path, dpi=150):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


# ── figure 1: per-mode error boxplots ────────────────────────────────────────

def fig1_mode_errors(df, K, out_dir):
    """Box plots of per-mode rel-L2, split by near-degenerate flag."""
    cols = [f'mode{k}_relL2' for k in range(K)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  [fig1] skipped — columns missing: {missing}")
        return

    is_deg = df['is_near_degenerate'].astype(bool)
    df_sep = df[~is_deg]
    df_deg = df[is_deg]

    fig, ax = plt.subplots(figsize=(max(6, K * 1.8), 5))
    positions_sep = np.arange(K) - 0.18
    positions_deg = np.arange(K) + 0.18
    w = 0.3

    data_sep = [df_sep[c].dropna().values for c in cols]
    data_deg = [df_deg[c].dropna().values for c in cols]

    bp1 = ax.boxplot(data_sep, positions=positions_sep, widths=w, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2),
                     flierprops=dict(marker='o', markersize=3, alpha=0.4))
    bp2 = ax.boxplot(data_deg, positions=positions_deg, widths=w, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2),
                     flierprops=dict(marker='o', markersize=3, alpha=0.4))

    color_sep, color_deg = '#4C72B0', '#DD8452'
    for patch in bp1['boxes']:
        patch.set_facecolor(color_sep)
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor(color_deg)
        patch.set_alpha(0.7)

    ax.set_yscale('log')
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([f'Mode {k}' for k in range(K)])
    ax.set_ylabel('Relative L2 error (sign-agnostic)')
    ax.set_title('Per-mode error distribution — validation split')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]],
              [f'Well-separated (n={len(df_sep)})',
               f'Near-degenerate (n={len(df_deg)})'],
              loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Annotate medians
    for k in range(K):
        vals = df[cols[k]].dropna().values
        if len(vals):
            ax.text(k, np.median(vals) * 1.05, f'{np.median(vals):.3f}',
                    ha='center', va='bottom', fontsize=7, color='dimgray')

    _savefig(fig, os.path.join(out_dir, 'fig1_mode_errors.png'))


# ── figure 2: worst-N geometry field plots ───────────────────────────────────

def fig2_worst_geoms(df, npz_data, K, top_n, out_dir):
    """For each of the top_n worst geometries: GT | Pred | Error scatter plots.

    Shows only the mode with the highest rel-L2 error for readability.
    """
    if not npz_data:
        print("  [fig2] skipped — no NPZ data")
        return

    sort_col = 'mean_relL2_areaw' if 'mean_relL2_areaw' in df.columns else 'mean_relL2'
    worst = df.nlargest(top_n, sort_col)

    n_show = min(top_n, len(worst))
    fig = plt.figure(figsize=(13, 3.2 * n_show), constrained_layout=True)
    gs = gridspec.GridSpec(n_show, 3, figure=fig, hspace=0.45, wspace=0.3)

    cmap_field = 'RdBu_r'
    cmap_err = 'hot_r'

    for row_idx, (_, row) in enumerate(worst.iterrows()):
        gid = int(row['geom_id'])
        if gid not in npz_data:
            continue
        d = npz_data[gid]
        coords = d.get('coords')
        pred = d.get('pred')
        target = d.get('target')
        if coords is None or pred is None or target is None:
            continue

        pred = np.asarray(pred)
        target = np.asarray(target)
        if pred.ndim == 1:
            pred = pred[:, np.newaxis]
            target = target[:, np.newaxis]

        # pick worst mode for this geometry
        mode_errs = [_sign_agnostic_rel_l2(pred[:, k], target[:, k])
                     for k in range(pred.shape[1])]
        km = int(np.argmax(mode_errs))
        t_k = target[:, km]
        p_k = pred[:, km]
        # sign-align prediction to target
        if np.sum((p_k + t_k) ** 2) < np.sum((p_k - t_k) ** 2):
            p_k = -p_k

        vmax = np.abs(t_k).max()
        err = p_k - t_k
        emax = np.abs(err).max()

        x, y = coords[:, 0], coords[:, 1]
        s = max(1, 5000 // len(x))  # dot size scales with density

        ax_gt = fig.add_subplot(gs[row_idx, 0])
        ax_pr = fig.add_subplot(gs[row_idx, 1])
        ax_er = fig.add_subplot(gs[row_idx, 2])

        for ax, vals, cm, vmin_, vmax_, title in [
            (ax_gt, t_k, cmap_field, -vmax, vmax, 'Ground truth'),
            (ax_pr, p_k, cmap_field, -vmax, vmax, 'Prediction'),
            (ax_er, err, cmap_err, 0, emax, 'Abs error'),
        ]:
            sc = ax.scatter(x, y, c=np.abs(vals) if cm == cmap_err else vals,
                            cmap=cm, vmin=vmin_ if cm != cmap_err else 0,
                            vmax=vmax_, s=s, rasterized=True)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            ax.set_aspect('equal')
            ax.set_title(title, fontsize=9)
            ax.axis('off')

        rl2 = mode_errs[km]
        mean_rl2 = row[sort_col]
        ax_gt.set_title(
            f'geom {gid}  mode {km}  rl2={rl2:.3f}  mean_rl2={mean_rl2:.3f}',
            fontsize=8, loc='left')

    fig.suptitle(f'Top-{n_show} worst geometries (sorted by {sort_col})',
                 fontsize=11)
    fig.savefig(os.path.join(out_dir, 'fig2_worst_geoms.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {os.path.join(out_dir, 'fig2_worst_geoms.png')}")


# ── figure 3: error vs geometric features ────────────────────────────────────

def fig3_error_vs_features(df, npz_data, K, out_dir):
    """Scatter of mean rel-L2 vs geometric feature, with Spearman r."""
    y_col = 'mean_relL2_areaw' if 'mean_relL2_areaw' in df.columns else 'mean_relL2'
    y = df[y_col].values

    features = {}

    # n_nodes — already in CSV
    if 'n_nodes' in df.columns:
        features['n_nodes'] = df['n_nodes'].values

    # Mean true frequency across modes
    freq_cols = [f'mode{k}_freq_true_ghz' for k in range(K)
                 if f'mode{k}_freq_true_ghz' in df.columns]
    if freq_cols:
        features['mean_freq_GHz'] = df[freq_cols].mean(axis=1).values

    # Frequency separation (f1 - f0) / f0 for modes 0 & 1
    if K >= 2 and 'mode0_freq_true_ghz' in df.columns and 'mode1_freq_true_ghz' in df.columns:
        f0 = df['mode0_freq_true_ghz'].values
        f1 = df['mode1_freq_true_ghz'].values
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            features['freq_sep (f1-f0)/f0'] = np.abs(f1 - f0) / (f0 + 1e-12)

    # Aspect ratio + effective radius from NPZ coords
    if npz_data:
        aspect_ratios = {}
        eff_radii = {}
        for gid, d in npz_data.items():
            coords = d.get('coords')
            if coords is None:
                continue
            x_range = coords[:, 0].max() - coords[:, 0].min()
            y_range = coords[:, 1].max() - coords[:, 1].min()
            ar = max(x_range, y_range) / (min(x_range, y_range) + 1e-12)
            aspect_ratios[gid] = ar
            eff_radii[gid] = np.sqrt(x_range * y_range / np.pi)

        if aspect_ratios:
            features['aspect_ratio'] = df['geom_id'].map(aspect_ratios).values
        if eff_radii:
            features['effective_radius'] = df['geom_id'].map(eff_radii).values

    n_feats = len(features)
    if n_feats == 0:
        print("  [fig3] skipped — no features available")
        return

    ncols = min(n_feats, 3)
    nrows = (n_feats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)

    is_deg = df['is_near_degenerate'].astype(bool).values
    colors = np.where(is_deg, '#DD8452', '#4C72B0')

    for ax_idx, (feat_name, x) in enumerate(features.items()):
        r, c = divmod(ax_idx, ncols)
        ax = axes[r][c]
        valid = ~np.isnan(x) & ~np.isnan(y)
        x_v, y_v, c_v = x[valid], y[valid], colors[valid]

        ax.scatter(x_v, y_v, c=c_v, alpha=0.6, s=25, edgecolors='none')
        ax.set_yscale('log')
        ax.set_xlabel(feat_name)
        ax.set_ylabel(y_col)
        ax.grid(alpha=0.2)

        if len(x_v) > 3:
            r_sp, p_sp = stats.spearmanr(x_v, y_v)
            ax.set_title(f'{feat_name}  Spearman r={r_sp:.2f}  (p={p_sp:.3f})',
                         fontsize=9)
        else:
            ax.set_title(feat_name, fontsize=9)

    # hide unused axes
    for i in range(n_feats, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].set_visible(False)

    # legend
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker='o', color='w',
                            markerfacecolor='#4C72B0', label='well-separated'),
                    Line2D([0], [0], marker='o', color='w',
                            markerfacecolor='#DD8452', label='near-degenerate')]
    fig.legend(handles=legend_elems, loc='lower right', fontsize=9)
    fig.suptitle('Error vs geometric features', fontsize=11)
    _savefig(fig, os.path.join(out_dir, 'fig3_error_vs_features.png'))


# ── figure 4: spatial error (boundary vs interior) ───────────────────────────

def fig4_spatial_error(df, npz_data, K, out_dir):
    """Per-node absolute error binned by boundary distance."""
    if not npz_data:
        print("  [fig4] skipped — no NPZ data")
        return

    all_dist = []
    all_err = []

    for gid, d in npz_data.items():
        coords = d.get('coords')
        pred = d.get('pred')
        target = d.get('target')
        if coords is None or pred is None or target is None:
            continue

        pred = np.asarray(pred)
        target = np.asarray(target)
        if pred.ndim == 1:
            pred = pred[:, np.newaxis]
            target = target[:, np.newaxis]

        dist = _boundary_dist(coords)  # 0=boundary, 1=deep interior

        # sign-align each mode then compute mean abs error across modes
        abs_err = np.zeros(len(coords))
        for k in range(pred.shape[1]):
            p_k, t_k = pred[:, k], target[:, k]
            if np.sum((p_k + t_k) ** 2) < np.sum((p_k - t_k) ** 2):
                p_k = -p_k
            abs_err += np.abs(p_k - t_k)
        abs_err /= pred.shape[1]

        all_dist.append(dist)
        all_err.append(abs_err)

    if not all_dist:
        print("  [fig4] skipped — no valid geometries in NPZ")
        return

    dist_all = np.concatenate(all_dist)
    err_all = np.concatenate(all_err)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: hexbin density
    ax = axes[0]
    hb = ax.hexbin(dist_all, err_all, gridsize=40, bins='log', cmap='YlOrRd',
                   mincnt=1)
    plt.colorbar(hb, ax=ax, label='log10(count)')
    ax.set_xlabel('Boundary distance (0=wall, 1=center)')
    ax.set_ylabel('Mean abs field error')
    ax.set_title('Error density vs boundary distance')

    # Right: binned mean ± std
    ax2 = axes[1]
    bins = np.linspace(0, 1, 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    idx = np.digitize(dist_all, bins, right=True).clip(0, len(bins) - 2)
    bin_means, bin_stds, bin_ns = [], [], []
    for b in range(len(bin_centers)):
        mask = idx == b
        vals = err_all[mask]
        bin_means.append(vals.mean() if len(vals) else np.nan)
        bin_stds.append(vals.std() if len(vals) else np.nan)
        bin_ns.append(len(vals))

    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    ax2.bar(bin_centers, bin_means, width=0.08, yerr=bin_stds,
            color='steelblue', alpha=0.7, capsize=3)
    ax2.set_xlabel('Boundary distance (0=wall, 1=center)')
    ax2.set_ylabel('Mean abs field error')
    ax2.set_title('Binned error vs boundary distance')
    ax2.axvline(0.1, color='red', linestyle='--', alpha=0.5,
                label='≤0.1 = near boundary')
    ax2.legend(fontsize=8)

    # Compute boundary vs interior ratio
    bnd_mask = dist_all < 0.1
    if bnd_mask.any() and (~bnd_mask).any():
        bnd_mean = err_all[bnd_mask].mean()
        int_mean = err_all[~bnd_mask].mean()
        ratio = bnd_mean / (int_mean + 1e-12)
        fig.suptitle(
            f'Spatial error analysis  '
            f'(boundary mean={bnd_mean:.4f}, interior mean={int_mean:.4f}, '
            f'ratio={ratio:.2f})',
            fontsize=10)

    _savefig(fig, os.path.join(out_dir, 'fig4_spatial_error.png'))


# ── figure 5: degenerate vs non-degenerate comparison ────────────────────────

def fig5_degen_comparison(df, K, out_dir):
    """Compare degenerate vs non-degenerate geometry error distributions."""
    y_col = 'mean_relL2_areaw' if 'mean_relL2_areaw' in df.columns else 'mean_relL2'
    is_deg = df['is_near_degenerate'].astype(bool)
    sep_vals = df.loc[~is_deg, y_col].dropna().values
    deg_vals = df.loc[is_deg, y_col].dropna().values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: violin / box comparison
    ax = axes[0]
    parts = ax.violinplot([sep_vals, deg_vals], positions=[0, 1],
                          showmedians=True, showextrema=True)
    colors_v = ['#4C72B0', '#DD8452']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_v[i])
        pc.set_alpha(0.7)
    ax.set_yscale('log')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        f'Well-separated\n(n={len(sep_vals)})',
        f'Near-degenerate\n(n={len(deg_vals)})',
    ])
    ax.set_ylabel(y_col)
    ax.set_title('Error distribution: degenerate vs well-separated')
    ax.grid(axis='y', alpha=0.3)

    if len(sep_vals) > 0 and len(deg_vals) > 0:
        mw = stats.mannwhitneyu(sep_vals, deg_vals, alternative='two-sided')
        ax.text(0.5, 0.97, f'Mann-Whitney U p={mw.pvalue:.3e}',
                transform=ax.transAxes, ha='center', va='top', fontsize=8)

    # Right: freq separation vs mean rel-L2 (for all samples with 2+ modes)
    ax2 = axes[1]
    if (K >= 2 and 'mode0_freq_true_ghz' in df.columns
            and 'mode1_freq_true_ghz' in df.columns):
        f0 = df['mode0_freq_true_ghz'].values
        f1 = df['mode1_freq_true_ghz'].values
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            freq_sep = np.abs(f1 - f0) / (np.abs(f0) + 1e-12)
        y_all = df[y_col].values
        valid = ~np.isnan(freq_sep) & ~np.isnan(y_all)
        colors_all = np.where(is_deg.values, '#DD8452', '#4C72B0')
        ax2.scatter(freq_sep[valid], y_all[valid],
                    c=colors_all[valid], alpha=0.5, s=20)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('|f1 - f0| / f0  (mode separation)')
        ax2.set_ylabel(y_col)
        ax2.set_title('Error vs mode frequency separation')
        ax2.grid(alpha=0.2)
        ax2.axvline(0.05, color='red', linestyle='--', alpha=0.5,
                    label='deg threshold (5%)')
        ax2.legend(fontsize=8)

        if valid.sum() > 3:
            r_sp, p_sp = stats.spearmanr(freq_sep[valid], y_all[valid])
            ax2.set_title(
                f'Error vs freq separation  Spearman r={r_sp:.2f} p={p_sp:.3f}',
                fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Need K≥2 with freq columns', ha='center',
                 transform=ax2.transAxes)
        ax2.set_axis_off()

    _savefig(fig, os.path.join(out_dir, 'fig5_degen_comparison.png'))


# ── figure 6: ranking table ───────────────────────────────────────────────────

def fig6_ranking_table(df, K, out_dir):
    """Top-20 worst geometries as matplotlib table + save ranked CSV."""
    sort_col = 'mean_relL2_areaw' if 'mean_relL2_areaw' in df.columns else 'mean_relL2'
    df_sorted = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # Save full ranked CSV
    csv_path = os.path.join(out_dir, 'val_metrics_ranked.csv')
    df_sorted.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    # Build display table (top-20)
    top = df_sorted.head(20)
    mode_cols = [f'mode{k}_relL2' for k in range(K)
                 if f'mode{k}_relL2' in df.columns]
    display_cols = ['geom_id', 'n_nodes', sort_col] + mode_cols + ['is_near_degenerate']
    display_cols = [c for c in display_cols if c in top.columns]
    tbl = top[display_cols].copy()

    # Format floats
    for c in tbl.select_dtypes(include=[float]).columns:
        tbl[c] = tbl[c].apply(lambda v: f'{v:.4f}' if not np.isnan(v) else 'NaN')

    n_rows = len(tbl)
    n_cols = len(display_cols)
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.4), max(4, n_rows * 0.35 + 1)))
    ax.axis('off')

    headers = [c.replace('_relL2', '\nrl2').replace('_areaw', '\n(aw)')
                .replace('is_near_degenerate', 'degen') for c in display_cols]
    data = tbl.values.tolist()
    tbl_obj = ax.table(
        cellText=data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    tbl_obj.auto_set_font_size(False)
    tbl_obj.set_fontsize(8)
    tbl_obj.auto_set_column_width(col=list(range(n_cols)))

    # Highlight worst rows
    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            cell = tbl_obj[r_idx + 1, c_idx]
            shade = 0.95 - 0.45 * (r_idx / max(n_rows - 1, 1))
            cell.set_facecolor((1.0, shade, shade))  # red gradient

    ax.set_title(f'Top-{n_rows} worst geometries (sorted by {sort_col})',
                 fontsize=10, pad=10)
    _savefig(fig, os.path.join(out_dir, 'fig6_ranking_table.png'), dpi=120)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', default='val_metrics.csv',
                    help='per-geometry metrics CSV from infer_val_all.py')
    ap.add_argument('--npz', default=None,
                    help='predictions NPZ from infer_val_all.py (optional; '
                         'enables field-plot figures)')
    ap.add_argument('--out_dir', default='error_analysis',
                    help='output directory for PNG figures')
    ap.add_argument('--top_n', type=int, default=8,
                    help='number of worst geometries to plot in fig2')
    # inference options (used only when CSV/NPZ do not exist)
    ap.add_argument('--checkpoint', default=None,
                    help='model checkpoint (runs inference if CSV missing)')
    ap.add_argument('--data_path', default='data/gnot_dataset_5k.pkl',
                    help='dataset path (used only when running inference)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- ensure CSV exists ---
    _infer_if_needed(args)

    if not os.path.exists(args.csv):
        raise FileNotFoundError(
            f"CSV not found: {args.csv}\n"
            "Run scripts/infer_val_all.py first or provide --checkpoint.")

    print(f"\nLoading {args.csv} …")
    df = pd.read_csv(args.csv)
    # Keep only val split rows if the CSV covers multiple splits
    if 'split' in df.columns and 'val' in df['split'].values:
        df = df[df['split'] == 'val'].reset_index(drop=True)
    print(f"  {len(df)} geometries, columns: {list(df.columns)}")

    K = _detect_K(df)
    print(f"  Detected K={K} modes\n")

    npz_data = {}
    if args.npz:
        print(f"Loading {args.npz} …")
        npz_data = _load_npz(args.npz)
        print(f"  {len(npz_data)} geometries in NPZ\n")

    print("Generating figures …")
    fig1_mode_errors(df, K, args.out_dir)
    fig2_worst_geoms(df, npz_data, K, args.top_n, args.out_dir)
    fig3_error_vs_features(df, npz_data, K, args.out_dir)
    fig4_spatial_error(df, npz_data, K, args.out_dir)
    fig5_degen_comparison(df, K, args.out_dir)
    fig6_ranking_table(df, K, args.out_dir)

    print(f"\nDone. All figures saved to: {args.out_dir}/")
    print("Quick start:\n"
          "  fig1 = per-mode error boxes (which modes are hardest?)\n"
          "  fig2 = worst geometries field plots (GT vs Pred vs Error)\n"
          "  fig3 = error vs geometry features (what correlates with failure?)\n"
          "  fig4 = error by distance from boundary wall\n"
          "  fig5 = degenerate vs well-separated comparison\n"
          "  fig6 = ranked table of worst geometries\n"
          "  val_metrics_ranked.csv = all geometries sorted by error")


if __name__ == '__main__':
    main()
