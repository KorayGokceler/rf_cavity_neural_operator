"""Mode 1 ve Mode 2 frekans farklarini histograma dokuyen analiz scripti.

Cavity'nin 2. ve 3. eigenmod frekanslari birbirine cok yakinsa modlar
"near-degenerate" olur, FEM cozucusu eigenvektorlerin yonelimini tutarsiz
sekilde dondurebilir, bu da modelin ogrenmesini bozar.

Kullanim:
    # Raw H5 dosyasindan (FEM cikti):
    python analyze_freq_separation.py --h5 rf_cavity_1000_dataset.h5
    # Converter ciktisindan (pkl veya h5):
    python analyze_freq_separation.py --dataset data/gnot_dataset.pkl
"""

import argparse
import os
import pickle
from collections import defaultdict

import h5py
import numpy as np
import matplotlib.pyplot as plt


def freqs_from_raw_h5(path):
    """Raw FEM H5 dosyasindan her geometri icin eigenmod frekanslarini cek."""
    geom_freqs = {}
    with h5py.File(path, 'r') as f:
        for key in sorted(f.keys()):
            grp = f[key]
            if 'freqs' not in grp:
                continue
            freqs = grp['freqs'][:]
            sample_id = int(key.split('_')[-1])
            geom_freqs[sample_id] = np.asarray(freqs, dtype=np.float64)
    return geom_freqs


def freqs_from_converted_pkl(path):
    """Converter ciktisi pkl'den her geometri icin (mode_idx -> freq) tablosu."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    samples = data['samples']
    geom_freqs = defaultdict(dict)
    for s in samples:
        g_id = int(s['geom_id'])
        theta = s['Theta']
        m_idx = int(theta[0])
        freq = float(theta[1])
        geom_freqs[g_id][m_idx] = freq
    out = {}
    for g_id, modes in geom_freqs.items():
        max_m = max(modes.keys())
        arr = np.zeros(max_m + 1, dtype=np.float64)
        for m, f_ in modes.items():
            arr[m] = f_
        out[g_id] = arr
    return out


def freqs_from_converted_h5(path):
    """Converter ciktisi H5'ten her geometri icin (mode_idx -> freq) tablosu."""
    geom_freqs = defaultdict(dict)
    with h5py.File(path, 'r') as f:
        samples_grp = f['samples']
        for key in samples_grp.keys():
            samp = samples_grp[key]
            g_id = int(samp.attrs['geom_id'])
            theta = samp['Theta'][:]
            m_idx = int(theta[0])
            freq = float(theta[1])
            geom_freqs[g_id][m_idx] = freq
    out = {}
    for g_id, modes in geom_freqs.items():
        max_m = max(modes.keys())
        arr = np.zeros(max_m + 1, dtype=np.float64)
        for m, f_ in modes.items():
            arr[m] = f_
        out[g_id] = arr
    return out


def load_freqs(args):
    if args.h5:
        print(f"Reading raw FEM H5: {args.h5}")
        return freqs_from_raw_h5(args.h5), 'raw'
    if args.dataset:
        print(f"Reading converted dataset: {args.dataset}")
        if args.dataset.endswith('.pkl'):
            return freqs_from_converted_pkl(args.dataset), 'converted'
        return freqs_from_converted_h5(args.dataset), 'converted'
    raise SystemExit("Either --h5 or --dataset must be provided.")


def plot_histograms(diffs_abs, freqs_m1, freqs_m2, output_path, n_bins, source):
    diffs_rel = diffs_abs / np.maximum(0.5 * (freqs_m1 + freqs_m2), 1e-12)
    n = len(diffs_abs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute frequency separation
    ax = axes[0]
    ax.hist(diffs_abs, bins=n_bins, color='#3a7bd5', edgecolor='black', alpha=0.85)
    ax.axvline(np.median(diffs_abs), color='red', linestyle='--',
               label=f"median = {np.median(diffs_abs):.3f} GHz")
    ax.axvline(np.percentile(diffs_abs, 5), color='orange', linestyle=':',
               label=f"p5 = {np.percentile(diffs_abs, 5):.3f} GHz")
    ax.set_xlabel("|f_mode2 - f_mode1|  [GHz]")
    ax.set_ylabel("Geometry count")
    ax.set_title(f"Absolute separation between mode-1 and mode-2 (N={n})")
    ax.legend()
    ax.grid(alpha=0.3)

    # Relative frequency separation
    ax = axes[1]
    ax.hist(diffs_rel * 100.0, bins=n_bins, color='#5dba6a', edgecolor='black', alpha=0.85)
    ax.axvline(np.median(diffs_rel) * 100, color='red', linestyle='--',
               label=f"median = {np.median(diffs_rel)*100:.2f} %")
    ax.axvline(np.percentile(diffs_rel, 5) * 100, color='orange', linestyle=':',
               label=f"p5 = {np.percentile(diffs_rel, 5)*100:.2f} %")
    ax.set_xlabel("|Δf| / mean(f)  [%]")
    ax.set_ylabel("Geometry count")
    ax.set_title("Relative separation between mode-1 and mode-2")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"Mode-1 / Mode-2 frequency separation  (source: {source})",
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved histogram: {output_path}")


def print_summary(diffs_abs, freqs_m1, freqs_m2, threshold_pct):
    diffs_rel = diffs_abs / np.maximum(0.5 * (freqs_m1 + freqs_m2), 1e-12)
    near_deg = (diffs_rel * 100.0) < threshold_pct

    print("\n" + "=" * 60)
    print("  Mode-1 / Mode-2 frequency separation summary")
    print("=" * 60)
    print(f"  N geometries           : {len(diffs_abs)}")
    print(f"  |Δf|  min / med / max  : {diffs_abs.min():.4f} / "
          f"{np.median(diffs_abs):.4f} / {diffs_abs.max():.4f}  GHz")
    print(f"  rel Δf min / med / max : {diffs_rel.min()*100:.3f} / "
          f"{np.median(diffs_rel)*100:.3f} / {diffs_rel.max()*100:.3f}  %")
    print(f"  Near-degenerate (<{threshold_pct}% rel): "
          f"{near_deg.sum()} / {len(diffs_abs)}  ({near_deg.mean()*100:.1f}%)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Histogram of mode-1 and mode-2 frequency separation per geometry.")
    parser.add_argument('--h5', type=str, default=None,
                        help="Raw FEM H5 file (e.g. rf_cavity_1000_dataset.h5).")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Converted dataset (.pkl or .h5).")
    parser.add_argument('--output', type=str, default='freq_separation_hist.png',
                        help="Output PNG path.")
    parser.add_argument('--bins', type=int, default=40,
                        help="Number of histogram bins.")
    parser.add_argument('--near_deg_threshold', type=float, default=1.0,
                        help="Relative Δf (%%) below which modes are flagged as near-degenerate.")
    parser.add_argument('--mode_a', type=int, default=1,
                        help="Index of first mode to compare (default 1).")
    parser.add_argument('--mode_b', type=int, default=2,
                        help="Index of second mode to compare (default 2).")
    args = parser.parse_args()

    geom_freqs, source = load_freqs(args)

    f_a, f_b = [], []
    skipped = 0
    for g_id, freqs in geom_freqs.items():
        if len(freqs) <= max(args.mode_a, args.mode_b):
            skipped += 1
            continue
        f_a.append(freqs[args.mode_a])
        f_b.append(freqs[args.mode_b])
    if skipped:
        print(f"Skipped {skipped} geometries lacking modes "
              f"{args.mode_a}/{args.mode_b}.")

    f_a = np.asarray(f_a, dtype=np.float64)
    f_b = np.asarray(f_b, dtype=np.float64)
    diffs_abs = np.abs(f_b - f_a)

    if len(diffs_abs) == 0:
        raise SystemExit("No geometries with the requested modes found.")

    print_summary(diffs_abs, f_a, f_b, args.near_deg_threshold)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plot_histograms(diffs_abs, f_a, f_b, args.output, args.bins, source)


if __name__ == '__main__':
    main()
