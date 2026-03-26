"""
GNOT RF Cavity - Feature Visualization Script
Generates detailed visualizations of all features extracted from the dataset
before they are fed into the GNOT model.

Usage:
    python visualize_features.py --pkl_path data/gnot_dataset.pkl --sample_idx 0
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.tri import Triangulation
from collections import defaultdict
import seaborn as sns
import os

sns.set_theme(style="darkgrid")


def load_dataset(pkl_path):
    """Load the processed PKL dataset."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_single_sample_features(data, sample_idx, save_dir):
    """
    Plot all geometric and physical features for a single sample.
    Features in Input_funcs (6 channels):
      [0:2] = Normalized X, Y coordinates
      [2]   = Distance to boundary
      [3]   = Boundary mask (0 or 1)
      [4]   = Distance to center
      [5]   = Local curvature
    """
    sample = data['samples'][sample_idx]
    geom_id = sample['geom_id']
    geom = data['geometry_pool'][geom_id]

    X = geom['X']                       # (N, 2) normalized coordinates
    input_funcs = geom['Input_funcs'][0] # (N, 6) all geometric features
    elements = geom['elements']          # (M, 3) triangle connectivity
    Y = sample['Y']                      # (N, 1) mode shape
    theta = sample['Theta']              # [mode_idx, freq, sample_id]

    nodes = X
    tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)

    # Feature names and their column indices in input_funcs
    feature_map = {
        'Normalized X': input_funcs[:, 0],
        'Normalized Y': input_funcs[:, 1],
        'Distance to Boundary': input_funcs[:, 2],
        'Boundary Mask': input_funcs[:, 3],
        'Distance to Center': input_funcs[:, 4],
        'Local Curvature': input_funcs[:, 5],
    }

    # =========================================================================
    # Figure 1: Geometric Features (6 panels)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle(
        f"Geometric Features — Sample {geom_id} | Mode {int(theta[0])+1} | "
        f"Freq: {theta[1]:.4f} GHz | Nodes: {len(nodes)}",
        fontsize=16, fontweight='bold', y=0.98
    )

    cmaps = ['viridis', 'plasma', 'cividis', 'coolwarm', 'magma', 'YlOrRd']

    for ax, (name, values), cmap in zip(axes.flatten(), feature_map.items(), cmaps):
        tc = ax.tripcolor(tri, values, shading='gouraud', cmap=cmap)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
        cb = fig.colorbar(tc, ax=ax, shrink=0.75, pad=0.02)
        cb.ax.tick_params(labelsize=9)

        # Min/Max annotations
        ax.text(0.02, 0.02, f"min: {values.min():.4f}\nmax: {values.max():.4f}",
                transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path1 = os.path.join(save_dir, f"features_geometric_sample_{geom_id}.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {path1}")

    # =========================================================================
    # Figure 2: Mode Shape + Mesh Overlay
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(
        f"Mode Shape & Mesh — Sample {geom_id} | Mode {int(theta[0])+1} | "
        f"Freq: {theta[1]:.4f} GHz",
        fontsize=16, fontweight='bold'
    )

    # Panel 1: Raw Mesh
    axes[0].triplot(tri, color='steelblue', linewidth=0.15, alpha=0.6)
    axes[0].set_title(f"Mesh ({len(elements)} elements)", fontsize=13, fontweight='bold')
    axes[0].set_aspect('equal')
    axes[0].axis('off')

    # Panel 2: Mode Shape
    mode = Y[:, 0]
    tc = axes[1].tripcolor(tri, mode, shading='gouraud', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title(f"Mode Shape (normalized)", fontsize=13, fontweight='bold')
    axes[1].set_aspect('equal')
    axes[1].axis('off')
    fig.colorbar(tc, ax=axes[1], shrink=0.75)

    # Panel 3: Mode Shape + Mesh overlay
    axes[2].tripcolor(tri, mode, shading='gouraud', cmap='RdBu_r', vmin=-1, vmax=1, alpha=0.7)
    axes[2].triplot(tri, color='black', linewidth=0.08, alpha=0.3)
    axes[2].set_title("Mode + Mesh Overlay", fontsize=13, fontweight='bold')
    axes[2].set_aspect('equal')
    axes[2].axis('off')

    plt.tight_layout()
    path2 = os.path.join(save_dir, f"features_mode_sample_{geom_id}.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {path2}")

    # =========================================================================
    # Figure 3: Feature Distributions (Histograms)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f"Feature Distributions — Sample {geom_id}", fontsize=16, fontweight='bold')

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']

    for ax, (name, values), color in zip(axes.flatten(), feature_map.items(), colors):
        ax.hist(values, bins=50, color=color, alpha=0.75, edgecolor='white', linewidth=0.5)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel("Value", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)

        # Stats box
        stats_text = (f"μ = {values.mean():.4f}\n"
                      f"σ = {values.std():.4f}\n"
                      f"med = {np.median(values):.4f}")
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path3 = os.path.join(save_dir, f"features_distributions_sample_{geom_id}.png")
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {path3}")


def plot_dataset_statistics(data, save_dir):
    """
    Plot dataset-wide statistics (frequency distributions, mesh sizes, etc.)
    """
    samples = data['samples']
    geometry_pool = data['geometry_pool']

    # Collect stats
    freqs = []
    mode_indices = []
    mesh_sizes = []

    for s in samples:
        freqs.append(s['Theta'][1])
        mode_indices.append(int(s['Theta'][0]))

    for gid, geom in geometry_pool.items():
        mesh_sizes.append(len(geom['X']))

    freqs = np.array(freqs)
    mode_indices = np.array(mode_indices)
    mesh_sizes = np.array(mesh_sizes)

    unique_modes = sorted(set(mode_indices))

    # =========================================================================
    # Figure 4: Dataset-Wide Statistics
    # =========================================================================
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("Dataset-Wide Statistics", fontsize=18, fontweight='bold', y=0.98)

    # --- Panel 1: Overall Frequency Distribution ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(freqs, bins=50, color='#3498db', alpha=0.8, edgecolor='white')
    ax1.set_title("All Frequencies", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Count")
    ax1.text(0.97, 0.97, f"μ={freqs.mean():.2f}\nσ={freqs.std():.2f}\n"
             f"min={freqs.min():.2f}\nmax={freqs.max():.2f}",
             transform=ax1.transAxes, fontsize=9, va='top', ha='right',
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    # --- Panel 2: Frequency by Mode (Violin Plot) ---
    ax2 = fig.add_subplot(gs[0, 1])
    freq_by_mode = [freqs[mode_indices == m] for m in unique_modes]
    parts = ax2.violinplot(freq_by_mode, positions=unique_modes, showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#e74c3c')
        pc.set_alpha(0.6)
    ax2.set_title("Frequency by Mode", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Mode Index")
    ax2.set_ylabel("Frequency (GHz)")
    ax2.set_xticks(unique_modes)
    ax2.set_xticklabels([f"Mode {m+1}" for m in unique_modes])

    # --- Panel 3: Mesh Sizes ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(mesh_sizes, bins=30, color='#2ecc71', alpha=0.8, edgecolor='white')
    ax3.set_title("Mesh Sizes (Nodes per Geometry)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Number of Nodes")
    ax3.set_ylabel("Count")
    ax3.text(0.97, 0.97, f"μ={mesh_sizes.mean():.0f}\nσ={mesh_sizes.std():.0f}\n"
             f"min={mesh_sizes.min()}\nmax={mesh_sizes.max()}",
             transform=ax3.transAxes, fontsize=9, va='top', ha='right',
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    # --- Panel 4: Frequency vs Mode (Scatter) ---
    ax4 = fig.add_subplot(gs[1, 0])
    scatter_colors = ['#e74c3c', '#3498db', '#2ecc71']
    for m_idx in unique_modes:
        mask = mode_indices == m_idx
        ax4.scatter(np.arange(mask.sum()), freqs[mask], s=5, alpha=0.5,
                    color=scatter_colors[m_idx % len(scatter_colors)],
                    label=f"Mode {m_idx+1}")
    ax4.set_title("Frequency Scatter by Mode", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Sample Index")
    ax4.set_ylabel("Frequency (GHz)")
    ax4.legend(fontsize=9)

    # --- Panel 5: Feature Correlation Heatmap (from one sample) ---
    ax5 = fig.add_subplot(gs[1, 1])
    sample_geom_id = list(geometry_pool.keys())[0]
    sample_features = geometry_pool[sample_geom_id]['Input_funcs'][0]
    feature_names = ['X', 'Y', 'Dist_Bnd', 'Bnd_Mask', 'Dist_Ctr', 'Curvature']
    corr_matrix = np.corrcoef(sample_features.T)
    im = ax5.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax5.set_xticks(range(len(feature_names)))
    ax5.set_yticks(range(len(feature_names)))
    ax5.set_xticklabels(feature_names, fontsize=9, rotation=45, ha='right')
    ax5.set_yticklabels(feature_names, fontsize=9)
    ax5.set_title("Feature Correlation (Sample 0)", fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax5, shrink=0.75)
    # Annotate cells
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            ax5.text(j, i, f"{corr_matrix[i, j]:.2f}", ha='center', va='center',
                     fontsize=7, color='black' if abs(corr_matrix[i, j]) < 0.5 else 'white')

    # --- Panel 6: Mode Shape Value Distribution ---
    ax6 = fig.add_subplot(gs[1, 2])
    all_y_vals = np.concatenate([s['Y'].flatten() for s in samples[:200]])
    ax6.hist(all_y_vals, bins=80, color='#9b59b6', alpha=0.8, edgecolor='white')
    ax6.set_title("Mode Shape Value Distribution", fontsize=13, fontweight='bold')
    ax6.set_xlabel("Normalized Field Value")
    ax6.set_ylabel("Count")

    # --- Panel 7: Summary Table ---
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    table_data = [
        ["Total Geometries", str(len(geometry_pool))],
        ["Total Samples", str(len(samples))],
        ["Modes per Geometry", str(len(unique_modes))],
        ["Input Feature Dim", "6 (X, Y, dist_bnd, bnd_mask, dist_ctr, curvature)"],
        ["Output Dim (Field)", "1 (normalized mode shape)"],
        ["Output Dim (Freq)", "1 (resonance frequency in GHz)"],
        ["Theta Dim", "1 (mode index)"],
        ["Avg Mesh Size", f"{mesh_sizes.mean():.0f} nodes"],
        ["Frequency Range", f"{freqs.min():.2f} — {freqs.max():.2f} GHz"],
    ]
    table = ax7.table(
        cellText=table_data,
        colLabels=["Property", "Value"],
        loc='center',
        cellLoc='left',
        colWidths=[0.35, 0.55]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#ecf0f1' if row % 2 == 0 else 'white')

    ax7.set_title("Dataset Summary", fontsize=14, fontweight='bold', pad=15)

    path4 = os.path.join(save_dir, "dataset_statistics.png")
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {path4}")


def main():
    parser = argparse.ArgumentParser(description="Visualize GNOT dataset features.")
    parser.add_argument("--pkl_path", type=str, default="data/gnot_dataset.pkl",
                        help="Path to processed PKL dataset.")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Sample index to visualize in detail.")
    parser.add_argument("--save_dir", type=str, default="feature_plots",
                        help="Directory to save output plots.")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Number of individual samples to visualize.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"📊 Loading dataset from {args.pkl_path}...")
    data = load_dataset(args.pkl_path)
    print(f"   Geometries: {len(data['geometry_pool'])}")
    print(f"   Samples:    {len(data['samples'])}")

    # Plot individual sample features
    print(f"\n🔬 Plotting features for {args.n_samples} individual samples...")
    for i in range(min(args.n_samples, len(data['samples']))):
        idx = args.sample_idx + i
        if idx >= len(data['samples']):
            break
        print(f"\n--- Sample {idx} ---")
        plot_single_sample_features(data, idx, args.save_dir)

    # Plot dataset-wide statistics
    print(f"\n📈 Plotting dataset-wide statistics...")
    plot_dataset_statistics(data, args.save_dir)

    print(f"\n🎉 All visualizations saved to: {args.save_dir}/")


if __name__ == '__main__':
    main()
