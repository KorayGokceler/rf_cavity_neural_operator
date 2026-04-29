"""
GNOT RF Cavity - Feature Visualization Script (Updated for 8 Features)
Generates detailed visualizations of all features extracted from the dataset.

Usage:
    python visualize_features.py --pkl_path data/gnot_dataset.pkl --sample_idx 0
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.tri import Triangulation
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
    Plot all geometric features for a single sample (8 channels).
    """
    sample = data['samples'][sample_idx]
    geom_id = sample['geom_id']
    geom = data['geometry_pool'][geom_id]

    nodes = geom['X']
    input_funcs = geom['Input_funcs']  # (N, 8)
    elements = geom['elements']
    Y = sample['Y']
    theta = sample['Theta']

    tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)

    # Updated feature map for the current 8-feature extractor
    feature_map = {
        'Normalized X': input_funcs[:, 0],
        'Normalized Y': input_funcs[:, 1],
        'Dist to Boundary': input_funcs[:, 2],
        'Dir Boundary X': input_funcs[:, 3],
        'Dir Boundary Y': input_funcs[:, 4],
        'Node Area (Density)': input_funcs[:, 5],
        'Cos(Principal)': input_funcs[:, 6],
        'Sin(Principal)': input_funcs[:, 7],
    }

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(
        f"Geometric Features — Sample {geom_id} | Mode {int(theta[0])+1} | "
        f"Freq: {theta[1]:.4f} GHz | Nodes: {len(nodes)}",
        fontsize=18, fontweight='bold', y=0.98
    )

    cmaps = ['viridis', 'plasma', 'cividis', 'magma', 'inferno', 'rocket', 'icefire', 'twilight']

    for ax, (name, values), cmap in zip(axes.flatten(), feature_map.items(), cmaps):
        tc = ax.tripcolor(tri, values, shading='flat', cmap=cmap)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
        cb = fig.colorbar(tc, ax=ax, shrink=0.7, pad=0.02)
        cb.ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path1 = os.path.join(save_dir, f"features_geometric_sample_{geom_id}.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {path1}")

    # Figure 2: Mode Shape + Mesh
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    mode = Y[:, 0]
    tc = axes[0].tripcolor(tri, mode, shading='flat', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title("Mode Shape (Field)", fontsize=14, fontweight='bold')
    axes[0].set_aspect('equal')
    axes[0].axis('off')
    fig.colorbar(tc, ax=axes[0])

    axes[1].triplot(tri, color='black', linewidth=0.1, alpha=0.5)
    axes[1].set_title(f"Mesh Structure ({len(elements)} elements)", fontsize=14, fontweight='bold')
    axes[1].set_aspect('equal')
    axes[1].axis('off')

    plt.tight_layout()
    path2 = os.path.join(save_dir, f"mode_shape_sample_{geom_id}.png")
    plt.savefig(path2, dpi=150)
    plt.close()

def plot_dataset_statistics(data, save_dir):
    """Dataset-wide statistics."""
    samples = data['samples']
    geometry_pool = data['geometry_pool']

    freqs = np.array([s['Theta'][1] for s in samples])
    mode_indices = np.array([int(s['Theta'][0]) for s in samples])
    mesh_sizes = np.array([len(g['X']) for g in geometry_pool.values()])

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Freq Dist
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(freqs, bins=40, ax=ax1, color='skyblue', kde=True)
    ax1.set_title("Overall Frequency Distribution", fontweight='bold')

    # Freq by Mode
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(x=mode_indices, y=freqs, ax=ax2, palette='Set2')
    ax2.set_title("Frequency by Mode", fontweight='bold')
    ax2.set_xticklabels([f"Mode {i+1}" for i in sorted(np.unique(mode_indices))])

    # Mesh sizes
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(mesh_sizes, bins=30, ax=ax3, color='salmon')
    ax3.set_title("Mesh Sizes (Nodes)", fontweight='bold')

    # Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    stats_text = [
        ["Total Geometries", len(geometry_pool)],
        ["Total Samples", len(samples)],
        ["Input Features", "8 (X, Y, Dist, DirX, DirY, Area, Cos, Sin)"],
        ["Freq Range", f"{freqs.min():.2f} - {freqs.max():.2f} GHz"]
    ]
    table = ax4.table(cellText=stats_text, colLabels=["Metric", "Value"], loc='center', cellLoc='left')
    table.scale(1, 2)
    ax4.set_title("Dataset Summary", fontweight='bold')

    plt.savefig(os.path.join(save_dir, "dataset_statistics.png"), dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, default="data/gnot_dataset.pkl")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="feature_plots")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    data = load_dataset(args.pkl_path)
    
    print(f"📊 Visualizing Sample {args.sample_idx}...")
    plot_single_sample_features(data, args.sample_idx, args.save_dir)
    
    print(f"📈 Generating Dataset Stats...")
    plot_dataset_statistics(data, args.save_dir)
    print(f"🎉 Done! Plots saved to: {args.save_dir}/")

if __name__ == '__main__':
    main()
