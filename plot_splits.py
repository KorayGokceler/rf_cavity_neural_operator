import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from tqdm import tqdm
from src.data.dataset import GNOTDataset
from src.config import load_config


def _load_elements(data_path):
    """geom_id -> triangle connectivity (GNOTDataset drops it from RAM)."""
    pool = {}
    if str(data_path).endswith('.pkl'):
        import pickle
        with open(data_path, 'rb') as f:
            raw = pickle.load(f)
        for g_id, geom in raw['geometry_pool'].items():
            if 'elements' in geom:
                pool[int(g_id)] = geom['elements']
    return pool


def plot_split_geometries(dataset, split_name, save_dir, n_geometries=50, elements_pool=None):
    """One figure per geometry: mesh + every mode (ascending frequency).

    Uses GNOTDataset.__getitem__ (one item == one geometry with all K modes),
    so the plotted split is exactly what training sees.
    """
    os.makedirs(save_dir, exist_ok=True)
    elements_pool = elements_pool or {}
    stats = dataset.stats
    n = min(n_geometries, len(dataset))
    print(f"Plotting {n} samples for {split_name} split...")

    for idx in tqdm(range(n), desc=f"{split_name} Split"):
        item = dataset[idx]
        geom_id = int(item['geom_id'].item())
        x = item['X'].numpy()
        y = item['Y_field'].numpy()                  # [N, K]
        freqs = item['Y_freq'].numpy()               # [K] normalized
        if stats:
            freqs = freqs * stats['std'] + stats['mean']
        K = y.shape[1]

        el = elements_pool.get(geom_id)
        if el is not None and int(el.max()) < len(x):
            triang = Triangulation(x[:, 0], x[:, 1], el)
        else:
            triang = Triangulation(x[:, 0], x[:, 1])

        fig, axes = plt.subplots(1, K + 1, figsize=(5 * (K + 1), 5), squeeze=False)
        axes = axes[0]
        fig.suptitle(f"{split_name.capitalize()} Data | Geometry ID: {geom_id}",
                     fontsize=16, fontweight='bold', y=0.98)
        axes[0].triplot(triang, color='gray', linewidth=0.15, alpha=0.5)
        axes[0].set_title("Mesh", fontsize=14)
        for k in range(K):
            axes[k + 1].tripcolor(triang, y[:, k], shading='gouraud', cmap='RdBu_r',
                                  vmin=-1, vmax=1)
            axes[k + 1].set_title(f"Mode {k} ({freqs[k]:.4f} GHz)", fontsize=14)
        for ax in axes:
            ax.set_aspect('equal')
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(save_dir, f"geom_{geom_id:04d}.png")
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Visualize Train and Validation Splits individually")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of geometries to plot per split")
    parser.add_argument("--out_dir", type=str, default="dataset_plots")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc = cfg.dataset
    data_path = dc.data_path

    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset {data_path} not found.")
        print("Please generate the dataset and convert it first!")
        print("Steps:")
        print("1. python src/data_gen/dataset_generator.py")
        print("2. python convert.py")
        return

    # Same split parameters as train.py (seed included) → same geometries.
    kw = dict(train_ratio=dc.train_ratio, val_ratio=dc.val_ratio,
              random_seed=dc.get('random_seed', 42))
    print("Loading Train Split...")
    train_dataset = GNOTDataset(data_path, split='train', **kw)
    print("Loading Validation Split...")
    val_dataset = GNOTDataset(data_path, split='val', **kw)
    elements_pool = _load_elements(data_path)

    print("\nPlotting Training Samples...")
    plot_split_geometries(train_dataset, "Train", os.path.join(args.out_dir, "train_split"),
                          n_geometries=args.n_samples, elements_pool=elements_pool)

    print("\nPlotting Validation Samples...")
    plot_split_geometries(val_dataset, "Validation", os.path.join(args.out_dir, "val_split"),
                          n_geometries=args.n_samples, elements_pool=elements_pool)

    print("\n🎉 All previews matched and saved individually!")


if __name__ == '__main__':
    main()
