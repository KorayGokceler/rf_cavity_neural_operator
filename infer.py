import torch
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from torch.utils.data import DataLoader

from src.data.dataset import GNOTDataset, gnot_collate_fn
from src.training.lightning_module import GNOTLightning


def plot_geometry_comparison(geom_id, modes_data, save_path, elements=None, deg_note=""):
    """Plot every mode of a geometry, ordered by ascending predicted frequency.

    Each mode is one row: Ground Truth | Prediction | Error.
    """
    # Modes are keyed by display index (0,1,2 = ascending predicted frequency)
    sorted_modes = sorted(modes_data.items())
    n_modes = len(sorted_modes)

    fig, axes = plt.subplots(n_modes, 3, figsize=(18, 5 * n_modes), squeeze=False)
    title = f"Geometry ID: {geom_id}"
    if deg_note:
        title += f"   ({deg_note})"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    for i, (disp_idx, data) in enumerate(sorted_modes):
        coords = data['coords']
        target = data['target'].flatten()
        pred = data['pred'].flatten()
        f_true = data['f_true']
        f_pred = data['f_pred']
        rel_l2 = data['rel_l2']
        sign_info = data['sign_info']

        if elements is not None:
            tri = Triangulation(coords[:, 0], coords[:, 1], elements)
        else:
            tri = Triangulation(coords[:, 0], coords[:, 1])

        im1 = axes[i, 0].tripcolor(tri, target, cmap='RdBu_r', shading='gouraud', vmin=-1, vmax=1)
        axes[i, 0].set_title(
            f"Mode {disp_idx} (by ascending frequency) - Ground Truth\nFreq: {f_true:.3f} GHz",
            fontsize=12)
        fig.colorbar(im1, ax=axes[i, 0])

        im2 = axes[i, 1].tripcolor(tri, pred, cmap='RdBu_r', shading='gouraud', vmin=-1, vmax=1)
        axes[i, 1].set_title(
            f"Mode {disp_idx} - GNOT Prediction{sign_info}\nFreq: {f_pred:.3f} GHz (Rel L2: {rel_l2:.3f})",
            fontsize=12)
        fig.colorbar(im2, ax=axes[i, 1])

        error = pred - target
        err_max = max(np.abs(error).max(), 1e-8)
        im3 = axes[i, 2].tripcolor(tri, error, cmap='RdBu_r', shading='gouraud', vmin=-err_max, vmax=err_max)
        axes[i, 2].set_title(f"Mode {disp_idx} - Error (pred - truth)\nmax |err|: {err_max:.3f}", fontsize=12)
        fig.colorbar(im3, ax=axes[i, 2])

        for ax in axes[i]:
            ax.set_aspect('equal')
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _near_degenerate_note(freqs, rel_threshold=0.05):
    """Return a human-readable note about near-degenerate frequency pairs.

    freqs: 1-D array of (ascending) frequencies in physical units.
    """
    pairs = []
    K = len(freqs)
    f_mean = float(np.mean(np.abs(freqs))) if K else 0.0
    denom = max(abs(f_mean), 1e-6)
    for i in range(1, K):
        if abs(freqs[i] - freqs[i - 1]) / denom < rel_threshold:
            pairs.append((i - 1, i))
    if not pairs:
        return "well-separated modes"
    return "near-degenerate: " + ", ".join(f"modes {a}&{b}" for a, b in pairs)


def main(args):
    print(f"Loading checkpoint from: {args.checkpoint}")
    model = GNOTLightning.load_from_checkpoint(args.checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    print(f"Loading dataset from: {args.data_path}")
    dataset = GNOTDataset(args.data_path, split=args.split)

    if hasattr(dataset, 'stats') and dataset.stats:
        model.freq_stats = dataset.stats

    if args.freq_mean is not None and args.freq_std is not None:
        model.freq_stats = {'mean': args.freq_mean, 'std': args.freq_std}
        print(f"Using manual frequency stats override: mean={args.freq_mean}, std={args.freq_std}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=gnot_collate_fn)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load triangle connectivity directly from raw data for proper FEM viz.
    elements_pool = {}
    data_path_str = str(args.data_path)
    if data_path_str.endswith('.pkl'):
        import pickle as _pkl
        with open(args.data_path, 'rb') as ef:
            raw = _pkl.load(ef)
        for g_id, geom in raw['geometry_pool'].items():
            if 'elements' in geom:
                elements_pool[int(g_id)] = geom['elements']
    elif data_path_str.endswith('.h5'):
        import h5py
        with h5py.File(args.data_path, 'r') as ef:
            for g_id_str in ef['geometry_pool']:
                grp = ef['geometry_pool'][g_id_str]
                if 'elements' in grp:
                    elements_pool[int(g_id_str)] = grp['elements'][:]
    print(f"Loaded mesh connectivity for {len(elements_pool)} geometries.")

    print(f"Running inference for up to {args.num_samples} geometries...")
    geometries_results = {}

    with torch.no_grad():
        for batch in dataloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(batch)
            preds = outputs['field']           # [B, N, K]
            targets = batch['Y_field']         # [B, N, K]
            coords = batch['X']                # [B, N, grid]
            mask = batch.get('Mask', None)
            geom_ids = batch['geom_id'].squeeze(-1).cpu().numpy()

            f_pred = outputs['freq']           # [B, K] (model-sorted ascending)
            f_true = batch['Y_freq']           # [B, K] (dataset-sorted ascending)
            if model.freq_stats:
                f_pred_ph = f_pred * model.freq_stats['std'] + model.freq_stats['mean']
                f_true_ph = f_true * model.freq_stats['std'] + model.freq_stats['mean']
            else:
                f_pred_ph, f_true_ph = f_pred, f_true

            B, N, K = preds.shape
            for i in range(B):
                g_id = int(geom_ids[i])
                if g_id in geometries_results:
                    continue
                if len(geometries_results) >= args.num_samples:
                    continue

                m = mask[i] if mask is not None else slice(None)
                fp_i = f_pred_ph[i].cpu().numpy()    # [K] ascending
                ft_i = f_true_ph[i].cpu().numpy()    # [K] ascending

                modes = {}
                for k in range(K):  # k = display index, ascending frequency
                    p_tensor = preds[i, m, k]
                    t_tensor = targets[i, m, k]

                    rel_pos = torch.norm(p_tensor - t_tensor) / (torch.norm(t_tensor) + 1e-8)
                    rel_neg = torch.norm(p_tensor + t_tensor) / (torch.norm(t_tensor) + 1e-8)
                    if rel_neg < rel_pos:
                        rel_l2 = rel_neg.item()
                        final_pred_viz = -p_tensor.cpu().numpy()
                        sign_info = "*"
                    else:
                        rel_l2 = rel_pos.item()
                        final_pred_viz = p_tensor.cpu().numpy()
                        sign_info = ""

                    modes[k] = {
                        'coords': coords[i, m].cpu().numpy(),
                        'target': t_tensor.cpu().numpy(),
                        'pred': final_pred_viz,
                        'f_true': float(ft_i[k]),
                        'f_pred': float(fp_i[k]),
                        'rel_l2': rel_l2,
                        'sign_info': sign_info,
                    }

                deg_note = _near_degenerate_note(ft_i)
                geometries_results[g_id] = {
                    'modes': modes,
                    'elements': elements_pool.get(g_id),
                    'deg_note': deg_note,
                }

            if len(geometries_results) >= args.num_samples:
                break

    print(f"Plotting {len(geometries_results)} geometries...")
    for idx, (g_id, data) in enumerate(geometries_results.items()):
        save_path = os.path.join(args.output_dir, f"sample_geom_{g_id:04d}_all_modes.png")
        plot_geometry_comparison(g_id, data['modes'], save_path,
                                 data.get('elements'), data.get('deg_note', ""))
        note = data.get('deg_note', "")
        print(f"[{idx + 1}/{len(geometries_results)}] Geometry {g_id} ({note}) -> {save_path}")

    print("Inference and grouped visualization completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer and Visualize GNOT Predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .ckpt model checkpoint file")
    parser.add_argument("--data_path", type=str, default="data/gnot_dataset_5k.pkl", help="Path to input dataset")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of geometries to visualize and save as PNG")
    parser.add_argument("--output_dir", type=str, default="inference_plots", help="Output directory for PNG plots")

    parser.add_argument("--freq_mean", type=float, default=None, help="Manual override for frequency mean")
    parser.add_argument("--freq_std", type=float, default=None, help="Manual override for frequency std")

    args = parser.parse_args()
    main(args)
