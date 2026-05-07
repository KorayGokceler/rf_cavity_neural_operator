import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from torch.utils.data import DataLoader

from src.data.dataset import GNOTDataset, gnot_collate_fn
from src.training.lightning_module import GNOTLightning

def plot_geometry_comparison(geom_id, modes_data, save_path):
    """
    Plots all modes of a geometry in a single figure.
    Each mode gets a row with Ground Truth, Prediction, and Error columns.
    """
    n_modes = len(modes_data)
    # Sort modes by their index
    sorted_modes = sorted(modes_data.items())

    fig, axes = plt.subplots(n_modes, 3, figsize=(18, 5 * n_modes), squeeze=False)
    fig.suptitle(f"Geometry ID: {geom_id}", fontsize=16, fontweight='bold', y=0.98)

    for i, (m_idx, data) in enumerate(sorted_modes):
        coords = data['coords']
        target = data['target'].flatten()
        pred = data['pred'].flatten()
        f_true = data['f_true']
        f_pred = data['f_pred']
        rel_l2 = data['rel_l2']
        sign_info = data['sign_info']

        tri = Triangulation(coords[:, 0], coords[:, 1])

        # Ground Truth
        im1 = axes[i, 0].tripcolor(tri, target, cmap='RdBu_r', shading='gouraud', vmin=-1, vmax=1)
        axes[i, 0].set_title(f"Mode {m_idx} - Ground Truth\nFreq: {f_true:.2f} GHz", fontsize=12)
        fig.colorbar(im1, ax=axes[i, 0])

        # Prediction
        im2 = axes[i, 1].tripcolor(tri, pred, cmap='RdBu_r', shading='gouraud', vmin=-1, vmax=1)
        axes[i, 1].set_title(f"Mode {m_idx} - GNOT Prediction{sign_info}\nFreq: {f_pred:.2f} GHz (Rel L2: {rel_l2:.3f})", fontsize=12)
        fig.colorbar(im2, ax=axes[i, 1])

        # Error (pred - truth) — symmetric colormap centred at 0
        error = pred - target
        err_max = max(np.abs(error).max(), 1e-8)
        im3 = axes[i, 2].tripcolor(tri, error, cmap='RdBu_r', shading='gouraud', vmin=-err_max, vmax=err_max)
        axes[i, 2].set_title(f"Mode {m_idx} - Error (pred - truth)\nmax |err|: {err_max:.3f}", fontsize=12)
        fig.colorbar(im3, ax=axes[i, 2])

        for ax in axes[i]:
            ax.set_aspect('equal')
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

def main(args):
    print(f"Loading checkpoint from: {args.checkpoint}")
    # Load model from checkpoint
    model = GNOTLightning.load_from_checkpoint(args.checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    print(f"Loading dataset from: {args.data_path}")
    dataset = GNOTDataset(args.data_path, split=args.split)
    
    # Put frequency stats from dataset to model (important for denormalizing freq predictions)
    if hasattr(dataset, 'stats') and dataset.stats:
        model.freq_stats = dataset.stats
    
    # Manual override
    if args.freq_mean is not None and args.freq_std is not None:
        model.freq_stats = {'mean': args.freq_mean, 'std': args.freq_std}
        print(f"Using manual frequency stats override: mean={args.freq_mean}, std={args.freq_std}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=gnot_collate_fn)

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running inference for {args.num_samples} geometries...")
    
    geometries_results = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if torch.cuda.is_available():
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
            outputs = model(batch)
            preds = outputs['field']
            targets = batch['Y_field']
            coords = batch['X']
            mask = batch.get('Mask', None)
            geom_ids = batch['geom_id'].squeeze(-1).cpu().numpy()
            theta_ins = batch['Theta_in'].squeeze(-1).cpu().numpy()
            
            # De-normalize predicted frequencies if available
            if outputs.get('freq') is not None and model.freq_stats:
                freq_preds = outputs['freq'] * model.freq_stats['std'] + model.freq_stats['mean']
                freq_trues = batch['Y_freq'] * model.freq_stats['std'] + model.freq_stats['mean']
            elif outputs.get('freq') is not None:
                freq_preds = outputs['freq']
                freq_trues = batch['Y_freq']
            else:
                freq_preds = batch['Y_freq'] * 0  # Zeros placeholder
                freq_trues = batch['Y_freq']

            B = preds.shape[0]
            for i in range(B):
                g_id = int(geom_ids[i])
                m_idx = int(theta_ins[i])
                
                # Check if we already have enough geometries
                if g_id not in geometries_results and len(geometries_results) >= args.num_samples:
                    continue
                
                m = mask[i] if mask is not None else slice(None)
                p_tensor = preds[i, m]
                t_tensor = targets[i, m]
                
                # Sign-Agnostic selection
                rel_pos = torch.norm(p_tensor - t_tensor) / (torch.norm(t_tensor) + 1e-8)
                rel_neg = torch.norm(p_tensor + t_tensor) / (torch.norm(t_tensor) + 1e-8)
                
                if rel_neg < rel_pos:
                    rel_l2 = rel_neg.item()
                    final_pred_viz = -p_tensor.cpu().numpy()
                    sign_info = "*" # mini indicator for sign flip
                else:
                    rel_l2 = rel_pos.item()
                    final_pred_viz = p_tensor.cpu().numpy()
                    sign_info = ""

                if g_id not in geometries_results:
                    geometries_results[g_id] = {'modes': {}}
                
                geometries_results[g_id]['modes'][m_idx] = {
                    'coords': coords[i, m].cpu().numpy(),
                    'target': t_tensor.cpu().numpy(),
                    'pred': final_pred_viz,
                    'f_true': freq_trues[i, 0].item(),
                    'f_pred': freq_preds[i, 0].item(),
                    'rel_l2': rel_l2,
                    'sign_info': sign_info
                }

            # Break early if we collected all needed geometries and they all have 3 modes
            # (or whatever number of modes is expected)
            all_complete = len(geometries_results) >= args.num_samples
            if all_complete:
                # Check if each geometry has at least some modes (e.g. 3)
                for res in geometries_results.values():
                    if len(res['modes']) < 3: # Assuming 3 modes is standard
                        all_complete = False
                        break
                if all_complete: break

    # Now plot the grouped results
    print(f"Plotting {len(geometries_results)} geometries...")
    for idx, (g_id, data) in enumerate(geometries_results.items()):
        save_path = os.path.join(args.output_dir, f"sample_geom_{g_id:04d}_all_modes.png")
        plot_geometry_comparison(g_id, data['modes'], save_path)
        print(f"[{idx+1}/{len(geometries_results)}] Saved grouped plot for Geometry {g_id} to {save_path}")

    print("Inference and grouped visualization completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer and Visualize GNOT Predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .ckpt model checkpoint file")
    parser.add_argument("--data_path", type=str, default="data/gnot_dataset.pkl", help="Path to input dataset")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize and save as PNG")
    parser.add_argument("--output_dir", type=str, default="inference_plots", help="Output directory for PNG plots")
    
    # Frequency stats override
    parser.add_argument("--freq_mean", type=float, default=None, help="Manual override for frequency mean")
    parser.add_argument("--freq_std", type=float, default=None, help="Manual override for frequency std")
    
    args = parser.parse_args()
    main(args)
