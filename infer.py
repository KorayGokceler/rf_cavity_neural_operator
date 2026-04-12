import torch
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from torch.utils.data import DataLoader

from src.data.dataset import GNOTDataset, gnot_collate_fn
from src.training.lightning_module import GNOTLightning

def plot_single_comparison(coords, target, pred, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    tri = Triangulation(coords[:, 0], coords[:, 1])
    
    im1 = axes[0].tripcolor(tri, target.flatten(), cmap='RdBu_r', shading='gouraud')
    axes[0].set_title("Ground Truth (Simulated)")
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].tripcolor(tri, pred.flatten(), cmap='RdBu_r', shading='gouraud')
    axes[1].set_title("GNOT Prediction")
    fig.colorbar(im2, ax=axes[1])
    
    for ax in axes:
        ax.set_aspect('equal')
        ax.axis('off')
        
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main(args):
    print(f"Loading checkpoint from: {args.checkpoint}")
    # Load model from checkpoint
    model = GNOTLightning.load_from_checkpoint(args.checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    print(f"Loading dataset from: {args.data_path}")
    # EXPLICIT FULL MESH: max_nodes=None ensures we take all nodes from H5/PKL
    dataset = GNOTDataset(args.data_path, split=args.split, max_nodes=None)
    
    # Put frequency stats from dataset to model
    if hasattr(dataset, 'stats') and dataset.stats:
        model.freq_stats = dataset.stats

    # batch_size=1 is recommended for full-mesh inference to avoid VRAM peaks
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=gnot_collate_fn)

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running FULL MESH inference and generating {args.num_samples} plots...")
    
    samples_plotted = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if torch.cuda.is_available():
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
            outputs = model(batch)
            preds = outputs['field']      # [B, N, 1]
            targets = batch['Y_field']    # [B, N, 1]
            coords = batch['X']           # [B, N, 2]
            mask = batch.get('Mask', None)
            
            # De-normalize frequencies (Only if available)
            if outputs['freq'] is not None:
                if model.freq_stats:
                    freq_preds = outputs['freq'] * model.freq_stats['std'] + model.freq_stats['mean']
                    freq_trues = batch['Y_freq'] * model.freq_stats['std'] + model.freq_stats['mean']
                else:
                    freq_preds = outputs['freq']
                    freq_trues = batch['Y_freq']
            else:
                freq_preds = None
                freq_trues = None

            B = preds.shape[0]
            for i in range(B):
                m = mask[i] if mask is not None else slice(None)
                
                # Sign-Agnostic Logic for Visualization
                p_tensor = preds[i, m]
                t_tensor = targets[i, m]
                
                rel_pos = torch.norm(p_tensor - t_tensor) / (torch.norm(t_tensor) + 1e-8)
                rel_neg = torch.norm(p_tensor + t_tensor) / (torch.norm(t_tensor) + 1e-8)
                
                if rel_neg < rel_pos:
                    rel_l2 = rel_neg.item()
                    final_pred_viz = -p_tensor.cpu().numpy()
                    sign_info = " (Flipped for Viz)"
                else:
                    rel_l2 = rel_pos.item()
                    final_pred_viz = p_tensor.cpu().numpy()
                    sign_info = ""

                valid_coords = coords[i, m].cpu().numpy()
                valid_targets = t_tensor.cpu().numpy()
                n_nodes = valid_coords.shape[0]
                
                m_idx = int(batch['Theta_in'][i].item())
                
                if freq_preds is not None:
                    f_true = freq_trues[i, 0].item()
                    f_pred = freq_preds[i, 0].item()
                    f_err = abs(f_true - f_pred)
                    title = f"Mode {m_idx} | Nodes: {n_nodes} | {sign_info}\nFreq True: {f_true:.2f}GHz, Pred: {f_pred:.2f}GHz (Err: {f_err:.3f})\nField Rel L2: {rel_l2:.3f}"
                else:
                    title = f"Mode {m_idx} | Nodes: {n_nodes} | {sign_info}\nField Rel L2: {rel_l2:.3f}"
                save_path = os.path.join(args.output_dir, f"{args.split}_mode{m_idx}_sample_{samples_plotted}.png")
                
                plot_single_comparison(valid_coords, valid_targets, final_pred_viz, title, save_path)
                print(f"[{samples_plotted+1}/{args.num_samples}] Saved FULL MESH plot ({n_nodes} nodes) to {save_path}")
                
                samples_plotted += 1
                if samples_plotted >= args.num_samples:
                    print("\nFull mesh inference completed successfully.")
                    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Mesh GNOT Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_path", type=str, default="data/gnot_dataset.pkl", help="Dataset path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (1 is safest for full mesh)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str, default="inference_full_res", help="Output directory")
    
    args = parser.parse_args()
    main(args)
