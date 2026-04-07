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
    dataset = GNOTDataset(args.data_path, split=args.split)
    
    # Put frequency stats from dataset to model (important for denormalizing freq predictions)
    if hasattr(dataset, 'stats') and dataset.stats:
        model.freq_stats = dataset.stats

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=gnot_collate_fn)

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running inference and generating {args.num_samples} visualization plots...")
    
    samples_plotted = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if torch.cuda.is_available():
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
            outputs = model(batch)
            preds = outputs['field']
            targets = batch['Y_field']
            coords = batch['X']
            mask = batch.get('Mask', None)
            
            # De-normalize predicted frequencies if available
            if model.freq_stats:
                freq_preds = outputs['freq'] * model.freq_stats['std'] + model.freq_stats['mean']
                freq_trues = batch['Y_freq'] * model.freq_stats['std'] + model.freq_stats['mean']
            else:
                freq_preds = outputs['freq']
                freq_trues = batch['Y_freq']

            B = preds.shape[0]
            for i in range(B):
                m = mask[i] if mask is not None else slice(None)
                
                # Sign-Agnostic Logic for Inference Metrics & Visualization
                p_tensor = preds[i, m]
                t_tensor = targets[i, m]
                
                rel_pos = torch.norm(p_tensor - t_tensor) / (torch.norm(t_tensor) + 1e-8)
                rel_neg = torch.norm(p_tensor + t_tensor) / (torch.norm(t_tensor) + 1e-8)
                
                # En iyi işareti seç (faz keyfiliğini görselde yenmek için)
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
                
                f_true = freq_trues[i, 0].item()
                f_pred = freq_preds[i, 0].item()
                f_err = abs(f_true - f_pred)
                
                # Extraction of mode index
                m_idx = int(batch['Theta_in'][i].item())
                
                title = f"Mode {m_idx} | Sample {samples_plotted}{sign_info}\nFreq True: {f_true:.2f}GHz, Pred: {f_pred:.2f}GHz (Err: {f_err:.3f})\nField Rel L2: {rel_l2:.3f}"
                save_path = os.path.join(args.output_dir, f"{args.split}_mode{m_idx}_sample_{samples_plotted}.png")
                
                plot_single_comparison(valid_coords, valid_targets, final_pred_viz, title, save_path)
                print(f"[{samples_plotted+1}/{args.num_samples}] Saved visualization to {save_path}")
                
                samples_plotted += 1
                if samples_plotted >= args.num_samples:
                    print("Inference completed successfully!")
                    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer and Visualize GNOT Predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .ckpt model checkpoint file")
    parser.add_argument("--data_path", type=str, default="data/gnot_dataset.pkl", help="Path to input dataset")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize and save as PNG")
    parser.add_argument("--output_dir", type=str, default="inference_plots", help="Output directory for PNG plots")
    
    args = parser.parse_args()
    main(args)
