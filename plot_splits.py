import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from tqdm import tqdm
from src.data.dataset import GNOTDataset
from src.config import load_config

def plot_split_geometries(dataset, split_name, save_dir, n_geometries=50):
    os.makedirs(save_dir, exist_ok=True)
    try:
        available_real_indices = dataset.active_indices
        geom_dict = {}
        
        # Geometrileri ve sahip olduklari mod dizinlerini grupla
        if dataset.is_h5:
            f = dataset._get_h5_handle()
            for idx in available_real_indices:
                sample = f['samples'][str(idx)]
                geom_id = sample.attrs['geom_id']
                mode_idx = sample['Theta'][0]
                if geom_id not in geom_dict:
                    geom_dict[geom_id] = {}
                geom_dict[geom_id][mode_idx] = idx
        else:
            for idx in available_real_indices:
                sample = dataset.samples_metadata[idx]
                geom_id = sample['geom_id']
                mode_idx = int(sample['Theta'][0])
                if geom_id not in geom_dict:
                    geom_dict[geom_id] = {}
                geom_dict[geom_id][mode_idx] = sample

        geoms_to_plot = list(geom_dict.items())[:n_geometries]
        print(f"Plotting {len(geoms_to_plot)} samples for {split_name} split...")

        for geom_id, modes in tqdm(geoms_to_plot, desc=f"{split_name} Split"):
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(f"{split_name.capitalize()} Data | Geometry ID: {geom_id}", fontsize=16, fontweight='bold', y=0.98)
            
            if dataset.is_h5:
                geom = f['geometry_pool'][str(geom_id)]
                x = geom['X'][:]
                triang = Triangulation(x[:,0], x[:,1])
                
                # Mesh Cizimi
                axes[0].triplot(triang, color='gray', linewidth=0.15, alpha=0.5)
                axes[0].set_title(f"Mesh", fontsize=14)
                axes[0].set_aspect('equal')
                axes[0].axis('off')
                
                # Mod Cizimleri
                for i in range(3):
                    ax = axes[i+1]
                    if i in modes:
                        idx = modes[i]
                        sample = f['samples'][str(idx)]
                        y_field = sample['Y'][:, 0]
                        freq = sample['Theta'][1]
                        
                        tc = ax.tripcolor(triang, y_field, shading='gouraud', cmap='RdBu_r', vmin=-1, vmax=1)
                        ax.set_title(f"Mode {i} ({freq:.4f} GHz)", fontsize=14)
                    else:
                        ax.set_title(f"Mode {i} (Not in {split_name})", fontsize=14)
                    ax.set_aspect('equal')
                    ax.axis('off')
                    
            else:
                geom = dataset.geometry_pool[geom_id]
                x = geom['X']
                elements = geom.get('elements', None)
                if elements is not None:
                     triang = Triangulation(x[:,0], x[:,1], elements)
                else:
                     triang = Triangulation(x[:,0], x[:,1])

                axes[0].triplot(triang, color='gray', linewidth=0.15, alpha=0.5)
                axes[0].set_title(f"Mesh", fontsize=14)
                axes[0].set_aspect('equal')
                axes[0].axis('off')
                
                for i in range(3):
                    ax = axes[i+1]
                    if i in modes:
                        sample = modes[i]
                        y_field = sample['Y'][:, 0]
                        freq = sample['Theta'][1]
                        
                        tc = ax.tripcolor(triang, y_field, shading='gouraud', cmap='RdBu_r', vmin=-1, vmax=1)
                        ax.set_title(f"Mode {i} ({freq:.4f} GHz)", fontsize=14)
                    else:
                        ax.set_title(f"Mode {i} (Not in {split_name})", fontsize=14)
                    ax.set_aspect('equal')
                    ax.axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = os.path.join(save_dir, f"geom_{geom_id:04d}.png")
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()

    except Exception as e:
        print(f"Error plotting {split_name} split: {e}")

def main():
    parser = argparse.ArgumentParser("Visualize Train and Validation Splits individually")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of geometries to plot per split")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_path = cfg.dataset.data_path
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset {data_path} not found.")
        print("Please generate the dataset and convert it first!")
        print("Steps:")
        print("1. python src/data_gen/dataset_generator.py")
        print("2. python convert.py")
        return

    print("Loading Train Split...")
    train_dataset = GNOTDataset(data_path, split='train', train_ratio=cfg.dataset.train_ratio, val_ratio=cfg.dataset.val_ratio)
    print("Loading Validation Split...")
    val_dataset = GNOTDataset(data_path, split='val', train_ratio=cfg.dataset.train_ratio, val_ratio=cfg.dataset.val_ratio)

    print(f"\nPlotting Training Samples...")
    plot_split_geometries(train_dataset, "Train", "dataset_plots/train_split", n_geometries=args.n_samples)
    
    print(f"\nPlotting Validation Samples...")
    plot_split_geometries(val_dataset, "Validation", "dataset_plots/val_split", n_geometries=args.n_samples)
    
    print("\n🎉 All previews matched and saved individually!")

if __name__ == '__main__':
    main()
