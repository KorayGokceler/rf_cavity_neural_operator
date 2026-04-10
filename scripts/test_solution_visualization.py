import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.data_gen.dataset_generator as dg
from src.data_gen.dataset_generator import generate_sample_data

def main():
    # Setup mock args for the generator
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="random")
    dg.ARGS = parser.parse_args()
    
    # We will generate 4 random samples
    samples = []
    print("Generating 4 geometries and solving Maxwell equations. Please wait...")
    for i in range(4):
        # generate_sample_data takes an integer ID
        data = generate_sample_data(i + 100)
        if data is not None:
            samples.append(data)
            print(f"Sample {i+1} solved! Freq: {data['freqs'][0]:.4f} GHz")
        else:
            print(f"Sample {i+1} failed solving.")

    if not samples:
        print("No samples generated.")
        return

    # Plotting
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, data in enumerate(samples):
        nodes = data['nodes']
        elements = data['elements']
        vecs = data['vecs']
        
        # Fundamental mode is the first column
        # vecs length matches ndof (P2 has edge nodes), but nodes matches Gmsh vertices
        # We slice to match node count for tripcolor
        E_field = vecs[:len(nodes), 0]
        E_field = np.abs(E_field) # magnitude
        
        # Plot Mesh
        ax_mesh = axes[0, i]
        ax_mesh.triplot(nodes[:, 0], nodes[:, 1], elements, color='gray', linewidth=0.2)
        ax_mesh.set_aspect('equal')
        ax_mesh.set_title(f"Mesh {i+1}")
        ax_mesh.axis('off')
        
        # Plot Field Solution (contour plot)
        ax_field = axes[1, i]
        tpc = ax_field.tripcolor(nodes[:, 0], nodes[:, 1], elements, E_field, shading='flat', cmap='jet')
        ax_field.set_aspect('equal')
        ax_field.set_title(f"Elec Field (TM010)\nFreq: {data['freqs'][0]:.4f} GHz")
        ax_field.axis('off')

    plt.tight_layout()
    output_path = "solutions_demo.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSolutions visual saved to {output_path}")

if __name__ == '__main__':
    main()
