import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jn_zeros
import argparse
import gmsh
from matplotlib.tri import Triangulation

# Import existing project modules
from src.data.dataset_converter import RFCavityToGNOT
from src.training.lightning_module import GNOTLightning

class AnalyticalValidator:
    def __init__(self, c=299792458):
        self.c = c

    def get_circle_tm010(self, R, coords):
        """
        Analytical solution for TM010 mode in a circular cavity.
        f = c * j01 / (2 * pi * R)
        E_z = J0(j01 * r / R)
        """
        j01 = jn_zeros(0, 1)[0]
        freq = (self.c * j01) / (2 * np.pi * R) / 1e9  # GHz
        
        # Coords are centered at self.extract_geometry_features (0,0)
        # But let's assume raw coords were around (0.05, 0.05) 
        # Actually RFCavityToGNOT centers them.
        r = np.linalg.norm(coords, axis=1)
        field = jv(0, j01 * r / R)
        
        # Sign-Agnostic: ensure peak is positive for comparison
        if np.abs(np.min(field)) > np.max(field):
            field *= -1.0
        
        # Normalize peak to 1.0 (as the model does)
        field = field / (np.abs(field).max() + 1e-10)
        
        return freq, field.reshape(-1, 1)

    def get_square_tm110(self, L, coords):
        """
        Analytical solution for TM110 mode in a square cavity.
        f = (c/2) * sqrt((1/L)^2 + (1/L)^2) = c / (sqrt(2) * L)
        E_z = sin(pi * x / L) * sin(pi * y / L)
        """
        freq = (self.c / (np.sqrt(2) * L)) / 1e9  # GHz
        
        # RFCavityToGNOT centers nodes around 0.
        # Original square [0, L] x [0, L] becomes [-L/2, L/2] x [-L/2, L/2]
        x = coords[:, 0] + L/2
        y = coords[:, 1] + L/2
        
        field = np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
        
        if np.abs(np.min(field)) > np.max(field):
            field *= -1.0
            
        field = field / (np.abs(field).max() + 1e-10)
        
        return freq, field.reshape(-1, 1)

def generate_mesh(shape_type, size_param):
    """Generates mesh nodes and elements for Circle or Square."""
    if not gmsh.isInitialized():
        gmsh.initialize()
    
    gmsh.model.add("analytical_geom")
    gmsh.option.setNumber("General.Terminal", 0)
    
    if shape_type == 'circle':
        # size_param is radius
        cx, cy = 0.05, 0.05
        p1 = gmsh.model.occ.addPoint(cx, cy, 0)
        p2 = gmsh.model.occ.addPoint(cx + size_param, cy, 0)
        p3 = gmsh.model.occ.addPoint(cx, cy + size_param, 0)
        p4 = gmsh.model.occ.addPoint(cx - size_param, cy, 0)
        p5 = gmsh.model.occ.addPoint(cx, cy - size_param, 0)
        
        c1 = gmsh.model.occ.addCircleArc(p2, p1, p3)
        c2 = gmsh.model.occ.addCircleArc(p3, p1, p4)
        c3 = gmsh.model.occ.addCircleArc(p4, p1, p5)
        c4 = gmsh.model.occ.addCircleArc(p5, p1, p2)
        
        loop = gmsh.model.occ.addCurveLoop([c1, c2, c3, c4])
        surf = gmsh.model.occ.addPlaneSurface([loop])
        lines = [c1, c2, c3, c4]
    else:
        # size_param is L
        cx, cy = 0.05, 0.05
        L = size_param
        p1 = gmsh.model.occ.addPoint(cx - L/2, cy - L/2, 0)
        p2 = gmsh.model.occ.addPoint(cx + L/2, cy - L/2, 0)
        p3 = gmsh.model.occ.addPoint(cx + L/2, cy + L/2, 0)
        p4 = gmsh.model.occ.addPoint(cx - L/2, cy + L/2, 0)
        
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)
        
        loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        surf = gmsh.model.occ.addPlaneSurface([loop])
        lines = [l1, l2, l3, l4]

    gmsh.model.occ.synchronize()
    
    # Simple mesh sizing
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.002)
    gmsh.model.mesh.generate(2)
    
    _, coords, _ = gmsh.model.mesh.getNodes()
    _, _, conns = gmsh.model.mesh.getElements(2)
    
    nodes = coords.reshape(-1, 3)[:, :2]
    elements = (conns[0].reshape(-1, 3) - 1).astype(np.int32)
    
    gmsh.model.remove()
    return nodes, elements

def main(args):
    print(f"--- Analytical Validation Started ---")
    
    # 1. Load Model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = GNOTLightning.load_from_checkpoint(args.checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    validator = AnalyticalValidator()
    converter = RFCavityToGNOT("dummy.h5")
    
    shapes = []
    if args.test_circle:
        shapes.append(('circle', args.r_circle))
    if args.test_square:
        shapes.append(('square', args.l_square))
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    for s_type, s_param in shapes:
        print(f"\nEvaluating {s_type} (param={s_param})...")
        
        # 2. Generate Mesh
        nodes, elements = generate_mesh(s_type, s_param)
        print(f"Mesh generated with {len(nodes)} nodes.")
        
        # 3. Extract Features
        # extract_geometry_features internally centers nodes around 0.
        geom_data = converter.extract_geometry_features(nodes, elements)
        x_norm = geom_data['X']
        feat_norm = geom_data['Input_funcs']
        
        # 4. Get Analytical Solution (Ground Truth)
        if s_type == 'circle':
            f_true, field_true = validator.get_circle_tm010(s_param, x_norm)
        else:
            f_true, field_true = validator.get_square_tm110(s_param, x_norm)
        
        # 5. Prepare Inference Batch
        # Mode 0 is the fundamental mode (TM010 or TM110 depending on shape)
        batch = {
            'X': torch.from_numpy(x_norm).float().unsqueeze(0),
            'Input_funcs': torch.from_numpy(feat_norm).float().unsqueeze(0),
            'Theta_in': torch.tensor([[0]], dtype=torch.long),
            'Mask': torch.ones((1, len(nodes)), dtype=torch.bool)
        }
        
        if torch.cuda.is_available():
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
        # 6. Run Inference
        with torch.no_grad():
            outputs = model(batch)
            pred_field = outputs['field'].squeeze(0).cpu().numpy()
            
            # Sign correction for comparison
            rel_pos = np.linalg.norm(pred_field - field_true)
            rel_neg = np.linalg.norm(pred_field + field_true)
            if rel_neg < rel_pos:
                pred_field = -pred_field
            
            # Frequency denormalization
            if model.predict_frequency:
                f_pred_norm = outputs['freq'].item()
                f_pred = f_pred_norm * args.freq_std + args.freq_mean
            else:
                f_pred = 0.0
                
        # 7. Calculate Metrics
        rel_l2 = np.linalg.norm(pred_field - field_true) / (np.linalg.norm(field_true) + 1e-8)
        f_err = np.abs(f_pred - f_true) / f_true * 100 if f_true > 0 else 0
        
        print(f"Results for {s_type}:")
        print(f"  Theoretical Freq: {f_true:.4f} GHz")
        if model.predict_frequency:
            print(f"  Predicted Freq:   {f_pred:.4f} GHz (Error: {f_err:.2f}%)")
        print(f"  Field Rel L2:     {rel_l2:.4f}")
        
        # 8. Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        tri = Triangulation(x_norm[:, 0], x_norm[:, 1], elements)
        
        im0 = axes[0].tripcolor(tri, field_true.flatten(), cmap='RdBu_r', vmin=-1, vmax=1, shading='gouraud')
        axes[0].set_title(f"Analytical (f={f_true:.2f})")
        fig.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].tripcolor(tri, pred_field.flatten(), cmap='RdBu_r', vmin=-1, vmax=1, shading='gouraud')
        axes[1].set_title(f"GNOT Prediction (RelL2={rel_l2:.3f})")
        fig.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].tripcolor(tri, np.abs(pred_field - field_true).flatten(), cmap='viridis')
        axes[2].set_title("Absolute Error Map")
        fig.colorbar(im2, ax=axes[2])
        
        for ax in axes:
            ax.set_aspect('equal')
            ax.axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f"validation_{s_type}.png")
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    print("\n--- Validation Completed ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate GNOT against Analytical Solutions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--output_dir", type=str, default="analytical_validation", help="Output directory")
    
    # Normalization Stats
    parser.add_argument("--freq_mean", type=float, default=3.0, help="Mean freq used for normalization")
    parser.add_argument("--freq_std", type=float, default=1.0, help="Std freq used for normalization")
    
    # Tests
    parser.add_argument("--test_circle", action="store_true", default=True)
    parser.add_argument("--test_square", action="store_true", default=True)
    parser.add_argument("--r_circle", type=float, default=0.04, help="Radius of circular cavity (m)")
    parser.add_argument("--l_square", type=float, default=0.08, help="Side length of square cavity (m)")
    
    args = parser.parse_args()
    main(args)
