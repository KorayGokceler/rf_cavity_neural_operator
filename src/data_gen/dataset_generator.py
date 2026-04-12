import os, gmsh, h5py, numpy as np, matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from skfem import *
from skfem import utils
from skfem.models.poisson import laplace, mass
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate 2D RF Cavity Dataset.")
    parser.add_argument("--h5_filename", type=str, default="rf_cavity_1000_dataset.h5", help="Output H5 filename.")
    parser.add_argument("--plot_dir", type=str, default="dataset_plots", help="Directory to save plots.")
    parser.add_argument("--n_total", type=int, default=1000, help="Total number of samples to generate.")
    parser.add_argument("--n_plot", type=int, default=100, help="Number of samples to plot.")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "calibration"], help="Generation mode: random shapes or calibration shapes (square/circle).")
    return parser.parse_args()

# Global settings placeholder
ARGS = None

def worker_init():
    """Initialize Gmsh once per worker process to save time and prevent hangs."""
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)  # Only errors
    except Exception:
        pass

def generate_sample_data(s_id):
    try:
        # Clear existing models and initialize for this specific task
        if not gmsh.isInitialized():
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 1)

        model_name = f"rf_{s_id}_{os.getpid()}"
        if model_name in gmsh.model.list():
            gmsh.model.remove()
        gmsh.model.add(model_name)
        
        np.random.seed(s_id * 13 + 7)
        cx, cy = 0.05, 0.05

        # Geometric Diversity: Sharp corners and chaotic blobs
        if ARGS.mode == 'calibration':
            # Calibration: Square or Circle
            shape_type = 'square' if s_id % 2 == 0 else 'circle'
            if shape_type == 'square':
                side = 0.08
                pts_c = [
                    (cx - side/2, cy - side/2),
                    (cx + side/2, cy - side/2),
                    (cx + side/2, cy + side/2),
                    (cx - side/2, cy + side/2)
                ]
            else:
                radius = 0.04
                t = np.linspace(0, 2*np.pi, 100, endpoint=False)
                pts_c = [(cx + radius*np.cos(ti), cy + radius*np.sin(ti)) for ti in t]
        else:
            method = np.random.choice(['sharp', 'smooth', 'pillbox', 'elliptical'], p=[0.20, 0.20, 0.30, 0.30])
            if method == 'sharp':
                n_pts = np.random.randint(7, 13)
                angles = np.sort(np.random.uniform(0, 2*np.pi, n_pts))
                r = np.random.uniform(0.02, 0.046, n_pts)
                pts_c = [(cx + ri*np.cos(ai), cy + ri*np.sin(ai)) for ri, ai in zip(r, angles)]
            elif method == 'smooth':
                t = np.linspace(0, 2*np.pi, 120, endpoint=False)
                r_fluctuation = sum(np.random.uniform(-0.009, 0.009) * np.cos(k*t + np.random.uniform(0, 2*np.pi)) for k in range(2, 8))
                r = np.maximum(0.035 + r_fluctuation, 0.012) 
                pts_c = [(cx + ri*np.cos(ti), cy + ri*np.sin(ti)) for ri, ti in zip(r, t)]
            elif method == 'pillbox':
                body_w = np.random.uniform(0.05, 0.09)
                body_h = np.random.uniform(0.04, 0.07)
                pipe_w = np.random.uniform(0.015, body_w - 0.01) 
                pipe_top = np.random.uniform(0.01, 0.025)
                pipe_bot = np.random.uniform(0.01, 0.025)
                x_b_min, x_b_max = cx - body_w/2, cx + body_w/2
                y_b_min, y_b_max = cy - body_h/2, cy + body_h/2
                x_p_min, x_p_max = cx - pipe_w/2, cx + pipe_w/2
                pts_c = [
                    (x_p_min, y_b_max + pipe_top), (x_p_min, y_b_max),
                    (x_b_min, y_b_max), (x_b_min, y_b_min),
                    (x_p_min, y_b_min), (x_p_min, y_b_min - pipe_bot),
                    (x_p_max, y_b_min - pipe_bot), (x_p_max, y_b_min),
                    (x_b_max, y_b_min), (x_b_max, y_b_max),
                    (x_p_max, y_b_max), (x_p_max, y_b_max + pipe_top)
                ]
            else: # elliptical
                req = np.random.uniform(0.040, 0.055)
                riris = np.random.uniform(0.012, 0.025)
                L_cell = np.random.uniform(0.04, 0.08)
                L_pipe = np.random.uniform(0.01, 0.03)
                power = np.random.uniform(1.8, 2.5)
                x_curve = np.linspace(L_cell/2, -L_cell/2, 40)
                y_top_curve = cy + riris + (req - riris) * (np.abs(np.cos(x_curve * np.pi / L_cell))**power)
                pts_c = []
                pts_c.append((cx + L_cell/2 + L_pipe, cy + riris))
                for x, y in zip(x_curve, y_top_curve): pts_c.append((cx + x, y))
                pts_c.append((cx - L_cell/2 - L_pipe, cy + riris))
                pts_c.append((cx - L_cell/2 - L_pipe, cy - riris))
                x_curve_bot = np.linspace(-L_cell/2, L_cell/2, 40)
                y_bot_curve = cy - (riris + (req - riris) * (np.abs(np.cos(x_curve_bot * np.pi / L_cell))**power))
                for x, y in zip(x_curve_bot, y_bot_curve): pts_c.append((cx + x, y))
                pts_c.append((cx + L_cell/2 + L_pipe, cy - riris))

        pts = [gmsh.model.occ.addPoint(p[0], p[1], 0) for p in pts_c]
        lines = [gmsh.model.occ.addLine(pts[i], pts[(i+1)%len(pts)]) for i in range(len(pts))]
        gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop(lines)])
        gmsh.model.occ.synchronize()

        # Fine P1 Mesh (Robust & Uniformly DENSE)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", lines)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.0010) # 1mm precision
        gmsh.model.mesh.field.setNumber(2, "SizeMax", 0.0035) # 3.5mm interior max
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.002)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.04)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(2)
        _, coords, _ = gmsh.model.mesh.getNodes()
        _, _, conns = gmsh.model.mesh.getElements(2)
        
        if len(coords) == 0 or len(conns[0]) == 0:
            return None
            
        nodes = np.ascontiguousarray(coords.reshape(-1, 3)[:, :2])
        elements = np.ascontiguousarray((conns[0].reshape(-1, 3) - 1).astype(np.int32))

        # Physics Solver (P1 Precision - 1:1 Node:Element mapping)
        m = MeshTri(nodes.T, elements.T)
        basis = Basis(m, ElementTriP1())
        K, M = laplace.assemble(basis), mass.assemble(basis)
        D = basis.get_dofs(facets=m.boundary_facets())
        Kc, Mc, xc, Ic = utils.condense(K, M, D=D, expand=True)
        vals, vecs = utils.solve_eigen(Kc, Mc, x=xc, I=Ic, k=3, sigma=300.0) # Shift closer to fundamental

        freqs = (299792458 * np.sqrt(np.abs(vals.real))) / (2 * np.pi) / 1e9
        
        return {
            'id': s_id, 
            'nodes': nodes.astype(np.float32), 
            'elements': elements,
            'freqs': freqs.real, 
            'vecs': vecs.real, 
            'n_nodes': len(nodes),
            'shape_type': 'calibration' if ARGS.mode == 'calibration' else 'random'
        }
    except Exception as e:
        # print(f"Error in worker {s_id}: {str(e)}")
        return None
    finally:
        try:
            gmsh.model.remove()
        except Exception:
            pass

def save_sample_plot(data, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    nodes, elements = data['nodes'], data['elements']
    triang = Triangulation(nodes[:, 0], nodes[:, 1], elements)

    for i in range(4):
        if i == 0:
            axes[i].triplot(triang, color='gray', linewidth=0.15, alpha=0.5)
            axes[i].set_title(f"ID {data['id']}: {len(elements)} Elements")
        else:
            mode_v = data['vecs'][:, i-1]
            mode_norm = mode_v / (np.max(np.abs(mode_v)) + 1e-10)
            axes[i].tripcolor(triang, mode_norm, shading='gouraud', cmap='RdBu_r', vmin=-1, vmax=1)
            axes[i].set_title(f"Mode {i}: {data['freqs'][i-1]:.4f} GHz")

    for ax in axes: ax.set_aspect('equal'); ax.axis('off')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    ARGS = parse_args()
    os.makedirs(ARGS.plot_dir, exist_ok=True)
    print(f"🚀 Generating {ARGS.n_total} samples using {cpu_count()} cores...")

    import json
    metadata = {'n_samples': ARGS.n_total, 'mode': ARGS.mode}

    with h5py.File(ARGS.h5_filename, "w") as f_h5:
        f_h5.attrs['metadata'] = json.dumps(metadata)
        n_workers = min(cpu_count(), 8) # slightly more workers for P1
        print(f"Starting robust generation with {n_workers} workers...")
        
        with Pool(n_workers, initializer=worker_init) as pool:
            successful_samples = 0
            # imap_unordered is key to avoiding hangs! 
            # It yields results as they finish, regardless of input order.
            # We ask for a bit more samples (1.2x) to account for failed geometries.
            task_range = range(int(ARGS.n_total * 1.5)) 
            pbar = tqdm(total=ARGS.n_total, desc="Generating Valid Geometries")
            
            # Using imap_unordered for a stream of results
            for res in pool.imap_unordered(generate_sample_data, task_range):
                if res is None: continue
                if successful_samples >= ARGS.n_total: break

                grp = f_h5.create_group(f"samples/{res['id']}")
                grp.create_dataset("nodes", data=res['nodes'], compression="gzip", compression_opts=4)
                grp.create_dataset("elements", data=res['elements'], compression="gzip", compression_opts=4)
                grp.create_dataset("freqs", data=res['freqs'])
                grp.create_dataset("vecs", data=res['vecs'], compression="gzip", compression_opts=4)
                grp.attrs['shape_type'] = res['shape_type']
                grp.attrs['geom_id'] = res['id']

                if successful_samples < ARGS.n_plot:
                    plot_path = os.path.join(ARGS.plot_dir, f"sample_{res['id']:03d}.png")
                    save_sample_plot(res, plot_path)
            
                successful_samples += 1
                pbar.update(1)
                if successful_samples % 20 == 0: f_h5.flush()
                
            pbar.close()
                
    print(f"\n✅ All {ARGS.n_total} samples successfully generated!")
    os._exit(0)
