import os, gmsh, h5py, numpy as np, matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from skfem import *
from skfem import utils
from skfem.models.poisson import laplace, mass
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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

def generate_sample_data(s_id):
    try:
        gmsh.initialize()
        gmsh.model.add(f"rf_{s_id}")
        np.random.seed(s_id * 13)
        L, cx, cy = 0.1, 0.05, 0.05

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
                t = np.linspace(0, 2*np.pi, 100, endpoint=False)
                r = 0.035 + sum(np.random.uniform(-0.008, 0.008) * np.cos(k*t + np.random.uniform(0, 2*np.pi)) for k in range(2, 8))
                pts_c = [(cx + ri*np.cos(ti), cy + ri*np.sin(ti)) for ri, ti in zip(r, t)]
            elif method == 'pillbox':
                # Endüstriyel RF Kavitelerine benzer: Geniş hücre (body) ve Işın Geçiş Boruları (beam pipes)
                body_w = np.random.uniform(0.05, 0.09)
                body_h = np.random.uniform(0.04, 0.07)
                pipe_w = np.random.uniform(0.015, body_w - 0.01) # boru ana gövdeden ince olmalı
                pipe_top = np.random.uniform(0.01, 0.025)
                pipe_bot = np.random.uniform(0.01, 0.025)
                
                # Koordinat Sınırları (CCW saat yönü tersine dizilim MESH için ZORUNLUDUR)
                x_b_min, x_b_max = cx - body_w/2, cx + body_w/2
                y_b_min, y_b_max = cy - body_h/2, cy + body_h/2
                x_p_min, x_p_max = cx - pipe_w/2, cx + pipe_w/2
                
                # Sol üst köşe ışın borusundan başlayarak Saat Yönünün Tersine (CCW) tüm dış hattı dön
                pts_c = [
                    (x_p_min, y_b_max + pipe_top), # 1. Üst Sol Uç
                    (x_p_min, y_b_max),            # 2. Üst Sol İç Köşe
                    (x_b_min, y_b_max),            # 3. Gövde Üst Sol
                    (x_b_min, y_b_min),            # 4. Gövde Alt Sol
                    (x_p_min, y_b_min),            # 5. Alt Sol İç Köşe
                    (x_p_min, y_b_min - pipe_bot), # 6. Alt Sol Uç
                    (x_p_max, y_b_min - pipe_bot), # 7. Alt Sağ Uç
                    (x_p_max, y_b_min),            # 8. Alt Sağ İç Köşe
                    (x_b_max, y_b_min),            # 9. Gövde Alt Sağ
                    (x_b_max, y_b_max),            # 10. Gövde Üst Sağ
                    (x_p_max, y_b_max),            # 11. Üst Sağ İç Köşe
                    (x_p_max, y_b_max + pipe_top)  # 12. Üst Sağ Uç
                ]
            else: # method == 'elliptical'
                # GERÇEKÇİ TESLA KAVİTESİ: Beam pipe + Yumuşak Eliptik Genişleme
                req = np.random.uniform(0.040, 0.055)    # Ekvator genişliği
                riris = np.random.uniform(0.012, 0.025)  # Beam pipe / Iris genişliği
                L = np.random.uniform(0.04, 0.08)        # Eliptik hücre uzunluğu
                L_pipe = np.random.uniform(0.01, 0.03)   # Dış ışın borusu uzunlukları
                power = np.random.uniform(1.5, 3.0)      # Eğrilik kontrolü (2.0 = düzgün harmonik)
                
                # MESH için Saat Yönünün Tersine (CCW) tüm hücreyi dönmemiz gerekiyor.
                x_curve = np.linspace(L/2, -L/2, 40) # Sağdan Sola üst kavis
                y_top_curve = cy + riris + (req - riris) * (np.cos(x_curve * np.pi / L)**power)
                
                pts_c = []
                # 1. Sağ-Üst Işın Borusu
                pts_c.append((cx + L/2 + L_pipe, cy + riris))
                pts_c.append((cx + L/2, cy + riris))
                # 2. Üst TESLA Kavisi (Sağdan Sola)
                for x, y in zip(x_curve, y_top_curve):
                    pts_c.append((cx + x, y))
                # 3. Sol-Üst Işın Borusu
                pts_c.append((cx - L/2, cy + riris))
                pts_c.append((cx - L/2 - L_pipe, cy + riris))
                # 4. Sol-Alt Işın Borusu
                pts_c.append((cx - L/2 - L_pipe, cy - riris))
                pts_c.append((cx - L/2, cy - riris))
                # 5. Alt TESLA Kavisi (Soldan Sağa dönmeli)
                x_curve_bot = np.linspace(-L/2, L/2, 40)
                y_bot_curve = cy - (riris + (req - riris) * (np.cos(x_curve_bot * np.pi / L)**power))
                for x, y in zip(x_curve_bot, y_bot_curve):
                    pts_c.append((cx + x, y))
                # 6. Sağ-Alt Işın Borusu
                pts_c.append((cx + L/2, cy - riris))
                pts_c.append((cx + L/2 + L_pipe, cy - riris))

        pts = [gmsh.model.occ.addPoint(p[0], p[1], 0) for p in pts_c]
        lines = [gmsh.model.occ.addLine(pts[i], pts[(i+1)%len(pts)]) for i in range(len(pts))]
        gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop(lines)])
        gmsh.model.occ.synchronize()

        # Adaptive Mesh: Dense at boundary, balanced at center
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", lines)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.0012)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", 0.005)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.002)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.03)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(2)
        _, coords, _ = gmsh.model.mesh.getNodes()
        _, _, conns = gmsh.model.mesh.getElements(2)
        nodes = np.ascontiguousarray(coords.reshape(-1, 3)[:, :2])
        elements = np.ascontiguousarray((conns[0].reshape(-1, 3) - 1).astype(np.int32))

        # Physics Solver (P2 Precision)
        m = MeshTri(nodes.T, elements.T)
        basis = Basis(m, ElementTriP2())
        K, M = laplace.assemble(basis), mass.assemble(basis)
        D = basis.get_dofs(facets=m.boundary_facets())
        Kc, Mc, xc, Ic = utils.condense(K, M, D=D, expand=True)
        vals, vecs = utils.solve_eigen(Kc, Mc, x=xc, I=Ic, k=3, sigma=500.0)

        freqs = (299792458 * np.sqrt(np.abs(vals.real))) / (2 * np.pi) / 1e9

        return {
            'id': s_id, 'nodes': nodes, 'elements': elements,
            'freqs': freqs.real, 'vecs': vecs.real, 'n_nodes': len(nodes),
            'shape_type': 'calibration' if ARGS.mode == 'calibration' else 'random'
        }
    except Exception as e:
        return None
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass

def save_sample_plot(data, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    nodes, elements = data['nodes'], data['elements']
    triang = Triangulation(nodes[:, 0], nodes[:, 1], elements)

    # Mesh Panel
    axes[0].triplot(triang, color='gray', linewidth=0.15, alpha=0.5)
    axes[0].set_title(f"ID {data['id']}: {len(elements)} Elements")

    for i in range(3):
        # Taking up to first n_nodes from P2 precision for visualization
        mode_v = data['vecs'][:data['n_nodes'], i]
        mode_norm = mode_v / np.max(np.abs(mode_v))
        axes[i+1].tripcolor(triang, mode_norm, shading='gouraud', cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i+1].set_title(f"Mode {i+1}: {data['freqs'][i]:.4f} GHz")

    for ax in axes: ax.set_aspect('equal'); ax.axis('off')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    ARGS = parse_args()
    
    os.makedirs(ARGS.plot_dir, exist_ok=True)
    print(f"🚀 Generating {ARGS.n_total} samples using {cpu_count()} cores...")

    with h5py.File(ARGS.h5_filename, "w") as f_h5:
        with Pool(cpu_count()) as pool:
            # Process in chunks to manage memory
            for i in tqdm(range(0, ARGS.n_total, 50), desc="Batch Generation"):
                chunk_range = range(i, min(i + 50, ARGS.n_total))
                chunk_results = pool.map(generate_sample_data, chunk_range)

                for res in chunk_results:
                    if res is None: continue

                    # H5 Save (Gzip compression)
                    grp = f_h5.create_group(f"sample_{res['id']:04d}")
                    grp.create_dataset("nodes", data=res['nodes'], compression="gzip", compression_opts=4)
                    grp.create_dataset("elements", data=res['elements'], compression="gzip", compression_opts=4)
                    grp.create_dataset("freqs", data=res['freqs'])
                    grp.create_dataset("vecs", data=res['vecs'], compression="gzip", compression_opts=4)
                    grp.attrs['shape_type'] = res['shape_type']

                    # Plot first N_PLOT
                    if res['id'] < ARGS.n_plot:
                        plot_path = os.path.join(ARGS.plot_dir, f"sample_{res['id']:03d}.png")
                        save_sample_plot(res, plot_path)

    print(f"\n✅ Completed! \n📦 File: {ARGS.h5_filename} \n🖼️ Plots: {ARGS.plot_dir}/")
