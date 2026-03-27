import h5py
import numpy as np
import argparse
import os

def calculate_analytical_freq(shape_type, mode_idx):
    c = 299792458
    if shape_type == 'square':
        # Side a = 0.08
        a = 0.08
        # TM modes: (1,1), (1,2)/(2,1), (2,2) ...
        # Based on sorted frequencies, we expect:
        # idx 0: (1,1)
        # idx 1: (1,2) or (2,1)
        # idx 2: (1,2) or (2,1)
        if mode_idx == 0:
            m, n = 1, 1
        elif mode_idx == 1 or mode_idx == 2:
            m, n = 1, 2
        else:
            return None
        return (c / 2) * np.sqrt((m/a)**2 + (n/a)**2) / 1e9
    
    elif shape_type == 'circle':
        # Radius R = 0.04
        R = 0.04
        # Bessel function zeros j_mn
        # idx 0: j_01 = 2.4048
        # idx 1: j_11 = 3.8317
        # idx 2: j_11 = 3.8317 (degenerate)
        if mode_idx == 0:
            j = 2.40482
        elif mode_idx == 1 or mode_idx == 2:
            j = 3.83171
        else:
            return None
        return (c * j) / (2 * np.pi * R) / 1e9
    return None

def check_mesh_quality(nodes, elements):
    # Calculate aspect ratio of triangles
    # AR = (a*b*c) / (8 * (s-a)(s-b)(s-c) * R_in) -- too complex, use simpler one:
    # AR = longest_edge / shortest_altitude or just max_edge / min_edge
    qualities = []
    for tri in elements:
        pts = nodes[tri]
        edges = [
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[1] - pts[2]),
            np.linalg.norm(pts[2] - pts[0])
        ]
        # Quality measure: 2 * (area / semi-perimeter) / max_edge (inradius normalized)
        # Or simpler: min_angle / max_angle
        # Let's just use max_edge / min_edge as a proxy
        qualities.append(max(edges) / (min(edges) + 1e-10))
    return np.mean(qualities), np.max(qualities)

def validate_dataset(h5_path):
    print(f"🔍 Validating Dataset: {h5_path}")
    if not os.path.exists(h5_path):
        print(f"❌ File not found: {h5_path}")
        return

    with h5py.File(h5_path, 'r') as f:
        keys = sorted(f.keys())
        print(f"Total Samples: {len(keys)}")
        
        results = []
        mesh_stats = []
        
        for k in keys:
            grp = f[k]
            nodes = grp['nodes'][:]
            elements = grp['elements'][:]
            freqs = grp['freqs'][:]
            shape_type_attr = grp.attrs.get('shape_type', 'unknown')
            
            # Determine if it's a calibration shape based on id if attr is generic
            s_id = int(k.split('_')[-1])
            if shape_type_attr == 'calibration':
                actual_shape = 'square' if s_id % 2 == 0 else 'circle'
            else:
                actual_shape = 'random'
            
            # Mesh quality
            avg_qr, max_qr = check_mesh_quality(nodes, elements)
            mesh_stats.append((avg_qr, max_qr))
            
            if actual_shape != 'random':
                for i in range(min(3, len(freqs))):
                    analytical = calculate_analytical_freq(actual_shape, i)
                    if analytical:
                        error = abs(freqs[i] - analytical) / analytical * 100
                        results.append({
                            'id': k,
                            'shape': actual_shape,
                            'mode': i,
                            'calc': freqs[i],
                            'target': analytical,
                            'error_pct': error
                        })

        # Report Results
        print("\n--- Physical Validation (Analytical Comparison) ---")
        if not results:
            print("No calibration samples found to compare with analytical solutions.")
        else:
            print(f"{'ID':<15} {'Shape':<10} {'Mode':<5} {'Calc (GHz)':<12} {'Target':<12} {'Error %':<8}")
            for res in results[:20]: # Show first 20
                print(f"{res['id']:<15} {res['shape']:<10} {res['mode']:<5} {res['calc']:<12.4f} {res['target']:<12.4f} {res['error_pct']:<8.2f}%")
            
            avg_err = np.mean([r['error_pct'] for r in results])
            print(f"\nAverage Relative Error: {avg_err:.4f}%")
            if avg_err < 1.0:
                print("✅ Physical accuracy is excellent (< 1%)")
            elif avg_err < 5.0:
                print("⚠️ Physical accuracy is acceptable (< 5%)")
            else:
                print("❌ Physical accuracy is low (> 5%). Check mesh density or solver settings.")

        print("\n--- Mesh Quality Analysis ---")
        avg_q_all = np.mean([s[0] for s in mesh_stats])
        max_q_all = np.max([s[1] for s in mesh_stats])
        print(f"Average Aspect Ratio: {avg_q_all:.2f}")
        print(f"Worst Aspect Ratio:   {max_q_all:.2f}")
        if avg_q_all < 2.0:
            print("✅ Mesh quality is good.")
        else:
            print("⚠️ Mesh quality might be low. Consider adjusting SizeMin/SizeMax.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_filename", type=str, default="rf_cavity_1000_dataset.h5")
    args = parser.parse_args()
    validate_dataset(args.h5_filename)
