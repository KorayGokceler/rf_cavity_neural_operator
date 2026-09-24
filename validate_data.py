import h5py
import numpy as np
import argparse
import os

C0 = 299792458.0

# Defaults = dataset_generator.py calibration geometry (used if the H5 group has no attrs)
DEFAULT_CALIB_PARAMS = {
    'square': {'side': 0.08},
    'circle': {'radius': 0.04},
    'annulus': {'r_outer': 0.045, 'r_inner': 0.02},
}


def _annulus_k_roots(a, b, n, n_roots):
    """Roots k of the TM (Dirichlet-Dirichlet) annulus condition
    J_n(k a) Y_n(k b) - J_n(k b) Y_n(k a) = 0, inner radius a, outer radius b."""
    from scipy.special import jv, yv
    from scipy.optimize import brentq
    g = lambda k: jv(n, k * a) * yv(n, k * b) - jv(n, k * b) * yv(n, k * a)
    # Roots are spaced ~pi/(b-a); a fine grid brackets every sign change.
    k_grid = np.linspace(1e-6, (n_roots + n + 2) * np.pi / (b - a), 4000)
    vals = g(k_grid)
    roots = []
    for i in np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]:
        roots.append(brentq(g, k_grid[i], k_grid[i + 1]))
        if len(roots) == n_roots:
            break
    return roots


def analytical_spectrum(shape_type, n_modes, params=None):
    """Lowest n_modes TM (Dirichlet Laplacian) resonant frequencies [GHz], ascending,
    with degenerate modes repeated (e.g. circle TM11 appears twice)."""
    params = {**DEFAULT_CALIB_PARAMS.get(shape_type, {}), **(params or {})}
    n_extra = n_modes + 4
    if shape_type == 'square':
        a = params['side']
        freqs = [(C0 / 2) * np.sqrt((m / a) ** 2 + (n / a) ** 2)
                 for m in range(1, n_extra + 1) for n in range(1, n_extra + 1)]
    elif shape_type == 'circle':
        from scipy.special import jn_zeros
        R = params['radius']
        freqs = []
        for n in range(n_extra):
            for j in jn_zeros(n, n_extra):
                f = C0 * j / (2 * np.pi * R)
                freqs += [f] if n == 0 else [f, f]  # cos/sin polarisations
    elif shape_type == 'annulus':
        a, b = params['r_inner'], params['r_outer']
        freqs = []
        for n in range(n_extra):
            for k in _annulus_k_roots(a, b, n, n_extra):
                f = C0 * k / (2 * np.pi)
                freqs += [f] if n == 0 else [f, f]
    else:
        return None
    return np.sort(np.asarray(freqs))[:n_modes] / 1e9


def calculate_analytical_freq(shape_type, mode_idx, params=None):
    spec = analytical_spectrum(shape_type, mode_idx + 1, params)
    return None if spec is None else float(spec[mode_idx])

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
        return None

    with h5py.File(h5_path, 'r') as f:
        keys = sorted(f.keys())
        print(f"Total Samples: {len(keys)}")
        
        results = []
        mesh_stats = []
        
        for k in keys:
            grp = f[k]
            if not isinstance(grp, h5py.Group) or 'nodes' not in grp:
                continue
            nodes = grp['nodes'][:]
            elements = grp['elements'][:]
            freqs = grp['freqs'][:]
            shape_type_attr = grp.attrs.get('shape_type', 'unknown')
            if isinstance(shape_type_attr, bytes):
                shape_type_attr = shape_type_attr.decode()

            # Generator writes the concrete calibration shape ('square'/'circle'/'annulus');
            # very old files used a generic 'calibration' tag with square/circle alternating.
            s_id = int(k.split('_')[-1])
            if shape_type_attr in DEFAULT_CALIB_PARAMS:
                actual_shape = shape_type_attr
            elif shape_type_attr == 'calibration':
                actual_shape = 'square' if s_id % 2 == 0 else 'circle'
            else:
                actual_shape = 'random'
            
            # Mesh quality
            avg_qr, max_qr = check_mesh_quality(nodes, elements)
            mesh_stats.append((avg_qr, max_qr))
            
            if actual_shape != 'random':
                params = {p: float(grp.attrs[p]) for p in DEFAULT_CALIB_PARAMS[actual_shape] if p in grp.attrs}
                spectrum = analytical_spectrum(actual_shape, len(freqs), params)
                for i in range(len(freqs)):
                    analytical = spectrum[i]
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
            max_err = np.max([r['error_pct'] for r in results])
            print(f"\nAverage Relative Error: {avg_err:.4f}%  (max {max_err:.4f}%)")
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
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_filename", type=str, default="rf_cavity_1000_dataset.h5")
    args = parser.parse_args()
    validate_dataset(args.h5_filename)
