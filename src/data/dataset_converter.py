import h5py
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from collections import defaultdict


def _assemble_p1_K_M(nodes, elements):
    """Build P1 (linear-element) stiffness K and consistent mass M.

    Implements the analytical per-element formulas for 2D triangular linear
    elements so we do NOT need scikit-fem at conversion time.  K and M are
    returned as scipy.sparse.csr_matrix on the FULL node set (Dirichlet BC
    is enforced softly in the loss via the boundary band, not via condensation).

    Args:
        nodes:    [N, 2] node coordinates.
        elements: [E, 3] triangle vertex indices into `nodes`.
    Returns:
        K_csr, M_csr: csr_matrix of shape (N, N).
    """
    N = nodes.shape[0]
    rows, cols, K_vals, M_vals = [], [], [], []
    for tri in elements:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        x0, y0 = nodes[i0]
        x1, y1 = nodes[i1]
        x2, y2 = nodes[i2]
        # Twice the signed area (used as Jacobian)
        twoA = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        area = 0.5 * abs(twoA)
        if area <= 0.0:
            continue
        # Shape-function gradient coefficients
        b = np.array([y1 - y2, y2 - y0, y0 - y1])
        c = np.array([x2 - x1, x0 - x2, x1 - x0])
        K_el = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
        # Consistent mass: (area / 12) * [[2,1,1],[1,2,1],[1,1,2]]
        M_el = (area / 12.0) * (np.ones((3, 3)) + np.eye(3))
        idx = [i0, i1, i2]
        for a in range(3):
            for b_ in range(3):
                rows.append(idx[a])
                cols.append(idx[b_])
                K_vals.append(K_el[a, b_])
                M_vals.append(M_el[a, b_])
    K_csr = csr_matrix((K_vals, (rows, cols)), shape=(N, N))
    M_csr = csr_matrix((M_vals, (rows, cols)), shape=(N, N))
    return K_csr, M_csr

class RFCavityToGNOT:
    def __init__(self, h5_filepath: str):
        self.h5_filepath = h5_filepath
        self.geometry_pool = {}
        self.samples = []
        self.stats = {
            'n_geometries': 0,
            'n_samples': 0,
            'freq_range': [float('inf'), float('-inf')],
            'mesh_sizes': [],
            'freq_by_mode': defaultdict(list),
        }

    def _find_boundary_nodes(self, elements):
        edge_count = {}
        for tri in elements:
            edges = [
                tuple(sorted((tri[0], tri[1]))),
                tuple(sorted((tri[1], tri[2]))),
                tuple(sorted((tri[2], tri[0])))
            ]
            for e in edges:
                edge_count[e] = edge_count.get(e, 0) + 1

        boundary_nodes = {node for edge, count in edge_count.items()
                         if count == 1 for node in edge}
        return np.array(list(boundary_nodes))

    def _compute_node_areas(self, nodes, elements):
        """Her noktanın Voronoi benzeri alanı: komşu üçgen alanlarının 1/3'ü toplamı.
        Mesh yoğunluğu bilgisi verir — fizik çözücü sınıra yakın daha sık mesh kullanır."""
        n_nodes = len(nodes)
        areas = np.zeros(n_nodes)
        for tri in elements:
            v0, v1, v2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
            # 2D cross product: |a_x * b_y - a_y * b_x| (NumPy 2.0 np.cross için 3D zorunlu)
            a, b = v1 - v0, v2 - v0
            tri_area = 0.5 * abs(a[0] * b[1] - a[1] * b[0])
            for k in range(3):
                areas[tri[k]] += tri_area / 3.0
        # Normalize to [0, 1]
        if areas.max() > 0:
            areas = areas / (areas.max() + 1e-10)
        return areas.reshape(-1, 1).astype(np.float32)

    def extract_geometry_features(self, nodes, elements, compute_fem_matrices=True):
        # Isotropic Normalization: En-boy oranını (aspect ratio) koruyarak merkezi 0'a çek.
        center_raw = nodes.mean(axis=0)
        nodes_centered = nodes - center_raw
        global_scale = np.max(np.abs(nodes_centered)) + 1e-12
        nodes_norm = nodes_centered / global_scale

        # Boundary detection
        boundary_indices = self._find_boundary_nodes(elements)
        boundary_nodes = nodes_norm[boundary_indices]
        tree = cKDTree(boundary_nodes)
        dist_to_boundary, nearest_idx = tree.query(nodes_norm)

        # Direction to nearest boundary (normalized) — "duvar hangi yönde?"
        nearest_bnd_points = boundary_nodes[nearest_idx]  # [N, 2]
        dir_vec = nearest_bnd_points - nodes_norm          # [N, 2]
        dir_norms = np.linalg.norm(dir_vec, axis=1, keepdims=True) + 1e-10
        dir_to_boundary = (dir_vec / dir_norms).astype(np.float32)  # [N, 2] unit vectors

        # Node area (local mesh density)
        node_area = self._compute_node_areas(nodes_norm, elements)

        # Principal axis angle — kavite ana eksenine göre açı
        # PCA on boundary nodes → major axis direction
        bnd_centered = boundary_nodes - boundary_nodes.mean(axis=0)
        cov = np.cov(bnd_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_axis = eigenvectors[:, -1]  # en büyük eigenvalue'nun eigenvector'ü
        
        # PCA ekseni için işaret sabitleme (Sign Disambiguation)
        # Vektörün x bileşeni negatifse (veya x sıfırken y negatifse) yönünü ters çevir.
        # Bu, rastgele 180 derece dönüşleri (takla atmayı) engeller.
        if principal_axis[0] < 0 or (principal_axis[0] == 0 and principal_axis[1] < 0):
            principal_axis = -principal_axis

        # Her nokta için: (nokta - merkez) vektörünün ana eksene göre açısı
        node_vecs = nodes_norm - nodes_norm.mean(axis=0)
        # cos(angle) = dot(node_vec, principal) / |node_vec|
        dots = node_vecs @ principal_axis
        node_mags = np.linalg.norm(node_vecs, axis=1) + 1e-10
        cos_angle = (dots / node_mags).reshape(-1, 1).astype(np.float32)
        # sin bileşeni de ekle — tam yön bilgisi için
        cross = (node_vecs[:, 0] * principal_axis[1] - node_vecs[:, 1] * principal_axis[0])
        sin_angle = (cross / node_mags).reshape(-1, 1).astype(np.float32)

        # Combine: [x, y, dist_bnd, dir_bnd_x, dir_bnd_y, node_area, cos_principal, sin_principal]
        # val_dim = 8
        geom_features = np.concatenate([
            nodes_norm,                         # [0,1] x, y
            dist_to_boundary.reshape(-1, 1),    # [2]   dist to boundary
            dir_to_boundary,                    # [3,4] direction to nearest boundary (x, y)
            node_area,                          # [5]   local mesh density
            cos_angle,                          # [6]   cos(angle to principal axis)
            sin_angle,                          # [7]   sin(angle to principal axis)
        ], axis=1).astype(np.float32)

        result = {
            'X': nodes_norm.astype(np.float32),
            'Input_funcs': geom_features,
            'elements': elements,
        }

        # P1 stiffness K and consistent mass M, in normalised coordinates.
        # These power the FEM-Rayleigh path in training without needing
        # autograd through the network.  See physics_losses.rayleigh_quotient
        # mode="fem" and Rowan et al. Section 4.2.
        if compute_fem_matrices:
            try:
                K_csr, M_csr = _assemble_p1_K_M(nodes_norm, elements)
                result['K_data']    = K_csr.data.astype(np.float32)
                result['K_indices'] = K_csr.indices.astype(np.int32)
                result['K_indptr']  = K_csr.indptr.astype(np.int32)
                result['M_data']    = M_csr.data.astype(np.float32)
                result['M_indices'] = M_csr.indices.astype(np.int32)
                result['M_indptr']  = M_csr.indptr.astype(np.int32)
                result['K_shape']   = np.array(K_csr.shape, dtype=np.int32)
            except Exception as e:
                # FEM matrix assembly is optional — fall back to autograd path.
                print(f"  [warn] P1 K/M assembly failed: {e}")
        return result

    def convert_dataset(self, output_filepath, mode_indices=[0, 1, 2], max_samples=None, format='pkl',
                        freq_mean=None, freq_std=None):
        print(f"Converting: {self.h5_filepath} to {format.upper()}")

        with h5py.File(self.h5_filepath, 'r') as f:
            sample_keys = sorted(f.keys())
            if max_samples:
                sample_keys = sample_keys[:max_samples]

            for key in tqdm(sample_keys, desc="Converting"):
                grp = f[key]
                sample_id = int(key.split('_')[-1])

                nodes = grp['nodes'][:]
                elements = grp['elements'][:]
                freqs = grp['freqs'][:]
                vecs = grp['vecs'][:len(nodes), :]

                if sample_id not in self.geometry_pool:
                    self.geometry_pool[sample_id] = self.extract_geometry_features(nodes, elements)
                    self.stats['n_geometries'] += 1
                    self.stats['mesh_sizes'].append(len(nodes))

                for i, m_idx in enumerate(mode_indices):
                    if m_idx >= len(freqs):
                        continue

                    Y = vecs[:, m_idx].reshape(-1, 1).astype(np.float32)

                    # Normalize magnitude; sign is handled by the gauge-invariant loss
                    Y_max = np.abs(Y).max()
                    if Y_max > 1e-10:
                        Y = Y / Y_max

                    freq = float(freqs[m_idx])
                    theta = np.array([float(i), freq, float(sample_id)], dtype=np.float32)

                    self.samples.append({
                        'geom_id': sample_id,
                        'Y': Y,
                        'Theta': theta
                    })

                    self.stats['n_samples'] += 1
                    self.stats['freq_range'][0] = min(self.stats['freq_range'][0], freq)
                    self.stats['freq_range'][1] = max(self.stats['freq_range'][1], freq)
                    self.stats['freq_by_mode'][m_idx].append(freq)

        # Use manual stats if provided, otherwise compute from dataset
        final_mean = float(freq_mean) if freq_mean is not None else float(np.mean([f for l in self.stats['freq_by_mode'].values() for f in l]))
        final_std = float(freq_std) if freq_std is not None else float(np.std([f for l in self.stats['freq_by_mode'].values() for f in l]) + 1e-10)

        # Dataset Metadata
        metadata = {
            'mode_indices': mode_indices,
            'n_geometries': self.stats['n_geometries'],
            'n_samples': self.stats['n_samples'],
            'freq_stats': {
                'mean': final_mean,
                'std': final_std,
                'mode_stats': {
                    str(m): {'mean': float(np.mean(l)), 'std': float(np.std(l) + 1e-10)}
                    for m, l in self.stats['freq_by_mode'].items()
                }
            }
        }

        if format == 'pkl':
            dataset = {
                'geometry_pool': self.geometry_pool,
                'samples': self.samples,
                'metadata': metadata
            }
            with open(output_filepath, 'wb') as f:
                pickle.dump(dataset, f)
        else:
            # Save to H5 (Memory Efficient)
            with h5py.File(output_filepath, 'w') as f_out:
                # Store metadata as attributes or JSON
                import json
                f_out.attrs['metadata'] = json.dumps(metadata)
                
                # Geometries
                geom_grp = f_out.create_group('geometry_pool')
                for g_id, g_data in self.geometry_pool.items():
                    g_sub = geom_grp.create_group(str(g_id))
                    g_sub.create_dataset('X', data=g_data['X'], compression="gzip")
                    # Input_funcs is now a plain numpy array
                    g_sub.create_dataset('Input_funcs', data=g_data['Input_funcs'], compression="gzip")
                    # P1 FEM K, M caches (optional — present only if computed)
                    if 'K_data' in g_data:
                        for key in ('K_data', 'K_indices', 'K_indptr',
                                    'M_data', 'M_indices', 'M_indptr', 'K_shape'):
                            g_sub.create_dataset(key, data=g_data[key], compression="gzip")
                
                # Samples
                samp_grp = f_out.create_group('samples')
                for i, s in enumerate(self.samples):
                    s_sub = samp_grp.create_group(str(i))
                    s_sub.attrs['geom_id'] = s['geom_id']
                    s_sub.create_dataset('Y', data=s['Y'], compression="gzip")
                    s_sub.create_dataset('Theta', data=s['Theta'])

        self._print_stats()
        print(f"\n✅ Saved ({format.upper()}): {output_filepath}")
        return output_filepath

    def _print_stats(self):
        print(f"\n{'='*50}")
        print("DATASET STATISTICS")
        print(f"{'='*50}")
        print(f"Geometries: {self.stats['n_geometries']}")
        print(f"Samples:    {self.stats['n_samples']}")
        print(f"Mesh sizes: {min(self.stats['mesh_sizes'])} - {max(self.stats['mesh_sizes'])}")
        print(f"Freq range: {self.stats['freq_range'][0]:.2f} - {self.stats['freq_range'][1]:.2f} GHz")
        print(f"{'='*50}")
