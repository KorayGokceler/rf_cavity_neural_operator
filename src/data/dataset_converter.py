import h5py
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import cKDTree
from collections import defaultdict

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
            tri_area = 0.5 * abs(np.cross(v1 - v0, v2 - v0))
            for k in range(3):
                areas[tri[k]] += tri_area / 3.0
        # Normalize to [0, 1]
        if areas.max() > 0:
            areas = areas / (areas.max() + 1e-10)
        return areas.reshape(-1, 1).astype(np.float32)

    def extract_geometry_features(self, nodes, elements):
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

        return {
            'X': nodes_norm.astype(np.float32),
            'Input_funcs': geom_features,
            'elements': elements,
        }

    def convert_dataset(self, output_filepath, mode_indices=[0, 1, 2], max_samples=None, format='pkl'):
        print(f"Converting: {self.h5_filepath} to {format.upper()}")

        with h5py.File(self.h5_filepath, 'r') as f:
            if 'samples' not in f:
                print("Error: H5 file does not contain 'samples' group.")
                return
            
            samples_grp = f['samples']
            sample_keys = sorted(samples_grp.keys(), key=lambda x: int(x))
            
            if max_samples:
                sample_keys = sample_keys[:max_samples]

            for key in tqdm(sample_keys, desc="Converting"):
                grp = samples_grp[key]
                sample_id = int(key)

                nodes = grp['nodes'][:]
                elements = grp['elements'][:]
                freqs = grp['freqs'][:]
                vecs = grp['vecs'][:]

                if sample_id not in self.geometry_pool:
                    self.geometry_pool[sample_id] = self.extract_geometry_features(nodes, elements)
                    self.stats['n_geometries'] += 1
                    self.stats['mesh_sizes'].append(len(nodes))

                for i, m_idx in enumerate(mode_indices):
                    if m_idx >= len(freqs):
                        continue

                    Y = vecs[:, m_idx].reshape(-1, 1).astype(np.float32)



                    # Peak-Sign Normalization: Alanın faz (sign) keyfiliğini yenmek için en yüksek mutlak değerli noktanın işareti baz alınıyor.
                    # Bu, her örnekteki en belirgin "dağın" her zaman yukarı bakmasını sağlar.
                    max_idx = np.argmax(np.abs(Y))
                    peak_val = Y.flatten()[max_idx]
                    if peak_val < 0:
                        Y = Y * -1.0

                    # Normalize
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

        # Dataset Metadata
        metadata = {
            'mode_indices': mode_indices,
            'n_geometries': self.stats['n_geometries'],
            'n_samples': self.stats['n_samples'],
            'freq_stats': {
                'mean': float(np.mean([f for l in self.stats['freq_by_mode'].values() for f in l])),
                'std': float(np.std([f for l in self.stats['freq_by_mode'].values() for f in l]) + 1e-10),
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
