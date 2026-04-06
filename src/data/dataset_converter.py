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

    def _estimate_local_curvature(self, nodes, elements):
        n_nodes = len(nodes)
        curvature = np.zeros((n_nodes, 1))

        neighbors = [set() for _ in range(n_nodes)]
        for tri in elements:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        neighbors[tri[i]].add(tri[j])

        for i in range(n_nodes):
            if len(neighbors[i]) > 1:
                neighbor_list = list(neighbors[i])
                vectors = nodes[neighbor_list] - nodes[i]
                angles = np.arctan2(vectors[:, 1], vectors[:, 0])
                curvature[i] = np.std(angles)

        if curvature.max() > 0:
            curvature = curvature / (curvature.max() + 1e-10)

        return curvature.astype(np.float32)

    def extract_geometry_features(self, nodes, elements):
        # Per-axis normalization: map each axis independently to [0, 1]
        min_val = nodes.min(axis=0)
        max_val = nodes.max(axis=0)
        nodes_norm = (nodes - min_val) / (max_val - min_val + 1e-10)

        # Boundary
        boundary_indices = self._find_boundary_nodes(elements)
        tree = cKDTree(nodes_norm[boundary_indices])
        dist_to_boundary, _ = tree.query(nodes_norm)

        boundary_mask = np.zeros((len(nodes), 1))
        boundary_mask[boundary_indices] = 1.0

        # Center
        center = nodes_norm.mean(axis=0)
        dist_to_center = np.linalg.norm(nodes_norm - center, axis=1, keepdims=True)

        # Curvature
        curvature = self._estimate_local_curvature(nodes_norm, elements)

        # Combine: [x_norm, y_norm, dist_to_boundary, boundary_mask, dist_to_center, curvature]
        geom_features = np.concatenate([
            nodes_norm,
            dist_to_boundary.reshape(-1, 1),
            boundary_mask,
            dist_to_center,
            curvature,
        ], axis=1).astype(np.float32)

        return {
            'X': nodes_norm.astype(np.float32),
            'Input_funcs': geom_features,  # numpy array directly, no tuple wrapping
            'elements': elements,
        }

    def convert_dataset(self, output_filepath, mode_indices=[0, 1, 2], max_samples=None, format='pkl'):
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

                for m_idx in mode_indices:
                    if m_idx >= len(freqs):
                        continue

                    Y = vecs[:, m_idx].reshape(-1, 1).astype(np.float32)

                    # Spatial Sign Alignment: İşareti en güçlü aksis ile mekansal olarak sabitle.
                    # Bu, dipol/kuadrupol gibi simetrik alanlardaki rastgele işaret taklalarını engeller.
                    nodes_y = nodes[:, 1].reshape(-1, 1)
                    nodes_x = nodes[:, 0].reshape(-1, 1)
                    sum_y = (Y * nodes_y).sum()
                    sum_x = (Y * nodes_x).sum()

                    # En güçlü korelasyon olan ekseni seç ve pozitif yöne zorla
                    if abs(sum_y) > abs(sum_x):
                        sign = np.sign(sum_y) if abs(sum_y) > 1e-12 else 1
                    else:
                        sign = np.sign(sum_x) if abs(sum_x) > 1e-12 else 1
                    
                    Y = Y * sign
                    if sign == 0: Y = Y * 1.0 # fallback for zero case

                    # Normalize
                    Y_max = np.abs(Y).max()
                    if Y_max > 1e-10:
                        Y = Y / Y_max

                    freq = float(freqs[m_idx])
                    theta = np.array([float(m_idx), freq, float(sample_id)], dtype=np.float32)

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
