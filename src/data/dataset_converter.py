import h5py
import numpy as np
import pickle
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

    @staticmethod
    def _boundary_edges(elements):
        """Undirected edges used by exactly one triangle, [E_bnd, 2] (sorted pairs)."""
        el = np.asarray(elements)
        edges = np.sort(np.concatenate([el[:, [0, 1]], el[:, [1, 2]], el[:, [2, 0]]]), axis=1)
        uniq, counts = np.unique(edges, axis=0, return_counts=True)
        return uniq[counts == 1]

    def _find_boundary_nodes(self, elements):
        # Bir kenar sadece tek bir üçgene aitse sınır kenarıdır. Sıralı (deterministik) çıktı.
        return np.unique(self._boundary_edges(elements))

    @staticmethod
    def _torsion_function(nodes, elements, boundary_indices):
        """P1 solution of −Δw = 1 in Ω, w = 0 on ∂Ω at the mesh nodes."""
        import skfem
        from skfem.models.poisson import laplace, unit_load
        mesh = skfem.MeshTri(np.ascontiguousarray(nodes.T, dtype=np.float64),
                             np.ascontiguousarray(np.asarray(elements).T, dtype=np.int64))
        basis = skfem.Basis(mesh, skfem.ElementTriP1())
        w = skfem.solve(*skfem.condense(skfem.asm(laplace, basis), skfem.asm(unit_load, basis),
                                        D=np.asarray(boundary_indices)))
        return np.clip(w, 0.0, None)

    def _compute_node_areas(self, nodes, elements):
        """Her noktanın Voronoi benzeri alanı: komşu üçgen alanlarının 1/3'ü toplamı.
        Mesh yoğunluğu bilgisi verir — fizik çözücü sınıra yakın daha sık mesh kullanır."""
        el = np.asarray(elements)
        v0, v1, v2 = nodes[el[:, 0]], nodes[el[:, 1]], nodes[el[:, 2]]
        # 2D cross product: |a_x * b_y - a_y * b_x| (NumPy 2.0 np.cross için 3D zorunlu)
        a, b = v1 - v0, v2 - v0
        tri_area = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        areas = np.zeros(len(nodes))
        for k in range(3):
            np.add.at(areas, el[:, k], tri_area / 3.0)
        # Normalize to [0, 1]
        if areas.max() > 0:
            areas = areas / (areas.max() + 1e-10)
        return areas.reshape(-1, 1).astype(np.float32)

    def _boundary_loops(self, nodes, elements):
        """Boundary as closed node loops, each oriented with the domain on its LEFT
        (outer wall counter-clockwise, holes clockwise).

        Uses mesh topology (directed boundary edges of CCW-oriented triangles), so the
        orientation is canonical: it does not depend on node numbering or on the
        triangle winding gmsh happened to produce, and multiply-connected domains
        (e.g. annulus) give one loop per wall.
        """
        el = np.asarray(elements).copy()
        p0, p1, p2 = nodes[el[:, 0]], nodes[el[:, 1]], nodes[el[:, 2]]
        signed = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
        cw = signed < 0
        el[cw] = el[cw][:, [0, 2, 1]]  # make every triangle CCW
        directed = np.concatenate([el[:, [0, 1]], el[:, [1, 2]], el[:, [2, 0]]])
        und = np.sort(directed, axis=1)
        _, inv, counts = np.unique(und, axis=0, return_inverse=True, return_counts=True)
        bnd_directed = directed[counts[inv.reshape(-1)] == 1]  # interior of a CCW tri is left of a->b

        succ = defaultdict(list)
        for a, b in bnd_directed:
            succ[int(a)].append(int(b))
        loops = []
        for start in sorted(succ):
            while succ[start]:
                loop, cur = [start], succ[start].pop()
                while cur != start and succ.get(cur):
                    loop.append(cur)
                    cur = succ[cur].pop()
                loops.append(np.asarray(loop, dtype=np.int64))
        return loops

    def _compute_boundary_curvature(self, nodes, elements, boundary_indices):
        """Signed discrete curvature at each boundary node, aligned with boundary_indices.

        Sign convention (canonical, domain on the left of the traversal):
            > 0 : convex wall (e.g. every point of a circle),
            < 0 : concave / re-entrant wall (inner wall of an annulus, re-entrant corner).
        kappa_i = cross(t_in, t_out) / (0.5 * (|e_in| + |e_out|)) with unit tangents t.

        Returns:
            curvature: [N_bnd] float32
        """
        curv_by_node = {}
        for loop in self._boundary_loops(nodes, elements):
            N = len(loop)
            if N < 3:
                continue
            pts = nodes[loop]
            t1 = pts - np.roll(pts, 1, axis=0)    # p_i - p_{i-1}
            t2 = np.roll(pts, -1, axis=0) - pts   # p_{i+1} - p_i
            l1 = np.linalg.norm(t1, axis=1) + 1e-10
            l2 = np.linalg.norm(t2, axis=1) + 1e-10
            t1_u, t2_u = t1 / l1[:, None], t2 / l2[:, None]
            # Signed curvature: cross product of unit tangents / avg arc length
            cross = t1_u[:, 0] * t2_u[:, 1] - t1_u[:, 1] * t2_u[:, 0]
            kappa = cross / (0.5 * (l1 + l2) + 1e-10)
            curv_by_node.update(zip(loop.tolist(), kappa.tolist()))
        return np.array([curv_by_node.get(int(i), 0.0) for i in boundary_indices], dtype=np.float32)

    def extract_geometry_features(self, nodes, elements):
        # Isotropic Normalization: En-boy oranını (aspect ratio) koruyarak merkezi 0'a çek.
        # Merkez = alan-ağırlıklı ağırlık merkezi (düğüm ortalaması sınıra sık mesh
        # yüzünden kayıyordu: diskte ~0.005 kayma).
        el = np.asarray(elements)
        v0, v1, v2 = nodes[el[:, 0]], nodes[el[:, 1]], nodes[el[:, 2]]
        a, b = v1 - v0, v2 - v0
        tri_area = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        center_raw = (tri_area[:, None] * (v0 + v1 + v2) / 3.0).sum(axis=0) / tri_area.sum()
        nodes_centered = nodes - center_raw
        global_scale = np.max(np.abs(nodes_centered)) + 1e-12
        nodes_norm = nodes_centered / global_scale

        # Boundary detection
        boundary_indices = self._find_boundary_nodes(elements)
        boundary_nodes = nodes_norm[boundary_indices]
        tree = cKDTree(boundary_nodes)

        # Query k=3 nearest boundary nodes at once (for dist_1st, 2nd, 3rd)
        k_query = min(3, len(boundary_nodes))
        dists_k, nearest_idxs_k = tree.query(nodes_norm, k=k_query)
        if k_query == 1:
            dists_k = dists_k.reshape(-1, 1)
            nearest_idxs_k = nearest_idxs_k.reshape(-1, 1)

        dist_to_boundary = dists_k[:, 0]
        nearest_idx = nearest_idxs_k[:, 0]
        dist_2nd = dists_k[:, 1] if k_query >= 2 else dist_to_boundary
        dist_3rd = dists_k[:, 2] if k_query >= 3 else dist_2nd

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

        # ── NEW: extra features (val_dim 8 → 12) ────────────────────────────
        # [8] dist_to_2nd_boundary — distance to 2nd nearest boundary node
        dist_2nd_feat = dist_2nd.reshape(-1, 1).astype(np.float32)
        # [9] dist_to_3rd_boundary — distance to 3rd nearest boundary node
        dist_3rd_feat = dist_3rd.reshape(-1, 1).astype(np.float32)

        # [10] local boundary curvature at nearest boundary node, assigned to
        #      every interior node as the curvature of its closest boundary pt.
        #      Sign is canonical (topological loop orientation): + convex wall,
        #      − concave/re-entrant wall. (Eski greedy-NN sıralaması işareti
        #      geometriden geometriye rastgele çeviriyordu.)
        bnd_curvature = self._compute_boundary_curvature(nodes_norm, elements, boundary_indices)
        # Clamp to avoid outliers; normalize by max abs curvature
        bnd_curvature = bnd_curvature / (np.abs(bnd_curvature).max() + 1e-8)
        node_curvature = bnd_curvature[nearest_idx].reshape(-1, 1).astype(np.float32)

        # [11] convexity: curvature of the nearest wall weighted by dist to boundary.
        #      < 0 → node faces a concave/re-entrant wall ("pocket"), > 0 → convex wall.
        convexity = (dist_to_boundary * bnd_curvature[nearest_idx]).reshape(-1, 1).astype(np.float32)
        convexity = np.clip(convexity, -1.0, 1.0)

        # [12] torsion function w/max(w): −Δw = 1 in Ω, w = 0 on ∂Ω (one P1
        #      solve).  Smooth "landscape" of the domain; λ₁ ≈ j₀₁²/(4·max w)
        #      to ~1% (docs/16) → max w is stored as 'torsion_max'.
        torsion = self._torsion_function(nodes_norm, elements, boundary_indices)
        torsion_max = float(torsion.max())

        # Combine: [x, y, dist_bnd, dir_bnd_x, dir_bnd_y, node_area,
        #           cos_principal, sin_principal,
        #           dist_2nd, dist_3rd, curvature, convexity, torsion]
        # val_dim = 13
        geom_features = np.concatenate([
            nodes_norm,                         # [0,1] x, y
            dist_to_boundary.reshape(-1, 1),    # [2]   dist to boundary
            dir_to_boundary,                    # [3,4] direction to nearest boundary (x, y)
            node_area,                          # [5]   local mesh density
            cos_angle,                          # [6]   cos(angle to principal axis)
            sin_angle,                          # [7]   sin(angle to principal axis)
            dist_2nd_feat,                      # [8]   dist to 2nd nearest boundary
            dist_3rd_feat,                      # [9]   dist to 3rd nearest boundary
            node_curvature,                     # [10]  local boundary curvature
            convexity,                          # [11]  convexity sign
            (torsion / (torsion_max + 1e-12)).reshape(-1, 1),  # [12] torsion w / max w
        ], axis=1).astype(np.float32)

        return {
            'X': nodes_norm.astype(np.float32),
            'Input_funcs': geom_features,
            'elements': elements,
            # Physical length of one normalized unit [m] and the removed center [m].
            # Features are scale-invariant, but eigenfrequencies scale as 1/size:
            #   x_phys = X * scale + center,  k^2_phys = k^2_norm / scale^2,
            #   f_phys = c * sqrt(k^2_norm) / (2*pi*scale).
            'scale': float(global_scale),
            'torsion_max': torsion_max,       # max w in normalised coords (λ₁ prior)
            'center': center_raw.astype(np.float64),
        }

    def convert_dataset(self, output_filepath, mode_indices=[0, 1, 2], max_samples=None, format='pkl',
                        freq_mean=None, freq_std=None):
        print(f"Converting: {self.h5_filepath} to {format.upper()}")

        with h5py.File(self.h5_filepath, 'r') as f:
            # Only raw-FEM sample groups (sample_XXXX with a 'nodes' dataset)
            sample_keys = sorted(k for k in f.keys()
                                 if isinstance(f[k], h5py.Group) and 'nodes' in f[k])
            if not sample_keys:
                raise ValueError(f"No raw FEM sample groups (sample_XXXX/nodes) found in {self.h5_filepath}; "
                                 f"top-level keys: {list(f.keys())[:10]}")
            if max_samples:
                sample_keys = sample_keys[:max_samples]

            for key in tqdm(sample_keys, desc="Converting"):
                grp = f[key]
                sample_id = int(key.split('_')[-1])

                nodes = grp['nodes'][:]
                elements = grp['elements'][:]
                freqs = np.asarray(grp['freqs'][:], dtype=np.float64)
                # vecs is the full P2 DOF vector; rows 0..n_nodes-1 are the mesh vertices
                # (skfem ElementTriP2 numbers vertex DOFs first) → P1 field on X.
                if grp['vecs'].shape[0] < len(nodes) or grp['vecs'].shape[1] != len(freqs):
                    raise ValueError(f"{key}: vecs shape {grp['vecs'].shape} inconsistent with "
                                     f"{len(nodes)} nodes / {len(freqs)} freqs")
                vecs = grp['vecs'][:len(nodes), :]
                # Mode index = rank by frequency. Older H5 files (skfem non-symmetric
                # `eigs`) are not guaranteed ascending, e.g. inside degenerate pairs.
                order = np.argsort(freqs, kind='stable')
                freqs, vecs = freqs[order], vecs[:, order]

                if sample_id not in self.geometry_pool:
                    geom = self.extract_geometry_features(nodes, elements)
                    geom['shape_type'] = grp.attrs.get('shape_type', 'unknown')
                    self.geometry_pool[sample_id] = geom
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
                    # Theta[0] = slot position i within mode_indices (0..len-1), NOT the
                    # raw FEM mode index; the raw index is kept separately in 'mode_idx'.
                    theta = np.array([float(i), freq, float(sample_id)], dtype=np.float32)

                    self.samples.append({
                        'geom_id': sample_id,
                        'Y': Y,
                        'Theta': theta,
                        'mode_idx': int(m_idx),
                    })

                    self.stats['n_samples'] += 1
                    self.stats['freq_range'][0] = min(self.stats['freq_range'][0], freq)
                    self.stats['freq_range'][1] = max(self.stats['freq_range'][1], freq)
                    self.stats['freq_by_mode'][m_idx].append(freq)

        if self.stats['n_samples'] == 0:
            raise ValueError(f"No samples produced: requested modes {mode_indices} not present "
                             f"in {self.h5_filepath} (it stores {len(freqs)} modes per geometry).")

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
                    g_sub.create_dataset('Input_funcs', data=g_data['Input_funcs'], compression="gzip")
                    g_sub.create_dataset('elements', data=g_data['elements'], compression="gzip")
                    if 'shape_type' in g_data:
                        g_sub.attrs['shape_type'] = g_data['shape_type']
                    g_sub.attrs['scale'] = g_data['scale']
                    g_sub.attrs['torsion_max'] = g_data['torsion_max']
                    g_sub.attrs['center'] = g_data['center']

                # Samples
                samp_grp = f_out.create_group('samples')
                for i, s in enumerate(self.samples):
                    s_sub = samp_grp.create_group(str(i))
                    s_sub.attrs['geom_id'] = s['geom_id']
                    s_sub.attrs['mode_idx'] = s['mode_idx']
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
