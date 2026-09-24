import torch
import numpy as np
import pickle
import collections
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class GNOTDataset(Dataset):
    """One sample = (geometry, ALL K modes together).

    Unlike the previous per-mode formulation, every item now bundles the full
    set of K eigenmodes that belong to a single geometry.  The model predicts
    all K modes simultaneously and the loss uses a Grassmannian / set-prediction
    formulation, so there is no `mode_idx` input anymore.

    Output of __getitem__:
        X            : [N, grid_dim]
        Input_funcs  : [N, val_dim]
        Y_field      : [N, K]              (one column per mode)
        Y_freq       : [K]                 (normalised eigenfrequencies, ascending)
        geom_id      : [1]
    """
    # Feature channel reference (Input_funcs columns, val_dim=12):
    #   0: x_norm, 1: y_norm, 2: dist_to_boundary, 3: dir_bnd_x,
    #   4: dir_bnd_y, 5: node_area, 6: cos_principal, 7: sin_principal,
    #   8: dist_2nd_boundary, 9: dist_3rd_boundary, 10: curvature, 11: convexity
    FEATURE_NAMES = [
        'x_norm', 'y_norm', 'dist_boundary', 'dir_bnd_x', 'dir_bnd_y',
        'node_area', 'cos_principal', 'sin_principal',
        'dist_2nd_boundary', 'dist_3rd_boundary', 'curvature', 'convexity',
    ]

    # Gauge-dependent principal-axis channels (zeroed under augmentation).
    GAUGE_FEATURES = (6, 7)

    def __init__(self, data_path, split='train', train_ratio=0.8, val_ratio=0.1,
                 feature_indices=None, max_nodes=None, random_seed=42,
                 augment=False, zero_gauge_features=None):
        """
        zero_gauge_features: zero cos/sin_principal (cols 6-7) for every item.
            None → same as ``augment and split == 'train'``.  The rotation
            augmentation zeroes these channels on the train split, so val/test
            (and inference) of an augmented model must zero them too, otherwise
            the model sees channels at eval time that were always 0 in training.
        """
        # Allow passing a full config dict (or ConfigDict) instead of a raw path.
        # This keeps the convenience constructor `RFCavityDataset(cfg, split=...)`
        # working while the canonical signature stays path-based.
        if isinstance(data_path, dict):
            cfg = data_path
            dcfg = cfg.get('dataset', {})
            data_path = dcfg.get('data_path')
            train_ratio = dcfg.get('train_ratio', train_ratio)
            val_ratio = dcfg.get('val_ratio', val_ratio)
            feature_indices = dcfg.get('feature_indices', feature_indices)
            max_nodes = dcfg.get('max_nodes', max_nodes)
            random_seed = dcfg.get('random_seed', random_seed)
            augment = dcfg.get('augment', augment)
            zero_gauge_features = dcfg.get('zero_gauge_features', zero_gauge_features)
        print(f"Loading dataset from {data_path}...")
        self.data_path = data_path
        self.is_h5 = str(data_path).endswith('.h5')
        self.feature_indices = feature_indices  # e.g. [0,1] for xy-only, None=all
        self.max_nodes = max_nodes  # Optional: cap sequence length to prevent VRAM overflow

        # 1. Collect all per-mode samples and group them by geometry ID.
        #    Each geometry yields exactly one training item containing every mode.
        geom_to_samples = collections.defaultdict(list)

        if self.is_h5:
            import h5py, json
            with h5py.File(data_path, 'r') as f:
                metadata = json.loads(f.attrs['metadata'])
                self.stats = metadata.get('freq_stats', None)
                self.n_samples_total = metadata['n_samples']
                for i in range(self.n_samples_total):
                    g_id = f['samples'][str(i)].attrs['geom_id']
                    geom_to_samples[int(g_id)].append(i)
        else:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            self.geometry_pool = data['geometry_pool']
            all_samples = data['samples']
            self.stats = data.get('metadata', {}).get('freq_stats', None)
            self.samples_metadata = all_samples
            for i, s in enumerate(all_samples):
                geom_to_samples[int(s['geom_id'])].append(i)
            # Triangles are only kept for the full mesh (SpectralNO assembly='p1');
            # a random node subset breaks the connectivity → free them then.
            if max_nodes is not None:
                for geom in self.geometry_pool.values():
                    geom.pop('elements', None)

        # 2. Order each geometry's per-mode samples by their mode index so the
        #    K field columns are in a consistent (mode 0, 1, 2 ...) order before
        #    we sort by frequency.
        for g_id in geom_to_samples:
            geom_to_samples[g_id].sort(
                key=lambda s_idx: self._raw_mode_index(s_idx)
            )

        # Geometries missing a mode (converter skips m_idx >= len(freqs)) would
        # give a shorter Y_freq and crash torch.stack in collate → drop them.
        n_modes = max((len(v) for v in geom_to_samples.values()), default=0)
        incomplete = [g for g, v in geom_to_samples.items() if len(v) != n_modes]
        if incomplete:
            print(f"WARNING: dropping {len(incomplete)} geometries with < {n_modes} modes "
                  f"(e.g. {incomplete[:5]})")
            for g in incomplete:
                del geom_to_samples[g]

        unique_geoms = sorted(list(geom_to_samples.keys()))
        n_geoms = len(unique_geoms)

        # 3. Split by geometry (not by sample) to prevent leakage
        # Local RandomState: same permutation as the old np.random.seed() +
        # np.random.permutation() (existing splits unchanged) but without
        # resetting the global numpy RNG that augmentation relies on.
        perm_geoms = np.random.RandomState(random_seed).permutation(unique_geoms)

        n_train_geoms = int(n_geoms * train_ratio)
        n_val_geoms = int(n_geoms * val_ratio)

        if split == 'train':
            active_geoms = perm_geoms[:n_train_geoms]
        elif split == 'val':
            active_geoms = perm_geoms[n_train_geoms : n_train_geoms + n_val_geoms]
        else:
            active_geoms = perm_geoms[n_train_geoms + n_val_geoms :]

        # 4. One active "sample" == one geometry == list of per-mode sample indices.
        self.geom_to_samples = geom_to_samples
        self.active_geoms = [int(g) for g in active_geoms]

        if self.stats:
            print(f"Freq Stats: mean={self.stats['mean']:.4f}, std={self.stats['std']:.4f}")
        print(f"Split: {split}, Geometries: {len(self.active_geoms)} "
              f"(each yields all {self._infer_num_modes()} modes)")

        self.split = split
        self.augment = augment
        self.random_seed = random_seed
        if zero_gauge_features is None:
            zero_gauge_features = bool(augment and split == 'train')
        self.zero_gauge_features = bool(zero_gauge_features)
        self.h5_handle = None

    def _raw_mode_index(self, s_idx):
        """Return the integer mode index stored for a given per-mode sample."""
        if self.is_h5:
            f = self._get_h5_handle()
            sample = f['samples'][str(s_idx)]
            return int(sample.attrs.get('mode_idx', sample['Theta'][0]))
        return int(self.samples_metadata[s_idx]['Theta'][0])

    def _infer_num_modes(self):
        if not self.geom_to_samples:
            return 0
        return max(len(v) for v in self.geom_to_samples.values())

    def data_dims(self):
        """(val_dim, K) that __getitem__ yields, without reading node fields."""
        if self.feature_indices is not None:
            val_dim = len(self.feature_indices)
        elif self.is_h5:
            f = self._get_h5_handle()
            g_key = next(iter(f['geometry_pool'].keys()))
            val_dim = int(f['geometry_pool'][g_key]['Input_funcs'].shape[-1])
        else:
            geom = next(iter(self.geometry_pool.values()))
            val_dim = int(np.asarray(geom['Input_funcs']).shape[-1])
        return val_dim, self._infer_num_modes()

    def __len__(self):
        return len(self.active_geoms)

    def _get_h5_handle(self):
        """Returns a worker-local H5 file handle.

        Each DataLoader worker (or the main process) opens its own handle
        so that multi-process access is safe.  The handle is opened lazily
        on first access and kept alive for the lifetime of the worker.
        """
        import h5py
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 'main'
        attr = f'_h5_handle_{worker_id}'
        if not hasattr(self, attr) or getattr(self, attr) is None:
            setattr(self, attr, h5py.File(self.data_path, 'r', swmr=True))
        return getattr(self, attr)

    def __del__(self):
        """Close any open H5 handles when the dataset object is garbage collected."""
        for attr in list(vars(self)):
            if attr.startswith('_h5_handle_'):
                try:
                    getattr(self, attr).close()
                except Exception:
                    pass

    def __getitem__(self, idx):
        g_id = self.active_geoms[idx]
        sample_indices = self.geom_to_samples[g_id]  # ordered by mode index

        if self.is_h5:
            f = self._get_h5_handle()
            first = f['samples'][str(sample_indices[0])]
            geom = f['geometry_pool'][str(first.attrs['geom_id'])]
            x = np.asarray(geom['X'][:])
            input_features = np.asarray(geom['Input_funcs'][:])
            scale = geom.attrs.get('scale', None)
            elements = np.asarray(geom['elements'][:]) if 'elements' in geom else None
            mode_fields = []
            raw_freqs = []
            for s_idx in sample_indices:
                sample = f['samples'][str(s_idx)]
                mode_fields.append(np.asarray(sample['Y'][:]).reshape(-1))  # [N]
                raw_freqs.append(float(sample['Theta'][1]))
        else:
            first = self.samples_metadata[sample_indices[0]]
            geom = self.geometry_pool[first['geom_id']]
            x = np.asarray(geom['X'])
            input_features = np.asarray(geom['Input_funcs'])
            scale = geom.get('scale', None)
            elements = geom.get('elements', None)
            mode_fields = []
            raw_freqs = []
            for s_idx in sample_indices:
                sample = self.samples_metadata[s_idx]
                mode_fields.append(np.asarray(sample['Y']).reshape(-1))  # [N]
                raw_freqs.append(float(sample['Theta'][1]))

        # Stack the K mode fields as columns: [N, K]
        y_field = np.stack(mode_fields, axis=1).astype(np.float32)  # [N, K]
        raw_freqs = np.asarray(raw_freqs, dtype=np.float32)         # [K]

        if self.stats:
            norm_freqs = (raw_freqs - self.stats['mean']) / self.stats['std']
        else:
            norm_freqs = raw_freqs

        # Keep modes ordered by ascending (normalised) frequency so the targets
        # are in a canonical order; the model also produces sorted frequencies.
        order = np.argsort(norm_freqs, kind='stable')
        norm_freqs = norm_freqs[order]
        y_field = y_field[:, order]

        # Always work on a private copy: in-place edits below must never mutate
        # the cached geometry_pool arrays.
        x = np.array(x, dtype=np.float32, copy=True)
        input_features = np.array(input_features, dtype=np.float32, copy=True)

        # VRAM Optimization: Node Sub-sampling — SAME randperm for all modes so
        # every mode column refers to the identical node set.  val/test use a
        # fixed per-geometry generator so the monitored val metric does not
        # fluctuate from epoch to epoch just because different nodes were drawn.
        n_nodes = x.shape[0]
        if self.max_nodes is not None and n_nodes > self.max_nodes:
            gen = None
            if self.split != 'train':
                gen = torch.Generator().manual_seed(int(self.random_seed) * 100003 + int(g_id))
            rand_idx = torch.randperm(n_nodes, generator=gen)[:self.max_nodes].numpy()
            x = x[rand_idx]
            input_features = input_features[rand_idx]
            y_field = y_field[rand_idx]

        # ── Rotation / Reflection Augmentation (train split only) ────────────
        # Augmentation makes the encoder more robust to rotation by zeroing
        # the gauge-dependent cos/sin_principal features (cols 6-7) and
        # rotating the coordinate and boundary direction features.
        # NOTE: Y_field values are FEM scalars — they are scalar (rotation-
        # invariant) so they do NOT change under coordinate rotation.
        # Augmentation runs on the FULL feature layout (fixed column meaning);
        # the ablation subset (feature_indices) is selected afterwards.
        if self.augment and self.split == 'train':
            theta = np.random.uniform(0.0, 2.0 * np.pi)
            c_th, s_th = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
            R = np.array([[c_th, -s_th], [s_th, c_th]], dtype=np.float32)  # [2, 2]

            # Rotate spatial coordinates
            x = x @ R.T                                       # [N, 2]

            # Rotate boundary direction features (cols 3-4: dir_bnd_x, dir_bnd_y)
            if input_features.shape[1] > 4:
                input_features[:, 3:5] = input_features[:, 3:5] @ R.T   # [N, 2]

            # Reflection (50 % probability): flip x-axis
            if np.random.rand() < 0.5:
                x[:, 0] = -x[:, 0]
                if input_features.shape[1] > 4:
                    input_features[:, 3] = -input_features[:, 3]   # dir_bnd_x

            # Update X column 0-1 in input_features as well (they are x_norm, y_norm)
            if input_features.shape[1] >= 2:
                input_features[:, 0] = x[:, 0]
                input_features[:, 1] = x[:, 1]

        # Zero out gauge-dependent principal-axis features (cols 6-7): undefined
        # after rotation.  Applied on EVERY split of an augmented run (see
        # zero_gauge_features) so train and val/test see the same inputs.
        if self.zero_gauge_features and input_features.shape[1] > max(self.GAUGE_FEATURES):
            input_features[:, list(self.GAUGE_FEATURES)] = 0.0

        # dist_to_boundary from the FULL layout (col 2) — the boundary PINN term
        # needs it even when the ablation subset drops that column.
        if input_features.shape[1] > 2:
            dist_bnd = input_features[:, 2].copy()
        else:
            dist_bnd = np.full(input_features.shape[0], np.inf, dtype=np.float32)
        # node_area (col 5, full layout): lumped-mass quadrature weight for the
        # area-weighted field loss (training.area_weighted_field).
        if input_features.shape[1] > 5:
            area = input_features[:, 5].copy()
        else:
            area = np.ones(input_features.shape[0], dtype=np.float32)

        # Ablation: select feature subset if feature_indices is set
        if self.feature_indices is not None:
            input_features = input_features[:, self.feature_indices]

        item = {
            'X': torch.from_numpy(np.ascontiguousarray(x)).float(),
            'Input_funcs': torch.from_numpy(np.ascontiguousarray(input_features)).float(),
            'Y_field': torch.from_numpy(np.ascontiguousarray(y_field)).float(),  # [N, K]
            'Y_freq': torch.from_numpy(np.ascontiguousarray(norm_freqs)).float(),  # [K]
            'Dist_bnd': torch.from_numpy(np.ascontiguousarray(dist_bnd)).float(),  # [N]
            'Area': torch.from_numpy(np.ascontiguousarray(area)).float(),          # [N]
            'geom_id': torch.tensor([int(g_id)], dtype=torch.long),
        }
        # Physical length of one normalised unit [m] (newer converter output
        # only): SpectralNO turns its eigenvalue into GHz with it.
        if scale is not None:
            item['Scale'] = torch.tensor(float(scale), dtype=torch.float32)
        # Mesh triangles (full mesh only — sub-sampling reindexes the nodes).
        if elements is not None and self.max_nodes is None:
            item['Elements'] = torch.from_numpy(np.asarray(elements, dtype=np.int64))  # [T, 3]
        return item


# Backwards/forwards-compatible alias used by some scripts/tests.
RFCavityDataset = GNOTDataset


def gnot_collate_fn(batch):
    batch_x = [item['X'] for item in batch]
    batch_inputs = [item['Input_funcs'] for item in batch]
    batch_y_field = [item['Y_field'] for item in batch]
    batch_freq = [item['Y_freq'] for item in batch]
    batch_geom_id = [item['geom_id'] for item in batch]

    X_padded = pad_sequence(batch_x, batch_first=True, padding_value=0.0)
    Inputs_padded = pad_sequence(batch_inputs, batch_first=True, padding_value=0.0)
    Y_field_padded = pad_sequence(batch_y_field, batch_first=True, padding_value=0.0)  # [B, N, K]
    Freq_stacked = torch.stack(batch_freq)       # [B, K]
    geom_id_stacked = torch.stack(batch_geom_id) # [B, 1]

    lengths = [len(x) for x in batch_x]
    max_len = X_padded.size(1)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True

    out = {
        'X': X_padded,
        'Input_funcs': Inputs_padded,
        'Y_field': Y_field_padded,       # [B, N, K]
        'Y_freq': Freq_stacked,          # [B, K]
        'geom_id': geom_id_stacked,
        'Mask': mask,
    }
    if all('Dist_bnd' in item for item in batch):
        out['Dist_bnd'] = pad_sequence([item['Dist_bnd'] for item in batch],
                                       batch_first=True, padding_value=0.0)  # [B, N]
    if all('Area' in item for item in batch):
        out['Area'] = pad_sequence([item['Area'] for item in batch],
                                   batch_first=True, padding_value=0.0)      # [B, N]
    if all('Scale' in item for item in batch):
        out['Scale'] = torch.stack([item['Scale'] for item in batch])       # [B]
    if all('Elements' in item for item in batch):
        # Padded triangles (0, 0, 0) have zero area → contribute nothing.
        out['Elements'] = pad_sequence([item['Elements'] for item in batch],
                                       batch_first=True, padding_value=0)     # [B, T, 3]
    return out
