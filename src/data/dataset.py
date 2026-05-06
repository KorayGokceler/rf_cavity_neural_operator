import torch
import numpy as np
import pickle
import collections
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class GNOTDataset(Dataset):
    # Feature channel reference (Input_funcs columns):
    #   0: x_norm, 1: y_norm, 2: dist_to_boundary, 3: dir_bnd_x,
    #   4: dir_bnd_y, 5: node_area, 6: cos_principal, 7: sin_principal
    FEATURE_NAMES = ['x_norm', 'y_norm', 'dist_boundary', 'dir_bnd_x', 'dir_bnd_y', 'node_area', 'cos_principal', 'sin_principal']

    def __init__(self, data_path, split='train', train_ratio=0.8, val_ratio=0.1,
                 feature_indices=None, max_nodes=None, active_mode_index=None, random_seed=42):
        print(f"Loading dataset from {data_path}...")
        self.data_path = data_path
        self.is_h5 = str(data_path).endswith('.h5')
        self.feature_indices = feature_indices  # e.g. [0,1] for xy-only, None=all
        self.max_nodes = max_nodes  # Optional: cap sequence length to prevent VRAM overflow
        self.active_mode_index = active_mode_index # Filter for single-mode training

        # 1. Collect all samples and their geometry IDs
        sample_to_geom = []
        
        if self.is_h5:
            import h5py, json
            with h5py.File(data_path, 'r') as f:
                metadata = json.loads(f.attrs['metadata'])
                self.stats = metadata.get('freq_stats', None)
                self.n_samples_total = metadata['n_samples']
                for i in range(self.n_samples_total):
                    g_id = f['samples'][str(i)].attrs['geom_id']
                    sample_to_geom.append((i, g_id))
        else:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            self.geometry_pool = data['geometry_pool']
            all_samples = data['samples']
            self.stats = data.get('metadata', {}).get('freq_stats', None)
            self.samples_metadata = all_samples
            for i, s in enumerate(all_samples):
                sample_to_geom.append((i, s['geom_id']))
            # GNN removed — free triangle connectivity from RAM, it's no longer needed at training time
            for geom in self.geometry_pool.values():
                geom.pop('elements', None)

        # 2. Group samples by geometry to split by geometry (no data leakage)
        geom_to_samples = collections.defaultdict(list)
        for s_idx, g_id in sample_to_geom:
            geom_to_samples[g_id].append(s_idx)
        
        unique_geoms = sorted(list(geom_to_samples.keys()))
        n_geoms = len(unique_geoms)
        
        # 3. Split by geometry (not by sample) to prevent leakage
        np.random.seed(random_seed)
        perm_geoms = np.random.permutation(unique_geoms)
        
        n_train_geoms = int(n_geoms * train_ratio)
        n_val_geoms = int(n_geoms * val_ratio)

        if split == 'train':
            active_geoms = perm_geoms[:n_train_geoms]
        elif split == 'val':
            active_geoms = perm_geoms[n_train_geoms : n_train_geoms + n_val_geoms]
        else:
            active_geoms = perm_geoms[n_train_geoms + n_val_geoms :]

        # 4. Flatten: collect all individual sample indices belonging to active geometries
        self.active_samples = []
        # H5 dosyasını N kere açmamak için bir kez açıp tüm filtrelemeyi tek pass'te yap
        if active_mode_index is not None and self.is_h5:
            import h5py
            with h5py.File(self.data_path, 'r') as f_filter:
                samples_grp = f_filter['samples']
                for g_id in active_geoms:
                    for s_idx in geom_to_samples[g_id]:
                        sample = samples_grp[str(s_idx)]
                        m_idx = int(sample.attrs.get('mode_idx', sample['Theta'][0]))
                        if m_idx == active_mode_index:
                            self.active_samples.append(s_idx)
        else:
            for g_id in active_geoms:
                indices = geom_to_samples[g_id]
                if active_mode_index is not None:
                    # PKL: samples_metadata bellekte, hızlı
                    for s_idx in indices:
                        m_idx = int(self.samples_metadata[s_idx]['Theta'][0])
                        if m_idx == active_mode_index:
                            self.active_samples.append(s_idx)
                else:
                    self.active_samples.extend(indices)
        
        # Store geometry pool reference for PKL access
        if not self.is_h5:
            # Build a quick sample_idx -> geom_id lookup
            self.sample_geom_lookup = {s_idx: g_id for s_idx, g_id in sample_to_geom}

        if self.stats:
            print(f"Freq Stats: mean={self.stats['mean']:.4f}, std={self.stats['std']:.4f}")
        print(f"Split: {split}, Samples: {len(self.active_samples)} (from {len(active_geoms)} geometries)")

        self.h5_handle = None

    def __len__(self):
        return len(self.active_samples)

    def _get_h5_handle(self):
        """Returns a worker-local H5 file handle.
        
        Each DataLoader worker (or the main process) opens its own handle
        so that multi-process access is safe.  The handle is opened lazily
        on first access and kept alive for the lifetime of the worker.
        """
        import h5py
        import os
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
        s_idx = self.active_samples[idx]
        
        if self.is_h5:
            f = self._get_h5_handle()
            sample = f['samples'][str(s_idx)]
            g_id = sample.attrs['geom_id']
            geom = f['geometry_pool'][str(g_id)]
            x = geom['X'][:]
            input_features = geom['Input_funcs'][:]
            y_val = sample['Y'][:]           # [N, 1]
            raw_theta = sample['Theta'][:]   # [mode_idx, freq]
        else:
            sample = self.samples_metadata[s_idx]
            g_id = sample['geom_id']
            geom = self.geometry_pool[g_id]
            x = geom['X']
            input_features = geom['Input_funcs']
            y_val = sample['Y']              # [N, 1]
            raw_theta = sample['Theta']      # [mode_idx, freq]

        # Mode index (0, 1, or 2) and frequency
        mode_idx = int(raw_theta[0])
        raw_freq = raw_theta[1]

        # If we are in single-mode training, map the active mode to 0
        if self.active_mode_index is not None:
            mode_idx = 0
        
        if self.stats:
            norm_freq = (raw_freq - self.stats['mean']) / self.stats['std']
        else:
            norm_freq = raw_freq

        # Ablation: select feature subset if feature_indices is set
        if self.feature_indices is not None:
            input_features = input_features[:, self.feature_indices]

        # VRAM Optimization: Node Sub-sampling
        n_nodes = x.shape[0]
        if self.max_nodes is not None and n_nodes > self.max_nodes:
            rand_idx = torch.randperm(n_nodes)[:self.max_nodes].numpy()
            x = x[rand_idx]
            input_features = input_features[rand_idx]
            y_val = y_val[rand_idx]

        return {
            'X': torch.from_numpy(x).float(),
            'Input_funcs': torch.from_numpy(input_features).float(),
            'Y_field': torch.from_numpy(y_val).float(),         # [N, 1]
            'Theta_in': torch.tensor([mode_idx], dtype=torch.long),  # [1]
            'Y_freq': torch.tensor([norm_freq], dtype=torch.float32), # [1]
            'geom_id': torch.tensor([int(g_id)], dtype=torch.long),
        }

def gnot_collate_fn(batch):
    batch_x = [item['X'] for item in batch]
    batch_inputs = [item['Input_funcs'] for item in batch]
    batch_y_field = [item['Y_field'] for item in batch]
    batch_theta = [item['Theta_in'] for item in batch]
    batch_freq = [item['Y_freq'] for item in batch]
    batch_geom_id = [item['geom_id'] for item in batch]

    X_padded = pad_sequence(batch_x, batch_first=True, padding_value=0.0)
    Inputs_padded = pad_sequence(batch_inputs, batch_first=True, padding_value=0.0)
    Y_field_padded = pad_sequence(batch_y_field, batch_first=True, padding_value=0.0)
    Theta_stacked = torch.stack(batch_theta)    # [B, 1]
    Freq_stacked = torch.stack(batch_freq)       # [B, 1]
    geom_id_stacked = torch.stack(batch_geom_id)

    lengths = [len(x) for x in batch_x]
    max_len = X_padded.size(1)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True

    return {
        'X': X_padded,
        'Input_funcs': Inputs_padded,
        'Y_field': Y_field_padded,       # [B, N, 1]
        'Theta_in': Theta_stacked,       # [B, 1]
        'Y_freq': Freq_stacked,          # [B, 1]
        'geom_id': geom_id_stacked,
        'Mask': mask,
    }
