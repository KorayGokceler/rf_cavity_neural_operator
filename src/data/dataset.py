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

    def __init__(self, data_path, split='train', train_ratio=0.8, val_ratio=0.1, feature_indices=None, max_nodes=None):
        print(f"Loading dataset from {data_path}...")
        self.data_path = data_path
        self.is_h5 = str(data_path).endswith('.h5')
        self.feature_indices = feature_indices  # e.g. [0,1] for xy-only, None=all
        self.max_nodes = max_nodes  # Optional: cap sequence length to prevent VRAM overflow

        # 1. Collect mapping: sample_index -> geom_id
        sample_to_geom = []
        
        if self.is_h5:
            import h5py, json
            with h5py.File(data_path, 'r') as f:
                metadata = json.loads(f.attrs['metadata'])
                self.stats = metadata.get('freq_stats', None)
                self.n_samples_total = metadata['n_samples']
                # Store sample indices for subsetting
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

        # 2. Group samples by geometry
        geom_to_samples = collections.defaultdict(list)
        for s_idx, g_id in sample_to_geom:
            geom_to_samples[g_id].append(s_idx)
        
        unique_geoms = sorted(list(geom_to_samples.keys()))
        n_geoms = len(unique_geoms)
        
        # 3. Shuffle geometries (not samples!) to keep modes of same geometry together
        np.random.seed(42)
        perm_geoms = np.random.permutation(unique_geoms)
        
        n_train_geoms = int(n_geoms * train_ratio)
        n_val_geoms = int(n_geoms * val_ratio)

        if split == 'train':
            active_geoms = perm_geoms[:n_train_geoms]
        elif split == 'val':
            active_geoms = perm_geoms[n_train_geoms : n_train_geoms + n_val_geoms]
        else:
            active_geoms = perm_geoms[n_train_geoms + n_val_geoms :]

        # 4. Collect all sample indices for the selected geometries
        self.active_indices = []
        for g_id in active_geoms:
            self.active_indices.extend(geom_to_samples[g_id])
        
        # Optional: shuffle active_indices so modes are mixed within batches
        if split == 'train':
            np.random.shuffle(self.active_indices)

        if self.stats:
            print(f"Freq Stats: mean={self.stats['mean']:.4f}, std={self.stats['std']:.4f}")
        print(f"Split: {split}, Count: {len(self.active_indices)}")

        self.h5_handle = None

    def __len__(self):
        return len(self.active_indices)

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
        real_idx = self.active_indices[idx]
        
        if self.is_h5:
            f = self._get_h5_handle()
            sample = f['samples'][str(real_idx)]
            geom_id = sample.attrs['geom_id']
            geom = f['geometry_pool'][str(geom_id)]
            
            x = geom['X'][:]
            input_features = geom['Input_funcs'][:]
            y_field = sample['Y'][:]
            raw_theta = sample['Theta'][:]
        else:
            sample = self.samples_metadata[real_idx]
            geom = self.geometry_pool[sample['geom_id']]
            x = geom['X']
            input_features = geom['Input_funcs']  # plain numpy array, no tuple
            y_field = sample['Y']
            raw_theta = sample['Theta']

        # raw_theta format: [m_idx, freq, sample_id]
        raw_freq = raw_theta[1]
        if self.stats:
            norm_freq = (raw_freq - self.stats['mean']) / self.stats['std']
        else:
            norm_freq = raw_freq

        # Ablation: select feature subset if feature_indices is set
        if self.feature_indices is not None:
            input_features = input_features[:, self.feature_indices]

        # VRAM Optimization: Node Sub-sampling
        # Randomly select a subset of nodes if the point cloud exceeds max_nodes.
        # This completely flattens VRAM peaks and eliminates massive zero-padding waste!
        n_nodes = x.shape[0]
        if self.max_nodes is not None and n_nodes > self.max_nodes:
            # We use torch.randperm instead of np.random to avoid DataLoader worker RNG duplication
            rand_idx = torch.randperm(n_nodes)[:self.max_nodes].numpy()
            x = x[rand_idx]
            input_features = input_features[rand_idx]
            y_field = y_field[rand_idx]

        return {
            'X': torch.from_numpy(x).float(),
            'Input_funcs': torch.from_numpy(input_features).float(),
            'Y_field': torch.from_numpy(y_field).float(),
            'Y_freq': torch.from_numpy(np.array([norm_freq], dtype=np.float32)),
            'Theta_in': torch.tensor([int(raw_theta[0])], dtype=torch.long),
            'geom_id': torch.tensor([int(raw_theta[2])], dtype=torch.long),
            'elements': torch.from_numpy(geom['elements']).long()
        }

def gnot_collate_fn(batch):
    batch_x = [item['X'] for item in batch]
    batch_inputs = [item['Input_funcs'] for item in batch]
    batch_y_field = [item['Y_field'] for item in batch]
    batch_y_freq = [item['Y_freq'] for item in batch]
    batch_theta_in = [item['Theta_in'] for item in batch]
    batch_geom_id = [item['geom_id'] for item in batch]
    batch_elements = [item['elements'] for item in batch]

    X_padded = pad_sequence(batch_x, batch_first=True, padding_value=0.0)
    Inputs_padded = pad_sequence(batch_inputs, batch_first=True, padding_value=0.0)
    Y_field_padded = pad_sequence(batch_y_field, batch_first=True, padding_value=0.0)
    Y_freq_stacked = torch.stack(batch_y_freq)
    Theta_in_stacked = torch.stack(batch_theta_in)
    geom_id_stacked = torch.stack(batch_geom_id)

    lengths = [len(x) for x in batch_x]
    max_len = X_padded.size(1)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True

    return {
        'X': X_padded,
        'Input_funcs': Inputs_padded,
        'Y_field': Y_field_padded,
        'Y_freq': Y_freq_stacked,
        'Theta_in': Theta_in_stacked,
        'geom_id': geom_id_stacked,
        'Mask': mask,
        'elements': batch_elements
    }
