import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class GNOTDataset(Dataset):
    def __init__(self, pkl_path, split='train', train_ratio=0.8, val_ratio=0.1):
        print(f"Loading dataset from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        self.geometry_pool = data['geometry_pool']
        all_samples = data['samples']

        n_total = len(all_samples)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        np.random.seed(42)
        perm = np.random.permutation(n_total)
        all_samples = [all_samples[i] for i in perm]

        if split == 'train':
            self.samples = all_samples[:n_train]
        elif split == 'val':
            self.samples = all_samples[n_train:n_train+n_val]
        else:
            self.samples = all_samples[n_train+n_val:]

        print(f"Split: {split}, Count: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        geom = self.geometry_pool[sample['geom_id']]
        input_features = geom['Input_funcs'][0]
        raw_theta = sample['Theta']

        return {
            'X': torch.from_numpy(geom['X']),
            'Input_funcs': torch.from_numpy(input_features),
            'Y_field': torch.from_numpy(sample['Y']),
            'Y_freq': torch.from_numpy(np.array([raw_theta[1]], dtype=np.float32)),
            'Theta_in': torch.from_numpy(np.array([raw_theta[0]], dtype=np.float32))
        }

def gnot_collate_fn(batch):
    batch_x = [item['X'] for item in batch]
    batch_inputs = [item['Input_funcs'] for item in batch]
    batch_y_field = [item['Y_field'] for item in batch]
    batch_y_freq = [item['Y_freq'] for item in batch]
    batch_theta_in = [item['Theta_in'] for item in batch]

    X_padded = pad_sequence(batch_x, batch_first=True, padding_value=0.0)
    Inputs_padded = pad_sequence(batch_inputs, batch_first=True, padding_value=0.0)
    Y_field_padded = pad_sequence(batch_y_field, batch_first=True, padding_value=0.0)
    Y_freq_stacked = torch.stack(batch_y_freq)
    Theta_in_stacked = torch.stack(batch_theta_in)

    return {
        'X': X_padded,
        'Input_funcs': Inputs_padded,
        'Y_field': Y_field_padded,
        'Y_freq': Y_freq_stacked,
        'Theta_in': Theta_in_stacked
    }
