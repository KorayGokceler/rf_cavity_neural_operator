"""3D Maxwell (H-field, Whitney N0) eigenmode dataset for model_type='eigenspace3d'.

Reads the PKL written by convert_3d.py (contract: docs/20_3D_MODEL.md):
geometry_pool[g] = {X [Nv,3], Input_funcs [Nv,F], edges [Ne,2] (low→high),
tets, M, K [Ne×Ne], G [Ne×Nv], Kp [Nv×Nv] (CSR), scale, center, shape_type,
torsion_max}; samples = [{geom_id, Y [Ne] (H-field N0 DOFs, unit M-norm),
Theta = [mode_idx, freq_GHz, sample_id]}].

One item = one geometry with ALL its stored modes (sorted by frequency).
The split is by geometry with the same seeded permutation as GNOTDataset.
No node subsampling (the sparse operators need the full mesh).

Batching (maxwell3d_collate): vertex tensors padded to Nv_max, edge tensors
padded to Ne_max, and each sparse matrix becomes ONE block-diagonal torch
COO tensor on the padded index space (sample b's edge e ↔ row b·Ne_max + e,
vertex v ↔ column b·Nv_max + v; padding = empty rows/columns).  One sparse
product then serves the whole batch (src/models/hcurl.py: spmm).
"""
import collections
import pickle

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# Converter layout (used when the PKL has no metadata['feature_names']).
from src.data.dataset_converter_3d import FEATURE_NAMES_3D, geometry_operators


def to_csr(obj, shape=None):
    """scipy CSR from a scipy matrix, an (indptr, indices, data[, shape]) tuple
    or a {'indptr', 'indices', 'data'[, 'shape']} dict."""
    if sp.issparse(obj):
        return obj.tocsr()
    if isinstance(obj, dict):
        obj = (obj['indptr'], obj['indices'], obj['data']) + ((obj['shape'],) if 'shape' in obj else ())
    indptr, indices, data = (np.asarray(a) for a in obj[:3])
    if len(obj) > 3:
        shape = tuple(int(s) for s in obj[3])
    if shape is None:
        shape = (len(indptr) - 1, int(indices.max()) + 1 if len(indices) else 0)
    return sp.csr_matrix((data.astype(np.float64), indices, indptr), shape=shape)


def _feature_columns(names):
    """(coordinate columns, boundary-direction columns, node-volume column) by name."""
    low = [str(n).lower() for n in names]
    coords = [low.index(c) for c in ('x', 'y', 'z')] if all(c in low for c in ('x', 'y', 'z')) else \
        [i for i, n in enumerate(low) if n in ('x_norm', 'y_norm', 'z_norm')]
    dirs = [i for i, n in enumerate(low) if n.startswith('dir')]
    vol = next((i for i, n in enumerate(low) if 'volume' in n), None)
    return (coords if len(coords) == 3 else None), (dirs if len(dirs) == 3 else None), vol


def random_orthogonal(rng):
    """Haar-random 3×3 orthogonal matrix (rotation or rotoreflection).  N0 DOFs
    ∫_e H·t are invariant under a rotation of mesh + field; a reflection flips
    all of them (H is axial), which the sign-agnostic losses ignore."""
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    return (Q * np.sign(np.diag(R))).astype(np.float32)


class Maxwell3DDataset(Dataset):
    """Items: X [Nv,3], Input_funcs [Nv,F], Area [Nv] (node volume), Edges
    [Ne,2], Y_field [Ne,K], Y_freq [K] (z-scored, ascending), Scale,
    TorsionMax, geom_id, shape_type, the scipy CSR operators M, K, G, Kp and,
    when the PKL has it, FreqNext (z-scored frequency of mode K+1: flags a
    degenerate pair split by the last stored mode).

    PKLs written with convert_3d.py --no_operators are supported: the operators
    are rebuilt from (X, tets) by the converter's geometry_operators and kept
    in memory (per DataLoader worker) when cache_operators is set."""

    def __init__(self, data_path, split='train', train_ratio=0.8, val_ratio=0.1,
                 random_seed=42, augment=False, feature_indices=None, cache_operators=True):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.data_path, self.split = data_path, split
        self.is_h5 = False
        self.geometry_pool = data['geometry_pool']
        self.samples_metadata = data['samples']
        meta = data.get('metadata', {}) or {}
        self.metadata = meta
        self.stats = meta.get('freq_stats', None)
        self.feature_names = list(meta.get('feature_names', None) or FEATURE_NAMES_3D)
        self.feature_indices = feature_indices
        self.augment = bool(augment) and split == 'train'
        self.cache_operators = bool(cache_operators)
        self._ops_cache = {}

        geom_to_samples = collections.defaultdict(list)
        for i, s in enumerate(self.samples_metadata):
            geom_to_samples[int(s['geom_id'])].append(i)
        n_modes = max((len(v) for v in geom_to_samples.values()), default=0)
        incomplete = [g for g, v in geom_to_samples.items() if len(v) != n_modes]
        if incomplete:
            print(f"WARNING: dropping {len(incomplete)} geometries with < {n_modes} modes")
            for g in incomplete:
                del geom_to_samples[g]
        for v in geom_to_samples.values():
            v.sort(key=lambda j: float(self.samples_metadata[j]['Theta'][1]))
        unique = sorted(geom_to_samples)
        perm = np.random.RandomState(random_seed).permutation(unique)
        n_tr, n_va = int(len(unique) * train_ratio), int(len(unique) * val_ratio)
        active = {'train': perm[:n_tr], 'val': perm[n_tr:n_tr + n_va]}.get(split, perm[n_tr + n_va:])
        self.geom_to_samples = geom_to_samples
        self.active_geoms = [int(g) for g in active]
        self.n_modes = n_modes
        self.coord_cols, self.dir_cols, self.vol_col = _feature_columns(self.feature_names)
        if self.augment and (self.coord_cols is None or self.dir_cols is None):
            raise ValueError(f"augment needs x,y,z and 3 dir_* feature columns, got {self.feature_names}")
        print(f"Maxwell3DDataset {split}: {len(self.active_geoms)} geometries × {n_modes} modes")

    def data_dims(self):
        """(val_dim, n_modes) of the items."""
        if self.feature_indices is not None:
            return len(self.feature_indices), self.n_modes
        g = next(iter(self.geometry_pool.values()))
        return int(np.asarray(g['Input_funcs']).shape[-1]), self.n_modes

    def __len__(self):
        return len(self.active_geoms)

    def operators(self, g_id):
        """(M, K, G, Kp) scipy CSR of one geometry; rebuilt from (X, tets) if not stored."""
        g = self.geometry_pool[g_id]
        nv, ne = len(g['X']), len(g['edges'])
        ops = g if 'M' in g else self._ops_cache.get(g_id)
        if ops is None:
            ops, _ = geometry_operators(np.asarray(g['X'], np.float64), np.asarray(g['tets'], np.int64))
            if not np.array_equal(ops['edges'], np.asarray(g['edges'])):
                raise ValueError(f"geometry {g_id}: rebuilt edges differ from the stored ones")
            if self.cache_operators:
                self._ops_cache[g_id] = ops
        shapes = {'M': (ne, ne), 'K': (ne, ne), 'G': (ne, nv), 'Kp': (nv, nv)}
        return tuple(to_csr(ops[k], shapes[k]) for k in ('M', 'K', 'G', 'Kp'))

    def __getitem__(self, idx):
        g_id = self.active_geoms[idx]
        s_idx = self.geom_to_samples[g_id]
        g_key = self.samples_metadata[s_idx[0]]['geom_id']
        g = self.geometry_pool[g_key]
        X = np.array(g['X'], dtype=np.float32, copy=True)
        F = np.array(g['Input_funcs'], dtype=np.float32, copy=True)
        edges = np.asarray(g['edges'], dtype=np.int64)
        nv = X.shape[0]
        Y = np.stack([np.asarray(self.samples_metadata[j]['Y'], dtype=np.float32).reshape(-1)
                      for j in s_idx], axis=1)                                   # [Ne, K]
        f = np.array([float(self.samples_metadata[j]['Theta'][1]) for j in s_idx], dtype=np.float32)
        f_next = g.get('freq_next', None)
        f_next = np.float32(np.nan if f_next is None else f_next)
        if self.stats:
            f = (f - self.stats['mean']) / self.stats['std']
            f_next = (f_next - self.stats['mean']) / self.stats['std']
        if self.augment:
            Q = random_orthogonal(np.random)   # worker-seeded global RNG, as GNOTDataset
            X = X @ Q.T
            F[:, self.coord_cols] = F[:, self.coord_cols] @ Q.T
            F[:, self.dir_cols] = F[:, self.dir_cols] @ Q.T
        vol = F[:, self.vol_col].copy() if self.vol_col is not None else np.ones(nv, np.float32)
        if self.feature_indices is not None:
            F = F[:, self.feature_indices]
        item = {
            'X': torch.from_numpy(np.ascontiguousarray(X)),
            'Input_funcs': torch.from_numpy(np.ascontiguousarray(F)),
            'Area': torch.from_numpy(vol),
            'Edges': torch.from_numpy(edges),
            'Y_field': torch.from_numpy(Y),
            'Y_freq': torch.from_numpy(f),
            'Scale': torch.tensor(float(g['scale']), dtype=torch.float32),
            'TorsionMax': torch.tensor(float(g.get('torsion_max', 0.0) or 0.0), dtype=torch.float32),
            'geom_id': torch.tensor([g_id], dtype=torch.long),
            'FreqNext': torch.tensor(float(f_next), dtype=torch.float32),   # NaN if unknown
            'shape_type': str(g.get('shape_type', '')),
        }
        item['M'], item['K'], item['G'], item['Kp'] = self.operators(g_key)
        return item


def _block_diag(mats, rows, cols):
    """Block-diagonal torch COO (float64) of scipy matrices on a padded
    (rows × cols per block) layout."""
    r, c, v = [], [], []
    for b, A in enumerate(mats):
        A = A.tocoo()
        r.append(A.row.astype(np.int64) + b * rows)
        c.append(A.col.astype(np.int64) + b * cols)
        v.append(A.data.astype(np.float64))
    idx = torch.from_numpy(np.stack([np.concatenate(r), np.concatenate(c)]))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(np.concatenate(v)),
                                   (len(mats) * rows, len(mats) * cols), check_invariants=False).coalesce()


def _mask(lengths, n):
    return torch.arange(n)[None, :] < torch.as_tensor(lengths)[:, None]


def maxwell3d_collate(batch):
    """Pads vertex / edge tensors and builds block-diagonal sparse operators
    (see module docstring).  Keys: X, Input_funcs, Area, Mask [B,Nv]; Edges
    [B,Ne,2] (padding (0,0)), EdgeMask, Y_field [B,Ne,K]; Y_freq [B,K], Scale,
    TorsionMax, FreqNext [B]; geom_id [B,1]; shape_type list[str]; M, K, G, Gt, Kp
    (sparse COO float64) and Kp_diag [B,Nv] (Jacobi preconditioner)."""
    pad = lambda k: pad_sequence([it[k] for it in batch], batch_first=True)   # noqa: E731
    nv = [it['X'].shape[0] for it in batch]
    ne = [it['Edges'].shape[0] for it in batch]
    Nv, Ne = max(nv), max(ne)
    out = {k: pad(k) for k in ('X', 'Input_funcs', 'Area', 'Edges', 'Y_field')}
    out['Mask'], out['EdgeMask'] = _mask(nv, Nv), _mask(ne, Ne)
    for k in ('Y_freq', 'Scale', 'TorsionMax', 'FreqNext', 'geom_id'):
        out[k] = torch.stack([it[k] for it in batch])
    out['shape_type'] = [it['shape_type'] for it in batch]
    out['M'] = _block_diag([it['M'] for it in batch], Ne, Ne)
    out['K'] = _block_diag([it['K'] for it in batch], Ne, Ne)
    out['G'] = _block_diag([it['G'] for it in batch], Ne, Nv)
    out['Gt'] = _block_diag([it['G'].T for it in batch], Nv, Ne)
    out['Kp'] = _block_diag([it['Kp'] for it in batch], Nv, Nv)
    out['Kp_diag'] = pad_sequence([torch.from_numpy(it['Kp'].diagonal().astype(np.float64))
                                   for it in batch], batch_first=True)
    return out
