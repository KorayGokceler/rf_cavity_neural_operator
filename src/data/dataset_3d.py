"""3D Maxwell (Whitney N0) eigenmode dataset for model_type='eigenspace3d'.

Reads the PKL written by convert_3d.py (contract: docs/20_3D_MODEL.md):
geometry_pool[g] = {X [Nv,3], Input_funcs [Nv,F], edges [Ne,2] (low→high),
tets, M, K [Ne×Ne], G [Ne×Nv], Kp [Nv×Nv] (CSR), scale, center, shape_type,
torsion_max}; samples = [{geom_id, Y [Ne] (N0 DOFs, unit M-norm),
Theta = [mode_idx, freq_GHz, sample_id]}].

Field (metadata['field']):
  'H' (or missing): H-field DOFs on all edges, G = full discrete gradient,
      Kp = GᵀMG singular on constants → batch['KpNull'] = 'const'.
  'E': E-field DOFs, PEC wall edges (geometry 'bnd_edge') are essential
      zeros; G = interior-vertex (+ boundary-component) potentials, zero
      columns after n_pot; Kp SPD on its valid block → batch['KpNull'] =
      'none', and batch['BndEdge'] [B, Ne] (padding False) masks the basis.
  A batch never mixes fields (maxwell3d_collate raises).

One item = one geometry with ALL its stored modes (sorted by frequency).
The split is by geometry with the same seeded permutation as GNOTDataset.
No node subsampling (the sparse operators need the full mesh).

Batching (maxwell3d_collate): vertex tensors padded to Nv_max, edge tensors
padded to Ne_max, and each sparse matrix becomes ONE block-diagonal torch
sparse tensor on the padded index space (sample b's edge e ↔ row b·Ne_max + e,
vertex v ↔ column b·Nv_max + v; padding = empty rows/columns).  One sparse
product then serves the whole batch (src/models/hcurl.py: spmm).

Cavity QoI (docs/24_CAVITY_QOI.md §0.4): items carry Y_qoi [K, n_qoi] (the
PKL's samples[i]['qoi'] labels in ds.qoi_names order, NaN where missing) and,
with qoi_ops=True, 'qoi_ops' = src.qoi.build_qoi_operators(...) of the stored
(un-augmented) mesh — N0 DOFs are rotation invariant, so the operators stay
valid under augment; the beam axis moves with the cavity.  The collate adds
Y_qoi and, when every item has qoi_ops, the QoI_* operator keys.
"""
import collections
import inspect
import pickle

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# Converter layout (used when the PKL has no metadata['feature_names']).
from src.data.dataset_converter_3d import FEATURE_NAMES_3D, geometry_operators, qoi_operators_of

# docs/24 §0.3 src.qoi.QOI_LABELS — fallback only when src.qoi cannot be imported (the dataset
# must load without it); a PKL's metadata['qoi']['labels'] takes precedence.
_QOI_LABELS_FALLBACK = ('Q0', 'G_ohm', 'R_over_Q_ohm', 'R_sh_ohm', 'T_transit', 'Epk_Eacc',
                        'Bpk_Eacc_mT_per_MVm')


def default_qoi_names():
    try:
        from src.qoi import QOI_LABELS
        return tuple(QOI_LABELS)
    except ImportError:
        return _QOI_LABELS_FALLBACK


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


FIELDS = ('H', 'E')
KP_NULL = {'H': 'const', 'E': 'none'}     # null space of Kp handled by hcurl._pcg


def field_of(meta):
    """'H' | 'E' from PKL metadata (missing / None → 'H', the old behaviour)."""
    f = str((meta or {}).get('field', None) or 'H').upper()
    if f not in FIELDS:
        raise ValueError(f"metadata['field'] must be one of {FIELDS}, got {f!r}")
    return f


def rebuild_operators(X, tets, field='H'):
    """Contract operators of one mesh from (X, tets) (lean PKLs): the converter's
    geometry_operators(X, tets, field=...) dict (M, K, G, Kp, edges[, bnd_edge])."""
    X, tets = np.asarray(X, np.float64), np.asarray(tets, np.int64)
    params = inspect.signature(geometry_operators).parameters
    if 'field' in params or any(p.kind is p.VAR_KEYWORD for p in params.values()):
        res = geometry_operators(X, tets, field=field)
    elif field == 'H':
        res = geometry_operators(X, tets)                   # pre-E converter: always H
    else:
        raise ValueError("this PKL (field='E') has no stored operators and "
                         "dataset_converter_3d.geometry_operators has no `field` argument: "
                         "update the converter or store the operators in the PKL.")
    ops = res[0] if isinstance(res, tuple) else res
    if field == 'E' and 'bnd_edge' not in ops:
        raise ValueError("geometry_operators(X, tets, field='E') returned no 'bnd_edge' "
                         "(PEC wall-edge mask), required by the E formulation.")
    return ops


def random_orthogonal(rng):
    """Haar-random 3×3 orthogonal matrix (rotation or rotoreflection).  N0 DOFs
    ∫_e F·t of a polar field (E) are invariant under any orthogonal map of mesh
    + field; an axial field (H) flips all of them under a reflection, which the
    sign-agnostic losses ignore.  (The operators are invariant either way.)"""
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    return (Q * np.sign(np.diag(R))).astype(np.float32)


class Maxwell3DDataset(Dataset):
    """Items: X [Nv,3], Input_funcs [Nv,F], Area [Nv] (node volume), Edges
    [Ne,2], Y_field [Ne,K], Y_freq [K] (z-scored, ascending), Scale,
    TorsionMax, geom_id, shape_type, field ('H' | 'E'), the scipy CSR
    operators M, K, G, Kp, for field 'E' BndEdge [Ne] bool (PEC wall edges)
    and, when the PKL has it, FreqNext (z-scored frequency of mode K+1: flags
    a degenerate pair split by the last stored mode).  Y_qoi float32 [K, n_qoi]:
    the stored cavity QoI labels (ds.qoi_names; NaN where the PKL has none) and,
    with qoi_ops=True, 'qoi_ops' (src.qoi.build_qoi_operators dict, built
    lazily and cached per geometry when cache_operators is set).

    PKLs written with convert_3d.py --no_operators are supported: the operators
    are rebuilt from (X, tets) by the converter's geometry_operators and kept
    in memory (per DataLoader worker) when cache_operators is set."""

    def __init__(self, data_path, split='train', train_ratio=0.8, val_ratio=0.1,
                 random_seed=42, augment=False, feature_indices=None, cache_operators=True,
                 qoi_ops=False):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.data_path, self.split = data_path, split
        self.is_h5 = False
        self.geometry_pool = data['geometry_pool']
        self.samples_metadata = data['samples']
        meta = data.get('metadata', {}) or {}
        self.metadata = meta
        self.field = field_of(meta)
        g_fields = {str(g['field']).upper() for g in self.geometry_pool.values() if g.get('field', None)}
        if g_fields - {self.field}:
            raise ValueError(f"metadata field {self.field!r} but geometry_pool fields {sorted(g_fields)}")
        self.stats = meta.get('freq_stats', None)
        self.feature_names = list(meta.get('feature_names', None) or FEATURE_NAMES_3D)
        self.feature_indices = feature_indices
        self.augment = bool(augment) and split == 'train'
        self.cache_operators = bool(cache_operators)
        self._ops_cache = {}
        self.qoi_ops = bool(qoi_ops)
        self._qoi_cache = {}
        q_meta = meta.get('qoi', None) or {}
        self.qoi_names = tuple(q_meta.get('labels', None) or default_qoi_names())
        self.has_qoi = any('qoi' in s for s in self.samples_metadata)

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
        print(f"Maxwell3DDataset {split}: {len(self.active_geoms)} geometries × {n_modes} modes"
              f" (field {self.field})")

    def data_dims(self):
        """(val_dim, n_modes) of the items."""
        if self.feature_indices is not None:
            return len(self.feature_indices), self.n_modes
        g = next(iter(self.geometry_pool.values()))
        return int(np.asarray(g['Input_funcs']).shape[-1]), self.n_modes

    def __len__(self):
        return len(self.active_geoms)

    def _ops(self, g_id):
        g = self.geometry_pool[g_id]
        ops = g if 'M' in g else self._ops_cache.get(g_id)
        if ops is None:
            ops = rebuild_operators(g['X'], g['tets'], self.field)
            if not np.array_equal(ops['edges'], np.asarray(g['edges'])):
                raise ValueError(f"geometry {g_id}: rebuilt edges differ from the stored ones")
            if self.cache_operators:
                self._ops_cache[g_id] = ops
        return ops

    def operators(self, g_id):
        """(M, K, G, Kp) scipy CSR of one geometry; rebuilt from (X, tets) if not stored."""
        g = self.geometry_pool[g_id]
        nv, ne = len(g['X']), len(g['edges'])
        ops = self._ops(g_id)
        shapes = {'M': (ne, ne), 'K': (ne, ne), 'G': (ne, nv), 'Kp': (nv, nv)}
        return tuple(to_csr(ops[k], shapes[k]) for k in ('M', 'K', 'G', 'Kp'))

    def bnd_edge(self, g_id):
        """bool [Ne] PEC wall-edge mask of an E geometry (stored, else rebuilt)."""
        g = self.geometry_pool[g_id]
        b = g.get('bnd_edge', None)
        if b is None:
            if 'M' in g:
                raise ValueError(f"geometry {g_id}: field 'E' PKL without 'bnd_edge'")
            b = self._ops(g_id)['bnd_edge']
        b = np.asarray(b, dtype=bool).reshape(-1)
        if len(b) != len(g['edges']):
            raise ValueError(f"geometry {g_id}: bnd_edge has {len(b)} entries, {len(g['edges'])} edges")
        return b

    def qoi_operators(self, g_id):
        """src.qoi.build_qoi_operators dict of one geometry (stored mesh, M from operators()),
        cached when cache_operators is set."""
        ops = self._qoi_cache.get(g_id)
        if ops is None:
            ops = qoi_operators_of(self.geometry_pool[g_id], self.field, M=self.operators(g_id)[0])
            if self.cache_operators:
                self._qoi_cache[g_id] = ops
        return ops

    def qoi_labels(self, s_idx):
        """float32 [len(s_idx), n_qoi] stored labels of the given samples (NaN where missing)."""
        nan = float('nan')
        return np.array([[float((self.samples_metadata[j].get('qoi', None) or {}).get(n, nan))
                          for n in self.qoi_names] for j in s_idx], dtype=np.float32).reshape(len(s_idx),
                                                                                              len(self.qoi_names))

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
            'Y_qoi': torch.from_numpy(self.qoi_labels(s_idx)),
        }
        item['M'], item['K'], item['G'], item['Kp'] = self.operators(g_key)
        item['field'] = self.field
        if self.qoi_ops:
            item['qoi_ops'] = self.qoi_operators(g_key)
        if self.field == 'E':
            bnd = self.bnd_edge(g_key)
            wall = np.abs(Y[bnd]).max(initial=0.0)
            if wall > 1e-6 * max(np.abs(Y).max(initial=0.0), 1e-30):
                raise ValueError(f"geometry {g_key}: E targets are nonzero on PEC wall edges "
                                 f"(max {wall:.3g}); contract: wall rows exactly 0")
            item['Y_field'][torch.from_numpy(bnd)] = 0.0
            item['BndEdge'] = torch.from_numpy(bnd)
        return item


def _block_diag(mats, rows, cols, layout='csr'):
    """Block-diagonal torch CSR (float64) of scipy matrices on a padded
    (rows × cols per block) layout.  CSR: ~3–8× faster sparse products than
    COO (review docs/22, m5) and no coalesce step.  layout='coo': a coalesced
    torch COO tensor (row-major) of the same matrix."""
    blocks = []
    for A in mats:
        A = sp.csr_matrix(A, dtype=np.float64)
        A.resize((rows, cols))
        blocks.append(A)
    S = sp.block_diag(blocks, format='csr')
    if layout == 'coo':
        S.sort_indices()
        C = S.tocoo()
        idx = torch.from_numpy(np.vstack([C.row, C.col]).astype(np.int64))
        return torch.sparse_coo_tensor(idx, torch.from_numpy(C.data.astype(np.float64)),
                                       size=S.shape).coalesce()
    if layout != 'csr':
        raise ValueError(f"layout must be 'csr' or 'coo', got {layout!r}")
    return torch.sparse_csr_tensor(torch.from_numpy(S.indptr.astype(np.int64)),
                                   torch.from_numpy(S.indices.astype(np.int64)),
                                   torch.from_numpy(S.data), size=S.shape,
                                   check_invariants=False)


def _mask(lengths, n):
    return torch.arange(n)[None, :] < torch.as_tensor(lengths)[:, None]


def _pad_rows(arrays, n, fill=0.0, dtype=torch.float64):
    """Stack 1-D (or [L, ...]) arrays padded along dim 0 to n with `fill`."""
    arrays = [torch.as_tensor(np.asarray(a)).to(dtype) for a in arrays]
    out = torch.full((len(arrays), n) + tuple(arrays[0].shape[1:]), fill, dtype=dtype)
    for b, a in enumerate(arrays):
        out[b, :a.shape[0]] = a
    return out


def _collate_qoi(batch, out, field, Ne):
    """QoI operator keys of docs/24 §0.4 (every item has 'qoi_ops'): block-diagonal COO
    float64 on the padded edge space, padded per-point arrays (float64), masks."""
    ops = [it['qoi_ops'] for it in batch]
    for o in ops:
        if str(o.get('field', field)).upper() != field:
            raise ValueError(f"maxwell3d_collate: qoi_ops field {o.get('field')!r} != batch field {field!r}")
    P = max(int(o['Az'].shape[0]) for o in ops)
    nf = [len(np.asarray(o['face_area']).reshape(-1)) for o in ops]
    Nf = max(nf)
    out['QoI_S'] = _block_diag([o['S'] for o in ops], Ne, Ne, layout='coo')
    out['QoI_Az'] = _block_diag([o['Az'] for o in ops], P, Ne, layout='coo')
    out['QoI_zeta'] = _pad_rows([np.asarray(o['zeta'], np.float64).reshape(-1) for o in ops], P)
    out['QoI_q'] = _pad_rows([np.asarray(o['q'], np.float64).reshape(-1) for o in ops], P)   # q = 0 pad
    out['QoI_Esurf'] = _block_diag([o['Esurf'] for o in ops], 3 * Nf, Ne, layout='coo')
    out['QoI_Hsurf'] = _block_diag([o['Hsurf'] for o in ops], 3 * Nf, Ne, layout='coo')
    out['QoI_area'] = _pad_rows([np.asarray(o['face_area'], np.float64).reshape(-1) for o in ops], Nf)
    out['QoI_SurfMask'] = _mask(nf, Nf)
    out['QoI_scale'] = torch.tensor([float(o['scale']) for o in ops], dtype=torch.float64)
    out['QoI_Laxis'] = torch.tensor([float(o['L_axis']) for o in ops], dtype=torch.float64)


def maxwell3d_collate(batch):
    """Pads vertex / edge tensors and builds block-diagonal sparse operators
    (see module docstring).  Keys: X, Input_funcs, Area, Mask [B,Nv]; Edges
    [B,Ne,2] (padding (0,0)), EdgeMask, Y_field [B,Ne,K]; Y_freq [B,K], Scale,
    TorsionMax, FreqNext [B]; geom_id [B,1]; shape_type list[str]; M, K, G, Gt, Kp
    (sparse CSR float64) and Kp_diag [B,Nv] (Jacobi preconditioner); field
    ('H' | 'E', one per batch), KpNull ('const' for H, 'none' for E) and, for
    E, BndEdge [B,Ne] bool (PEC wall edges, padding False).
    Cavity QoI (docs/24 §0.4): Y_qoi [B, K_max, n_qoi] float32 (NaN padding) when
    any item has it; when EVERY item has 'qoi_ops' also QoI_S [B·Ne × B·Ne],
    QoI_Az [B·P × B·Ne], QoI_Esurf / QoI_Hsurf [B·3Nf × B·Ne] (block-diagonal
    COO float64, same edge layout as M), QoI_zeta, QoI_q [B, P] (0 padding),
    QoI_area [B, Nf] (0 padding), QoI_SurfMask [B, Nf] bool, QoI_scale,
    QoI_Laxis [B] (float64; P, Nf = max over the batch)."""
    fields = {it.get('field', 'H') for it in batch}
    if len(fields) != 1:
        raise ValueError(f"maxwell3d_collate: a batch mixes fields {sorted(fields)}; "
                         "H and E items need separate datasets / loaders.")
    field = fields.pop()
    if field not in KP_NULL:
        raise ValueError(f"unknown field {field!r}, expected one of {FIELDS}")
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
    out['field'], out['KpNull'] = field, KP_NULL[field]
    if field == 'E':
        if any(it.get('BndEdge', None) is None for it in batch):
            raise ValueError("maxwell3d_collate: field 'E' item without 'BndEdge'")
        out['BndEdge'] = pad('BndEdge').bool()                                 # padding False
    if any(it.get('Y_qoi', None) is not None for it in batch):
        yq = [it.get('Y_qoi', None) for it in batch]
        nq = next(y.shape[-1] for y in yq if y is not None)
        K = max(y.shape[0] for y in yq if y is not None)
        out['Y_qoi'] = torch.full((len(batch), K, nq), float('nan'), dtype=torch.float32)
        for b, y in enumerate(yq):
            if y is not None:
                out['Y_qoi'][b, :y.shape[0]] = torch.as_tensor(y, dtype=torch.float32)
    if all(it.get('qoi_ops', None) is not None for it in batch):
        _collate_qoi(batch, out, field, Ne)
    return out


def item_from_geometry(g, feature_names=None, feature_indices=None, g_id=0, qoi_ops=False):
    """Model input for one converter geometry dict (dataset_converter_3d.extract_geometry_3d)
    without labels (Y_field / Y_freq dummies): label-free inference, e.g. active sampling.
    The field comes from g['field'] (default 'H'); an 'E' geometry needs g['bnd_edge'].
    qoi_ops=True adds 'qoi_ops' (src.qoi.build_qoi_operators; needs g['center'], else 0)
    so that the collated batch carries the QoI_* keys (no Y_qoi: no labels)."""
    field = str(g.get('field', None) or 'H').upper()
    if field not in KP_NULL:
        raise ValueError(f"geometry field must be one of {FIELDS}, got {field!r}")
    names = list(feature_names or FEATURE_NAMES_3D)
    _, _, vol_col = _feature_columns(names)
    F = np.asarray(g['Input_funcs'], dtype=np.float32)
    nv, ne = len(g['X']), len(g['edges'])
    vol = F[:, vol_col].copy() if vol_col is not None else np.ones(nv, np.float32)
    if feature_indices is not None:
        F = F[:, feature_indices]
    shapes = {'M': (ne, ne), 'K': (ne, ne), 'G': (ne, nv), 'Kp': (nv, nv)}
    item = {'X': torch.from_numpy(np.ascontiguousarray(g['X'], dtype=np.float32)),
            'Input_funcs': torch.from_numpy(np.ascontiguousarray(F)), 'Area': torch.from_numpy(vol),
            'Edges': torch.from_numpy(np.asarray(g['edges'], dtype=np.int64)),
            'Y_field': torch.zeros(ne, 1), 'Y_freq': torch.zeros(1),
            'Scale': torch.tensor(float(g['scale']), dtype=torch.float32),
            'TorsionMax': torch.tensor(float(g.get('torsion_max', 0.0) or 0.0), dtype=torch.float32),
            'geom_id': torch.tensor([g_id], dtype=torch.long), 'FreqNext': torch.tensor(float('nan')),
            'shape_type': str(g.get('shape_type', '')), 'field': field}
    for k in ('M', 'K', 'G', 'Kp'):
        item[k] = to_csr(g[k], shapes[k])
    if field == 'E':
        if g.get('bnd_edge', None) is None:
            raise ValueError("item_from_geometry: field 'E' geometry without 'bnd_edge'")
        bnd = np.asarray(g['bnd_edge'], dtype=bool).reshape(-1)
        if len(bnd) != ne:
            raise ValueError(f"bnd_edge has {len(bnd)} entries, {ne} edges")
        item['BndEdge'] = torch.from_numpy(bnd)
    if qoi_ops:
        item['qoi_ops'] = qoi_operators_of(g, field, M=item['M'])
    return item
