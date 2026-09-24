"""Tests for GNOTDataset and gnot_collate_fn (spectral-subspace arch).

Invariants:
- One item == one geometry == ALL K modes (Y_field [N,K], Y_freq [K])
- Geometry-based split (no data leakage)
- random_seed reproducibility
- max_nodes sub-sampling uses the SAME node set for every mode
- Collate output shapes & mask correctness
- 'elements' / 'Theta_in' NOT in batch
"""
import pickle

import numpy as np
import pytest
import torch

from src.data.dataset import GNOTDataset, gnot_collate_fn


def _build_synthetic_pkl(tmp_path, n_geoms=10, n_modes=3):
    """Sentetik pkl dataset oluştur (gerçek FEM/converter pipeline'ını çalıştırmadan)."""
    geometry_pool = {}
    samples = []
    rng = np.random.default_rng(0)
    for g_id in range(n_geoms):
        n_nodes = int(rng.integers(20, 40))
        nodes = rng.uniform(-1, 1, (n_nodes, 2)).astype(np.float32)
        # 8-channel feature: [x, y, dist_bnd, dir_bnd_x, dir_bnd_y, area, cos_p, sin_p]
        input_funcs = rng.uniform(-1, 1, (n_nodes, 8)).astype(np.float32)
        # Make some nodes "boundary" with dist_bnd ~ 0
        bnd_count = max(1, n_nodes // 5)
        input_funcs[:bnd_count, 2] = 0.0
        elements = rng.integers(0, n_nodes, (n_nodes // 2, 3)).astype(np.int32)
        geometry_pool[g_id] = {
            'X': nodes,
            'Input_funcs': input_funcs,
            'elements': elements,
        }
        for m in range(n_modes):
            Y = rng.uniform(-1, 1, (n_nodes, 1)).astype(np.float32)
            samples.append({
                'geom_id': g_id,
                'Y': Y,
                'Theta': np.array([float(m), 5.0 + m, float(g_id)], dtype=np.float32),
            })

    metadata = {
        'mode_indices': list(range(n_modes)),
        'n_geometries': n_geoms,
        'n_samples': n_geoms * n_modes,
        'freq_stats': {'mean': 5.0, 'std': 1.0, 'mode_stats': {}},
    }

    pkl_path = tmp_path / "synthetic.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'geometry_pool': geometry_pool,
            'samples': samples,
            'metadata': metadata,
        }, f)
    return str(pkl_path), n_geoms, n_modes


@pytest.fixture
def synthetic_pkl(tmp_path):
    return _build_synthetic_pkl(tmp_path, n_geoms=10, n_modes=3)


def test_geometry_split_no_leakage(synthetic_pkl):
    """A geometry ID must appear in exactly one of train/val/test."""
    pkl_path, n_geoms, n_modes = synthetic_pkl
    train = GNOTDataset(pkl_path, split='train', train_ratio=0.6, val_ratio=0.2)
    val = GNOTDataset(pkl_path, split='val', train_ratio=0.6, val_ratio=0.2)
    test = GNOTDataset(pkl_path, split='test', train_ratio=0.6, val_ratio=0.2)

    train_ids = set(train.active_geoms)
    val_ids = set(val.active_geoms)
    test_ids = set(test.active_geoms)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert (train_ids | val_ids | test_ids) == set(range(n_geoms))


def test_one_item_per_geometry(synthetic_pkl):
    """len(dataset) == number of geometries in the split (not per-mode)."""
    pkl_path, n_geoms, n_modes = synthetic_pkl
    train = GNOTDataset(pkl_path, split='train', train_ratio=0.6, val_ratio=0.2)
    val = GNOTDataset(pkl_path, split='val', train_ratio=0.6, val_ratio=0.2)
    test = GNOTDataset(pkl_path, split='test', train_ratio=0.6, val_ratio=0.2)
    assert len(train) + len(val) + len(test) == n_geoms
    assert len(train) == len(train.active_geoms)


def test_random_seed_reproducibility(synthetic_pkl):
    """Same random_seed → same split."""
    pkl_path, _, _ = synthetic_pkl
    a = GNOTDataset(pkl_path, split='train', random_seed=42)
    b = GNOTDataset(pkl_path, split='train', random_seed=42)
    assert a.active_geoms == b.active_geoms


def test_random_seed_changes_split(synthetic_pkl):
    """Different seed → generally different split."""
    pkl_path, _, _ = synthetic_pkl
    a = GNOTDataset(pkl_path, split='train', random_seed=42)
    b = GNOTDataset(pkl_path, split='train', random_seed=123)
    assert a.active_geoms != b.active_geoms


def test_elements_removed_from_geometry_pool(synthetic_pkl):
    """Node sub-sampling breaks connectivity → elements popped (RAM)."""
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train', max_nodes=16)
    for g_id, geom in ds.geometry_pool.items():
        assert 'elements' not in geom, f"elements found in geom {g_id}"
    assert 'Elements' not in ds[0]


def test_full_mesh_items_carry_elements(synthetic_pkl):
    """Full mesh (max_nodes=None): triangles go to the batch for P1 assembly;
    padded triangles are (0, 0, 0)."""
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train')
    items = [ds[i] for i in range(3)]
    for it in items:
        assert it['Elements'].dtype == torch.long and it['Elements'].shape[-1] == 3
        assert int(it['Elements'].max()) < it['X'].shape[0]
    batch = gnot_collate_fn(items)
    T = max(it['Elements'].shape[0] for it in items)
    assert batch['Elements'].shape == (3, T, 3)
    short = min(range(3), key=lambda i: items[i]['Elements'].shape[0])
    n_t = items[short]['Elements'].shape[0]
    assert (batch['Elements'][short, n_t:] == 0).all()


def test_getitem_no_elements_no_theta(synthetic_pkl):
    """__getitem__ must not contain 'elements' or 'Theta_in'."""
    pkl_path, _, n_modes = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train')
    item = ds[0]
    assert 'elements' not in item
    assert 'Theta_in' not in item
    expected = {'X', 'Input_funcs', 'Y_field', 'Y_freq', 'geom_id'}
    assert expected.issubset(item.keys())
    # All K modes bundled
    assert item['Y_field'].shape[1] == n_modes
    assert item['Y_freq'].shape == (n_modes,)
    assert item['Y_field'].shape[0] == item['X'].shape[0]


def test_freq_sorted_ascending(synthetic_pkl):
    """Y_freq is sorted ascending and Y_field columns follow that order."""
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train')
    for i in range(len(ds)):
        f = ds[i]['Y_freq']
        assert torch.all(f[1:] >= f[:-1])


def test_getitem_tensor_dtypes(synthetic_pkl):
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train')
    item = ds[0]
    assert item['X'].dtype == torch.float32
    assert item['Input_funcs'].dtype == torch.float32
    assert item['Y_field'].dtype == torch.float32
    assert item['Y_freq'].dtype == torch.float32
    assert item['geom_id'].dtype == torch.long


def test_max_nodes_truncation_shared_nodes(synthetic_pkl):
    """max_nodes caps every mode column to the SAME node set."""
    pkl_path, _, n_modes = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train', max_nodes=15)
    for i in range(len(ds)):
        item = ds[i]
        assert item['X'].shape[0] <= 15
        assert item['Input_funcs'].shape[0] <= 15
        assert item['Y_field'].shape[0] == item['X'].shape[0]
        assert item['Y_field'].shape[1] == n_modes


def test_max_nodes_no_truncation_when_smaller(synthetic_pkl):
    """max_nodes daha büyükse hiç bir şey kırpılmamalı."""
    pkl_path, _, _ = synthetic_pkl
    ds_full = GNOTDataset(pkl_path, split='train', max_nodes=None)
    ds_big = GNOTDataset(pkl_path, split='train', max_nodes=10000)
    for i in range(len(ds_full)):
        a = ds_full[i]
        b = ds_big[i]
        assert a['X'].shape == b['X'].shape


def test_collate_output_shapes(synthetic_pkl):
    pkl_path, _, n_modes = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train')
    batch = [ds[i] for i in range(min(4, len(ds)))]
    out = gnot_collate_fn(batch)

    B = len(batch)
    assert out['X'].dim() == 3 and out['X'].shape[0] == B
    assert out['Input_funcs'].shape[:2] == out['X'].shape[:2]
    assert out['Y_field'].shape[:2] == out['X'].shape[:2]
    assert out['Y_field'].shape[2] == n_modes
    assert 'Theta_in' not in out
    assert out['Y_freq'].shape == (B, n_modes)
    assert out['geom_id'].shape == (B, 1)
    assert out['Mask'].shape == out['X'].shape[:2]
    assert out['Mask'].dtype == torch.bool


def test_collate_no_elements(synthetic_pkl):
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train')
    batch = [ds[i] for i in range(min(3, len(ds)))]
    out = gnot_collate_fn(batch)
    assert 'elements' not in out


def test_collate_mask_marks_real_nodes_true(synthetic_pkl):
    """Her örneğin gerçek N kadar node'unda mask=True, padding'de False."""
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train')
    batch = [ds[i] for i in range(min(4, len(ds)))]
    out = gnot_collate_fn(batch)
    for i, item in enumerate(batch):
        n_real = item['X'].shape[0]
        assert out['Mask'][i, :n_real].all()
        if n_real < out['Mask'].shape[1]:
            assert not out['Mask'][i, n_real:].any()


def test_collate_padded_values_zero(synthetic_pkl):
    """Padding bölgelerinde padding_value=0 olduğundan X[i, n_real:] hep 0 olmalı."""
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train')
    batch = [ds[i] for i in range(min(4, len(ds)))]
    out = gnot_collate_fn(batch)
    for i, item in enumerate(batch):
        n_real = item['X'].shape[0]
        if n_real < out['X'].shape[1]:
            assert (out['X'][i, n_real:] == 0).all()


def test_dict_config_constructor(synthetic_pkl):
    """RFCavityDataset(cfg_dict, split=...) convenience must work."""
    from src.data.dataset import RFCavityDataset
    pkl_path, _, n_modes = synthetic_pkl
    cfg = {'dataset': {'data_path': pkl_path, 'train_ratio': 0.6,
                       'val_ratio': 0.2, 'random_seed': 7}}
    ds = RFCavityDataset(cfg, split='train')
    item = ds[0]
    assert item['Y_field'].shape[1] == n_modes


def test_freq_normalization(synthetic_pkl):
    """Y_freq z-score normalised with builder stats (mean=5.0, std=1.0).

    Builder freqs are [5, 6, 7] for modes [0,1,2]; after normalisation and
    ascending sort the values become [0, 1, 2]."""
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train')
    item = ds[0]
    f = item['Y_freq']
    assert torch.allclose(f, torch.tensor([0.0, 1.0, 2.0]), atol=1e-5)


def test_train_val_test_ratios_sum(synthetic_pkl):
    """train+val+test geometry count == total geometries."""
    pkl_path, n_geoms, n_modes = synthetic_pkl
    train = GNOTDataset(pkl_path, split='train', train_ratio=0.7, val_ratio=0.2)
    val = GNOTDataset(pkl_path, split='val', train_ratio=0.7, val_ratio=0.2)
    test = GNOTDataset(pkl_path, split='test', train_ratio=0.7, val_ratio=0.2)
    total = len(train) + len(val) + len(test)
    assert total == n_geoms


# ── Regression tests (train-infra review) ────────────────────────────────────

def _raw_input_funcs(ds, idx):
    g_id = ds.active_geoms[idx]
    s0 = ds.samples_metadata[ds.geom_to_samples[g_id][0]]
    return ds.geometry_pool[s0['geom_id']]['Input_funcs']


def test_split_identical_to_legacy_global_seed(synthetic_pkl):
    """Local RandomState must reproduce the old np.random.seed() split."""
    pkl_path, n_geoms, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train', random_seed=7)
    np.random.seed(7)
    legacy = np.random.permutation(list(range(n_geoms)))[:int(n_geoms * 0.8)]
    assert ds.active_geoms == [int(g) for g in legacy]


def test_augment_does_not_mutate_pool_and_zeros_gauge(synthetic_pkl):
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train', augment=True)
    before = np.array(_raw_input_funcs(ds, 0), copy=True)
    item = ds[0]
    assert np.array_equal(_raw_input_funcs(ds, 0), before)
    assert torch.all(item['Input_funcs'][:, 6:8] == 0)
    # x_norm / y_norm follow the rotated coordinates
    assert torch.allclose(item['Input_funcs'][:, :2], item['X'])


def test_zero_gauge_features_for_eval_split_of_augmented_run(synthetic_pkl):
    """val/test of an augmented run must see the same zeroed channels as train."""
    pkl_path, _, _ = synthetic_pkl
    val = GNOTDataset(pkl_path, split='val', zero_gauge_features=True)
    assert torch.all(val[0]['Input_funcs'][:, 6:8] == 0)
    plain = GNOTDataset(pkl_path, split='val')
    assert not torch.all(plain[0]['Input_funcs'][:, 6:8] == 0)


def test_feature_subset_keeps_dist_bnd(synthetic_pkl):
    """Ablation subsets without column 2 still expose dist_to_boundary."""
    pkl_path, _, n_modes = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train', feature_indices=[0, 1], augment=True)
    item = ds[0]
    assert item['Input_funcs'].shape[1] == 2
    assert torch.allclose(item['Dist_bnd'], torch.from_numpy(_raw_input_funcs(ds, 0)[:, 2]))
    assert ds.data_dims() == (2, n_modes)
    batch = gnot_collate_fn([ds[0], ds[1]])
    assert batch['Dist_bnd'].shape == batch['Mask'].shape


def test_data_dims_full_layout(synthetic_pkl):
    pkl_path, _, n_modes = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train')
    assert ds.data_dims() == (8, n_modes)


def test_incomplete_geometries_are_dropped(tmp_path):
    pkl_path, n_geoms, n_modes = _build_synthetic_pkl(tmp_path, n_geoms=10, n_modes=3)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    # geometry 0 loses its last mode (converter skips m_idx >= len(freqs))
    data['samples'] = [s for s in data['samples']
                       if not (s['geom_id'] == 0 and int(s['Theta'][0]) == n_modes - 1)]
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    all_geoms = set()
    for split in ('train', 'val', 'test'):
        ds = GNOTDataset(pkl_path, split=split)
        all_geoms |= set(ds.active_geoms)
        if len(ds) >= 2:
            gnot_collate_fn([ds[0], ds[1]])   # must not fail on torch.stack
    assert 0 not in all_geoms and len(all_geoms) == n_geoms - 1


def test_eval_subsampling_is_deterministic(synthetic_pkl):
    """val/test node sub-sampling must not change between epochs."""
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='val', max_nodes=10)
    assert torch.equal(ds[0]['X'], ds[0]['X'])
