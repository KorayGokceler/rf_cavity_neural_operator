"""Tests for GNOTDataset and gnot_collate_fn.

Kritik invariantlar:
- Geometri-bazlı split (data leakage yok)
- random_seed reproducibility
- max_nodes sub-sampling
- Collate output shapes ve mask doğruluğu
- 'elements' batch'te DEĞIL
"""
import pickle
import tempfile
from pathlib import Path

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
    """Aynı geometri ID'si train/val/test setlerinden yalnızca birinde olmalı."""
    pkl_path, n_geoms, n_modes = synthetic_pkl
    train = GNOTDataset(pkl_path, split='train', train_ratio=0.6, val_ratio=0.2)
    val = GNOTDataset(pkl_path, split='val', train_ratio=0.6, val_ratio=0.2)
    test = GNOTDataset(pkl_path, split='test', train_ratio=0.6, val_ratio=0.2)

    def geom_ids(ds):
        ids = set()
        for s_idx in ds.active_samples:
            ids.add(ds.sample_geom_lookup[s_idx])
        return ids

    train_ids = geom_ids(train)
    val_ids = geom_ids(val)
    test_ids = geom_ids(test)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert (train_ids | val_ids | test_ids) == set(range(n_geoms))


def test_random_seed_reproducibility(synthetic_pkl):
    """Aynı random_seed → aynı split."""
    pkl_path, _, _ = synthetic_pkl
    a = GNOTDataset(pkl_path, split='train', random_seed=42)
    b = GNOTDataset(pkl_path, split='train', random_seed=42)
    assert a.active_samples == b.active_samples


def test_random_seed_changes_split(synthetic_pkl):
    """Farklı seed → genelde farklı split."""
    pkl_path, _, _ = synthetic_pkl
    a = GNOTDataset(pkl_path, split='train', random_seed=42)
    b = GNOTDataset(pkl_path, split='train', random_seed=123)
    # Çok küçük ihtimalle aynı çıkabilirler — ama 10 geometride bu pratikte imkansız
    assert a.active_samples != b.active_samples


def test_elements_removed_from_geometry_pool(synthetic_pkl):
    """elements RAM optimizasyonu için pop edilmeli."""
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train')
    for g_id, geom in ds.geometry_pool.items():
        assert 'elements' not in geom, f"elements found in geom {g_id}"


def test_getitem_no_elements_key(synthetic_pkl):
    """__getitem__ output dict'inde 'elements' OLMAMALI."""
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train')
    item = ds[0]
    assert 'elements' not in item
    # Beklenen anahtarlar
    expected = {'X', 'Input_funcs', 'Y_field', 'Theta_in', 'Y_freq', 'geom_id'}
    assert expected.issubset(item.keys())


def test_getitem_tensor_dtypes(synthetic_pkl):
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train')
    item = ds[0]
    assert item['X'].dtype == torch.float32
    assert item['Input_funcs'].dtype == torch.float32
    assert item['Y_field'].dtype == torch.float32
    assert item['Theta_in'].dtype == torch.long
    assert item['Y_freq'].dtype == torch.float32


def test_max_nodes_truncation(synthetic_pkl):
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train', max_nodes=15)
    for i in range(len(ds)):
        item = ds[i]
        assert item['X'].shape[0] <= 15
        assert item['Input_funcs'].shape[0] <= 15
        assert item['Y_field'].shape[0] <= 15


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
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train')
    batch = [ds[i] for i in range(min(4, len(ds)))]
    out = gnot_collate_fn(batch)

    B = len(batch)
    assert out['X'].dim() == 3 and out['X'].shape[0] == B
    assert out['Input_funcs'].shape[:2] == out['X'].shape[:2]
    assert out['Y_field'].shape[:2] == out['X'].shape[:2]
    assert out['Theta_in'].shape == (B, 1)
    assert out['Y_freq'].shape == (B, 1)
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


def test_active_mode_filter(synthetic_pkl):
    """active_mode_index seçilirse sadece o moddan örnek olmalı + Theta_in 0'a remap edilmeli."""
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train', active_mode_index=1)
    for i in range(len(ds)):
        item = ds[i]
        # Tek mod aktif → Theta_in 0'a remap edilir (model num_field_modes=1 ile çalışır)
        assert item['Theta_in'].item() == 0


def test_freq_normalization(synthetic_pkl):
    """Y_freq z-score normalize edilmeli (mean=5.0, std=1.0)."""
    pkl_path, _, _ = synthetic_pkl
    ds = GNOTDataset(pkl_path, split='train')
    item = ds[0]
    raw_freq = ds.samples_metadata[ds.active_samples[0]]['Theta'][1]
    expected_norm = (raw_freq - 5.0) / 1.0
    assert abs(item['Y_freq'].item() - expected_norm) < 1e-5


def test_train_val_test_ratios_sum(synthetic_pkl):
    """train+val+test = total_geoms (round'ing tolerasyonu ile)."""
    pkl_path, n_geoms, n_modes = synthetic_pkl
    train = GNOTDataset(pkl_path, split='train', train_ratio=0.7, val_ratio=0.2)
    val = GNOTDataset(pkl_path, split='val', train_ratio=0.7, val_ratio=0.2)
    test = GNOTDataset(pkl_path, split='test', train_ratio=0.7, val_ratio=0.2)
    total = len(train.active_samples) + len(val.active_samples) + len(test.active_samples)
    assert total == n_geoms * n_modes
