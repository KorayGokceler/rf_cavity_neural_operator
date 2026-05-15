"""End-to-end integration tests: dataset → collate → model → loss.

Bu testler "her şey birlikte çalışıyor mu" sorusunu cevaplar.
Sentetik bir pkl dataset üzerinden tek iterasyon koşulur.
"""
import pickle
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import GNOTDataset, gnot_collate_fn
from src.training.lightning_module import GNOTLightning


def _build_synthetic_pkl(tmp_path, n_geoms=8, n_modes=3):
    geometry_pool = {}
    samples = []
    rng = np.random.default_rng(0)
    for g_id in range(n_geoms):
        n_nodes = int(rng.integers(15, 30))
        nodes = rng.uniform(-1, 1, (n_nodes, 2)).astype(np.float32)
        # 8-channel feature with realistic dist_bnd
        input_funcs = rng.uniform(0.1, 1.0, (n_nodes, 8)).astype(np.float32)
        # Make first 2 nodes "boundary" with dist_bnd ~ 0
        input_funcs[:2, 2] = 0.0
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


def test_dataset_to_model_forward(tmp_path):
    """Bir batch al, modele ver, çıktıyı kontrol et."""
    pkl_path, n_geoms, n_modes = _build_synthetic_pkl(tmp_path)
    ds = GNOTDataset(pkl_path, split='train', train_ratio=0.7, val_ratio=0.15)
    loader = DataLoader(ds, batch_size=4, collate_fn=gnot_collate_fn, shuffle=False)

    model = GNOTLightning(
        val_dim=8, grid_dim=2, hidden_dim=16,
        n_shared_layers=1, n_mode_layers=1, n_field_head_layers=2,
        n_heads=2, num_experts=2, num_field_modes=3,
        lr=1e-3, freq_weight=0.5, smoothness_weight=0.0,
        scheduler='custom_cosine',
        predict_frequency=True,
        rff_dim=8,
    )
    model.eval()

    batch = next(iter(loader))
    with torch.no_grad():
        out = model(batch)

    assert out['field'].shape == (batch['X'].shape[0], batch['X'].shape[1], n_modes)
    assert out['freq'].shape == (batch['X'].shape[0], n_modes)
    assert (out['freq'][:, 1:] >= out['freq'][:, :-1]).all()
    assert torch.isfinite(out['field']).all()
    assert torch.isfinite(out['freq']).all()


def test_dataset_to_loss_backward(tmp_path):
    """Forward → loss → backward — gradient akışı kontrolü."""
    pkl_path, _, _ = _build_synthetic_pkl(tmp_path)
    ds = GNOTDataset(pkl_path, split='train')
    loader = DataLoader(ds, batch_size=4, collate_fn=gnot_collate_fn, shuffle=False)

    model = GNOTLightning(
        val_dim=8, grid_dim=2, hidden_dim=16,
        n_shared_layers=1, n_mode_layers=1, n_field_head_layers=2,
        n_heads=2, num_experts=2, num_field_modes=3,
        lr=1e-3, freq_weight=0.5, smoothness_weight=0.1,
        scheduler='custom_cosine',
        predict_frequency=True,
        rff_dim=8,
        degeneracy_mode='soft',
    )

    batch = next(iter(loader))
    loss, preds, targets = model._compute_loss(batch, "train")
    loss.backward()

    # En azından bir parametrenin gradı olmalı
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert has_grad
    assert torch.isfinite(loss).all()


def test_collate_compatible_with_pad_sequence(tmp_path):
    """Farklı boyutlardaki örnekler aynı batch'te padding ile birleştirilebilmeli."""
    pkl_path, _, _ = _build_synthetic_pkl(tmp_path)
    ds = GNOTDataset(pkl_path, split='train')

    # Birkaç örneği farklı boyutlarda al
    batch_items = [ds[i] for i in range(min(4, len(ds)))]
    sizes = [item['X'].shape[0] for item in batch_items]

    out = gnot_collate_fn(batch_items)
    max_n = max(sizes)
    assert out['X'].shape[1] == max_n
    # Mask doğru boyutu işaretliyor mu
    for i, n in enumerate(sizes):
        assert out['Mask'][i].sum().item() == n
