"""End-to-end integration tests: dataset → collate → model → loss.

Bu testler "her şey birlikte çalışıyor mu" sorusunu cevaplar.
Sentetik bir pkl dataset üzerinden tek iterasyon koşulur.
"""
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import GNOTDataset, gnot_collate_fn
from src.training.lightning_module import GNOTLightning


def _build_synthetic_pkl(tmp_path, n_geoms=8, n_modes=3, n_feat=8):
    geometry_pool = {}
    samples = []
    rng = np.random.default_rng(0)
    for g_id in range(n_geoms):
        n_nodes = int(rng.integers(15, 30))
        nodes = rng.uniform(-1, 1, (n_nodes, 2)).astype(np.float32)
        # n_feat-channel feature with realistic dist_bnd (converter: 12)
        input_funcs = rng.uniform(0.1, 1.0, (n_nodes, n_feat)).astype(np.float32)
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


# ── End-to-end smoke: train.py (fit + test) → checkpoint → infer.py ──────────

def _write_cfg(tmp_path, data_path, model_type, **train_over):
    import yaml
    cfg = {
        'model_type': model_type,
        'dataset': {'data_path': data_path, 'train_ratio': 0.8, 'val_ratio': 0.1,
                    'random_seed': 0, 'max_nodes': None},
        'model': {'val_dim': 12, 'grid_dim': 2, 'embed_dim': 16, 'n_shared_layers': 0,
                  'n_mode_layers': 1, 'n_field_head_layers': 2, 'n_heads': 2,
                  'num_experts': 2, 'num_field_modes': 3, 'rff_dim': 8, 'n_basis': 4,
                  'use_checkpoint': False, 'predict_frequency': True},
        'training': {'learning_rate': 1e-3, 'weight_decay': 0.0, 'batch_size': 4,
                     'max_epochs': 1, 'check_val_every_n_epoch': 1,
                     'gradient_clip_val': 1.0, 'strategy': 'auto', 'freq_weight': 0.5,
                     'scheduler': 'custom_cosine', 'patience': 5, 'num_workers': 0,
                     'pin_memory': True, 'log_dir': str(tmp_path / 'logs'),
                     'exp_name': f'smoke_{model_type}', 'log_every_n_steps': 1,
                     'viz_every_n_epochs': 1, 'progress_bar_refresh_rate': 0,
                     'fast_dev_run': False},
        'inference': {'checkpoint_path': None, 'output_dir': str(tmp_path / 'plots'),
                      'num_visualize': 1},
    }
    cfg['training'].update(train_over)
    path = tmp_path / f'{model_type}.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _run_train(monkeypatch, cfg_path, *extra):
    import sys
    import train
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', cfg_path, *extra])
    train.main()


def test_train_spectral_fit_test_then_infer(tmp_path, monkeypatch):
    """SpectralNO: fit + trainer.test (grad-mode eval) + checkpoint + infer.py."""
    import glob
    import infer
    pkl_path, _, _ = _build_synthetic_pkl(tmp_path, n_geoms=10, n_feat=12)
    cfg_path = _write_cfg(tmp_path, pkl_path, 'spectral_no', augment=True)
    _run_train(monkeypatch, cfg_path)

    exp_dir = tmp_path / 'logs' / 'smoke_spectral_no'
    best = glob.glob(str(exp_dir / 'best-epoch=*-val_rel_l2=*.ckpt'))
    assert len(best) == 1, "ModelCheckpoint name must not create sub-directories"
    assert (exp_dir / 'last.ckpt').exists()

    model, _ = infer.load_model(str(exp_dir))
    assert model.hparams.data_cfg['zero_gauge_features'] is True
    assert model.scale_invariant_field

    infer.main(infer.parse_args(['--config', cfg_path, '--data_path', pkl_path]))
    assert glob.glob(str(tmp_path / 'plots' / 'sample_geom_*_all_modes.png'))


def test_train_gnot_reducelr_with_sparse_validation(tmp_path, monkeypatch):
    """ReduceLROnPlateau must only step on epochs that ran validation."""
    pkl_path, _, _ = _build_synthetic_pkl(tmp_path, n_geoms=10, n_feat=12)
    cfg_path = _write_cfg(tmp_path, pkl_path, 'gnot', scheduler='reducelr',
                          max_epochs=2, check_val_every_n_epoch=2)
    _run_train(monkeypatch, cfg_path)


def test_train_fast_dev_run_gnot(tmp_path, monkeypatch):
    pkl_path, _, _ = _build_synthetic_pkl(tmp_path, n_geoms=10, n_feat=12)
    cfg_path = _write_cfg(tmp_path, pkl_path, 'gnot', scheduler='onecycle')
    _run_train(monkeypatch, cfg_path, '--fast_dev_run')


def test_train_rejects_val_dim_mismatch(tmp_path, monkeypatch):
    import pytest
    pkl_path, _, _ = _build_synthetic_pkl(tmp_path, n_geoms=10, n_feat=8)
    cfg_path = _write_cfg(tmp_path, pkl_path, 'gnot')      # config says 12
    with pytest.raises(ValueError, match='val_dim=12'):
        _run_train(monkeypatch, cfg_path)
