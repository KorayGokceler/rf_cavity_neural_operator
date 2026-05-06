"""Tests for src/config.py — YAML loading, override parsing, flatten."""
import pytest
import yaml

from src.config import ConfigDict, _deep_update, load_config, config_to_flat_dict


def test_config_dict_attribute_access():
    cfg = ConfigDict({'a': 1, 'b': {'c': 2}})
    assert cfg.a == 1
    assert cfg.b.c == 2


def test_config_dict_missing_key_raises_attribute_error():
    cfg = ConfigDict({'a': 1})
    with pytest.raises(AttributeError):
        _ = cfg.nonexistent


def test_config_dict_set_attribute():
    cfg = ConfigDict({})
    cfg.x = 5
    assert cfg['x'] == 5


def test_deep_update_recursive():
    base = {'a': 1, 'b': {'c': 2, 'd': 3}}
    override = {'b': {'c': 20, 'e': 4}}
    out = _deep_update(base, override)
    assert out == {'a': 1, 'b': {'c': 20, 'd': 3, 'e': 4}}


def test_load_config_from_file(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("model:\n  embed_dim: 256\n  n_heads: 8\n")
    cfg = load_config(str(cfg_file))
    assert cfg.model.embed_dim == 256
    assert cfg.model.n_heads == 8


def test_load_config_with_overrides(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("model:\n  embed_dim: 256\ntraining:\n  lr: 0.001\n")
    cfg = load_config(str(cfg_file), overrides={'model.embed_dim': 128, 'training.lr': 0.0001})
    assert cfg.model.embed_dim == 128
    assert cfg.training.lr == 0.0001


def test_load_config_override_creates_new_keys(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("model:\n  embed_dim: 256\n")
    cfg = load_config(str(cfg_file), overrides={'model.new_key': 'new_value'})
    assert cfg.model.new_key == 'new_value'


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_config_to_flat_dict():
    cfg = ConfigDict({
        'a': 1,
        'b': {'c': 2, 'd': {'e': 3}},
    })
    flat = config_to_flat_dict(cfg)
    assert flat == {'a': 1, 'b.c': 2, 'b.d.e': 3}


def test_default_config_loads(tmp_path):
    """Repodaki configs/default.yaml düzgün parse edilebilmeli."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    default_cfg = repo_root / "configs" / "default.yaml"
    if not default_cfg.exists():
        pytest.skip("configs/default.yaml not found")
    cfg = load_config(str(default_cfg))
    # Beklenen kritik anahtarların varlığı
    assert hasattr(cfg, 'model')
    assert hasattr(cfg, 'training')
    assert hasattr(cfg, 'dataset')
    assert hasattr(cfg, 'data_gen')
    # Kritik tipler
    assert isinstance(cfg.model.embed_dim, int)
    assert isinstance(cfg.training.learning_rate, float)
    assert isinstance(cfg.training.batch_size, int)
