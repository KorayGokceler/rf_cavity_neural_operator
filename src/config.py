"""Merkezi config yönetimi.

Kullanım:
    from src.config import load_config
    cfg = load_config("configs/default.yaml")
    print(cfg.model.embed_dim)        # 256
    print(cfg.training.learning_rate)  # 0.001
    
    # CLI override: 
    cfg = load_config("configs/default.yaml", overrides={"model.embed_dim": 128})
"""
import yaml
from pathlib import Path


class ConfigDict(dict):
    """Dict that supports attribute-style access (cfg.model.embed_dim)."""
    
    def __getattr__(self, key):
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, ConfigDict):
                val = ConfigDict(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key, val):
        self[key] = val

    def __repr__(self):
        return f"ConfigDict({dict.__repr__(self)})"


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override dict."""
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], val)
        else:
            base[key] = val
    return base


def load_config(config_path: str = "configs/default.yaml", overrides: dict = None) -> ConfigDict:
    """Load YAML config and apply optional overrides.
    
    Args:
        config_path: Path to YAML config file.
        overrides: Dict of dotted-key overrides, e.g. {"model.embed_dim": 128}
    
    Returns:
        ConfigDict with attribute-style access.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Apply dotted-key overrides: "model.embed_dim" → cfg["model"]["embed_dim"]
    if overrides:
        for dotted_key, val in overrides.items():
            keys = dotted_key.split('.')
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = val
    
    return ConfigDict(cfg)


def config_to_flat_dict(cfg: ConfigDict, prefix: str = "") -> dict:
    """Flatten nested config to dotted-key dict for logging.
    
    Example: {"model": {"embed_dim": 256}} → {"model.embed_dim": 256}
    """
    flat = {}
    for key, val in cfg.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            flat.update(config_to_flat_dict(val, full_key))
        else:
            flat[full_key] = val
    return flat
