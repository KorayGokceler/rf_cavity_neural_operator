"""Train the 3D cavity eigenmode model (EigenspaceOperator3D, E field).

    python train.py --config configs/eigenspace_3d.yaml --override dataset.data_path=data/x.pkl
    python train.py ... --resume training_logs/<exp>/last.ckpt          # continue a run
"""
import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.config import config_to_flat_dict, load_config
from src.data.dataset_3d import Maxwell3DDataset, maxwell3d_collate
from src.training.lightning_module import CavityLightning, count_near_degenerate
from src.training.progress import ProgressFile


def parse_args():
    p = argparse.ArgumentParser(description="Train the 3D cavity eigenmode model.")
    p.add_argument("--config", default="configs/eigenspace_3d.yaml", help="YAML config.")
    p.add_argument("--override", nargs="*", default=[],
                   help="config overrides key=value (e.g. model.embed_dim=192)")
    p.add_argument("--fast_dev_run", action="store_true", help="one batch of train / val / test")
    p.add_argument("--resume", default=None, help="checkpoint to continue from (last.ckpt)")
    p.add_argument("--qoi_weight", type=float, default=None,
                   help="training.qoi_weight: weight of the QoI log-ratio loss (0 = off)")
    p.add_argument("--qoi_terms", default=None,
                   help="training.qoi_terms, comma separated (default Q0,R_over_Q_ohm,G_ohm)")
    p.add_argument("--qoi_metrics", action="store_true",
                   help="training.qoi_metrics: QoI operators + qoi_*_rel_err metrics with qoi_weight = 0")
    return p.parse_args()


def parse_overrides(override_list):
    """Parse CLI overrides like ['model.embed_dim=128', 'training.mode_loss_weights=[1.0,2.0,2.0]']."""
    import json
    overrides = {}
    for item in override_list:
        key, val_str = item.split("=", 1)
        # JSON list/dict desteği: [1.0, 2.0] veya {"a": 1}
        if val_str.startswith("[") or val_str.startswith("{"):
            try:
                val = json.loads(val_str)
                overrides[key] = val
                continue
            except json.JSONDecodeError:
                pass
        # Otomatik tip dönüşümü
        try:
            val = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                if val_str.lower() == "true":
                    val = True
                elif val_str.lower() == "false":
                    val = False
                elif val_str.lower() == "null" or val_str.lower() == "none":
                    val = None
                else:
                    val = val_str
        overrides[key] = val
    return overrides

_EIGENSPACE3D_KEYS = ('n_layers', 'edge_rff', 'mass_ridge', 'drop_tol', 'eig_broadening',
                      'kp_tol', 'kp_maxiter')


def _eigenspace3d_kwargs(mc):
    """EigenspaceOperator3D kwargs: top-level model keys overridden by `model.eigenspace`."""
    kw = {k: mc[k] for k in _EIGENSPACE3D_KEYS if k in mc}
    kw.update(dict(getattr(mc, 'eigenspace', None) or {}))
    return kw or None


def _qoi_enabled(tc):
    """QoI operators needed: the loss is on (training.qoi_weight > 0) or only the
    metrics are requested (training.qoi_metrics)."""
    return float(getattr(tc, 'qoi_weight', 0.0) or 0.0) > 0 or bool(getattr(tc, 'qoi_metrics', False))


def _datasets_3d(dc, random_seed, augment, feature_indices, qoi_ops=False):
    """train / val / test Maxwell3DDataset."""
    kw = dict(train_ratio=dc.train_ratio, val_ratio=dc.val_ratio, random_seed=random_seed,
              feature_indices=feature_indices,
              # false for large PKLs written with --no_operators: rebuild M/K/G/Kp
              # per item instead of keeping every geometry's operators in RAM.
              cache_operators=getattr(dc, 'cache_operators', True))
    if qoi_ops:                    # per-geometry QoI operators (docs/24 §0.4), cached per geometry
        kw['qoi_ops'] = True
    return tuple(Maxwell3DDataset(dc.data_path, split=s, augment=augment and s == 'train', **kw)
                 for s in ('train', 'val', 'test'))


def main():
    args = parse_args()
    overrides = parse_overrides(args.override)
    if args.fast_dev_run:
        overrides["training.fast_dev_run"] = True
    if args.qoi_weight is not None:
        overrides["training.qoi_weight"] = float(args.qoi_weight)
    if args.qoi_terms is not None:
        overrides["training.qoi_terms"] = [t.strip() for t in args.qoi_terms.split(",") if t.strip()]
    if args.qoi_metrics:
        overrides["training.qoi_metrics"] = True
    cfg = load_config(args.config, overrides=overrides)
    tc, mc, dc = cfg.training, cfg.model, cfg.dataset

    rank0 = int(os.environ.get('LOCAL_RANK', 0)) == 0
    if rank0:
        print(f"Config: {args.config} | embed_dim={mc.embed_dim}, heads={mc.n_heads}, n_basis={mc.n_basis}, "
              f"K={mc.num_field_modes} | lr={tc.learning_rate}, batch={tc.batch_size}, epochs={tc.max_epochs} | "
              f"GPUs: {torch.cuda.device_count()}")
    # 'high' = TF32 on Ampere+ GPUs, exact fp32 on CPU ('medium' would run fp32 matmuls in bf16)
    torch.set_float32_matmul_precision(getattr(tc, 'matmul_precision', None) or 'high')
    feature_indices = getattr(dc, 'feature_indices', None)
    augment = getattr(tc, 'augment', False)
    random_seed = getattr(dc, 'random_seed', 42)
    pl.seed_everything(random_seed, workers=True)       # model init / shuffle / augmentation
    qoi_ops = _qoi_enabled(tc)
    train_dataset, val_dataset, test_dataset = _datasets_3d(dc, random_seed, augment, feature_indices,
                                                            qoi_ops=qoi_ops)
    for name, ds in (('train', train_dataset), ('val', val_dataset), ('test', test_dataset)):
        if len(ds) == 0:     # an empty split never logs the monitored metric: fail early instead
            raise ValueError(f"'{name}' split is empty ({len(train_dataset.geom_to_samples)} geometries, "
                             f"train_ratio={dc.train_ratio}, val_ratio={dc.val_ratio}): use more data.")

    data_val_dim, data_n_modes = train_dataset.data_dims()
    if getattr(mc, 'val_dim', None) in (None, 'auto'):
        mc.val_dim = data_val_dim
    elif int(mc.val_dim) != data_val_dim:
        raise ValueError(f"model.val_dim={mc.val_dim} but the dataset provides {data_val_dim} input features "
                         "per vertex (feature_indices?): set model.val_dim: null.")
    if data_n_modes < int(mc.num_field_modes):
        raise ValueError(f"model.num_field_modes={mc.num_field_modes} but the dataset stores only "
                         f"{data_n_modes} modes per geometry.")

    loader_kw = dict(collate_fn=maxwell3d_collate, num_workers=tc.num_workers,
                     pin_memory=False,                   # sparse COO batches cannot be pinned
                     persistent_workers=tc.num_workers > 0,
                     prefetch_factor=2 if tc.num_workers > 0 else None)
    train_loader = DataLoader(train_dataset, batch_size=tc.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_dataset, batch_size=tc.batch_size, **loader_kw)
    test_loader = DataLoader(test_dataset, batch_size=tc.batch_size, **loader_kw)

    model = CavityLightning(
        val_dim=mc.val_dim, num_field_modes=mc.num_field_modes, hidden_dim=mc.embed_dim,
        n_heads=mc.n_heads, n_basis=mc.n_basis, rff_dim=getattr(mc, 'rff_dim', 64),
        rff_length_scale=getattr(mc, 'rff_length_scale', 0.1), dropout=getattr(mc, 'dropout', 0.0),
        eigenspace_kwargs=_eigenspace3d_kwargs(mc),
        lr=tc.learning_rate, weight_decay=tc.weight_decay, scheduler=tc.scheduler,
        cosine_eta_min=getattr(tc, 'cosine_eta_min', 1e-6),
        onecycle_pct_start=getattr(tc, 'onecycle_pct_start', 0.3),
        onecycle_div_factor=getattr(tc, 'onecycle_div_factor', 25.0),
        onecycle_final_div_factor=getattr(tc, 'onecycle_final_div_factor', 1e4),
        reducelr_patience=getattr(tc, 'reducelr_patience', 10),
        reducelr_factor=getattr(tc, 'reducelr_factor', 0.5),
        gradient_clip_val=tc.gradient_clip_val, freq_weight=tc.freq_weight,
        span_weight=getattr(tc, 'span_weight', 1.0), selfsup_weight=getattr(tc, 'selfsup_weight', 0.01),
        ortho_weight=getattr(tc, 'ortho_weight', 0.01), span_norm=getattr(tc, 'span_norm', 'both'),
        span_root=getattr(tc, 'span_root', True), span_ridge=getattr(tc, 'span_ridge', 1e-9),
        selfsup_form=getattr(tc, 'selfsup_form', 'compliance'),
        near_deg_threshold=getattr(tc, 'near_deg_threshold', 0.05),
        near_deg_rel_threshold=getattr(tc, 'near_deg_rel_threshold', None),
        qoi_weight=float(getattr(tc, 'qoi_weight', 0.0) or 0.0),
        qoi_terms=getattr(tc, 'qoi_terms', None) or ('Q0', 'R_over_Q_ohm', 'G_ohm'),
        qoi_peak_p=getattr(tc, 'qoi_peak_p', None), qoi_rq_floor=getattr(tc, 'qoi_rq_floor', 1e-2),
        data_cfg=dict(train_ratio=dc.train_ratio, val_ratio=dc.val_ratio, random_seed=random_seed,
                      feature_indices=feature_indices, augment=augment, zero_gauge_features=bool(augment),
                      field=getattr(train_dataset, 'field', 'E'), qoi_ops=qoi_ops),
    )
    if train_dataset.stats:              # z-scored frequency targets ↔ GHz
        model.freq_stats = train_dataset.stats

    if rank0:
        n_deg, n_deg_modes, n_tot = count_near_degenerate(train_dataset, getattr(tc, 'near_deg_threshold', 0.05),
                                                          getattr(tc, 'near_deg_rel_threshold', None))
        print(f"[degeneracy] train: {n_deg}/{n_tot} geometries with a near-degenerate cluster ({n_deg_modes} modes)")

    # metric names contain '/': the checkpoint file name is given explicitly
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{tc.log_dir}/{tc.exp_name}",
        filename="best-epoch={epoch:02d}-val_rel_l2={val/field_rel_l2:.4f}",
        auto_insert_metric_name=False, save_top_k=1, save_last=True,
        monitor="val/field_rel_l2", mode="min")
    progress_file = ProgressFile(f"{tc.log_dir}/{tc.exp_name}/progress.json")   # read by the web UI
    try:
        logger = TensorBoardLogger(save_dir=tc.log_dir, name=tc.exp_name)
    except ModuleNotFoundError:
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger(save_dir=tc.log_dir, name=tc.exp_name)
    logger.log_hyperparams(config_to_flat_dict(cfg))

    n_gpus = torch.cuda.device_count()
    strategy = getattr(tc, 'strategy', 'auto')
    if strategy == "ddp" and n_gpus > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer(
        max_epochs=tc.max_epochs, check_val_every_n_epoch=getattr(tc, 'check_val_every_n_epoch', 1),
        accelerator="auto", devices=n_gpus if n_gpus > 0 else "auto", strategy=strategy,
        gradient_clip_val=tc.gradient_clip_val,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step'),
                   EarlyStopping(monitor="val/field_rel_l2", patience=tc.patience, mode="min"),
                   TQDMProgressBar(refresh_rate=tc.progress_bar_refresh_rate), progress_file],
        logger=logger, log_every_n_steps=tc.log_every_n_steps, fast_dev_run=tc.fast_dev_run)

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    # fast_dev_run writes no checkpoint: test the in-memory model
    trainer.test(model if tc.fast_dev_run else None, dataloaders=test_loader,
                 ckpt_path=None if tc.fast_dev_run else "best")
    if rank0:
        progress_file.finish()


if __name__ == '__main__':
    main()
