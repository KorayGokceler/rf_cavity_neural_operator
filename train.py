"""Train the 3D cavity eigenmode model (E field, EigenspaceOperatorField: learned field in HCurl(p) on
curved meshes; field-label H5 files of dataset_generator_3d, docs/28–29).

    python train.py --config configs/field_3d.yaml --override "dataset.data_path=data/field/*.h5"
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
from src.data.field_dataset import (OPERATORS, FieldDataset, field_collate, field_collate_mf, field_collate_parts,
                                    loader_worker_init)
from src.training.lightning_module import CavityLightning, count_near_degenerate, monitor_metric
from src.training.progress import ProgressFile
from src.training.step_profile import StepProfiler, SyncDebug


def parse_args():
    p = argparse.ArgumentParser(description="Train the 3D cavity eigenmode model.")
    p.add_argument("--config", default="configs/field_3d.yaml", help="YAML config.")
    p.add_argument("--override", nargs="*", default=[],
                   help="config overrides key=value (e.g. model.embed_dim=192)")
    p.add_argument("--fast_dev_run", action="store_true", help="one batch of train / val / test")
    p.add_argument("--resume", default=None, help="checkpoint to continue from (last.ckpt)")
    p.add_argument("--init_from", default=None,
                   help="checkpoint whose WEIGHTS start this run (fresh optimizer / schedule / epoch counter): "
                        "fine-tune an LF-trained model on HF data (docs/34); --resume continues a run instead")
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

_EIGENSPACE_KEYS = ('n_layers', 'edge_rff', 'mass_ridge', 'drop_tol', 'eig_broadening', 'grad_checkpoint',
                    'kp_tol', 'kp_maxiter')


def _eigenspace_kwargs(mc):
    """EigenspaceOperatorField kwargs: top-level model keys overridden by `model.eigenspace`."""
    kw = {k: mc[k] for k in _EIGENSPACE_KEYS if k in mc}
    kw.update(dict(getattr(mc, 'eigenspace', None) or {}))
    return kw or None


def _datasets_field(dc, random_seed, feature_indices, projection=True):
    """train / val / test FieldDataset (field-label H5 files: a path, glob, directory or list).
    projection=False (training.projection, docs/38): the TRAIN and VAL splits do not build the kernel potentials
    G, Kp (their loss never projects); the TEST split always does (the test step projects: Ritz frequencies,
    fields).  dataset.operators (default "assembled"): "matfree" (docs/40 §10) gives the TRAIN and VAL splits the matrix-free
    items (compact element records, no global matrix) and needs training.projection: false (ValueError otherwise); the test
    split is always assembled."""
    operators = str(getattr(dc, 'operators', None) or 'assembled')
    if operators not in OPERATORS:
        raise ValueError(f"dataset.operators must be one of {OPERATORS}, got {operators!r}")
    if operators == 'matfree' and projection:
        raise ValueError("dataset.operators: matfree needs training.projection: false: the matrix-free operators serve the "
                         "projection-free train / validation steps; the projected step needs the assembled G, Kp, M, K "
                         "(docs/40 §10)")
    kw = dict(train_ratio=dc.train_ratio, val_ratio=dc.val_ratio, random_seed=random_seed,
              feature_indices=feature_indices, cache_dir=getattr(dc, 'cache_dir', None),
              n_modes=getattr(dc, 'n_modes', None), hf_repeat=getattr(dc, 'hf_repeat', 1),
              train_extra=getattr(dc, 'train_extra', None),
              require_label_ok=bool(getattr(dc, 'require_label_ok', False)),
              max_tets=getattr(dc, 'max_tets', None), cache_compress=getattr(dc, 'cache_compress', True))
    return tuple(FieldDataset(dc.data_path, split=s, build_potential=projection or s == 'test',
                              operators=operators if s != 'test' else 'assembled',
                              **({'verify_labels': dc.verify_labels} if getattr(dc, 'verify_labels', None) is not None else {}),
                              **kw)
                 for s in ('train', 'val', 'test'))


def loader_kwargs(tc, workers, persistent, gpu, matfree=False):
    """DataLoader keyword arguments of a loader with `workers` worker processes (training.prefetch_factor, default 2).
    gpu (training.gpu_prefetch): the collate returns the sparse operators as dense parts (SparseParts) and
    pin_memory=True pins the batches; otherwise the batches are plain and pin_memory is False (sparse tensors cannot
    be pinned).  matfree (the dataset has operators: matfree, docs/40 §10): the batches hold dense tensors only
    (field_collate_mf, no SparseParts needed), so pin_memory is True whenever a CUDA device exists, with or without
    gpu_prefetch (which then only decides how the batch is copied: side stream or the default transfer)."""
    return dict(collate_fn=field_collate_mf if matfree else field_collate_parts if gpu else field_collate,
                num_workers=workers, pin_memory=bool(gpu or (matfree and torch.cuda.is_available())),
                persistent_workers=persistent and workers > 0,
                prefetch_factor=int(getattr(tc, 'prefetch_factor', 2) or 2) if workers > 0 else None,
                worker_init_fn=loader_worker_init if workers > 0 else None)


def gpu_prefetch_enabled(tc, announce=True):
    """training.gpu_prefetch, which needs a CUDA device (otherwise a notice and the usual transfer)."""
    on = bool(getattr(tc, 'gpu_prefetch', False))
    if on and not torch.cuda.is_available():
        if announce:
            print("training.gpu_prefetch: no CUDA device, the batches are transferred as usual")
        return False
    return on


def make_loaders(tc, train_dataset, val_dataset, test_dataset):
    """(train, val, test) DataLoaders from the training config (docs/38 §11).

    training.num_workers; prefetch_factor (default 2: batches held ready per worker, so each loader keeps up to
    num_workers × prefetch_factor collated batches in /dev/shm); val_num_workers (null = num_workers);
    val_persistent_workers (default true: the validation workers stay alive between epochs together with their
    prefetched batches; false: they are started for each validation and freed after it).  The train workers are
    persistent.  gpu_prefetch: see loader_kwargs."""
    nw = int(tc.num_workers)
    val_nw = getattr(tc, 'val_num_workers', None)
    val_nw = nw if val_nw is None else int(val_nw)
    gpu = gpu_prefetch_enabled(tc)
    mf = lambda ds: getattr(ds, 'operators', 'assembled') == 'matfree'      # noqa: E731
    return (DataLoader(train_dataset, batch_size=tc.batch_size, shuffle=True,
                       **loader_kwargs(tc, nw, True, gpu, mf(train_dataset))),
            DataLoader(val_dataset, batch_size=tc.batch_size,
                       **loader_kwargs(tc, val_nw, bool(getattr(tc, 'val_persistent_workers', True)), gpu, mf(val_dataset))),
            DataLoader(test_dataset, batch_size=tc.batch_size, **loader_kwargs(tc, nw, True, gpu, mf(test_dataset))))


def _checkpoint_filename(monitor):
    """ModelCheckpoint file name of the best epoch.  The metric names contain '/': the name is given explicitly;
    the score tag (val_rel_l2 | val_span_loss) tells src.training.checkpoint.resolve_checkpoint which metric the
    number is."""
    tag = {'val/field_rel_l2': 'val_rel_l2', 'val/span_loss': 'val_span_loss'}[monitor]
    return f"best-epoch={{epoch:02d}}-{tag}={{{monitor}:.4f}}"


def _selection_callbacks(tc, projection):
    """(ModelCheckpoint, EarlyStopping): model selection by val/field_rel_l2 (Ritz fields), or by val/span_loss when
    training and validation are projection-free (training.projection: false: the validation step has no Ritz
    fields, docs/38).  The best checkpoint is the one trainer.test(ckpt_path="best") loads."""
    monitor = monitor_metric(projection)
    return (ModelCheckpoint(dirpath=f"{tc.log_dir}/{tc.exp_name}", filename=_checkpoint_filename(monitor),
                            auto_insert_metric_name=False, save_top_k=1, save_last=True, monitor=monitor, mode="min"),
            EarlyStopping(monitor=monitor, patience=tc.patience, mode="min"))


def _check_projection_matches(ckpt_path, projection):
    """--resume continues a run with the loss it was started with: training.projection must equal the checkpoint's
    (a missing key = a checkpoint from before the option = projection true), else ValueError.  ModelCheckpoint and
    EarlyStopping restore the best score of the monitored metric from the checkpoint, and the two modes monitor
    different metrics (val/field_rel_l2 vs val/span_loss): a restored best score of the other metric would block or
    force every later "improvement" (docs/38 §4).  Start a new run from the weights with --init_from instead."""
    hp = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("hyper_parameters") or {}
    was = bool(hp.get("projection", True))
    if was != projection:
        raise ValueError(f"--resume {ckpt_path} was trained with projection={was}, but training.projection={projection}: "
                         "the monitored metric (and its restored best score) would differ. Resume with the same "
                         "training.projection, or start a new run from these weights with --init_from (docs/38 §4).")


def main():
    args = parse_args()
    overrides = parse_overrides(args.override)
    if args.fast_dev_run:
        overrides["training.fast_dev_run"] = True
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
    random_seed = getattr(dc, 'random_seed', 42)
    pl.seed_everything(random_seed, workers=True)       # model init / shuffle
    model_type = str(getattr(mc, 'type', None) or 'field3d')
    if model_type != 'field3d':
        raise ValueError(f"model.type {model_type!r}: only field3d is supported (the N0 edge model "
                         "eigenspace3d was removed)")
    for key in ('qoi_weight', 'qoi_metrics', 'augment'):
        if getattr(tc, key, None):
            raise ValueError(f"training.{key} belonged to the removed N0 model; set it to 0 / false or drop it "
                             "(the p3 QoI labels are in the data)")
    projection = getattr(tc, 'projection', True)
    if not isinstance(projection, bool):
        raise ValueError(f"training.projection must be true or false, not {projection!r}")
    if args.resume:                  # on every rank: an error on one rank only would hang the others
        _check_projection_matches(args.resume, projection)
    train_dataset, val_dataset, test_dataset = _datasets_field(dc, random_seed, feature_indices, projection)
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

    train_loader, val_loader, test_loader = make_loaders(tc, train_dataset, val_dataset, test_dataset)

    model = CavityLightning(
        val_dim=mc.val_dim, num_field_modes=mc.num_field_modes, hidden_dim=mc.embed_dim,
        n_heads=mc.n_heads, n_basis=mc.n_basis, rff_dim=getattr(mc, 'rff_dim', 64),
        rff_length_scale=getattr(mc, 'rff_length_scale', 0.1), dropout=getattr(mc, 'dropout', 0.0),
        eigenspace_kwargs=_eigenspace_kwargs(mc),
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
        projection=projection,
        diag_metrics_every_n_steps=getattr(tc, 'diag_metrics_every_n_steps', 1) or 1,
        fidelity_weights=getattr(tc, 'fidelity_weights', None) or (1.0, 1.0),
        fidelity_freq_scale=getattr(tc, 'fidelity_freq_scale', None) or (1.0, 1.0),
        data_cfg=dict(train_ratio=dc.train_ratio, val_ratio=dc.val_ratio, random_seed=random_seed,
                      feature_indices=feature_indices, field=getattr(train_dataset, 'field', 'E'),
                      data_path=dc.data_path),
        model_type=model_type,
    )
    model.gpu_prefetch = gpu_prefetch_enabled(tc, announce=False)
    model.matfree_chunk = int(getattr(tc, 'matfree_chunk', 2048) or 2048)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location="cpu", weights_only=False)["state_dict"])
    if train_dataset.stats:              # z-scored frequency targets ↔ GHz
        model.freq_stats = train_dataset.stats

    if rank0:
        n_deg, n_deg_modes, n_tot = count_near_degenerate(train_dataset, getattr(tc, 'near_deg_threshold', 0.05),
                                                          getattr(tc, 'near_deg_rel_threshold', None))
        print(f"[degeneracy] train: {n_deg}/{n_tot} geometries with a near-degenerate cluster ({n_deg_modes} modes)")

    checkpoint_callback, early_stopping = _selection_callbacks(tc, projection)
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
    profile_only = bool(getattr(tc, 'profile_only', False))
    callbacks = [checkpoint_callback, LearningRateMonitor(logging_interval='step'),
                 early_stopping,
                 TQDMProgressBar(refresh_rate=tc.progress_bar_refresh_rate), progress_file]
    if int(getattr(tc, 'profile_steps', 0) or 0) > 0:            # step-time breakdown of the first N batches (docs/38 §11)
        callbacks.append(StepProfiler(tc.profile_steps, path=f"{tc.log_dir}/{tc.exp_name}/step_profile.json",
                                      stop=profile_only))
    if int(getattr(tc, 'sync_debug_steps', 0) or 0) > 0:          # list the remaining host syncs on a GPU (docs/38 §11.2)
        callbacks.append(SyncDebug(tc.sync_debug_steps))
    trainer = pl.Trainer(
        max_epochs=tc.max_epochs, check_val_every_n_epoch=getattr(tc, 'check_val_every_n_epoch', 1),
        accelerator="auto", devices=n_gpus if n_gpus > 0 else "auto", strategy=strategy,
        gradient_clip_val=tc.gradient_clip_val,
        accumulate_grad_batches=int(getattr(tc, 'accumulate_grad_batches', 1) or 1),
        callbacks=callbacks, logger=logger, log_every_n_steps=tc.log_every_n_steps, fast_dev_run=tc.fast_dev_run)

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    if profile_only and int(getattr(tc, 'profile_steps', 0) or 0) > 0:      # the step profile was the point: no best checkpoint, no test
        if rank0:
            progress_file.finish()
        return
    # fast_dev_run writes no checkpoint: test the in-memory model
    trainer.test(model if tc.fast_dev_run else None, dataloaders=test_loader,
                 ckpt_path=None if tc.fast_dev_run else "best")
    if rank0:
        progress_file.finish()


if __name__ == '__main__':
    main()
