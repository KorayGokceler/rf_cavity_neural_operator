import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, TQDMProgressBar
from src.training.callbacks import FieldVisualizationCallback
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.dataset import GNOTDataset, gnot_collate_fn
from src.training.lightning_module import GNOTLightning, count_near_degenerate
from src.config import load_config, config_to_flat_dict

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNOT Model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file.")
    # Her config değerini CLI'dan override edebilirsin:
    #   python train.py --config configs/default.yaml --override model.embed_dim=128
    parser.add_argument("--override", nargs="*", default=[],
                        help="Override config values: key=value (e.g. model.embed_dim=128)")
    # Eski CLI argümanları hala destekleniyor (config override olarak):
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 1 epoch to verify pipeline.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from.")
    return parser.parse_args()

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

def _spectral_kwargs(mc, feature_indices):
    """config `model.spectral` + SpectralNO feature columns remapped into the
    `feature_indices` subset (SpectralNO reads coords 0-1, dist 2, dir 3-4 and
    area 5 of the full layout; a dropped column → None = not used)."""
    kw = dict(getattr(mc, 'spectral', None) or {})
    if feature_indices is not None:
        pos = {f: i for i, f in enumerate(feature_indices)}
        cols = lambda *fs: [pos[f] for f in fs] if all(f in pos for f in fs) else None
        kw.setdefault('coord_feature_idx', cols(0, 1))
        kw.setdefault('dist_feature_idx', pos.get(2))
        kw.setdefault('dir_feature_idx', cols(3, 4))
        kw.setdefault('area_feature_idx', pos.get(5))
        if kw.get('torsion_feature_idx') is not None:
            kw['torsion_feature_idx'] = pos.get(kw['torsion_feature_idx'])
    return kw or None


_EIGENSPACE_KEYS = ('n_layers', 'torsion_feature_idx', 'mass_ridge', 'stiff_ridge',
                    'eig_broadening')


def _eigenspace_kwargs(mc, feature_indices):
    """EigenspaceOperator kwargs: top-level model keys (n_layers, ...) overridden
    by a `model.eigenspace` block; torsion_feature_idx (full-layout column)
    remapped into the `feature_indices` subset."""
    kw = {k: mc[k] for k in _EIGENSPACE_KEYS if k in mc}
    kw.update(dict(getattr(mc, 'eigenspace', None) or {}))
    if feature_indices is not None and kw.get('torsion_feature_idx') is not None:
        kw['torsion_feature_idx'] = {f: i for i, f in enumerate(feature_indices)}.get(
            kw['torsion_feature_idx'])
    return kw or None


def main():
    args = parse_args()
    
    # Config yükle + CLI override'larını uygula
    overrides = parse_overrides(args.override)
    if args.fast_dev_run:
        overrides["training.fast_dev_run"] = True
    cfg = load_config(args.config, overrides=overrides)
    
    # Kısayollar
    tc = cfg.training
    mc = cfg.model
    dc = cfg.dataset

    # Rank-0 guard: only print config on main process
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:
        print(f"Config loaded from: {args.config}")
        print(f"Model: embed_dim={mc.embed_dim}, heads={mc.n_heads}, "
              f"L_shared={mc.n_shared_layers}, L_mode={mc.n_mode_layers}")
        print(f"Experts={mc.num_experts}, num_field_modes={mc.num_field_modes}")
        print(f"Training: lr={tc.learning_rate}, batch_size={tc.batch_size}, epochs={tc.max_epochs}, "
              f"scheduler={tc.scheduler}")
        n_gpus = torch.cuda.device_count()
        strategy = getattr(tc, 'strategy', 'auto')
        print(f"GPUs: {n_gpus}, Strategy: {strategy}")

    model_type = getattr(cfg, 'model_type', 'gnot')  # 'gnot' | 'spectral_no' | 'eigenspace'
    # fp32 matmul precision.  'medium' (previous global default) lets PyTorch
    # run fp32 matmuls in bf16 — on CPU (oneDNN) that is ~0.25 abs error on a
    # 512x512 product, and it corrupts SpectralNO's Galerkin M/L assembly +
    # Cholesky/eigh.  'high' = TF32 on Ampere GPUs, exact fp32 on CPU;
    # SpectralNO uses 'highest'.  Override: training.matmul_precision.
    matmul_precision = getattr(tc, 'matmul_precision', None) or (
        'highest' if model_type == 'spectral_no' else 'high')
    torch.set_float32_matmul_precision(matmul_precision)
    if local_rank == 0:
        print(f"Model type: {model_type} (float32 matmul precision: {matmul_precision})")
        print("\nLoading datasets...")
    feature_indices = getattr(dc, 'feature_indices', None)
    max_nodes = getattr(dc, 'max_nodes', None)
    augment = getattr(tc, 'augment', False)
    if feature_indices is not None and local_rank == 0:
        names = [GNOTDataset.FEATURE_NAMES[i] if i < len(GNOTDataset.FEATURE_NAMES) else f'feat_{i}'
                 for i in feature_indices]
        print(f"Ablation: using features {feature_indices} → {names}")
    if max_nodes is not None and local_rank == 0:
        print(f"Node sub-sampling: max_nodes={max_nodes}")
    if augment and local_rank == 0:
        print("Augmentation: rotation + reflection enabled for train split")
    random_seed = getattr(dc, 'random_seed', 42)
    # Model init / DataLoader shuffle / augmentation için global seed
    # (dataset split'i ayrıca aynı seed ile deterministik).
    pl.seed_everything(random_seed, workers=True)
    train_dataset = GNOTDataset(dc.data_path, split='train',
                                train_ratio=dc.train_ratio, val_ratio=dc.val_ratio,
                                feature_indices=feature_indices, max_nodes=max_nodes,
                                random_seed=random_seed, augment=augment)
    val_dataset   = GNOTDataset(dc.data_path, split='val',
                                train_ratio=dc.train_ratio, val_ratio=dc.val_ratio,
                                feature_indices=feature_indices, max_nodes=max_nodes,
                                random_seed=random_seed, augment=False,
                                zero_gauge_features=augment)
    test_dataset  = GNOTDataset(dc.data_path, split='test',
                                train_ratio=dc.train_ratio, val_ratio=dc.val_ratio,
                                feature_indices=feature_indices, max_nodes=max_nodes,
                                random_seed=random_seed, augment=False,
                                zero_gauge_features=augment)

    # Boş split → ModelCheckpoint/EarlyStopping monitor'ü hiç loglanmaz ve
    # eğitim sonunda anlaşılmaz bir hata verir; erken ve açık hata ver.
    for _name, _ds in (('train', train_dataset), ('val', val_dataset), ('test', test_dataset)):
        if len(_ds) == 0:
            raise ValueError(
                f"'{_name}' split is empty ({len(train_dataset.geom_to_samples)} geometries, "
                f"train_ratio={dc.train_ratio}, val_ratio={dc.val_ratio}). "
                f"Use more data or adjust dataset.train_ratio / dataset.val_ratio.")

    # val_dim / num_field_modes veriden doğrulanır (val_dim: null → otomatik).
    data_val_dim, data_n_modes = train_dataset.data_dims()
    if getattr(mc, 'val_dim', None) in (None, 'auto'):
        mc.val_dim = data_val_dim
        if local_rank == 0:
            print(f"val_dim auto-detected from data: {data_val_dim}")
    elif int(mc.val_dim) != data_val_dim:
        fi_note = (f" (after feature_indices={feature_indices})"
                   if feature_indices is not None else "")
        raise ValueError(
            f"model.val_dim={mc.val_dim} but dataset '{dc.data_path}' provides "
            f"{data_val_dim} input features per node{fi_note}. Set "
            f"model.val_dim={data_val_dim} (or null to auto-detect), or re-convert the dataset.")
    # eigenspace: the span loss uses every stored mode (e.g. 6), the model
    # outputs K ≤ that many Ritz modes (compared with the K lowest targets).
    span_loss = getattr(tc, 'span_loss', None)
    if model_type == 'eigenspace' or span_loss:
        if data_n_modes < int(mc.num_field_modes):
            raise ValueError(
                f"model.num_field_modes={mc.num_field_modes} but the dataset provides only "
                f"{data_n_modes} modes per geometry (data_convert.mode_indices); "
                f"eigenspace needs data modes >= K.")
    elif int(mc.num_field_modes) != data_n_modes:
        raise ValueError(
            f"model.num_field_modes={mc.num_field_modes} but the dataset provides "
            f"{data_n_modes} modes per geometry (data_convert.mode_indices). "
            f"Set model.num_field_modes={data_n_modes}.")
    if getattr(tc, 'mode_loss_weights', None) is not None and local_rank == 0:
        print("WARNING: training.mode_loss_weights is set but is NOT used by the "
              "set-prediction (OT / Grassmannian) loss - it has no effect.")

    # pin_memory sadece CUDA varken anlamlı (CPU'da uyarı + gereksiz kopya).
    pin_memory = bool(tc.pin_memory) and torch.cuda.is_available()
    loader_kw = dict(collate_fn=gnot_collate_fn, num_workers=tc.num_workers,
                     pin_memory=pin_memory, persistent_workers=tc.num_workers > 0,
                     prefetch_factor=2 if tc.num_workers > 0 else None)
    train_loader = DataLoader(train_dataset, batch_size=tc.batch_size, shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_dataset,   batch_size=tc.batch_size, **loader_kw)
    test_loader  = DataLoader(test_dataset,  batch_size=tc.batch_size, **loader_kw)

    model = GNOTLightning(
        val_dim=mc.val_dim,
        grid_dim=mc.grid_dim,
        hidden_dim=mc.embed_dim,
        n_shared_layers=mc.n_shared_layers,
        n_mode_layers=mc.n_mode_layers,
        n_field_head_layers=getattr(mc, 'n_field_head_layers', 2),
        n_heads=mc.n_heads,
        num_experts=getattr(mc, 'num_experts', 4),
        num_field_modes=mc.num_field_modes,
        lr=tc.learning_rate,
        freq_weight=tc.freq_weight,
        smoothness_weight=getattr(tc, 'smoothness_weight', 0.0),
        mode_loss_weights=getattr(tc, 'mode_loss_weights', None),
        lr_mode_specific=getattr(tc, 'lr_mode_specific', None),
        lr_freq_heads=getattr(tc, 'lr_freq_heads', None),
        scheduler=tc.scheduler,
        reducelr_patience=getattr(tc, 'reducelr_patience', 10),
        reducelr_factor=getattr(tc, 'reducelr_factor', 0.5),
        weight_decay=tc.weight_decay,
        use_checkpoint=mc.use_checkpoint,
        predict_frequency=getattr(mc, 'predict_frequency', True),
        gradient_clip_val=tc.gradient_clip_val,
        # Scheduler-specific params (using getattr for flexibility with different configs)
        onecycle_pct_start=getattr(tc, 'onecycle_pct_start', 0.3),
        onecycle_div_factor=getattr(tc, 'onecycle_div_factor', 25.0),
        onecycle_final_div_factor=getattr(tc, 'onecycle_final_div_factor', 1e4),
        cosine_eta_min=getattr(tc, 'cosine_eta_min', 1.0e-6),
        dropout=getattr(mc, 'dropout', 0.0),
        rff_dim=getattr(mc, 'rff_dim', 64),
        rff_length_scale=getattr(mc, 'rff_length_scale', 0.1),
        n_basis=getattr(mc, 'n_basis', 16),
        degeneracy_mode=getattr(tc, 'degeneracy_mode', 'soft'),
        near_deg_threshold=getattr(tc, 'near_deg_threshold', 0.05),
        near_deg_rel_threshold=getattr(tc, 'near_deg_rel_threshold', None),
        deg_sigma_rel=getattr(tc, 'deg_sigma_rel', 0.5),
        deg_sigma_abs=getattr(tc, 'deg_sigma_abs', 0.3),
        slot_ortho_weight=getattr(tc, 'slot_ortho_weight', 0.1),
        freq_match_weight=getattr(tc, 'freq_match_weight', 0.5),
        # SpectralNO / new params
        model_type=model_type,
        bc_scale=getattr(mc, 'bc_scale', 0.02),
        rayleigh_weight=getattr(tc, 'rayleigh_weight', 0.1),
        orthonormalize_output=getattr(mc, 'orthonormalize_output', False),
        scale_invariant_field=getattr(tc, 'scale_invariant_field', None),
        area_weighted_field=getattr(tc, 'area_weighted_field', False),
        physics_freq=getattr(mc, 'physics_freq', False),
        ritz_basis=getattr(mc, 'ritz_basis', 0),
        # Extra SpectralNO kwargs (mass_ridge, area_feature_idx, ...) from model.spectral
        spectral_kwargs=_spectral_kwargs(mc, feature_indices),
        # EigenspaceOperator kwargs (model.eigenspace) + NEO loss weights
        eigenspace_kwargs=(_eigenspace_kwargs(mc, feature_indices)
                           if model_type == 'eigenspace' else None),
        span_weight=getattr(tc, 'span_weight', 1.0),
        selfsup_weight=getattr(tc, 'selfsup_weight', 0.01),
        ortho_weight=getattr(tc, 'ortho_weight', 0.01),
        ritz_field_weight=getattr(tc, 'ritz_field_weight', 0.0),
        span_loss=span_loss,
        span_norm=getattr(tc, 'span_norm', 'both'),
        span_root=getattr(tc, 'span_root', True),
        span_ridge=getattr(tc, 'span_ridge', 1e-9),
        selfsup_form=getattr(tc, 'selfsup_form', 'compliance'),
        # Stored in the checkpoint hparams so infer.py rebuilds the same split
        # and the same input features (feature_indices, gauge zeroing).
        data_cfg=dict(train_ratio=dc.train_ratio, val_ratio=dc.val_ratio,
                      random_seed=random_seed, feature_indices=feature_indices,
                      max_nodes=max_nodes, augment=augment,
                      zero_gauge_features=bool(augment)),
    )

    # Pass frequency statistics to the model for physical units logging
    if hasattr(train_dataset, 'stats') and train_dataset.stats:
        model.freq_stats = train_dataset.stats

    # One-time diagnostic: how many train geometries are near-degenerate
    # (same detect_clusters + threshold the loss/metric uses).
    if local_rank == 0:
        deg_thr = getattr(tc, 'near_deg_threshold', 0.05)
        rel_thr = getattr(tc, 'near_deg_rel_threshold', None)
        n_deg, n_deg_modes, n_tot = count_near_degenerate(train_dataset, deg_thr, rel_thr)
        rule = f"relative gap < {rel_thr:.1%}" if rel_thr is not None else f"|Δz| < {deg_thr:.4f}"
        print(f"[degeneracy] train: {n_deg}/{n_tot} geometries near-degenerate "
              f"({n_deg_modes} modes), {rule}")

    # NOTE: metrik adındaki '/' otomatik isim eklemede alt klasör açıyordu
    # ("best-epoch=00-val/field_rel_l2=0.1234.ckpt"); isim elle verilir.
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{tc.log_dir}/{tc.exp_name}",
        filename="best-epoch={epoch:02d}-val_rel_l2={val/field_rel_l2:.4f}",
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=True,               # last.ckpt → --resume / infer fallback
        monitor="val/field_rel_l2",
        mode="min",
        verbose=False
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop = EarlyStopping(monitor="val/field_rel_l2", patience=tc.patience, mode="min")
    viz_callback = FieldVisualizationCallback(log_every_n_epochs=tc.viz_every_n_epochs)
    progress_bar = TQDMProgressBar(refresh_rate=tc.progress_bar_refresh_rate)
    
    try:
        tb_logger = TensorBoardLogger(save_dir=tc.log_dir, name=tc.exp_name)
    except ModuleNotFoundError:
        # tensorboard kurulu değilse eğitimi durdurma — CSV'ye logla
        from pytorch_lightning.loggers import CSVLogger
        if local_rank == 0:
            print("WARNING: tensorboard not installed -> falling back to CSVLogger "
                  "(pip install tensorboard for TB logging/figures).")
        tb_logger = CSVLogger(save_dir=tc.log_dir, name=tc.exp_name)
    
    # Config'i TensorBoard'a logla (reproduceability)
    tb_logger.log_hyperparams(config_to_flat_dict(cfg))

    # Multi-GPU support: strategy & devices from config
    n_gpus = torch.cuda.device_count()
    devices = n_gpus if n_gpus > 0 else "auto"
    strategy_name = getattr(tc, 'strategy', 'auto')
    
    # Optimize DDP: skip unused parameter detection (all params are used in GNOT)
    if strategy_name == "ddp" and n_gpus > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = strategy_name
    
    trainer = pl.Trainer(
        max_epochs=tc.max_epochs,
        check_val_every_n_epoch=getattr(tc, 'check_val_every_n_epoch', 5),
        accelerator="auto",
        devices=devices,
        strategy=strategy,
        gradient_clip_val=tc.gradient_clip_val,
        callbacks=[checkpoint_callback, lr_monitor, early_stop, viz_callback, progress_bar],
        logger=tb_logger,
        log_every_n_steps=tc.log_every_n_steps,
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=tc.fast_dev_run,
        # SpectralNO computes ∇ψ with autograd, so val/test need grad mode too:
        # the default inference_mode=True crashes trainer.test() with
        # "element 0 of tensors does not require grad".
        inference_mode=(model_type != 'spectral_no'),
    )

    if tc.fast_dev_run and local_rank == 0:
        print("Starting fast_dev_run training to verify pipeline...")
        
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)

    if tc.fast_dev_run:
        # fast_dev_run disables checkpointing, so "best" is unavailable; run a
        # quick sanity test on the in-memory model instead of loading a ckpt.
        if local_rank == 0:
            print("Running fast_dev_run sanity test on in-memory model...")
        trainer.test(model, dataloaders=test_loader)
        if local_rank == 0:
            print("Verification complete! To run full training, do not use --fast_dev_run flag.")
    else:
        if local_rank == 0:
            print("Running final evaluation on held-out test split...")
        trainer.test(dataloaders=test_loader, ckpt_path="best")

if __name__ == '__main__':
    main()
