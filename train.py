import os
import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, TQDMProgressBar
from src.training.callbacks import FieldVisualizationCallback
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.dataset import GNOTDataset, gnot_collate_fn
from src.training.lightning_module import GNOTLightning
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

    if local_rank == 0:
        print("\nLoading datasets...")
    feature_indices = getattr(dc, 'feature_indices', None)
    max_nodes = getattr(dc, 'max_nodes', None)
    if feature_indices is not None and local_rank == 0:
        names = [GNOTDataset.FEATURE_NAMES[i] for i in feature_indices]
        print(f"Ablation: using features {feature_indices} → {names}")
    if max_nodes is not None and local_rank == 0:
        print(f"Node sub-sampling: max_nodes={max_nodes}")
    random_seed = getattr(dc, 'random_seed', 42)
    train_dataset = GNOTDataset(dc.data_path, split='train',
                                train_ratio=dc.train_ratio, val_ratio=dc.val_ratio,
                                feature_indices=feature_indices, max_nodes=max_nodes,
                                random_seed=random_seed)
    val_dataset   = GNOTDataset(dc.data_path, split='val',
                                train_ratio=dc.train_ratio, val_ratio=dc.val_ratio,
                                feature_indices=feature_indices, max_nodes=max_nodes,
                                random_seed=random_seed)
    test_dataset  = GNOTDataset(dc.data_path, split='test',
                                train_ratio=dc.train_ratio, val_ratio=dc.val_ratio,
                                feature_indices=feature_indices, max_nodes=max_nodes,
                                random_seed=random_seed)

    train_loader = DataLoader(train_dataset, batch_size=tc.batch_size, shuffle=True,
                              collate_fn=gnot_collate_fn, num_workers=tc.num_workers,
                              pin_memory=tc.pin_memory, persistent_workers=tc.num_workers > 0,
                              prefetch_factor=2 if tc.num_workers > 0 else None)
    val_loader   = DataLoader(val_dataset,   batch_size=tc.batch_size,
                              collate_fn=gnot_collate_fn, num_workers=tc.num_workers,
                              pin_memory=tc.pin_memory, persistent_workers=tc.num_workers > 0,
                              prefetch_factor=2 if tc.num_workers > 0 else None)
    test_loader  = DataLoader(test_dataset,  batch_size=tc.batch_size,
                              collate_fn=gnot_collate_fn, num_workers=tc.num_workers,
                              pin_memory=tc.pin_memory, persistent_workers=tc.num_workers > 0,
                              prefetch_factor=2 if tc.num_workers > 0 else None)

    model = GNOTLightning(
        val_dim=mc.val_dim,
        grid_dim=mc.grid_dim,
        hidden_dim=mc.embed_dim,
        n_shared_layers=mc.n_shared_layers,
        n_mode_layers=mc.n_mode_layers,
        n_field_head_layers=getattr(mc, 'n_field_head_layers', 2),
        n_heads=mc.n_heads,
        num_experts=mc.num_experts,
        num_field_modes=mc.num_field_modes,
        lr=tc.learning_rate,
        freq_weight=tc.freq_weight,
        smoothness_weight=getattr(tc, 'smoothness_weight', 0.0),
        mode_loss_weights=tc.mode_loss_weights,
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
        degeneracy_mode=getattr(tc, 'degeneracy_mode', 'soft'),
        near_deg_threshold=getattr(tc, 'near_deg_threshold', 0.05),
        deg_sigma_rel=getattr(tc, 'deg_sigma_rel', 0.5),
        deg_sigma_abs=getattr(tc, 'deg_sigma_abs', 0.3),
        slot_ortho_weight=getattr(tc, 'slot_ortho_weight', 0.1),
        freq_match_weight=getattr(tc, 'freq_match_weight', 0.5),
    )

    # Pass frequency statistics to the model for physical units logging
    if hasattr(train_dataset, 'stats') and train_dataset.stats:
        model.freq_stats = train_dataset.stats

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{tc.log_dir}/{tc.exp_name}",
        filename="best-{epoch:02d}-{val/field_rel_l2:.4f}",
        save_top_k=1,
        monitor="val/field_rel_l2",
        mode="min",
        verbose=False
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop = EarlyStopping(monitor="val/field_rel_l2", patience=tc.patience, mode="min")
    viz_callback = FieldVisualizationCallback(log_every_n_epochs=tc.viz_every_n_epochs)
    progress_bar = TQDMProgressBar(refresh_rate=tc.progress_bar_refresh_rate)
    
    tb_logger = TensorBoardLogger(save_dir=tc.log_dir, name=tc.exp_name)
    
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
        fast_dev_run=tc.fast_dev_run
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
