import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, DeviceStatsMonitor, TQDMProgressBar
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
    return parser.parse_args()

def parse_overrides(override_list):
    """Parse CLI overrides like ['model.embed_dim=128', 'training.batch_size=32']."""
    overrides = {}
    for item in override_list:
        key, val_str = item.split("=", 1)
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

    print(f"Config loaded from: {args.config}")
    print(f"Model: embed_dim={mc.embed_dim}, n_layers={mc.n_layers}, n_heads={mc.n_heads}, "
          f"num_experts={mc.num_experts}, num_field_modes={mc.num_field_modes}")
    print(f"Training: lr={tc.learning_rate}, batch_size={tc.batch_size}, epochs={tc.max_epochs}, "
          f"scheduler={tc.scheduler}")

    print("\nLoading datasets...")
    train_dataset = GNOTDataset(dc.data_path, split='train',
                                train_ratio=dc.train_ratio, val_ratio=dc.val_ratio)
    val_dataset   = GNOTDataset(dc.data_path, split='val',
                                train_ratio=dc.train_ratio, val_ratio=dc.val_ratio)
    test_dataset  = GNOTDataset(dc.data_path, split='test',
                                train_ratio=dc.train_ratio, val_ratio=dc.val_ratio)

    train_loader = DataLoader(train_dataset, batch_size=tc.batch_size, shuffle=True,
                              collate_fn=gnot_collate_fn, num_workers=tc.num_workers,
                              pin_memory=tc.pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=tc.batch_size,
                              collate_fn=gnot_collate_fn, num_workers=tc.num_workers,
                              pin_memory=tc.pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=tc.batch_size,
                              collate_fn=gnot_collate_fn, num_workers=tc.num_workers,
                              pin_memory=tc.pin_memory)

    model = GNOTLightning(
        val_dim=mc.val_dim,
        grid_dim=mc.grid_dim,
        theta_dim=mc.theta_dim,
        hidden_dim=mc.embed_dim,
        n_layers=mc.n_layers,
        n_heads=mc.n_heads,
        num_experts=mc.num_experts,
        num_field_modes=mc.num_field_modes,
        lr=tc.learning_rate,
        freq_weight=tc.freq_weight,
        mode_loss_weights=tc.mode_loss_weights,
        scheduler=tc.scheduler,
        weight_decay=tc.weight_decay,
        use_checkpoint=mc.use_checkpoint,
        # Scheduler-specific params
        onecycle_pct_start=tc.onecycle_pct_start,
        onecycle_div_factor=tc.onecycle_div_factor,
        onecycle_final_div_factor=tc.onecycle_final_div_factor,
        cosine_eta_min=tc.cosine_eta_min,
    )
    
    # Pass frequency statistics to the model for physical units logging
    if hasattr(train_dataset, 'stats') and train_dataset.stats:
        model.freq_stats = train_dataset.stats

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{tc.log_dir}/{tc.exp_name}",
        filename="best-{epoch:02d}-{val/loss:.4f}",
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        verbose=False
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop = EarlyStopping(monitor="val/loss", patience=tc.patience, mode="min")
    viz_callback = FieldVisualizationCallback(log_every_n_epochs=tc.viz_every_n_epochs)
    gpu_stats = DeviceStatsMonitor()
    progress_bar = TQDMProgressBar(refresh_rate=tc.progress_bar_refresh_rate)
    
    tb_logger = TensorBoardLogger(save_dir=tc.log_dir, name=tc.exp_name)
    
    # Config'i TensorBoard'a logla (reproduceability)
    tb_logger.log_hyperparams(config_to_flat_dict(cfg))

    trainer = pl.Trainer(
        max_epochs=tc.max_epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=tc.gradient_clip_val,
        callbacks=[checkpoint_callback, lr_monitor, early_stop, viz_callback, gpu_stats, progress_bar],
        logger=tb_logger,
        log_every_n_steps=tc.log_every_n_steps,
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=tc.fast_dev_run
    )

    if tc.fast_dev_run:
        print("Starting fast_dev_run training to verify pipeline...")
    trainer.fit(model, train_loader, val_loader)
    
    print("Running final evaluation on held-out test split...")
    ckpt_path = "best" if not tc.fast_dev_run else None
    trainer.test(dataloaders=test_loader, ckpt_path=ckpt_path)
    
    if tc.fast_dev_run:
        print("Verification complete! To run full training, do not use --fast_dev_run flag.")

if __name__ == '__main__':
    main()
