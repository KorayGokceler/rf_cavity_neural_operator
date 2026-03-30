import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, DeviceStatsMonitor, TQDMProgressBar
from src.training.callbacks import FieldVisualizationCallback
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.dataset import GNOTDataset, gnot_collate_fn
from src.training.lightning_module import GNOTLightning

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNOT Model.")
    parser.add_argument("--output_pkl", type=str, default="data/gnot_dataset.pkl", help="Path to input PKL dataset.")
    parser.add_argument("--log_dir", type=str, default="training_logs", help="Directory for logs and checkpoints.")
    parser.add_argument("--exp_name", type=str, default="gnot_training", help="Name of the experiment.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs to train.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size of the model.")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of GNOT layers.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--freq_weight", type=float, default=0.5, help="Weight for frequency loss component.")
    parser.add_argument("--scheduler", type=str, default="onecycle", choices=["onecycle", "cosine"], help="LR scheduler type.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for EarlyStopping.")
    parser.add_argument("--viz_every_n_epochs", type=int, default=5, help="Visualize mode shapes every N epochs.")
    parser.add_argument("--use_checkpoint", action="store_true", help="Use gradient checkpointing to save GPU VRAM.")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 1 epoch to verify pipeline.")
    
    # Model architecture constants (usually not changed frequently)
    parser.add_argument("--grid_dim", type=int, default=2, help="Grid dimension size.")
    parser.add_argument("--val_dim", type=int, default=6, help="Value dimension size.")
    parser.add_argument("--theta_dim", type=int, default=1, help="Theta dimension size.")
    
    return parser.parse_args()

def main(args):
    print("Loading datasets...")
    # NOTE: Before running training, make sure data/gnot_dataset.pkl exists
    # Running data generation and conversion is required prior to training.
    
    train_dataset = GNOTDataset(args.output_pkl, split='train')
    val_dataset = GNOTDataset(args.output_pkl, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=gnot_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            collate_fn=gnot_collate_fn, num_workers=0)

    model = GNOTLightning(
        val_dim=args.val_dim,
        grid_dim=args.grid_dim,
        theta_dim=args.theta_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        lr=args.learning_rate,
        freq_weight=args.freq_weight,
        scheduler=args.scheduler,
        weight_decay=args.weight_decay,
        use_checkpoint=args.use_checkpoint
    )
    
    # Pass frequency statistics to the model for physical units logging
    if hasattr(train_dataset, 'stats') and train_dataset.stats:
        model.freq_stats = train_dataset.stats

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.log_dir}/{args.exp_name}",
        filename="best-{epoch:02d}-{val/loss:.4f}",
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        verbose=False
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop = EarlyStopping(monitor="val/loss", patience=args.patience, mode="min")
    viz_callback = FieldVisualizationCallback(log_every_n_epochs=args.viz_every_n_epochs)
    gpu_stats = DeviceStatsMonitor()
    progress_bar = TQDMProgressBar(refresh_rate=50) # Reduces flickering in Colab
    
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name=args.exp_name)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, lr_monitor, early_stop, viz_callback, gpu_stats, progress_bar],
        logger=tb_logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=args.fast_dev_run
    )

    if args.fast_dev_run:
        print("Starting fast_dev_run training to verify pipeline...")
    trainer.fit(model, train_loader, val_loader)
    
    print("Running final evaluation to measure model success...")
    # Verify the model using test step metrics over validation dataset
    ckpt_path = "best" if not args.fast_dev_run else None
    trainer.test(dataloaders=val_loader, ckpt_path=ckpt_path)
    
    if args.fast_dev_run:
        print("Verification complete! To run full training, do not use --fast_dev_run flag.")

if __name__ == '__main__':
    args = parse_args()
    main(args)
