import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.dataset import GNOTDataset, gnot_collate_fn
from src.training.lightning_module import GNOTLightning

# Configuration
OUTPUT_PKL = "data/gnot_dataset.pkl"
LOG_DIR = "training_logs"
EXP_NAME = "gnot_training"

BATCH_SIZE = 16
MAX_EPOCHS = 50
HIDDEN_DIM = 256
N_LAYERS = 6
LEARNING_RATE = 1e-3
FREQ_WEIGHT = 0.5
GRID_DIM = 2
VAL_DIM = 6
THETA_DIM = 1

def main():
    print("Loading datasets...")
    # NOTE: Before running training, make sure data/gnot_dataset.pkl exists
    # Running data generation and conversion is required prior to training.
    
    train_dataset = GNOTDataset(OUTPUT_PKL, split='train')
    val_dataset = GNOTDataset(OUTPUT_PKL, split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=gnot_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            collate_fn=gnot_collate_fn, num_workers=0)

    model = GNOTLightning(
        val_dim=VAL_DIM,
        grid_dim=GRID_DIM,
        theta_dim=THETA_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        lr=LEARNING_RATE,
        freq_weight=FREQ_WEIGHT
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{LOG_DIR}/{EXP_NAME}",
        filename="best-{epoch:02d}-{val/loss:.4f}",
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        verbose=False
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    tb_logger = TensorBoardLogger(save_dir=LOG_DIR, name=EXP_NAME)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=True  # Ensure it runs correctly for 1 step
    )

    print("Starting fast_dev_run training to verify pipeline...")
    trainer.fit(model, train_loader, val_loader)
    
    print("Verification complete! To run full training, set fast_dev_run=False inside train.py.")

if __name__ == '__main__':
    main()
