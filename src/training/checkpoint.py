"""Locate the checkpoint of a training run."""
import glob
import os
import re


def resolve_checkpoint(path):
    """A .ckpt file, or a training dir (training_logs/<exp_name>): the best checkpoint there (lowest
    val_rel_l2 in the file name, as written by train.py's ModelCheckpoint), else last.ckpt."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpts = glob.glob(os.path.join(path, '**', '*.ckpt'), recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files under {path}")

    def score(p):
        m = re.search(r'val_rel_l2=([0-9.]+?)(?:-v\d+)?\.ckpt$', os.path.basename(p))
        return float(m.group(1)) if m else float('inf')
    return min(ckpts, key=lambda p: (score(p), os.path.basename(p) != 'last.ckpt'))
