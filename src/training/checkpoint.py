"""Locate the checkpoint of a training run."""
import glob
import os
import re

# the score in the file name of train.py's ModelCheckpoint: val_rel_l2 (monitor val/field_rel_l2), or, for
# projection-free training (docs/38), val_span_loss (monitor val/span_loss; val_loss = val/loss was the monitor of
# an earlier version and is still read); all are minimised
_SCORE = re.compile(r'(val_rel_l2|val_span_loss|val_loss)=([0-9.]+?)(?:-v\d+)?\.ckpt$')


def resolve_checkpoint(path):
    """A .ckpt file, or a training dir (training_logs/<exp_name>): the best checkpoint there (lowest
    val_rel_l2, val_span_loss or val_loss in the file name, as written by train.py's ModelCheckpoint), else last.ckpt.
    A directory with checkpoints of both kinds (a run resumed with another training.projection) is judged
    within the kind of the most recently written best checkpoint: the two scores are not comparable."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpts = glob.glob(os.path.join(path, '**', '*.ckpt'), recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files under {path}")

    def parse(p):
        m = _SCORE.search(os.path.basename(p))
        return (m.group(1), float(m.group(2))) if m else (None, float('inf'))
    kinds = {parse(p)[0] for p in ckpts} - {None}
    if len(kinds) > 1:
        newest = max((p for p in ckpts if parse(p)[0]), key=os.path.getmtime)
        ckpts = [p for p in ckpts if parse(p)[0] in (parse(newest)[0], None)]
    return min(ckpts, key=lambda p: (parse(p)[1], os.path.basename(p) != 'last.ckpt'))
