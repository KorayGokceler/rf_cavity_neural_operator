"""<run dir>/progress.json — a small, always-current summary of a training run, read by the web UI
(src/service/training.py) and handy on any machine (`cat progress.json`):

    status        running | finished | failed      (a killed process leaves 'running': check the pid)
    epoch / max_epochs, step, batch / n_batches    (batch progress refreshed every few seconds)
    history       one entry per finished epoch: the epoch-level metrics (train/*, val/*, lr)
    best          {'path', 'score', 'monitor'} of the ModelCheckpoint
    test          test metrics after trainer.test

A resumed run keeps the earlier history (entries of re-run epochs are replaced).
"""
import json
import math
import os
import time

import pytorch_lightning as pl


def _scalars(metrics):
    out = {}
    for k, v in metrics.items():
        if k.endswith('_step'):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[k[:-6] if k.endswith('_epoch') else k] = f
    return out


class ProgressFile(pl.Callback):
    def __init__(self, path, every_s=5.0):
        self.path, self.every_s = path, every_s
        self.state = {'status': 'starting', 'history': []}
        try:
            with open(path) as f:
                old = json.load(f)
            self.state['history'] = list(old.get('history') or [])
            self.state['started_first'] = old.get('started_first') or old.get('started')
        except (OSError, ValueError):
            pass
        self._last = 0.0

    def _write(self, **upd):
        self.state.update(upd, updated=time.time(), pid=os.getpid())
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        tmp = f"{self.path}.{os.getpid()}.tmp"
        with open(tmp, 'w') as f:
            json.dump(self.state, f)
        os.replace(tmp, self.path)

    def _best(self, trainer):
        cb = trainer.checkpoint_callback
        if cb is None or not getattr(cb, 'best_model_path', ''):
            return None
        s = cb.best_model_score
        return {'path': cb.best_model_path, 'score': None if s is None else float(s), 'monitor': cb.monitor}

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank:
            return
        now = time.time()
        self._write(status='running', started=now, started_first=self.state.get('started_first') or now,
                    epoch=trainer.current_epoch, max_epochs=trainer.max_epochs,
                    n_params=int(sum(p.numel() for p in pl_module.parameters())))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank or time.time() - self._last < self.every_s:
            return
        self._last = time.time()
        nb = trainer.num_training_batches
        self._write(epoch=trainer.current_epoch, step=trainer.global_step, batch=batch_idx + 1,
                    n_batches=None if nb in (None, float('inf')) else int(nb))

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank:
            return
        e = trainer.current_epoch
        entry = {'epoch': e, 'step': trainer.global_step, 'time': time.time(), **_scalars(trainer.callback_metrics)}
        hist = [h for h in self.state['history'] if h.get('epoch') != e] + [entry]
        self._write(history=hist, epoch=e + 1, step=trainer.global_step, best=self._best(trainer))

    def on_train_end(self, trainer, pl_module):
        if not trainer.global_rank:
            self._write(status='trained', best=self._best(trainer))

    def on_test_end(self, trainer, pl_module):
        if not trainer.global_rank:
            self._write(test=_scalars(trainer.callback_metrics))

    def on_exception(self, trainer, pl_module, exception):
        if not trainer.global_rank:
            self._write(status='failed', error=f"{type(exception).__name__}: {exception}"[:500])

    def finish(self):
        """After the final test (train.py): the run is complete."""
        if self.state.get('status') not in ('failed',):
            self._write(status='finished')
