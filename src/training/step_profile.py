"""Step-time breakdown of the training loop (training.profile_steps = N, docs/38 §11).

`StepProfiler` (a Lightning callback) and `StepClock` (stamped into the CavityLightning module) time, on the main
process, for the first N training batches: loader wait, host→device transfer, trunk (forward_basis), Grams, loss terms,
logging, backward, gradient clipping + optimizer step, and the rest (Lightning bookkeeping, callbacks, progress bar).  On CUDA
every stamp is preceded by torch.cuda.synchronize(), so each stage gets the GPU time it caused; that serialises the CPU
and the GPU: the sum is an upper bound of the normal step time and the profile is for ranking the stages, not for
predicting the throughput.  After N batches the clock is removed and training goes on at full speed; the table is printed
and written to `<log_dir>/<exp>/step_profile.json`.
"""
import json
import os
import statistics
import time
from collections import defaultdict

import pytorch_lightning as pl
import torch

STAGES = ('loader_wait', 'transfer', 'trunk', 'grams', 'loss', 'log', 'step_other', 'backward', 'accum_other',
          'optimizer', 'post')


class StepClock:
    def __init__(self, cuda_sync=None):
        self.cuda_sync = torch.cuda.is_available() if cuda_sync is None else bool(cuda_sync)
        self.acc = defaultdict(float)
        self._t = None
        self._t_transfer = None

    def now(self):
        if self.cuda_sync:
            torch.cuda.synchronize()
        return time.perf_counter()

    def reset(self):
        self._t = self.now()

    def mark(self, stage):
        t = self.now()
        if self._t is not None:
            self.acc[stage] += t - self._t
        self._t = t

    def begin_transfer(self):
        self._t_transfer = self.now()

    def end_transfer(self):
        self.acc['transfer'] += self.now() - self._t_transfer


class SyncDebug(pl.Callback):
    """training.sync_debug_steps = N: `torch.cuda.set_sync_debug_mode("warn")` for N training batches after the first three
    (warm-up), so that every operation that synchronises the host with the GPU (an `.item()`, an implicit copy, a `nonzero` ...)
    emits a warning with its Python stack.  CUDA only; nothing happens on a CPU."""

    def __init__(self, n_batches, warmup=3):
        self.n_batches, self.warmup = int(n_batches), int(warmup)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if torch.cuda.is_available() and batch_idx == self.warmup:
            torch.cuda.set_sync_debug_mode("warn")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available() and batch_idx == self.warmup + self.n_batches - 1:
            torch.cuda.set_sync_debug_mode("default")

    def on_train_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.set_sync_debug_mode("default")


class StepProfiler(pl.Callback):
    def __init__(self, n_batches, warmup=2, path=None, stop=False):
        self.n_batches, self.warmup, self.path, self.stop = int(n_batches), int(warmup), path, bool(stop)
        self.rows, self.clock, self._t_end, self._t_start = [], None, None, None

    def on_train_start(self, trainer, pl_module):
        self.clock = pl_module._clock = StepClock()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.clock is None:
            return
        now = self.clock.now()
        fetch = 0.0 if self._t_end is None else now - self._t_end        # previous batch end → here: loader + transfer
        self._fetch = fetch
        self._t_start = now
        self.clock._t = now

    def on_before_backward(self, trainer, pl_module, loss):
        if self.clock is not None:
            self.clock.mark('step_other')

    def on_after_backward(self, trainer, pl_module):
        if self.clock is not None:
            self.clock.mark('backward')

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self.clock is not None:
            self.clock.mark('accum_other')

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        if self.clock is not None:
            self.clock.mark('optimizer')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.clock is None:
            return
        self.clock.mark('post')
        now = self.clock.now()
        acc = dict(self.clock.acc)
        self.clock.acc.clear()
        self._t_end = now
        row = {k: acc.get(k, 0.0) for k in STAGES}
        row['loader_wait'] = max(self._fetch - row['transfer'], 0.0)
        row['total'] = self._fetch + (now - self._t_start)
        self.rows.append(row)
        if len(self.rows) >= self.n_batches:
            self._finish(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if self.clock is not None and self.rows:
            self._finish(trainer, pl_module)

    def _finish(self, trainer, pl_module):
        pl_module._clock = self.clock = None
        if self.stop:
            trainer.should_stop = True
        if trainer.global_rank:
            return
        rows = self.rows[self.warmup:] or self.rows
        stats = {k: {'mean': statistics.fmean(r[k] for r in rows), 'median': statistics.median(r[k] for r in rows),
                     'max': max(r[k] for r in rows)} for k in (*STAGES, 'total')}
        info = {'n_batches': len(rows), 'warmup_skipped': len(self.rows) - len(rows), 'stages': stats,
                'cuda_sync': torch.cuda.is_available(), 'gpu_prefetch': bool(pl_module.gpu_prefetch),
                'torch_threads': torch.get_num_threads(),
                'peak_gpu_gb': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None}
        tot = stats['total']['mean'] or 1.0
        print(f"\nStep-time breakdown ({len(rows)} batches after {info['warmup_skipped']} warm-up, main process, "
              f"{'CUDA-synchronised stamps' if info['cuda_sync'] else 'CPU, no device'}; gpu_prefetch={info['gpu_prefetch']})")
        for k in (*STAGES, 'total'):
            s = stats[k]
            print(f"  {k:12s} mean {s['mean'] * 1e3:9.1f} ms  median {s['median'] * 1e3:9.1f}  max {s['max'] * 1e3:9.1f}"
                  f"  {100 * s['mean'] / tot:5.1f} %")
        if self.path:
            os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
            with open(self.path, 'w') as f:
                json.dump(info, f, indent=1)
