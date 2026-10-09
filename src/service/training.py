"""Model training from the web UI: one train.py run at a time (one GPU), in <runs_root>/<name>/ — the
same layout as the notebooks' training_logs/<EXP> and TRUBA's runs/<EXP>, so every run is listed:

    <runs_root>/<name>/best-epoch=…-val_rel_l2=….ckpt, last.ckpt   (ModelCheckpoint)
    <runs_root>/<name>/progress.json                                 (src/training/progress.py)
    <runs_root>/<name>/train_params.json                             (what the UI started: for resume)

Training data: one PKL as is, or several datasets merged from their H5 shards (convert_3d.py, lean)
into a local cache file (scratch; rebuilt after a VM restart). Stopping keeps last.ckpt; resume
continues from it (optionally with more epochs).
"""
import collections
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time

from src.service.jobs import run_logged

# (embed_dim, n_heads, eigenspace.n_layers, n_basis) — small … xl as cluster/truba/config.sh model_overrides
PRESETS = {
    'tiny': (32, 2, 2, 16),
    'small': (128, 4, 4, 24),
    'base': (192, 4, 6, 32),
    'large': (256, 8, 8, 48),
    'xl': (384, 8, 12, 64),
}
PRESET_PARAMS_M = {'tiny': 0.03, 'small': 0.85, 'base': 2.8, 'large': 6.5, 'xl': 21.7}
# metrics sent to the UI per epoch (progress.json keeps all of them)
CURVE_KEYS = ('train/loss', 'val/loss', 'train/field_rel_l2', 'val/field_rel_l2', 'val/span_rel_l2',
              'train/freq_rel_err', 'val/freq_rel_err', 'val/freq_mae_ghz', 'lr-AdamW')
NAME_RE = r'^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,63}$'      # no '..', no hidden dirs


def preset_overrides(name):
    e, h, n_layers, m = PRESETS[name]
    return [f"model.embed_dim={e}", f"model.n_heads={h}", f"model.eigenspace.n_layers={n_layers}",
            f"model.n_basis={m}"]


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class TrainRun:
    def __init__(self, name, params, data):
        self.name, self.params, self.data = name, dict(params), data
        self.status, self.stage, self.error = 'queued', 'starting', None
        self.created, self.finished = time.time(), None
        self.log = collections.deque(maxlen=400)
        self.data_cached = False
        self._proc, self._cancel = None, False


class TrainManager:
    def __init__(self, runs_root, scratch=None, python=sys.executable, on_done=None):
        self.runs_root = os.path.abspath(runs_root)
        self.scratch = scratch or os.path.join(tempfile.gettempdir(), 'rfcav_train')
        self.python, self.on_done = python, on_done
        self.active = None
        self._lock = threading.Lock()

    # ── listing ──────────────────────────────────────────────────────
    def run_dir(self, name):
        return os.path.join(self.runs_root, name)

    def _summary(self, name):
        d = self.run_dir(name)
        prog = _read_json(os.path.join(d, 'progress.json')) or {}
        params = _read_json(os.path.join(d, 'train_params.json'))
        ckpts = glob.glob(os.path.join(d, '**', '*.ckpt'), recursive=True)
        if not prog and not ckpts and not params:
            return None
        live = self.active if self.active is not None and self.active.name == name else None
        status = prog.get('status') or ('checkpoint' if ckpts else 'empty')
        if live is not None and live.status in ('queued', 'running'):
            status = 'preparing' if live.stage != 'training' else 'running'
        elif status in ('running', 'starting'):
            status = 'stopped'                                   # no process: interrupted / VM restarted
        elif status == 'trained':
            status = 'finished'                                  # the final test did not run
        if live is not None and live.status in ('failed', 'cancelled') and status not in ('finished',):
            status = 'stopped' if live.status == 'cancelled' else 'failed'
        hist = prog.get('history') or []
        best = prog.get('best') or None
        last = hist[-1] if hist else {}
        return {'name': name, 'status': status, 'epoch': prog.get('epoch', len(hist)),
                'max_epochs': prog.get('max_epochs') or (params or {}).get('epochs'),
                'batch': prog.get('batch'), 'n_batches': prog.get('n_batches'),
                'best_score': best and best.get('score'), 'monitor': best and best.get('monitor'),
                'val_field_rel_l2': last.get('val/field_rel_l2'), 'val_freq_rel_err': last.get('val/freq_rel_err'),
                'n_params': prog.get('n_params'), 'updated': prog.get('updated') or max(
                    [os.path.getmtime(c) for c in ckpts] or [os.path.getmtime(d)]),
                'n_ckpt': len(ckpts), 'has_last': os.path.exists(os.path.join(d, 'last.ckpt')),
                'resumable': params is not None and os.path.exists(os.path.join(d, 'last.ckpt')),
                'preset': (params or {}).get('preset'), 'from_ui': params is not None,
                'error': (live.error if live is not None and live.error else prog.get('error'))}

    def runs(self):
        if not os.path.isdir(self.runs_root):
            return []
        out = [s for s in (self._summary(n) for n in sorted(os.listdir(self.runs_root))
                           if os.path.isdir(self.run_dir(n))) if s]
        return sorted(out, key=lambda r: -(r['updated'] or 0))

    def detail(self, name):
        s = self._summary(name)
        if s is None:
            raise KeyError(name)
        d = self.run_dir(name)
        prog = _read_json(os.path.join(d, 'progress.json')) or {}
        live = self.active if self.active is not None and self.active.name == name else None
        s.update(history=[{k: h[k] for k in ('epoch', 'time', *CURVE_KEYS) if k in h}
                          for h in prog.get('history') or []],
                 test=prog.get('test'), best=prog.get('best'), params=_read_json(os.path.join(d, 'train_params.json')),
                 stage=live.stage if live is not None else None, log=list(live.log)[-60:] if live is not None else [],
                 data_cached=live.data_cached if live is not None else None)
        return s

    # ── control ──────────────────────────────────────────────────────
    def busy(self):
        return self.active is not None and self.active.status in ('queued', 'running')

    def start(self, name, params, data, resume=False):
        """data: {'pkl': path} or {'h5': [files]}; params: preset, epochs, batch_size, lr, n_modes, patience,
        qoi_weight, cache_operators (None = auto), sources (display names)."""
        with self._lock:
            if self.busy():
                raise RuntimeError(f"a training is running ({self.active.name}): stop it first")
            d = self.run_dir(name)
            if resume:
                if not os.path.exists(os.path.join(d, 'last.ckpt')):
                    raise FileNotFoundError(f"{name}: no last.ckpt to resume from")
            elif os.path.exists(d) and os.listdir(d):
                raise FileExistsError(f"run '{name}' exists: resume it or choose another name")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, 'train_params.json'), 'w') as f:
                json.dump({**params, 'data': data}, f, indent=1)
            run = TrainRun(name, params, data)
            run.resume = resume
            self.active = run
        threading.Thread(target=self._worker, args=(run,), daemon=True).start()
        return run

    def resume_params(self, name):
        p = _read_json(os.path.join(self.run_dir(name), 'train_params.json'))
        if p is None:
            raise FileNotFoundError(f"{name}: not started from the UI (no train_params.json)")
        return p

    def stop(self):
        run = self.active
        if run is None or run.status not in ('queued', 'running'):
            return None
        run._cancel = True
        if run._proc is not None and run._proc.poll() is None:
            run._proc.terminate()
        return run

    # ── worker ───────────────────────────────────────────────────────
    def _data_pkl(self, run):
        data = run.data
        if data.get('pkl'):
            return data['pkl'], os.path.getsize(data['pkl'])
        files = sorted(data['h5'])
        h = hashlib.sha1(json.dumps([(f, os.path.getsize(f), int(os.path.getmtime(f))) for f in files])
                         .encode()).hexdigest()[:16]
        os.makedirs(self.scratch, exist_ok=True)
        out = os.path.join(self.scratch, f'train_{h}.pkl')
        size = sum(os.path.getsize(f) for f in files)
        if os.path.exists(out):
            run.log.append(f"training data (cached): {out}")
            run.data_cached = True
            return out, size
        run.stage = f'merging {len(files)} H5 shards → training PKL'
        run_logged(run, [self.python, 'convert_3d.py', '--h5_filepath', *files, '--output_path', out + '.tmp',
                         '--no_operators'], threads=4)
        os.replace(out + '.tmp', out)
        return out, size

    def _worker(self, run):
        run.status = 'running'
        p = run.params
        try:
            pkl, size = self._data_pkl(run)
            cache = p.get('cache_operators')
            if cache is None:                                    # operators in RAM ≈ 9 × the H5 size
                cache = size < 300 * 2 ** 20
            d = self.run_dir(run.name)
            cmd = [self.python, 'train.py', '--config', 'configs/eigenspace_3d.yaml']
            if run.resume:
                cmd += ['--resume', os.path.join(d, 'last.ckpt')]
            if float(p.get('qoi_weight') or 0) > 0:
                cmd += ['--qoi_weight', str(p['qoi_weight'])]
            cmd += ['--override', f"dataset.data_path={pkl}", f"dataset.cache_operators={str(bool(cache)).lower()}",
                    f"training.log_dir={self.runs_root}", f"training.exp_name={run.name}",
                    f"training.max_epochs={int(p['epochs'])}", f"training.batch_size={int(p['batch_size'])}",
                    f"training.learning_rate={float(p['lr'])}", f"training.patience={int(p['patience'])}",
                    f"training.num_workers={int(p.get('num_workers', 2))}",
                    f"model.num_field_modes={int(p['n_modes'])}", *preset_overrides(p['preset'])]
            run.stage = 'training'
            run_logged(run, cmd, threads=max(1, (os.cpu_count() or 2) // 2))
            run.status = 'done'
        except Exception as e:                                   # noqa: BLE001 — shown in the UI
            run.status = 'cancelled' if run._cancel else 'failed'
            run.error = None if run._cancel else str(e)
            if not run._cancel:
                run.log.append(f"ERROR: {e}")
        run.finished, run.stage = time.time(), run.status
        if self.on_done is not None and run.status == 'done':
            try:
                self.on_done(run)
            except Exception as e:                               # noqa: BLE001
                run.log.append(f"after training: {e}")

    def clear_cache(self):
        shutil.rmtree(self.scratch, ignore_errors=True)
