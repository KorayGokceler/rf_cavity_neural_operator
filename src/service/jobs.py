"""Dataset generation from the web UI: a queue of background jobs, one geometry family per job, in
the TRUBA layout (cluster/truba) so the results mix with cluster datasets:

    <gen_root>/<TAG>/h5/<family>/<family>_s00000.h5 …   (shards of `shard_size` geometries)
    <gen_root>/<TAG>/pkl/<family>.pkl                    (converted, lean by default)

Geometry ids = block · 2^19 + i with the family's fixed block from cluster/truba/families.tsv, so the
same (family, i) is the same geometry everywhere. Shards are written to a local scratch directory
first and moved to gen_root when complete (gen_root may be a slow / network mount such as Google
Drive); existing shards are skipped, so re-submitting the same job continues an interrupted one.
"""
import collections
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ID_BLOCK = 2 ** 19
_PROGRESS = re.compile(r'(\d+)/(\d+) done \((\d+) ok, (\d+) failed\)')
_TQDM = re.compile(r'\d+%\|.*\|\s*\d+/\d+')


def family_blocks(tsv=None):
    """{family: block} from cluster/truba/families.tsv (the cluster's id blocks)."""
    tsv = tsv or os.path.join(ROOT, 'cluster', 'truba', 'families.tsv')
    out = {}
    if os.path.exists(tsv):
        for line in open(tsv):
            p = line.split()
            if p and not p[0].startswith('#') and len(p) >= 2:
                out[p[0]] = int(p[1])
    return out


def tag_of(mesh_size, n_modes):
    """Dataset tag of the TRUBA layout (cluster/truba/config.sh): E_ms<mesh>_k<modes>_v1."""
    return f"E_ms{mesh_size:g}_k{int(n_modes)}_v1"


def run_logged(job, cmd, on_line=None, threads=1, env=None):
    """Run cmd (cwd = repo root) with its output in job.log (progress bars collapsed to one live line);
    job needs .log (deque), ._proc, ._cancel. Raises on a non-zero exit or a cancel."""
    job.log.append('$ ' + ' '.join(os.path.basename(c) if i == 1 else c for i, c in enumerate(cmd)))
    t = str(threads)
    env = dict(os.environ, OMP_NUM_THREADS=t, OPENBLAS_NUM_THREADS=t, MKL_NUM_THREADS=t, PYTHONUNBUFFERED='1',
               **(env or {}))
    job._proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
    for raw in job._proc.stdout:
        line = raw.rstrip().split('\r')[-1].strip()          # progress bars redraw with \r: keep the last state
        if not line:
            continue
        if _TQDM.search(line):                              # one live line per progress bar
            if job.log and _TQDM.search(job.log[-1]):
                job.log[-1] = line[-300:]
            else:
                job.log.append(line[-300:])
            continue
        job.log.append(line[-300:])
        if on_line:
            on_line(line)
    rc = job._proc.wait()
    job._proc = None
    if job._cancel:
        raise RuntimeError('cancelled')
    if rc != 0:
        raise RuntimeError(f"{os.path.basename(cmd[1])} exited with code {rc}")


class Job:
    def __init__(self, params):
        self.id = uuid.uuid4().hex[:12]
        self.params = dict(params)
        self.status = 'queued'
        self.created, self.started, self.finished = time.time(), None, None
        self.done = self.ok = self.failed = 0
        self.stage = ''
        self.error = None
        self.outputs = []
        self.log = collections.deque(maxlen=200)
        self._proc = None
        self._cancel = False

    def public(self):
        p = self.params
        return {'id': self.id, 'status': self.status, 'params': p, 'stage': self.stage,
                'progress': {'done': self.done, 'total': int(p['n_total']), 'ok': self.ok, 'failed': self.failed},
                'created': self.created, 'started': self.started, 'finished': self.finished,
                'error': self.error, 'outputs': self.outputs, 'log': list(self.log)[-40:]}


class JobManager:
    """One worker thread runs the queued jobs in order (generation is CPU-heavy)."""

    def __init__(self, gen_root, scratch=None, on_done=None, python=sys.executable):
        self.gen_root = os.path.abspath(gen_root)
        self.scratch = scratch or os.path.join(tempfile.gettempdir(), 'rfcav_gen')
        self.on_done, self.python = on_done, python
        self.jobs = collections.OrderedDict()
        self._q = collections.deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    # ── API ──────────────────────────────────────────────────────────
    def submit(self, params):
        job = Job(params)
        with self._lock:
            self.jobs[job.id] = job
            self._q.append(job)
        self._wake.set()
        return job

    def cancel(self, jid):
        job = self.jobs[jid]
        job._cancel = True
        if job.status == 'queued':
            job.status, job.finished = 'cancelled', time.time()
        if job._proc and job._proc.poll() is None:
            job._proc.terminate()
        return job

    def paths(self, p):
        base = os.path.join(self.gen_root, p['tag'])
        return os.path.join(base, 'h5', p['family']), os.path.join(base, 'pkl', f"{p['family']}.pkl")

    # ── worker ───────────────────────────────────────────────────────
    def _loop(self):
        while True:
            self._wake.wait()
            with self._lock:
                job = self._q.popleft() if self._q else None
                if not self._q:
                    self._wake.clear()
            if job is None or job.status == 'cancelled':
                continue
            job.status, job.started = 'running', time.time()
            try:
                self._run(job)
                job.status = 'cancelled' if job._cancel else 'done'
            except Exception as e:                                   # noqa: BLE001 — reported to the UI
                job.status, job.error = ('cancelled' if job._cancel else 'failed'), str(e)
                job.log.append(f"ERROR: {e}")
            job.finished = time.time()
            if self.on_done and job.status == 'done':
                try:
                    self.on_done(job)
                except Exception as e:                               # noqa: BLE001
                    job.log.append(f"rescan failed: {e}")

    def _exec(self, job, cmd, on_line=None):
        run_logged(job, cmd, on_line)

    def _run(self, job):
        p = job.params
        h5_dir, pkl = self.paths(p)
        os.makedirs(h5_dir, exist_ok=True)
        tmp = os.path.join(self.scratch, job.id)
        os.makedirs(tmp, exist_ok=True)
        n, size = int(p['n_total']), int(p['shard_size'])
        n_shards = -(-n // size)
        try:
            for s in range(n_shards):
                if job._cancel:
                    raise RuntimeError('cancelled')
                dest = os.path.join(h5_dir, f"{p['family']}_s{s:05d}.h5")
                count = min(size, n - s * size)
                if os.path.exists(dest):
                    job.done += count
                    job.log.append(f"exists, skipped: {os.path.basename(dest)}")
                    continue
                job.stage = f"shard {s + 1}/{n_shards}"
                local = os.path.join(tmp, os.path.basename(dest))
                start = int(p['block']) * ID_BLOCK + s * size
                base_done = job.done

                def prog(line, base=base_done):
                    m = _PROGRESS.search(line)
                    if m:
                        job.done = base + int(m.group(1))
                        job.ok_shard, job.failed_shard = int(m.group(3)), int(m.group(4))

                ok0, f0 = job.ok, job.failed
                job.ok_shard = job.failed_shard = 0
                self._exec(job, [self.python, 'src/data_gen/dataset_generator_3d.py', '--families', p['family'],
                                 '--n_total', str(count), '--start_id', str(start),
                                 '--seed', str(p['seed']), '--n_eigen_modes', str(p['n_modes']),
                                 '--mesh_size', str(p['mesh_size']), '--sampling', p['sampling'],
                                 '--deform_prob', str(p['deform_prob']), '--deform_max', str(p['deform_max']),
                                 '--n_workers', str(p['workers']), '--sample_timeout', '900',
                                 '--h5_filename', local, '--resume'], prog)
                job.ok, job.failed = ok0 + job.ok_shard, f0 + job.failed_shard
                job.done = base_done + count
                job.stage = f"copying shard {s + 1} to the storage"
                shutil.move(local, dest + '.part')
                os.replace(dest + '.part', dest)
                job.outputs.append(dest)
            if p.get('convert', True):
                job.stage = 'converting to PKL'
                shards = sorted(f for f in os.listdir(h5_dir) if f.endswith('.h5'))
                local_pkl = os.path.join(tmp, os.path.basename(pkl))
                cmd = [self.python, 'convert_3d.py', '--h5_filepath', *[os.path.join(h5_dir, f) for f in shards],
                       '--output_path', local_pkl] + (['--no_operators'] if p.get('lean', True) else [])
                self._exec(job, cmd)
                os.makedirs(os.path.dirname(pkl), exist_ok=True)
                shutil.move(local_pkl, pkl + '.part')
                os.replace(pkl + '.part', pkl)
                job.outputs.append(pkl)
            job.stage = 'done'
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
