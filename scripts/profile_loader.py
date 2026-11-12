"""Throughput of the training DataLoader WITHOUT the model (docs/38 §11): how many samples/s can the workers deliver,
where does their time go, and what does the main process pay to receive and move a batch.

    python scripts/profile_loader.py --config configs/field_3d_noproj.yaml \\
        --override "dataset.data_path=/content/data/*.h5" dataset.cache_dir=/content/ops_cache \\
        --workers 4 8 16 --n_batches 60 --step_time 0.26 [--device cuda] [--gpu_prefetch] [--json out.json]

For every worker count N the train loader of train.py (make_loaders' kwargs: collate, prefetch_factor, pinning) is iterated
for `--warmup` + `--n_batches` batches.  Worker side (per sample, summed over the batch; measured inside the workers):
cache_read (pickle.load of the .ops2.pkl, or of the .mf1.pkl with dataset.operators=matfree), cache_decode (ops_cache.unpack: zlib,
delta, upper triangle → K, M; matfree: ops_cache.unpack_mf, a type check and the float64 labels), load_build
(no cache file: the operators are assembled, and written when cache_dir is set — the first epoch), field_item (field_item
with Cᵀ; matfree: field_item_matfree, no sparse matrix), collate (field_collate: padding, block-diagonal CSR; matfree:
field_collate_mf, dense tensors and the list of element records).  With dataset.operators=matfree (docs/40 §10; train / val
splits of a projection-free config) the batches carry no sparse tensor and are pinned whenever CUDA exists.  Main side: `wait` = time blocked in next() (the loader
does not keep up; includes receiving the batch: unpickling from shared memory), `to_device` = moving the batch to --device
(synchronised; with --gpu_prefetch: the SparseParts path of src/data/gpu_transfer.py; 0 on cpu).  --step_time emulates the
GPU step with a sleep (the CPU is free for the workers, as with an asynchronous GPU step): with 0 the result is the
capacity of the loader, with the real step time (docs/38 §6.2/§8: ~0.26 s at 23k tets) `wait` is what the training loop
would lose per batch; predicted GPU utilisation = step_time / (step_time + wait + to_device).  `main_cpu` is the CPU time of
the main process (all its threads, incl. the pin thread) per batch.  Prints a table; --json writes it.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.data import gpu_transfer, ops_cache  # noqa: E402
from src.data.field_dataset import FieldDataset  # noqa: E402
from train import loader_kwargs, parse_overrides  # noqa: E402

WORKER_STAGES = ('cache_read', 'cache_decode', 'load_build', 'field_item', 'collate')


class TimedDataset(torch.utils.data.Dataset):
    """Wraps a FieldDataset: __getitem__ returns (item, {stage: seconds}) with the cache read and decode timed apart."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ds = self.ds
        g = ds.active_geoms[idx]
        tm = {}
        mfree = ds.operators == 'matfree'
        cp = ds._cache_path(g, ds.build_potential, matfree=mfree)
        t0 = time.perf_counter()
        if cp and os.path.exists(cp):
            payload = ops_cache.read_mf(cp) if mfree else ops_cache.read(cp)
            t1 = time.perf_counter()
            ops, rec = (ops_cache.unpack_mf if mfree else ops_cache.unpack)(payload)
            t2 = time.perf_counter()
            tm['cache_read'], tm['cache_decode'] = t1 - t0, t2 - t1
        else:
            ops, rec = ds.load(g)
            tm['load_build'] = time.perf_counter() - t0
        t3 = time.perf_counter()
        item = ds.make_item(g, ops, rec)
        tm['field_item'] = time.perf_counter() - t3
        return item, tm


class TimedCollate:
    """Collate of (item, times) samples: the batch and the summed stage times + the collate time."""

    def __init__(self, collate):
        self.collate = collate

    def __call__(self, samples):
        tm = {}
        for _, t in samples:
            for k, v in t.items():
                tm[k] = tm.get(k, 0.0) + v
        t0 = time.perf_counter()
        batch = self.collate([s[0] for s in samples])
        tm['collate'] = time.perf_counter() - t0
        return batch, tm


def batch_bytes(obj):
    if isinstance(obj, gpu_transfer.SparseParts):
        return sum(batch_bytes(t) for t in obj[:3])
    if isinstance(obj, torch.Tensor):
        if obj.layout == torch.sparse_csr:
            return sum(batch_bytes(t) for t in (obj.crow_indices(), obj.col_indices(), obj.values()))
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(batch_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(batch_bytes(v) for v in obj)
    return 0


def summarise(xs):
    return {'mean': statistics.fmean(xs), 'median': statistics.median(xs), 'max': max(xs)}


def run(tc, ds, workers, args, gpu):
    lk = loader_kwargs(tc, workers, True, gpu, ds.operators == 'matfree')
    collate = TimedCollate(lk['collate_fn'])
    kw = dict(lk, collate_fn=collate)
    loader = torch.utils.data.DataLoader(TimedDataset(ds), batch_size=args.batch_size, shuffle=True, **kw)
    device = torch.device(args.device)
    n_total = args.warmup + args.n_batches
    rows, it = [], iter(loader)
    cpu0 = wall0 = None
    for i in range(n_total):
        if i == args.warmup:
            cpu0, wall0 = time.process_time(), time.perf_counter()
        t0 = time.perf_counter()
        try:
            batch, tm = next(it)
        except StopIteration:
            it = iter(loader)
            batch, tm = next(it)
        t1 = time.perf_counter()
        if device.type != 'cpu':
            batch = gpu_transfer.wait_ready(gpu_transfer.to_device(batch, device, None))
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if args.step_time > 0:
            time.sleep(args.step_time)
        if i >= args.warmup:
            rows.append(dict(tm, wait=t1 - t0, to_device=t2 - t1, nbytes=batch_bytes(batch)))
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    del it, loader
    out = {'workers': workers, 'batch_size': args.batch_size, 'n_batches': len(rows), 'step_time': args.step_time,
           'samples_per_s': len(rows) * args.batch_size / wall, 'main_cpu_s_per_batch': cpu / len(rows),
           'main_cpu_share': cpu / wall, 'batch_MB': statistics.fmean(r['nbytes'] for r in rows) / 1e6,
           'stages': {k: summarise([r[k] for r in rows if k in r]) for k in (*WORKER_STAGES, 'wait', 'to_device')
                      if any(k in r for r in rows)}}
    st = out['stages']
    out['worker_s_per_sample'] = sum(st[k]['mean'] for k in WORKER_STAGES if k in st) / args.batch_size
    out['predicted_gpu_util'] = (args.step_time / (args.step_time + st['wait']['mean'] + st['to_device']['mean'])
                                 if args.step_time > 0 else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/field_3d_noproj.yaml')
    ap.add_argument('--override', nargs='*', default=[], help='config overrides key=value (dataset.data_path=..., ...)')
    ap.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    ap.add_argument('--workers', type=int, nargs='+', default=None, help='worker counts to compare (default: the config)')
    ap.add_argument('--batch_size', type=int, default=None, help='default: training.batch_size')
    ap.add_argument('--n_batches', type=int, default=40)
    ap.add_argument('--warmup', type=int, default=4, help='batches not counted (worker start, first epoch cache writes)')
    ap.add_argument('--step_time', type=float, default=0.0, help='seconds slept per batch, emulating the GPU step')
    ap.add_argument('--device', default='cpu', help="'cpu' (no transfer measured) or 'cuda'")
    ap.add_argument('--gpu_prefetch', action='store_true', help='SparseParts collate + pinning (training.gpu_prefetch)')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=parse_overrides(args.override))
    tc, dc = cfg.training, cfg.dataset
    args.batch_size = args.batch_size or int(tc.batch_size)
    gpu = bool(args.gpu_prefetch) and torch.cuda.is_available()
    if args.gpu_prefetch and not gpu:
        print("--gpu_prefetch: no CUDA device, plain batches")
    projection = getattr(tc, 'projection', True)
    operators = (str(getattr(dc, 'operators', None) or 'assembled') if args.split != 'test' else 'assembled')
    if operators == 'matfree' and projection:
        raise SystemExit("dataset.operators: matfree needs training.projection: false")
    ds = FieldDataset(dc.data_path, split=args.split, train_ratio=dc.train_ratio, val_ratio=dc.val_ratio,
                      random_seed=getattr(dc, 'random_seed', 42), feature_indices=getattr(dc, 'feature_indices', None),
                      cache_dir=getattr(dc, 'cache_dir', None), n_modes=getattr(dc, 'n_modes', None),
                      hf_repeat=getattr(dc, 'hf_repeat', 1), train_extra=getattr(dc, 'train_extra', None),
                      require_label_ok=bool(getattr(dc, 'require_label_ok', False)),
                      max_tets=getattr(dc, 'max_tets', None), build_potential=projection or args.split == 'test',
                      operators=operators, **({'verify_labels': dc.verify_labels}
                                              if getattr(dc, 'verify_labels', None) is not None else {}))
    print(f"nproc={os.cpu_count()} torch_threads={torch.get_num_threads()} samples={len(ds)} split={args.split} "
          f"batch_size={args.batch_size} operators={operators} cache_dir={getattr(dc, 'cache_dir', None)} step_time={args.step_time} "
          f"device={args.device} gpu_prefetch={gpu}")
    results = [run(tc, ds, w, args, gpu) for w in (args.workers or [int(tc.num_workers)])]
    for r in results:
        print(f"\nworkers={r['workers']}  {r['samples_per_s']:.2f} samples/s  batch {r['batch_MB']:.0f} MB  "
              f"worker time {r['worker_s_per_sample']:.2f} s/sample  main CPU {r['main_cpu_s_per_batch'] * 1e3:.0f} ms/batch "
              f"({100 * r['main_cpu_share']:.0f} % of one core)"
              + (f"  predicted GPU utilisation {100 * r['predicted_gpu_util']:.0f} %" if r['predicted_gpu_util'] else ''))
        for k, v in r['stages'].items():
            print(f"    {k:13s} mean {v['mean'] * 1e3:8.1f} ms  median {v['median'] * 1e3:8.1f}  max {v['max'] * 1e3:8.1f}")
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'args': vars(args), 'nproc': os.cpu_count(), 'torch_threads': torch.get_num_threads(),
                       'results': results}, f, indent=1)


if __name__ == '__main__':
    main()
