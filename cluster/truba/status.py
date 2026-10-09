"""Progress of the per-family dataset generation (run via status.sh, or directly).

Per family of families.tsv: shards finished / partial (running or interrupted, *.h5.partial) /
missing, geometries stored, failures of the last runs, PKL present + size.

    python cluster/truba/status.py --out_dir /arf/scratch/$USER/rfcav3d/<TAG> [--families_tsv …]
"""
import argparse
import glob
import os

import h5py


def families(tsv, test=False, test_n=112):
    rows = []
    for line in open(tsv):
        p = line.split()
        if not p or p[0].startswith('#') or len(p) < 5:
            continue
        fam, block, n, shard, group = p[0], int(p[1]), int(p[2]), int(p[3]), p[4]
        if test:
            n = shard = test_n
        rows.append((fam, block, n, shard, group))
    return rows


def count(path):
    try:
        with h5py.File(path, 'r') as f:
            ok = sum(k.startswith('sample_') for k in f.keys())
            return ok, int(f.attrs.get('n_failed_last_run', 0))
    except OSError:                       # partial file being written right now
        return 0, 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out_dir', required=True, help='$DATA_ROOT/$TAG')
    ap.add_argument('--families_tsv', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'families.tsv'))
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--test_n', type=int, default=112)
    a = ap.parse_args()
    h5_dir, pkl_dir = os.path.join(a.out_dir, 'h5'), os.path.join(a.out_dir, 'pkl')
    head = f"{'family':<16}{'group':<6}{'shards':>12}{'partial':>8}{'geoms':>14}{'failed':>8}{'pkl':>10}"
    print(a.out_dir)
    print(head)
    print('-' * len(head))
    tot = [0, 0]
    for fam, _, n, shard, group in families(a.families_tsv, a.test, a.test_n):
        n_sh = -(-n // shard)
        done = sorted(glob.glob(os.path.join(h5_dir, fam, f'{fam}_s*.h5')))
        part = sorted(glob.glob(os.path.join(h5_dir, fam, f'{fam}_s*.h5.partial')))
        ok = fail = 0
        for p in done + part:
            o, f = count(p)
            ok, fail = ok + o, fail + f
        pkl = os.path.join(pkl_dir, f'{fam}.pkl')
        size = f"{os.path.getsize(pkl) / 2**30:.1f} GB" if os.path.exists(pkl) else '-'
        tot = [tot[0] + ok, tot[1] + n]
        print(f"{fam:<16}{group:<6}{len(done):>6}/{n_sh:<5}{len(part):>8}{ok:>8}/{n:<6}{fail:>8}{size:>10}")
    print('-' * len(head))
    print(f"{'total':<22}{'':>20}{tot[0]:>8}/{tot[1]:<6}")
    mixes = sorted(glob.glob(os.path.join(pkl_dir, 'mix_*.pkl')))
    for m in mixes:
        print(f"mix: {os.path.basename(m)}  {os.path.getsize(m) / 2**30:.1f} GB")


if __name__ == '__main__':
    main()
