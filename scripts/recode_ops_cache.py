"""Rewrite FieldDataset operator-cache files (`*.ops2.pkl`) with another compression policy, without the H5 files or the
meshes (docs/38 §11): e.g. the zlib-compressed cache of a finished first epoch -> raw files that load without
decompression (about 2x the disk) or `fast` ones (zlib only where it shrinks an array to 30 %: ~1.3x the disk, half the
decompression).  The content is unchanged: every file is read back and compared bit for bit (--no_verify skips that).

    python scripts/recode_ops_cache.py /content/ops_cache /content/ops_cache_raw --compress false --workers 8
    python scripts/recode_ops_cache.py CACHE_DIR CACHE_DIR --compress fast           # in place (atomic per file)

The reader accepts every kind of file, so a directory may mix them; files of another format are skipped with a message.
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import ops_cache  # noqa: E402


def _same(a, b):
    ops_a, rec_a = ops_cache.unpack(a)
    ops_b, rec_b = ops_cache.unpack(b)
    if ops_a.keys() != ops_b.keys() or not np.array_equal(rec_a["u_model"], rec_b["u_model"]):
        return False
    for k, x in ops_a.items():
        y = ops_b[k]
        if sp.issparse(x):
            if not (x.dtype == y.dtype and np.array_equal(x.indptr, y.indptr) and np.array_equal(x.indices, y.indices)
                    and x.data.tobytes() == y.data.tobytes()):
                return False
        elif isinstance(x, np.ndarray):
            if not (x.dtype == y.dtype and x.tobytes() == y.tobytes()):
                return False
    return True


def recode(args):
    src, dst, compress, verify = args
    try:
        payload = ops_cache.read(src)
    except ops_cache.OpsCacheError as e:
        return src, None, str(e)
    if compress is False:
        new = ops_cache.uncompress(payload)
    else:
        ops, rec = ops_cache.unpack(payload)
        new = ops_cache.pack(ops, rec, narrow=payload["narrow"], compress=compress)
    if verify and not _same(payload, new):
        return src, None, "content differs after recoding (file not written)"
    ops_cache.write(dst, new)
    return src, (os.path.getsize(src), os.path.getsize(dst)), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src_dir")
    ap.add_argument("dst_dir")
    ap.add_argument("--compress", choices=["true", "fast", "false"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_verify", action="store_true")
    a = ap.parse_args()
    compress = {"true": True, "fast": "fast", "false": False}[a.compress]
    files = sorted(glob.glob(os.path.join(a.src_dir, "*" + ops_cache.SUFFIX)))
    os.makedirs(a.dst_dir, exist_ok=True)
    jobs = [(f, os.path.join(a.dst_dir, os.path.basename(f)), compress, not a.no_verify) for f in files]
    tot = [0, 0]
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (src, sizes, err) in enumerate(ex.map(recode, jobs), 1):
            if err:
                print(f"[{i}/{len(jobs)}] {os.path.basename(src)}: skipped: {err}")
                continue
            tot[0] += sizes[0]
            tot[1] += sizes[1]
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {tot[0] / 1e9:.2f} GB -> {tot[1] / 1e9:.2f} GB ({tot[1] / max(tot[0], 1):.2f}x)")


if __name__ == "__main__":
    main()
