"""Add the stored token-feature columns (qfeat_wall, docs/29) to existing field-label H5 files.

The wall-distance / unit-direction / torsion columns of the model's token features cost ~1.3–2 s of
every sample load at 9k tets (docs/31 §3.7); new files written by dataset_generator_3d --labels field
carry them, older ones get them here.  Only the new dataset is written: labels, meshes and QoI are
untouched, and files without it keep loading (the loader then computes the columns).

    python scripts/add_field_features.py data/field/*.h5 [--overwrite]
"""
import argparse
import glob
import os
import sys

import torch  # noqa: F401  (before ngsolve: tests/conftest.py)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_gen.field_labels import add_wall_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="field-label H5 files or globs")
    ap.add_argument("--overwrite", action="store_true", help="recompute samples that already have qfeat_wall")
    ap.add_argument("--threads", type=int, default=0, help="NGSolve / kd-tree threads (0 = default)")
    a = ap.parse_args()
    if a.threads:
        import ngsolve
        ngsolve.SetNumThreads(a.threads)
    files = sorted({f for p in a.paths for f in (glob.glob(p) or [p])})
    total = sum(add_wall_features(f, overwrite=a.overwrite) for f in files)
    print(f"qfeat_wall written for {total} samples in {len(files)} files")


if __name__ == "__main__":
    main()
