"""STEP copies [mm] of the geometries in existing field-label H5 files (for CST and other CAD tools).

Files generated without --step_dir (or shards a UI job skipped because they already existed) still carry the
CAD solid of every sample ('brep', docs/29); this writes it as <shape_type>_<id:08d>.step, the names
dataset_generator_3d --step_dir uses.  Labels are untouched.  Deformed samples get no STEP (their labels
belong to the warped mesh); N0-label files store no CAD and need a new generation with --step_dir.

    python scripts/export_step.py <gen_root>/<TAG>/h5/elliptical/*.h5 --out <gen_root>/<TAG>/step/elliptical
"""
import argparse
import glob
import os
import sys

import torch  # noqa: F401  (before ngsolve: tests/conftest.py)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_gen.field_labels import export_step_files  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="field-label H5 files or globs")
    ap.add_argument("--out", required=True, help="directory for the STEP files")
    ap.add_argument("--overwrite", action="store_true", help="rewrite STEP files that already exist")
    a = ap.parse_args(argv)
    files = sorted({f for p in a.paths for f in (glob.glob(p) or [p])})
    total = {}
    for f in files:
        for k, v in export_step_files(f, a.out, overwrite=a.overwrite).items():
            total[k] = total.get(k, 0) + v
    print(f"{len(files)} files: {total.get('written', 0)} STEP written, {total.get('existing', 0)} existing, "
          f"{total.get('deformed', 0)} deformed (no STEP), {total.get('no_cad', 0)} without CAD (N0 labels)")


if __name__ == "__main__":
    main()
