"""Check stored field labels against a high-order reference on the SAME stored model mesh.

For a field-label H5 file (or a directory of them, searched recursively; pass explicit files to control the set) pick samples (default: the first one per
shape_type), rebuild the stored model mesh, solve HCurl p<order> (default 4) on it and compare with the stored label
frequencies: per-mode relative error, max / median, the stored est_rel_f and est / true.  Samples whose p<order> space
exceeds --max_ndof are skipped with the reason (--fallback_order tries order-1 instead).  --cross_check adds the
p<order-1> solve and reports |p(order-1) - p(order)| as the reference's own error bound.

    python scripts/reference_check.py data/field/ --per_family 1 --threads 4 --out ref.json
    python scripts/reference_check.py a.h5 --samples sample_0001,sample_0007 --cross_check

Cost (docs/37): p4 on a 6k-tet mesh = 330k DOF, ~100 s on 4 threads, ~5 GB.  The JSON (--out) is rewritten after
every sample; progress lines "ref <i>/<n> <family> ..." are flushed.  Schema: src/data_gen/reference.py `run`.
"""
import argparse
import os
import sys

import torch  # noqa: F401  (before ngsolve: tests/conftest.py)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="field-label H5 files, globs or directories (recursive)")
    ap.add_argument("--order", type=int, default=4, help="reference HCurl order (default 4)")
    ap.add_argument("--per_family", type=int, default=1, help="first N samples per shape_type (default 1), counted over all files")
    ap.add_argument("--samples", default=None, help="comma-separated sample names, e.g. sample_0001,sample_0007")
    ap.add_argument("--all", action="store_true", help="every sample of every file")
    ap.add_argument("--max_ndof", type=int, default=700000, help="skip samples whose reference space is larger")
    ap.add_argument("--fallback_order", action="store_true", help="when skipped, try order-1")
    ap.add_argument("--cross_check", action="store_true",
                    help="also solve order-1 on the same mesh; report |p(order-1) - p(order)| as the reference's own error")
    ap.add_argument("--allow_other_mesh", action="store_true",
                    help="also check adaptive p3 labels whose frequencies come from a finer label mesh (est/true is then null)")
    ap.add_argument("--threads", type=int, default=0, help="NGSolve threads (0 = default)")
    ap.add_argument("--out", default=None, help="JSON result file (rewritten after every sample)")
    ap.add_argument("--json_only", action="store_true", help="no table on stdout, only progress lines")
    a = ap.parse_args(argv)
    if a.threads:
        import ngsolve
        ngsolve.SetNumThreads(a.threads)
    from src.data_gen import reference as ref

    def log(msg):
        print(msg, flush=True)

    names = [s for s in a.samples.split(",") if s] if a.samples else None
    doc = ref.run(a.paths, out=a.out, order=a.order, max_ndof=a.max_ndof, per_family=a.per_family, names=names,
                  all_samples=a.all, fallback_order=a.fallback_order, cross_check=a.cross_check,
                  allow_other_mesh=a.allow_other_mesh, log=log)
    if not a.json_only:
        print(ref.format_table(doc), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
