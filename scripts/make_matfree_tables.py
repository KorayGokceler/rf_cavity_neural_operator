"""(Re)build and verify the 24-class reference tables of the matrix-free operators (src/data/matfree.py, docs/40).

    python scripts/make_matfree_tables.py              # build if the cache lacks them, print the path
    python scripts/make_matfree_tables.py --rebuild    # ignore the cache and rebuild (atomic write)
    python scripts/make_matfree_tables.py --verify     # + errors against highorder.field_operators on a synthetic
                                                       #   24-class mesh and on a small curved pillbox

The tables come from NGSolve alone and are cached in $RFCAV_CACHE or ~/.cache/rfcav, keyed by the ngsolve version and
matfree.FORMAT; nothing is committed.  Exit status 1 if a verification error exceeds --tol."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402,F401  (before ngsolve)

from src.data import matfree as mf  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rebuild", action="store_true", help="ignore the cache and rebuild")
    ap.add_argument("--verify", action="store_true", help="compare with field_operators on small curved meshes")
    ap.add_argument("--tol", type=float, default=1e-11, help="max relative error accepted by --verify")
    args = ap.parse_args()
    t = time.perf_counter()
    tab = mf.reference_tables(rebuild=args.rebuild, verbose=True)
    print(f"tables: {mf.table_path()}  ({time.perf_counter() - t:.1f} s; T {tab['T'].shape}, phi {tab['phi'].shape})")
    if not args.verify:
        return 0
    from netgen.occ import Axes, Cylinder, OCCGeometry, Z
    from ngsolve import Mesh
    geo = Cylinder(Axes((0, 0, 0), Z), r=0.1, h=0.1)
    geo.faces.name = "wall"
    pill = Mesh(OCCGeometry(geo).GenerateMesh(maxh=0.04))
    pill.Curve(3)
    worst = 0.0
    for name, mesh in (("synthetic 24 classes (random cubic maps)", mf._synthetic_mesh(2, 99, 0.05)),
                       ("curved pillbox", pill)):
        err = mf.check_against_ngsolve(mesh)
        worst = max(worst, *(v for k, v in err.items() if k != "classes"))
        print(f"{name}: " + ", ".join(f"{k} {v:.1e}" if k != "classes" else f"{v} classes" for k, v in err.items()))
    print("PASS" if worst <= args.tol else f"FAIL (worst {worst:.1e} > {args.tol:g})")
    return 0 if worst <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
