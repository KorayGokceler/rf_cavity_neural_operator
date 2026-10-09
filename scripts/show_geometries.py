"""Gallery of generated 3D cavity geometries (two views each) from a generator H5.

    python scripts/show_geometries.py data/maxwell3d_part00.h5 [--n 12 | --ids 0 5 9] [--out gallery.png]
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use('Agg')

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.viz.geometry import gallery, load_h5  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('h5')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--ids', type=int, nargs='*', default=None)
    ap.add_argument('--out', default='geometry_gallery.png')
    args = ap.parse_args()
    geoms = load_h5(args.h5, args.n, args.ids)
    gallery(geoms).savefig(args.out, dpi=90)
    print(args.out, f"({len(geoms)} geometries)")


if __name__ == '__main__':
    main()
