"""3D field visualisation of an EigenspaceOperator3D checkpoint (pure model prediction).

Per selected geometry: the N0 edge DOFs (target and aligned prediction, see
src/viz/predict.py) are turned back into E field vectors, cut by axis-aligned planes
through the centroid and plotted true | predicted | |error| per mode (PNG), and
the whole field is written to a ParaView .vtu (cell + point vectors, mm).
Fields are nodal-averaged for display (--raw: the per-tet N0 field, faceted).
Interactive CST-style view (phase / cut sliders): src/viz/viewer.py.

    python scripts/plot_3d.py --checkpoint ckpt_or_dir --data_path data/maxwell3d.pkl \
        [--split test] [--n 3 | --indices 0 5] [--axes y z] [--modes 0 1 2] \
        [--res 121] [--out_dir viz3d] [--no_vtu] [--no_png] [--raw] [--arrows]

Cavities are built along z: a y-cut is the axial (x–z) plane, a z-cut the transverse one.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.viz.nedelec import cell_field, plane_sample   # noqa: E402
from src.viz.predict import load, pick, predict         # noqa: E402
from src.viz.slices import figure_modes                 # noqa: E402
from src.viz.vtk_export import export_prediction        # noqa: E402


def render(lm, ds, idx, out_dir, axes=('y', 'z'), modes=None, res=121, png=True, vtu=True, device='cpu',
           smooth=True, arrows=False):
    """PNG / VTU files written for dataset item idx (list of paths)."""
    p = predict(lm, ds, idx, device)
    name = f"geom{p['geom_id']}_{p['shape_type'] or 'shape'}"
    fld = p.get('field', 'E')
    to_mm = (p['scale'], p['center'])
    paths = []
    if png:
        for ax in axes:
            st, sp = (plane_sample(p['X'], p['tets'], p['edges'], p[k], axis=ax, res=res, smooth=smooth)
                      for k in ('true', 'pred'))
            fig = figure_modes(st, sp, p, modes=modes, to_mm=to_mm, arrows=arrows,
                               suptitle=f"{name}  {fld} field ({ax}-cut, split={ds.split})", field=fld)
            paths.append(os.path.join(out_dir, f"{name}_{ax}cut.png"))
            fig.savefig(paths[-1], dpi=110, bbox_inches='tight')
            plt.close(fig)
    if vtu:
        paths.append(export_prediction(os.path.join(out_dir, f"{name}.vtu"), p, cell_field))
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', required=True, help='.ckpt file or a training dir')
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test', 'all'])
    ap.add_argument('--n', type=int, default=3, help='geometries (one per shape type first)')
    ap.add_argument('--indices', type=int, nargs='*', default=None, help='dataset indices instead of --n')
    ap.add_argument('--axes', nargs='*', default=['y', 'z'], choices=['x', 'y', 'z'])
    ap.add_argument('--modes', type=int, nargs='*', default=None, help='default: all outputs')
    ap.add_argument('--res', type=int, default=121, help='plane grid points per side')
    ap.add_argument('--out_dir', default='viz3d')
    ap.add_argument('--no_vtu', action='store_true')
    ap.add_argument('--no_png', action='store_true')
    ap.add_argument('--raw', action='store_true', help='raw per-tet N0 field (no nodal averaging)')
    ap.add_argument('--arrows', action='store_true', help='in-plane field arrows')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    lm, ds = load(args.checkpoint, args.data_path, args.split, args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    for idx in (args.indices if args.indices else pick(ds, args.n)):
        for path in render(lm, ds, idx, args.out_dir, args.axes, args.modes, args.res,
                           not args.no_png, not args.no_vtu, args.device, not args.raw, args.arrows):
            print(path)


if __name__ == '__main__':
    main()
