"""Pure-model prediction on ONE new cavity with the field3d model: frequencies (Ritz values), mode fields and
figures of merit (Q0, G, R/Q, R_sh, T, Epk/Eacc, Bpk/Eacc from the predicted HCurl(p2) fields) without an FE
eigen-solve — the end of the pipeline, the same code as the web UI's Predict (src/service/model.py).

Geometry sources (exactly one; the field model needs the CAD solid, it predicts on its curved mesh):
  --step  FILE.step|.stp|.brep|.iges   CAD (one closed solid: the cavity's vacuum).  STEP / IGES are read in the
                                       unit they declare (--unit ignored); a BREP has no unit: --unit (default m,
                                       the generator's BREPs)
  --family NAME --id N                 a generator geometry (field-label settings: the field datasets' shape)
  --h5    SHARD.h5 [--sample sample_0000123 | index]   a field-label geometry (stored curved mesh, labels known)

The beam axis and accelerating length follow the family (src.qoi.figures.beam_axis); a CAD file uses the
z axis through the origin.  Mesh density: h = --mesh_size · (V / --vol_div)^(1/3) (n-cell cavity: --vol_div n),
or --h_abs [m]; the curved mesh is netgen's at maxh = 3 h (field_labels.start_mesh).

    python scripts/predict_geometry.py --checkpoint DIR --step tesla.step --vol_div 9 \\
        [--fe] [--out_dir pred_tesla]

--checkpoint omitted: an UNTRAINED field3d model (numbers meaningless; for testing the pipeline).
--fe also solves the FE eigenproblem in the model's own space (HCurl(p2) on the same curved mesh) and prints
the model-vs-FE frequency errors; the Ritz values are upper bounds of these discrete eigenvalues.  With --h5
the stored p3 label frequencies and figures of merit are printed as well.  Outputs in --out_dir:
prediction.json and modes.vtu (E / H of every predicted mode at the P2 nodes of the curved mesh, 8 sub-tets per tet) for ParaView.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch  # noqa: F401  (before ngsolve: tests/conftest.py)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.service import geometry as G                                         # noqa: E402
from src.service.model import QOI_SHOW, ModelService                          # noqa: E402


def h5_geometry(path, sample):
    """Field-label H5 group (name or index) → geometry dict (as geometry.mesh_family) + stored labels."""
    import h5py

    from src.data.mesh_features import geom_params_of
    from src.data_gen.field_labels import read_field_group
    with h5py.File(path, 'r') as f:
        keys = sorted(k for k in f if k.startswith('sample_'))
        key = keys[int(sample)] if str(sample).isdigit() and str(sample) not in keys else str(sample)
        g = f[key]
        if str(g.attrs.get('labels', '')) != 'field':
            raise SystemExit(f"{key}: not a field-label sample (generator output): no curved mesh stored")
        mesh, rec = read_field_group(g)
        src = {'vol': np.asarray(g['vol']).tobytes(), 'brep': np.asarray(g['brep']).tobytes(),
               'curve': int(g.attrs['curve']), 'model_order': int(rec['model_order']),
               'deform': (tuple(np.asarray(g[k]) for k in ('deform_om', 'deform_a', 'deform_ph'))
                          if 'deform_om' in g else None)}
        params = geom_params_of(g.attrs)
    nodes, tets = G.linear_mesh(mesh)
    names = [n for n in rec['qoi_names'] if n]
    return ({'nodes': nodes, 'tets': tets, 'mesh_src': src, 'params': params, 'n_tets': int(mesh.ne),
             'blocker': None, 'family': rec['shape_type'], 'label': f"{os.path.basename(path)}:{key}"},
            {'f_GHz': rec['freqs'].tolist(),
             'qoi': [{n: float(rec['qoi'][k, j]) for j, n in enumerate(names)} for k in range(len(rec['freqs']))]})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', default=None, help='field3d .ckpt file or training dir (none: untrained model)')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--step', help='CAD file (.step/.stp/.brep/.iges)')
    src.add_argument('--h5', help='field-label H5 shard')
    src.add_argument('--family', help='generator family (with --id)')
    ap.add_argument('--sample', default='0', help='--h5: group name (sample_…) or index')
    ap.add_argument('--id', type=int, default=0, help='--family: geometry id (e.g. block·2^19 + i)')
    ap.add_argument('--unit', default=None, choices=sorted(G.UNITS),
                    help='length unit of a --step BREP (default m); ignored for STEP / IGES (unit from the file)')
    ap.add_argument('--mesh_size', type=float, default=0.10, help='relative mesh size (field datasets: 0.10)')
    ap.add_argument('--vol_div', type=float, default=1.0, help='volume divisor of h (n cells → n)')
    ap.add_argument('--h_abs', type=float, default=None, help='absolute mesh size h [m] (overrides --mesh_size)')
    ap.add_argument('--fe', action='store_true', help='also solve FE in the model space and compare')
    ap.add_argument('--out_dir', default='prediction')
    ap.add_argument('--no_vtu', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args(argv)

    svc = ModelService(args.checkpoint, args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) geometry → curved mesh (bytes) + its linear mesh
    t0 = time.perf_counter()
    labels = None
    if args.step:
        m = G.mesh_upload(args.step, args.unit or 'm', args.mesh_size, args.vol_div, args.h_abs)
        if m['unit_used'] == 'file':
            print(f"{args.step}: length unit read from the file (STEP / IGES)"
                  + (f"; --unit {args.unit} ignored" if args.unit else ''))
        else:
            print(f"{args.step}: length unit {m['unit_used']} (--unit)")
        n_sharp = m.get('n_sharp_edges')
        if n_sharp:
            print(f"{args.step}: {n_sharp} sharp re-entrant edges: Epk/Eacc and Bpk/Eacc are not converged and "
                  "are not reported")
        print("beam axis assumed: z through the origin; L_acc = axis chord incl. pipes; R/Q, R_sh, T and "
              "peaks/Eacc depend on it")
        geo = dict(m, family='', label=os.path.basename(args.step),
                   params=None if n_sharp is None else {'n_sharp_edges': float(n_sharp)})
    elif args.h5:
        geo, labels = h5_geometry(args.h5, args.sample)
    else:
        geo = dict(G.mesh_family(args.family, args.id, args.mesh_size), family=args.family,
                   label=f"{args.family} id {args.id}")
    t_geo = time.perf_counter() - t0
    if geo['mesh_src'] is None:
        raise SystemExit(f"{geo['label']}: cannot predict — {geo['blocker']}")
    chk = G.check(geo['nodes'], geo['tets'])
    if not chk['ok']:
        raise SystemExit(f"{geo['label']}: geometry failed the checks: {'; '.join(chk['problems'])}")

    # 2) prediction (Ritz modes in HCurl(p2), figures of merit of the p2 fields, display fields)
    sol = svc.predict(geo['mesh_src'], geo['family'], geo['params'])
    f_pred = np.array([m['f_GHz'] for m in sol['modes']])
    res = {'geometry': geo['label'], 'family': geo['family'], 'untrained': svc.untrained,
           'n_tets': sol['n_tets'], 'n_dof': sol['n_dof'], 'betti1': chk['betti1'],
           'f_pred_GHz': f_pred.tolist(), 'modes': sol['modes'], 'qoi_source': sol['qoi_source'],
           'conventions': {'U_J': 1.0, 'sigma': 5.8e7, 'beta': 1.0, 'R_over_Q': 'linac'},
           'time_s': dict(geometry=t_geo, **sol['time_s'])}

    # 3) optional FE reference in the model space (Ritz values are upper bounds of its eigenvalues)
    if args.fe:
        from src.data_gen import highorder as ho
        t0 = time.perf_counter()
        mesh = G.load_mesh_src(geo['mesh_src'])
        lab = ho.solve_modes(mesh, len(f_pred), order=int(geo['mesh_src'].get('model_order') or 2))
        f_fe = lab['f_hz'][:len(f_pred)] / 1e9
        res.update(f_fe_GHz=f_fe.tolist(), f_rel_err=(f_pred / f_fe - 1).tolist())
        res['time_s']['fe_solve'] = time.perf_counter() - t0
    if labels is not None:
        k = min(len(f_pred), len(labels['f_GHz']))
        res.update(f_label_GHz=labels['f_GHz'][:k], qoi_label=labels['qoi'][:k],
                   f_rel_err_label=(f_pred[:k] / np.asarray(labels['f_GHz'][:k]) - 1).tolist())

    # 4) report + files
    print(f"{geo['label']}: {sol['n_tets']} curved tets, {sol['n_dof']} HCurl(p2) DOFs, b1 {chk['betti1']}"
          + ('  [UNTRAINED model]' if svc.untrained else ''))
    hdr = f"{'mode':>4} {'f_pred GHz':>11} {'eta':>8}" + (f" {'f_FE GHz':>10} {'f err':>8}" if args.fe else '')
    print(hdr + ''.join(f" {q[:12]:>12}" for q in QOI_SHOW))
    for k, m in enumerate(sol['modes']):
        line = f"{k:>4} {m['f_GHz']:>11.5f} {m['eta']:>8.2e}"
        if args.fe:
            line += f" {res['f_fe_GHz'][k]:>10.5f} {res['f_rel_err'][k]:>+8.2%}"
        line += ''.join(f" {('—' if m[q] is None else f'{m[q]:.4g}'):>12}" for q in QOI_SHOW)
        print(line + ('  (degenerate)' if m['degenerate'] else '') + ('  (accelerating)' if m['accelerating'] else ''))
    if labels is not None:
        print('stored p3 labels: f [GHz] ' + ' '.join(f"{f:.5f}" for f in res['f_label_GHz']))
    print("time [s]: " + ', '.join(f"{k} {v:.3f}" for k, v in res['time_s'].items()))
    with open(os.path.join(args.out_dir, 'prediction.json'), 'w') as fo:
        json.dump(res, fo, indent=1)
    if not args.no_vtu:
        from src.viz.vtk_export import write_modes_vtu
        r = svc.fields(sol)                                   # the p2 fields at the P2 nodes of the curved mesh
        pts = {}
        for k in range(len(f_pred)):
            pts[f"E_pred_{k}"], pts[f"H_pred_{k}"] = r['E'][k], r['H'][k]
        vtu = write_modes_vtu(os.path.join(args.out_dir, 'modes.vtu'), r['x_mm'], r['tets'], {},
                              pts, field_data={'f_pred_GHz': f_pred,
                                               'units': 'mm; E [V/m], H [A/m] at U = 1 J, HCurl(p2) fields at the P2 '
                                                        'nodes of the curved mesh (8 linear sub-tets per tet)'})
        print(f"wrote {vtu}")
    print(f"wrote {os.path.join(args.out_dir, 'prediction.json')}")
    return res


if __name__ == '__main__':
    main()
