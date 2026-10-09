"""Pure-model prediction on ONE new cavity: frequencies, mode fields and figures of merit
(Q0, G, R/Q, R_sh, T, Epk/Eacc, Bpk/Eacc) without an FE eigen-solve — the end of the pipeline.

Geometry sources (exactly one):
  --step  FILE.step|.stp|.brep|.iges   CAD (gmsh OCC import + tet mesh, like the generator)
  --mesh  FILE.msh|.vtu|…              a tetrahedral mesh (meshio; tetra cells)
  --h5    SHARD.h5 [--sample sample_0000123 | index]   a generated geometry (FE truth known)
  --family NAME --id N                 a generator geometry by family and id (mesh only)

The cavity must be closed (PEC walls; beam pipes capped) and, for R/Q / V_acc, have its beam axis
on x = y = 0 along z (else --axis_xy). Lengths: --unit (default m; CAD files are often mm).
Mesh density: h = --mesh_size · (V / --vol_div)^(1/3) as in training (for an n-cell cavity use
--vol_div n so the per-cell density matches the elliptical family), or --h_abs [m].

    python scripts/predict_geometry.py --checkpoint DIR --step tesla.step --unit mm --vol_div 9 \
        [--fe] [--out_dir pred_tesla] [--n_axis 401] [--Rs 1e-8]

--fe also solves the FE eigenproblem on the same mesh and prints model-vs-FE errors (frequency,
field rel-L2 per mode, QoI) and both run times. Outputs in --out_dir: prediction.json, modes.vtu
(+ .json sidecar; E/H of every predicted mode, and the FE ones with --fe) for ParaView.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.dataset_converter_3d import extract_geometry_3d, h5_edges_to_canonical  # noqa: E402
from src.data_gen import dataset_generator_3d as gen                          # noqa: E402
from src.service.geometry import UNITS, from_h5, mesh_file, mesh_step         # noqa: E402,F401
from src.service.geometry import gen_args as _gen_args                        # noqa: E402
from src.service.model import QOI_SHOW, degenerate_mask, load_model           # noqa: E402,F401
from src.service.model import forward as _forward                             # noqa: E402


def predict(lm, geom, field, feature_indices=None, device='cpu', warmup=True):
    """(U [Ne,K], f [K] GHz, model seconds) — src.service.model.forward without λ."""
    U, f, _, t = _forward(lm, geom, feature_indices, device, warmup)
    return U, f, t


def qoi(geom, M, U, f_ghz, field, args):
    from src.data.dataset_converter_3d import qoi_operators_of
    from src.qoi import qoi_from_dofs
    kw = {'n_axis': args.n_axis}
    if args.axis_dir or args.axis_point:
        kw.update(axis_dir=args.axis_dir, axis_point=args.axis_point)
    elif args.axis_xy:
        kw['axis_xy'] = tuple(args.axis_xy)
    ops = qoi_operators_of(geom, field, M=M, **kw)       # default: the family's beam axis (hwr: x)
    return qoi_from_dofs(ops, U, np.asarray(f_ghz) * 1e9, Rs=args.Rs, sigma=args.sigma, beta=args.beta,
                         L_acc=args.L_acc, convention=args.convention)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', required=True, help='.ckpt file or a training dir')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--step', help='CAD file (.step/.stp/.brep/.iges)')
    src.add_argument('--mesh', help='tetrahedral mesh file (meshio)')
    src.add_argument('--h5', help='generator H5 shard')
    src.add_argument('--family', help='generator family (with --id)')
    ap.add_argument('--sample', default='0', help='--h5: group name (sample_…) or index')
    ap.add_argument('--id', type=int, default=0, help='--family: geometry id (e.g. block·2^19 + i)')
    ap.add_argument('--unit', default='m', choices=sorted(UNITS), help='length unit of --step / --mesh')
    ap.add_argument('--mesh_size', type=float, default=0.10, help='relative tet size (training: 0.10)')
    ap.add_argument('--vol_div', type=float, default=1.0, help='volume divisor of h (n cells → n)')
    ap.add_argument('--h_abs', type=float, default=None, help='absolute tet size [m] (overrides --mesh_size)')
    ap.add_argument('--field', choices=['E', 'H'], default=None,
                    help='only for checkpoints that do not record their training field')
    ap.add_argument('--fe', action='store_true', help='also solve FE on the same mesh and compare')
    ap.add_argument('--n_axis', type=int, default=401)
    ap.add_argument('--axis_xy', type=float, nargs=2, default=None, help='beam axis along z at (x, y) [m]')
    ap.add_argument('--axis_dir', default=None, choices=['x', 'y', 'z'], help='beam axis direction')
    ap.add_argument('--axis_point', type=float, nargs=3, default=None, help='a point of the beam axis [m]')
    ap.add_argument('--sigma', type=float, default=5.8e7, help='wall conductivity [S/m] (copper)')
    ap.add_argument('--Rs', type=float, default=None, help='fixed surface resistance [Ω] (e.g. SRF Nb)')
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--L_acc', type=float, default=None, help='accelerating length [m] (default axis chord)')
    ap.add_argument('--convention', default='linac', choices=['linac', 'circuit'])
    ap.add_argument('--deg_rel', type=float, default=1e-3)
    ap.add_argument('--out_dir', default='prediction')
    ap.add_argument('--no_vtu', action='store_true')
    ap.add_argument('--no_warmup', action='store_true', help='time the first forward (includes set-up)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args(argv)

    lm, field, fi = load_model(args.checkpoint, args.device, args.field)
    K_out = int(lm.hparams.get('num_field_modes', 6))
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) geometry → tet mesh [m]
    t0 = time.perf_counter()
    truth, label, family = None, '', ''
    if args.step:
        _gen_args(field, args.mesh_size)
        nodes, tets = mesh_step(args.step, args.unit, args.mesh_size, args.vol_div, args.h_abs)
        label = os.path.basename(args.step)
    elif args.mesh:
        nodes, tets = mesh_file(args.mesh, args.unit)
        label = os.path.basename(args.mesh)
    elif args.h5:
        nodes, tets, truth = from_h5(args.h5, args.sample)
        label = f"{os.path.basename(args.h5)}:{truth['key']} ({truth['shape_type']})"
        family = truth['shape_type']
        if truth['field'] != field:
            raise ValueError(f"checkpoint trained on field {field}, H5 is {truth['field']}")
    else:
        _gen_args(field, args.mesh_size, families=[args.family])
        g = gen.mesh_sample(args.id)
        nodes, tets, label = g['nodes'], g['tets'], f"{args.family} id {args.id}"
        family = args.family
    t_mesh = time.perf_counter() - t0
    topo = gen.mesh_topology(tets, len(nodes))
    gen.certify(topo, field)

    # 2) optional FE reference on the same mesh (skfem order → canonical below)
    t_fe = None
    if args.fe and truth is None:
        _gen_args(field, args.mesh_size, n_modes=K_out)
        t0 = time.perf_counter()
        A = gen.assemble_h_n0(nodes, tets)
        vals, vecs, _ = gen.solve_modes(A, K_out + 1, field)
        t_fe = time.perf_counter() - t0
        nodes, tets = np.ascontiguousarray(A['mesh'].p.T), np.ascontiguousarray(A['mesh'].t.T)
        truth = {'edges': A['edges'], 'vecs': vecs[:, :K_out], 'freqs': gen.eigenvalues_to_ghz(vals[:K_out])}

    # 3) model input: normalised mesh, features, N0 operators (same code as the training PKLs)
    t0 = time.perf_counter()
    geom, M = extract_geometry_3d(nodes, tets, field)
    geom.update(field=field, shape_type=family)
    t_ops = time.perf_counter() - t0

    # 4) prediction (Ritz fields + frequencies), 5) figures of merit from the predicted field
    U, f_pred, t_model = predict(lm, geom, field, fi, args.device, warmup=not args.no_warmup)
    t0 = time.perf_counter()
    q_pred = qoi(geom, M, U, f_pred, field, args)
    t_qoi = time.perf_counter() - t0
    deg = degenerate_mask(f_pred, args.deg_rel)

    res = {'geometry': label, 'field': field, 'n_vertices': int(len(nodes)), 'n_edges': int(len(geom['edges'])),
           'n_tets': int(len(tets)), 'scale_m': float(geom['scale']), 'betti1': int(topo['betti1']),
           'n_bnd_components': int(topo['n_shells']),
           'f_pred_GHz': f_pred.tolist(), 'degenerate': deg.tolist(),
           'qoi_pred': {k: np.asarray(v).tolist() for k, v in q_pred.items()},
           'qoi_settings': {'sigma': args.sigma, 'Rs': args.Rs, 'beta': args.beta, 'L_acc': args.L_acc,
                            'convention': args.convention, 'axis_xy': args.axis_xy, 'axis_dir': args.axis_dir,
                            'axis_point': args.axis_point, 'U_J': 1.0},
           'time_s': {'mesh': t_mesh, 'operators': t_ops, 'model': t_model, 'qoi': t_qoi}}

    # 6) comparison with FE (when known)
    T = None
    if truth is not None:
        rows, sign = h5_edges_to_canonical(truth['edges'], len(nodes), geom['edges'])
        Kt = min(truth['vecs'].shape[1], len(f_pred))
        T = np.zeros((len(geom['edges']), Kt))
        T[rows] = sign[:, None] * truth['vecs'][:, :Kt]
        f_true = np.asarray(truth['freqs'][:Kt], float)
        q_true = qoi(geom, M, T, f_true, field, args)
        MT = M @ T
        tt = np.einsum('ek,ek->k', T, MT)
        c = U[:, :Kt].T @ MT                                                 # [K,K] u_iᵀ M t_j
        rl = []
        for k in range(Kt):                                                  # sign-free rel-L2 (pair: span)
            idx = [j for j in range(Kt) if abs(f_true[j] - f_true[k]) < args.deg_rel * f_true[k]]
            proj = (c[idx, k] ** 2).sum() / tt[k]
            rl.append(float(np.sqrt(max(1.0 - proj, 0.0))))
        res.update(f_fe_GHz=f_true.tolist(), f_rel_err=(f_pred[:Kt] / f_true - 1).tolist(), field_rel_l2=rl,
                   qoi_fe={k: np.asarray(v).tolist() for k, v in q_true.items()})
        if t_fe is not None:
            res['time_s']['fe_solve'] = t_fe

    # 7) report + files
    print(f"{label}: field {field}, Nv {len(nodes)}, Ne {len(geom['edges'])}, b1 {topo['betti1']}")
    hdr = f"{'mode':>4} {'f_pred GHz':>11}" + (f" {'f_FE GHz':>10} {'f err':>8} {'relL2':>6}" if T is not None else '')
    print(hdr + ''.join(f" {k[:12]:>12}" for k in QOI_SHOW))
    for k in range(len(f_pred)):
        line = f"{k:>4} {f_pred[k]:>11.5f}"
        if T is not None and k < len(res['f_fe_GHz']):
            line += f" {res['f_fe_GHz'][k]:>10.5f} {res['f_rel_err'][k]:>+8.2%} {res['field_rel_l2'][k]:>6.3f}"
        line += ''.join(f" {float(q_pred[q][k]):>12.4g}" for q in QOI_SHOW) + ('  (degenerate)' if deg[k] else '')
        print(line)
        if T is not None and k < len(res['f_fe_GHz']):
            print(f"{'':>4} {'FE':>11}" + ' ' * 35 + ''.join(f" {float(res['qoi_fe'][q][k]):>12.4g}" for q in QOI_SHOW))
    ts = res['time_s']
    print("time [s]: " + ', '.join(f"{k} {v:.3f}" for k, v in ts.items())
          + (f"  → model+operators+QoI {ts['operators'] + ts['model'] + ts['qoi']:.2f} s vs FE {ts['fe_solve']:.2f} s"
             if 'fe_solve' in ts else ''))
    with open(os.path.join(args.out_dir, 'prediction.json'), 'w') as fo:
        json.dump(res, fo, indent=1)
    if not args.no_vtu:
        from src.viz.nedelec import vertex_field
        from src.viz.vtk_export import write_modes_vtu
        prim, sec = field, ('H' if field == 'E' else 'E')
        pts = {}
        for name, D in (('pred', U), ('fe', T)):
            if D is None:
                continue
            P = vertex_field(geom['X'], geom['tets'], geom['edges'], D)
            C = vertex_field(geom['X'], geom['tets'], geom['edges'], D, curl=True)
            for k in range(D.shape[1]):
                pts[f"{prim}_{name}_{k}"], pts[f"{sec}_{name}_{k}"] = P[:, :, k], C[:, :, k]
        X_mm = (geom['X'] * geom['scale'] + geom['center']) * 1e3
        vtu = write_modes_vtu(os.path.join(args.out_dir, 'modes.vtu'), X_mm, geom['tets'], {}, pts,
                              field_data={'f_pred_GHz': f_pred, 'units': 'mm; fields unit-energy shape, '
                                          f'{sec} ∝ curl {prim}'})
        print(f"wrote {vtu}")
    print(f"wrote {os.path.join(args.out_dir, 'prediction.json')}")
    return res


if __name__ == '__main__':
    main()
