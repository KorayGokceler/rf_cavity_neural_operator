"""Data / label-noise-floor diagnostic for the RF-cavity neural operator.

Goal (why this exists): the model plateaus at ~0.10 relative field error and
architecture changes have stopped helping.  Before another architecture change
we must answer: *is ~0.10 a label-noise floor (a DATA problem) or a model
problem?*  This script measures, from a trained checkpoint:

  SECTION A  (always; needs only the checkpoint + a split)
    A1  per-mode sign-agnostic relative-L2  (reproduces the training metric)
    A2  the SAME error re-weighted by node_area  (M-norm proxy, mesh-invariant).
        If A2 << A1, the L-inf + node-uniform labelling/metric is inflating the
        number -> an M-norm relabel is the high-leverage data fix.
    A3  well-separated vs near-degenerate split  (where does 0.10 come from?)
    A4  prediction-vs-target scale sanity        (refutes a cold-start zero pred)

  SECTION B  (label-noise floor vs ANALYTIC eigenfunctions)
    For calibration geometries (square / circle) the exact eigenfunctions are
    known.  We compare the *stored FEM label* against the analytic field under
    both the dataset's L-inf normalisation and an M-norm normalisation, and we
    also compare the *model prediction* against analytic.  This is the decisive
    number: if FEM-label-vs-analytic ~= the model's error, no architecture will
    help and the labels must be fixed.

Usage:
    python scripts/diagnose_data_floor.py --checkpoint path/to.ckpt \
        [--data_path data/gnot_dataset_5k.pkl] [--split val|train|test|all] \
        [--raw_h5 rf_cavity_*.h5] [--json out.json]
"""
import os
import sys
import json
import argparse
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

# repo root on path (this file lives in scripts/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.dataset import GNOTDataset, gnot_collate_fn
from src.training.lightning_module import GNOTLightning
from infer import _orthonormalize_np, _subspace_rel_l2, _near_degenerate_clusters

# First Dirichlet eigenvalue roots (Bessel zeros) for the unit disk.
_J0_1 = 2.404825557695773   # 1st zero of J0  -> monopole
_J1_1 = 3.831705970207512   # 1st zero of J1  -> dipole pair


# ── small numeric helpers ────────────────────────────────────────────────────

def _sign_agnostic_rel_l2(p, t, w=None):
    """min over global sign of weighted ||p-t|| / ||t||.  w: per-node weights."""
    if w is None:
        w = np.ones_like(t)
    den = np.sqrt(np.sum(w * t * t)) + 1e-12
    num_p = np.sqrt(np.sum(w * (p - t) ** 2))
    num_n = np.sqrt(np.sum(w * (p + t) ** 2))
    return float(min(num_p, num_n) / den)


def _subspace_rel_l2_w(E_basis, e_tgt, w=None):
    """Subspace projection error, optionally M-weighted.

    Scaling every row by sqrt(w) turns the plain Euclidean QR / projection in
    `infer._subspace_rel_l2` into the w-weighted inner-product version, because
    <a,b>_w == (sqrt(w)a) . (sqrt(w)b).
    """
    if w is None:
        Q = _orthonormalize_np(E_basis)
        return _subspace_rel_l2(Q, e_tgt)
    sw = np.sqrt(np.clip(w, 0.0, None)).reshape(-1, 1)
    Q = _orthonormalize_np(E_basis * sw)
    return _subspace_rel_l2(Q, (e_tgt.reshape(-1, 1) * sw).ravel())


def _boundary_node_indices(elements):
    """Indices of nodes on the mesh boundary (edges shared by exactly 1 tri)."""
    cnt = {}
    for tri in elements:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = (a, b) if a < b else (b, a)
            cnt[e] = cnt.get(e, 0) + 1
    bnd = {n for e, c in cnt.items() if c == 1 for n in e}
    return np.array(sorted(bnd), dtype=np.int64)


def _classify_shape(Xn, elements):
    """Conservatively classify a normalised geometry from its boundary signature.

    This is only a *fallback* — the authoritative source is the raw generator
    H5 `shape_type` attr (pass --raw_h5).  We therefore only declare a
    calibration shape on a tight, unambiguous signature and otherwise return
    'random' (random blobs must NOT be mislabelled).  Normalisation is
    isotropic (max|coord| == 1), so:
      circle  -> all boundary radii ~= 1            (cv < 3%)
      square  -> r(theta) ~= 1/max(|cos|,|sin|)     (axis-aligned [-1,1]^2)
      annulus -> boundary radii are tightly bimodal (clear inner ring)
    Returns one of: 'circle' | 'square' | 'annulus' | 'random'.
    """
    try:
        bnd = _boundary_node_indices(np.asarray(elements))
        P = Xn[bnd]
        r = np.linalg.norm(P, axis=1)
    except Exception:
        return 'random'
    if r.size < 12:
        return 'random'
    rmax, rmin = float(r.max()), float(r.min())
    cv = float(r.std() / (r.mean() + 1e-12))

    # circle: a single tight ring at radius 1
    if cv < 0.03 and rmin > 0.94 and rmax < 1.06:
        return 'circle'

    # square: boundary obeys the axis-aligned [-1,1]^2 law r=1/max(|cos|,|sin|)
    th = np.arctan2(P[:, 1], P[:, 0])
    sq_resid = float(np.median(
        np.abs(r - 1.0 / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th))))))
    if sq_resid < 0.05 and 1.30 < rmax < 1.50 and 0.95 < rmin < 1.05:
        return 'square'

    # annulus: tightly bimodal radii with a clear empty gap between rings
    rs = np.sort(r)
    gaps = np.diff(rs)
    gi = int(np.argmax(gaps))
    if gaps[gi] > 0.25:
        lo, hi = rs[:gi + 1], rs[gi + 1:]
        if (lo.size > 0.10 * r.size and hi.size > 0.10 * r.size
                and lo.mean() < 0.75 and hi.mean() > 0.85
                and lo.std() / (lo.mean() + 1e-12) < 0.06
                and hi.std() / (hi.mean() + 1e-12) < 0.06):
            return 'annulus'
    return 'random'


def _analytic_fields(shape, Xn):
    """Analytic Dirichlet eigenfields at normalised nodes Xn [N,2].

    Returns (mono [N], dipole_basis [N,2]) ordered by ascending frequency, or
    None if the shape has no closed form here.
    """
    x, y = Xn[:, 0], Xn[:, 1]
    if shape == 'square':
        # normalised square is [-1,1]^2  ->  u,v in [0,1]
        u, v = 0.5 * (x + 1.0), 0.5 * (y + 1.0)
        mono = np.sin(np.pi * u) * np.sin(np.pi * v)            # (1,1)
        da = np.sin(np.pi * u) * np.sin(2 * np.pi * v)          # (1,2)
        db = np.sin(2 * np.pi * u) * np.sin(np.pi * v)          # (2,1)
        return mono, np.stack([da, db], axis=1)
    if shape == 'circle':
        from scipy.special import jv
        rho = np.sqrt(x * x + y * y)
        th = np.arctan2(y, x)
        mono = jv(0, _J0_1 * rho)                               # J0
        rad1 = jv(1, _J1_1 * rho)
        return mono, np.stack([rad1 * np.cos(th), rad1 * np.sin(th)], axis=1)
    return None


def _shape_from_h5(raw_h5):
    """geom_id -> shape string, read from the raw generator H5 if available."""
    import h5py
    out = {}
    with h5py.File(raw_h5, 'r') as f:
        for k in f.keys():
            try:
                gid = int(k.split('_')[-1])
            except ValueError:
                continue
            st = f[k].attrs.get('shape_type', 'unknown')
            st = st.decode() if isinstance(st, bytes) else str(st)
            if st == 'calibration':            # generator convention
                st = 'square' if gid % 2 == 0 else 'circle'
            out[gid] = st
    return out


# ── Section A: model-error decomposition (from checkpoint) ────────────────────

def section_a(model, dataset, device, batch_size, deg_threshold):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    collate_fn=gnot_collate_fn)
    fs = getattr(model, 'freq_stats', None)
    seen = set()
    K = None
    acc = {'uni': [], 'w': [], 'sep_uni': [], 'sep_w': [],
           'deg_uni': [], 'deg_w': [], 'scale_ratio': [], 'norm_ratio': []}
    per_mode_uni, per_mode_w = None, None

    with torch.no_grad():
        for batch in dl:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            out = model(batch)
            P = out['field']                       # [B,N,K]
            T = batch['Y_field']                   # [B,N,K]
            M = batch['Mask']                      # [B,N]
            IF = batch['Input_funcs']              # [B,N,8]  (idx5 = node_area)
            gids = batch['geom_id'].squeeze(-1).cpu().numpy()
            ft = batch['Y_freq']                   # [B,K] normalised, ascending
            if fs:
                ft_ph = (ft * fs['std'] + fs['mean']).cpu().numpy()
            else:
                ft_ph = ft.cpu().numpy()

            B, N, k = P.shape
            if K is None:
                K = k
                per_mode_uni = [[] for _ in range(K)]
                per_mode_w = [[] for _ in range(K)]
            for i in range(B):
                gid = int(gids[i])
                if gid in seen:
                    continue
                seen.add(gid)
                m = M[i].cpu().numpy().astype(bool)
                Eh = P[i][m].cpu().numpy()                 # [Nv,K]
                Et = T[i][m].cpu().numpy()                 # [Nv,K]
                w = IF[i][m, 5].cpu().numpy().astype(np.float64)
                w = np.clip(w, 1e-8, None)

                for kk in range(K):
                    ru = _sign_agnostic_rel_l2(Eh[:, kk], Et[:, kk])
                    rw = _sign_agnostic_rel_l2(Eh[:, kk], Et[:, kk], w)
                    per_mode_uni[kk].append(ru)
                    per_mode_w[kk].append(rw)
                    acc['uni'].append(ru)
                    acc['w'].append(rw)

                clusters = _near_degenerate_clusters(ft_ph[i], deg_threshold)
                for cl in clusters:
                    if len(cl) == 1:
                        kk = cl[0]
                        acc['sep_uni'].append(
                            _sign_agnostic_rel_l2(Eh[:, kk], Et[:, kk]))
                        acc['sep_w'].append(
                            _sign_agnostic_rel_l2(Eh[:, kk], Et[:, kk], w))
                    else:
                        Esub = Eh[:, cl]
                        for kk in cl:
                            acc['deg_uni'].append(
                                _subspace_rel_l2_w(Esub, Et[:, kk]))
                            acc['deg_w'].append(
                                _subspace_rel_l2_w(Esub, Et[:, kk], w))

                acc['scale_ratio'].append(
                    float(np.abs(Eh).max() / (np.abs(Et).max() + 1e-12)))
                acc['norm_ratio'].append(
                    float(np.linalg.norm(Eh) / (np.linalg.norm(Et) + 1e-12)))

    def ms(v):
        return (float(np.mean(v)), float(np.std(v))) if v else (float('nan'), 0.0)

    rep = {
        'n_geoms': len(seen),
        'overall_uniform_rel_l2': ms(acc['uni']),
        'overall_areaweighted_rel_l2': ms(acc['w']),
        'per_mode_uniform': [ms(per_mode_uni[k]) for k in range(K or 0)],
        'per_mode_areaweighted': [ms(per_mode_w[k]) for k in range(K or 0)],
        'well_separated_uniform': ms(acc['sep_uni']),
        'well_separated_areaweighted': ms(acc['sep_w']),
        'near_degenerate_subspace_uniform': ms(acc['deg_uni']),
        'near_degenerate_subspace_areaweighted': ms(acc['deg_w']),
        'pred_target_scale_ratio': ms(acc['scale_ratio']),
        'pred_target_norm_ratio': ms(acc['norm_ratio']),
    }
    return rep


# ── Section B: analytic label-noise floor ────────────────────────────────────

def section_b(raw_pkl_path, active_geoms, shape_map):
    if not str(raw_pkl_path).endswith('.pkl') or not os.path.exists(raw_pkl_path):
        return {'status': 'skipped: analytic floor needs the .pkl '
                          '(with elements) at --data_path'}
    with open(raw_pkl_path, 'rb') as f:
        raw = pickle.load(f)
    gpool = raw['geometry_pool']
    by_geom = {}
    for s in raw['samples']:
        by_geom.setdefault(int(s['geom_id']), []).append(s)

    buckets = {}  # shape -> list of dicts
    active = set(int(g) for g in active_geoms)
    for gid, geom in gpool.items():
        gid = int(gid)
        if gid not in active:
            continue
        shape = shape_map.get(gid)
        if shape is None:
            shape = _classify_shape(np.asarray(geom['X']),
                                    geom.get('elements'))
        if shape not in ('square', 'circle'):
            continue
        samples = sorted(by_geom.get(gid, []), key=lambda s: float(s['Theta'][1]))
        if len(samples) < 3:
            continue
        Xn = np.asarray(geom['X'], dtype=np.float64)         # [N,2] normalised
        Et = np.stack([np.asarray(s['Y']).reshape(-1) for s in samples[:3]],
                      axis=1).astype(np.float64)             # [N,3] asc-freq
        af = _analytic_fields(shape, Xn)
        if af is None or Xn.shape[0] != Et.shape[0]:
            continue
        mono_a, dip_a = af

        # node_area weights for the M-norm variant (Input_funcs col 5)
        w = np.asarray(geom['Input_funcs'])[:, 5].astype(np.float64)
        w = np.clip(w, 1e-8, None)

        rec = {
            'monopole_Linf': _sign_agnostic_rel_l2(Et[:, 0], mono_a),
            'monopole_Mnorm': _sign_agnostic_rel_l2(Et[:, 0], mono_a, w),
            'dipole_subspace_Linf': float(np.mean([
                _subspace_rel_l2_w(dip_a, Et[:, 1]),
                _subspace_rel_l2_w(dip_a, Et[:, 2])])),
            'dipole_subspace_Mnorm': float(np.mean([
                _subspace_rel_l2_w(dip_a, Et[:, 1], w),
                _subspace_rel_l2_w(dip_a, Et[:, 2], w)])),
        }
        buckets.setdefault(shape, []).append(rec)

    if not buckets:
        return {'status': 'no square/circle calibration geometries in this '
                          'split (pass --raw_h5 or use a calibration dataset)'}

    summary = {}
    for shape, recs in buckets.items():
        keys = recs[0].keys()
        summary[shape] = {'n': len(recs)}
        for kk in keys:
            vals = [r[kk] for r in recs]
            summary[shape][kk] = (float(np.mean(vals)), float(np.std(vals)))
    return {'status': 'ok', 'per_shape': summary}


# ── reporting ────────────────────────────────────────────────────────────────

def _fmt(ms):
    return f"{ms[0]:.4f} ± {ms[1]:.4f}"


def print_report(split, a, b):
    print("\n" + "=" * 72)
    print(f"  DATA / LABEL-NOISE-FLOOR DIAGNOSTIC   (split = {split})")
    print("=" * 72)
    print(f"\n[A] MODEL ERROR DECOMPOSITION   ({a['n_geoms']} geometries)")
    print(f"  overall rel-L2  uniform        : {_fmt(a['overall_uniform_rel_l2'])}")
    print(f"  overall rel-L2  area-weighted  : {_fmt(a['overall_areaweighted_rel_l2'])}")
    print("   ^ if area-weighted << uniform -> L-inf/node-uniform labelling is "
          "inflating\n     the error; an M-norm relabel is the high-leverage data fix.")
    for k, (u, w) in enumerate(zip(a['per_mode_uniform'],
                                   a['per_mode_areaweighted'])):
        print(f"  mode {k}: uniform {_fmt(u)}   area-weighted {_fmt(w)}")
    print(f"  well-separated   : uniform {_fmt(a['well_separated_uniform'])}"
          f"   area-w {_fmt(a['well_separated_areaweighted'])}")
    print(f"  near-degenerate  : subspace-uniform "
          f"{_fmt(a['near_degenerate_subspace_uniform'])}"
          f"   subspace-area-w {_fmt(a['near_degenerate_subspace_areaweighted'])}")
    print(f"  pred/target scale ratio        : "
          f"{_fmt(a['pred_target_scale_ratio'])}   (≈1 healthy; ≪1 = cold-start)")
    print(f"  pred/target L2-norm ratio      : {_fmt(a['pred_target_norm_ratio'])}")

    print("\n[B] ANALYTIC LABEL-NOISE FLOOR  (FEM label vs exact eigenfunction)")
    if b.get('status') != 'ok':
        print(f"  {b.get('status')}")
    else:
        for shape, s in b['per_shape'].items():
            print(f"  {shape}  (n={s['n']})")
            print(f"    monopole : L-inf {_fmt(s['monopole_Linf'])}"
                  f"   M-norm {_fmt(s['monopole_Mnorm'])}")
            print(f"    dipole   : L-inf {_fmt(s['dipole_subspace_Linf'])}"
                  f"   M-norm {_fmt(s['dipole_subspace_Mnorm'])}")
        print("   ^ Decisive: if this floor ≈ the model's rel-L2 in [A], ~0.10 is "
              "LABEL-bound\n     (a data problem) and no architecture change will "
              "pass it.  If M-norm\n     floor ≪ L-inf floor, switch target "
              "normalisation to M-norm.")
    print("=" * 72 + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data_path', default='data/gnot_dataset_5k.pkl')
    ap.add_argument('--split', default='val',
                    choices=['train', 'val', 'test', 'all'])
    ap.add_argument('--raw_h5', default=None,
                    help='optional raw generator .h5 for authoritative shape_type')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--deg_threshold', type=float, default=0.05)
    ap.add_argument('--json', default=None, help='optional path to dump report')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint: {args.checkpoint}")
    model = GNOTLightning.load_from_checkpoint(args.checkpoint)
    model.eval().to(device)

    shape_map = {}
    if args.raw_h5 and os.path.exists(args.raw_h5):
        shape_map = _shape_from_h5(args.raw_h5)
        print(f"Loaded shape_type for {len(shape_map)} geometries from "
              f"{args.raw_h5}")

    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    full_report = {}
    for sp in splits:
        ds = GNOTDataset(args.data_path, split=sp)
        if getattr(ds, 'stats', None):
            model.freq_stats = ds.stats
        a = section_a(model, ds, device, args.batch_size, args.deg_threshold)
        b = section_b(args.data_path, ds.active_geoms, shape_map)
        print_report(sp, a, b)
        full_report[sp] = {'section_a': a, 'section_b': b}

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(full_report, f, indent=2)
        print(f"Report written to {args.json}")


if __name__ == '__main__':
    main()
