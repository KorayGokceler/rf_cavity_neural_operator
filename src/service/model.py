"""Model side of the prediction service: checkpoint loading, pure-model prediction of one cavity with the
field3d model (EigenspaceOperatorField: HCurl(p2) Ritz modes on the curved mesh), figures of merit from
the predicted high-order fields, label-free residual, and display fields in physical units — shared by
scripts/predict_geometry.py and src/service/api.py.

Prediction (ModelService.predict; the label-free path of scripts/eval_fidelity.py):
    curved mesh (geometry.load_mesh_src) → field_labels.model_normalisation → highorder.field_operators(p2)
    → field_item / field_collate → model → Ritz frequencies out['freq'] (GHz with freq_stats = (0, 1)) and
    HCurl(p2) DOFs (the same vectors in physical and normalised coordinates)
    → figures of merit: highorder.mode_qoi on the p2 fields (the label QoI code; qoi_source 'p2_model')
    → η_k = ‖K u − λ M u‖_{D⁻¹} / λ with the field operators' K, M
    → display: Whitney N0 interpolant on the linear mesh (datasets.whitney_dofs, as the dataset viewer).

Display convention (all fields at stored energy U = 1 J, standing wave):
    E(t) = E · cos φ,   H(t) = H · sin φ      (the sign of the curl relation is folded into H)
"""
import collections
import math
import os
import threading
import time

import numpy as np
import torch

from src.training.checkpoint import resolve_checkpoint
from src.data.dataset_converter_3d import extract_geometry_3d, h5_edges_to_canonical, qoi_operators_of
from src.training.lightning_module import CavityLightning

EPS0, MU0 = 8.8541878128e-12, 4e-7 * math.pi
QOI_SHOW = ('Q0', 'G_ohm', 'R_over_Q_ohm', 'R_sh_ohm', 'T_transit', 'Epk_Eacc', 'Bpk_Eacc_mT_per_MVm')
VOLTAGE_QOI = ('R_over_Q_ohm', 'R_sh_ohm', 'T_transit', 'Epk_Eacc', 'Bpk_Eacc_mT_per_MVm')
PEAK_QOI = ('Epk_Eacc', 'Bpk_Eacc_mT_per_MVm')
RQ_FLOOR_OHM = 1e-6        # below: no accelerating field on the axis → voltage figures undefined (shown '—')


def checkpoint_model_type(path):
    """hyper_parameters['model_type'] of a .ckpt (None when absent) without building the model."""
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError:                                  # not a zip checkpoint: no mmap
        ck = torch.load(path, map_location='cpu', weights_only=False)
    hp = (ck.get('hyper_parameters') or {}) if isinstance(ck, dict) else {}
    return hp.get('model_type')


def load_model(checkpoint, device='cpu'):
    """(lm, feature_indices) of a field3d .ckpt or training dir (the best checkpoint there); other model
    types (the removed N0 model 'eigenspace3d', 2D models) are rejected with a clear error."""
    path = resolve_checkpoint(checkpoint)
    mt = checkpoint_model_type(path)
    if mt != 'field3d':
        raise ValueError(f"{os.path.basename(str(path))}: model_type {mt!r}, the service runs only 'field3d' "
                         "checkpoints (EigenspaceOperatorField; the N0 model eigenspace3d was removed)")
    lm = CavityLightning.load_from_checkpoint(path, map_location=device).eval()
    lm.freq_stats = {'mean': 0.0, 'std': 1.0}            # out['freq'] is then f in GHz
    return lm, dict(lm.hparams.get('data_cfg') or {}).get('feature_indices')


def untrained_model(n_modes=6, seed=0):
    """A small randomly initialised field3d model — for UI / API development without a checkpoint only."""
    torch.manual_seed(seed)
    lm = CavityLightning(val_dim=9, hidden_dim=32, n_heads=2, n_basis=16, num_field_modes=n_modes, rff_dim=16,
                         eigenspace_kwargs={'n_layers': 2, 'grad_checkpoint': False}).eval()
    lm.freq_stats = {'mean': 0.0, 'std': 1.0}
    return lm


def degenerate_mask(f, rel=1e-3):
    """Modes whose frequency is within rel of a neighbour: their individual field (and so their
    per-mode QoI) is any rotation inside the pair — only the pair's span is defined."""
    f = np.asarray(f, float)
    d = np.zeros(len(f), bool)
    close = np.abs(np.diff(f)) < rel * np.abs(f[1:])
    d[1:] |= close
    d[:-1] |= close
    return d


def residual(K, M, U, lam, free=None):
    """Label-free error indicator η_k = ‖K u − λ M u‖_{D⁻¹} / λ (D = diag M) per mode."""
    R = K @ U - (M @ U) * lam[None, :]
    d = M.diagonal()[:, None]
    if free is not None:
        R, d = R[free], d[free]
    return np.sqrt((R ** 2 / d).sum(0)) / np.abs(lam)


@torch.no_grad()
def forward(lm, ops, feature_indices=None, shape_type='', device='cpu'):
    """Field model on one geometry's field_operators: (U [n_dof, K] HCurl DOFs, f [K] GHz, λ [K] normalised
    Ritz values, model seconds)."""
    from src.data.field_dataset import field_collate, field_item
    batch = field_collate([field_item(ops, feature_indices=feature_indices, shape_type=shape_type)])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    lm.to(device)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = lm.model(batch)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    t = time.perf_counter() - t0
    n = int(ops['n_dof'])
    return (out['field'][0, :n].double().cpu().numpy(), out['freq'][0].double().cpu().numpy(),
            out['eigenvalues'][0].double().cpu().numpy(), t)


def field_qoi(mesh, U, f_ghz, order, family='', params=None):
    """Figures of merit of HCurl(order) DOF vectors U [n, K] on the curved mesh at f_ghz, from the high-order
    fields (highorder.mode_qoi with the labels' beam axis and accelerating length: field_labels._beam_axis):
    QOI_SHOW rows per mode (None when undefined).  A geometry left with sharp re-entrant edges
    (params n_sharp_edges > 0) gets no surface peaks, as its labels."""
    from ngsolve import GridFunction, HCurl

    from src.data_gen import field_labels as fl
    from src.data_gen import highorder as ho
    fes = HCurl(mesh, order=int(order), dirichlet='wall')
    gfs = []
    for k in range(U.shape[1]):
        g = GridFunction(fes)
        g.vec.FV().NumPy()[:] = U[:, k]
        gfs.append(g)
    q = ho.mode_qoi(mesh, gfs, np.asarray(f_ghz, float) * 1e9, *fl._beam_axis(family, mesh, params))
    sharp = float((params or {}).get('n_sharp_edges', 0) or 0) > 0
    return [{n: (None if sharp and n in PEAK_QOI else _num(q[n][k])) for n in QOI_SHOW} for k in range(U.shape[1])]


def override_qoi(sol, labels):
    """Replace the figures of merit of a solution's mode rows by better ones (stored p3 labels, or the
    predicted p2 fields' mode_qoi) — the N0 display fields' own values are only an approximation — and
    pick the accelerating mode again (largest R/Q among the non-degenerate modes)."""
    for row, lab in zip(sol['modes'], labels, strict=False):
        row.update({n: lab[n] for n in QOI_SHOW if n in lab})        # finite or None
        if not (row.get('R_over_Q_ohm') or 0.0) > RQ_FLOOR_OHM:      # no field on the axis
            row.update({n: None for n in VOLTAGE_QOI})
    rq = [-1.0 if m['degenerate'] or m['R_over_Q_ohm'] is None else m['R_over_Q_ohm'] for m in sol['modes']]
    acc = int(np.argmax(rq)) if max(rq, default=-1.0) > 0 else -1
    for k, m in enumerate(sol['modes']):
        m['accelerating'] = k == acc
    return sol


def qoi(geom, M, U, f_ghz, n_axis=401, axis=None, Rs=None, sigma=5.8e7, beta=1.0, L_acc=None,
        convention='linac', return_ops=False):
    """Figures of merit of DOFs U at f_ghz (U = 1 J; family beam axis unless axis = dict of
    axis_xy / axis_dir / axis_point)."""
    from src.qoi import qoi_from_dofs
    ops = qoi_operators_of(geom, M=M, n_axis=n_axis, **(axis or {}))
    q = qoi_from_dofs(ops, U, np.asarray(f_ghz) * 1e9, Rs=Rs, sigma=sigma, beta=beta, L_acc=L_acc,
                      convention=convention)
    return (q, ops) if return_ops else q


def physical_fields(geom, M, U, f_ghz):
    """Nodal (vertex-averaged) display fields at U = 1 J, [Nv,3,K] each: (E [V/m], H [A/m]) with
    E(t) = E cos φ and H(t) = H sin φ."""
    from src.viz.nedelec import vertex_field
    X, tets, edges, s = geom['X'], geom['tets'], geom['edges'], float(geom['scale'])
    w = 2 * np.pi * np.asarray(f_ghz, float) * 1e9                                    # [K]
    m = np.einsum('ek,ek->k', U, M @ U)
    E = vertex_field(X, tets, edges, U)
    C = vertex_field(X, tets, edges, U, curl=True)                                    # curl E (normalised)
    a = np.sqrt(1.0 / (0.5 * EPS0 * s ** 3 * m))                                      # → V/m
    return a * E, -(a / (w * MU0 * s)) * C                                            # Faraday


# ─────────────────────────── solutions (model, FE, difference) ────────

def axis_values(geom, M, U, f_ghz, ops):
    """Field component along the beam axis at U = 1 J for every mode: [K, P] V/m (NaN outside Ω)."""
    s = float(geom['scale'])
    m = np.einsum('ek,ek->k', U, M @ U)
    v = np.asarray(ops['Az'] @ U)                                                    # [P, K]
    a = np.sqrt(1.0 / (0.5 * EPS0 * s ** 3 * m))
    out = (v * a[None, :]).T
    out[:, np.asarray(ops['q']) <= 0] = np.nan
    return out


def build_solution(geom, M, U, f_ghz, source, eta=None, qoi_kw=None, ops=None):
    """Everything the UI shows for one set of modes U [Ne,K] at f_ghz [K] (source 'model' | 'fe'):
    public 'modes' rows + private display arrays (_E, _H [Nv,3,K], _axis [K,P])."""
    U = np.asarray(U, float)
    f = np.asarray(f_ghz, float)
    if ops is None:
        q, ops = qoi(geom, M, U, f, return_ops=True, **(qoi_kw or {}))
    else:
        from src.qoi import qoi_from_dofs
        q = qoi_from_dofs(ops, U, f * 1e9, **{k: v for k, v in (qoi_kw or {}).items() if k != 'axis'})
    q = {k: np.asarray(v, float).copy() for k, v in q.items()}
    no_v = ~(np.asarray(q['R_over_Q_ohm']) > RQ_FLOOR_OHM)          # no field along the beam axis
    for n in VOLTAGE_QOI:
        q[n][no_v] = np.nan
    deg = degenerate_mask(f)
    rq = np.asarray(q['R_over_Q_ohm'], float)
    ok_rq = np.where(deg | ~np.isfinite(rq), -np.inf, rq)
    accel = int(np.argmax(ok_rq)) if np.isfinite(ok_rq).any() else -1
    modes = []
    for k in range(len(f)):
        row = {'mode': k, 'f_GHz': float(f[k]), 'degenerate': bool(deg[k]), 'accelerating': k == accel,
               'eta': None if eta is None else float(eta[k])}
        row.update({n: _num(q[n][k]) for n in QOI_SHOW})
        modes.append(row)
    E, H = physical_fields(geom, M, U, f)
    return {'source': source, 'n_edges': int(len(geom['edges'])), 'modes': modes,
            'axis_length_mm': float(ops['L_axis'] * geom['scale'] * 1e3),
            'L_acc_mm': float(ops['L_acc'] * geom['scale'] * 1e3),
            '_geom': geom, '_M': M, '_U': U, '_f': f, '_E': E, '_H': H, '_ops': ops,
            '_axis': axis_values(geom, M, U, f, ops),
            '_zeta_mm': np.asarray(ops['zeta']) * geom['scale'] * 1e3}


def align(Up, Ut, M, f_true, rel=1e-3):
    """Predicted modes matched to the FE ones (same index): isolated mode → sign flip; inside a
    degenerate cluster of the FE spectrum → M-projection of t_k on the predicted cluster span,
    rescaled to ‖t_k‖_M. Returns (aligned [Ne,K], rel-L2 [K] in the M-norm)."""
    K = min(Up.shape[1], Ut.shape[1])
    P, T = Up[:, :K] / np.sqrt(np.einsum('ek,ek->k', Up[:, :K], M @ Up[:, :K])), Ut[:, :K]
    MT = M @ T
    tt = np.einsum('ek,ek->k', T, MT)
    C = P.T @ MT
    out = np.zeros_like(T)
    f = np.asarray(f_true[:K], float)
    k = 0
    while k < K:
        j = k
        while j + 1 < K and abs(f[j + 1] - f[j]) < rel * f[j]:
            j += 1
        idx = np.arange(k, j + 1)
        if len(idx) == 1:
            out[:, k] = (np.sign(C[k, k]) or 1.0) * P[:, k] * np.sqrt(tt[k])
        else:
            Cc = C[np.ix_(idx, idx)]
            proj = P[:, idx] @ Cc
            out[:, idx] = proj * np.sqrt(tt[idx] / np.maximum((Cc ** 2).sum(0), 1e-300))[None, :]
        k = j + 1
    D = out - T
    rl = np.sqrt(np.einsum('ek,ek->k', D, M @ D) / tt)
    return out, rl


def compare(pred, true, rel=1e-3):
    """(rows, difference solution) for a model solution vs the FE one on the same mesh."""
    geom, M = true['_geom'], true['_M']
    Ua, rl = align(pred['_U'], true['_U'], M, true['_f'], rel)
    K = Ua.shape[1]
    pa = build_solution(geom, M, Ua, pred['_f'][:K], 'model', ops=true['_ops'])
    rows = []
    for k in range(K):
        tr, pr = true['modes'][k], pa['modes'][k]
        row = {'mode': k, 'f_fe': tr['f_GHz'], 'f_model': pr['f_GHz'], 'f_rel_err': pr['f_GHz'] / tr['f_GHz'] - 1,
               'rel_l2': float(rl[k]), 'degenerate': tr['degenerate'], 'accelerating': tr['accelerating'],
               'eta': pred['modes'][k]['eta'], 'qoi': {}}
        for n in QOI_SHOW:
            a, b = tr[n], pr[n]
            row['qoi'][n] = {'fe': a, 'model': b,
                             'rel_err': (None if a in (None, 0) or b is None else b / a - 1)}
        rows.append(row)
    diff = dict(pa, source='diff', modes=[dict(m, f_GHz=true['modes'][m['mode']]['f_GHz']) for m in pa['modes']],
                _E=pa['_E'] - true['_E'][:, :, :K], _H=pa['_H'] - true['_H'][:, :, :K],
                _axis=pa['_axis'] - true['_axis'][:K], _f=true['_f'][:K])
    return rows, diff


# ─────────────────────────── the service ───────────────────────────────

class ModelService:
    """One loaded field3d model + per-geometry prediction. Thread-safe (one prediction at a time; the NGSolve
    parts under src.service.datasets._NG_LOCK).  The curved mesh and its field operators (~4 s at 10k tets)
    are kept for the last `cache_size` geometries (RFCAV_FIELD_CACHE, default 2)."""

    def __init__(self, checkpoint=None, device=None, optional=False, cache_size=None):
        """optional: a checkpoint that does not load falls back to the untrained model (load_error says why)
        instead of failing — for launchers that pick a checkpoint automatically."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_error = None
        if checkpoint:
            try:
                self.lm, self.feature_indices = load_model(checkpoint, self.device)
                self.untrained = False
            except Exception as e:                           # noqa: BLE001 — reported in info()
                if not optional:
                    raise
                self.load_error, checkpoint = f"{type(e).__name__}: {e}", None
        if not checkpoint:
            self.lm, self.feature_indices, self.untrained = untrained_model(), None, True
        self.lm.to(self.device)
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.n_modes = int(self.lm.hparams.get('num_field_modes', 6))
        self.n_params = int(sum(p.numel() for p in self.lm.parameters()))
        self._lock = threading.Lock()
        self.cache_size = int(cache_size if cache_size is not None else os.environ.get('RFCAV_FIELD_CACHE') or 2)
        self._ops = collections.OrderedDict()                # key → (mesh, ops)

    def load(self, checkpoint):
        """Switch to another checkpoint (e.g. a run just trained from the UI); on failure the current
        model stays."""
        path = resolve_checkpoint(checkpoint)
        lm, fi = load_model(path, self.device)
        lm.to(self.device)
        with self._lock:
            self.lm, self.feature_indices = lm, fi
            self.untrained, self.load_error, self.checkpoint = False, None, str(path)
            self.n_modes = int(lm.hparams.get('num_field_modes', 6))
            self.n_params = int(sum(p.numel() for p in lm.parameters()))
        return self.info()

    def info(self):
        return {'n_modes': self.n_modes, 'n_params': self.n_params, 'device': self.device,
                'untrained': self.untrained, 'checkpoint': self.checkpoint and self.checkpoint.split('/')[-1],
                'load_error': self.load_error, 'model_type': 'field3d',
                'run': self.checkpoint and os.path.basename(os.path.dirname(os.path.abspath(self.checkpoint)))}

    def geometry(self, nodes, tets, family='', params=None):
        """(converter geometry dict with features + operators, CSR mass) of a linear mesh [m] — the display
        geometry (N0 fields, cut planes, input-feature views); params: the family's geom_params (beam axis and
        accelerating length, src.qoi.operators)."""
        geom, M = extract_geometry_3d(nodes, tets)
        geom.update(shape_type=family or '', params=params)
        return geom, (M.tocsr() if hasattr(M, 'tocsr') else M)

    def _operators(self, mesh_src, key, order, t):
        """(curved mesh, field_operators) of a mesh_src, cached per key (caller holds the locks)."""
        from src.data_gen import field_labels as fl
        from src.data_gen import highorder as ho
        from src.service.geometry import load_mesh_src
        if key is not None and key in self._ops:
            self._ops.move_to_end(key)
            t.update(mesh=0.0, operators=0.0)
            return self._ops[key]
        t0 = time.perf_counter()
        mesh = load_mesh_src(mesh_src)
        t['mesh'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        center, scale = fl.model_normalisation(mesh)
        ops = ho.field_operators(mesh, order=int(order), scale=scale, center=center)
        t['operators'] = time.perf_counter() - t0
        if key is not None and self.cache_size > 0:
            self._ops[key] = (mesh, ops)
            while len(self._ops) > self.cache_size:
                self._ops.popitem(last=False)
        return mesh, ops

    def predict(self, mesh_src, geom, M, family='', params=None, key=None):
        """Model solution of one geometry (build_solution dict + 'time_s', 'qoi_source', 'n_dof', 'n_tets'):
        mesh_src its curved mesh (geometry.curved_from_cad / the field H5), geom / M its display geometry
        (the linear mesh of that curved mesh: ModelService.geometry or the dataset item's), params the
        family's geom_params, key a cache key (the geometry id)."""
        from src.data.sparse import to_csr
        from src.service.datasets import _NG_LOCK, whitney_dofs
        t = {}
        order = int(mesh_src.get('model_order') or 2)
        with _NG_LOCK, self._lock:
            mesh, ops = self._operators(mesh_src, key, order, t)
            U, f, lam, t['model'] = forward(self.lm, ops, self.feature_indices, family or '', self.device)
            t0 = time.perf_counter()
            labels = field_qoi(mesh, U, f, order, family, params)
            t['qoi'] = time.perf_counter() - t0
            t0 = time.perf_counter()
            nodes, _, edges_ng, V = whitney_dofs(mesh, U, order)
            n_tets = int(mesh.ne)
        n = int(ops['n_dof'])
        Kf, Mf = to_csr(ops['K'], (n, n)), to_csr(ops['M'], (n, n))
        Un = U / np.sqrt(np.maximum(np.einsum('ik,ik->k', U, Mf @ U), 1e-300))
        eta = residual(Kf, Mf, Un, lam, ~np.asarray(ops['bnd'], bool))
        X = np.asarray(geom['X'], float) * float(geom['scale']) + np.asarray(geom['center']).reshape(1, 3)
        if nodes.shape != X.shape or not np.allclose(nodes, X, rtol=0, atol=1e-5 * float(geom['scale'])):
            raise ValueError("the display geometry is not the linear mesh of the curved mesh the model ran on")
        ne = len(geom['edges'])
        rows, sign = h5_edges_to_canonical(edges_ng, geom['n_nodes'], geom['edges'])
        Ud = np.zeros((ne, V.shape[1]))
        Ud[rows] = sign[:, None] * V
        sol = build_solution(geom, to_csr(M, (ne, ne)), Ud, f, 'model', eta=eta)
        override_qoi(sol, labels)
        t['display'] = time.perf_counter() - t0
        sol.update(qoi_source='p2_model', n_dof=n, n_tets=n_tets, time_s=t)
        return sol

    # ── display data (any solution: model, FE, difference) ───────────
    @staticmethod
    def to_mm(geom, X):
        return (np.asarray(X) * geom['scale'] + np.asarray(geom['center']).reshape(1, 3)) * 1e3

    @staticmethod
    def axis_profile(sol, k):
        """(axial coordinate [mm] relative to the centroid, field along the axis [V/m]) of mode k."""
        return sol['_zeta_mm'], sol['_axis'][k]

    def plane(self, sol, k, axis='y', pos_mm=None, res=121):
        """Cut plane: (points [P,3] mm, triangles [T,3], E [P,3], H [P,3]) of mode k."""
        pts, tri, (E, H) = self.plane_values(sol['_geom'], [sol['_E'][:, :, k], sol['_H'][:, :, k]],
                                             axis, pos_mm, res)
        return pts, tri, E, H

    def plane_values(self, geom, nodal, axis='y', pos_mm=None, res=121):
        """Vertex arrays `nodal` (list of [Nv,…]) interpolated on a cut plane (points mm, triangles, values)."""
        from src.viz.nedelec import interp_vertex, locate_grid
        ia = 'xyz'.index(axis)
        c, s = float(np.asarray(geom['center']).reshape(3)[ia]), float(geom['scale'])
        off = 0.0 if pos_mm is None else (pos_mm * 1e-3 - c) / s
        X = np.asarray(geom['X'], float)
        lo, hi = X.min(0), X.max(0)
        d = np.maximum(hi - lo, 1e-12) / (res - 1)
        lo[ia], d[ia] = off, 1.0
        dims = np.full(3, res)
        dims[ia] = 1                                     # flat index u + res·v: the (v, u) grid below
        tid, bary = locate_grid(X, geom['tets'], lo, d, dims)
        zz, yy, xx = np.meshgrid(*(lo[i] + d[i] * np.arange(dims[i]) for i in (2, 1, 0)), indexing='ij')
        pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1)
        inside = (tid >= 0).reshape(res, res)
        idx = -np.ones(inside.shape, np.int64)
        idx[inside] = np.arange(int(inside.sum()))
        a, b, cc, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
        full = (a >= 0) & (b >= 0) & (cc >= 0) & (d >= 0)
        tri = np.concatenate([np.stack([a[full], b[full], d[full]], 1), np.stack([a[full], d[full], cc[full]], 1)])
        sel = tid >= 0
        vals = [interp_vertex(geom['tets'], tid[sel], bary[sel], np.asarray(v)) for v in nodal]
        return self.to_mm(geom, pts[sel]), tri, vals

    def volume(self, sol, k, n_points=80_000):
        """Mode k on a regular 3D grid (volume_values): dict(dims, origin_mm, spacing_mm, inside, E, H)."""
        out = self.volume_values(sol['_geom'], [sol['_E'][:, :, k], sol['_H'][:, :, k]], n_points)
        out['E'], out['H'] = out.pop('values')
        return out

    def volume_values(self, geom, nodal, n_points=80_000):
        """Vertex arrays `nodal` (list of [Nv,3]) sampled on a regular grid of about n_points over the mesh
        bounding box, in vtkImageData order (x fastest, then y, then z); 0 outside the cavity:
        dict(dims [3], origin_mm [3], spacing_mm, inside [N] bool, values [list of [N,3]])."""
        from src.viz.nedelec import interp_vertex, locate_grid
        X = np.asarray(geom['X'], float)
        lo, hi = X.min(0), X.max(0)
        ext = np.maximum(hi - lo, 1e-12)
        d = float((np.prod(ext) / max(int(n_points), 8)) ** (1 / 3))
        dims = np.maximum(2, np.ceil(ext / d).astype(int) + 1)
        tid, bary = locate_grid(X, geom['tets'], lo, d, dims)
        vals = [np.nan_to_num(interp_vertex(geom['tets'], tid, bary, np.asarray(v, float))) for v in nodal]
        return {'dims': [int(x) for x in dims], 'origin_mm': self.to_mm(geom, lo[None])[0].tolist(),
                'spacing_mm': d * float(geom['scale']) * 1e3, 'inside': tid >= 0, 'values': vals}

    @staticmethod
    def surface_fields(sol, vid, k):
        """E, H [Nb,3] at the display-surface vertices."""
        return sol['_E'][vid, :, k], sol['_H'][vid, :, k]


def _num(v):
    v = float(v)
    return v if math.isfinite(v) else None
