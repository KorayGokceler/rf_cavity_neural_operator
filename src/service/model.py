"""Model side of the prediction service: checkpoint loading, pure-model prediction of one mesh
(frequencies, Ritz fields, figures of merit, label-free residual), and display fields in
physical units — shared by scripts/predict_geometry.py and src/service/api.py.

Display convention (all fields at stored energy U = 1 J, standing wave):
    E(t) = E · cos φ,   H(t) = H · sin φ      (the sign of the curl relation is folded into H)
"""
import math
import os
import threading
import time

import numpy as np
import torch

from infer import resolve_checkpoint
from src.data.dataset_3d import item_from_geometry, maxwell3d_collate
from src.data.dataset_converter_3d import extract_geometry_3d, qoi_operators_of
from src.training.lightning_module import GNOTLightning

EPS0, MU0 = 8.8541878128e-12, 4e-7 * math.pi
QOI_SHOW = ('Q0', 'G_ohm', 'R_over_Q_ohm', 'R_sh_ohm', 'T_transit', 'Epk_Eacc', 'Bpk_Eacc_mT_per_MVm')
VOLTAGE_QOI = ('R_over_Q_ohm', 'R_sh_ohm', 'T_transit', 'Epk_Eacc', 'Bpk_Eacc_mT_per_MVm')
RQ_FLOOR_OHM = 1e-6        # below: no accelerating field on the axis → voltage figures undefined (shown '—')


def load_model(checkpoint, device='cpu', field=None):
    """(lm, field, feature_indices). The field is the checkpoint's training field (data_cfg, recorded
    by train.py); `field` overrides it and is required for checkpoints that do not record one."""
    lm = GNOTLightning.load_from_checkpoint(resolve_checkpoint(checkpoint), map_location=device).eval()
    lm.freq_stats = {'mean': 0.0, 'std': 1.0}            # physics_freq: out['freq'] is then f in GHz
    dc = dict(lm.hparams.get('data_cfg') or {})
    trained = dc.get('field')
    if field and trained and field.upper() != str(trained).upper():
        raise ValueError(f"checkpoint was trained on field {trained}, requested field {field}")
    field = field or trained
    if not field:
        raise ValueError("checkpoint does not record its training field: pass field='E'|'H'")
    return lm, str(field).upper(), dc.get('feature_indices')


def untrained_model(field='E', n_modes=6, seed=0):
    """A small randomly initialised model — for UI / API development without a checkpoint only."""
    torch.manual_seed(seed)
    lm = GNOTLightning(val_dim=9, grid_dim=3, hidden_dim=32, n_heads=2, n_basis=16, num_field_modes=n_modes,
                       rff_dim=16, model_type='eigenspace3d', physics_freq=True,
                       eigenspace_kwargs={'n_layers': 2}, data_cfg={'field': field}).eval()
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
def forward(lm, geom, feature_indices=None, device='cpu', warmup=True):
    """(U [Ne,K] Ritz fields, f [K] GHz, λ [K] normalised eigenvalues, model seconds)."""
    item = item_from_geometry(geom, feature_indices=feature_indices)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in maxwell3d_collate([item]).items()}
    lm.to(device)
    if warmup:                                           # first call: allocator / sparse setup / CG warm-up
        lm.model(batch)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = lm.model(batch)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    t = time.perf_counter() - t0
    ne = len(geom['edges'])
    return (out['field'][0, :ne].double().cpu().numpy(), out['freq'][0].double().cpu().numpy(),
            out['eigenvalues'][0].double().cpu().numpy(), t)


def qoi(geom, M, U, f_ghz, field, n_axis=401, axis=None, Rs=None, sigma=5.8e7, beta=1.0, L_acc=None,
        convention='linac', return_ops=False):
    """Figures of merit of DOFs U at f_ghz (U = 1 J; family beam axis unless axis = dict of
    axis_xy / axis_dir / axis_point)."""
    from src.qoi import qoi_from_dofs
    ops = qoi_operators_of(geom, field, M=M, n_axis=n_axis, **(axis or {}))
    q = qoi_from_dofs(ops, U, np.asarray(f_ghz) * 1e9, Rs=Rs, sigma=sigma, beta=beta, L_acc=L_acc,
                      convention=convention)
    return (q, ops) if return_ops else q


def physical_fields(geom, M, U, f_ghz, field):
    """Nodal (vertex-averaged) display fields at U = 1 J, [Nv,3,K] each: (E [V/m], H [A/m]) with
    E(t) = E cos φ and H(t) = H sin φ."""
    from src.viz.nedelec import vertex_field
    X, tets, edges, s = geom['X'], geom['tets'], geom['edges'], float(geom['scale'])
    w = 2 * np.pi * np.asarray(f_ghz, float) * 1e9                                    # [K]
    m = np.einsum('ek,ek->k', U, M @ U)
    P = vertex_field(X, tets, edges, U)                                               # primary
    C = vertex_field(X, tets, edges, U, curl=True)                                    # its curl (normalised)
    if field == 'E':
        a = np.sqrt(1.0 / (0.5 * EPS0 * s ** 3 * m))                                  # → V/m
        return a * P, -(a / (w * MU0 * s)) * C                                        # Faraday
    a = np.sqrt(1.0 / (0.5 * MU0 * s ** 3 * m))                                       # → A/m
    return (a / (w * EPS0 * s)) * C, a * P                                            # Ampère


# ─────────────────────────── solutions (model, FE, difference) ────────

def axis_values(geom, M, U, f_ghz, field, ops):
    """Field component along the beam axis at U = 1 J for every mode: [K, P] V/m (NaN outside Ω)."""
    s = float(geom['scale'])
    m = np.einsum('ek,ek->k', U, M @ U)
    v = np.asarray(ops['Az'] @ U)                                                    # [P, K]
    if field == 'E':
        a = np.sqrt(1.0 / (0.5 * EPS0 * s ** 3 * m))
    else:
        a = np.sqrt(1.0 / (0.5 * MU0 * s ** 3 * m)) / (2 * np.pi * np.asarray(f_ghz) * 1e9 * EPS0 * s)
    out = (v * a[None, :]).T
    out[:, np.asarray(ops['q']) <= 0] = np.nan
    return out


def build_solution(geom, M, U, f_ghz, field, source, eta=None, qoi_kw=None, ops=None):
    """Everything the UI shows for one set of modes U [Ne,K] at f_ghz [K] (source 'model' | 'fe'):
    public 'modes' rows + private display arrays (_E, _H [Nv,3,K], _axis [K,P])."""
    U = np.asarray(U, float)
    f = np.asarray(f_ghz, float)
    if ops is None:
        q, ops = qoi(geom, M, U, f, field, return_ops=True, **(qoi_kw or {}))
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
    E, H = physical_fields(geom, M, U, f, field)
    return {'source': source, 'field': field, 'n_edges': int(len(geom['edges'])), 'modes': modes,
            'axis_length_mm': float(ops['L_axis'] * geom['scale'] * 1e3),
            '_geom': geom, '_M': M, '_U': U, '_f': f, '_E': E, '_H': H, '_ops': ops,
            '_axis': axis_values(geom, M, U, f, field, ops),
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
    geom, M, field = true['_geom'], true['_M'], true['field']
    Ua, rl = align(pred['_U'], true['_U'], M, true['_f'], rel)
    K = Ua.shape[1]
    pa = build_solution(geom, M, Ua, pred['_f'][:K], field, 'model', ops=true['_ops'])
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
    """One loaded model + per-mesh prediction. Thread-safe (one forward at a time)."""

    def __init__(self, checkpoint=None, device=None, field=None, optional=False):
        """optional: a checkpoint that does not load falls back to the untrained model (load_error says why)
        instead of failing — for launchers that pick a checkpoint automatically."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_error = None
        if checkpoint:
            try:
                self.lm, self.field, self.feature_indices = load_model(checkpoint, self.device, field)
                self.untrained = False
            except Exception as e:                           # noqa: BLE001 — reported in info()
                if not optional:
                    raise
                self.load_error, checkpoint = f"{type(e).__name__}: {e}", None
        if not checkpoint:
            self.field = (field or 'E').upper()
            self.lm, self.feature_indices, self.untrained = untrained_model(self.field), None, True
        self.lm.to(self.device)
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.n_modes = int(self.lm.hparams.get('num_field_modes', 6))
        self.n_params = int(sum(p.numel() for p in self.lm.parameters()))
        self._lock = threading.Lock()

    def load(self, checkpoint):
        """Switch to another checkpoint (e.g. a run just trained from the UI); on failure the current
        model stays."""
        path = resolve_checkpoint(checkpoint)
        lm, field, fi = load_model(path, self.device)
        lm.to(self.device)
        with self._lock:
            self.lm, self.field, self.feature_indices = lm, field, fi
            self.untrained, self.load_error, self.checkpoint = False, None, str(path)
            self.n_modes = int(lm.hparams.get('num_field_modes', 6))
            self.n_params = int(sum(p.numel() for p in lm.parameters()))
        return self.info()

    def info(self):
        return {'field': self.field, 'n_modes': self.n_modes, 'n_params': self.n_params, 'device': self.device,
                'untrained': self.untrained, 'checkpoint': self.checkpoint and self.checkpoint.split('/')[-1],
                'load_error': self.load_error,
                'run': self.checkpoint and os.path.basename(os.path.dirname(os.path.abspath(self.checkpoint)))}

    def geometry(self, nodes, tets, family=''):
        """(converter geometry dict with features + operators, CSR mass) of a mesh [m]."""
        geom, M = extract_geometry_3d(nodes, tets, self.field)
        geom.update(field=self.field, shape_type=family or '')
        return geom, (M.tocsr() if hasattr(M, 'tocsr') else M)

    def predict(self, nodes, tets, family='', qoi_kw=None, warmup=False, geom=None, M=None):
        """Model solution of one mesh (build_solution dict + 'time_s'); geom / M reuse a prepared
        geometry (dataset item)."""
        from src.data.dataset_3d import to_csr
        t0 = time.perf_counter()
        if geom is None:
            geom, M = self.geometry(nodes, tets, family)
        t_ops = time.perf_counter() - t0
        with self._lock:
            U, f, lam, t_model = forward(self.lm, geom, self.feature_indices, self.device, warmup)
        t0 = time.perf_counter()
        ne = len(geom['edges'])
        Mmat = to_csr(M, (ne, ne))
        free = ~np.asarray(geom['bnd_edge'], bool) if self.field == 'E' else None
        Kmat = to_csr(geom['K'], (ne, ne)) if 'K' in geom else None
        eta = (residual(Kmat, Mmat, U / np.sqrt(np.einsum('ek,ek->k', U, Mmat @ U)), lam, free)
               if Kmat is not None else None)
        sol = build_solution(geom, Mmat, U, f, self.field, 'model', eta=eta, qoi_kw=qoi_kw)
        sol['time_s'] = {'operators': t_ops, 'model': t_model, 'qoi': time.perf_counter() - t0}
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
        from src.viz.nedelec import interp_vertex, locate, plane_grid
        ia = 'xyz'.index(axis)
        c, s = float(np.asarray(geom['center']).reshape(3)[ia]), float(geom['scale'])
        off = 0.0 if pos_mm is None else (pos_mm * 1e-3 - c) / s
        pts, Ug, _, _ = plane_grid(geom['X'], axis, off, res, pad=0.0)
        tid, bary = locate(geom['X'], geom['tets'], pts)
        inside = (tid >= 0).reshape(Ug.shape)
        idx = -np.ones(inside.shape, np.int64)
        idx[inside] = np.arange(int(inside.sum()))
        a, b, cc, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
        full = (a >= 0) & (b >= 0) & (cc >= 0) & (d >= 0)
        tri = np.concatenate([np.stack([a[full], b[full], d[full]], 1), np.stack([a[full], d[full], cc[full]], 1)])
        sel = tid >= 0
        vals = [interp_vertex(geom['tets'], tid[sel], bary[sel], np.asarray(v)) for v in nodal]
        return self.to_mm(geom, pts[sel]), tri, vals

    @staticmethod
    def surface_fields(sol, vid, k):
        """E, H [Nb,3] at the display-surface vertices."""
        return sol['_E'][vid, :, k], sol['_H'][vid, :, k]


def _num(v):
    v = float(v)
    return v if math.isfinite(v) else None
