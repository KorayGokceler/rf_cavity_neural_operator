"""Model side of the prediction service: checkpoint loading, pure-model prediction of one mesh
(frequencies, Ritz fields, figures of merit, label-free residual), and display fields in
physical units — shared by scripts/predict_geometry.py and src/service/api.py.

Display convention (all fields at stored energy U = 1 J, standing wave):
    E(t) = E · cos φ,   H(t) = H · sin φ      (the sign of the curl relation is folded into H)
"""
import math
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


# ─────────────────────────── the service ───────────────────────────────

class ModelService:
    """One loaded model + per-mesh prediction. Thread-safe (one forward at a time)."""

    def __init__(self, checkpoint=None, device=None, field=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if checkpoint:
            self.lm, self.field, self.feature_indices = load_model(checkpoint, self.device, field)
            self.untrained = False
        else:
            self.field = (field or 'E').upper()
            self.lm, self.feature_indices, self.untrained = untrained_model(self.field), None, True
        self.lm.to(self.device)
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.n_modes = int(self.lm.hparams.get('num_field_modes', 6))
        self.n_params = int(sum(p.numel() for p in self.lm.parameters()))
        self._lock = threading.Lock()

    def info(self):
        return {'field': self.field, 'n_modes': self.n_modes, 'n_params': self.n_params, 'device': self.device,
                'untrained': self.untrained, 'checkpoint': self.checkpoint and self.checkpoint.split('/')[-1]}

    def predict(self, nodes, tets, family='', qoi_kw=None, warmup=False):
        """Everything for one mesh: dict with 'modes' (table rows), 'qoi', timings, and private
        arrays (geom, M, U, fields) kept for the field / plane requests."""
        t0 = time.perf_counter()
        geom, M = extract_geometry_3d(nodes, tets, self.field)
        geom.update(field=self.field, shape_type=family or '')
        t_ops = time.perf_counter() - t0
        with self._lock:
            U, f, lam, t_model = forward(self.lm, geom, self.feature_indices, self.device, warmup)
        t0 = time.perf_counter()
        q, ops = qoi(geom, M, U, f, self.field, return_ops=True, **(qoi_kw or {}))
        t_qoi = time.perf_counter() - t0
        free = ~np.asarray(geom['bnd_edge'], bool) if self.field == 'E' else None
        from src.data.dataset_3d import to_csr
        Kmat = to_csr(geom['K'], (len(geom['edges']),) * 2)
        Mmat = to_csr(M, (len(geom['edges']),) * 2) if not hasattr(M, 'tocsr') else M.tocsr()
        eta = residual(Kmat, Mmat, U / np.sqrt(np.einsum('ek,ek->k', U, Mmat @ U)), lam, free)
        deg = degenerate_mask(f)
        rq = np.asarray(q['R_over_Q_ohm'], float)
        ok_rq = np.where(deg | ~np.isfinite(rq), -np.inf, rq)
        accel = int(np.argmax(ok_rq)) if np.isfinite(ok_rq).any() else -1
        modes = []
        for k in range(len(f)):
            row = {'mode': k, 'f_GHz': float(f[k]), 'degenerate': bool(deg[k]), 'eta': float(eta[k]),
                   'accelerating': k == accel}
            row.update({n: _num(q[n][k]) for n in QOI_SHOW})
            modes.append(row)
        E, H = physical_fields(geom, Mmat, U, f, self.field)
        zeta_mm = np.asarray(ops['zeta']) * geom['scale'] * 1e3          # axial, relative to the volume centroid
        return {
            'field': self.field, 'n_edges': int(len(geom['edges'])), 'modes': modes,
            'axis_length_mm': float(ops['L_axis'] * geom['scale'] * 1e3),
            'time_s': {'operators': t_ops, 'model': t_model, 'qoi': t_qoi},
            '_geom': geom, '_M': Mmat, '_U': U, '_f': f, '_E': E, '_H': H, '_ops': ops, '_zeta_mm': zeta_mm,
        }

    # ── display data ─────────────────────────────────────────────────
    @staticmethod
    def to_mm(geom, X):
        return (np.asarray(X) * geom['scale'] + np.asarray(geom['center']).reshape(1, 3)) * 1e3

    def axis_profile(self, pred, k):
        """(axial coordinate [mm], E_z [V/m]) of mode k on the beam axis (NaN outside the cavity)."""
        geom, ops, U, f = pred['_geom'], pred['_ops'], pred['_U'], pred['_f']
        s = float(geom['scale'])
        m = float(U[:, k] @ (pred['_M'] @ U[:, k]))
        v = np.asarray(ops['Az'] @ U[:, k]).ravel()
        if self.field == 'E':
            ez = v * np.sqrt(1.0 / (0.5 * EPS0 * s ** 3 * m))
        else:
            w = 2 * np.pi * f[k] * 1e9
            ez = v * np.sqrt(1.0 / (0.5 * MU0 * s ** 3 * m)) / (w * EPS0 * s)
        ez = np.where(np.asarray(ops['q']) > 0, ez, np.nan)
        return pred['_zeta_mm'], ez

    def plane(self, pred, k, axis='y', pos_mm=None, res=121):
        """Cut plane through the cavity: (points [P,3] mm, triangles [T,3], E [P,3], H [P,3])."""
        from src.viz.nedelec import interp_vertex, locate, plane_grid
        geom = pred['_geom']
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
        E = interp_vertex(geom['tets'], tid[sel], bary[sel], pred['_E'][:, :, k])
        H = interp_vertex(geom['tets'], tid[sel], bary[sel], pred['_H'][:, :, k])
        return self.to_mm(geom, pts[sel]), tri, E, H

    def surface_fields(self, pred, vid, k):
        """E, H [Nb,3] at the display-surface vertices."""
        return pred['_E'][vid, :, k], pred['_H'][vid, :, k]


def _num(v):
    v = float(v)
    return v if math.isfinite(v) else None
