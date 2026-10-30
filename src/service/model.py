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
    → display (src/viz/hcurl_display.py): the HCurl(p2) fields are drawn directly — no Whitney-N0 projection —
      on the P2 nodes / 8 sub-tets of the curved mesh, the wall as a curved lattice, cut planes and the 3D grid
      by exact evaluation (ModelService.fields / plane / volume).  Solutions are lazy: predict keeps the DOFs,
      the fields are evaluated at the first /fields request.

Display convention (all fields at stored energy U = ½ε0 ∫|E|² = 1 J, standing wave):
    E(t) = E · cos φ,   H(t) = H · sin φ      (the sign of the curl relation is folded into H)
"""
import collections
import math
import os
import threading
import time

import numpy as np
import torch

from src.data.mesh_features import display_geometry
from src.training.checkpoint import resolve_checkpoint
from src.training.lightning_module import CavityLightning

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
    """Set the figures of merit of a solution's mode rows (stored p3 labels, or the predicted p2 fields' mode_qoi),
    blank the voltage-dependent ones where there is no field on the axis, and pick the accelerating mode (largest
    R/Q among the non-degenerate modes)."""
    for row, lab in zip(sol['modes'], labels, strict=False):
        row.update({n: lab[n] for n in QOI_SHOW if n in lab})        # finite or None
        if not (row.get('R_over_Q_ohm') or 0.0) > RQ_FLOOR_OHM:      # no field on the axis
            row.update({n: None for n in VOLTAGE_QOI})
    rq = [-1.0 if m['degenerate'] or m['R_over_Q_ohm'] is None else m['R_over_Q_ohm'] for m in sol['modes']]
    acc = int(np.argmax(rq)) if max(rq, default=-1.0) > 0 else -1
    for k, m in enumerate(sol['modes']):
        m['accelerating'] = k == acc
    return sol


# ─────────────────────────── solutions (model, FE, difference) ────────

def make_solution(source, U, f_ghz, Mp, mesh_src, family='', params=None, labels=None, eta=None, order=2, key=None,
                  axis_length_mm=0.0, L_acc_mm=0.0):
    """Everything the UI shows for one set of modes (source 'model' | 'fe'), without evaluating a field: the public
    'modes' rows + private data (DOFs _U [n, K] of HCurl(order, 'wall') on the curved mesh of mesh_src, frequencies
    _f [GHz], physical mass matrix _Mp [n × n] CSR, and what the lazy display needs).  labels: figures of merit per
    mode (the stored p3 labels, or the predicted p2 fields' mode_qoi); eta: residual indicators."""
    U = np.asarray(U, float)
    f = np.asarray(f_ghz, float)
    deg = degenerate_mask(f)
    modes = [{'mode': k, 'f_GHz': float(f[k]), 'degenerate': bool(deg[k]), 'accelerating': False,
              'eta': None if eta is None else float(eta[k]), **{n: None for n in QOI_SHOW}} for k in range(len(f))]
    sol = {'source': source, 'n_dof': int(U.shape[0]), 'modes': modes, 'axis_length_mm': float(axis_length_mm),
           'L_acc_mm': float(L_acc_mm), '_U': U, '_f': f, '_Mp': Mp, '_mesh_src': mesh_src, '_key': key,
           '_order': int(order), '_family': family, '_params': params}
    return override_qoi(sol, labels or [])


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
    """(rows, difference solution) for a model solution vs the FE one on the same mesh.
    The model's DOFs are aligned to the FE modes (align) in the physical HCurl(p2) mass matrix of the FE side — both
    fields live in the same space of the same mesh — so rel_l2 is the error of two p2 fields in the true M-norm.
    Figures of merit: the FE side keeps its own (the stored p3 labels of a field dataset); the model side
    uses the prediction's own p2 figures of merit (pred['modes'], ModelService.predict) — the aligned vector
    of an isolated mode is the predicted one up to its sign, which leaves them unchanged — and None inside a
    degenerate cluster of the FE spectrum (the aligned vectors there are projections, not predicted modes).
    The difference solution's fields are those of the aligned model minus the FE ones, both at U = 1 J."""
    Mp = true['_Mp']
    Ua, rl = align(pred['_U'], true['_U'], Mp, true['_f'], rel)
    K = Ua.shape[1]
    pa = dict(pred, source='model', _U=Ua, _f=pred['_f'][:K], _Mp=Mp, modes=pred['modes'][:K])
    pa.pop('_resp', None)
    cluster = degenerate_mask(np.asarray(true['_f'][:K], float), rel)       # the clusters of align
    mq = [{n: (None if cluster[k] else pred['modes'][k][n]) for n in QOI_SHOW} for k in range(K)]
    rows = []
    for k in range(K):
        tr, pr = true['modes'][k], pred['modes'][k]
        row = {'mode': k, 'f_fe': tr['f_GHz'], 'f_model': pr['f_GHz'], 'f_rel_err': pr['f_GHz'] / tr['f_GHz'] - 1,
               'rel_l2': float(rl[k]), 'degenerate': tr['degenerate'], 'accelerating': tr['accelerating'],
               'eta': pr['eta'], 'qoi': {}}
        for n in QOI_SHOW:
            a, b = tr[n], mq[k][n]
            row['qoi'][n] = {'fe': a, 'model': b,
                             'rel_err': (None if a in (None, 0) or b is None else b / a - 1)}
        rows.append(row)
    diff = dict(pa, source='diff', modes=[dict(m, f_GHz=true['modes'][m['mode']]['f_GHz'], **mq[m['mode']])
                                          for m in pa['modes']],
                _parts=(pa, true), _f=true['_f'][:K])
    return rows, diff


# ─────────────────────────── the service ───────────────────────────────

class ModelService:
    """One loaded field3d model + per-geometry prediction. Thread-safe (one prediction at a time; the NGSolve
    parts under src.service.datasets._NG_LOCK).  The curved mesh and its field operators (~4 s at 10k tets)
    are kept for the last `cache_size` geometries (RFCAV_FIELD_CACHE, default 2); the display mesh, wall lattice
    and meshes of FE-truth items are cached the same way."""

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
        self._views = collections.OrderedDict()              # key → {'mesh', 'dm', 'lat'} (the display of a mesh)

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
        """Normalised linear mesh with the model's input features (mesh_features.display_geometry) of a mesh [m] — the
        input-feature views (cut planes of the model's input channels); params: the family's geom_params."""
        geom = display_geometry(nodes, tets)
        geom.update(shape_type=family or '', params=params)
        return geom

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
        mesh = self._views[key]['mesh'] if key in self._views else load_mesh_src(mesh_src)
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

    def _view(self, mesh_src, key, mesh=None):
        """{'mesh', 'dm' (P2 display mesh), 'lat' (wall lattice)} of a geometry, cached per key (caller holds the
        locks); mesh: an already loaded curved mesh (else the cached one of a prediction, else reloaded)."""
        from src.service.geometry import load_mesh_src
        from src.viz import hcurl_display as hd
        if key is not None and key in self._views:
            self._views.move_to_end(key)
            return self._views[key]
        if mesh is None:
            mesh = self._ops[key][0] if key in self._ops else load_mesh_src(mesh_src)
        v = {'mesh': mesh, 'dm': hd.display_mesh(mesh), 'lat': hd.surface_lattice(mesh)}
        if key is not None and self.cache_size > 0:
            self._views[key] = v
            while len(self._views) > self.cache_size:
                self._views.popitem(last=False)
        return v

    def predict(self, mesh_src, family='', params=None, key=None):
        """Model solution of one geometry (make_solution dict + 'time_s', 'qoi_source', 'n_dof', 'n_tets'):
        mesh_src its curved mesh (geometry.curved_from_cad / the field H5), params the family's geom_params,
        key a cache key (the geometry id).  The fields are evaluated later, at the first display request."""
        from src.data.sparse import to_csr
        from src.data_gen import field_labels as fl
        from src.service.datasets import _NG_LOCK
        t = {}
        order = int(mesh_src.get('model_order') or 2)
        with _NG_LOCK, self._lock:
            mesh, ops = self._operators(mesh_src, key, order, t)
            U, f, lam, t['model'] = forward(self.lm, ops, self.feature_indices, family or '', self.device)
            t0 = time.perf_counter()
            labels = field_qoi(mesh, U, f, order, family, params)
            t['qoi'] = time.perf_counter() - t0
            t0 = time.perf_counter()
            n_tets = int(mesh.ne)
            axis = fl._beam_axis(family, mesh, params)
            L_axis = self.axis_chord(self._view(mesh_src, key, mesh), axis, mesh_src.get('deform'))
        n = int(ops['n_dof'])
        Kf, Mf = to_csr(ops['K'], (n, n)), to_csr(ops['M'], (n, n))
        Un = U / np.sqrt(np.maximum(np.einsum('ik,ik->k', U, Mf @ U), 1e-300))
        eta = residual(Kf, Mf, Un, lam, ~np.asarray(ops['bnd'], bool))
        Mp = Mf * float(ops['scale'])                         # the field operators' M is the physical one / scale
        sol = make_solution('model', U, f, Mp, mesh_src, family, params, labels, eta, order, key,
                            L_axis * 1e3, (axis[2] if axis[2] is not None else L_axis) * 1e3)
        t['display'] = time.perf_counter() - t0
        sol.update(qoi_source='p2_model', n_tets=n_tets, time_s=t)
        return sol

    def fe_solution(self, item, key):
        """FE-truth solution of an opened dataset item (FieldH5Dataset.open; key: its geometry id): the stored
        label fields as they are, the stored p3 figures of merit; the item's mesh seeds the display cache."""
        from src.data_gen import field_labels as fl
        from src.service.datasets import _NG_LOCK
        with _NG_LOCK, self._lock:
            axis = fl._beam_axis(item['family'], item['mesh'], item['geom_params'])
            L_axis = self.axis_chord(self._view(item['mesh_src'], key, item['mesh']), axis, item['mesh_src'].get('deform'))
        return make_solution('fe', item['U'], item['f_GHz'], item['Mp'], item['mesh_src'], item['family'],
                             item['geom_params'], item['qoi_labels'], None, item['order'], key, L_axis * 1e3,
                             (axis[2] if axis[2] is not None else L_axis) * 1e3)

    @staticmethod
    def axis_chord(view, axis, deform=None, n_axis=401):
        """Length [m] of the beam axis (d, p0, L_acc of field_labels._beam_axis) inside the cavity."""
        from src.viz import hcurl_display as hd
        return hd.axis_chord(view['mesh'], view['dm'], axis[0], axis[1], n_axis, deform)

    # ── display data (any solution: model, FE, difference) ───────────
    def _parts(self, sol):
        return sol['_parts'] if sol['source'] == 'diff' else (sol,)

    def _raw(self, sol, view):
        """Fields of a solution at the P2 nodes / wall lattice / beam axis, float32 (caller holds the locks):
        dict(E, H [K, Nn, 3], sE, sH [K, Ns] magnitudes, axis (zeta_mm [P], Ez [K, P]))."""
        from src.viz import hcurl_display as hd
        lat = view['lat']
        if sol['source'] == 'diff':
            a, b = (self._raw_vec(p, view) for p in sol['_parts'])
            K = a['E'].shape[0]
            return {'E': a['E'] - b['E'][:K], 'H': a['H'] - b['H'][:K],
                    'sE': hd.surface_magnitudes(lat, a['vE'], b['vE'][:K]),
                    'sH': hd.surface_magnitudes(lat, a['vH'], b['vH'][:K]),
                    'axis': (a['axis'][0], a['axis'][1] - b['axis'][1][:K])}
        r = self._raw_vec(sol, view)
        return {'E': r['E'], 'H': r['H'], 'sE': hd.surface_magnitudes(lat, r['vE']),
                'sH': hd.surface_magnitudes(lat, r['vH']), 'axis': r['axis']}

    def _raw_vec(self, sol, view):
        from src.data_gen import field_labels as fl
        from src.viz import hcurl_display as hd
        mesh, dm, lat = view['mesh'], view['dm'], view['lat']
        U, f, Mp, order = sol['_U'], sol['_f'], sol['_Mp'], sol['_order']
        nf = hd.nodal_fields(mesh, dm, U, f, Mp, order)
        vE, vH = hd.surface_vectors(mesh, lat, U, f, Mp, order)
        d, p0, _ = fl._beam_axis(sol['_family'], mesh, sol['_params'])
        zeta, ez, _ = hd.axis_profile(mesh, dm, U, Mp, d, p0, order, deform=sol['_mesh_src'].get('deform'))
        return {'E': nf['E'], 'H': nf['H'], 'vE': vE, 'vH': vH, 'axis': (zeta, ez)}

    def fields(self, sol):
        """Display payload of a solution (cached in the solution): dict(x_mm [Nn, 3], tets [Nc, 4] sub-tets, E, H
        [K, Nn, 3] float32, sE, sH [K, Ns] float32 wall magnitudes, E_max, H_max [K], axis (zeta_mm, Ez [K, P]))."""
        from src.service.datasets import _NG_LOCK
        with _NG_LOCK, self._lock:
            if '_resp' in sol:
                return sol['_resp']
            part = self._parts(sol)[0]
            view = self._view(part['_mesh_src'], part['_key'])
            r = self._raw(sol, view)
            n = {k: np.linalg.norm(r[k], axis=2).max(1) for k in ('E', 'H')}
            sm = {'E': r['sE'].max(1), 'H': r['sH'].max(1)}
            out = dict(r, x_mm=(view['dm']['x'] * 1e3).astype(np.float32), tets=view['dm']['tets'].astype(np.uint32),
                       E_max=np.maximum(np.maximum(n['E'], sm['E']), 1e-300),
                       H_max=np.maximum(np.maximum(n['H'], sm['H']), 1e-300))
            sol['_resp'] = out
            return out

    # ── exact evaluation (cut planes, 3D grid) ─────────────────────────
    def _points(self, sol, k, pts):
        """E, H [N, 3] of mode k at physical points [m] (0 outside) and the inside mask; a difference solution gives
        aligned model − FE."""
        from src.viz import hcurl_display as hd
        out = None
        for sgn, p in zip((1.0, -1.0), self._parts(sol), strict=False):
            part = self._view(p['_mesh_src'], p['_key'])
            E, H, inside = hd.eval_points(part['mesh'], p['_U'][:, k:k + 1], p['_f'][k:k + 1], p['_Mp'], pts,
                                          p['_order'], p['_mesh_src'].get('deform'))
            out = (E[0], H[0], inside) if out is None else (out[0] + sgn * E[0], out[1] + sgn * H[0], inside)
        return out

    def plane(self, sol, k, axis='y', pos_mm=None, res=121):
        """Cut plane of mode k by exact point evaluation: (points [P, 3] mm, triangles [T, 3], E [P, 3], H [P, 3])."""
        from src.service.datasets import _NG_LOCK
        ia = 'xyz'.index(axis)
        with _NG_LOCK, self._lock:
            part = self._parts(sol)[0]
            dm = self._view(part['_mesh_src'], part['_key'])['dm']
            lo, hi = dm['x'].min(0), dm['x'].max(0)
            pos = float(dm['centroid'][ia]) if pos_mm is None else pos_mm * 1e-3
            iu, iv = [c for c in range(3) if c != ia]
            uu, vv = np.meshgrid(np.linspace(lo[iu], hi[iu], res), np.linspace(lo[iv], hi[iv], res))
            pts = np.zeros((res * res, 3))
            pts[:, ia], pts[:, iu], pts[:, iv] = pos, uu.ravel(), vv.ravel()
            E, H, inside = self._points(sol, k, pts)
        ins = inside.reshape(res, res)
        idx = -np.ones(ins.shape, np.int64)
        idx[ins] = np.arange(int(ins.sum()))
        a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
        full = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
        tri = np.concatenate([np.stack([a[full], b[full], d[full]], 1), np.stack([a[full], d[full], c[full]], 1)])
        return pts[inside] * 1e3, tri, E[inside], H[inside]

    def volume(self, sol, k, n_points=80_000):
        """Mode k on a regular 3D grid by exact point evaluation, in vtkImageData order (x fastest, then y, z), 0
        outside the cavity: dict(dims, origin_mm, spacing_mm, inside [N] bool, E, H [N, 3])."""
        from src.service.datasets import _NG_LOCK
        with _NG_LOCK, self._lock:
            part = self._parts(sol)[0]
            dm = self._view(part['_mesh_src'], part['_key'])['dm']
            lo, hi = dm['x'].min(0), dm['x'].max(0)
            ext = np.maximum(hi - lo, 1e-12)
            d = float((np.prod(ext) / max(int(n_points), 8)) ** (1 / 3))
            dims = np.maximum(2, np.ceil(ext / d).astype(int) + 1)
            zz, yy, xx = np.meshgrid(*(lo[i] + d * np.arange(dims[i]) for i in (2, 1, 0)), indexing='ij')
            pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1)
            E, H, inside = self._points(sol, k, pts)
        return {'dims': [int(x) for x in dims], 'origin_mm': (lo * 1e3).tolist(), 'spacing_mm': d * 1e3,
                'inside': inside, 'E': E, 'H': H}

    # ── input-feature views (linear mesh of display_geometry) ──
    @staticmethod
    def to_mm(geom, X):
        return (np.asarray(X) * geom['scale'] + np.asarray(geom['center']).reshape(1, 3)) * 1e3

    def plane_values(self, geom, nodal, axis='y', pos_mm=None, res=121):
        """Vertex arrays `nodal` (list of [Nv,…]) interpolated on a cut plane (points mm, triangles, values)."""
        from src.viz.meshsample import interp_vertex, locate_grid
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


def _num(v):
    v = float(v)
    return v if math.isfinite(v) else None
