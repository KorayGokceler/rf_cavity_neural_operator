"""Differentiable cavity QoI (src/qoi/torch_qoi.py, docs/24 §0.5) and the
QoI loss / metrics of CavityLightning.

The formula tests use random per-geometry operators of the §0.3 shapes and an
independent dense numpy transcription of §0.2 (so they do not depend on the
FE operator assembly); _qoi_keys stacks them exactly as §0.4 prescribes for the
collate.  The consistency tests against src/qoi/operators.qoi_from_dofs and the
dataset's qoi_ops collate run once those exist (skipped otherwise)."""
import math

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from src.data.dataset_3d import _block_diag
from src.qoi.torch_qoi import (C0, EPS0, MU0, QOI_KEYS, QOI_LABELS, has_qoi_ops,
                               qoi_torch)


# ── helpers ──────────────────────────────────────────────────────────────────

def random_ops(ne, n_axis, n_surf, seed, scale=0.1):
    """§0.3-shaped operator dict with random sparse entries (S, M SPD-ish)."""
    rng = np.random.default_rng(seed)
    rs = lambda r, c, d: sp.random(r, c, density=d, random_state=rng, format='csr')   # noqa: E731
    Bm = rs(2 * ne, ne, 0.2) + sp.eye(2 * ne, ne)
    M = (Bm.T @ Bm).tocsr()
    Bs = rs(ne, ne, 0.1)
    S = (Bs.T @ Bs + 0.1 * sp.eye(ne)).tocsr()
    Az = (rs(n_axis, ne, 0.3) - rs(n_axis, ne, 0.3)).tocsr()
    q = np.full(n_axis, 1.0 / n_axis)
    q[:2] = 0.0                                                     # points outside Ω
    zeta = np.linspace(-0.5, 0.5, n_axis)
    Es = (rs(3 * n_surf, ne, 0.3) - rs(3 * n_surf, ne, 0.3)).tocsr()
    Hs = (rs(3 * n_surf, ne, 0.3) - rs(3 * n_surf, ne, 0.3)).tocsr()
    return {'scale': float(scale), 'M': M, 'S': S, 'Az': Az, 'zeta': zeta,
            'q': q, 'L_axis': float(q.sum()), 'Esurf': Es, 'Hsurf': Hs,
            'face_area': rng.uniform(0.5, 1.0, n_surf)}


def _pad(arrs, n, fill=0.0, dtype=torch.float64):
    out = torch.full((len(arrs), n), fill, dtype=dtype)
    for b, a in enumerate(arrs):
        out[b, :len(a)] = torch.as_tensor(np.asarray(a), dtype=dtype)
    return out


def _qoi_keys(ops_list, ne_max):
    """The §0.4 batch keys for per-geometry ops dicts (block-diagonal on the padded layout)."""
    P = max(o['Az'].shape[0] for o in ops_list)
    Nf = max(o['Esurf'].shape[0] // 3 for o in ops_list)
    return {
        'QoI_S': _block_diag([o['S'] for o in ops_list], ne_max, ne_max),
        'QoI_Az': _block_diag([o['Az'] for o in ops_list], P, ne_max),
        'QoI_zeta': _pad([o['zeta'] for o in ops_list], P),
        'QoI_q': _pad([o['q'] for o in ops_list], P),
        'QoI_Esurf': _block_diag([o['Esurf'] for o in ops_list], 3 * Nf, ne_max),
        'QoI_Hsurf': _block_diag([o['Hsurf'] for o in ops_list], 3 * Nf, ne_max),
        'QoI_area': _pad([o['face_area'] for o in ops_list], Nf),
        'QoI_SurfMask': _pad([np.ones(o['Esurf'].shape[0] // 3) for o in ops_list], Nf).bool(),
        'QoI_scale': torch.tensor([o['scale'] for o in ops_list], dtype=torch.float64),
        'QoI_Laxis': torch.tensor([o['L_axis'] for o in ops_list], dtype=torch.float64),
    }


def make_batch(ops_list):
    """Minimal batch: M + the QoI keys, for qoi_torch only."""
    ne_max = max(o['M'].shape[0] for o in ops_list)
    batch = {'M': _block_diag([o['M'] for o in ops_list], ne_max, ne_max),
             'Scale': torch.tensor([o['scale'] for o in ops_list], dtype=torch.float32)}
    batch.update(_qoi_keys(ops_list, ne_max))
    return batch, ne_max


def ref_qoi(o, U, f_hz, Rs=None, sigma=5.8e7, beta=1.0, convention='linac', L_acc=None):
    """Dense numpy transcription of docs/24 §0.2 for one geometry, U [Ne, K], f_hz [K]."""
    s, w = o['scale'], 2 * np.pi * f_hz
    Rs = np.sqrt(np.pi * f_hz * MU0 / sigma) if Rs is None else np.broadcast_to(Rs, f_hz.shape)
    uMu = np.einsum('ek,ek->k', U, o['M'] @ U)
    uSu = np.einsum('ek,ek->k', U, o['S'] @ U)
    a = o['Az'] @ U
    ph = np.exp(1j * np.outer(o['zeta'], w * s / (beta * C0)))
    vt = np.abs((o['q'][:, None] * a * ph).sum(0))
    T = vt / (o['q'][:, None] * np.abs(a)).sum(0)
    pk = lambda A: np.sqrt(((A @ U).reshape(-1, 3, U.shape[1]) ** 2).sum(1)).max(0)   # noqa: E731
    Uj, P = 0.5 * EPS0 * s ** 3 * uMu, 0.5 * Rs / (w * MU0) ** 2 * uSu
    V, Epk, Bpk = s * vt, pk(o['Esurf']), pk(o['Hsurf']) / (w * s)
    V, P, Epk, Bpk = V / np.sqrt(Uj), P / Uj, Epk / np.sqrt(Uj), Bpk / np.sqrt(Uj)
    h = 0.5 if convention == 'circuit' else 1.0
    L = s * o['L_axis'] if L_acc is None else L_acc
    Eacc = V / L
    Q0 = w / P
    return {'f_Hz': f_hz, 'Rs_ohm': Rs, 'U_J': np.ones_like(f_hz), 'P_c_W': P, 'Q0': Q0,
            'G_ohm': Q0 * Rs, 'V_acc_V': V, 'T_transit': T, 'R_over_Q_ohm': h * V ** 2 / w,
            'R_sh_ohm': h * V ** 2 / P, 'L_acc_m': np.full_like(f_hz, L), 'E_acc_Vm': Eacc,
            'E_pk_Vm': Epk, 'B_pk_T': Bpk, 'Epk_Eacc': Epk / Eacc,
            'Bpk_Eacc_mT_per_MVm': Bpk * 1e3 / (Eacc * 1e-6)}


def _setup(K=3, sizes=((30, 11, 7), (22, 9, 5), (35, 13, 8)), seed=0):
    ops = [random_ops(ne, p, nf, seed + i, scale=0.05 + 0.05 * i)
           for i, (ne, p, nf) in enumerate(sizes)]
    batch, ne_max = make_batch(ops)
    rng = np.random.default_rng(seed + 100)
    Us = [rng.standard_normal((o['M'].shape[0], K)) for o in ops]
    F = torch.zeros(len(ops), ne_max, K, dtype=torch.float64)
    for b, U in enumerate(Us):
        F[b, :len(U)] = torch.from_numpy(U)
    f_ghz = torch.from_numpy(rng.uniform(0.5, 3.0, (len(ops), K)))
    return ops, batch, F, f_ghz, Us


# ── formulas, batching, invariances ──────────────────────────────────────────

@pytest.mark.parametrize("convention,Rs,L_acc", [("linac", None, None), ("circuit", 0.01, 0.2)])
def test_matches_dense_reference(convention, Rs, L_acc):
    ops, batch, F, f_ghz, Us = _setup()
    assert has_qoi_ops(batch)
    out = qoi_torch(batch, F, f_ghz, Rs=Rs, convention=convention, beta=0.8, L_acc=L_acc)
    assert set(out) == set(QOI_KEYS)
    for b, (o, U) in enumerate(zip(ops, Us, strict=True)):
        ref = ref_qoi(o, U, f_ghz[b].numpy() * 1e9, Rs=Rs, convention=convention, beta=0.8, L_acc=L_acc)
        for k in QOI_KEYS:
            assert out[k].shape == f_ghz.shape and out[k].dtype == torch.float64
            np.testing.assert_allclose(out[k][b].numpy(), ref[k], rtol=1e-10, err_msg=k)


def test_amplitude_sign_and_dtype_invariance():
    ops, batch, F, f_ghz, _ = _setup()
    ref = qoi_torch(batch, F, f_ghz)
    c = torch.tensor([-3.0, 0.01, 7.0], dtype=torch.float64)
    out = qoi_torch(batch, F * c, f_ghz)
    for k in QOI_KEYS:
        assert torch.allclose(out[k], ref[k], rtol=1e-10)
    out32 = qoi_torch(batch, F.float(), f_ghz.float())              # float32 input, float64 inside
    assert out32['Q0'].dtype == torch.float64
    assert torch.allclose(out32['Q0'], ref['Q0'], rtol=1e-5)


def test_padding_invariance():
    """A geometry alone == the same geometry inside a padded batch (larger neighbours)."""
    ops, batch, F, f_ghz, Us = _setup()
    alone, _ = make_batch([ops[1]])
    out1 = qoi_torch(alone, F[1:2, :len(Us[1])], f_ghz[1:2])
    outB = qoi_torch(batch, F, f_ghz)
    for k in QOI_KEYS:
        assert torch.allclose(out1[k][0], outB[k][1], rtol=1e-12)
    # garbage on padded surface points is masked out of the peaks
    batch2 = dict(batch)
    batch2['QoI_SurfMask'] = batch['QoI_SurfMask'].clone()
    batch2['QoI_SurfMask'][0, -1] = False
    assert (qoi_torch(batch2, F, f_ghz)['E_pk_Vm'][0] <= outB['E_pk_Vm'][0] + 1e-12).all()


def test_frequency_broadcast_and_fixed_rs():
    ops, batch, F, f_ghz, _ = _setup()
    f1 = f_ghz[:, :1]
    a = qoi_torch(batch, F, f1.squeeze(-1))                         # [B] → every mode
    b = qoi_torch(batch, F, f1.expand_as(f_ghz))
    assert torch.allclose(a['Q0'], b['Q0'])
    nb = qoi_torch(batch, F, f_ghz, Rs=20e-9)                       # SRF Nb-like fixed R_s
    assert torch.allclose(nb['Rs_ohm'], torch.full_like(f_ghz, 20e-9))
    cu = qoi_torch(batch, F, f_ghz)
    assert torch.allclose(nb['G_ohm'], cu['G_ohm'], rtol=1e-12)     # G does not depend on R_s
    assert torch.allclose(nb['Q0'] * nb['Rs_ohm'], cu['Q0'] * cu['Rs_ohm'], rtol=1e-12)


def test_peak_p_norm():
    ops, batch, F, f_ghz, Us = _setup()
    ex = qoi_torch(batch, F, f_ghz)['B_pk_T']
    p8 = qoi_torch(batch, F, f_ghz, peak_p=8)['B_pk_T']
    p200 = qoi_torch(batch, F, f_ghz, peak_p=200)['B_pk_T']
    assert (p8 >= ex * (1 - 1e-12)).all() and (p200 >= ex * (1 - 1e-12)).all()
    assert ((p200 - ex) < (p8 - ex) + 1e-30).all() and torch.allclose(p200, ex, rtol=0.02)
    # = (Σ_i |x_i|^p)^{1/p} exactly (U = 1 J normalised)
    o, U, b = ops[0], Us[0], 0
    mag = np.sqrt(((o['Hsurf'] @ U).reshape(-1, 3, U.shape[1]) ** 2).sum(1))
    uMu = np.einsum('ek,ek->k', U, o['M'] @ U)
    w = 2 * np.pi * f_ghz[b].numpy() * 1e9
    ref = (mag ** 8).sum(0) ** (1 / 8) / (w * o['scale']) / np.sqrt(0.5 * EPS0 * o['scale'] ** 3 * uMu)
    np.testing.assert_allclose(p8[b].numpy(), ref, rtol=1e-10)
    with pytest.raises(ValueError):
        qoi_torch(batch, F, f_ghz, peak_p=0.5)


@pytest.mark.parametrize("peak_p", [None, 6.0])
def test_gradcheck(peak_p):
    ops, batch, F, f_ghz, _ = _setup(K=2, sizes=((12, 6, 4), (9, 5, 3)), seed=3)
    F = F.clone().requires_grad_(True)
    f = f_ghz.clone().requires_grad_(True)
    keys = ('Q0', 'G_ohm', 'R_over_Q_ohm', 'R_sh_ohm', 'T_transit', 'Epk_Eacc', 'Bpk_Eacc_mT_per_MVm')

    def fn(F, f):
        out = qoi_torch(batch, F, f, peak_p=peak_p)
        return tuple(torch.log(out[k]) for k in keys)
    assert torch.autograd.gradcheck(fn, (F, f), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_zero_voltage_gradient_is_finite():
    """A mode with A_z u = 0 (V = 0): finite values and gradients (√(x² + tiny))."""
    ops, batch, F, f_ghz, _ = _setup(K=2)
    batch = dict(batch)
    batch['QoI_q'] = torch.zeros_like(batch['QoI_q'])
    F = F.clone().requires_grad_(True)
    out = qoi_torch(batch, F, f_ghz)
    assert (out['V_acc_V'] < 1e-100).all()
    loss = sum(torch.log(out[k].clamp(min=1e-300)).sum() for k in ('Q0', 'R_over_Q_ohm', 'Epk_Eacc'))
    loss.backward()
    assert torch.isfinite(F.grad).all()


def test_missing_ops_raise():
    ops, batch, F, f_ghz, _ = _setup()
    del batch['QoI_Az']
    assert not has_qoi_ops(batch)
    with pytest.raises(ValueError, match="qoi_ops=True"):
        qoi_torch(batch, F, f_ghz)


# ── CavityLightning: loss / metrics wiring ─────────────────────────────────────

skfem = pytest.importorskip("skfem")


@pytest.fixture(scope="module")
def ds3d(tmp_path_factory):
    import pickle

    from src.data.dataset_3d import Maxwell3DDataset
    from tests.maxwell3d_synth import make_dataset
    path = tmp_path_factory.mktemp("qoi3d") / "synth.pkl"
    with open(path, "wb") as f:
        pickle.dump(make_dataset(n_geoms=3, n=3, n_modes=6, seed=0), f)
    return Maxwell3DDataset(str(path), split="test", train_ratio=0.0, val_ratio=0.0)


def _qoi_batch(ds, idx=(0, 1), seed=0):
    """maxwell3d_collate batch + random §0.3 operators per geometry (wiring only)."""
    from src.data.dataset_3d import maxwell3d_collate
    items = [ds[i] for i in idx]
    batch = maxwell3d_collate(items)
    ne_max = batch['Edges'].shape[1]
    ops = []
    for b, it in enumerate(items):
        o = random_ops(it['Edges'].shape[0], 9, 6, seed + b, float(it['Scale']))
        o['M'] = it['M']
        ops.append(o)
    batch.update(_qoi_keys(ops, ne_max))
    return batch, ops


def _module(ds, **kw):
    from src.training.lightning_module import CavityLightning
    torch.manual_seed(0)
    lm = CavityLightning(val_dim=ds[0]['Input_funcs'].shape[-1], hidden_dim=16, n_heads=2, n_basis=10,
                         num_field_modes=4, rff_dim=8, eigenspace_kwargs={"n_layers": 1},
                         near_deg_rel_threshold=0.02, **kw)
    lm.freq_stats = dict(ds.stats)
    return lm


def _logged(lm, batch, prefix="val", grad=False):
    logs = {}
    lm.log = lambda name, value, **kw: logs.__setitem__(name, (float(torch.as_tensor(value).detach()),
                                                               kw.get("batch_size")))
    with torch.set_grad_enabled(grad):
        loss, _, _ = lm._compute_loss_3d(batch, prefix)
    return loss, logs


def test_default_off_is_unchanged_and_logs_metrics(ds3d):
    batch, _ = _qoi_batch(ds3d)
    plain = {k: v for k, v in batch.items() if not k.startswith('QoI_')}
    lm = _module(ds3d)
    assert lm.qoi_weight == 0.0 and lm.qoi_terms == ('Q0', 'R_over_Q_ohm', 'G_ohm')
    l0, logs0 = _logged(lm, plain)
    l1, logs1 = _logged(lm, batch)
    assert torch.equal(l0, l1)                                      # weight 0: loss untouched
    assert not any('qoi' in k for k in logs0)
    for name in QOI_LABELS:
        v, n = logs1[f"val/qoi_{name}_rel_err"]
        assert math.isfinite(v) and 0 <= n <= 2 * 4
    assert 'val/qoi_loss' not in logs1


def test_qoi_loss_on_finite_gradients(ds3d):
    batch, _ = _qoi_batch(ds3d)
    lm0 = _module(ds3d)
    l0, _ = _logged(lm0, batch, "train", grad=True)
    lm = _module(ds3d, qoi_weight=0.5, qoi_terms="Q0,R_over_Q_ohm,G_ohm,Epk_Eacc", qoi_peak_p=8)
    assert lm.qoi_terms == ('Q0', 'R_over_Q_ohm', 'G_ohm', 'Epk_Eacc')
    loss, logs = _logged(lm, batch, "train", grad=True)
    q, _ = logs['train/qoi_loss']
    assert math.isfinite(q) and q > 0
    assert torch.isclose(loss, l0 + 0.5 * q, rtol=1e-4)              # same model init (seed 0)
    for t in lm.qoi_terms:
        assert math.isfinite(logs[f'train/qoi_{t}_log_mse'][0])
    loss.backward()
    grads = [p.grad for p in lm.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_qoi_loss_is_zero_for_exact_prediction(ds3d):
    """Ritz field = target, frequency = FE frequency → QoI loss 0 and rel err 0."""
    batch, _ = _qoi_batch(ds3d)
    lm = _module(ds3d, qoi_weight=1.0)
    K = 4
    out = {'field': batch['Y_field'][..., :K] * -2.0}               # any amplitude / sign
    f = batch['Y_freq'][:, :K]
    logs = {}
    lm.log = lambda name, value, **kw: logs.__setitem__(name, (float(value), kw.get("batch_size")))
    loss = lm._qoi_terms_3d(batch, out, batch['Y_field'][..., :K], f, f, 'val')
    assert float(loss) < 1e-10
    for name in QOI_LABELS:
        v, n = logs[f'val/qoi_{name}_rel_err']
        assert v < 1e-5, (name, v)


def test_masking_rules(ds3d):
    """Near-degenerate / split modes and non-accelerating modes are excluded."""
    from src.qoi.torch_qoi import VOLTAGE_KEYS
    batch, _ = _qoi_batch(ds3d)
    K = 4
    lm = _module(ds3d, qoi_weight=1.0)
    f = batch['Y_freq'].clone()
    f[0, 1] = f[0, 2] = f[0, 1]                                      # sample 0: modes 1, 2 degenerate
    batch['Y_freq'] = f
    iso = lm._isolated_3d(batch, K)
    assert not iso[0, 1] and not iso[0, 2]
    # make every mode of sample 1 non-accelerating except mode 0: zero axis rows elsewhere
    out = {'field': batch['Y_field'][..., :K]}
    logs = {}
    lm.log = lambda name, value, **kw: logs.__setitem__(name, (float(value), kw.get("batch_size")))
    batch['QoI_q'] = torch.zeros_like(batch['QoI_q'])               # no mode accelerates anywhere
    lm._qoi_terms_3d(batch, out, batch['Y_field'][..., :K], f[:, :K], f[:, :K], 'val')
    n_iso = int(iso.sum())
    for name in QOI_LABELS:
        v, n = logs[f'val/qoi_{name}_rel_err']
        assert math.isfinite(v)
        assert n == (0 if name in VOLTAGE_KEYS else n_iso), (name, n, n_iso)


def test_qoi_weight_without_ops_raises(ds3d):
    batch, _ = _qoi_batch(ds3d)
    plain = {k: v for k, v in batch.items() if not k.startswith('QoI_')}
    lm = _module(ds3d, qoi_weight=0.1)
    with pytest.raises(ValueError, match="qoi_ops=True"):
        _logged(lm, plain)
    with pytest.raises(ValueError, match="qoi_terms"):
        _module(ds3d, qoi_weight=0.1, qoi_terms=('Q0', 'nope'))


def test_train_py_cli_flags(monkeypatch):
    import sys

    import train
    monkeypatch.setattr(sys, "argv", ["train.py", "--qoi_weight", "0.3", "--qoi_terms", "Q0, G_ohm",
                                      "--qoi_metrics"])
    args = train.parse_args()
    assert args.qoi_weight == 0.3 and args.qoi_terms == "Q0, G_ohm" and args.qoi_metrics
    from src.config import ConfigDict
    assert train._qoi_enabled(ConfigDict({'qoi_weight': 0.3}))
    assert train._qoi_enabled(ConfigDict({'qoi_metrics': True}))
    assert not train._qoi_enabled(ConfigDict({}))


# ── consistency with the numpy reference + the dataset collate (docs/24 §0.3/§0.4) ──

def _qoi_ds(tmp_path_factory, n_geoms=3):
    import pickle

    from src.data.dataset_3d import Maxwell3DDataset
    from tests.maxwell3d_synth import make_dataset
    path = tmp_path_factory.mktemp("qoiops") / "synth.pkl"
    data = make_dataset(n_geoms=n_geoms, n=4, n_modes=6, seed=1)
    try:                                         # stored labels (converter back-fill), if available
        from src.data.dataset_converter_3d import attach_qoi_labels
        attach_qoi_labels(data, verbose=False)
    except ImportError:
        pass
    with open(path, "wb") as f:
        pickle.dump(data, f)
    try:
        return str(path), Maxwell3DDataset(str(path), split="test", train_ratio=0.0, val_ratio=0.0,
                                           qoi_ops=True)
    except TypeError:
        pytest.skip("Maxwell3DDataset has no qoi_ops flag yet")


def test_torch_matches_numpy_on_collated_batch(tmp_path_factory):
    """qoi_torch on a maxwell3d_collate batch == src.qoi.qoi_from_dofs per geometry (float64)."""
    qoi_np = pytest.importorskip("src.qoi.operators")
    from src.data.dataset_3d import maxwell3d_collate
    _, ds = _qoi_ds(tmp_path_factory)
    items = [ds[i] for i in range(len(ds))]
    batch = maxwell3d_collate(items)
    assert has_qoi_ops(batch)
    f_ghz = batch['Y_freq'].double() * ds.stats['std'] + ds.stats['mean']
    for kw in ({}, {'Rs': 2e-8, 'beta': 0.7, 'convention': 'circuit'}):
        out = qoi_torch(batch, batch['Y_field'], f_ghz, **kw)
        for b, it in enumerate(items):
            ref = qoi_np.qoi_from_dofs(it['qoi_ops'], it['Y_field'].double().numpy(),
                                       f_ghz[b].numpy() * 1e9, **kw)
            for k in QOI_KEYS:
                np.testing.assert_allclose(out[k][b].numpy(), ref[k], rtol=1e-6, err_msg=f"{k} {kw}")
    # the stored labels (when the PKL has them) agree too (float32 targets)
    yq = batch.get('Y_qoi', None)
    assert yq is not None and yq.shape == (len(items), 6, len(ds.qoi_names))
    if torch.isfinite(yq).any():
        out = qoi_torch(batch, batch['Y_field'], f_ghz)
        for j, name in enumerate(ds.qoi_names):
            ok = torch.isfinite(yq[..., j])
            assert torch.allclose(out[name][ok].float(), yq[..., j][ok], rtol=1e-3), name


def test_training_smoke_step_with_qoi_loss(tmp_path_factory):
    """End to end: dataset(qoi_ops=True) → collate → CavityLightning(qoi_weight > 0) → a few
    optimizer steps with finite loss / gradients and the QoI metrics logged."""
    from src.data.dataset_3d import maxwell3d_collate
    _, ds = _qoi_ds(tmp_path_factory)
    batch = maxwell3d_collate([ds[0], ds[1]])
    lm = _module(ds, qoi_weight=0.1, qoi_peak_p=8)
    opt = torch.optim.Adam(lm.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        loss, logs = _logged(lm, batch, "train", grad=True)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in lm.parameters() if p.grad is not None)
        opt.step()
    assert math.isfinite(logs['train/qoi_loss'][0])
    for name in QOI_LABELS:
        assert math.isfinite(logs[f'train/qoi_{name}_rel_err'][0])


def test_fast_dev_run_train_py_with_qoi(tmp_path_factory, tmp_path, monkeypatch):
    import sys

    import train
    path, _ = _qoi_ds(tmp_path_factory, n_geoms=5)
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--config", "configs/eigenspace_3d.yaml", "--fast_dev_run",
        "--qoi_weight", "0.1", "--qoi_terms", "Q0,G_ohm,R_over_Q_ohm", "--override",
        f"dataset.data_path={path}", "dataset.train_ratio=0.6", "dataset.val_ratio=0.2",
        "model.embed_dim=16", "model.n_basis=10", "model.num_field_modes=4", "model.rff_dim=8",
        "model.eigenspace.n_layers=1", "training.batch_size=2", "training.num_workers=0",
        f"training.log_dir={tmp_path}"])
    train.main()
