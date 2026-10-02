"""Cavity QoI data plumbing (docs/24_CAVITY_QOI.md §0.4): converter labels, dataset Y_qoi /
qoi_ops, maxwell3d_collate QoI_* keys, scripts/eval_qoi.py.

Plumbing tests run against a FAKE src.qoi of the contract shape (monkeypatched into
sys.modules), so they do not depend on the reference implementation; the tests marked
`real` use src/qoi/operators.py when it is importable (else skipped).
"""
import importlib
import pickle
import sys
import types
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.sparse as sp
import torch

pytest.importorskip("skfem")

import src.data.dataset_converter_3d as conv                            # noqa: E402
from src.data.dataset_3d import (Maxwell3DDataset, item_from_geometry,  # noqa: E402
                                 maxwell3d_collate)
from src.models.hcurl import spmm                                       # noqa: E402
from tests.maxwell3d_synth import C0, box_geometry, make_dataset        # noqa: E402

LABELS = ('Q0', 'G_ohm', 'R_over_Q_ohm', 'R_sh_ohm', 'T_transit', 'Epk_Eacc',
          'Bpk_Eacc_mT_per_MVm')
QOI_KEYS = {'Y_qoi', 'QoI_S', 'QoI_Az', 'QoI_zeta', 'QoI_q', 'QoI_Esurf', 'QoI_Hsurf', 'QoI_area',
            'QoI_SurfMask', 'QoI_scale', 'QoI_Laxis'}


# ── fake src.qoi (contract shapes, amplitude / sign invariant functionals) ─────

def _fake_build(X, tets, edges, scale, center, field, M=None, n_axis=13, axis_xy=(0.0, 0.0)):
    X = np.asarray(X, np.float64)
    edges = np.asarray(edges, np.int64)
    ne = len(edges)
    if M is None:
        M = conv.assemble_n0(X, tets, edges)[0]
    bf = conv.boundary_faces(X, np.asarray(tets))
    nf = len(bf)
    seed = int(1e3 * abs(X).sum()) % (2 ** 31) + nf
    rng = np.random.default_rng(seed)
    S = sp.random(ne, ne, density=4.0 / ne, random_state=rng)
    S = (S @ S.T + sp.eye(ne)).tocsr()
    Az = sp.random(n_axis, ne, density=6.0 / ne, random_state=rng, format='csr')
    Es = sp.random(3 * nf, ne, density=6.0 / ne, random_state=rng, format='csr')
    Hs = sp.random(3 * nf, ne, density=6.0 / ne, random_state=rng, format='csr')
    zeta = np.linspace(-1, 1, n_axis)
    q = np.full(n_axis, 2.0 / n_axis)
    q[0] = 0.0
    tri = X[bf]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    _fake_build.calls.append(dict(M=sp.csr_matrix(M), n_edges=ne, field=field))
    if _fake_build.fail_if is not None and _fake_build.fail_if(ne):
        raise RuntimeError("fake failure")
    return {'field': field, 'scale': float(scale), 'M': sp.csr_matrix(M), 'S': S, 'Az': Az,
            'zeta': zeta, 'q': q, 'L_axis': float(q.sum()), 'Esurf': Es, 'Hsurf': Hs, 'face_area': area}


def _fake_qoi_from_dofs(ops, U, f_hz, Rs=None, sigma=5.8e7, beta=1.0, L_acc=None, convention='linac'):
    U = np.asarray(U, np.float64)
    U = U[:, None] if U.ndim == 1 else U
    f = np.broadcast_to(np.asarray(f_hz, np.float64), (U.shape[1],))
    uMu = np.einsum('ik,ik->k', U, ops['M'] @ U)
    uSu = np.einsum('ik,ik->k', U, ops['S'] @ U)
    a = ops['Az'] @ U
    V2 = (ops['q'][:, None] * a).sum(0) ** 2 / uMu
    e = np.linalg.norm((ops['Esurf'] @ U).reshape(-1, 3, U.shape[1]), axis=1).max(0) ** 2 / uMu
    h = np.linalg.norm((ops['Hsurf'] @ U).reshape(-1, 3, U.shape[1]), axis=1).max(0) ** 2 / uMu
    Q0 = f * 1e-9 * uMu / uSu
    out = {'Q0': Q0, 'G_ohm': Q0 * np.sqrt(f) / sigma ** 0.5, 'R_over_Q_ohm': V2 / f * 1e9,
           'R_sh_ohm': V2 * Q0, 'T_transit': V2 / (1 + V2), 'Epk_Eacc': e / (V2 + 1e-30),
           'Bpk_Eacc_mT_per_MVm': h / (V2 + 1e-30), 'f_Hz': f}
    if convention == 'circuit':
        out['R_over_Q_ohm'] = 0.5 * out['R_over_Q_ohm']
    return out


@pytest.fixture
def fake_qoi(monkeypatch):
    import src
    mod = types.ModuleType('src.qoi')
    mod.__path__ = [str(Path(src.__file__).parent / 'qoi')]          # real submodules (torch_qoi) stay importable
    _fake_build.calls, _fake_build.fail_if = [], None
    mod.build_qoi_operators = _fake_build
    mod.qoi_from_dofs = _fake_qoi_from_dofs
    mod.cavity_qoi = None
    mod.QOI_LABELS = LABELS
    monkeypatch.setitem(sys.modules, 'src.qoi', mod)
    monkeypatch.setattr(src, 'qoi', mod, raising=False)
    return mod


def _real_qoi():
    try:
        q = importlib.import_module('src.qoi')
        importlib.import_module('src.qoi.operators')
    except ImportError as e:
        pytest.skip(f"src.qoi.operators not available ({e})")
    for k in ('build_qoi_operators', 'qoi_from_dofs', 'QOI_LABELS'):
        if not hasattr(q, k):
            pytest.skip(f"src.qoi has no {k}")
    return q


# ── data fixtures ──────────────────────────────────────────────────────────────

def _write_pkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return str(path)


@pytest.fixture(scope='module', params=['H', 'E'])
def synth(request, tmp_path_factory):
    """(field, PKL path) of 3 synthetic boxes of different sizes, no QoI labels."""
    d = make_dataset(n_geoms=3, n=4, n_modes=4, seed=2, field=request.param)
    return request.param, _write_pkl(tmp_path_factory.mktemp('q' + request.param) / 's.pkl', d)


def _all(path, **kw):
    return Maxwell3DDataset(path, split='test', train_ratio=0.0, val_ratio=0.0, **kw)


def _write_h5(path, field='H', n_geoms=2, start=0, n_modes=4, seed=0):
    """Generator-format H5 (sample_XXXX groups) of synthetic PEC boxes, physical size 'a'."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, 'w') as f:
        f.attrs['metadata'] = '{"synthetic": true}'
        for i in range(n_geoms):
            dims = np.sort(rng.uniform(0.5, 1.0, 3))[::-1]
            geom, lam, vec = box_geometry(dims, 4, n_modes=n_modes, field=field)
            a = float(rng.uniform(0.05, 0.2))                    # metres per synth unit
            freqs = C0 * np.sqrt(lam) / (2 * np.pi * a) / 1e9      # GHz
            g = f.create_group(f'sample_{start + i:04d}')
            g.create_dataset('nodes', data=np.asarray(geom['X'], np.float64) * a + np.array([0.0, 0.0, 0.3]))
            g.create_dataset('tets', data=geom['tets'])
            g.create_dataset('edges', data=geom['edges'])
            g.create_dataset('freqs', data=freqs[::-1].copy())  # converter sorts
            g.create_dataset('e_edges' if field == 'E' else 'h_edges', data=vec[:, ::-1].copy())
            g.attrs['shape_type'] = 'box'
            g.attrs['field'] = field
    return str(path)


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ── dataset: Y_qoi, qoi_names, qoi_ops ──────────────────────────────────────

def test_y_qoi_nan_without_labels(synth):
    _, path = synth
    ds = _all(path)
    it = ds[0]
    K = it['Y_freq'].shape[0]
    assert ds.qoi_names == LABELS and not ds.has_qoi
    assert it['Y_qoi'].shape == (K, len(LABELS)) and it['Y_qoi'].dtype == torch.float32
    assert torch.isnan(it['Y_qoi']).all()
    assert 'qoi_ops' not in it


def test_y_qoi_follows_frequency_order(tmp_path):
    d = make_dataset(n_geoms=2, n=4, n_modes=4, seed=3)
    rng = np.random.default_rng(0)
    for s in d['samples']:
        s['qoi'] = {n: float(1000 * s['Theta'][1] + j) for j, n in enumerate(LABELS)}
    d['samples'] = [d['samples'][i] for i in rng.permutation(len(d['samples']))]   # scrambled order
    d['metadata']['qoi'] = {'labels': list(LABELS[:3])}                         # subset → ds.qoi_names
    ds = _all(_write_pkl(tmp_path / 'l.pkl', d))
    assert ds.qoi_names == LABELS[:3] and ds.has_qoi
    for i in range(len(ds)):
        it = ds[i]
        f = np.array([ds.samples_metadata[j]['Theta'][1] for j in ds.geom_to_samples[ds.active_geoms[i]]])
        assert np.all(np.diff(f) >= 0)
        want = 1000 * f[:, None] + np.arange(3)[None]
        np.testing.assert_allclose(it['Y_qoi'].numpy(), want, rtol=1e-6)


def test_qoi_ops_lazy_and_cached(synth, fake_qoi):
    field, path = synth
    ds = _all(path)
    ds[0]
    assert _fake_build.calls == []                                   # off by default
    ds = _all(path, qoi_ops=True)
    a, b = ds[0]['qoi_ops'], ds[0]['qoi_ops']
    assert a is b and len(_fake_build.calls) == 1
    c = _fake_build.calls[0]
    assert c['field'] == field
    M = ds.operators(ds.active_geoms[0])[0]
    assert abs(c['M'] - M).max() == 0                                # the dataset's M is passed
    ds_nc = _all(path, qoi_ops=True, cache_operators=False)
    ds_nc[0], ds_nc[0]
    assert len(_fake_build.calls) == 3


def test_qoi_ops_lean_pkl_rebuilds_m(synth, fake_qoi, tmp_path):
    field, path = synth
    d = _load(path)
    full = _all(path)
    M_full = full.operators(full.active_geoms[0])[0]
    for g in d['geometry_pool'].values():
        for k in ('M', 'K', 'G', 'Kp'):
            g.pop(k)
    ds = _all(_write_pkl(tmp_path / 'lean.pkl', d), qoi_ops=True)
    it = ds[0]
    assert 'qoi_ops' in it
    M_rebuilt = _fake_build.calls[-1]['M']
    assert abs(M_rebuilt - M_full).max() < 1e-6 * abs(M_full).max()     # float32 X


# ── collate ───────────────────────────────────────────────────────────────────

def _dense(A):
    return A.to_dense().numpy()


def test_collate_qoi_keys_and_block_layout(synth, fake_qoi):
    field, path = synth
    ds = _all(path, qoi_ops=True)
    items = [ds[i] for i in range(len(ds))]
    batch = maxwell3d_collate(items)
    assert QOI_KEYS <= set(batch)
    B = len(items)
    ne = [it['Edges'].shape[0] for it in items]
    assert len(set(ne)) > 1                                          # padding is exercised
    Ne = max(ne)
    ops = [it['qoi_ops'] for it in items]
    P = max(o['Az'].shape[0] for o in ops)
    nf = [len(o['face_area']) for o in ops]
    Nf = max(nf)
    K = items[0]['Y_freq'].shape[0]
    shapes = {'QoI_S': (B * Ne, B * Ne), 'QoI_Az': (B * P, B * Ne), 'QoI_Esurf': (B * 3 * Nf, B * Ne),
              'QoI_Hsurf': (B * 3 * Nf, B * Ne)}
    for k, shp in shapes.items():
        A = batch[k]
        assert A.layout == torch.sparse_coo and A.is_coalesced() and A.dtype == torch.float64, k
        assert tuple(A.shape) == shp, k
    for k, shp in {'QoI_zeta': (B, P), 'QoI_q': (B, P), 'QoI_area': (B, Nf), 'QoI_SurfMask': (B, Nf),
                   'QoI_scale': (B,), 'QoI_Laxis': (B,), 'Y_qoi': (B, K, len(LABELS))}.items():
        assert tuple(batch[k].shape) == shp, k
    assert batch['QoI_SurfMask'].dtype == torch.bool
    assert batch['Y_qoi'].dtype == torch.float32
    rows = {'QoI_S': Ne, 'QoI_Az': P, 'QoI_Esurf': 3 * Nf, 'QoI_Hsurf': 3 * Nf}
    src = {'QoI_S': 'S', 'QoI_Az': 'Az', 'QoI_Esurf': 'Esurf', 'QoI_Hsurf': 'Hsurf'}
    for k, r in rows.items():
        D = _dense(batch[k])
        for b, o in enumerate(ops):
            A = o[src[k]].toarray()
            blk = D[b * r:(b + 1) * r, b * Ne:(b + 1) * Ne]
            np.testing.assert_array_equal(blk[:A.shape[0], :A.shape[1]], A)
            assert not blk[A.shape[0]:].any() and not blk[:, A.shape[1]:].any()      # padding empty
            off = D[b * r:(b + 1) * r].copy()
            off[:, b * Ne:(b + 1) * Ne] = 0
            assert not off.any()                                                     # block-diagonal
    for b, o in enumerate(ops):
        p = o['Az'].shape[0]
        np.testing.assert_array_equal(batch['QoI_q'][b, :p].numpy(), o['q'])
        assert not batch['QoI_q'][b, p:].any()
        np.testing.assert_array_equal(batch['QoI_zeta'][b, :p].numpy(), o['zeta'])
        np.testing.assert_array_equal(batch['QoI_area'][b, :nf[b]].numpy(), o['face_area'])
        assert batch['QoI_SurfMask'][b].sum() == nf[b] and batch['QoI_SurfMask'][b, :nf[b]].all()
        assert batch['QoI_scale'][b] == pytest.approx(o['scale'])
        assert batch['QoI_Laxis'][b] == pytest.approx(o['L_axis'])
    # one spmm serves the batch: (QoI_S F)_b = S_b u_b
    F = batch['Y_field'].double()
    SF = spmm(batch['QoI_S'], F)
    for b, o in enumerate(ops):
        np.testing.assert_allclose(SF[b, :ne[b]].numpy(), o['S'] @ F[b, :ne[b]].numpy(), rtol=1e-12, atol=1e-14)


def test_collate_y_qoi_padding_and_partial_ops(synth, fake_qoi):
    field, path = synth
    ds = _all(path)
    items = [ds[0], ds[1]]
    batch = maxwell3d_collate(items)
    assert 'Y_qoi' in batch and not (set(batch) & (QOI_KEYS - {'Y_qoi'}))
    # Y_qoi with different K → NaN padding
    items[1] = dict(items[1], Y_qoi=torch.ones(2, len(LABELS)))
    yq = maxwell3d_collate(items)['Y_qoi']
    assert torch.isnan(yq[1, 2:]).all() and (yq[1, :2] == 1).all()
    # QoI_* only when EVERY item has qoi_ops
    ds_q = _all(path, qoi_ops=True)
    mixed = maxwell3d_collate([ds_q[0], ds[1]])
    assert not (set(mixed) & (QOI_KEYS - {'Y_qoi'}))
    assert QOI_KEYS <= set(maxwell3d_collate([ds_q[0], ds_q[1]]))
    # the existing operator keys are unchanged (CSR)
    assert batch['M'].layout == torch.sparse_csr


def test_item_from_geometry_qoi_ops(synth, fake_qoi):
    field, path = synth
    g = _load(path)['geometry_pool'][0]
    it = item_from_geometry(g, qoi_ops=True)
    assert 'qoi_ops' in it and 'Y_qoi' not in it
    batch = maxwell3d_collate([it, item_from_geometry(_load(path)['geometry_pool'][1], qoi_ops=True)])
    assert (QOI_KEYS - {'Y_qoi'}) <= set(batch) and 'Y_qoi' not in batch
    assert 'qoi_ops' not in item_from_geometry(g)


# ── converter ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('field', ['H', 'E'])
def test_converter_stores_qoi_labels(tmp_path, fake_qoi, field):
    h5a = _write_h5(tmp_path / 'a.h5', field, n_geoms=2, start=0, seed=1)
    h5b = _write_h5(tmp_path / 'b.h5', field, n_geoms=1, start=10, seed=2)       # second shard
    out = conv.RFCavity3DConverter([h5a, h5b]).convert_dataset(str(tmp_path / 'o.pkl'))
    d = _load(out)
    meta = d['metadata']['qoi']
    assert meta['labels'] == list(LABELS) and meta['n_failed'] == 0
    assert {'labels', 'sigma', 'beta', 'convention', 'axis', 'L_acc'} <= set(meta)
    assert (meta['sigma'], meta['beta'], meta['convention']) == (5.8e7, 1.0, 'linac')
    assert len(_fake_build.calls) == 3                                           # once per geometry
    assert sorted(d['geometry_pool']) == [0, 1, 10]
    for gid, g in d['geometry_pool'].items():
        ss = [s for s in d['samples'] if s['geom_id'] == gid]
        Y = np.stack([s['Y'] for s in ss], 1).astype(np.float64)
        ne = len(g['edges'])
        ops = _fake_build(g['X'], g['tets'], g['edges'], g['scale'], g['center'], field,
                          M=conv.from_csr_tuple(g['M'], (ne, ne)))
        want = _fake_qoi_from_dofs(ops, Y, np.array([s['Theta'][1] for s in ss], np.float64) * 1e9)
        for k, s in enumerate(ss):
            assert set(s['qoi']) == set(LABELS)
            for n in LABELS:
                assert s['qoi'][n] == pytest.approx(float(want[n][k]), rel=1e-5)
        assert np.allclose(g['center'], [0, 0, g['center'][2]]) and g['center'][2] == pytest.approx(0.3)
    ds = _all(out)
    assert np.isfinite(ds[0]['Y_qoi'].numpy()).all()


def test_convert_cli_no_qoi_and_lean(tmp_path, fake_qoi):
    import convert_3d
    h5 = _write_h5(tmp_path / 'a.h5', 'E', n_geoms=2)
    convert_3d.main(['--h5_filepath', h5, '--output_path', str(tmp_path / 'n.pkl'), '--no_qoi'])
    d = _load(tmp_path / 'n.pkl')
    assert 'qoi' not in d['metadata'] and all('qoi' not in s for s in d['samples'])
    assert _fake_build.calls == []
    convert_3d.main(['--h5_filepath', h5, '--output_path', str(tmp_path / 'f.pkl')])
    convert_3d.main(['--h5_filepath', h5, '--output_path', str(tmp_path / 'l.pkl'), '--no_operators'])
    full, lean = _load(tmp_path / 'f.pkl'), _load(tmp_path / 'l.pkl')
    assert 'M' not in lean['geometry_pool'][0]
    for a, b in zip(full['samples'], lean['samples'], strict=True):
        assert a['qoi'] == b['qoi']
    # back-fill into the no-QoI PKL (in place) == converter labels
    convert_3d.main(['--add_qoi', str(tmp_path / 'n.pkl')])
    back = _load(tmp_path / 'n.pkl')
    assert back['metadata']['qoi']['labels'] == list(LABELS)
    for a, b in zip(full['samples'], back['samples'], strict=True):
        for n in LABELS:
            assert b['qoi'][n] == pytest.approx(a['qoi'][n], rel=1e-6)


def test_attach_qoi_lean_pkl(synth, fake_qoi):
    field, path = synth
    d = _load(path)
    for g in d['geometry_pool'].values():
        for k in ('M', 'K', 'G', 'Kp'):
            g.pop(k)
    conv.attach_qoi_labels(d, verbose=False)
    assert all(set(s['qoi']) == set(LABELS) for s in d['samples'])
    assert d['metadata']['qoi']['n_failed'] == 0


def test_converter_qoi_failure_is_nan(tmp_path, fake_qoi):
    h5 = _write_h5(tmp_path / 'a.h5', 'H', n_geoms=2, seed=4)
    first = []
    _fake_build.fail_if = lambda ne: (not first and not first.append(ne))       # fail the 1st geometry
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        d = _load(conv.RFCavity3DConverter(h5).convert_dataset(str(tmp_path / 'o.pkl')))
    assert any('QoI labels failed' in str(x.message) for x in w)
    assert d['metadata']['qoi']['n_failed'] == 1
    bad = [s for s in d['samples'] if s['geom_id'] == 0]
    good = [s for s in d['samples'] if s['geom_id'] == 1]
    assert all(np.isnan(v) for s in bad for v in s['qoi'].values())
    assert all(np.isfinite(v) for s in good for v in s['qoi'].values())


# ── scripts/eval_qoi.py ───────────────────────────────────────────────────────

def test_degenerate_mask():
    ev = importlib.import_module('scripts.eval_qoi')
    m = ev.degenerate_mask(5, [[0], [1, 2], [3]], np.array([0, 0, 0, 0, 1], bool))
    assert m.tolist() == [False, True, True, False, True]
    assert ev.degenerate_mask(3, [[0], [1], [2]], [2]).tolist() == [False, False, True]


def _fake_out(ds, idx, pred=None, f_scale=1.0):
    it = ds[idx]
    K = it['Y_field'].shape[1]
    g = ds.geometry_pool[ds.active_geoms[idx]]
    fs = ds.stats
    f = it['Y_freq'].double().numpy() * fs['std'] + fs['mean']
    T = it['Y_field'].double().numpy()
    return {'field': ds.field, 'geom_id': ds.active_geoms[idx], 'shape_type': it['shape_type'],
            'X': it['X'].double().numpy(), 'tets': g['tets'], 'edges': g['edges'], 'scale': g['scale'],
            'center': g['center'], 'true': T, 'pred': T if pred is None else pred, 'f_true': f,
            'f_pred': f * f_scale, 'rel_l2': np.zeros(K), 'split': np.zeros(K, bool),
            'clusters': [[0], [1, 2]] + [[k] for k in range(3, K)]}


def test_geometry_rows_exact_and_degenerate(synth, fake_qoi):
    ev = importlib.import_module('scripts.eval_qoi')
    field, path = synth
    ds = _all(path, qoi_ops=True)
    out = _fake_out(ds, 0, pred=-3.0 * _fake_out(ds, 0)['true'])               # amplitude / sign free
    rows = ev.geometry_rows(ds[0]['qoi_ops'], out)
    assert len(rows) == len(out['f_true'])
    for r in rows:
        for n in LABELS:
            if r['degenerate']:
                assert np.isnan(r[f'{n}_rel'])
            else:
                assert abs(r[f'{n}_rel']) < 1e-9
    assert [r['degenerate'] for r in rows][:3] == [False, True, True]
    acc = [r for r in rows if r['accel']]
    assert len(acc) == 1 and not acc[0]['degenerate'] and acc[0]['rq_frac'] == pytest.approx(1.0)
    iso = [r for r in rows if not r['degenerate']]
    assert acc[0]['R_over_Q_ohm_true'] == max(r['R_over_Q_ohm_true'] for r in iso)
    rows2 = ev.geometry_rows(ds[0]['qoi_ops'], _fake_out(ds, 0, f_scale=1.01))
    assert all(r['f_rel_err'] == pytest.approx(0.01) for r in rows2)
    s = ev.summarize_qoi(rows + rows2)
    assert set(s.index.get_level_values('qoi')) == {'f', *LABELS}
    assert {'n_modes', 'modes_median', 'modes_mean', 'n_accel', 'accel_median', 'accel_mean'} <= set(s.columns)


def _tiny_lm(val_dim, stats):
    from src.training.lightning_module import GNOTLightning
    torch.manual_seed(0)
    lm = GNOTLightning(val_dim=val_dim, grid_dim=3, hidden_dim=16, n_heads=2, n_basis=10, num_field_modes=4,
                       rff_dim=8, model_type="eigenspace3d", physics_freq=True,
                       eigenspace_kwargs={"n_layers": 1}, near_deg_rel_threshold=0.02)
    lm.freq_stats = dict(stats)
    return lm.eval()


def _checkpoint(ds, tmp_path):
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    lm = _tiny_lm(ds[0]['Input_funcs'].shape[-1], ds.stats)
    trainer = pl.Trainer(max_epochs=1, limit_train_batches=1, accelerator='cpu', devices=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         default_root_dir=str(tmp_path))
    trainer.fit(lm, DataLoader(ds, batch_size=2, collate_fn=maxwell3d_collate))
    path = tmp_path / 'tiny.ckpt'
    trainer.save_checkpoint(str(path))
    return str(path)


def test_eval_qoi_main_fake(synth, fake_qoi, tmp_path):
    ev = importlib.import_module('scripts.eval_qoi')
    field, path = synth
    d = _load(path)
    conv.attach_qoi_labels(d, verbose=False)
    lab = _write_pkl(tmp_path / 'lab.pkl', d)
    ds = _all(lab)
    ckpt = _checkpoint(ds, tmp_path)
    csv = tmp_path / 'q.csv'
    rows, summary = ev.main(['--checkpoint', ckpt, '--data_path', lab, '--split', 'all', '--csv', str(csv),
                             '--summary_csv', str(tmp_path / 's.csv'), '--device', 'cpu'])
    assert csv.exists() and (tmp_path / 's.csv').exists()
    assert len(rows) == len(ds) * 4
    assert {'geom_id', 'shape_type', 'mode', 'degenerate', 'accel', 't_ops_s', 't_qoi_s', 'label_rel_diff',
            'Q0_true', 'Q0_pred', 'Q0_rel', 'Q0_label'} <= set(rows[0])
    assert max(r['label_rel_diff'] for r in rows) < 1e-5                          # stored == recomputed
    assert sum(r['accel'] for r in rows) == len(ds)
    assert ('all', 'Q0') in summary.index


# ── with the reference src/qoi/operators.py ───────────────────────────────────

@pytest.mark.parametrize('field', ['H', 'E'])
def test_real_converter_labels_and_eval(tmp_path, field):
    q = _real_qoi()
    h5 = _write_h5(tmp_path / 'a.h5', field, n_geoms=3, seed=5)
    out = conv.RFCavity3DConverter(h5).convert_dataset(str(tmp_path / 'o.pkl'))
    d = _load(out)
    assert d['metadata']['qoi']['n_failed'] == 0
    assert d['metadata']['qoi']['labels'] == list(q.QOI_LABELS)
    for s in d['samples']:
        assert s['qoi']['Q0'] > 0 and s['qoi']['G_ohm'] > 0 and np.isfinite(s['qoi']['Q0'])
    # dataset ops → same labels from the stored field
    ds = _all(out, qoi_ops=True)
    for i in range(len(ds)):
        it = ds[i]
        fs = ds.stats
        f = (it['Y_freq'].double().numpy() * fs['std'] + fs['mean']) * 1e9
        res = q.qoi_from_dofs(it['qoi_ops'], it['Y_field'].double().numpy(), f)
        for j, n in enumerate(ds.qoi_names):
            np.testing.assert_allclose(np.asarray(res[n]), it['Y_qoi'][:, j].numpy(), rtol=1e-4)
    # eval_qoi end to end; label check
    ev = importlib.import_module('scripts.eval_qoi')
    ckpt = _checkpoint(_all(out), tmp_path)
    rows, summary = ev.main(['--checkpoint', ckpt, '--data_path', out, '--split', 'all', '--device', 'cpu'])
    assert max(r['label_rel_diff'] for r in rows) < ev.LABEL_TOL
    assert all(r['t_ops_s'] > 0 for r in rows)


@pytest.mark.parametrize('field', ['H', 'E'])
def test_real_collate_matches_numpy_qoi(synth, field, tmp_path):
    """Batched QoI from the collated QoI_* keys (src/qoi/torch_qoi.py, if present) equals the
    per-geometry numpy qoi_from_dofs — checks the block layout with the real operators."""
    q = _real_qoi()
    try:
        tq = importlib.import_module('src.qoi.torch_qoi')
    except ImportError as e:
        pytest.skip(f"src.qoi.torch_qoi not available ({e})")
    d = make_dataset(n_geoms=3, n=4, n_modes=4, seed=2, field=field)
    ds = _all(_write_pkl(tmp_path / 's.pkl', d), qoi_ops=True)
    items = [ds[i] for i in range(len(ds))]
    batch = maxwell3d_collate(items)
    fs = ds.stats
    f = batch['Y_freq'].double() * fs['std'] + fs['mean']
    res = tq.qoi_torch(batch, batch['Y_field'], f)
    for b, it in enumerate(items):
        ne = it['Edges'].shape[0]
        ref = q.qoi_from_dofs(it['qoi_ops'], it['Y_field'].double().numpy()[:ne], f[b].numpy() * 1e9)
        for n in q.QOI_LABELS:
            np.testing.assert_allclose(res[n][b].numpy(), np.asarray(ref[n]), rtol=1e-6, atol=1e-12)
