"""Tests for src/viz/predict.py (A2, docs/viz3d_contract.md scratchpad)."""
import importlib
import math
import pickle

import numpy as np
import pytest
import torch

pytest.importorskip("skfem")

import pytorch_lightning as pl                                        # noqa: E402
from torch.utils.data import DataLoader                                # noqa: E402

from src.data.dataset_3d import Maxwell3DDataset, maxwell3d_collate    # noqa: E402
from src.models.hcurl import mode_rel_l2                               # noqa: E402
from src.training.lightning_module import GNOTLightning                # noqa: E402
from src.viz.predict import _align_modes, load, pick, predict          # noqa: E402
from tests.maxwell3d_synth import make_dataset                         # noqa: E402

K_MODES = 6
ALL_KEYS = {'geom_id', 'shape_type', 'X', 'tets', 'edges', 'scale', 'center',
            'true', 'pred', 'err', 'f_true', 'f_pred', 'rel_l2', 'split', 'clusters'}


@pytest.fixture(scope="module")
def pkl(tmp_path_factory):
    path = tmp_path_factory.mktemp("viz3d") / "synth3d.pkl"
    with open(path, "wb") as f:
        pickle.dump(make_dataset(n_geoms=6, n=4, n_modes=K_MODES, seed=1), f)
    return str(path)


@pytest.fixture(scope="module")
def ds(pkl):
    return Maxwell3DDataset(pkl, split="test", train_ratio=0.0, val_ratio=0.0)   # all geometries


def _module(val_dim, **kw):
    torch.manual_seed(0)
    return GNOTLightning(val_dim=val_dim, grid_dim=3, hidden_dim=16, n_heads=2, n_basis=10,
                         num_field_modes=4, rff_dim=8, model_type="eigenspace3d", physics_freq=True,
                         eigenspace_kwargs={"n_layers": 1}, near_deg_rel_threshold=0.02, **kw)


@pytest.fixture(scope="module")
def lm(ds):
    m = _module(ds[0]["Input_funcs"].shape[-1])
    m.freq_stats = dict(ds.stats)
    return m.eval()


def _geom(ds, g_id):
    s_idx = ds.geom_to_samples[g_id]
    g_key = ds.samples_metadata[s_idx[0]]['geom_id']
    return ds.geometry_pool[g_key]


# ── pick() ───────────────────────────────────────────────────────────────────

def test_pick_one_per_shape_type_then_fill(ds):
    idx = pick(ds, n=3)
    assert len(idx) == 3 and len(set(idx)) == 3
    order_seen = []
    for i in range(len(ds)):
        t = _geom(ds, ds.active_geoms[i])['shape_type']
        if t not in order_seen:
            order_seen.append(t)
    got_types = [_geom(ds, ds.active_geoms[i])['shape_type'] for i in idx]
    assert got_types[:len(order_seen)] == order_seen[:len(idx)]


def test_pick_deterministic(ds):
    assert pick(ds, n=3) == pick(ds, n=3)
    assert pick(ds, n=4, by='shape_type') == pick(ds, n=4)


def test_pick_clips_and_covers_when_n_ge_len(ds):
    idx = pick(ds, n=1000)
    assert sorted(idx) == list(range(len(ds)))


# ── _align_modes: cluster / isolated properties (self-contained) ───────────

def test_align_modes_isolated_flips_sign_to_match_target():
    t = np.array([1.0, 0.0, 0.0, 0.0])
    f = -t.copy()
    F, T = torch.from_numpy(f[:, None]), torch.from_numpy(t[:, None])
    MT = torch.from_numpy((np.eye(4) @ t)[:, None])
    pred = _align_modes(F, T, MT, [[0]])
    np.testing.assert_allclose(pred.numpy(), T.numpy(), atol=1e-12)


def test_align_modes_isolated_error_equals_rel_l2():
    """‖pred - true‖_M == rel_l2 for an isolated (sign-aligned, unit-M-norm) mode."""
    rng = np.random.default_rng(2)
    Ne = 6
    t = rng.standard_normal(Ne)
    t /= np.linalg.norm(t)
    raw = rng.standard_normal(Ne)
    raw /= np.linalg.norm(raw)
    f = 0.3 * t + 0.7 * raw
    f /= np.linalg.norm(f)
    F, T = torch.from_numpy(f[:, None]), torch.from_numpy(t[:, None])
    MT = torch.from_numpy((np.eye(Ne) @ t)[:, None])
    pred = _align_modes(F, T, MT, [[0]])
    rl = mode_rel_l2(F, T, MT, [[0]])
    err_norm = float(np.linalg.norm(pred.numpy()[:, 0] - t))
    assert math.isclose(err_norm, float(rl[0]), rel_tol=1e-9, abs_tol=1e-9)


def test_align_modes_cluster_reproduces_target_when_span_matches():
    """F spans exactly the same 2-D subspace as the degenerate pair T (a
    rotation within that span) -> the M-projection reproduces T to 1e-10."""
    Ne = 5
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((Ne, Ne)))
    e1, e2 = Q[:, 0], Q[:, 1]
    theta = 0.7
    f1 = math.cos(theta) * e1 + math.sin(theta) * e2
    f2 = -math.sin(theta) * e1 + math.cos(theta) * e2
    T = np.stack([e1, e2], axis=1)
    F = np.stack([f1, f2], axis=1)
    Ft, Tt = torch.from_numpy(F), torch.from_numpy(T)
    MT = torch.from_numpy(np.eye(Ne) @ T)
    pred = _align_modes(Ft, Tt, MT, [[0, 1]])
    np.testing.assert_allclose(pred.numpy(), T, atol=1e-10)
    rl = mode_rel_l2(Ft, Tt, MT, [[0, 1]])
    assert float(rl.abs().max()) < 1e-10


def test_align_modes_cluster_partial_span_gives_nonzero_subspace_error():
    """F's span only partially covers T -> pred != true and rel_l2 > 0,
    but pred still has the target's M-norm (rescaled projection)."""
    Ne = 5
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.standard_normal((Ne, Ne)))
    e1, e2, e3 = Q[:, 0], Q[:, 1], Q[:, 2]
    T = np.stack([e1, e2], axis=1)
    F = np.stack([e1, e3], axis=1)                      # only e1 in common
    Ft, Tt = torch.from_numpy(F), torch.from_numpy(T)
    MT = torch.from_numpy(np.eye(Ne) @ T)
    pred = _align_modes(Ft, Tt, MT, [[0, 1]])
    rl = mode_rel_l2(Ft, Tt, MT, [[0, 1]])
    assert float(rl[1]) > 1e-3   # e2 not in span(F) -> real subspace error
    np.testing.assert_allclose(np.linalg.norm(pred.numpy(), axis=0), [1.0, 1.0], atol=1e-10)
    assert not np.allclose(pred.numpy(), T, atol=1e-6)


# ── predict(): shape / dtype / basic invariants ─────────────────────────────

def test_predict_output_keys_shapes_dtypes(ds, lm):
    out = predict(lm, ds, 0)
    assert set(out) == ALL_KEYS
    g_id = out['geom_id']
    geom = _geom(ds, g_id)
    Nv, Ne = len(geom['X']), len(geom['edges'])
    K = out['pred'].shape[1]
    assert out['X'].shape == (Nv, 3)
    assert out['tets'].shape[1:] == (4,)
    assert out['edges'].shape == (Ne, 2)
    assert out['true'].shape == (Ne, K) == out['pred'].shape == out['err'].shape
    assert out['f_true'].shape == (K,) == out['f_pred'].shape
    assert out['rel_l2'].shape == (K,) == out['split'].shape
    assert isinstance(out['scale'], float)
    assert out['center'].shape == (3,)
    assert out['split'].dtype == bool
    np.testing.assert_allclose(out['err'], out['pred'] - out['true'])
    for arr in (out['X'], out['true'], out['pred'], out['err'], out['f_true'],
               out['f_pred'], out['rel_l2'], out['center']):
        assert arr.dtype == np.float64
    assert np.array_equal(np.isnan(out['rel_l2']), out['split'])


def test_predict_split_flag_consistent_with_clusters_3d(ds, lm):
    for idx in range(len(ds)):
        out = predict(lm, ds, idx)
        assert np.array_equal(np.isnan(out['rel_l2']), out['split'])
        flat_inside = {k for cl in out['clusters'] for k in cl}
        K = out['rel_l2'].shape[0]
        split_idx = set(np.nonzero(out['split'])[0].tolist())
        assert flat_inside | split_idx == set(range(K))
        assert not (flat_inside & split_idx)


def test_predict_isolated_error_equals_rel_l2(ds, lm):
    """For every isolated (non-clustered, non-split) mode of a real prediction,
    ‖pred - true‖_M == rel_l2 (both unit-M-norm, sign aligned)."""
    for idx in range(len(ds)):
        out = predict(lm, ds, idx)
        g_id = out['geom_id']
        M, _, _, _ = ds.operators(g_id)
        for cl in out['clusters']:
            if len(cl) != 1:
                continue
            k = cl[0]
            err = out['pred'][:, k] - out['true'][:, k]
            m_err = math.sqrt(float(err @ (M @ err)))
            assert math.isclose(m_err, out['rel_l2'][k], rel_tol=1e-6, abs_tol=1e-8)


# ── predict() vs scripts/eval_3d.py: exact same numbers ────────────────────

def test_rel_l2_and_freq_match_eval_3d(ds, lm):
    # batch_size=1: identical padding to predict()'s single-item batch, so the
    # model's (float32) forward pass takes the exact same path and the numbers
    # match to double precision. A batch_size > 1 in eval_3d pads samples to a
    # shared Ne_max/Nv_max, which perturbs the float32 basis by ~1e-7 relative
    # (ordinary batching/padding float32 noise, unrelated to this module) —
    # not tested here since predict() itself only ever forwards one geometry.
    eval_3d = importlib.import_module("scripts.eval_3d")
    rows, _ = eval_3d.evaluate(lm, ds, device="cpu", batch_size=1)
    by_geom = {r["geom_id"]: r for r in rows}
    for idx in range(len(ds)):
        out = predict(lm, ds, idx)
        row = by_geom[out["geom_id"]]
        K = out["rel_l2"].shape[0]
        expect_rl = np.array([row[f"rel_l2_{k}"] for k in range(K)])
        expect_fp = np.array([row[f"f_pred_{k}"] for k in range(K)])
        expect_ft = np.array([row[f"f_true_{k}"] for k in range(K)])
        np.testing.assert_allclose(out["rel_l2"], expect_rl, rtol=1e-8, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(out["f_pred"], expect_fp, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(out["f_true"], expect_ft, rtol=1e-8, atol=1e-10)


# ── load(): checkpoint round-trip ───────────────────────────────────────────

def test_load_roundtrip(pkl, ds, tmp_path):
    lm0 = _module(ds[0]["Input_funcs"].shape[-1])
    lm0.freq_stats = dict(ds.stats)
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=maxwell3d_collate)
    trainer = pl.Trainer(max_epochs=1, limit_train_batches=1, accelerator="cpu", devices=1,
                         logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, default_root_dir=str(tmp_path))
    trainer.fit(lm0, dl)
    ckpt_path = tmp_path / "tiny.ckpt"
    trainer.save_checkpoint(str(ckpt_path))

    lm2, ds2 = load(str(ckpt_path), pkl, split="all", device="cpu")
    assert len(ds2) == len(ds)
    assert lm2.freq_stats == ds2.stats
    assert not lm2.training

    out = predict(lm2, ds2, 0)
    assert out["pred"].shape == out["true"].shape
    assert set(out) == ALL_KEYS
