"""Matrix-free operators in the projection-free training path (dataset.operators: matfree, docs/40 §10): items without any
global matrix, the cache, collate, the model's V and the loss's Grams from the element records against the assembled
path (same weights, same geometries: V, loss, every logged component, all parameter gradients), the loader options, and
train.py end to end.  The H5 fixture is the generator's calibration set (box + pillbox, as tests/test_field_labels.py)."""
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pytest
import torch

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")
from src.data import gpu_transfer, matfree, ops_cache  # noqa: E402
from src.data.field_dataset import FieldDataset, field_collate, field_collate_mf  # noqa: E402
from src.training.lightning_module import CavityLightning  # noqa: E402
from tests.test_field_labels import field_h5  # noqa: E402,F401  (module-scoped fixture: the generator's calibration set)

K = 4
pytestmark = pytest.mark.filterwarnings("ignore:FieldDataset")          # the fixture's labels were transferred between meshes
BASE = dict(train_ratio=1.0, val_ratio=0.0, build_potential=False)

# Tolerances (measured: V 1e-7 on the fixture, 3e-7 on a 10k-tet pillbox; loss and logged components <= 1e-7; gradients 2e-7; docs/40 §10.3; one float32 ulp is 6e-8).  The assembled path
# computes P·ψ in float32 with the float32-stored QP, the matrix-free path in float64 with a rebuilt P, so the two bases differ by
# float32 rounding of the (float32) network output; everything downstream is float64 on both sides.
TOL_V, TOL_LOSS, TOL_GRAD = 1e-6, 1e-6, 1e-6


@pytest.fixture(scope="module")
def pair(field_h5):
    """(assembled, matfree) train-split datasets over the same 4 geometries."""
    return (FieldDataset(str(field_h5), split="train", **BASE),
            FieldDataset(str(field_h5), split="train", operators="matfree", **BASE))


def _tree(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _tree(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _tree(v)


def _module(**kw):
    torch.manual_seed(0)
    lm = CavityLightning(val_dim=9, hidden_dim=16, n_heads=2, n_basis=8, num_field_modes=K, rff_dim=8,
                         eigenspace_kwargs={"n_layers": 1}, projection=False, freq_weight=0.0, selfsup_weight=0.0,
                         fidelity_weights=(1.0, 2.0), **kw)
    lm.freq_stats = {"mean": 0.0, "std": 1.0}
    return lm


def _capture(lm):
    logs = {}

    def log(name, value, **kw):
        logs[name] = float(value.detach() if torch.is_tensor(value) else value)
    lm.log = log
    return logs


def _step(batch, train=True, chunk=2048, matfree_batch=False):
    """(loss, logs, flat gradient or None, V) of one projection-free step of a freshly seeded model."""
    lm = _module()
    lm.matfree_chunk = chunk
    logs = _capture(lm)
    if train:
        lm.train()
        loss = lm._compute_loss_3d_plain(batch, "train")
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in lm.parameters() if p.grad is not None])
    else:
        lm.eval()
        with torch.no_grad():
            loss = lm._compute_loss_3d_plain(batch, "val")
        g = None
    with torch.no_grad():
        V = lm.model.forward_basis(batch)["basis"]
    return loss.detach(), logs, g, V


# ── items ────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_matfree_items_equal_the_assembled_items_apart_from_the_operators(pair):
    da, dm = pair
    for i in range(len(da)):
        ia, im = da[i], dm[i]
        assert set(ia) - set(im) == {"QP", "M", "K", "C", "Ct"} and set(im) - set(ia) == {"MF"}
        for k in set(im) - {"MF"}:
            a, b = ia[k], im[k]
            if torch.is_tensor(a):
                assert a.dtype == b.dtype and torch.allclose(a, b, rtol=0, atol=0, equal_nan=True), k   # bit for bit (QoI may hold NaN)
            else:
                assert a == b, k
        assert not any(t.layout != torch.strided for t in _tree(im))          # no sparse tensor anywhere
        mf = im["MF"]
        nt = int(mf["n_tets"])
        assert mf["dofs"].shape == (nt, 30) and mf["xc"].dtype == torch.float64 and mf["n_dof"] == im["BndDof"].shape[0]
        nbytes = sum(t.numel() * t.element_size() for t in mf.values() if torch.is_tensor(t))
        assert nbytes / nt <= 700


def test_matfree_item_calls_no_assembly(pair, monkeypatch):
    """The matrix-free item builder must never reach field_operators (nor an assembled-operator cache)."""
    from src.data_gen import highorder as ho
    _, dm = pair

    def boom(*a, **k):
        raise AssertionError("field_operators called by the matrix-free item")
    monkeypatch.setattr(ho, "field_operators", boom)
    item = dm[1]
    assert "MF" in item and "M" not in item


def test_matfree_record_and_tokens_equal_the_assembled_operators(pair):
    """The record's tokens (qx, qw from the cubic element map) and the label Rayleigh quotients agree with the assembled
    operators the dataset would have built (same normalisation)."""
    from src.data_gen import highorder as ho
    from src.data_gen.field_labels import model_normalisation, read_field_group
    import h5py
    da, dm = pair
    path, key = da.index[0]
    with h5py.File(path, "r") as f:
        mesh, rec = read_field_group(f[key])
    center, scale = model_normalisation(mesh)
    ops = ho.field_operators(mesh, 2, scale=scale, center=center, wall_feat=rec["wall_feat"], potential=False)
    bundle, _ = dm.load_matfree(0)
    mrec = bundle["mf"]
    assert mrec["scale"] == ops["scale"] and np.array_equal(mrec["center"], ops["center"])
    assert abs(matfree.token_points(mrec).numpy() - ops["qx"]).max() < 1e-12
    assert abs(matfree.token_weights(mrec).numpy() / ops["qw"] - 1).max() < 1e-12
    assert np.array_equal(bundle["qfeat"], ops["qfeat"].reshape(-1, ops["qfeat"].shape[-1]).astype(np.float32))
    assert np.array_equal(bundle["qw"], ops["qw"].reshape(-1).astype(np.float32))
    from src.data.field_dataset import check_labels_matfree
    err = check_labels_matfree(mrec, rec["u_model"], rec["rq_model"], 1e-5)
    assert err < 1e-6
    with pytest.raises(ValueError, match="labels do not match"):
        check_labels_matfree(mrec, rec["u_model"], rec["rq_model"] * 1.01, 1e-5)


def test_assembled_features_are_the_former_formula(field_h5):
    """highorder.token_features replaced three inlined lines of field_operators; the assembled operators still carry exactly
    the former expressions (qx = (x − c)/s, qw = W/s³, qfeat = point_features(.., qw, ..))."""
    import h5py
    from src.data_gen import highorder as ho
    from src.data_gen.field_labels import model_normalisation, read_field_group
    with h5py.File(field_h5, "r") as f:
        mesh, rec = read_field_group(f["sample_0001"])
    center, s = model_normalisation(mesh)
    ops = ho.field_operators(mesh, 2, scale=s, center=center, wall_feat=rec["wall_feat"], potential=False)
    _, _, wq, mips, qx, detJ = ho._quadrature(mesh, 2)
    W = detJ * wq[None, :]
    qxn = (qx - center) / s
    feats = ho.point_features(mesh, mips, qxn, W / s ** 3, center, s, 2, wall=rec["wall_feat"])
    assert np.array_equal(ops["qx"], qxn) and np.array_equal(ops["qw"], W / s ** 3)
    assert np.array_equal(ops["qfeat"], feats.reshape(mesh.ne, wq.size, -1))
    qxm, qwm, qfm = ho.token_inputs(mesh, 2, s, center, rec["wall_feat"])
    assert np.array_equal(qxm, qxn) and np.array_equal(qwm, ops["qw"]) and np.array_equal(qfm, ops["qfeat"])


# ── collate ──────────────────────────────────────────────────────────────────────────────────────────────────────

def test_collate_mf_pads_like_field_collate_and_holds_no_sparse_tensor(pair):
    da, dm = pair
    ba, bm = field_collate([da[0], da[2]]), field_collate_mf([dm[0], dm[2]])
    assert da[0]["X"].shape != da[2]["X"].shape                      # pillbox (57 tets) and box (46): one sample is padded
    assert bm["X"].shape[1] == max(da[0]["X"].shape[0], da[2]["X"].shape[0]) and not bm["Mask"].all()
    assert set(ba) - set(bm) == {"QP", "M", "K", "C", "Ct", "SymmetricMK"} and set(bm) - set(ba) == {"MF"}
    for k in set(bm) - {"MF"}:
        if torch.is_tensor(bm[k]):
            assert torch.equal(ba[k], bm[k]) or torch.allclose(ba[k], bm[k], equal_nan=True), k
    assert isinstance(bm["MF"], list) and len(bm["MF"]) == 2
    assert not any(t.layout != torch.strided for t in _tree(bm))
    nbytes = sum(t.numel() * t.element_size() for t in _tree(bm))
    nbytes_a = sum(t.values().numel() * t.values().element_size() + t.col_indices().numel() * 4 if t.layout != torch.strided
                   else t.numel() * t.element_size() for t in _tree(ba))
    assert nbytes < nbytes_a / 4                       # the operators dominate the assembled batch (tiny meshes: still 4x)


# ── model + loss: equivalence with the assembled path ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("train", [True, False])
def test_matfree_step_equals_the_assembled_step(pair, train):
    """Same weights, same geometries (batch of 2 samples of different size): V, the loss, every logged component and every
    parameter gradient of the matrix-free step equal the assembled step's (the train step; the validation step has no
    gradient).  Measured differences are printed (pytest -s)."""
    da, dm = pair
    for ids in ((0, 2), (3, 1), (1,)):      # pillbox (57 tets) + box (46): different sizes, padded
        ba, bm = field_collate([da[i] for i in ids]), field_collate_mf([dm[i] for i in ids])
        la, loga, ga, Va = _step(ba, train)
        lm_, logm, gm, Vm = _step(bm, train)
        dV = ((Va - Vm).abs().max() / Va.abs().max()).item()
        dl = abs(la.item() - lm_.item()) / abs(la.item())
        dlog = max(abs(loga[k] - logm[k]) / max(abs(loga[k]), 1e-12) for k in loga)
        dg = ((ga - gm).norm() / ga.norm()).item() if train else 0.0
        print(f"ids {ids} train={train}: V {dV:.1e}  loss {dl:.1e}  logs {dlog:.1e}  grad {dg:.1e}")
        assert loga.keys() == logm.keys() and len(loga) == 12 + 0
        assert dV < TOL_V and dl < TOL_LOSS and dlog < TOL_LOSS and dg < TOL_GRAD
        assert Va.shape == Vm.shape and Va.dtype == Vm.dtype == torch.float32
        assert (Vm[~bm["DofMask"]] == 0).all()                       # padded DOFs are zero
        for i, b in enumerate(ids):                                  # wall DOFs are zero
            assert (Vm[i, :dm[b]["BndDof"].shape[0]][dm[b]["BndDof"]] == 0).all()


def test_matfree_step_is_independent_of_the_chunk_size(pair):
    _, dm = pair
    bm = field_collate_mf([dm[0], dm[2]])
    ref = _step(bm, True, chunk=2048)
    for chunk in (3, 50, None):
        out = _step(bm, True, chunk=chunk)
        assert abs(out[0].item() - ref[0].item()) <= 1e-6 * abs(ref[0].item())
        assert ((out[2] - ref[2]).norm() / ref[2].norm()).item() < 1e-6
        assert ((out[3] - ref[3]).abs().max() / ref[3].abs().max()).item() < 1e-6


def test_step_operators_are_freed_and_shared(pair):
    """The element matrices and ψ maps are built once per sample per step (shared by V and the Grams), under no_grad, and the
    step object holds nothing after the loss function returned."""
    _, dm = pair
    bm = field_collate_mf([dm[0], dm[2]])
    calls = {"mats": 0, "psi_map": 0}
    orig = (matfree.element_matrices, matfree.psi_map)

    def count(name, fn):
        def wrapped(*a, **k):
            calls[name] += 1
            assert not torch.is_grad_enabled()
            return fn(*a, **k)
        return wrapped
    matfree.element_matrices, matfree.psi_map = count("mats", orig[0]), count("psi_map", orig[1])
    try:
        lm = _module()
        _capture(lm)
        lm.train()
        loss = lm._compute_loss_3d_plain(bm, "train")
        loss.backward()
    finally:
        matfree.element_matrices, matfree.psi_map = orig
    assert calls == {"mats": 2, "psi_map": 2}
    so = matfree.StepOperators(bm["MF"], 8, torch.device("cpu"))
    so.mats(0), so.Ce(0)
    so.free()
    assert not so._mats and not so._Ce and not so._rec


def test_gram_backward_without_graph_equals_the_autograd_through_the_gathers(pair):
    _, dm = pair
    bundle, _ = dm.load_matfree(1)
    tr = matfree.to_torch(bundle["mf"])
    Me, Ke = matfree.element_matrices(tr)
    n = tr["n_dof"]
    gen = torch.Generator().manual_seed(1)
    V0 = torch.randn(n, 5, dtype=torch.float64, generator=gen)
    T = torch.randn(n, 2, dtype=torch.float64, generator=gen)
    Wt = torch.randn(7, 7, dtype=torch.float64, generator=gen)
    grads = []
    for plain in (False, True):
        V = V0.clone().requires_grad_()
        A = Ke.clone().requires_grad_(plain)           # an A that requires grad takes the autograd route
        G = matfree.gram(A, tr["dofs"], V, T, chunk=40)
        (G * Wt).sum().backward()
        grads.append((V.grad, G.detach()))
    assert torch.allclose(grads[0][1], grads[1][1], rtol=1e-13, atol=0)
    assert ((grads[0][0] - grads[1][0]).abs().max() / grads[1][0].abs().max()).item() < 1e-12


# ── cache ────────────────────────────────────────────────────────────────────────────────────────────────────────

def _forbid_ngsolve(monkeypatch):
    from src.data_gen import field_labels
    from src.data_gen import highorder as ho

    def boom(*a, **k):
        raise AssertionError("NGSolve work on a cache hit")
    monkeypatch.setattr(field_labels, "read_field_group", boom)
    monkeypatch.setattr(matfree, "build_record", boom)
    monkeypatch.setattr(ho, "token_inputs", boom)
    monkeypatch.setattr(ho, "field_operators", boom)


def test_matfree_cache_writes_its_own_files_and_later_epochs_skip_ngsolve(field_h5, tmp_path, monkeypatch):
    cdir = tmp_path / "c"
    ds = FieldDataset(str(field_h5), split="train", operators="matfree", cache_dir=str(cdir), **BASE)
    assert ds.verify_labels                                         # default with a cache_dir: checked once at the build
    first = [ds[i] for i in range(len(ds))]
    files = sorted(os.listdir(cdir))
    assert len(files) == 4 and all(f.endswith(".mf1.pkl") for f in files)       # nothing else: no .tmp, no .ops2.pkl
    assert not any(f.endswith(ops_cache.SUFFIX) for f in files)
    for f in files:
        mb = os.path.getsize(cdir / f) / 1e6
        assert mb < 5, f
    _forbid_ngsolve(monkeypatch)
    again = [ds[i] for i in range(len(ds))]
    for a, b in zip(first, again):
        assert a.keys() == b.keys()
        for ta, tb in zip(_tree(a), _tree(b)):
            assert torch.equal(ta, tb)
    fresh = FieldDataset(str(field_h5), split="train", operators="matfree", cache_dir=str(cdir), **BASE)   # a new process
    assert all(torch.equal(x, y) for i in range(len(fresh)) for x, y in zip(_tree(fresh[i]), _tree(first[i])))


def test_matfree_cache_is_never_confused_with_the_operator_cache(field_h5, tmp_path, monkeypatch):
    cdir = tmp_path / "c"
    asm = FieldDataset(str(field_h5), split="train", cache_dir=str(cdir), train_ratio=1.0, val_ratio=0.0, build_potential=False)
    asm[0]
    mf = FieldDataset(str(field_h5), split="train", operators="matfree", cache_dir=str(cdir), **BASE)
    g = asm.active_geoms[0]
    assert mf.active_geoms[0] == g
    assert mf._cache_path(g, matfree=True) != asm._cache_path(g, False) and mf._cache_path(g, matfree=True).endswith(".mf1.pkl")
    mf[0]
    names = sorted(os.listdir(cdir))
    assert sum(n.endswith(".mf1.pkl") for n in names) == 1 and sum(n.endswith(".nopot.ops2.pkl") for n in names) == 1
    # an operator cache is not a matrix-free cache and the other way round
    with pytest.raises(ops_cache.OpsCacheError):
        ops_cache.read_mf(asm._cache_path(g, False))
    with pytest.raises(ops_cache.OpsCacheError):
        ops_cache.read(mf._cache_path(g, matfree=True))
    assert asm[0]["M"].shape[0] == mf[0]["BndDof"].shape[0]


def test_matfree_cache_rebuilds_on_corrupt_files_old_formats_and_other_ngsolve_versions(field_h5, tmp_path):
    cdir = tmp_path / "c"
    ds = FieldDataset(str(field_h5), split="train", operators="matfree", cache_dir=str(cdir), verify_labels=False, **BASE)
    ref = ds[0]
    cp = ds._cache_path(ds.active_geoms[0], matfree=True)
    good = pickle.load(open(cp, "rb"))
    for name, payload in (("format", {**good, "mf_format": 99}), ("matfree format", {**good, "matfree_format": 99}),
                          ("ngsolve", {**good, "ngsolve": "0.0.0"}), ("kind", {"kind": "ops"})):
        with open(cp, "wb") as fh:
            pickle.dump(payload, fh)
        with pytest.warns(UserWarning, match="rebuilding the matrix-free record|not a matrix-free"):
            got = ds[0]
        assert all(torch.equal(a, b) for a, b in zip(_tree(got), _tree(ref))), name
        assert pickle.load(open(cp, "rb"))["mf_format"] == ops_cache.MF_FORMAT          # rewritten
    with open(cp, "wb") as fh:
        fh.write(b"not a pickle")
    with pytest.warns(UserWarning, match="rebuilding"):
        ds[0]
    assert not [n for n in os.listdir(cdir) if n.endswith(".tmp")]


def test_matfree_cache_write_is_atomic(field_h5, tmp_path, monkeypatch):
    cdir = tmp_path / "c"
    ds = FieldDataset(str(field_h5), split="train", operators="matfree", cache_dir=str(cdir), verify_labels=False, **BASE)

    def broken(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(pickle, "dump", broken)
    with pytest.raises(RuntimeError):
        ds[0]
    assert os.listdir(cdir) == []                                   # neither a partial file nor a temporary one


def test_verify_labels_catches_a_mismatching_label_record(field_h5, tmp_path):
    import h5py
    bad = tmp_path / "bad.h5"
    with h5py.File(field_h5, "r") as src, h5py.File(bad, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        for key in ("sample_0001",):
            src.copy(key, dst)
        dst["sample_0001"]["rq_model"][...] = np.asarray(dst["sample_0001"]["rq_model"]) * 1.5
    ds = FieldDataset(str(bad), split="train", operators="matfree", verify_labels=True, **BASE)
    with pytest.raises(ValueError, match="labels do not match"):
        ds[0]
    ds = FieldDataset(str(bad), split="train", operators="matfree", **BASE)       # the default, without a cache_dir: checked too
    assert ds.verify_labels
    with pytest.raises(ValueError, match="labels do not match"):
        ds[0]
    FieldDataset(str(bad), split="train", operators="matfree", verify_labels=False, **BASE)[0]       # unchecked: loads


# ── guards, loaders, transfer ────────────────────────────────────────────────────────────────────────────────────

def test_option_guards(field_h5):
    import train
    from src.config import load_config
    with pytest.raises(ValueError, match="builds no kernel potentials"):
        FieldDataset(str(field_h5), split="train", operators="matfree", train_ratio=1.0, val_ratio=0.0)
    with pytest.raises(ValueError, match="operators must be one of"):
        FieldDataset(str(field_h5), split="train", operators="sparse", **BASE)
    cfg = load_config("configs/field_3d.yaml", overrides={"dataset.data_path": str(field_h5), "dataset.operators": "matfree"})
    with pytest.raises(ValueError, match="needs training.projection: false"):
        train._datasets_field(cfg.dataset, 42, None, projection=True)
    cfg.dataset.operators = "banana"
    with pytest.raises(ValueError, match="operators must be one of"):
        train._datasets_field(cfg.dataset, 42, None, projection=False)
    cfg.dataset.operators = "matfree"
    tr, va, te = train._datasets_field(cfg.dataset, 42, None, projection=False)
    assert (tr.operators, va.operators, te.operators) == ("matfree", "matfree", "assembled")
    assert not tr.build_potential and not va.build_potential and te.build_potential
    cfg.dataset.operators = None
    assert {d.operators for d in train._datasets_field(cfg.dataset, 42, None, projection=True)} == {"assembled"}
    for f in ("field_3d.yaml", "field_3d_noproj.yaml"):
        c = load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", f))
        assert c.dataset.operators == "assembled" and c.dataset.verify_labels is None and c.training.matfree_chunk == 2048


def test_forward_of_a_matfree_batch_is_refused_and_the_test_step_needs_the_assembled_batch(pair):
    _, dm = pair
    bm = field_collate_mf([dm[0]])
    lm = CavityLightning(val_dim=9, hidden_dim=16, n_heads=2, n_basis=8, num_field_modes=K, rff_dim=8,
                         eigenspace_kwargs={"n_layers": 1})
    with pytest.raises(ValueError, match="matrix-free batch"):
        lm.model(bm)
    lm = _module()
    with pytest.raises(ValueError, match="computes its own basis"):
        lm._compute_loss_3d_plain(bm, None, basis=torch.zeros(1, 1, 8))


def test_loader_options_for_matfree_datasets(pair, monkeypatch):
    """Collate and pinning follow the dataset: matfree train / val → field_collate_mf (dense), the assembled test split keeps its
    collate; pin_memory is on with CUDA, with or without gpu_prefetch (the two options do not conflict)."""
    import types

    import train
    from src.data.field_dataset import field_collate_parts
    da, dm = pair
    tc = types.SimpleNamespace(num_workers=0, batch_size=2)
    tr, va, te = train.make_loaders(tc, dm, dm, da)
    assert tr.collate_fn is field_collate_mf and va.collate_fn is field_collate_mf and te.collate_fn is field_collate
    assert not tr.pin_memory                                                    # no CUDA here
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for gp in (False, True):
        tc = types.SimpleNamespace(num_workers=0, batch_size=2, gpu_prefetch=gp)
        tr, va, te = train.make_loaders(tc, dm, dm, da)
        assert tr.collate_fn is field_collate_mf and va.collate_fn is field_collate_mf and tr.pin_memory and va.pin_memory
        assert te.collate_fn is (field_collate_parts if gp else field_collate) and te.pin_memory == gp


def test_matfree_batches_survive_workers_pinning_and_the_transfer_paths(pair, monkeypatch):
    from torch.utils.data._utils.pin_memory import pin_memory
    da, dm = pair
    ref = field_collate_mf([dm[0], dm[1]])
    loader = torch.utils.data.DataLoader(dm, batch_size=2, num_workers=2, collate_fn=field_collate_mf)    # items 0, 1
    got = next(iter(loader))
    assert [r["n_tets"] for r in got["MF"]] == [r["n_tets"] for r in ref["MF"]]
    assert all(torch.equal(a, b) for a, b in zip(_tree(got), _tree(ref)))
    pinned = []
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda t, *a, **k: (pinned.append(t), t)[1])
    out = pin_memory(ref, None)
    assert len(pinned) == len(list(_tree(ref)))                                 # every dense tensor, also inside the records
    assert out["MF"][0]["center"] is ref["MF"][0]["center"]                     # numpy / ints / floats pass through
    # both transfer paths (CPU: no stream, no pinning) keep the batch and the step unchanged
    moved = gpu_transfer.wait_ready(gpu_transfer.to_device(ref, "cpu"))
    assert all(torch.equal(a, b) for a, b in zip(_tree(moved), _tree(ref)))
    for prefetch in (False, True):
        lm = _module()
        lm.gpu_prefetch = prefetch
        b = lm.transfer_batch_to_device(ref, torch.device("cpu"), 0)
        assert all(torch.equal(x, y) for x, y in zip(_tree(b), _tree(ref)))
        assert isinstance(b["MF"], list) and b["MF"][0]["n_dof"] == ref["MF"][0]["n_dof"]


def test_lightning_fit_matfree_follows_the_assembled_run(pair):
    """Through Lightning's call path (loader → transfer → step → optimizer): 2 epochs of train + validation with matrix-free batches
    track the assembled run (losses per step; the weights stay within a few float32 ulps of each other per step)."""
    import pytorch_lightning as pl
    da, dm = pair

    class Rec(pl.Callback):
        def __init__(self):
            self.losses = []

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.losses.append(outputs["loss"].detach().clone())

    runs = []
    for ds, collate in ((da, field_collate), (dm, field_collate_mf)):
        lm = _module()
        rec = Rec()
        tl = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate)
        vl = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate)
        pl.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False,
                   callbacks=[rec], num_sanity_val_steps=0, enable_model_summary=False, gradient_clip_val=1.0,
                   ).fit(lm, tl, vl)
        runs.append((torch.stack(rec.losses), {n: p.detach().clone() for n, p in lm.named_parameters()}))
    la, lb = runs[0][0], runs[1][0]
    assert la.shape == lb.shape == (4,)
    assert ((la - lb).abs() / la.abs()).max().item() < 1e-4
    for n in runs[0][1]:
        a, b = runs[0][1][n], runs[1][1][n]
        assert ((a - b).abs().max() / (a.abs().max() + 1e-12)).item() < 1e-3, n


# ── train.py end to end ──────────────────────────────────────────────────────────────────────────────────────────

_SMALL = ["model.embed_dim=16", "model.n_heads=2", "model.n_basis=8", f"model.num_field_modes={K}", "model.rff_dim=8",
          "model.eigenspace.n_layers=1", "model.eigenspace.grad_checkpoint=false", "training.batch_size=1",
          "training.num_workers=0", "dataset.train_ratio=0.5", "dataset.val_ratio=0.25", "dataset.operators=matfree"]


def _argv(monkeypatch, h5, tmp_path, *extra, fast=False, config="configs/field_3d_noproj.yaml"):
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", config, *(["--fast_dev_run"] if fast else []), "--override",
                                      f"dataset.data_path={h5}", *_SMALL, f"training.log_dir={tmp_path}", *extra])


def test_train_py_fast_dev_run_with_matfree_train_and_val_and_an_assembled_projected_test(field_h5, tmp_path, monkeypatch):
    import train
    from src.models import hcurl
    seen = {"keys": {}, "stage": None, "cg": {"train": 0, "val": 0, "test": 0}}
    orig_pcg = hcurl._pcg

    def stage(name, orig):
        def wrapped(self, batch, batch_idx):
            seen["stage"] = name
            seen["keys"][name] = set(batch)
            return orig(self, batch, batch_idx)
        return wrapped
    for name, attr in (("train", "training_step"), ("val", "validation_step"), ("test", "test_step")):
        monkeypatch.setattr(CavityLightning, attr, stage(name, getattr(CavityLightning, attr)))
    monkeypatch.setattr(hcurl, "_pcg", lambda *a, **k: (seen["cg"].__setitem__(seen["stage"], seen["cg"][seen["stage"]] + 1),
                                                         orig_pcg(*a, **k))[1])
    _argv(monkeypatch, field_h5, tmp_path, fast=True)
    train.main()
    sparse = {"M", "K", "C", "Ct", "QP", "G", "Gt", "Kp"}
    assert "MF" in seen["keys"]["train"] and "MF" in seen["keys"]["val"] and "MF" not in seen["keys"]["test"]
    assert not (sparse & seen["keys"]["train"]) and not (sparse & seen["keys"]["val"])
    assert {"M", "K", "G", "Gt", "Kp", "C"} <= seen["keys"]["test"]
    assert seen["cg"]["train"] == seen["cg"]["val"] == 0 and seen["cg"]["test"] > 0


def test_train_py_two_epochs_with_matfree_then_resume(field_h5, tmp_path, monkeypatch):
    import train
    _argv(monkeypatch, field_h5, tmp_path, "training.exp_name=mf", "training.max_epochs=2", "training.patience=5",
          f"dataset.cache_dir={tmp_path / 'cache'}")
    train.main()
    run = tmp_path / "mf"
    prog = json.load(open(run / "progress.json"))
    assert prog["status"] == "finished" and len(prog["history"]) == 2
    h = prog["history"][-1]
    for pre in ("train", "val"):
        for key in ("loss", "span_loss", "ortho_loss", "span_rel_l2", "mass_diag_ratio", "grad_frac_lb"):
            assert f"{pre}/{key}" in h, f"{pre}/{key}"
    assert prog["best"]["monitor"] == "val/span_loss" and (run / "last.ckpt").exists()
    t = prog["test"]
    assert t["test/kp_cg_iters"] > 0 and t["test/loss_plain"] > 0                      # the test split: assembled, projected
    names = os.listdir(tmp_path / "cache")
    assert any(n.endswith(".mf1.pkl") for n in names) and any(n.endswith(ops_cache.SUFFIX) for n in names)   # train/val | test
    # resume from last.ckpt: one more epoch, the same data path
    _argv(monkeypatch, field_h5, tmp_path, "training.exp_name=mf", "training.max_epochs=3", "training.patience=5",
          f"dataset.cache_dir={tmp_path / 'cache'}")
    monkeypatch.setattr(sys, "argv", sys.argv + ["--resume", str(run / "last.ckpt")])
    train.main()
    prog2 = json.load(open(run / "progress.json"))
    assert prog2["status"] == "finished" and len(prog2["history"]) == 3


def test_matfree_with_the_projected_config_is_refused(field_h5, tmp_path, monkeypatch):
    import train
    _argv(monkeypatch, field_h5, tmp_path, fast=True, config="configs/field_3d.yaml")
    with pytest.raises(ValueError, match="needs training.projection: false"):
        train.main()


def test_default_operators_is_the_assembled_path_unchanged(field_h5, tmp_path, monkeypatch):
    """No dataset.operators key / assembled: items carry the sparse operators and the plain collate is used."""
    import train
    seen = {}
    orig = CavityLightning.training_step

    def spy(self, batch, batch_idx):
        seen["keys"] = set(batch)
        return orig(self, batch, batch_idx)
    monkeypatch.setattr(CavityLightning, "training_step", spy)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "configs/field_3d_noproj.yaml", "--fast_dev_run", "--override",
                                      f"dataset.data_path={field_h5}", *[a for a in _SMALL if "operators" not in a],
                                      f"training.log_dir={tmp_path}"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train.main()
    assert {"M", "K", "C", "Ct", "QP"} <= seen["keys"] and "MF" not in seen["keys"]
