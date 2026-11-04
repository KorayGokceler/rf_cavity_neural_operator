"""Throughput options of the projection-free training (docs/38 §11): the host-sync-free Cholesky fallback and training
step, the diagnostic-metric interval, the sparse-parts transfer path, the operator cache without zlib and its faster
decoder, the loader options and the step-time profiler.  CPU only: what needs a GPU is listed in docs/38 §11."""
import json
import os
import types

import numpy as np
import pytest
import scipy.sparse as sp
import torch

pytest.importorskip("ngsolve")
from src.data import gpu_transfer, ops_cache  # noqa: E402
from src.data.field_dataset import field_collate, field_collate_parts  # noqa: E402
from src.training.lightning_module import CavityLightning, _jacobi_cholesky  # noqa: E402
from tests.test_projection_free import K_MODES, _build, _capture, _item, _module  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def pill():
    from netgen.occ import Axes, Cylinder, Z
    return _build(Cylinder(Axes((0, 0, 0), Z), r=0.1, h=0.1), 0.05, K_MODES)


@pytest.fixture(scope="module")
def box():
    from netgen.occ import Box, Pnt
    return _build(Box(Pnt(0, 0, 0), Pnt(1.0, 0.8, 0.6)), 0.25, K_MODES)


def _items(pill, box, full=False, ct=True):
    """Two real items (pillbox LF, box HF: the fidelity table is used) without the kernel potentials unless full."""
    i = 1 if full else 2
    out = []
    for g, fid in ((pill, 0), (box, 1)):
        it = _item(g[i], g[3], g[4], with_ct=ct)
        it["Fidelity"] = torch.tensor([fid])
        out.append(it)
    return out


def _noproj(**kw):
    lm = _module(projection=False, freq_weight=0.0, selfsup_weight=0.0, fidelity_weights=(1.0, 2.0), **kw)
    lm.optimizers = lambda: types.SimpleNamespace(param_groups=[{"lr": 1e-3}])
    return lm


def _grads(lm):
    return {n: p.grad.clone() for n, p in lm.named_parameters() if p.grad is not None}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ── the Cholesky fallback without a host sync ───────────────────────────────────────────────────────

def _former_jacobi_cholesky(G, ridge):
    """The implementation before docs/38 §11 (Python `if` on `(info > 0).any()`)."""
    diag = torch.diagonal(G, dim1=-2, dim2=-1)
    alive = diag > 1e-12 * diag.amax(-1, keepdim=True).clamp(min=0)
    d = torch.where(alive & (diag > 0), diag.clamp(min=1e-300).rsqrt(), torch.zeros_like(diag))
    eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    Gs = G * d.unsqueeze(-1) * d.unsqueeze(-2)
    Gs = 0.5 * (Gs + Gs.transpose(-1, -2))
    C, info = torch.linalg.cholesky_ex(Gs + ridge * eye)
    if bool((info > 0).any()):
        C = torch.linalg.cholesky(Gs + max(ridge, 1e-9) * 1e3 * eye)
    return C, d


def _gram(n, seed, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, n, n + 2, dtype=torch.float64, generator=g)
    return A @ A.transpose(-1, -2) + shift * torch.eye(n, dtype=torch.float64)


def test_jacobi_cholesky_is_bit_identical_when_the_fallback_is_not_taken_and_differentiates_the_same():
    G = _gram(6, 0)
    G.requires_grad_(True)
    C, d = _jacobi_cholesky(G, 1e-9)
    (C.sum() + d.sum()).backward()
    G2 = G.detach().clone().requires_grad_(True)
    C0, d0 = _former_jacobi_cholesky(G2, 1e-9)
    (C0.sum() + d0.sum()).backward()
    assert torch.equal(C, C0) and torch.equal(d, d0) and torch.equal(G.grad, G2.grad)


def test_jacobi_cholesky_fallback_still_works_and_gives_the_former_result():
    """One sample is numerically indefinite (smallest Jacobi-scaled eigenvalue -5e-7: fails with ridge 1e-9, passes with the
    1e3 times bigger one): the whole batch is refactorised with the bigger ridge, as before."""
    G = _gram(5, 1)
    X = torch.randn(5, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    X = X / X.norm(dim=1, keepdim=True)                        # 5 unit vectors in R^4: a rank-4 correlation matrix
    u = torch.linalg.svd(X)[0][:, -1]                          # its null vector
    G[1] = X @ X.T - 5e-7 * torch.outer(u, u)                  # smallest eigenvalue -5e-7 (unit diagonal up to 5e-7)
    with pytest.raises(Exception):                            # the plain factorisation of this sample fails
        torch.linalg.cholesky(_scaled(G, 1e-9)[1])
    C, d = _jacobi_cholesky(G, 1e-9)
    C0, d0 = _former_jacobi_cholesky(G, 1e-9)
    assert torch.equal(C, C0) and torch.equal(d, d0) and torch.isfinite(C).all()
    G.requires_grad_(True)                                    # differentiable through the fallback
    _jacobi_cholesky(G, 1e-9)[0].sum().backward()
    assert torch.isfinite(G.grad).all()


def _scaled(G, ridge):
    diag = torch.diagonal(G, dim1=-2, dim2=-1)
    d = diag.clamp(min=1e-300).rsqrt()
    Gs = G * d.unsqueeze(-1) * d.unsqueeze(-2)
    return 0.5 * (Gs + Gs.transpose(-1, -2)) + ridge * torch.eye(G.shape[-1], dtype=G.dtype)


def test_jacobi_cholesky_returns_nan_when_the_bigger_ridge_fails_too():
    G = _gram(4, 2)
    G[2] = 2.0 * torch.ones(4, 4, dtype=torch.float64) - torch.eye(4, dtype=torch.float64)   # unit diagonal, eigenvalue -1: no ridge helps
    C, _ = _jacobi_cholesky(G, 1e-9)
    assert torch.isnan(C).all()                               # whole batch (the fallback is batch-wide); was a LinAlgError


class _HostSyncCounter:
    """Counts what forces a device→host copy or a pageable host→device copy on CUDA: Tensor.item / tolist / numpy / __bool__ /
    __float__ / __int__ / __index__ and torch.tensor(..., device=...)."""

    def __init__(self, monkeypatch):
        self.calls = []
        for name in ("item", "tolist", "numpy", "__bool__", "__float__", "__int__", "__index__"):
            orig = getattr(torch.Tensor, name)
            monkeypatch.setattr(torch.Tensor, name, self._wrap(name, orig))
        orig_tensor = torch.tensor

        def tensor(*a, **k):
            if k.get("device") is not None:
                self.calls.append("torch.tensor(device=)")
            return orig_tensor(*a, **k)
        monkeypatch.setattr(torch, "tensor", tensor)

    def _wrap(self, name, orig):
        def f(t, *a, **k):
            self.calls.append(name)
            return orig(t, *a, **k)
        return f


def test_the_counter_sees_the_former_implementation(monkeypatch):
    G = _gram(5, 4)
    c = _HostSyncCounter(monkeypatch)
    _former_jacobi_cholesky(G, 1e-9)
    assert "__bool__" in c.calls
    c.calls.clear()
    _jacobi_cholesky(G, 1e-9)
    assert c.calls == []


def test_noproj_training_step_has_no_host_sync(pill, box, monkeypatch):
    """After the first step (which builds the cached device constants), a projection-free training step, its backward and the
    validation step call nothing that synchronises a CUDA stream, in both diagnostic modes."""
    batch = field_collate(_items(pill, box))
    for every, idx in ((1, 0), (3, 0), (3, 1)):
        lm = _noproj(diag_metrics_every_n_steps=every)
        lm.log = lambda *a, **k: None                          # Lightning's own logging is not under test (and float() syncs)
        lm.training_step(batch, idx).backward()                # warm-up: tables, lr tensor
        c = _HostSyncCounter(monkeypatch)
        lm.training_step(batch, idx).backward()
        lm.eval()
        with torch.no_grad():
            lm.validation_step(batch, idx)
        monkeypatch.undo()
        assert c.calls == [], (every, idx, c.calls)


def test_fidelity_table_and_lr_tensor_are_built_once(pill, box, monkeypatch):
    lm = _noproj()
    t = lm._table((1.0, 2.0), torch.device("cpu"))
    assert t is lm._table((1.0, 2.0), torch.device("cpu")) and torch.equal(t, torch.tensor([1.0, 2.0]))
    a = lm._lr_tensor(1e-3)
    assert a is lm._lr_tensor(1e-3) and lm._lr_tensor(2e-3) is not a and float(a) == pytest.approx(1e-3)
    lm.fidelity_weights = (1.0, 5.0)                           # a changed setting is not served from the cache
    assert torch.equal(lm._table(lm.fidelity_weights, torch.device("cpu")), torch.tensor([1.0, 5.0]))


# ── diagnostic-metric interval ──────────────────────────────────────────────────────────────────────

DIAG_KEYS = {"train/span_rel_l2", "train/span_rel_l2_energy", "train/mass_diag_ratio", "train/grad_frac_lb",
             "train/grad_frac_lb_max", *(f"train/span_mode_{k}_rel_l2" for k in range(K_MODES))}
LOSS_KEYS = {"train/loss", "train/span_loss", "train/ortho_loss", "lr"}


def test_diag_interval_logs_the_same_keys_on_the_same_steps_and_changes_nothing_else(pill, box):
    batch = field_collate(_items(pill, box))
    ref = _noproj()
    ref_logs = _capture(ref)
    ref_loss = ref.training_step(batch, 0)
    ref_loss.backward()
    ref_grads = _grads(ref)
    assert set(ref_logs) == LOSS_KEYS | DIAG_KEYS                # default: everything every step
    for idx in range(7):
        lm = _noproj(diag_metrics_every_n_steps=3)
        logs = _capture(lm)
        loss = lm.training_step(batch, idx)
        loss.backward()
        assert torch.equal(loss, ref_loss) and _same(_grads(lm), ref_grads)        # the loss and its gradient: every step
        assert set(logs) == (LOSS_KEYS | DIAG_KEYS if idx % 3 == 0 else LOSS_KEYS)
        for k in logs:                                                             # a logged value is the value of every=1
            assert logs[k] == ref_logs[k], k
    # validation is not thinned
    lm = _noproj(diag_metrics_every_n_steps=3)
    logs = _capture(lm)
    lm.eval()
    with torch.no_grad():
        lm.validation_step(batch, 1)
    assert {k.replace("val/", "train/") for k in logs} >= (DIAG_KEYS | {"train/loss", "train/span_loss", "train/ortho_loss"})


def test_diag_interval_guards():
    with pytest.raises(ValueError, match=">= 1"):
        _noproj(diag_metrics_every_n_steps=0)
    with pytest.raises(ValueError, match="projection-free"):
        _module(diag_metrics_every_n_steps=2)                  # projected default path
    assert _module().hparams["diag_metrics_every_n_steps"] == 1 and _module().diag_every == 1
    # an old checkpoint's hyper_parameters (no such key) construct the module with the default
    hp = dict(_noproj().hparams)
    hp.pop("diag_metrics_every_n_steps")
    assert CavityLightning(**hp).diag_every == 1


# ── sparse parts: the batch is the same ─────────────────────────────────────────────────────────────

def _assert_batches_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, torch.Tensor) and x.layout == torch.sparse_csr:
            assert y.layout == torch.sparse_csr and x.shape == y.shape, k
            assert torch.equal(x.crow_indices(), y.crow_indices()) and torch.equal(x.col_indices(), y.col_indices()), k
            assert torch.equal(x.values(), y.values()) and x.dtype == y.dtype, k
        elif isinstance(x, torch.Tensor):
            assert x.dtype == y.dtype and torch.allclose(x, y, rtol=0, atol=0, equal_nan=True), k   # FreqNext is NaN here
        else:
            assert x == y, k


def test_sparse_parts_roundtrip_is_exact_and_the_step_is_unchanged(pill, box):
    for full in (False, True):
        batch = field_collate(_items(pill, box, full=full))
        parts = gpu_transfer.split_sparse(batch)
        assert not any(isinstance(v, torch.Tensor) and v.layout == torch.sparse_csr for v in parts.values())
        n_sparse = sum(isinstance(v, gpu_transfer.SparseParts) for v in parts.values())
        assert n_sparse == (7 if full else 4)                              # M K C Ct (+ G Gt Kp)
        _assert_batches_equal(batch, gpu_transfer.to_device(parts, "cpu"))
        _assert_batches_equal(batch, gpu_transfer.wait_ready(gpu_transfer.to_device(parts, torch.device("cpu"))))
    batch = field_collate(_items(pill, box))
    moved = gpu_transfer.to_device(gpu_transfer.split_sparse(batch), "cpu")
    out = []
    for b in (batch, moved):
        lm = _noproj()
        _capture(lm)
        loss = lm.training_step(b, 0)
        loss.backward()
        out.append((loss, _grads(lm)))
    assert torch.equal(out[0][0], out[1][0]) and _same(out[0][1], out[1][1])


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def test_parts_collate_through_dataloader_workers_equals_the_plain_collate(pill, box):
    items = _items(pill, box)
    plain = field_collate(items)
    loader = torch.utils.data.DataLoader(_ListDataset(items), batch_size=2, num_workers=2, collate_fn=field_collate_parts)
    (got,) = list(loader)
    assert any(isinstance(v, gpu_transfer.SparseParts) for v in got.values())
    _assert_batches_equal(plain, gpu_transfer.to_device(got, "cpu"))


def test_dataloader_pinning_walks_into_sparse_parts(pill, box, monkeypatch):
    """DataLoader(pin_memory=True) pins a SparseParts NamedTuple field by field (the CUDA pinning itself is replaced by a marker
    here): every dense tensor and every part of every operator is pinned, the integer size is left alone."""
    from torch.utils.data._utils.pin_memory import pin_memory
    pinned = []

    def fake_pin(t, *a, **k):
        pinned.append(t)
        return t
    monkeypatch.setattr(torch.Tensor, "pin_memory", fake_pin)
    batch = gpu_transfer.split_sparse(field_collate(_items(pill, box)))
    out = pin_memory(batch, None)
    n_dense = sum(isinstance(v, torch.Tensor) for v in batch.values())
    n_parts = sum(isinstance(v, gpu_transfer.SparseParts) for v in batch.values())
    assert len(pinned) == n_dense + 3 * n_parts and n_parts == 4
    assert all(isinstance(out[k], gpu_transfer.SparseParts) and out[k].size == batch[k].size
               for k in batch if isinstance(batch[k], gpu_transfer.SparseParts))
    assert out["shape_type"] == batch["shape_type"]


def test_lightning_fit_with_gpu_prefetch_on_cpu_equals_the_plain_path(pill, box):
    """Through Lightning's real call path (loader -> transfer_batch_to_device -> step): the SparseParts batches of
    `gpu_prefetch` give the same losses as plain batches, step for step (CPU: no pinning, no stream)."""
    import pytorch_lightning as pl
    items = _items(pill, box)

    class Rec(pl.Callback):
        def __init__(self):
            self.losses = []

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.losses.append(outputs["loss"].detach().clone())
            assert not any(isinstance(v, gpu_transfer.SparseParts) for v in batch.values())     # rebuilt before the step

    runs = []
    for parts in (False, True):
        lm = _module(projection=False, freq_weight=0.0, selfsup_weight=0.0, fidelity_weights=(1.0, 2.0))
        lm.gpu_prefetch = parts
        rec = Rec()
        loader = torch.utils.data.DataLoader(_ListDataset(items * 2), batch_size=1,
                                             collate_fn=field_collate_parts if parts else field_collate)
        pl.Trainer(max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False,
                   callbacks=[rec], accumulate_grad_batches=2, num_sanity_val_steps=0, enable_model_summary=False,
                   gradient_clip_val=1.0).fit(lm, loader)
        runs.append((rec.losses, {n: p.detach().clone() for n, p in lm.named_parameters()}))
    assert len(runs[0][0]) == 4 and all(torch.equal(a, b) for a, b in zip(runs[0][0], runs[1][0]))
    assert all(torch.equal(runs[0][1][n], runs[1][1][n]) for n in runs[0][1])            # and the trained weights


def test_sync_debug_callback_is_a_no_op_without_cuda():
    from src.training.step_profile import SyncDebug
    cb = SyncDebug(2)
    cb.on_train_batch_start(None, None, None, 3)
    cb.on_train_batch_end(None, None, None, None, 4)
    cb.on_train_end(None, None)


def test_wait_ready_orders_the_streams(monkeypatch):
    """The CUDA branch cannot run here: the order of its calls is checked with stand-ins."""
    log = []

    class Stream:
        def wait_event(self, e):
            log.append(("wait", e))

    class T:
        def record_stream(self, s):
            log.append(("record", s))

    cur = Stream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: cur)
    b = gpu_transfer.DeviceBatch({"x": 1})
    b.ready_event, b.keep = "EV", [T(), T()]
    assert gpu_transfer.wait_ready(b) is b
    assert log == [("wait", "EV"), ("record", cur), ("record", cur)] and b.ready_event is None
    gpu_transfer.wait_ready(b)
    assert len(log) == 3 and gpu_transfer.wait_ready({"x": 1}) == {"x": 1}


# ── operator cache: faster decoder, no zlib ─────────────────────────────────────────────────────────

def _former_expand_upper(ptr, ind, val, n):
    rows = np.repeat(np.arange(n, dtype=ind.dtype), np.diff(ptr))
    strict = ind > rows
    A = sp.coo_matrix((np.concatenate([val, val[strict]]),
                       (np.concatenate([rows, ind[strict]]), np.concatenate([ind, rows[strict]]))), shape=(n, n)).tocsr()
    return A.indptr, A.indices, A.data


def _random_sym(n, density, seed, empty_rows=False):
    """Symmetric canonical CSR with a full diagonal, explicit zeros (mirrored) and, optionally, rows without off-diagonals."""
    rng = np.random.default_rng(seed)
    U = sp.triu(sp.random(n, n, density=density, random_state=rng, format="coo"), k=1)
    r, c = U.row, U.col
    if empty_rows:
        keep = (r % 5 != 0) & (c % 5 != 0)
        r, c = r[keep], c[keep]
    v = rng.normal(size=r.size)
    v[::7] = 0.0                                                            # explicit zeros stay
    d = rng.normal(size=n)
    A = sp.coo_matrix((np.concatenate([v, v, d]), (np.concatenate([r, c, np.arange(n)]),
                                                   np.concatenate([c, r, np.arange(n)]))), shape=(n, n)).tocsr()
    A.sort_indices()
    assert A.has_canonical_format and (A != A.T).nnz == 0
    return A


@pytest.mark.parametrize("n,density,empty", [(1, 1.0, False), (7, 0.5, False), (60, 0.1, False), (200, 0.03, True),
                                             (500, 0.0, False)])
def test_upper_pattern_equals_the_former_coo_route_bit_for_bit(n, density, empty):
    A = _random_sym(n, density, seed=n, empty_rows=empty)
    ptr, ind, val = ops_cache._upper(A)
    ref = _former_expand_upper(ptr, ind, val, n)
    new = ops_cache._expand_upper(ptr, ind, val, n)
    assert all(np.array_equal(a, b) for a, b in zip(ref, new))
    assert np.array_equal(new[0], A.indptr) and np.array_equal(new[1], A.indices) and ops_cache._bits_equal(new[2], A.data)


def test_patterns_are_shared_between_matrices_of_one_pattern_only():
    A = _random_sym(80, 0.1, seed=1)
    B = A.copy()
    B.data = B.data * 3.0 + 1.0                                            # same pattern, other values (K and M)
    C = _random_sym(80, 0.1, seed=2)
    pat = ops_cache._Patterns()
    outs = []
    for X in (A, B, C):
        ptr, ind, val = ops_cache._upper(X)
        outs.append((pat.expand(ptr, ind, 80), ops_cache._expand_upper(ptr, ind, val, 80, pat)))
    assert outs[0][0] is outs[1][0] and outs[2][0] is not outs[0][0]
    for X, (_, (fp, fi, fv)) in zip((A, B, C), outs):
        assert np.array_equal(fp, X.indptr) and np.array_equal(fi, X.indices) and ops_cache._bits_equal(fv, X.data)


def _ops_of(pill, narrow):
    ops = dict(pill[1] if not narrow else pill[2])
    ops = {k: v for k, v in ops.items() if k not in ("bary", "X", "tets")}
    rec = {"u_model": pill[4].astype(np.float32).astype(np.float64), "shape_type": "pillbox"}
    return ops, rec


@pytest.mark.parametrize("narrow", [False, True])
def test_cache_without_zlib_reads_back_the_same_and_is_bigger(pill, narrow, tmp_path):
    with pytest.raises(ValueError, match="compress must be"):
        ops_cache.pack(*_ops_of(pill, narrow), narrow, compress="best")
    ops, rec = _ops_of(pill, narrow)
    z, raw = ops_cache.pack(ops, rec, narrow), ops_cache.pack(ops, rec, narrow, compress=False)
    for payload in (z, raw):
        assert payload["format"] == ops_cache.FORMAT == 2
    codecs = {p["codec"] for q in raw["ops"].values() for p in ([q["p"]] if q["kind"] == "array" else
                                                                  [q[n] for n in ("indptr", "indices", "data")] if q["kind"] == "csr" else [])}
    deltas = [q[n]["delta"] for q in raw["ops"].values() if q["kind"] == "csr" for n in ("indptr", "indices")]
    assert codecs == {"raw"} and not any(deltas)
    assert any(q["indices"]["delta"] for q in z["ops"].values() if q["kind"] == "csr")
    (oz, rz), (or_, rr) = ops_cache.unpack(z), ops_cache.unpack(raw)
    assert oz.keys() == or_.keys() and np.array_equal(rz["u_model"], rr["u_model"])
    for k in oz:
        if sp.issparse(oz[k]):
            a, b = oz[k], or_[k]
            assert a.dtype == b.dtype and a.indptr.dtype == b.indptr.dtype and a.indices.dtype == b.indices.dtype
            assert np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
            assert ops_cache._bits_equal(a.data, b.data), k
        elif isinstance(oz[k], np.ndarray):
            assert oz[k].dtype == or_[k].dtype and np.array_equal(oz[k], or_[k]), k
    # and the one the former code would have written (compress=True is its byte-for-byte writer): the file on disk
    zp, rp = tmp_path / "z.ops2.pkl", tmp_path / "r.ops2.pkl"
    ops_cache.write(str(zp), z)
    ops_cache.write(str(rp), raw)
    assert os.path.getsize(rp) > os.path.getsize(zp)
    for path in (zp, rp):
        ops_cache.unpack(ops_cache.read(str(path)))
    # uncompress() of a compressed payload = the raw payload's content
    (ou, ru) = ops_cache.unpack(ops_cache.uncompress(z))
    assert all(np.array_equal(ou[k].data, or_[k].data) for k in ou if sp.issparse(ou[k]))
    assert not any(p["codec"] == "zlib" for q in ops_cache.uncompress(z)["ops"].values() if q["kind"] == "csr"
                   for p in (q["indptr"], q["indices"], q["data"]))


def test_recode_script_converts_between_policies_without_changing_the_content(pill, tmp_path):
    import sys
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import recode_ops_cache as rc
    ops, rec = _ops_of(pill, narrow=True)
    src = tmp_path / "a" / "x.ops2.pkl"
    ops_cache.write(str(src), ops_cache.pack(ops, rec, True))
    sizes = {}
    for name, pol in (("raw", False), ("fast", "fast"), ("z", True)):
        dst = tmp_path / name / "x.ops2.pkl"
        _, sz, err = rc.recode((str(src), str(dst), pol, True))
        assert err is None
        sizes[name] = sz[1]
        assert rc._same(ops_cache.read(str(src)), ops_cache.read(str(dst)))
    assert sizes["raw"] > sizes["fast"] >= sizes["z"] * 0.99
    bad = tmp_path / "bad.ops2.pkl"
    bad.write_bytes(b"not a pickle")
    assert rc.recode((str(bad), str(tmp_path / "o.pkl"), False, True))[2] is not None


# ── loaders, config, profiler ───────────────────────────────────────────────────────────────────────

def test_make_loaders_options(pill, box):
    import train
    ds = _ListDataset(_items(pill, box))
    tc = types.SimpleNamespace(num_workers=2, batch_size=1)
    tr, va, te = train.make_loaders(tc, ds, ds, ds)
    assert (tr.prefetch_factor, va.prefetch_factor, tr.persistent_workers, va.persistent_workers) == (2, 2, True, True)
    assert not tr.pin_memory and tr.collate_fn is field_collate and tr.num_workers == va.num_workers == 2
    tc = types.SimpleNamespace(num_workers=2, batch_size=1, prefetch_factor=3, val_persistent_workers=False,
                               val_num_workers=1, gpu_prefetch=True)
    tr, va, te = train.make_loaders(tc, ds, ds, ds)
    assert tr.prefetch_factor == te.prefetch_factor == va.prefetch_factor == 3
    assert tr.persistent_workers and not va.persistent_workers and va.num_workers == 1 and tr.num_workers == 2
    assert tr.collate_fn is field_collate and not tr.pin_memory                      # no CUDA here: plain batches
    kw = train.loader_kwargs(tc, 2, True, True)
    assert kw["collate_fn"] is field_collate_parts and kw["pin_memory"] and kw["prefetch_factor"] == 3
    assert train.loader_kwargs(tc, 0, True, False)["prefetch_factor"] is None
    assert not train.loader_kwargs(tc, 0, True, False)["persistent_workers"]


def test_new_config_keys_are_in_both_configs_with_the_old_behaviour_as_default():
    from src.config import load_config
    new = {"log_every_n_steps": None, "diag_metrics_every_n_steps": 1, "prefetch_factor": 2, "val_persistent_workers": True,
           "val_num_workers": None, "gpu_prefetch": False, "profile_steps": 0, "profile_only": False, "sync_debug_steps": 0}
    new.pop("log_every_n_steps")
    cfgs = [load_config(os.path.join(REPO, "configs", f)).training for f in ("field_3d.yaml", "field_3d_noproj.yaml")]
    for tc in cfgs:
        for k, v in new.items():
            assert k in tc and tc[k] == v, k
    for f in ("field_3d.yaml", "field_3d_noproj.yaml"):
        assert load_config(os.path.join(REPO, "configs", f)).dataset["cache_compress"] is True


def test_step_profiler_writes_the_breakdown_and_removes_the_clock(pill, box, tmp_path):
    import pytorch_lightning as pl

    from src.training.step_profile import STAGES, StepProfiler
    items = _items(pill, box)
    lm = _module(projection=False, freq_weight=0.0, selfsup_weight=0.0, fidelity_weights=(1.0, 2.0))
    path = tmp_path / "prof.json"
    prof = StepProfiler(3, warmup=1, path=str(path), stop=True)
    loader = torch.utils.data.DataLoader(_ListDataset(items * 3), batch_size=1, collate_fn=field_collate)
    tr = pl.Trainer(max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False,
                    callbacks=[prof], accumulate_grad_batches=2, limit_train_batches=5, num_sanity_val_steps=0,
                    enable_model_summary=False)
    tr.fit(lm, loader)
    info = json.load(open(path))
    assert info["n_batches"] == 2 and info["warmup_skipped"] == 1 and lm._clock is None
    assert tr.global_step == 1 and tr.should_stop           # stop=True: 3 batches seen of 5, accumulate 2 -> 1 optimizer step
    st = info["stages"]
    assert set(st) == {*STAGES, "total"} and all(v["mean"] >= 0 for v in st.values())
    assert st["trunk"]["mean"] > 0 and st["grams"]["mean"] > 0 and st["backward"]["mean"] > 0
    assert sum(st[k]["mean"] for k in STAGES) == pytest.approx(st["total"]["mean"], rel=0.25)
