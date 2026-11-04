"""Operator cache format (src/data/ops_cache.py, docs/38 §9): bit-exact full variant, float32 projection-free
variant, upper-triangle K / M, int32 guard, versioning, atomic writes.  Pillbox on a curved p2 mesh (NGSolve)."""
import os
import pickle
import threading

import numpy as np
import pytest
import scipy.sparse as sp
import torch  # noqa: F401  (before ngsolve)

pytest.importorskip("ngsolve")
from netgen.occ import Axes, Cylinder, OCCGeometry, Z  # noqa: E402
from ngsolve import Mesh  # noqa: E402

from src.data import ops_cache as oc  # noqa: E402
from src.data.mesh_features import normalise  # noqa: E402
from src.data_gen import highorder as ho  # noqa: E402


@pytest.fixture(scope="module")
def pill():
    geo = Cylinder(Axes((0, 0, 0), Z), r=0.1, h=0.1)
    geo.faces.name = "wall"
    mesh = Mesh(OCCGeometry(geo).GenerateMesh(maxh=0.05))
    mesh.Curve(2)
    Xv = np.array([tuple(p.p) for p in mesh.ngmesh.Points()])
    tets = np.array([[v.nr - 1 for v in e.vertices] for e in mesh.ngmesh.Elements3D()])
    _, center, scale = normalise(Xv, tets)
    ops = ho.field_operators(mesh, order=2, scale=scale, center=center)
    ops = {k: v for k, v in ops.items() if k not in ("bary", "X", "tets")}          # as FieldDataset caches them
    rng = np.random.RandomState(0)
    rec = {"freqs": np.array([1.0, 2.0]), "u_model": rng.randn(ops["n_dof"], 2).astype(np.float32).astype(np.float64),
           "rq_model": np.array([3.0, 4.0]), "qoi_names": ["a"], "shape_type": "pillbox", "model_order": 2}
    return ops, rec


def _roundtrip(ops, rec, narrow, tmp_path):
    path = str(tmp_path / "x.ops2.pkl")
    oc.write(path, oc.pack(ops, rec, narrow))
    return oc.unpack(oc.read(path))


def _same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert np.array_equal(a.view(np.uint8), b.view(np.uint8)) if a.dtype.kind == "f" else np.array_equal(a, b)


def _same_csr(A, B):
    assert A.shape == B.shape and A.nnz == B.nnz and A.has_sorted_indices and B.has_sorted_indices
    assert A.indptr.dtype == B.indptr.dtype == np.int32 and A.indices.dtype == B.indices.dtype == np.int32
    assert np.array_equal(A.indptr, B.indptr) and np.array_equal(A.indices, B.indices)
    _same(A.data, B.data)


def test_full_variant_round_trips_bit_for_bit(pill, tmp_path):
    ops, rec = pill
    pl = oc.pack(ops, rec, narrow=False)
    assert pl["ops"]["K"]["sym"] and pl["ops"]["M"]["sym"]            # the symmetric ones are stored as a triangle
    assert not pl["ops"]["Kp"]["sym"] and not pl["ops"]["C"]["sym"]   # Kp = GᵀMG is not bit-symmetric, C is rectangular
    ops2, rec2 = _roundtrip(ops, rec, False, tmp_path)
    assert set(ops2) == set(ops)
    for k, v in ops.items():
        if sp.issparse(v):
            _same_csr(v, ops2[k])
            assert ops2[k].data.dtype == np.float64
        elif isinstance(v, np.ndarray):
            _same(v, ops2[k])
        else:
            assert v == ops2[k] or np.array_equal(v, ops2[k])
    _same(rec["u_model"], rec2["u_model"])
    assert rec2["u_model"].dtype == np.float64 and rec2["freqs"].tolist() == [1.0, 2.0] and rec2["shape_type"] == "pillbox"


def test_projection_free_variant_is_float32_valued_and_still_exactly_symmetric(pill, tmp_path):
    ops, rec = pill
    nopot = {k: v for k, v in ops.items() if k not in ("G", "Kp")} | {"n_pot": None}
    ops2, rec2 = _roundtrip(nopot, rec, True, tmp_path)
    for k in ("K", "M", "C"):
        A, B = nopot[k], ops2[k]
        assert B.data.dtype == np.float64 and np.array_equal(A.indptr, B.indptr) and np.array_equal(A.indices, B.indices)
        assert np.array_equal(B.data, A.data.astype(np.float32).astype(np.float64))      # exactly the float32 rounding
        # half an ulp of float32 (6e-8 relative), plus the float32 subnormal spacing for assembly noise ~1e-40
        assert np.all(np.abs(B.data - A.data) <= 6e-8 * np.abs(A.data) + 1e-44)
    for k in ("K", "M"):
        assert (ops2[k] != ops2[k].T).nnz == 0                                           # SymmetricMK stays valid
    for k in ("P", "qx", "qw"):
        assert ops2[k].dtype == np.float64
        assert np.array_equal(ops2[k].astype(np.float32), nopot[k].astype(np.float32))   # what field_item uses
    for k in ("qfeat", "bnd", "center"):
        _same(nopot[k], ops2[k])
    assert "G" not in ops2 and ops2["n_pot"] is None
    _same(rec["u_model"], rec2["u_model"])                                               # float32-valued: exact
    # the format is much smaller than the plain pickle
    plain = len(pickle.dumps((nopot, rec), protocol=pickle.HIGHEST_PROTOCOL))
    assert len(pickle.dumps(oc.pack(nopot, rec, True), protocol=pickle.HIGHEST_PROTOCOL)) < plain / 2.5


def test_upper_triangle_rebuild_is_exact_and_only_used_when_it_is(tmp_path):
    rng = np.random.RandomState(1)
    n = 60
    ii, jj = np.nonzero(np.triu(rng.rand(n, n) < 0.08, 1))
    vv = rng.randn(ii.size)
    vv[::5] = 0.0                                                        # explicit zeros keep their place
    dg = np.arange(0, n, 2)                                              # diagonal: every other row, half of them zero
    dv = np.where(dg % 4 == 0, 0.0, rng.randn(dg.size))
    S = sp.coo_matrix((np.r_[vv, vv, dv], (np.r_[ii, jj, dg], np.r_[jj, ii, dg])), shape=(n, n)).tocsr()
    S.sort_indices()
    assert S.nnz == 2 * ii.size + dg.size and (S.data == 0).sum() > 10
    p = oc._pack_csr(S, narrow=False, allow_sym=True)
    assert p["sym"] and p["indices"]["shape"] == ((S.nnz + dg.size) // 2,)
    R = oc._unpack_csr(p)
    assert np.array_equal(R.indptr, S.indptr) and np.array_equal(R.indices, S.indices)
    _same(R.data, S.data)
    # not exactly symmetric: one entry off by one ulp, a -0.0 against a +0.0, a missing mirror, a non-square matrix
    off = S.copy()
    i, j = np.argwhere(np.triu(S.toarray() != 0, 1))[0]
    pos = off.indptr[j] + np.searchsorted(off.indices[off.indptr[j]:off.indptr[j + 1]], i)
    off.data[pos] = np.nextafter(off.data[pos], np.inf)
    zer = S.copy()
    rows = np.repeat(np.arange(n), np.diff(zer.indptr))
    z = np.flatnonzero((zer.data == 0) & (zer.indices > rows))             # an off-diagonal zero: its mirror stays +0.0
    assert z.size
    zer.data[z[0]] = -0.0
    lone = sp.triu(S, 1, format="csr") + sp.eye(n, format="csr")
    for A in (off, zer, lone.tocsr(), sp.random(n, n + 3, density=0.1, random_state=rng, format="csr")):
        A.sort_indices()
        p = oc._pack_csr(A, narrow=False, allow_sym=True)
        R = oc._unpack_csr(p)
        assert not p["sym"] or (R.indices == A.indices).all()
        assert np.array_equal(R.indptr, A.indptr) and np.array_equal(R.indices, A.indices)
        _same(R.data, A.data)
    assert not oc._pack_csr(off, False, True)["sym"] and not oc._pack_csr(lone.tocsr(), False, True)["sym"]
    assert not oc._pack_csr(zer, False, True)["sym"]
    # a matrix with unsorted columns is stored as it is
    u = S.copy()
    u.indices[:2] = u.indices[:2][::-1]
    u.data[:2] = u.data[:2][::-1]
    assert not u.has_sorted_indices and not oc._pack_csr(u, False, True)["sym"]


def test_integer_ranges_are_asserted():
    i32 = np.dtype(np.int32)
    assert oc.index_dtypes(10 ** 6, 10 ** 6, 5 * 10 ** 8) == (i32, i32)
    assert oc.index_dtypes(10 ** 6, 10 ** 6, 2 ** 31 - 1)[0] == i32
    assert oc.index_dtypes(10 ** 6, 10 ** 6, 2 ** 31)[0] == np.dtype(np.int64)              # nnz needs int64 indptr
    assert oc.index_dtypes(10 ** 6, 2 ** 31, 10)[1] == np.dtype(np.int64)                   # columns need int64 indices
    with pytest.raises(OverflowError):
        oc.index_dtypes(10, 10, 2 ** 63)
    with pytest.raises(OverflowError):
        oc._enc(np.array([0, 2 ** 31 + 5], np.int64), np.int32, delta=True)                 # a difference outside int32
    assert oc._enc(np.array([0, 2 ** 31 + 5], np.int64), np.int64, delta=True)["dtype"] == "<i8"
    a = np.arange(10_000, dtype=np.int64) * 3
    assert np.array_equal(oc._dec(oc._enc(a, np.int32, delta=True)), a) and oc._enc(a, np.int32, delta=True)["codec"] == "zlib"
    r = np.random.RandomState(0).rand(10_000)
    assert oc._enc(r)["codec"] == "raw"                                                      # incompressible: no zlib
    with pytest.raises(OverflowError, match="narrow"):
        oc._enc(np.array([1.0, 1e300]), np.float32)


def test_payload_of_another_format_is_refused(pill, tmp_path):
    ops, rec = pill
    pl = oc.pack(ops, rec, False)
    assert pl["format"] == oc.FORMAT
    bad = tmp_path / "f1.ops2.pkl"
    oc.write(str(bad), pl | {"format": oc.FORMAT - 1})
    with pytest.raises(oc.OpsCacheError, match="another format"):
        oc.read(str(bad))
    legacy = tmp_path / "legacy.ops.pkl"                                      # the old (ops, rec) tuple
    legacy.write_bytes(pickle.dumps((ops, rec)))
    with pytest.raises(oc.OpsCacheError, match="another format"):
        oc.read(str(legacy))
    trunc = tmp_path / "t.ops2.pkl"
    oc.write(str(trunc), pl)
    trunc.write_bytes(trunc.read_bytes()[:1000])
    with pytest.raises(oc.OpsCacheError, match="unreadable"):
        oc.read(str(trunc))
    assert oc.SUFFIX != ".ops.pkl" and oc.SUFFIX.endswith(".pkl")


def test_write_is_atomic_and_race_safe(pill, tmp_path, monkeypatch):
    ops, rec = pill
    pl = oc.pack(ops, rec, False)
    path = str(tmp_path / "sub" / "a.ops2.pkl")
    oc.write(path, pl)                                                         # creates the directory
    first = open(path, "rb").read()
    # a writer that dies half-way leaves neither a partial file under the final name nor a temporary one
    real_dump = pickle.dump

    def dying(obj, fh, protocol=None):
        fh.write(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(oc.pickle, "dump", dying)
    with pytest.raises(OSError):
        oc.write(path, pl)
    monkeypatch.setattr(oc.pickle, "dump", real_dump)
    assert open(path, "rb").read() == first and os.listdir(tmp_path / "sub") == ["a.ops2.pkl"]
    # concurrent writers of the same file: always a complete readable file, no temporaries left
    errs = []

    def work():
        try:
            for _ in range(3):
                oc.write(path, pl)
                oc.unpack(oc.read(path))
        except Exception as e:      # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and os.listdir(tmp_path / "sub") == ["a.ops2.pkl"]
