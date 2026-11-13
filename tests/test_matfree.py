"""Matrix-free HCurl(p2) operators (src/data/matfree.py, docs/40) against the assembled sparse operators of
highorder.field_operators(mesh, 2, potential=False) on small curved meshes: pillbox (geometry order 3), the same
deformed (the generator's smooth map), a bisected + single_level mesh (all 24 orientation classes), a bisected
mesh without single_level (unused DOFs), and a synthetic mesh of disjoint random cubic tets (all 24 classes)."""
import os
import tempfile

import numpy as np
import pytest
import torch

pytest.importorskip("ngsolve")
from ngsolve import VOL, ElementId, Mesh  # noqa: E402
from netgen.occ import Axes, Cylinder, OCCGeometry, Z  # noqa: E402

from src.data import matfree as mf  # noqa: E402
from src.data.field_dataset import field_collate, field_item  # noqa: E402
from src.data_gen import highorder as ho  # noqa: E402
from src.data_gen.dataset_generator_3d import deform_map  # noqa: E402
from src.data_gen.field_labels import model_normalisation  # noqa: E402
from src.models.hcurl import hcurl_grams_plain, spmm  # noqa: E402

TOL_OP = 1e-11            # operator products and Grams (relative to the max entry)
M_BASIS = 6


@pytest.fixture(autouse=True, scope="module")
def _cache(tmp_path_factory):
    """Tables of this module in a private cache dir (also exercises the first-use build)."""
    old = os.environ.get("RFCAV_CACHE")
    os.environ["RFCAV_CACHE"] = str(tmp_path_factory.mktemp("rfcav_cache"))
    mf._MEMO.clear()
    yield
    if old is None:
        del os.environ["RFCAV_CACHE"]
    else:
        os.environ["RFCAV_CACHE"] = old
    mf._MEMO.clear()


def _cylinder():
    geo = Cylinder(Axes((0, 0, 0), Z), r=0.1, h=0.1)
    geo.faces.name = "wall"
    return geo


def _meshes():
    """name → (mesh, use model_normalisation)."""
    out = {}
    geo = _cylinder()
    mesh = Mesh(OCCGeometry(geo).GenerateMesh(maxh=0.045))
    mesh.Curve(3)
    out["pillbox"] = (mesh, False)
    om, a, ph, _ = deform_map(np.array([[-0.1, -0.1, 0.0], [0.1, 0.1, 0.1]]), np.random.default_rng(3), 0.4)
    with tempfile.TemporaryDirectory() as tmp:
        brep = os.path.join(tmp, "c.brep")
        geo.WriteBrep(brep)
        deformed, shape = ho.netgen_mesh(brep, 0.045, curve=3, deform=(om, a, ph), return_shape=True)
        out["deformed"] = (deformed, True)
        ref, shape2 = ho.netgen_mesh(brep, 0.07, curve=3, deform=(om, a, ph), return_shape=True)
        for e in range(ref.ne):
            ref.SetRefinementFlag(ElementId(VOL, e), e % 4 == 0)
        ref.Refine()
        ref.Curve(3)
        ho.set_deformation(ref, om, a, ph, order=3)
        out["bisected_hierarchy"] = (ref, True)          # unused DOFs of the parents' edges / faces
        for e in range(ref.ne):
            ref.SetRefinementFlag(ElementId(VOL, e), False)
        out["bisected_single_level"] = (ho.single_level(ref, shape2, 3, (om, a, ph)), True)
    out["synthetic24"] = (mf._synthetic_mesh(2, 5, 0.05), False)
    return out


@pytest.fixture(scope="module")
def cases():
    built = {}
    for name, (mesh, norm) in _meshes().items():
        kw = dict(zip(("center", "scale"), model_normalisation(mesh), strict=True)) if norm else {}
        rec = mf.build_record(mesh, **kw)
        ops = ho.field_operators(mesh, 2, potential=False, **kw)
        built[name] = (mesh, rec, ops)
    return built


NAMES = ["pillbox", "deformed", "bisected_hierarchy", "bisected_single_level", "synthetic24"]


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.fixture(params=NAMES)
def case(request, cases):
    return request.param, *cases[request.param]


def _random(rec, seed=0, m=M_BASIS):
    rng = np.random.default_rng(seed)
    n, nt = rec["n_dof"], rec["n_tets"]
    return rng.standard_normal((n, m)), rng.standard_normal((n, 3)), rng.standard_normal((nt, 14, m, 3))


def test_record_layout_and_size(cases):
    for name, (mesh, rec, ops) in cases.items():
        nt = mesh.ne
        assert rec["dofs"].dtype == np.int32 and rec["dofs"].shape == (nt, 30)
        assert rec["cls"].dtype == np.uint8 and rec["cls"].max() < 24
        assert rec["xc"].dtype == np.float64 and rec["xc"].shape == (nt, 20, 3)
        assert rec["x0"].dtype == np.float64 and rec["bnd"].dtype == bool
        assert rec["n_dof"] == ops["n_dof"] and rec["n_tets"] == nt
        assert mf.record_nbytes(rec) / nt <= 700, name
        np.testing.assert_array_equal(rec["bnd"], ops["bnd"])
        assert rec["scale"] == ops["scale"] and np.array_equal(rec["center"], ops["center"])
        assert not any(hasattr(v, "format") for v in rec.values())          # no scipy matrix in the record
    assert len(np.unique(cases["synthetic24"][1]["cls"])) == 24
    assert len(np.unique(cases["bisected_single_level"][1]["cls"])) == 24
    d = cases["bisected_hierarchy"][1]
    assert len(np.unique(d["dofs"])) < d["n_dof"]                       # unused DOFs exist and are tolerated
    assert (mf.to_torch(d)["n_d"] == 0).any()


def test_element_operators_match_the_assembled_matrices(case):
    name, _, rec, ops = case
    tr = mf.to_torch(rec)
    Me, Ke = mf.element_matrices(tr)
    V, _, _ = _random(rec)
    Vt = torch.from_numpy(V)
    assert _rel(mf.apply(Ke, tr["dofs"], Vt).numpy(), ops["K"] @ V) < TOL_OP
    assert _rel(mf.apply(Me, tr["dofs"], Vt).numpy(), ops["M"] @ V) < TOL_OP
    assert _rel(mf.gram(Ke, tr["dofs"], Vt).numpy(), V.T @ (ops["K"] @ V)) < TOL_OP
    assert _rel(mf.gram(Me, tr["dofs"], Vt).numpy(), V.T @ (ops["M"] @ V)) < TOL_OP
    G = mf.gram(Me, tr["dofs"], Vt[:, :4], Vt[:, 4:]).numpy()                        # [V | T] block Gram
    assert _rel(G, V.T @ (ops["M"] @ V)) < TOL_OP
    assert np.array_equal(G, G.T)
    # the element matrices are symmetric and the global ones are exactly the scatter of them
    assert float((Me - Me.transpose(1, 2)).abs().max()) == 0.0


def test_psi_to_dofs_matches_C_P_psi(case):
    name, _, rec, ops = case
    _, _, psi = _random(rec)
    m = psi.shape[2]
    ref = ops["C"] @ np.einsum("tlq,tqmc->tclm", ops["P"], psi).reshape(-1, m)
    u = mf.psi_to_dofs(mf.to_torch(rec), torch.from_numpy(psi)).numpy()
    assert _rel(u, ref) < 1e-10
    assert np.abs(u[rec["bnd"]]).max() > 0                                          # wall rows are NOT masked here
    # the same through the precomputed element map, and the P of the token rule
    Ce = mf.psi_map(rec)
    assert _rel(mf.psi_to_dofs(rec, torch.from_numpy(psi), Ce).numpy(), ref) < 1e-10
    tr = mf.to_torch(rec)
    P = mf._token_P(tr["cls"], mf.jacobians(tr, "token")).numpy()
    assert _rel(P, ops["P"]) < 1e-11


def test_token_points_and_weights(case):
    name, _, rec, ops = case
    assert _rel(mf.token_points(rec).numpy(), ops["qx"]) < 1e-12
    assert _rel(mf.token_weights(rec).numpy(), ops["qw"]) < 1e-12


def test_jacobians_are_those_of_ngsolve(cases):
    mesh, rec, _ = cases["deformed"]
    from ngsolve import IntegrationRule, TET
    for rule, deg in (("mass", 8), ("token", 4)):
        Jn, _ = mf._jacobians_ngsolve(mesh, IntegrationRule(TET, deg))
        assert _rel(mf.jacobians(rec, rule).numpy(), Jn) < 1e-12


def _batch_of(ops):
    """Collated single-item batch (M, K, SymmetricMK) of the sparse path; no labels (a synthetic mesh has only wall DOFs)."""
    return field_collate([field_item(ops)])


def test_grams_plain_equals_hcurl_grams_plain(case):
    name, _, rec, ops = case
    V, T, _ = _random(rec)
    Vt, Tt = torch.from_numpy(V), torch.from_numpy(T)
    assert float(Tt.abs().max()) > 0
    batch = _batch_of(ops)
    T32 = Tt.float().double()                                       # float32-representable targets, like stored labels
    ref_M, ref_A, ref_MT = hcurl_grams_plain(Vt[None], T32[None], batch)
    GM, GA, MT = mf.grams_plain(rec, Vt, T32)
    assert _rel(GM.numpy(), ref_M[0].numpy()) < TOL_OP
    assert _rel(GA.numpy(), ref_A[0].numpy()) < TOL_OP
    assert _rel(MT.numpy(), ref_MT[0].numpy()) < TOL_OP
    assert GM.shape == (V.shape[1] + 3, V.shape[1] + 3)
    # float32 inputs are accepted (cast like hcurl_grams_plain does)
    GM32, _, _ = mf.grams_plain(rec, Vt.float(), T32.float())
    assert _rel(GM32.numpy(), hcurl_grams_plain(Vt.float()[None], T32.float()[None], batch)[0][0].numpy()) < 1e-6
    # a padded batch of two records: rows ≥ n_dof ignored, result stacked
    pad = 5
    Vp = torch.nn.functional.pad(Vt, (0, 0, 0, pad))[None].repeat(2, 1, 1)
    Tp = torch.nn.functional.pad(T32, (0, 0, 0, pad))[None].repeat(2, 1, 1)
    GMb, GAb, MTb = mf.grams_plain([rec, rec], Vp, Tp)
    assert GMb.shape[0] == 2 and MTb.shape == (2, V.shape[0] + pad, 3)
    assert float((GMb[1] - GM).abs().max()) == 0 and float(MTb[0, -pad:].abs().max()) == 0


def test_gradients_equal_the_sparse_path(case):
    name, _, rec, ops = case
    V, T, psi = _random(rec, seed=1)
    m = V.shape[1]
    batch = _batch_of(ops)
    T32 = torch.from_numpy(T).float().double()
    rng = np.random.default_rng(9)
    wM, wA = (torch.from_numpy(rng.standard_normal((m + 3, m + 3))) for _ in range(2))

    def loss_of(GM, GA):
        return (GM * wM).sum() + 1e-3 * (GA * wA).sum()
    # w.r.t. V
    Vs = torch.from_numpy(V).requires_grad_(True)
    GM, GA, _ = hcurl_grams_plain(Vs[None], T32[None], batch)
    loss_of(GM[0], GA[0]).backward()
    Vm = torch.from_numpy(V).requires_grad_(True)
    GM, GA, _ = mf.grams_plain(rec, Vm, T32)
    loss_of(GM, GA).backward()
    assert _rel(Vm.grad.numpy(), Vs.grad.numpy()) < 1e-9
    # w.r.t. ψ through the whole chain ψ → u → Grams (sparse: C·vec(Pψ))
    keep = 1.0                  # (no wall masking: a synthetic mesh has only wall DOFs; the mask is a constant factor)
    ps = torch.from_numpy(psi).requires_grad_(True)
    C = _to_sparse(ops["C"])
    coef = torch.einsum("tlq,tqmc->tclm", torch.from_numpy(ops["P"]), ps).reshape(1, -1, m)
    u = spmm(C, coef, _to_sparse(ops["C"].T.tocsr()))[0] * keep
    GM, GA, _ = hcurl_grams_plain(u[None], T32[None], batch)
    loss_of(GM[0], GA[0]).backward()
    pm = torch.from_numpy(psi).requires_grad_(True)
    um = mf.psi_to_dofs(rec, pm) * keep
    GM, GA, _ = mf.grams_plain(rec, um, T32)
    loss_of(GM, GA).backward()
    assert _rel(pm.grad.numpy(), ps.grad.numpy()) < 1e-9


def _to_sparse(A):
    A = A.tocsr()
    return torch.sparse_csr_tensor(torch.from_numpy(A.indptr.astype(np.int64)), torch.from_numpy(A.indices.astype(np.int64)),
                                   torch.from_numpy(A.data), A.shape)


def test_chunked_equals_unchunked(cases):
    _, rec, _ = cases["bisected_single_level"]
    tr = mf.to_torch(rec)
    V, T, psi = (torch.from_numpy(a) for a in _random(rec))
    Me, Ke = mf.element_matrices(tr, chunk=None)
    Me7, Ke7 = mf.element_matrices(tr, chunk=7)
    assert float((Me - Me7).abs().max()) == 0 and float((Ke - Ke7).abs().max()) == 0     # per-element maths
    d = tr["dofs"]
    assert _rel(mf.apply(Me, d, V, chunk=13).numpy(), mf.apply(Me, d, V).numpy()) < 1e-13
    assert _rel(mf.gram(Ke, d, V, T, chunk=13).numpy(), mf.gram(Ke, d, V, T).numpy()) < 1e-13
    u = mf.psi_to_dofs(tr, psi, chunk=None)
    assert _rel(mf.psi_to_dofs(tr, psi, chunk=11).numpy(), u.numpy()) < 1e-13
    assert _rel(mf.psi_map(tr, chunk=11).numpy(), mf.psi_map(tr, chunk=None).numpy()) < 1e-13
    G = mf.grams_plain(rec, V, T, chunk=17)
    G0 = mf.grams_plain(rec, V, T)
    assert all(_rel(a.numpy(), b.numpy()) < 1e-13 for a, b in zip(G, G0, strict=True))


def test_wrong_geometry_order_is_refused():
    mesh = Mesh(OCCGeometry(_cylinder()).GenerateMesh(maxh=0.06))
    mesh.Curve(4)                                  # the element map is quartic: not representable by 20 monomials
    with pytest.raises(ValueError, match="cubic"):
        mf.build_record(mesh)
    mesh2 = Mesh(OCCGeometry(_cylinder()).GenerateMesh(maxh=0.06))
    mesh2.Curve(2)                                  # lower orders are exact
    rec = mf.build_record(mesh2)
    ops = ho.field_operators(mesh2, 2, potential=False)
    Me, Ke = mf.element_matrices(rec)
    V = np.random.default_rng(0).standard_normal((rec["n_dof"], 3))
    assert _rel(mf.apply(Ke, mf.to_torch(rec)["dofs"], torch.from_numpy(V)).numpy(), ops["K"] @ V) < TOL_OP
    with pytest.raises(NotImplementedError):
        mf.build_record(mesh2, order=3)


def test_tables_cache_round_trip_and_version_key(tmp_path, monkeypatch):
    monkeypatch.setenv("RFCAV_CACHE", str(tmp_path))
    mf._MEMO.clear()
    p = mf.table_path()
    assert p.startswith(str(tmp_path)) and f"_f{mf.FORMAT}_" in p and not os.path.exists(p)
    t1 = mf.reference_tables()                                      # built on first use, written atomically
    assert os.path.exists(p) and not [f for f in os.listdir(tmp_path) if f.endswith(".tmp.npz")]
    assert t1["T"].shape == (24, 30, 138) and t1["phi"].shape == (24, 46, 30, 3) and t1["L14"].shape == (24, 14, 10)
    mf._MEMO.clear()
    monkeypatch.setattr(mf, "_build_tables", lambda *a, **k: pytest.fail("rebuilt although cached"))
    t2 = mf.reference_tables()                                      # read from disk
    for k in ("T", "phi", "crl", "L46", "L14", "ref46", "w14"):
        assert np.array_equal(t1[k], t2[k])
    # another ngsolve version / format number → another file, rebuilt
    built = []
    monkeypatch.setattr(mf, "_build_tables", lambda *a, **k: built.append(1) or {k: t1[k] for k in mf._TABLE_KEYS})
    mf._MEMO.clear()
    monkeypatch.setattr(mf, "_ngsolve_version", lambda: "9.9.9-test")
    assert mf.table_path() != p
    mf.reference_tables()
    assert built == [1] and os.path.exists(mf.table_path())
    mf._MEMO.clear()
    monkeypatch.setattr(mf, "FORMAT", mf.FORMAT + 1)
    assert mf.table_path() != p
    # a corrupt cache file is rebuilt, not trusted
    with open(mf.table_path(), "wb") as fh:
        fh.write(b"not an npz")
    mf._MEMO.clear()
    mf.reference_tables()
    assert built == [1, 1]
    mf._MEMO.clear()
    with pytest.raises(NotImplementedError):
        mf.reference_tables(order=3)


def test_tables_are_independent_of_any_data_and_exact_on_the_reference_tet():
    tab = mf.reference_tables()
    assert tab["w46"].min() > 0 and abs(tab["w46"].sum() - 1 / 6) < 1e-13 and abs(tab["w14"].sum() - 1 / 6) < 1e-13
    # L2 partition: the constant function is in the span at every point (some coefficient vector gives 1)
    for k in range(24):
        c = np.linalg.lstsq(tab["L46"][k], np.ones(46), rcond=None)[0]
        assert np.abs(tab["L46"][k] @ c - 1).max() < 1e-12
    # classes are the rank permutations of the vertex numbers: 24 distinct ones
    perms = np.array(mf.PERMS)
    assert len(set(map(tuple, perms))) == 24 and (mf.element_classes(perms) == np.arange(24)).all()
    assert (mf.element_classes(perms * 3 + 10) == np.arange(24)).all()      # only the ranks matter


def test_check_against_ngsolve_helper(cases):
    mesh, rec, ops = cases["pillbox"]
    err = mf.check_against_ngsolve(mesh, ops=ops, scale=rec["scale"], center=rec["center"])
    assert max(v for k, v in err.items() if k != "classes") < TOL_OP


def test_tables_survive_a_truncated_cache_file_and_an_unwritable_cache_dir(tmp_path, monkeypatch):
    import warnings as _w
    monkeypatch.setenv("RFCAV_CACHE", str(tmp_path / "c"))
    monkeypatch.setattr(mf, "_MEMO", {})
    ref = mf.reference_tables()
    path = mf.table_path()
    with open(path, "rb") as f:
        head = f.read(4096)
    with open(path, "wb") as f:                                     # zip magic, no central directory: zipfile.BadZipFile
        f.write(head)
    mf._MEMO.clear()
    tab = mf.reference_tables()                                # rebuilt, not raised
    assert np.array_equal(tab["phi"], ref["phi"]) and np.array_equal(tab["T"], ref["T"])
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("RFCAV_CACHE", str(blocker / "sub"))         # makedirs fails: a file is in the way
    mf._MEMO.clear()
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        tab = mf.reference_tables()
    assert np.array_equal(tab["phi"], ref["phi"])
    assert any("not cached" in str(r.message) for r in rec)
