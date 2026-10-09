"""Tests for the 3D converter src/data/dataset_converter_3d.py (PKL contract of docs/19).

Invariants:
- closed-form Whitney M, K == skfem ElementTetN0 (after mapping to the contract edge order)
- K·G = 0, Kp = GᵀMG SPD on the n_pot block, G columns ≥ n_pot zero, G rows and Y rows of the
  PEC wall edges (bnd_edge) zero, one potential per extra boundary shell
- Rayleigh(Y) with the stored normalised K, M reproduces f = c√λ/(2π·scale); ‖Y‖_M = 1; GᵀMY = 0
- lean PKL: geometry_operators(X, tets) rebuilds the stored operators
- data of the removed H formulation is rejected
- scale invariance: scaled / translated input → same X, features, M, K; λ_phys ∝ 1/scale²
- edge orientation: low → high, lexicographic rows, flipped H5 edges handled by sign
- features finite and in range; distance exact on a box
"""
import pickle

import h5py
import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("skfem")

import src.data.dataset_converter_3d as conv  # noqa: E402
from src.data.dataset_converter_3d import FEATURE_NAMES_3D, from_csr_tuple  # noqa: E402

C0 = 299792458.0


def _box_mesh(a=1.0, b=0.8, d=0.6, n=5, seed=0):
    """Structured box tets with a random vertex permutation (non-trivial edge orientations)."""
    from skfem import MeshTet
    m = MeshTet.init_tensor(np.linspace(0, a, n + 1), np.linspace(0, b, n), np.linspace(0, d, n - 1))
    perm = np.random.default_rng(seed).permutation(m.p.shape[1])
    inv = np.argsort(perm)
    return m.p.T[perm], inv[m.t.T]


def _make_pkl(tmp_path_factory, name, families):
    pytest.importorskip("gmsh")
    import src.data_gen.dataset_generator_3d as gen
    d = tmp_path_factory.mktemp(f"d3{name}")
    h5 = d / "s.h5"
    old = gen.ARGS
    try:
        gen.main(["--n_total", "3", "--n_workers", "3", "--mesh_size", "0.2", "--n_eigen_modes", "4",
                  "--h5_filename", str(h5), "--seed", "3", "--families", *families])
    finally:
        gen.ARGS = old
    out = d / "s.pkl"
    conv.RFCavity3DConverter(str(h5)).convert_dataset(str(out))
    with open(out, "rb") as f:
        return pickle.load(f), h5


@pytest.fixture(scope="module")
def pkl(tmp_path_factory):
    """Handle families (b1 > 0)."""
    return _make_pkl(tmp_path_factory, "handles", ["hwr", "spoke", "dtl"])


@pytest.fixture(scope="module")
def pkl_balls(tmp_path_factory):
    return _make_pkl(tmp_path_factory, "balls", ["pillbox", "composite"])


@pytest.fixture(params=["handles", "balls"])
def any_pkl(request):
    return request.getfixturevalue("pkl" if request.param == "handles" else "pkl_balls")


def _ops(g):
    ne, nv = g["n_edges"], g["n_nodes"]
    return (from_csr_tuple(g["M"], (ne, ne)), from_csr_tuple(g["K"], (ne, ne)),
            from_csr_tuple(g["G"], (ne, nv)), from_csr_tuple(g["Kp"], (nv, nv)))


def test_assembly_matches_skfem():
    from skfem import Basis, BilinearForm, ElementTetN0, MeshTet
    from skfem.helpers import curl, dot
    nodes, tets = _box_mesh()
    M, K, edges = conv.assemble_n0(nodes, tets)
    mesh = MeshTet(nodes.T.copy(), tets.T.copy())
    basis = Basis(mesh, ElementTetN0())
    Ms = BilinearForm(lambda u, v, w: dot(u, v)).assemble(basis).tocsr()
    Ks = BilinearForm(lambda u, v, w: dot(curl(u), curl(v))).assemble(basis).tocsr()
    e_sk = np.empty((basis.N, 2), dtype=np.int64)
    e_sk[basis.edge_dofs[0]] = mesh.edges.T
    rows, sign = conv.h5_edges_to_canonical(e_sk, len(nodes), edges)
    Pm = sp.csr_matrix((sign, (rows, np.arange(len(rows)))), shape=(len(rows),) * 2)  # skfem → contract
    for A, As in ((M, Ms), (K, Ks)):
        assert abs(A - Pm @ As @ Pm.T).max() < 1e-12 * abs(A).max()


def test_edge_convention_and_gradient():
    nodes, tets = _box_mesh()
    M, K, edges = conv.assemble_n0(nodes, tets)
    assert np.all(edges[:, 0] < edges[:, 1])
    assert np.array_equal(edges, np.unique(edges, axis=0))                 # lexicographic rows
    G = conv.discrete_gradient(edges, len(nodes))
    assert abs(K @ G).max() < 1e-12 * abs(K).max()
    # grad of φ = x is e_x: DOFs x_high − x_low, ‖e_x‖²_M = |Ω|
    u = G @ nodes[:, 0]
    np.testing.assert_allclose(u, nodes[edges[:, 1], 0] - nodes[edges[:, 0], 0])
    assert np.isclose(u @ (M @ u), 1.0 * 0.8 * 0.6, rtol=1e-12)
    # flipped H5 orientation → sign −1
    e_h5 = edges.copy()
    e_h5[::2] = e_h5[::2, ::-1]
    rows, sign = conv.h5_edges_to_canonical(e_h5[::-1], len(nodes), edges)
    assert np.array_equal(rows, np.arange(len(edges))[::-1])
    assert np.array_equal(sign[::-1][::2], -np.ones(len(edges[::2])))


def test_contract_structure(any_pkl):
    data, _ = any_pkl
    assert set(data) == {"geometry_pool", "samples", "metadata"}
    md = data["metadata"]
    assert md["field"] == "E" and md["element"] == "N0" and md["n_modes"] == 4
    assert md["feature_names"] == FEATURE_NAMES_3D and len(FEATURE_NAMES_3D) == 9
    assert md["n_geometries"] == len(data["geometry_pool"]) == 3 and md["n_samples"] == len(data["samples"]) == 12
    assert {"mean", "std"} <= set(md["freq_stats"])
    for g in data["geometry_pool"].values():
        nv, ne = g["n_nodes"], g["n_edges"]
        assert g["X"].dtype == np.float32 and g["X"].shape == (nv, 3)
        assert g["Input_funcs"].dtype == np.float32 and g["Input_funcs"].shape == (nv, 9)
        assert g["edges"].dtype == np.int64 and g["edges"].shape == (ne, 2)
        assert g["tets"].dtype == np.int64 and g["tets"].shape[1] == 4
        for k in ("M", "K", "G", "Kp"):
            ip, ix, dt = g[k]
            assert ip.dtype == np.int64 and ix.dtype == np.int64 and dt.dtype == np.float64
        assert isinstance(g["scale"], float) and g["center"].dtype == np.float64 and g["center"].shape == (3,)
        assert isinstance(g["shape_type"], str) and g["torsion_max"] > 0
        assert np.isclose(np.abs(g["X"]).max(), 1.0, atol=1e-6)
        assert np.array_equal(g["edges"], conv.edges_from_tets(g["tets"]))
        assert {"bnd_edge", "n_pot", "n_bnd_components", "betti1"} <= set(g)
        assert g["bnd_edge"].dtype == bool and g["bnd_edge"].shape == (ne,) and 0 < g["bnd_edge"].sum() < ne
        assert isinstance(g["n_pot"], int) and 0 < g["n_pot"] < nv and g["n_bnd_components"] >= 1
    for s in data["samples"]:
        g = data["geometry_pool"][s["geom_id"]]
        assert s["Y"].dtype == np.float32 and s["Y"].shape == (g["n_edges"],)
        assert s["Theta"].dtype == np.float32 and s["Theta"].shape == (3,) and s["Theta"][2] == s["geom_id"]


def test_operators_and_rayleigh(any_pkl):
    data, h5 = any_pkl
    betti = set()
    for g in data["geometry_pool"].values():
        M, K, G, Kp = _ops(g)
        assert abs(K @ G).max() < 1e-11 * abs(K).max()
        assert abs(Kp - (G.T @ M @ G)).max() < 1e-12 * abs(Kp).max()
        n_pot, bnd = g["n_pot"], g["bnd_edge"]
        betti.add(g["betti1"])
        assert abs(G[:, n_pot:]).sum() == 0 and abs(G[bnd]).sum() == 0 and abs(Kp[n_pot:]).sum() == 0
        assert np.linalg.eigvalsh(Kp[:n_pot, :n_pot].toarray()).min() > 0            # SPD, no pinning
        # potentials: interior vertices in increasing order, then one per extra boundary shell
        bt = conv.boundary_topology(g["tets"], g["n_nodes"])
        interior = np.flatnonzero(bt["vert_comp"] < 0)
        assert n_pot == len(interior) + bt["n_comp"] - 1 == len(interior) + g["n_bnd_components"] - 1
        Gf = conv.discrete_gradient(g["edges"], g["n_nodes"])
        assert abs(G[:, :len(interior)] - Gf[:, interior]).sum() == 0
        # wall edges = edges of boundary faces
        bf_edges = np.unique(np.sort(bt["faces"][:, [[0, 1], [1, 2], [0, 2]]].reshape(-1, 2), axis=1), axis=0)
        assert np.array_equal(g["edges"][bnd], bf_edges)
    assert max(betti) >= 1 or all(g["shape_type"] in ("pillbox", "composite")
                                  for g in data["geometry_pool"].values())
    with h5py.File(h5, "r") as f:
        for s in data["samples"]:
            g = data["geometry_pool"][s["geom_id"]]
            M, K, G, Kp = _ops(g)
            y = s["Y"].astype(np.float64)
            assert np.all(s["Y"][g["bnd_edge"]] == 0)                           # n × E = 0 exactly
            my = M @ y
            assert np.isclose(y @ my, 1.0, rtol=1e-5)
            assert np.abs(G.T @ my).max() < 1e-5 * np.abs(my).max()        # float32 round-off level
            f_rq = C0 * np.sqrt((y @ (K @ y)) / (y @ my)) / (2 * np.pi * g["scale"]) / 1e9
            f_h5 = f[f"sample_{s['geom_id']:04d}"]["freqs"][int(s["Theta"][0])]
            assert np.isclose(f_rq, s["Theta"][1], rtol=1e-5) and np.isclose(f_rq, f_h5, rtol=1e-5)


def test_scale_invariance():
    nodes, tets = _box_mesh()
    g1, M1 = conv.extract_geometry_3d(nodes, tets)
    s, t = 2.7e-2, np.array([0.3, -1.0, 5.0])
    g2, M2 = conv.extract_geometry_3d(nodes * s + t, tets)
    assert np.isclose(g2["scale"], g1["scale"] * s, rtol=1e-12)
    np.testing.assert_allclose(g2["center"], g1["center"] * s + t, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g2["X"], g1["X"], atol=1e-6)
    # direction to the nearest wall is ambiguous on the medial axis (ties between walls)
    X = g1["X"].astype(np.float64)
    wd = np.sort(np.concatenate([X - X.min(0), X.max(0) - X], axis=1), axis=1)
    unique = (wd[:, 1] - wd[:, 0]) > 1e-6
    cols = [i for i in range(9) if i not in (4, 5, 6)]
    np.testing.assert_allclose(g2["Input_funcs"][:, cols], g1["Input_funcs"][:, cols], atol=1e-5)
    np.testing.assert_allclose(g2["Input_funcs"][unique, 4:7], g1["Input_funcs"][unique, 4:7], atol=1e-5)
    assert np.isclose(g2["torsion_max"], g1["torsion_max"], rtol=1e-9)
    for k in ("M", "K"):
        np.testing.assert_allclose(g2[k][2], g1[k][2], rtol=1e-8, atol=1e-12 * np.abs(g1[k][2]).max())
    # physical operators: M_phys = scale·M_norm, K_phys = K_norm/scale ⇒ λ_phys = λ_norm/scale²
    Mp, Kp_, _ = conv.assemble_n0(nodes * s + t, tets)
    ne = g1["n_edges"]
    for A_phys, key, fac in ((Mp, "M", 1 / g2["scale"]), (Kp_, "K", g2["scale"])):
        A_norm = from_csr_tuple(g2[key], (ne, ne))
        assert abs(A_phys * fac - A_norm).max() < 1e-9 * abs(A_norm).max()


def test_features_valid():
    nodes, tets = _box_mesh(n=6)
    g, _ = conv.extract_geometry_3d(nodes, tets)
    F = g["Input_funcs"].astype(np.float64)
    assert np.all(np.isfinite(F))
    X = g["X"].astype(np.float64)
    np.testing.assert_allclose(F[:, :3], X, atol=1e-7)
    # exact box distance in normalised units
    lo, hi = X.min(0), X.max(0)
    d_exact = np.minimum(X - lo, hi - X).min(1)
    np.testing.assert_allclose(F[:, 3], d_exact, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(F[:, 4:7], axis=1), 1.0, atol=1e-5)
    interior = d_exact > 1e-6
    # direction points to the nearest wall: moving by dist along it reaches ∂Ω
    q = X[interior] + F[interior, 3:4] * F[interior, 4:7]
    assert np.abs(np.minimum(q - lo, hi - q).min(1)).max() < 1e-5
    bnd = ~interior                                  # boundary vertices: outward normal
    assert np.all((F[bnd, 4:7] * X[bnd]).sum(1) > 0)
    assert F[:, 7].min() > 0 and np.isclose(F[:, 7].max(), 1.0)
    assert F[:, 8].min() >= 0 and np.isclose(F[:, 8].max(), 1.0) and np.all(F[bnd, 8] == 0)


def test_lean_pkl_rebuilds_operators(any_pkl, tmp_path):
    full, h5 = any_pkl
    out = tmp_path / "lean.pkl"
    conv.RFCavity3DConverter(str(h5)).convert_dataset(str(out), store_operators=False, mode_indices=[0, 1])
    with open(out, "rb") as f:
        lean = pickle.load(f)
    assert lean["metadata"]["operators_stored"] is False and lean["metadata"]["n_modes"] == 2
    assert lean["metadata"]["field"] == "E"
    for gid, g in lean["geometry_pool"].items():
        assert not {"M", "K", "G", "Kp"} & set(g)
        assert np.isfinite(g["freq_next"])
        ops, M = conv.geometry_operators(g["X"].astype(np.float64), g["tets"])  # stored float32 X
        gf = full["geometry_pool"][gid]
        assert set(gf) - set(g) == {"M", "K", "G", "Kp"}
        assert {k for k in ops} | set(g) == set(gf)                             # same keys as the full PKL
        for k in set(ops) - {"M", "K", "G", "Kp"}:                              # topology: identical
            assert np.array_equal(ops[k], gf[k]) and np.array_equal(g[k], gf[k]), k
        A, B = ops["G"], gf["G"]                                                # G: integer, exact
        assert all(np.array_equal(a, b) for a, b in zip(A, B, strict=True))
        Kp, Kp_full = from_csr_tuple(ops["Kp"], (g["n_nodes"],) * 2), from_csr_tuple(gf["Kp"], (g["n_nodes"],) * 2)
        assert abs(Kp - Kp_full).max() < 1e-5 * abs(Kp_full).max()             # float32 X round-off
        K = from_csr_tuple(ops["K"], M.shape)
        K_full = from_csr_tuple(gf["K"], M.shape)
        assert abs(K - K_full).max() < 1e-5 * abs(K_full).max()
        for s in (s for s in lean["samples"] if s["geom_id"] == gid):
            y = s["Y"].astype(np.float64)
            f_rq = C0 * np.sqrt((y @ (K @ y)) / (y @ (M @ y))) / (2 * np.pi * g["scale"]) / 1e9
            assert np.isclose(f_rq, s["Theta"][1], rtol=1e-5)


def test_shards_start_id_and_multi_file_conversion(pkl, tmp_path):
    """dataset_generator_3d --start_id shards (same seed) reproduce the unsharded samples;
    the converter merges them and rejects overlapping id ranges."""
    import src.data_gen.dataset_generator_3d as gen
    full, _ = pkl
    shards = [tmp_path / "a.h5", tmp_path / "b.h5"]
    old = gen.ARGS
    try:
        for h5, (start, n) in zip(shards, ((0, 1), (1, 2)), strict=True):
            gen.main(["--n_total", str(n), "--start_id", str(start), "--n_workers", "2", "--mesh_size", "0.2",
                      "--n_eigen_modes", "4", "--h5_filename", str(h5), "--seed", "3",
                      "--families", "hwr", "spoke", "dtl"])
    finally:
        gen.ARGS = old
    out = tmp_path / "merged.pkl"
    conv.RFCavity3DConverter([str(p) for p in shards]).convert_dataset(str(out))
    with open(out, "rb") as f:
        merged = pickle.load(f)
    assert sorted(merged["geometry_pool"]) == sorted(full["geometry_pool"]) == [0, 1, 2]
    for gid, g in merged["geometry_pool"].items():
        assert np.array_equal(g["tets"], full["geometry_pool"][gid]["tets"])
    with pytest.raises(ValueError, match="duplicate sample ids"):
        conv.RFCavity3DConverter([str(shards[1]), str(shards[1])]).convert_dataset(str(tmp_path / "dup.pkl"))


def test_e_operators_isolated_conductor():
    """Box with a floating inner box (two boundary shells, b2 = 1): one extra potential column
    = G_full @ indicator(inner shell); Kp SPD; wall rows zero; the kernel K·G = 0 still holds."""
    from skfem import MeshTet
    m = MeshTet.init_tensor(*(np.linspace(0, 1, 7),) * 3)
    c = m.p[:, m.t].mean(1)
    keep = ~np.all((c > 1 / 3) & (c < 2 / 3), axis=0)                  # remove the central 2×2×2 cells
    used = np.unique(m.t[:, keep])
    remap = np.full(m.p.shape[1], -1)
    remap[used] = np.arange(len(used))
    nodes, tets = m.p.T[used], remap[m.t[:, keep].T]
    ops, M = conv.geometry_operators(nodes, tets)
    bt = conv.boundary_topology(tets, len(nodes))
    assert ops["n_bnd_components"] == bt["n_comp"] == 2 and ops["betti1"] == 0 and bt["manifold"]
    inner = bt["vert_comp"] == 1
    assert np.all((nodes[inner] >= 1 / 3 - 1e-12) & (nodes[inner] <= 2 / 3 + 1e-12))   # component 1 = inner shell
    n_int = int((bt["vert_comp"] < 0).sum())
    assert ops["n_pot"] == n_int + 1
    ne, nv = len(ops["edges"]), len(nodes)
    G, Kp, K = from_csr_tuple(ops["G"], (ne, nv)), from_csr_tuple(ops["Kp"], (nv, nv)), from_csr_tuple(ops["K"], (ne, ne))
    Gf = conv.discrete_gradient(ops["edges"], nv)
    np.testing.assert_array_equal(G[:, n_int].toarray().ravel(), Gf @ inner.astype(float))
    assert abs(G[ops["bnd_edge"]]).sum() == 0 and abs(G[:, ops["n_pot"]:]).sum() == 0
    assert abs(K @ G).max() < 1e-11 * abs(K).max()
    assert np.linalg.eigvalsh(Kp[:ops["n_pot"], :ops["n_pot"]].toarray()).min() > 0


def test_h_formulation_data_rejected(pkl, tmp_path):
    """H5 of the removed H formulation (metadata field 'H'): a clear error, not silent garbage."""
    import json
    import shutil
    h5 = tmp_path / "h.h5"
    shutil.copy(pkl[1], h5)
    with h5py.File(h5, "a") as f:
        meta = json.loads(f.attrs["metadata"])
        f.attrs["metadata"] = json.dumps({**meta, "field": "H"})
    with pytest.raises(ValueError, match="removed H formulation"):
        conv.RFCavity3DConverter(str(h5)).convert_dataset(str(tmp_path / "x.pkl"))
