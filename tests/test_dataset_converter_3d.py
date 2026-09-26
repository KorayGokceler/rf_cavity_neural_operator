"""Tests for the 3D converter src/data/dataset_converter_3d.py (PKL contract of docs/19).

Invariants:
- closed-form Whitney M, K == skfem ElementTetN0 (after mapping to the contract edge order)
- K·G = 0, Kp = GᵀMG, Kp·1 = 0
- Rayleigh(Y) with the stored normalised K, M reproduces f = c√λ/(2π·scale); ‖Y‖_M = 1; GᵀMY = 0
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


@pytest.fixture(scope="module")
def pkl(tmp_path_factory):
    pytest.importorskip("gmsh")
    import src.data_gen.dataset_generator_3d as gen
    d = tmp_path_factory.mktemp("d3")
    h5 = d / "s.h5"
    old = gen.ARGS
    try:
        gen.main(["--n_total", "3", "--n_workers", "3", "--mesh_size", "0.2", "--n_eigen_modes", "4",
                  "--h5_filename", str(h5), "--seed", "3"])
    finally:
        gen.ARGS = old
    out = d / "s.pkl"
    conv.RFCavity3DConverter(str(h5)).convert_dataset(str(out))
    with open(out, "rb") as f:
        return pickle.load(f), h5


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


def test_contract_structure(pkl):
    data, _ = pkl
    assert set(data) == {"geometry_pool", "samples", "metadata"}
    md = data["metadata"]
    assert md["field"] == "H" and md["element"] == "N0" and md["n_modes"] == 4
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
    for s in data["samples"]:
        g = data["geometry_pool"][s["geom_id"]]
        assert s["Y"].dtype == np.float32 and s["Y"].shape == (g["n_edges"],)
        assert s["Theta"].dtype == np.float32 and s["Theta"].shape == (3,) and s["Theta"][2] == s["geom_id"]


def test_operators_and_rayleigh(pkl):
    data, h5 = pkl
    with h5py.File(h5, "r") as f:
        for s in data["samples"]:
            g = data["geometry_pool"][s["geom_id"]]
            M, K, G, Kp = _ops(g)
            assert abs(K @ G).max() < 1e-11 * abs(K).max()
            assert abs(Kp - (G.T @ M @ G)).max() < 1e-12 * abs(Kp).max()
            assert abs(Kp @ np.ones(g["n_nodes"])).max() < 1e-12 * abs(Kp).max()
            y = s["Y"].astype(np.float64)
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


def test_lean_pkl_rebuilds_operators(pkl, tmp_path):
    full, h5 = pkl
    out = tmp_path / "lean.pkl"
    conv.RFCavity3DConverter(str(h5)).convert_dataset(str(out), store_operators=False, mode_indices=[0, 1])
    with open(out, "rb") as f:
        lean = pickle.load(f)
    assert lean["metadata"]["operators_stored"] is False and lean["metadata"]["n_modes"] == 2
    for gid, g in lean["geometry_pool"].items():
        assert not {"M", "K", "G", "Kp"} & set(g)
        assert np.isfinite(g["freq_next"])
        ops, M = conv.geometry_operators(g["X"].astype(np.float64), g["tets"])   # from the stored float32 X
        assert np.array_equal(ops["edges"], g["edges"])
        K = from_csr_tuple(ops["K"], M.shape)
        K_full = from_csr_tuple(full["geometry_pool"][gid]["K"], M.shape)
        assert abs(K - K_full).max() < 1e-5 * abs(K_full).max()
        for s in (s for s in lean["samples"] if s["geom_id"] == gid):
            y = s["Y"].astype(np.float64)
            f_rq = C0 * np.sqrt((y @ (K @ y)) / (y @ (M @ y))) / (2 * np.pi * g["scale"]) / 1e9
            assert np.isclose(f_rq, s["Theta"][1], rtol=1e-5)
