"""src/data/mesh_features.py: display geometry + vertex features of a mesh, boundary topology, and the generator's
--start_id sharding.  Meshes: skfem tensor boxes with a random vertex permutation, and generator meshes.

- scale / translation invariance of the normalised mesh and the features (and of torsion_max)
- features finite and in range; distance to the wall exact on a box
- boundary_topology: an isolated conductor (two boundary shells, b2 = 1)
"""
import h5py
import numpy as np
import pytest

pytest.importorskip("skfem")

import src.data.mesh_features as mf  # noqa: E402
from src.data.mesh_features import FEATURE_NAMES_3D  # noqa: E402


def _box_mesh(a=1.0, b=0.8, d=0.6, n=5, seed=0):
    """Structured box tets with a random vertex permutation (non-trivial edge orientations)."""
    from skfem import MeshTet
    m = MeshTet.init_tensor(np.linspace(0, a, n + 1), np.linspace(0, b, n), np.linspace(0, d, n - 1))
    perm = np.random.default_rng(seed).permutation(m.p.shape[1])
    inv = np.argsort(perm)
    return m.p.T[perm], inv[m.t.T]


def test_scale_invariance():
    nodes, tets = _box_mesh()
    g1 = mf.display_geometry(nodes, tets)
    s, t = 2.7e-2, np.array([0.3, -1.0, 5.0])
    g2 = mf.display_geometry(nodes * s + t, tets)
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
    assert g1["Input_funcs"].shape == (len(nodes), len(FEATURE_NAMES_3D)) and g1["n_nodes"] == len(nodes)
    assert np.isclose(g2["volume"], g1["volume"], rtol=1e-9)                    # normalised volume


def test_features_valid():
    nodes, tets = _box_mesh(n=6)
    g = mf.display_geometry(nodes, tets)
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


def test_shards_start_id_reproduce_the_unsharded_samples(tmp_path, run_generator):
    """dataset_generator_3d --start_id shards (same seed) reproduce the geometry draws of the unsharded ids."""
    pytest.importorskip("gmsh")
    import src.data_gen.dataset_generator_3d as gen
    from src.data_gen import cavity_shapes as cs
    fam = ["pillbox", "box"]
    cheap = ["--n_workers", "1", "--mesh_size", "0.3", "--n_eigen_modes", "2", "--seed", "3", "--families", *fam,
             "--no_adapt", "--label_order", "2", "--model_order", "1"]
    got = {}
    for name, (start, n) in (("a.h5", (0, 1)), ("b.h5", (1, 2))):
        run_generator("--n_total", n, "--start_id", start, "--h5_filename", tmp_path / name, *cheap)
        with h5py.File(tmp_path / name, "r") as f:
            got.update({int(k.split("_")[-1]): (str(f[k].attrs["shape_type"]), float(f[k].attrs["volume"]))
                        for k in f if k.startswith("sample_")})
    assert sorted(got) == [0, 1, 2]
    old, old_fillet = gen.ARGS, cs.MIN_FILLET
    gen.ARGS = gen.parse_args(cheap)
    try:
        for sid, (shape, vol) in got.items():                              # = the unsharded draw of that id
            g = gen.mesh_sample(sid)
            assert g["shape_type"] == shape and g["volume"] == pytest.approx(vol, rel=1e-12)
    finally:
        gen.ARGS, cs.MIN_FILLET = old, old_fillet


def test_boundary_topology_isolated_conductor():
    """Box with a floating inner box (two boundary shells, b2 = 1, b1 = 0): two components, the second one is
    the inner shell, every boundary edge is shared by two boundary faces (compact manifold)."""
    from skfem import MeshTet
    m = MeshTet.init_tensor(*(np.linspace(0, 1, 7),) * 3)
    c = m.p[:, m.t].mean(1)
    keep = ~np.all((c > 1 / 3) & (c < 2 / 3), axis=0)                  # remove the central 2×2×2 cells
    used = np.unique(m.t[:, keep])
    remap = np.full(m.p.shape[1], -1)
    remap[used] = np.arange(len(used))
    nodes, tets = m.p.T[used], remap[m.t[:, keep].T]
    bt = mf.boundary_topology(tets, len(nodes))
    assert bt["n_comp"] == 2 and bt["betti1"] == 0 and bt["manifold"]
    inner = bt["vert_comp"] == 1
    assert np.all((nodes[inner] >= 1 / 3 - 1e-12) & (nodes[inner] <= 2 / 3 + 1e-12))   # component 1 = inner shell
    assert inner.sum() > 0 and (bt["vert_comp"] < 0).sum() > 0                          # interior vertices exist


def test_boundary_topology_solid_box_and_faces_outward():
    nodes, tets = _box_mesh(n=4)
    bt = mf.boundary_topology(tets, len(nodes))
    assert bt["n_comp"] == 1 and bt["betti1"] == 0 and bt["manifold"]
    bf = mf.boundary_faces(nodes, tets)
    p0, p1, p2 = nodes[bf[:, 0]], nodes[bf[:, 1]], nodes[bf[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    cen = (p0 + p1 + p2) / 3 - nodes.mean(0)
    assert np.all((n * cen).sum(1) > 0)                                              # outward
    assert np.isclose(0.5 * np.linalg.norm(n, axis=1).sum(), 2 * (1.0 * 0.8 + 1.0 * 0.6 + 0.8 * 0.6))


def test_features_finite_without_interior_vertex():
    """A mesh whose vertices are all on the wall has no torsion maximum to divide by: the torsion feature is 0
    (docs/31 §6 item 19: it was 0/0 = NaN, which made GET /features a 500)."""
    nodes = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    g = mf.display_geometry(nodes, tets)
    assert np.isfinite(g["Input_funcs"]).all() and g["torsion_max"] == 0.0 and np.all(g["Input_funcs"][:, 8] == 0)
