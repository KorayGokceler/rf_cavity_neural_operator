"""src/viz/nedelec.py (contract §A1): Whitney N0 reconstruction, point location,
plane sampling. Uses tests.maxwell3d_synth (skfem box mesh, jittered interior)."""
import numpy as np
import pytest

skfem = pytest.importorskip("skfem")

import src.viz.nedelec as nd                                          # noqa: E402
from tests.maxwell3d_synth import box_geometry                        # noqa: E402


@pytest.fixture(scope="module")
def mesh():
    """Jittered box (skfem edge order, NOT lexicographically sorted)."""
    geom, _, _ = box_geometry(dims=(1.0, 0.8, 0.6), n=4, jitter=0.15, seed=3)
    X = geom["X"].astype(np.float64)
    tets, edges = geom["tets"], geom["edges"]
    return X, tets, edges


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


def _random_interior_points(X, tets, rng, n=25):
    """Points strictly inside random tets (positive Dirichlet barycentric weights)."""
    idx = rng.integers(0, len(tets), size=n)
    bw = rng.dirichlet(np.ones(4), size=n)
    pts = np.einsum("pv,pvc->pc", bw, X[tets[idx]])
    return pts, idx, bw


# ───────────────────────── 1. constant field reproduction ─────────────────

def test_constant_field_reproduction(mesh, rng):
    X, tets, edges = mesh
    c = rng.normal(size=3)
    xa, xb = X[edges[:, 0]], X[edges[:, 1]]
    u = (c[None, :] * (xb - xa)).sum(-1)                    # DOF = line integral of c

    Hc = nd.cell_field(X, tets, edges, u)
    assert Hc.shape == (len(tets), 3)
    np.testing.assert_allclose(Hc, np.broadcast_to(c, Hc.shape), atol=1e-8)

    pts, _, _ = _random_interior_points(X, tets, rng)
    He = nd.eval_field(X, tets, edges, u, pts)
    assert He.shape == (len(pts), 3)
    np.testing.assert_allclose(He, np.broadcast_to(c, He.shape), atol=1e-6)


# ───────────────────────── 2. cross-check against skfem ElementTetN0 ──────

def _skfem_eval(X, tets, edges, u, pts):
    """skfem N0 interpolation at `pts` [P,3]; probes' comp-major flatten -> [P,3]."""
    m = skfem.MeshTet(np.ascontiguousarray(X.T), np.ascontiguousarray(tets.T))
    basis = skfem.Basis(m, skfem.ElementTetN0())
    assert np.array_equal(m.edges.T, edges)   # box_geometry's edges IS mesh.edges
    P = basis.probes(pts.T)
    return (P @ u).reshape(3, len(pts)).T


def test_matches_skfem_interpolation(mesh, rng):
    X, tets, edges = mesh
    u = rng.normal(size=len(edges))
    pts = np.vstack([X[tets].mean(1), _random_interior_points(X, tets, rng)[0]])

    ours = nd.eval_field(X, tets, edges, u, pts)
    ref = _skfem_eval(X, tets, edges, u, pts)
    np.testing.assert_allclose(ours, ref, atol=1e-6, rtol=1e-6)

    ours_centroid = nd.cell_field(X, tets, edges, u)
    ref_centroid = _skfem_eval(X, tets, edges, u, X[tets].mean(1))
    np.testing.assert_allclose(ours_centroid, ref_centroid, atol=1e-6, rtol=1e-6)


def test_dof_lookup_is_order_independent(mesh, rng):
    """tet_dofs must not assume `edges` is sorted (mesh.edges from skfem isn't)."""
    X, tets, edges = mesh
    u = rng.normal(size=len(edges))
    perm = rng.permutation(len(edges))
    edges2, u2 = edges[perm], u[perm]
    assert not np.array_equal(edges2, edges)

    pts = X[tets].mean(1)[:10]
    H1 = nd.eval_field(X, tets, edges, u, pts)
    H2 = nd.eval_field(X, tets, edges2, u2, pts)
    np.testing.assert_allclose(H1, H2, atol=1e-10)


# ───────────────────────── 3. locate() ─────────────────────────────────────

def test_locate_centroids_and_outside(mesh):
    X, tets, edges = mesh
    cent = X[tets].mean(1)
    tet_id, bary = nd.locate(X, tets, cent)
    np.testing.assert_array_equal(tet_id, np.arange(len(tets)))
    np.testing.assert_allclose(bary, 0.25, atol=1e-8)
    np.testing.assert_allclose(bary.sum(-1), 1.0, atol=1e-8)

    far = np.array([[10.0, 10.0, 10.0], [-5.0, 0.0, 0.0]])
    tet_id_out, bary_out = nd.locate(X, tets, far)
    np.testing.assert_array_equal(tet_id_out, -1)
    assert np.isnan(bary_out).all()


def test_locate_bary_sums_to_one_for_interior_points(mesh, rng):
    X, tets, edges = mesh
    pts, _, _ = _random_interior_points(X, tets, rng)
    tet_id, bary = nd.locate(X, tets, pts)
    assert (tet_id >= 0).all()
    np.testing.assert_allclose(bary.sum(-1), 1.0, atol=1e-8)
    assert (bary >= -1e-9).all()


# ───────────────────────── 4. plane_sample() ───────────────────────────────

def test_plane_sample_shapes_and_keys(mesh, rng):
    X, tets, edges = mesh
    K = 3
    u = rng.normal(size=(len(edges), K))
    res = 21
    s = nd.plane_sample(X, tets, edges, u, axis="y", offset=0.0, res=res, pad=0.0)

    assert set(s) == {"axis", "offset", "u_label", "v_label", "U", "V", "inside", "H", "in_plane"}
    assert s["axis"] == "y" and s["u_label"] == "x" and s["v_label"] == "z"
    assert s["in_plane"] == (0, 2)
    assert s["U"].shape == s["V"].shape == (res, res)
    assert s["inside"].shape == (res, res)
    assert s["H"].shape == (res, res, 3, K)
    assert np.isnan(s["H"][~s["inside"]]).all()
    assert not np.isnan(s["H"][s["inside"]]).any()
    # box cut through the middle, no padding: essentially the whole grid is inside
    assert s["inside"].mean() > 0.98


def test_plane_sample_outside_padding_has_nan(mesh, rng):
    X, tets, edges = mesh
    u = rng.normal(size=len(edges))
    s = nd.plane_sample(X, tets, edges, u, axis="z", offset=0.0, res=25, pad=0.25)
    assert not s["inside"].all()
    assert np.isnan(s["H"][~s["inside"]]).all()


# ───────────────────────── 5. 1-D vs 2-D u consistency ─────────────────────

def test_1d_vs_2d_u_consistency(mesh, rng):
    X, tets, edges = mesh
    u1 = rng.normal(size=len(edges))
    u2 = np.stack([u1, 2.0 * u1, -u1], axis=1)          # [Ne,3]
    pts, _, _ = _random_interior_points(X, tets, rng, n=10)

    Hc1, Hc2 = nd.cell_field(X, tets, edges, u1), nd.cell_field(X, tets, edges, u2)
    assert Hc1.shape == (len(tets), 3)
    assert Hc2.shape == (len(tets), 3, 3)
    np.testing.assert_allclose(Hc2[:, :, 0], Hc1, atol=1e-10)
    np.testing.assert_allclose(Hc2[:, :, 1], 2.0 * Hc1, atol=1e-10)
    np.testing.assert_allclose(Hc2[:, :, 2], -Hc1, atol=1e-10)

    He1, He2 = nd.eval_field(X, tets, edges, u1, pts), nd.eval_field(X, tets, edges, u2, pts)
    assert He1.shape == (len(pts), 3)
    assert He2.shape == (len(pts), 3, 3)
    np.testing.assert_allclose(He2[:, :, 0], He1, atol=1e-10)
    np.testing.assert_allclose(He2[:, :, 1], 2.0 * He1, atol=1e-10)
    np.testing.assert_allclose(He2[:, :, 2], -He1, atol=1e-10)

    s1 = nd.plane_sample(X, tets, edges, u1, axis="x", res=15)
    s2 = nd.plane_sample(X, tets, edges, u2, axis="x", res=15)
    assert s1["H"].shape == (15, 15, 3)
    assert s2["H"].shape == (15, 15, 3, 3)
    np.testing.assert_array_equal(s1["inside"], s2["inside"])
    np.testing.assert_allclose(s2["H"][..., 0], s1["H"], equal_nan=True, atol=1e-10)


# ───────────────────────── 6. nodal smoothing and curl (display) ──────────

def _rotational_dofs(X, edges, a, b):
    """DOFs of H = a + b×x (inside N0: exact line integrals via the edge midpoint)."""
    xa, xb = X[edges[:, 0]], X[edges[:, 1]]
    H_mid = a + np.cross(b, 0.5 * (xa + xb))
    return (H_mid * (xb - xa)).sum(-1)


def test_vertex_field_and_curl_exact_in_n0(mesh, rng):
    X, tets, edges = mesh
    a, b = rng.normal(size=3), rng.normal(size=3)
    u = _rotational_dofs(X, edges, a, b)
    Hv = nd.vertex_field(X, tets, edges, u)                  # continuous field → exact at vertices
    np.testing.assert_allclose(Hv, a + np.cross(b, X), atol=1e-8)
    Cv = nd.vertex_field(X, tets, edges, np.stack([u, 2 * u], 1), curl=True)
    assert Cv.shape == (len(X), 3, 2)
    np.testing.assert_allclose(Cv[..., 0], np.broadcast_to(2 * b, (len(X), 3)), atol=1e-8)
    np.testing.assert_allclose(Cv[..., 1], 2 * Cv[..., 0], atol=1e-8)


def test_smooth_plane_sample_matches_raw_for_n0_field(mesh, rng):
    X, tets, edges = mesh
    u = _rotational_dofs(X, edges, rng.normal(size=3), rng.normal(size=3))
    raw = nd.plane_sample(X, tets, edges, u, axis="z", offset=0.3, res=21)
    smo = nd.plane_sample(X, tets, edges, u, axis="z", offset=0.3, res=21, smooth=True)
    np.testing.assert_array_equal(raw["inside"], smo["inside"])
    np.testing.assert_allclose(smo["H"][smo["inside"]], raw["H"][raw["inside"]], atol=1e-8)
