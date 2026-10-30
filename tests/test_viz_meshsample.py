"""src/viz/meshsample.py: point location in a linear tet mesh, the regular-grid locator, plane grids and barycentric
interpolation of vertex values.  The mesh is a jittered Kuhn-split box built here (no meshing dependency)."""
import itertools

import numpy as np
import pytest

import src.viz.meshsample as ms


def _box_mesh(dims=(1.0, 0.8, 0.6), n=4, jitter=0.15, seed=3):
    """(X [Nv,3], tets [Nt,4]): n^3 cubes, 6 tets each (Kuhn), interior vertices jittered by `jitter` cell sizes."""
    ax = [np.linspace(0.0, d, n + 1) for d in dims]
    X = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    idx = np.arange(len(X)).reshape(n + 1, n + 1, n + 1)
    h = np.array(dims) / n
    rng = np.random.default_rng(seed)
    interior = ((X > 1e-12) & (X < np.array(dims) - 1e-12)).all(1)
    X = X + jitter * h * rng.uniform(-1, 1, X.shape) * interior[:, None]
    tets = []
    for i, j, k in itertools.product(range(n), repeat=3):
        for perm in itertools.permutations(range(3)):        # path from corner 000 to 111 through the cube
            c, path = [0, 0, 0], [(0, 0, 0)]
            for a in perm:
                c[a] = 1
                path.append(tuple(c))
            tets.append([idx[i + p[0], j + p[1], k + p[2]] for p in path])
    return X, np.array(tets, dtype=np.int64)


@pytest.fixture(scope="module")
def mesh():
    return _box_mesh()


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


def _random_interior_points(X, tets, rng, n=25):
    """Points strictly inside random tets (positive Dirichlet barycentric weights)."""
    idx = rng.integers(0, len(tets), size=n)
    bw = rng.dirichlet(np.ones(4), size=n)
    return np.einsum("pv,pvc->pc", bw, X[tets[idx]]), idx, bw


def test_mesh_fills_the_box(mesh):
    from src.data.mesh_features import tet_geometry
    vol, _ = tet_geometry(*mesh)
    assert np.isclose(vol.sum(), 1.0 * 0.8 * 0.6, rtol=1e-12) and (vol > 0).all()


def test_locate_centroids_and_outside(mesh):
    X, tets = mesh
    tet_id, bary = ms.locate(X, tets, X[tets].mean(1))
    np.testing.assert_array_equal(tet_id, np.arange(len(tets)))
    np.testing.assert_allclose(bary, 0.25, atol=1e-8)
    np.testing.assert_allclose(bary.sum(-1), 1.0, atol=1e-8)

    far = np.array([[10.0, 10.0, 10.0], [-5.0, 0.0, 0.0]])
    tet_id_out, bary_out = ms.locate(X, tets, far)
    np.testing.assert_array_equal(tet_id_out, -1)
    assert np.isnan(bary_out).all()


def test_locate_bary_sums_to_one_for_interior_points(mesh, rng):
    X, tets = mesh
    pts, _, bw = _random_interior_points(X, tets, rng)
    tet_id, bary = ms.locate(X, tets, pts)
    assert (tet_id >= 0).all()
    np.testing.assert_allclose(bary.sum(-1), 1.0, atol=1e-8)
    assert (bary >= -1e-9).all()
    rec = np.einsum("pi,pij->pj", bary, X[tets[tet_id]])      # barycentric -> the point itself
    np.testing.assert_allclose(rec, pts, atol=1e-12)


def test_locate_grid_matches_locate_on_volumes_and_planes(mesh):
    """The grid locator (each tet scans the grid points of its box) agrees with the k-d tree locate on a
    volume grid and on a plane (one dimension of 1, anisotropic spacing); points outside stay -1."""
    X, tets = mesh
    lo, hi = X.min(0) - 0.05, X.max(0) + 0.05                   # a frame of grid points outside the box
    for d, dims in ((0.07, None), (np.array([0.05, 1.0, 0.04]), (23, 1, 18))):
        dims = np.asarray(dims) if dims is not None else np.ceil((hi - lo) / d).astype(int) + 1
        lo_g = lo.copy()
        if dims[1] == 1:
            lo_g[1] = 0.37                                      # the plane y = 0.37
        ax = [lo_g[i] + np.broadcast_to(d, (3,))[i] * np.arange(dims[i]) for i in range(3)]
        zz, yy, xx = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")
        pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1)   # x fastest
        t_ref, b_ref = ms.locate(X, tets, pts)
        t, b = ms.locate_grid(X, tets, lo_g, d, dims)
        assert t.shape == (len(pts),) and ((t >= 0) == (t_ref >= 0)).mean() > 0.995
        assert (t >= 0).any() and (t < 0).any()
        both = (t >= 0) & (t_ref >= 0)
        rec = np.einsum("pi,pij->pj", b[both], X[tets[t[both]]])
        assert np.allclose(rec, pts[both], atol=1e-9)
        assert np.allclose(np.einsum("pi,pij->pj", b_ref[both], X[tets[t_ref[both]]]), rec, atol=1e-9)


def test_plane_grid_shapes_and_padding(mesh):
    X, _ = mesh
    pts, U, V, (iu, iv) = ms.plane_grid(X, axis="y", offset=0.37, res=31, pad=0.0)
    assert pts.shape == (31 * 31, 3) and U.shape == V.shape == (31, 31) and (iu, iv) == (0, 2)
    assert np.all(pts[:, 1] == 0.37)
    assert np.isclose(U.min(), X[:, 0].min()) and np.isclose(U.max(), X[:, 0].max())
    assert np.isclose(V.min(), X[:, 2].min()) and np.isclose(V.max(), X[:, 2].max())
    pts_p, *_ = ms.plane_grid(X, axis="y", offset=0.37, res=31, pad=0.1)
    assert pts_p[:, 0].min() < X[:, 0].min() and pts_p[:, 2].max() > X[:, 2].max()


def test_interp_vertex_reproduces_linear_fields_and_is_nan_outside(mesh, rng):
    """P1 interpolation of nodal values of a linear function is exact; outside the mesh it is NaN."""
    X, tets = mesh
    a, b = np.array([0.3, -1.2, 0.7]), 0.4
    nodal = np.column_stack([X @ a + b, 2 * X[:, 0] - X[:, 2]])       # [Nv, 2]
    pts, _, _ = _random_interior_points(X, tets, rng, n=40)
    pts = np.vstack([pts, [[9.0, 9.0, 9.0]]])
    tid, bary = ms.locate(X, tets, pts)
    out = ms.interp_vertex(tets, tid, bary, nodal)
    assert out.shape == (41, 2) and np.isnan(out[-1]).all()
    np.testing.assert_allclose(out[:-1, 0], pts[:-1] @ a + b, atol=1e-12)
    np.testing.assert_allclose(out[:-1, 1], 2 * pts[:-1, 0] - pts[:-1, 2], atol=1e-12)
