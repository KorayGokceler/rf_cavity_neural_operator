"""Tests for the FEM data generator (src/data_gen/dataset_generator.py) and the
analytic calibration reference (validate_data.py).

Kritik invariantlar:
- Calibration shapes (square / circle / annulus) match analytic TM frequencies
- Eigenpairs ascending, M-orthonormal (also inside degenerate pairs)
- Shift-invert sigma inside the spectrum does not skip the fundamental mode
- Per-sample seeding is reproducible, --seed changes the geometry
"""
import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")
pytest.importorskip("skfem")

from skfem import MeshTri, Basis, ElementTriP2
from skfem.models.poisson import mass

import src.data_gen.dataset_generator as gen
import validate_data

# Coarser than production mesh → each sample < 1 s
FAST_MESH = ["--mesh_size_min", "0.003", "--mesh_size_max", "0.008"]


@pytest.fixture
def set_args():
    old = gen.ARGS

    def _set(*argv):
        gen.ARGS = gen.parse_args(list(argv))
        return gen.ARGS

    yield _set
    gen.ARGS = old


def test_eigenvalues_to_ghz():
    f_ghz = np.array([1.0, 2.5])
    k = 2 * np.pi * f_ghz * 1e9 / gen.C0
    np.testing.assert_allclose(gen.eigenvalues_to_ghz(k ** 2), f_ghz, rtol=1e-12)


@pytest.mark.parametrize("s_id,shape", [(0, "square"), (1, "circle"), (2, "annulus")])
def test_calibration_matches_analytic(set_args, s_id, shape):
    set_args("--mode", "calibration", "--n_eigen_modes", "4", *FAST_MESH)
    res = gen.generate_sample_data(s_id)
    assert res is not None and res["shape_type"] == shape
    ref = validate_data.analytical_spectrum(shape, 4, res["geom_params"])
    # P2 on a (coarse) boundary-fitted mesh: well below 0.1 % in practice
    np.testing.assert_allclose(res["freqs"], ref, rtol=2e-3)


def test_eigenpairs_sorted_and_m_orthonormal(set_args):
    set_args("--mode", "calibration", "--n_eigen_modes", "3", *FAST_MESH)
    res = gen.generate_sample_data(1)  # circle: modes 1/2 are a degenerate pair
    assert np.all(np.diff(res["freqs"]) >= 0)
    assert np.isrealobj(res["vecs"])
    basis = Basis(MeshTri(res["nodes"].T, res["elements"].T), ElementTriP2())
    assert res["vecs"].shape[0] == basis.N and basis.N > res["n_nodes"]
    gram = res["vecs"].T @ mass.assemble(basis) @ res["vecs"]
    np.testing.assert_allclose(gram, np.eye(3), atol=1e-6)


def test_sigma_inside_spectrum_does_not_skip_fundamental():
    a = 0.08
    m = MeshTri.init_tensor(np.linspace(0, a, 13), np.linspace(0, a, 13))
    lam11, lam12 = 2 * (np.pi / a) ** 2, 5 * (np.pi / a) ** 2   # ~3084, ~7711
    # sigma closer to the degenerate lam12 pair than to lam11: plain shift-invert
    # with k=2 would return (lam12, lam12) and silently drop the fundamental.
    vals, vecs = gen.solve_dirichlet_eigenmodes(m.p.T, m.t.T, k=2, sigma=6000.0)
    np.testing.assert_allclose(vals, [lam11, lam12], rtol=1e-2)
    assert vecs.shape[1] == 2


def test_random_seed_reproducible(set_args):
    set_args("--mode", "random", *FAST_MESH)
    r1, r2 = gen.generate_sample_data(3), gen.generate_sample_data(3)
    np.testing.assert_array_equal(r1["nodes"], r2["nodes"])
    # ARPACK uses a random start vector → freqs agree to round-off, not bitwise
    np.testing.assert_allclose(r1["freqs"], r2["freqs"], rtol=1e-10)
    set_args("--mode", "random", "--seed", "1", *FAST_MESH)
    r3 = gen.generate_sample_data(3)
    assert r3["nodes"].shape != r1["nodes"].shape or not np.allclose(r3["nodes"], r1["nodes"])


def test_mesh_is_consistent(set_args):
    set_args("--mode", "calibration", *FAST_MESH)
    res = gen.generate_sample_data(2)  # annulus (OCC boolean cut)
    n = len(res["nodes"])
    assert res["elements"].min() == 0 and res["elements"].max() == n - 1
    assert len(np.unique(res["elements"])) == n  # no orphan nodes


def test_save_sample_plot_fewer_than_three_modes(set_args, tmp_path):
    set_args("--mode", "calibration", "--n_eigen_modes", "2", *FAST_MESH)
    res = gen.generate_sample_data(0)
    out = tmp_path / "p.png"
    gen.save_sample_plot(res, str(out))  # used to IndexError for K < 3
    assert out.exists()


def test_analytic_reference_values():
    # Independent literature values: j01 = 2.404826, j11 = 3.831706
    c = gen.C0
    np.testing.assert_allclose(
        validate_data.analytical_spectrum("circle", 3, {"radius": 0.04}),
        np.array([2.404826, 3.831706, 3.831706]) * c / (2 * np.pi * 0.04) / 1e9, rtol=1e-6)
    np.testing.assert_allclose(
        validate_data.analytical_spectrum("square", 3, {"side": 0.08}),
        np.array([np.sqrt(2), np.sqrt(5), np.sqrt(5)]) * c / (2 * 0.08) / 1e9, rtol=1e-12)
    # Thin annulus → k ≈ pi / (b - a) for the n = 0 mode
    f_thin = validate_data.analytical_spectrum("annulus", 1, {"r_inner": 0.100, "r_outer": 0.101})[0]
    np.testing.assert_allclose(f_thin, c / (2 * 0.001) / 1e9, rtol=1e-3)


def test_random_holes_multiply_connected(set_args):
    """--hole_prob 1: every random geometry has 1..max_holes holes → extra
    boundary loops, valid mesh, ascending eigenpairs; hole_prob 0 is legacy."""
    from src.data.dataset_converter import RFCavityToGNOT
    set_args("--mode", "random", "--hole_prob", "1.0", "--max_holes", "2", *FAST_MESH)
    for s_id in range(3):
        res = gen.generate_sample_data(s_id)
        assert "_hole" in res["shape_type"]
        n_holes = int(res["shape_type"].split("_hole")[1])
        loops = RFCavityToGNOT._boundary_loops(None, res["nodes"], res["elements"])
        assert len(loops) == 1 + n_holes
        assert np.all(np.diff(res["freqs"]) >= 0)
    set_args("--mode", "random", *FAST_MESH)
    legacy = gen.generate_sample_data(0)
    assert "_hole" not in legacy["shape_type"]
