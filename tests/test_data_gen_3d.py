"""Tests for the 3D Maxwell (H-field, Nédélec N0) generator src/data_gen/dataset_generator_3d.py.

Invariants:
- calibration box / pillbox frequencies match the analytic PEC spectra within the N0
  discretisation error at that mesh size (measured, --mesh_size 0.12, K = 8:
  box max |f/f_exact − 1| = 2.8e-3, pillbox 7.5e-3; 0.09: 1.2e-3 / 3.9e-3)
- no zero / gradient modes (divergence residual ~1e-14, λ_min ≫ 0)
- handles are rejected (torus: b1 = 1 → a harmonic λ = 0 field survives the projection)
- edge DOF orientation low → high vertex index, DOF = line integral of H
- per-sample seeding is reproducible
"""
import json

import h5py
import numpy as np
import pytest

pytest.importorskip("skfem")
pytestmark = pytest.mark.gmsh

import src.data_gen.dataset_generator_3d as gen  # noqa: E402


@pytest.fixture
def set_args():
    old = gen.ARGS

    def _set(*argv):
        gen.ARGS = gen.parse_args(list(argv))
        return gen.ARGS

    yield _set
    gen.ARGS = old


def test_eigenvalues_to_ghz():
    f = np.array([1.3, 2.5])
    k = 2 * np.pi * f * 1e9 / gen.C0
    np.testing.assert_allclose(gen.eigenvalues_to_ghz(k ** 2), f, rtol=1e-12)


def test_analytic_spectra():
    # pillbox TM010 at k = 2.405/R; cube: lowest k² = 2(π/a)² with multiplicity 3
    R, L = 0.04, 0.05
    k2, lab = gen.pillbox_spectrum(R, L)[0]
    assert lab == "TM010" and np.isclose(np.sqrt(k2) * R, 2.404825557695773)
    cube = gen.box_spectrum(1.0, 1.0, 1.0)
    np.testing.assert_allclose(cube[:3], 2 * np.pi ** 2)
    assert cube[3] > cube[2] * 1.2


@pytest.mark.parametrize("s_id,shape,tol", [(0, "calib_box", 5e-3), (1, "calib_pillbox", 1e-2)])
def test_calibration_matches_analytic(set_args, s_id, shape, tol):
    set_args("--mode", "calibration", "--n_eigen_modes", "8", "--mesh_size", "0.12")
    res = gen.generate_sample_data(s_id)
    assert res is not None and res["shape_type"] == shape
    ref = gen.eigenvalues_to_ghz(gen.analytic_k2(shape, res["geom_params"], 8))
    rel = res["freqs"] / ref - 1
    assert np.abs(rel).max() < tol, rel
    if shape == "calib_pillbox":   # TM010 at c·2.405/(2πR)
        assert abs(res["freqs"][0] / (gen.C0 * 2.404825557695773 / (2 * np.pi * 0.04) / 1e9) - 1) < tol


def test_no_zero_or_gradient_modes(set_args):
    set_args("--mode", "calibration", "--n_eigen_modes", "6", "--mesh_size", "0.15")
    res = gen.generate_sample_data(1)
    A = gen.assemble_h_n0(res["nodes"], res["tets"])
    U = res["h_edges"]
    lam = np.einsum("ij,ij->j", U, A["K"] @ U) / np.einsum("ij,ij->j", U, A["M"] @ U)
    np.testing.assert_allclose(gen.eigenvalues_to_ghz(lam), res["freqs"], rtol=1e-8)
    assert lam.min() > 0.5 * (2.405 / 0.04) ** 2               # far from the λ = 0 kernel
    MU = A["M"] @ U
    assert np.abs(A["G"].T @ MU).max() / np.abs(MU).max() < 1e-10  # ⟂_M all gradients (incl. boundary)
    np.testing.assert_allclose(U.T @ MU, np.eye(U.shape[1]), atol=1e-8)
    assert np.all(np.diff(res["freqs"]) >= 0) and res["freq_next"] >= res["freqs"][-1]


def test_kernel_is_gradients():
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("k")
        gmsh.model.occ.addBox(0, 0, 0, 1.0, 0.8, 0.6)
        gmsh.model.occ.synchronize()
        nodes, tets = gen._mesh_current_model(0.3)
    finally:
        gmsh.finalize()
    A = gen.assemble_h_n0(nodes, tets)
    assert abs(A["K"] @ A["G"]).max() < 1e-10 * abs(A["K"]).max()
    # DOF = line integral: the constant field e_x has DOFs x_head − x_tail = G·x
    e = A["edges"]
    assert np.all(e[:, 0] < e[:, 1])
    u = nodes[e[:, 1], 0] - nodes[e[:, 0], 0]
    np.testing.assert_allclose(A["G"] @ nodes[:, 0], u)
    assert np.isclose(u @ (A["M"] @ u), 1.0 * 0.8 * 0.6, rtol=1e-10)  # ∫|e_x|² = |Ω|


def test_handles_rejected():
    """A torus (b1 = 1) fails the topology certificate, and its harmonic field would be a
    spurious λ = 0 mode that the gradient projection does not remove."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("torus")
        gmsh.model.occ.addTorus(0, 0, 0, 0.05, 0.02)
        gmsh.model.occ.synchronize()
        nodes, tets = gen._mesh_current_model(0.012)
    finally:
        gmsh.finalize()
    topo = gen.mesh_topology(tets, len(nodes))
    assert topo["euler"] == 0 and topo["euler_boundary"] == 0 and not gen.is_topological_ball(topo)
    with pytest.raises(RuntimeError, match="zero"):
        gen.solve_h_modes(gen.assemble_h_n0(nodes, tets), 3)


@pytest.mark.parametrize("family", list(gen.FAMILIES))
def test_families_are_topological_balls(set_args, family):
    set_args("--families", family, "--mesh_size", "0.2", "--n_eigen_modes", "4")
    for s_id in range(2):
        res = gen.generate_sample_data(s_id)
        assert res is not None and res["shape_type"] == family
        assert gen.is_topological_ball(gen.mesh_topology(res["tets"], len(res["nodes"])))
        assert np.all(res["freqs"] > 0.5) and np.all(np.isfinite(res["h_edges"]))


def test_seeding_reproducible(set_args):
    set_args("--mesh_size", "0.2", "--n_eigen_modes", "3", "--seed", "7")
    a, b = gen.generate_sample_data(3), gen.generate_sample_data(3)
    assert a["shape_type"] == b["shape_type"] and a["geom_params"] == b["geom_params"]
    np.testing.assert_array_equal(a["nodes"], b["nodes"])
    np.testing.assert_array_equal(a["freqs"], b["freqs"])
    set_args("--mesh_size", "0.2", "--n_eigen_modes", "3", "--seed", "8")
    c = gen.generate_sample_data(3)
    assert c["geom_params"] != a["geom_params"]


def test_cli_writes_h5(tmp_path):
    out = tmp_path / "d3.h5"
    gen.main(["--n_total", "2", "--n_workers", "2", "--mesh_size", "0.2", "--n_eigen_modes", "3",
              "--h5_filename", str(out), "--families", "pillbox"])
    assert out.exists() and not (tmp_path / "d3.h5.partial").exists()
    with h5py.File(out, "r") as f:
        meta = json.loads(f.attrs["metadata"])
        assert meta["field"] == "H" and "edges[i,0] < edges[i,1]" in meta["dof_convention"]
        assert sorted(f.keys()) == ["sample_0000", "sample_0001"]
        g = f["sample_0000"]
        ne, nv = g["edges"].shape[0], g["nodes"].shape[0]
        assert g["nodes"].shape == (nv, 3) and g["tets"].shape[1] == 4
        assert g["h_edges"].shape == (ne, 3) and g["freqs"].shape == (3,)
        assert g.attrs["shape_type"] == "pillbox" and "R" in json.loads(g.attrs["geom_params"])
        assert np.all(g["edges"][:, 0] < g["edges"][:, 1])
