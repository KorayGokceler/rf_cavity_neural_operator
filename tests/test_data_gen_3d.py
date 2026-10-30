"""Tests for the 3D PEC cavity generator src/data_gen/dataset_generator_3d.py (geometry families, gmsh mesh,
topology certificate, field-label CLI) and the Whitney N0 reference eigensolve of src/qoi/analytic.fe_modes.

Invariants:
- calibration box / pillbox: the N0 reference frequencies match the analytic PEC spectra within the
  discretisation error at that mesh size (measured, --mesh_size 0.12, K = 8, max |f/f_exact - 1|; E converges
  from below)
- no zero / gradient modes (divergence residual ~1e-14, lambda_min >> 0); wall-edge DOFs exactly 0
- handles (b1 > 0) add nothing to the E kernel: torus, half-wave coax TEM at c/2L
- isolated conductor (b2 = 1): without the boundary-component potential a spurious lambda = 0
  mode, with it none
- per-sample seeding is reproducible; the CLI writes the field-label H5 layout
"""
import json
import os

import h5py
import numpy as np
import pytest

pytest.importorskip("skfem")
pytestmark = pytest.mark.gmsh

import src.data_gen.dataset_generator_3d as gen  # noqa: E402
from src.qoi.analytic import fe_modes  # noqa: E402

CHEAP = ["--no_adapt", "--label_order", "2", "--model_order", "1"]      # one p2 solve, p1 model space: seconds
SHARP = ["--no_round_edges", "--no_heal", "--min_fillet", "0"]          # mesh-only tests: no OCC filleting (composite: minutes)


@pytest.fixture
def set_args():
    from src.data_gen import cavity_shapes as cs
    old, old_fillet = gen.ARGS, cs.MIN_FILLET          # mesh_sample sets the module-level fillet floor

    def _set(*argv):
        gen.ARGS = gen.parse_args(list(argv))
        return gen.ARGS

    yield _set
    gen.ARGS, cs.MIN_FILLET = old, old_fillet


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


@pytest.mark.parametrize("s_id,shape,tol", [(0, "calib_box", 1e-2), (1, "calib_pillbox", 1e-2)])
def test_calibration_matches_analytic(set_args, s_id, shape, tol):
    """The N0 reference solve on the generator's calibration mesh vs the closed-form PEC spectrum."""
    set_args("--mode", "calibration", "--mesh_size", "0.12")
    g = gen.mesh_sample(s_id)
    assert g["shape_type"] == shape
    _, _, f = fe_modes(g["nodes"], g["tets"], 8)
    ref = gen.eigenvalues_to_ghz(gen.analytic_k2(shape, g["params"], 8))
    rel = f / 1e9 / ref - 1
    assert np.abs(rel).max() < tol, rel
    if shape == "calib_pillbox":   # TM010 at c·2.405/(2πR)
        assert abs(f[0] / (gen.C0 * 2.404825557695773 / (2 * np.pi * 0.04)) - 1) < tol


@pytest.mark.parametrize("s_id,shape", [(0, "calib_box"), (1, "calib_pillbox")])
def test_calibration_field_labels_match_analytic(set_args, s_id, shape):
    """The generator's own labels (curved mesh, one p2 solve) on the calibration shapes: stored analytic
    reference, frequencies within the p2 discretisation error (measured 5e-4 box, 7e-4 pillbox)."""
    set_args("--mode", "calibration", "--mesh_size", "0.2", "--maxh_factor", "1", "--n_eigen_modes", "4", *CHEAP)
    res = gen.generate_sample_data(s_id)
    assert res is not None and res["shape_type"] == shape and res["labels"] == "field"
    ref = gen.eigenvalues_to_ghz(gen.analytic_k2(shape, res["geom_params"], 4))
    assert np.abs(res["freqs"] / ref - 1).max() < 3e-3, res["freqs"] / ref - 1
    assert res["freq_next"] >= res["freqs"][-1] and np.all(np.diff(res["freqs"]) >= 0)


def test_no_zero_or_gradient_modes(set_args):
    set_args("--mode", "calibration", "--mesh_size", "0.15")
    g = gen.mesh_sample(1)
    geom, Y, f = fe_modes(g["nodes"], g["tets"], 6)
    fe, M = geom["fe"], geom["M"]
    K, G, n_pot, bnd, topo = fe["K"], fe["G"], fe["n_pot"], fe["bnd_edge"], fe["topo"]
    assert bnd.sum() > 0 and np.all(Y[bnd] == 0)                           # n × E = 0: wall DOFs exactly 0
    n_int = int((topo["vert_comp"] < 0).sum())
    assert n_pot == n_int and topo["n_comp"] == 1 and abs(G[:, n_pot:]).sum() == 0
    lam = np.einsum("ij,ij->j", Y, K @ Y) / np.einsum("ij,ij->j", Y, M @ Y)
    np.testing.assert_allclose(gen.C0 * np.sqrt(lam) / (2 * np.pi * geom["scale"]), f, rtol=1e-8)
    assert f[0] > np.sqrt(0.5) * gen.C0 * 2.405 / (2 * np.pi * 0.04)       # no λ ≈ 0 mode: TM010 is the lowest
    MY = M @ Y
    assert np.abs(G.T @ MY).max() / np.abs(MY).max() < 1e-10               # ⟂_M the E-kernel gradients
    np.testing.assert_allclose(Y.T @ MY, np.eye(Y.shape[1]), atol=1e-8)
    assert np.all(np.diff(f) >= 0)


def _occ_mesh(build, h):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("m")
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        return gen._mesh_current_model(h)
    finally:
        gmsh.finalize()


def test_handles_are_clean():
    """Torus (b1 = 1): the handle adds nothing to the E kernel, no spurious λ = 0 mode (measured
    f1 4.537 GHz at h = 12 mm, 4.386 at 5 mm)."""
    nodes, tets = _occ_mesh(lambda occ: occ.addTorus(0, 0, 0, 0.05, 0.02), 0.012)
    topo = gen.mesh_topology(tets, len(nodes))
    assert topo["euler"] == 0 and topo["euler_boundary"] == 0
    assert gen.is_valid_e_domain(topo) and topo["betti1"] == 1 and topo["n_bnd_components"] == 1
    gen.certify(topo)
    geom, _, f = fe_modes(nodes, tets, 3)
    assert geom["fe"]["topo"]["betti1"] == 1 and 4.3 < f[0] / 1e9 < 4.7


def test_half_wave_coax_tem():
    """Coax Ro = 50, ri = 15, L = 150 mm, inner conductor touching both end plates (b1 = 1), h = 8 mm:
    the TEM λ/2 mode at c/2L (measured −0.62 %; TE11-like pair ≈ 1.80 GHz; TEM λ at c/L −0.70 %)."""
    Ro, ri, L = 0.05, 0.015, 0.15

    def build(occ):
        occ.cut([(3, occ.addCylinder(0, 0, 0, 0, 0, L, Ro))], [(3, occ.addCylinder(0, 0, -0.01, 0, 0, L + 0.02, ri))])

    nodes, tets = _occ_mesh(build, 0.008)
    topo = gen.mesh_topology(tets, len(nodes))
    assert topo["betti1"] == 1 and gen.is_valid_e_domain(topo)
    _, _, f = fe_modes(nodes, tets, 4)
    f = f / 1e9
    assert abs(f[0] / (gen.C0 / (2 * L) / 1e9) - 1) < 1e-2 and abs(f[3] / (gen.C0 / L / 1e9) - 1) < 1e-2
    assert 1.7 < f[1] < f[2] < 1.9 and abs(f[2] / f[1] - 1) < 2e-3        # TE11-like degenerate pair


def test_isolated_conductor_needs_component_potential():
    """Sphere (r = 15 mm) floating inside a PEC box (b2 = 1, two boundary shells): the field ∇φ
    with φ = 1 on the sphere, 0 on the box is curl-free with zero wall trace but not a gradient of
    an interior potential — without the component potential it survives as λ = 0; with it the
    spectrum is clean (measured f1 2.028 GHz at h = 8 mm, 2.049 at 5 mm; the removed H formulation, valid
    here since b1 = 0, gave 2.123 / 2.090: both bracket the limit)."""
    from src.data.dataset_converter_3d import discrete_gradient, e_potential_matrix

    def build(occ):
        occ.cut([(3, occ.addBox(0, 0, 0, 0.1, 0.08, 0.06))], [(3, occ.addSphere(0.05, 0.04, 0.03, 0.015))])

    nodes, tets = _occ_mesh(build, 0.01)
    topo = gen.mesh_topology(tets, len(nodes))
    assert topo["n_bnd_components"] == 2 and topo["betti2"] == 1 and topo["betti1"] == 0
    assert gen.is_valid_e_domain(topo)
    with pytest.raises(RuntimeError, match="zero"):
        fe_modes(nodes, tets, 3, component_potentials=False)
    geom, _, f = fe_modes(nodes, tets, 3)
    assert geom["fe"]["topo"]["n_comp"] == 2 and 1.9 < f[0] / 1e9 < 2.12
    G, bnd, n_pot, bt = e_potential_matrix(geom["edges"], len(nodes), tets)
    assert n_pot == int((bt["vert_comp"] < 0).sum()) + 1
    sphere = bt["vert_comp"] == 1                                            # component 1 = the sphere (fewer vertices)
    assert np.allclose(np.linalg.norm(nodes[sphere] - [0.05, 0.04, 0.03], axis=1), 0.015, rtol=1e-3)
    np.testing.assert_array_equal(G[:, n_pot - 1].toarray().ravel(),
                                  discrete_gradient(geom["edges"], len(nodes)) @ sphere.astype(float))


def _expected_betti1(family, p):
    """First Betti number from the builder parameters (handle families, cavity_shapes)."""
    if family == "hwr":
        return 1 + int(p["beam_port"])                  # coax loop (+ the port through the inner conductor)
    if family == "spoke":
        return int(p["n_spokes"]) * (1 + int(p["bore"]))  # loop around each spoke (+ through its bore)
    if family == "dtl":
        return int(p["n_tubes"])                        # loop through each drift-tube bore
    return 0


@pytest.mark.parametrize("family", list(gen.FAMILIES))
def test_families_valid(set_args, family):
    """Every training family (the discrete freeform one included) passes the certificate (connected
    manifold); handle families have the b1 of their construction, the others b1 = 0."""
    set_args("--families", family, "--mesh_size", "0.2", *SHARP)
    for s_id in range(2):
        g = gen.mesh_sample(s_id)
        assert g["shape_type"] == family
        topo = gen.mesh_topology(g["tets"], len(g["nodes"]))
        assert gen.is_valid_e_domain(topo) and g["topo"]["betti1"] == topo["betti1"]
        assert topo["betti1"] == _expected_betti1(family, g["params"]), g["params"]


def test_default_families(set_args):
    from src.data_gen import cavity_shapes as cs
    a = set_args()
    assert a.families == list(gen.FAMILIES) and set(cs.HANDLE_FAMILIES) <= set(a.families)
    assert set_args("--families", "freeform").families == ["freeform"]       # kept by parse_args (service display)


def test_discrete_families_are_dropped_by_main(tmp_path, capsys):
    """Field labels need a CAD solid: main() drops freeform, and refuses when nothing is left."""
    with pytest.raises(SystemExit, match="no CAD family"):
        gen.main(["--families", "freeform", "--h5_filename", str(tmp_path / "x.h5")])
    assert "dropping the discrete families ['freeform']" in capsys.readouterr().out


def test_labels_n0_was_removed(capsys):
    """--labels n0 (Whitney N0 labels) is an error; --labels field stays accepted as a deprecated no-op."""
    assert gen.parse_args(["--labels", "field"]).labels == "field"
    with pytest.raises(SystemExit):
        gen.parse_args(["--labels", "n0"])
    assert "N0 label type was removed" in capsys.readouterr().err


def test_seeding_reproducible(set_args):
    set_args("--mesh_size", "0.2", "--seed", "7", *SHARP)
    a, b = gen.mesh_sample(3), gen.mesh_sample(3)
    assert a["shape_type"] == b["shape_type"] and a["params"] == b["params"]
    np.testing.assert_array_equal(a["nodes"], b["nodes"])
    set_args("--mesh_size", "0.2", "--seed", "8", *SHARP)
    c = gen.mesh_sample(3)
    assert c["params"] != a["params"]


def test_cli_writes_h5(tmp_path, run_generator):
    families = ["pillbox", "box"]
    out = tmp_path / "d3.h5"
    run_generator("--n_total", "2", "--n_workers", "2", "--mesh_size", "0.3", "--n_eigen_modes", "3",
                  "--h5_filename", out, "--families", *families, *CHEAP)
    assert out.exists() and not (tmp_path / "d3.h5.partial").exists()
    with h5py.File(out, "r") as f:
        meta = json.loads(f.attrs["metadata"])
        assert meta["field"] == "E" and meta["labels"] == "field" and meta["dataset"] == "u_model"
        assert sorted(f.keys()) == ["sample_0000", "sample_0001"]
        for key in f:
            g = f[key]
            nd = int(g.attrs["n_dof_model"])
            assert g["u_model"].shape == (nd, 3) and g["freqs"].shape == (3,) and g["qoi"].shape[0] == 3
            assert {"vol", "brep", "rq_model", "proj_err", "axis_t", "axis_ez"} <= set(g)
            assert "nodes" not in g and "edges" not in g and "e_edges" not in g          # no N0 layout
            assert g.attrs["field"] == "E" and g.attrs["labels"] == "field" and g.attrs["shape_type"] in families
            assert g.attrs["fem_element"].startswith("HCurl p1") and np.all(np.diff(g["freqs"][()]) >= 0)
            assert int(g.attrs["betti1"]) == 0 and int(g.attrs["n_bnd_components"]) == 1


def test_cli_step_dir_writes_the_cad_of_undeformed_samples(tmp_path, run_generator):
    from netgen.occ import OCCGeometry
    out, step = tmp_path / "s.h5", tmp_path / "step"
    common = ["--n_workers", "1", "--mesh_size", "0.3", "--n_eigen_modes", "2", "--families", "pillbox",
              "--step_dir", str(step), *CHEAP]
    run_generator("--n_total", "2", "--h5_filename", out, *common)
    with h5py.File(out, "r") as f:
        for key in f:
            sid = int(key.split("_")[-1])
            shape = OCCGeometry(str(step / f"pillbox_{sid:08d}.step")).shape
            assert shape.mass == pytest.approx(f[key].attrs["volume"] * 1e9, rel=1e-6)   # STEP in mm, H5 in m
    run_generator("--n_total", "1", "--start_id", "10", "--deform_prob", "1", "--h5_filename", tmp_path / "d.h5", *common)
    assert not (step / "pillbox_00000010.step").exists()       # deformed: labels ≠ the CAD solid


def test_cli_resume_appends_missing_ids(tmp_path, run_generator):
    """--resume (cluster wall-time kills): an interrupted run's .partial keeps its samples, only
    the missing ids are generated, and the result equals an uninterrupted run; large id offsets
    (TRUBA family blocks) work."""
    common = ["--n_workers", "1", "--mesh_size", "0.3", "--n_eigen_modes", "3", "--families", "pillbox",
              "--sampling", "sobol", "--start_id", str(3 * 524288), *CHEAP]
    full, part = tmp_path / "full.h5", tmp_path / "part.h5"
    run_generator("--n_total", "3", "--h5_filename", full, *common)
    run_generator("--n_total", "1", "--h5_filename", part, *common)
    os.replace(part, str(part) + ".partial")                 # = a job killed after its first sample
    run_generator("--n_total", "3", "--h5_filename", part, "--resume", *common)
    assert part.exists() and not os.path.exists(str(part) + ".partial")
    with h5py.File(full, "r") as a, h5py.File(part, "r") as b:
        assert sorted(a.keys()) == sorted(b.keys()) == [f"sample_{3 * 524288 + i:04d}" for i in range(3)]
        assert int(b.attrs["n_requested"]) == 3 and int(b.attrs["n_failed_last_run"]) == 0
        for k in a:
            np.testing.assert_array_equal(a[k]["vol"][()], b[k]["vol"][()])
            np.testing.assert_allclose(a[k]["freqs"][()], b[k]["freqs"][()], rtol=1e-10)


def test_mesher_error_restarts_gmsh(set_args, monkeypatch):
    """A 3D mesher error can leave the gmsh session broken (every later generate(3)
    yields no tets, gmsh.clear() does not help).  Simulated here: the session stays
    broken until gmsh.finalize(); the generator must restart gmsh and succeed."""
    import gmsh
    set_args("--n_eigen_modes", "3", "--mesh_size", "0.3", "--families", "pillbox", *CHEAP)
    state = {"calls": 0, "broken": False}
    real_mesh, real_finalize = gen._mesh_current_model, gmsh.finalize

    def mesh(h):
        state["calls"] += 1
        if state["calls"] == 1:
            state["broken"] = True
            raise RuntimeError("PLC Error:  A segment and a facet intersect at point")
        if state["broken"]:
            raise RuntimeError("expected only 4-node tets, got 3D element types []")
        return real_mesh(h)

    def finalize():
        state["broken"] = False
        real_finalize()

    monkeypatch.setattr(gen, "_mesh_current_model", mesh)
    monkeypatch.setattr(gmsh, "finalize", finalize)
    res = gen.generate_sample_data(0)
    assert res is not None and state["calls"] == 2 and len(res["freqs"]) == 3


# ─────────────────────────── realistic cavities / free-form (cavity_shapes) ───

def _mesh_params(builder, params, mesh_size=0.10):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("t")
        _, prm = builder(gmsh.model.occ, None, params)
        gmsh.model.occ.synchronize()
        vol = gmsh.model.occ.getMass(3, gmsh.model.getEntities(3)[0][1])
        h0 = mesh_size * vol ** (1 / 3)
        return gen._mesh_current_model(max(min(h0, prm["_h_cap"]), 0.7 * h0))
    finally:
        gmsh.finalize()


def test_tesla_cell_frequency():
    """TESLA mid-cell dimensions (Req 103.3, Riris 35, A = B = 42, a 12, b 19, L 57.7 mm) as a
    single cell with 80 mm PEC-capped pipes: converged TM010 = 1.288 GHz (Richardson; the periodic
    π-mode of the real structure is 1.300).  E-N0: +0.15 % at mesh 0.10, +0.03 % at 0.07."""
    from src.data_gen import cavity_shapes as cs
    tesla = dict(Req=0.1033, Riris=0.035, L=0.0577, A=0.042, B=0.042, a=0.012, b=0.019, n_cells=1.0, Lpipe=0.08)
    nodes, tets = _mesh_params(cs.build_elliptical, tesla)
    err_e = fe_modes(nodes, tets, 3)[2][0] / 1e9 / 1.288 - 1
    assert abs(err_e) < 4e-3


def test_half_cell_common_tangent():
    from src.data_gen import cavity_shapes as cs
    w = cs.half_cell_wall(0.1033, 0.035, 0.0577, 0.042, 0.042, 0.012, 0.019)
    assert np.allclose(w[0], [0, 0.035]) and np.allclose(w[-1], [0.0577, 0.1033])
    assert np.all(np.diff(w[:, 0]) >= 0) and np.all(np.diff(w[:, 1]) >= 0)
    d = np.diff(w, axis=0)                                   # C1 wall: no kink at the tangent points
    ang = np.arctan2(d[:, 1], d[:, 0])
    assert np.abs(np.diff(ang)).max() < 0.2
    assert cs.half_cell_wall(0.05, 0.035, 0.03, 0.03, 0.03, 0.01, 0.01) is None   # ellipses overlap
    mm = 1e-3                                                # published cells the old tangent solve rejected:
    for Req, Ri, L, A, B, a, b in ((101.205, 37, 62.665, 52.1, 47.9, 9.9, 11.3),        # Cornell ERL end (tolerance)
                                   (369.63, 150, 187, 73.52, 131.75, 106.25, 118.7),   # FCC-ee 400 MHz (start)
                                   (98.58, 32, 57.7, 49, 35.3, 10.5, 17)):             # NLSF-RE, re-entrant 101°
        w = cs.half_cell_wall(Req * mm, Ri * mm, L * mm, A * mm, B * mm, a * mm, b * mm)
        assert w is not None and np.all(np.diff(w[:, 1]) >= -1e-12) and w[24:, 0].min() >= cs.WEB_MIN * Req * mm


def test_fillet_tangency():
    from src.data_gen import cavity_shapes as cs
    segs = cs.filleted([(0, 0), (1, 0), (1, 1), (0, 1)], [0.2, 0.0, 0.3, 0.0])
    arcs = [s for s in segs if s[0] == "arc"]
    assert len(arcs) == 2
    for _, p, c, q in arcs:                                  # both ends on the circle
        assert np.isclose(np.linalg.norm(np.subtract(p, c)), np.linalg.norm(np.subtract(q, c)))
    lines = [s for s in segs if s[0] == "line"]
    assert all(np.linalg.norm(np.subtract(s[1], s[2])) > 0 for s in lines)


def test_freeform_surface_closed_and_varied():
    from src.data_gen import cavity_shapes as cs
    shapes = [cs.freeform_surface(np.random.default_rng(s), level=3) for s in range(6)]
    for P, F, _ in shapes:
        e = np.sort(F[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        assert (np.unique(e, axis=0, return_counts=True)[1] == 2).all()     # closed 2-manifold
        assert cs.surface_volume(P, F) > 0 and np.isfinite(P).all()
    ext = np.array([np.ptp(P, 0) / np.ptp(P, 0).max() for P, _, _ in shapes])
    assert ext.min() < 0.8                                   # not all spheres


@pytest.mark.parametrize("family", ["box", "coax_qw", "pillbox_port", "elliptical_long", "junction"])
def test_ood_families_are_topological_balls(set_args, family):
    from src.data_gen import cavity_shapes as cs
    assert family in cs.OOD_FAMILIES and family not in gen.FAMILIES
    set_args("--families", family, "--mesh_size", "0.2", "--seed", "1000", *SHARP)
    g = gen.mesh_sample(0)
    assert g["shape_type"] == family
    topo = gen.mesh_topology(g["tets"], len(g["nodes"]))
    assert topo["betti1"] == 0 and topo["n_shells"] == 1 and topo["euler"] == 1


def test_ood_box_matches_analytic(set_args):
    set_args("--families", "box", "--mesh_size", "0.12", "--seed", "1000", *SHARP)
    g = gen.mesh_sample(1)
    p = g["params"]
    ref = gen.eigenvalues_to_ghz(gen.box_spectrum(p["a"], p["b"], p["d"])[:5])
    assert np.abs(fe_modes(g["nodes"], g["tets"], 5)[2] / 1e9 / ref - 1).max() < 1e-2


# ─────────────────────────── diversity: deformation, Sobol, elliptical v2, active sampling ───

def test_smooth_deform_is_injective_and_orientation_preserving():
    from skfem import MeshTet
    m = MeshTet.init_tensor(*(np.linspace(0, 0.05, 7),) * 3)
    X, T = m.p.T.copy(), m.t.T

    def signed_vol(P):
        a, b, c, d = (P[T[:, i]] for i in range(4))
        return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)

    v0 = signed_vol(X)
    for seed in range(5):
        Y, prm = gen.smooth_deform(X, np.random.default_rng(seed), lip_max=0.5)
        assert np.all(np.sign(signed_vol(Y)) == np.sign(v0))           # no inverted tet
        assert 0 < prm["deform_lip"] <= 0.5 and prm["deform_max_disp"] > 0
        J = signed_vol(Y) / v0                                          # det(I + ∇δ) ∈ [(1-L)^3, (1+L)^3]
        assert J.min() > (1 - prm["deform_lip"]) ** 3 - 1e-9 and J.max() < (1 + prm["deform_lip"]) ** 3 + 1e-9


def test_sobol_rng_interface_and_family_balance(set_args):
    rng = gen.SobolRNG(np.linspace(0.05, 0.95, 8), np.random.default_rng(0))
    assert isinstance(rng.integers(1, 4), int) and 0.0 <= rng.random() < 1.0
    assert rng.uniform(2, 3, 3).shape == (3,) and rng.standard_normal((2, 3)).shape == (2, 3)
    assert sorted(rng.permutation(4).tolist()) == [0, 1, 2, 3] and rng.choice(["a", "b"]) in ("a", "b")
    assert rng.choice([1, 2, 3], p=[0, 0, 1]) == 3
    set_args("--sampling", "sobol", "--seed", "0")
    fams = [gen.ARGS.families[int(gen._sample_rng(s).integers(0, len(gen.ARGS.families)))] for s in range(64)]
    counts = [fams.count(f) for f in gen.ARGS.families]
    assert max(counts) - min(counts) <= 2                               # stratified (random: ~±6)


def test_elliptical_chain_end_cells_and_ood_split():
    from src.data_gen import cavity_shapes as cs
    rng = np.random.default_rng(3)
    for n in (1, 2, 5):
        p, halves, walls = cs.draw_elliptical(rng, n_cells=n)
        chain, z1 = cs.elliptical_chain(halves, walls, n)
        assert all(np.all(np.diff(w[:, 1]) >= -1e-12) for w in walls.values())   # r monotone in every half
        assert chain[:, 0].min() > -1e-12 and chain[:, 0].max() < z1 + 1e-12
        assert np.isclose(chain[0, 1], halves["e1"]["Riris"]) and np.isclose(chain[-1, 1], halves["e2"]["Riris"])
        assert np.isclose(chain[-1, 0], z1) and np.linalg.norm(np.diff(chain, axis=0), axis=1).max() < 0.2 * z1
    assert max(cs.N_CELLS) < 6                                          # OOD elliptical_long: 6–9 cells


def test_residual_indicator(set_args):
    """Label-free indicator of the service (src.service.model.residual): exact discrete pairs have η ≈ 0,
    perturbed ones not; the wall rows are not equations (essential BC) → excluded via the free mask."""
    from src.service.model import residual as residual_indicator
    set_args("--families", "hwr", "--mesh_size", "0.25", *SHARP)
    g = gen.mesh_sample(0)
    geom, Y, f = fe_modes(g["nodes"], g["tets"], 3)
    K, M, free = geom["fe"]["K"], geom["M"], ~geom["fe"]["bnd_edge"]
    vals = geom["fe"]["vals"]
    assert residual_indicator(K, M, Y, vals, free=free).max() < 1e-6
    noisy = Y + 0.05 * np.random.default_rng(0).standard_normal(Y.shape) * np.abs(Y).max()
    noisy[~free] = 0.0
    noisy /= np.sqrt(np.einsum("ij,ij->j", noisy, M @ noisy))
    assert residual_indicator(K, M, noisy, vals, free=free).min() > 1e-2
