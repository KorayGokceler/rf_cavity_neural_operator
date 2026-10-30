"""Tests for the 3D PEC cavity generator src/data_gen/dataset_generator_3d.py (geometry families, gmsh mesh,
topology certificate, field-label CLI) and the physical reference solves of the p2 path (tests/p2_reference.py:
generator solid -> BREP -> curved netgen mesh -> highorder.solve_modes at order 2).

Invariants (every tolerance below is ~2x the measured error on that mesh):
- calibration box / pillbox, OOD box, TESLA cell: the p2 frequencies match the analytic / converged values
  within the discretisation error of a ~400-3000 tet curved mesh
- no zero / gradient modes (wall DOFs exactly 0, Rayleigh quotient = f, M-orthogonal to the kernel
  gradients G, K G = 0, lowest mode = TM010)
- handles (b1 > 0) add nothing to the E kernel: torus, half-wave coax TEM at c/2L
- isolated conductor (b2 = 1): solve_modes meets and drops exactly one lambda = 0 mode (the shell potential
  the interior-gradient projection does not remove); field_operators' G carries that potential as an extra column
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
import src.data_gen.highorder as ho  # noqa: E402
from tests import p2_reference as p2  # noqa: E402

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


@pytest.mark.parametrize("s_id,shape", [(0, "calib_box"), (1, "calib_pillbox")])
def test_calibration_matches_analytic(set_args, tmp_path, s_id, shape):
    """The p2 solve on the generator's calibration solid (curved netgen mesh, maxh = the generator's h,
    ~400 tets) vs the closed-form PEC spectrum, K = 8.  Measured max |f/f_exact - 1|: box 4.8e-3 (442 tets),
    pillbox 5.7e-3 (355 tets); TM010 4.1e-4."""
    set_args("--mode", "calibration", "--mesh_size", "0.2")
    g = gen.mesh_sample(s_id, cad_path=str(tmp_path / "solid.brep"))
    assert g["shape_type"] == shape
    mesh = p2.curved_mesh(tmp_path / "solid.brep", g["h"])
    res = p2.solve(mesh, 8)
    ref = gen.eigenvalues_to_ghz(gen.analytic_k2(shape, g["params"], 8))
    rel = res["f_hz"] / 1e9 / ref - 1
    assert res["info"]["n_zero"] == 0 and np.abs(rel).max() < 1e-2, rel
    if shape == "calib_pillbox":   # TM010 at c·2.405/(2πR)
        assert abs(res["f_hz"][0] / (gen.C0 * 2.404825557695773 / (2 * np.pi * 0.04)) - 1) < 1e-3


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


@pytest.fixture(scope="module")
def pillbox_p2(tmp_path_factory):
    """Pillbox R = 40, L = 50 mm: curved mesh (355 tets, 4482 DOF), 6 p2 modes, field_operators' K, M, G."""
    R, L = 0.04, 0.05
    tmp = tmp_path_factory.mktemp("pillbox")
    mesh, res = p2.occ_modes(lambda occ: occ.addCylinder(0, 0, 0, 0, 0, L, R), 0.0126, 6, tmp)
    return mesh, res, ho.field_operators(mesh, order=2)


def test_no_zero_or_gradient_modes(pillbox_p2):
    """p2 analogue of the N0 kernel checks: the kernel projection leaves no lambda ~ 0 mode and no gradient
    in the Ritz block.  Measured: Rayleigh quotients = f to 3e-15, GᵀMU 4e-15 relative, K G 5e-16 relative,
    UᵀMU = I to 4e-15."""
    mesh, res, ops = pillbox_p2
    U, f, s = res["U"], res["f_hz"], ops["scale"]
    K, M, G, bnd = ops["K"], ops["M"], ops["G"], ops["bnd"]
    assert bnd.sum() > 0 and np.all(U[bnd] == 0)                           # n × E = 0: wall DOFs exactly 0
    assert res["info"]["n_zero"] == 0 and len(ho.wall_shells(mesh)) == 1 and 0 < ops["n_pot"] < res["info"]["n_h1"]
    lam = np.einsum("ij,ij->j", U, K @ U) / np.einsum("ij,ij->j", U, M @ U)
    np.testing.assert_allclose(gen.C0 * np.sqrt(lam) / (2 * np.pi * s), f, rtol=1e-8)
    assert f[0] > np.sqrt(0.5) * gen.C0 * 2.405 / (2 * np.pi * 0.04)       # no λ ≈ 0 mode: TM010 is the lowest
    assert abs(K @ G).max() / abs(K).max() < 1e-12                         # the kernel columns are in ker K
    MU = M @ U
    assert np.abs(G.T @ MU).max() / np.abs(MU).max() < 1e-10               # ⟂_M the E-kernel gradients
    np.testing.assert_allclose(s * (U.T @ MU), np.eye(U.shape[1]), atol=1e-8)   # unit mass norm (M is M/scale)
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


def test_handles_are_clean(tmp_path):
    """Torus (b1 = 1): the handle adds nothing to the E kernel, no spurious λ = 0 mode.  p2 on 1054 curved
    tets: f1 4.369 GHz (converged p2: 4.3575 at 2.4k-7.7k tets; the old N0 values 4.537 at h = 12 mm and
    4.386 at 5 mm bracket it from above)."""
    build = lambda occ: occ.addTorus(0, 0, 0, 0.05, 0.02)  # noqa: E731
    nodes, tets = _occ_mesh(build, 0.012)
    topo = gen.mesh_topology(tets, len(nodes))
    assert topo["euler"] == 0 and topo["euler_boundary"] == 0
    assert gen.is_valid_e_domain(topo) and topo["betti1"] == 1 and topo["n_bnd_components"] == 1
    gen.certify(topo)
    _, res = p2.occ_modes(build, 0.03, 3, tmp_path, curvaturesafety=0.7)
    f = res["f_hz"] / 1e9
    assert res["info"]["n_zero"] == 0 and np.all(np.diff(f) >= 0)
    assert 4.33 < f[0] < 4.40, f                                          # clean spectrum, handle mode included


def test_half_wave_coax_tem(tmp_path):
    """Coax Ro = 50, ri = 15, L = 150 mm, inner conductor touching both end plates (b1 = 1), p2 on 1292
    curved tets: the TEM λ/2 mode at c/2L (measured +0.085 %, N0 at 8 mm: -0.62 %), TEM λ at c/L (+0.11 %),
    TE11-like pair ≈ 1.81 GHz split by 1e-4 (+0.02 % at 3000 tets)."""
    Ro, ri, L = 0.05, 0.015, 0.15

    def build(occ):
        occ.cut([(3, occ.addCylinder(0, 0, 0, 0, 0, L, Ro))], [(3, occ.addCylinder(0, 0, -0.01, 0, 0, L + 0.02, ri))])

    nodes, tets = _occ_mesh(build, 0.008)
    topo = gen.mesh_topology(tets, len(nodes))
    assert topo["betti1"] == 1 and gen.is_valid_e_domain(topo)
    _, res = p2.occ_modes(build, 0.05, 4, tmp_path, curvaturesafety=0.7)
    f = res["f_hz"] / 1e9
    assert res["info"]["n_zero"] == 0
    assert abs(f[0] / (gen.C0 / (2 * L) / 1e9) - 1) < 3e-3 and abs(f[3] / (gen.C0 / L / 1e9) - 1) < 3e-3
    assert 1.7 < f[1] < f[2] < 1.9 and abs(f[2] / f[1] - 1) < 1e-3        # TE11-like degenerate pair


def test_isolated_conductor_has_a_component_potential(tmp_path):
    """Sphere (r = 15 mm) floating inside a PEC box (b2 = 1, two boundary shells): the field ∇φ with
    φ = 1 on the sphere, 0 on the box is curl-free with zero wall trace but not the gradient of an interior
    potential, so the interior-gradient projection of solve_modes cannot remove it: it shows up as exactly one
    λ = 0 Ritz value, which solve_modes drops (info n_zero = 1), and field_operators' G carries it as one extra
    column.  p2 on 516 tets: f1 2.0686 GHz (2.0688 at 2.3k tets; the old N0 values 2.028 / 2.049 and the
    removed H formulation's 2.123 / 2.090 bracket the limit)."""
    import scipy.sparse.linalg as spla

    def build(occ):
        occ.cut([(3, occ.addBox(0, 0, 0, 0.1, 0.08, 0.06))], [(3, occ.addSphere(0.05, 0.04, 0.03, 0.015))])

    nodes, tets = _occ_mesh(build, 0.01)
    topo = gen.mesh_topology(tets, len(nodes))
    assert topo["n_bnd_components"] == 2 and topo["betti2"] == 1 and topo["betti1"] == 0
    assert gen.is_valid_e_domain(topo)
    mesh, res = p2.occ_modes(build, 0.03, 3, tmp_path, curvaturesafety=1.5)
    f = res["f_hz"] / 1e9
    assert res["info"]["n_zero"] == 1                                       # the spurious λ = 0 mode, dropped
    assert abs(f[0] / 2.0688 - 1) < 2e-3 and 2.049 < f[0] < 2.090 and np.all(np.diff(f) >= 0), f
    ops = ho.field_operators(mesh, order=2)
    K, M, G, Kp, bnd, n_pot = ops["K"], ops["M"], ops["G"], ops["Kp"], ops["bnd"], ops["n_pot"]
    _, fesh1 = res["fes"].CreateGradient()
    assert n_pot == int(np.sum(list(fesh1.FreeDofs()))) + 1                 # interior potentials + the sphere's
    shells = ho.wall_shells(mesh)
    assert len(shells) == 2
    X = np.array([tuple(p.p) for p in mesh.ngmesh.Points()])
    sphere = shells[1]                                                      # the smaller shell
    assert np.allclose(np.linalg.norm(X[sphere] - [0.05, 0.04, 0.03], axis=1), 0.015, rtol=1e-3)
    g = G[:, n_pot - 1].toarray().ravel()
    assert np.all(g[bnd] == 0) and np.abs(K @ g).max() < 1e-10 * np.abs(g).max() * abs(K).max()   # curl-free, wall trace 0
    # not in the span of the interior gradients: the M-norm left after projecting them out is O(1) of its own norm
    Kii, kib = Kp[: n_pot - 1, : n_pot - 1].tocsc(), Kp[: n_pot - 1, n_pot - 1].toarray().ravel()
    left = Kp[n_pot - 1, n_pot - 1] - kib @ spla.spsolve(Kii, kib)
    assert left > 0.1 * Kp[n_pot - 1, n_pot - 1]
    MU = M @ res["U"]
    assert np.abs(G.T @ MU).max() / np.abs(MU).max() < 1e-8                 # the physical modes are ⟂_M the shell potential too


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

def test_tesla_cell_frequency(tmp_path):
    """TESLA mid-cell dimensions (Req 103.3, Riris 35, A = B = 42, a 12, b 19, L 57.7 mm) as a
    single cell with 80 mm PEC-capped pipes: converged TM010 = 1.288 GHz (Richardson; the periodic
    π-mode of the real structure is 1.300).  p2 on 2555 curved tets: 1.28825 (+0.019 %), 1.28814 (+0.011 %)
    at 10k tets; the old N0 value was +0.15 % at mesh 0.10, +0.03 % at 0.07."""
    from src.data_gen import cavity_shapes as cs
    tesla = dict(Req=0.1033, Riris=0.035, L=0.0577, A=0.042, B=0.042, a=0.012, b=0.019, n_cells=1.0, Lpipe=0.08)
    brep = p2.write_brep(lambda occ: cs.build_elliptical(occ, None, dict(tesla)), tmp_path / "tesla.brep")
    mesh = p2.curved_mesh(brep, 0.03, split="auto", curvaturesafety=1.0)
    err = p2.solve(mesh, 3)["f_hz"][0] / 1e9 / 1.288 - 1
    assert abs(err) < 1e-3, err


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


def test_ood_box_matches_analytic(set_args, tmp_path):
    """OOD box (seed 1000, id 1: 77 x 35 x 53 mm), p2 on 635 curved tets at 1.5 h: max |f/f_exact - 1| over the
    5 lowest modes 2.3e-3 (7.6e-3 at 279 tets, 2.4e-4 at 2266)."""
    set_args("--families", "box", "--mesh_size", "0.12", "--seed", "1000", *SHARP)
    g = gen.mesh_sample(1, cad_path=str(tmp_path / "box.brep"))
    p = g["params"]
    ref = gen.eigenvalues_to_ghz(gen.box_spectrum(p["a"], p["b"], p["d"])[:5])
    res = p2.solve(p2.curved_mesh(tmp_path / "box.brep", 1.5 * g["h"]), 5)
    assert np.abs(res["f_hz"] / 1e9 / ref - 1).max() < 5e-3


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


def test_residual_indicator(pillbox_p2):
    """Label-free indicator of the service (src.service.model.residual): exact discrete pairs have η ≈ 0,
    perturbed ones not; the wall rows are not equations (essential BC) → excluded via the free mask.
    field_operators' K, M live on the normalised geometry, so λ_ξ = λ·scale².  Measured: exact pairs
    1.4e-12, perturbed ≥ 1e2."""
    from src.service.model import residual as residual_indicator
    _, res, ops = pillbox_p2
    K, M, free = ops["K"], ops["M"], ~ops["bnd"]
    Y, vals = res["U"][:, :3], res["lam"][:3] * ops["scale"] ** 2
    assert residual_indicator(K, M, Y, vals, free=free).max() < 1e-6
    noisy = Y + 0.05 * np.random.default_rng(0).standard_normal(Y.shape) * np.abs(Y).max()
    noisy[~free] = 0.0
    noisy /= np.sqrt(np.einsum("ij,ij->j", noisy, M @ noisy))
    assert residual_indicator(K, M, noisy, vals, free=free).min() > 1e-2
