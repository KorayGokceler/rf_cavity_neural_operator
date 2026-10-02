"""Tests for the 3D Maxwell (Nédélec N0, E field by default, H behind --field H) generator
src/data_gen/dataset_generator_3d.py.

Invariants:
- calibration box / pillbox frequencies match the analytic PEC spectra within the N0
  discretisation error at that mesh size (measured, --mesh_size 0.12, K = 8, max |f/f_exact − 1|:
  E box 8.0e-3, pillbox 7.3e-3 (E converges from below); H box 2.8e-3, pillbox 7.5e-3 (from above))
- no zero / gradient modes (divergence residual ~1e-14, λ_min ≫ 0); E: wall-edge DOFs exactly 0
- handles: H leaves a harmonic λ = 0 field (torus, half-wave coax) and rejects them; E does not
  (b1 adds nothing to its kernel) — half-wave coax TEM at c/2L
- isolated conductor (b2 = 1): E without the boundary-component potential has a spurious λ = 0
  mode, with it none
- edge DOF orientation low → high vertex index, DOF = line integral of the field
- per-sample seeding is reproducible
"""
import json
import os

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


@pytest.mark.parametrize("field,s_id,shape,tol", [("E", 0, "calib_box", 1e-2), ("E", 1, "calib_pillbox", 1e-2),
                                                  ("H", 0, "calib_box", 5e-3), ("H", 1, "calib_pillbox", 1e-2)])
def test_calibration_matches_analytic(set_args, field, s_id, shape, tol):
    set_args("--mode", "calibration", "--n_eigen_modes", "8", "--mesh_size", "0.12", "--field", field)
    res = gen.generate_sample_data(s_id)
    assert res is not None and res["shape_type"] == shape and res["field"] == field
    ref = gen.eigenvalues_to_ghz(gen.analytic_k2(shape, res["geom_params"], 8))
    rel = res["freqs"] / ref - 1
    assert np.abs(rel).max() < tol, rel
    if shape == "calib_pillbox":   # TM010 at c·2.405/(2πR)
        assert abs(res["freqs"][0] / (gen.C0 * 2.404825557695773 / (2 * np.pi * 0.04) / 1e9) - 1) < tol


def test_no_zero_or_gradient_modes_e(set_args):
    set_args("--mode", "calibration", "--n_eigen_modes", "6", "--mesh_size", "0.15")
    res = gen.generate_sample_data(1)
    assert res["field"] == "E" and "h_edges" not in res
    A = gen.assemble_h_n0(res["nodes"], res["tets"])
    U = res["e_edges"]
    G, bnd, n_pot, topo = gen.e_potential_matrix(A["edges"], len(res["nodes"]), res["tets"])
    assert bnd.sum() > 0 and np.all(U[bnd] == 0)                           # n × E = 0: wall DOFs exactly 0
    n_int = int((topo["vert_comp"] < 0).sum())
    assert n_pot == n_int and topo["n_comp"] == 1 and abs(G[:, n_pot:]).sum() == 0
    lam = np.einsum("ij,ij->j", U, A["K"] @ U) / np.einsum("ij,ij->j", U, A["M"] @ U)
    np.testing.assert_allclose(gen.eigenvalues_to_ghz(lam), res["freqs"], rtol=1e-8)
    assert lam.min() > 0.5 * (2.405 / 0.04) ** 2
    MU = A["M"] @ U
    assert np.abs(G.T @ MU).max() / np.abs(MU).max() < 1e-10               # ⟂_M the E-kernel gradients
    np.testing.assert_allclose(U.T @ MU, np.eye(U.shape[1]), atol=1e-8)


def test_no_zero_or_gradient_modes(set_args):
    set_args("--mode", "calibration", "--n_eigen_modes", "6", "--mesh_size", "0.15", "--field", "H")
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


def test_handles_h_rejected_e_clean():
    """Torus (b1 = 1): the H certificate fails and its harmonic Neumann field is a spurious λ = 0
    mode the gradient projection does not remove; the E formulation has no such field (measured
    f1: E 4.537 GHz, H non-zero 4.371 at h = 12 mm, 4.386 / 4.352 at 5 mm — same limit)."""
    nodes, tets = _occ_mesh(lambda occ: occ.addTorus(0, 0, 0, 0.05, 0.02), 0.012)
    topo = gen.mesh_topology(tets, len(nodes))
    assert topo["euler"] == 0 and topo["euler_boundary"] == 0 and not gen.is_topological_ball(topo)
    assert gen.is_valid_e_domain(topo) and topo["betti1"] == 1 and topo["n_bnd_components"] == 1
    with pytest.raises(RuntimeError, match="not a topological ball"):
        gen.certify(topo, "H")
    gen.certify(topo, "E")
    A = gen.assemble_h_n0(nodes, tets)
    with pytest.raises(RuntimeError, match="zero"):
        gen.solve_h_modes(A, 3)
    vals, _, info = gen.solve_e_modes(A, 3)
    f = gen.eigenvalues_to_ghz(vals)
    assert info["betti1"] == 1 and 4.3 < f[0] < 4.7


def test_half_wave_coax_tem():
    """Coax Ro = 50, ri = 15, L = 150 mm, inner conductor touching both end plates (b1 = 1), h = 8 mm:
    E gives the TEM λ/2 mode at c/2L (measured −0.62 %; TE11-like pair ≈ 1.80 GHz; TEM λ at c/L
    −0.70 %), H a spurious λ ≈ 3e-12 harmonic mode."""
    Ro, ri, L = 0.05, 0.015, 0.15

    def build(occ):
        occ.cut([(3, occ.addCylinder(0, 0, 0, 0, 0, L, Ro))], [(3, occ.addCylinder(0, 0, -0.01, 0, 0, L + 0.02, ri))])

    nodes, tets = _occ_mesh(build, 0.008)
    topo = gen.mesh_topology(tets, len(nodes))
    assert topo["betti1"] == 1 and not gen.is_topological_ball(topo) and gen.is_valid_e_domain(topo)
    A = gen.assemble_h_n0(nodes, tets)
    vals, U, _ = gen.solve_e_modes(A, 4)
    f = gen.eigenvalues_to_ghz(vals)
    assert abs(f[0] / (gen.C0 / (2 * L) / 1e9) - 1) < 1e-2 and abs(f[3] / (gen.C0 / L / 1e9) - 1) < 1e-2
    assert 1.7 < f[1] < f[2] < 1.9 and abs(f[2] / f[1] - 1) < 2e-3        # TE11-like degenerate pair
    with pytest.raises(RuntimeError, match="zero"):
        gen.solve_h_modes(A, 2)


def test_isolated_conductor_needs_component_potential():
    """Sphere (r = 15 mm) floating inside a PEC box (b2 = 1, two boundary shells): the field ∇φ
    with φ = 1 on the sphere, 0 on the box is curl-free with zero wall trace but not a gradient of
    an interior potential — without the component potential it survives as λ = 0; with it the
    spectrum is clean and matches the H formulation (valid here: b1 = 0) within the N0 error
    (measured f1 E 2.028 / H 2.123 GHz at h = 8 mm, 2.049 / 2.090 at 5 mm)."""
    def build(occ):
        occ.cut([(3, occ.addBox(0, 0, 0, 0.1, 0.08, 0.06))], [(3, occ.addSphere(0.05, 0.04, 0.03, 0.015))])

    nodes, tets = _occ_mesh(build, 0.01)
    topo = gen.mesh_topology(tets, len(nodes))
    assert topo["n_bnd_components"] == 2 and topo["betti2"] == 1 and topo["betti1"] == 0
    assert gen.is_valid_e_domain(topo) and not gen.is_topological_ball(topo)
    A = gen.assemble_h_n0(nodes, tets)
    with pytest.raises(RuntimeError, match="zero"):
        gen.solve_e_modes(A, 3, component_potentials=False)
    vals, U, info = gen.solve_e_modes(A, 3)
    fe = gen.eigenvalues_to_ghz(vals)
    fh = gen.eigenvalues_to_ghz(gen.solve_h_modes(A, 3)[0])
    assert info["n_bnd_components"] == 2 and np.abs(fe / fh - 1).max() < 0.08 and fe[0] > 1.5
    G, bnd, n_pot, bt = gen.e_potential_matrix(A["edges"], len(nodes), tets)
    assert n_pot == int((bt["vert_comp"] < 0).sum()) + 1
    sphere = bt["vert_comp"] == 1                                            # component 1 = the sphere (fewer vertices)
    assert np.allclose(np.linalg.norm(nodes[sphere] - [0.05, 0.04, 0.03], axis=1), 0.015, rtol=1e-3)
    np.testing.assert_array_equal(G[:, n_pot - 1].toarray().ravel(), A["G"] @ sphere.astype(float))


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
    """Every training family passes the E certificate (connected manifold); ball families are
    topological balls, handle families have the b1 of their construction; wall DOFs are 0."""
    from src.data_gen import cavity_shapes as cs
    set_args("--families", family, "--mesh_size", "0.2", "--n_eigen_modes", "4")
    for s_id in range(2):
        res = gen.generate_sample_data(s_id)
        assert res is not None and res["shape_type"] == family and res["field"] == "E"
        topo = gen.mesh_topology(res["tets"], len(res["nodes"]))
        assert gen.is_valid_e_domain(topo) and res["betti1"] == topo["betti1"]
        assert res["betti1"] == _expected_betti1(family, res["geom_params"]), res["geom_params"]
        assert gen.is_topological_ball(topo) == (family not in cs.HANDLE_FAMILIES)
        assert np.all(res["freqs"] > 0.3) and np.all(np.isfinite(res["e_edges"]))
        bnd = gen.e_potential_matrix(res["edges"], len(res["nodes"]), res["tets"])[1]
        assert np.all(res["e_edges"][bnd] == 0)


def test_field_flag_defaults(set_args):
    from src.data_gen import cavity_shapes as cs
    a = set_args()
    assert a.field == "E" and a.families == list(gen.FAMILIES) and set(cs.HANDLE_FAMILIES) <= set(a.families)
    b = set_args("--field", "H")
    assert b.families == list(gen.H_FAMILIES) and not set(cs.HANDLE_FAMILIES) & set(b.families)
    set_args("--field", "H", "--families", "hwr", "--mesh_size", "0.3")
    assert gen.generate_sample_data(0) is None                            # handle family: E only


def test_seeding_reproducible(set_args):
    set_args("--mesh_size", "0.2", "--n_eigen_modes", "3", "--seed", "7")
    a, b = gen.generate_sample_data(3), gen.generate_sample_data(3)
    assert a["shape_type"] == b["shape_type"] and a["geom_params"] == b["geom_params"]
    np.testing.assert_array_equal(a["nodes"], b["nodes"])
    np.testing.assert_array_equal(a["freqs"], b["freqs"])
    set_args("--mesh_size", "0.2", "--n_eigen_modes", "3", "--seed", "8")
    c = gen.generate_sample_data(3)
    assert c["geom_params"] != a["geom_params"]


@pytest.mark.parametrize("field,families", [("E", ["pillbox", "hwr"]), ("H", ["pillbox"])])
def test_cli_writes_h5(tmp_path, field, families):
    out = tmp_path / "d3.h5"
    gen.main(["--n_total", "2", "--n_workers", "2", "--mesh_size", "0.2", "--n_eigen_modes", "3",
              "--h5_filename", str(out), "--families", *families, "--field", field])
    assert out.exists() and not (tmp_path / "d3.h5.partial").exists()
    ds, other = ("e_edges", "h_edges") if field == "E" else ("h_edges", "e_edges")
    with h5py.File(out, "r") as f:
        meta = json.loads(f.attrs["metadata"])
        assert meta["field"] == field and meta["dataset"] == ds and "edges[i,0] < edges[i,1]" in meta["dof_convention"]
        assert sorted(f.keys()) == ["sample_0000", "sample_0001"]
        for key in f:
            g = f[key]
            ne, nv = g["edges"].shape[0], g["nodes"].shape[0]
            assert g["nodes"].shape == (nv, 3) and g["tets"].shape[1] == 4
            assert g[ds].shape == (ne, 3) and other not in g and g["freqs"].shape == (3,)
            assert g.attrs["field"] == field and g.attrs["shape_type"] in families
            assert int(g.attrs["betti1"]) == (0 if g.attrs["shape_type"] == "pillbox" else 1 + int(g.attrs["beam_port"]))
            assert int(g.attrs["n_bnd_components"]) == 1
            assert np.all(g["edges"][:, 0] < g["edges"][:, 1])


def test_cli_resume_appends_missing_ids(tmp_path):
    """--resume (cluster wall-time kills): an interrupted run's .partial keeps its samples, only
    the missing ids are generated, and the result equals an uninterrupted run; large id offsets
    (TRUBA family blocks) work."""
    common = ["--n_workers", "1", "--mesh_size", "0.2", "--n_eigen_modes", "3", "--families", "pillbox",
              "--sampling", "sobol", "--start_id", str(3 * 524288)]
    full, part = tmp_path / "full.h5", tmp_path / "part.h5"
    gen.main(["--n_total", "3", "--h5_filename", str(full), *common])
    gen.main(["--n_total", "1", "--h5_filename", str(part), *common])
    os.replace(part, str(part) + ".partial")                 # = a job killed after its first sample
    gen.main(["--n_total", "3", "--h5_filename", str(part), "--resume", *common])
    assert part.exists() and not os.path.exists(str(part) + ".partial")
    with h5py.File(full, "r") as a, h5py.File(part, "r") as b:
        assert sorted(a.keys()) == sorted(b.keys()) == [f"sample_{3 * 524288 + i:04d}" for i in range(3)]
        assert int(b.attrs["n_requested"]) == 3 and int(b.attrs["n_failed_last_run"]) == 0
        for k in a:
            np.testing.assert_array_equal(a[k]["nodes"][()], b[k]["nodes"][()])
            np.testing.assert_allclose(a[k]["freqs"][()], b[k]["freqs"][()], rtol=1e-10)


def test_mesher_error_restarts_gmsh(set_args, monkeypatch):
    """A 3D mesher error can leave the gmsh session broken (every later generate(3)
    yields no tets, gmsh.clear() does not help).  Simulated here: the session stays
    broken until gmsh.finalize(); the generator must restart gmsh and succeed."""
    import gmsh
    set_args("--n_eigen_modes", "3", "--mesh_size", "0.2", "--families", "pillbox")
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
    single cell with 80 mm PEC-capped pipes: converged TM010 = 1.288 GHz (Richardson from H mesh
    0.05 / 0.035; the periodic π-mode of the real structure is 1.300).  H-N0 converges from above,
    O(h²): +1.3 % at mesh 0.10, +0.63 % at 0.07, +0.33 % at 0.05.  E-N0 is far more accurate for
    TM010 (E_z smooth, no H_φ wall singularity of the faceted wall): +0.15 % at 0.10, +0.03 % at 0.07."""
    from src.data_gen import cavity_shapes as cs
    tesla = dict(Req=0.1033, Riris=0.035, L=0.0577, A=0.042, B=0.042, a=0.012, b=0.019, n_cells=1.0, Lpipe=0.08)
    nodes, tets = _mesh_params(cs.build_elliptical, tesla)
    A = gen.assemble_h_n0(nodes, tets)
    err = gen.eigenvalues_to_ghz(gen.solve_h_modes(A, 3)[0][0]) / 1.288 - 1
    assert 0.005 < err < 0.02
    err_e = gen.eigenvalues_to_ghz(gen.solve_e_modes(A, 3)[0][0]) / 1.288 - 1
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
    for P, F, prm in shapes:
        e = np.sort(F[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        assert (np.unique(e, axis=0, return_counts=True)[1] == 2).all()     # closed 2-manifold
        assert cs.surface_volume(P, F) > 0 and np.isfinite(P).all()
    ext = np.array([np.ptp(P, 0) / np.ptp(P, 0).max() for P, _, _ in shapes])
    assert ext.min() < 0.8                                   # not all spheres


@pytest.mark.parametrize("family", ["box", "coax_qw", "pillbox_port", "elliptical_long", "junction"])
def test_ood_families_are_topological_balls(set_args, family):
    from src.data_gen import cavity_shapes as cs
    assert family in cs.OOD_FAMILIES and family not in gen.FAMILIES
    set_args("--families", family, "--mesh_size", "0.2", "--n_eigen_modes", "3", "--seed", "1000", "--field", "H")
    res = gen.generate_sample_data(0)
    assert res is not None and res["shape_type"] == family
    assert gen.is_topological_ball(gen.mesh_topology(res["tets"], len(res["nodes"])))


def test_ood_box_matches_analytic(set_args):
    set_args("--families", "box", "--mesh_size", "0.12", "--n_eigen_modes", "5", "--seed", "1000", "--field", "H")
    res = gen.generate_sample_data(1)
    p = res["geom_params"]
    ref = gen.eigenvalues_to_ghz(gen.box_spectrum(p["a"], p["b"], p["d"])[:5])
    assert np.abs(res["freqs"] / ref - 1).max() < 5e-3


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
        assert np.all(np.diff(chain[:, 0]) > -1e-12)                   # z monotone along the chain
        assert np.isclose(chain[0, 1], halves["e1"]["Riris"]) and np.isclose(chain[-1, 1], halves["e2"]["Riris"])
        assert np.isclose(chain[-1, 0], z1) and np.linalg.norm(np.diff(chain, axis=0), axis=1).max() < 0.2 * z1
    assert max(cs.N_CELLS) < 6                                          # OOD elliptical_long: 6–9 cells


def test_active_residual_indicator(set_args):
    from scripts.active_sampling import residual_indicator
    set_args("--families", "pillbox_pipes", "--mesh_size", "0.25", "--n_eigen_modes", "3")
    g = gen.mesh_sample(0)
    A = gen.assemble_h_n0(g["nodes"], g["tets"])
    vals, vecs, _ = gen.solve_h_modes(A, 3)
    assert residual_indicator(A["K"], A["M"], vecs, vals).max() < 1e-6    # exact discrete pairs
    noisy = vecs + 0.05 * np.random.default_rng(0).standard_normal(vecs.shape) * np.abs(vecs).max()
    noisy /= np.sqrt(np.einsum("ij,ij->j", noisy, A["M"] @ noisy))
    assert residual_indicator(A["K"], A["M"], noisy, vals).min() > 1e-2


def test_item_from_geometry_runs_the_model(set_args):
    import torch
    from src.data.dataset_3d import item_from_geometry, maxwell3d_collate
    from src.data.dataset_converter_3d import extract_geometry_3d
    from src.training.lightning_module import GNOTLightning
    set_args("--families", "composite", "--mesh_size", "0.25", "--n_eigen_modes", "3")
    g = gen.mesh_sample(1)
    geom, _ = extract_geometry_3d(g["nodes"], g["tets"])
    batch = maxwell3d_collate([item_from_geometry(geom)])
    torch.manual_seed(0)
    lm = GNOTLightning(val_dim=9, grid_dim=3, hidden_dim=16, n_heads=2, n_basis=8, num_field_modes=3, rff_dim=8,
                       model_type="eigenspace3d", physics_freq=True, eigenspace_kwargs={"n_layers": 1})
    lm.freq_stats = {"mean": 0.0, "std": 1.0}
    with torch.no_grad():
        out = lm.model(batch)
    assert out["field"].shape == (1, len(geom["edges"]), 3) and torch.isfinite(out["eigenvalues"]).all()


def test_active_residual_indicator_e(set_args):
    """E pairs: wall rows are not equations (essential BC) → excluded via the free mask."""
    from scripts.active_sampling import residual_indicator
    from src.data.dataset_converter_3d import geometry_operators, h5_edges_to_canonical
    set_args("--field", "E", "--families", "hwr", "--mesh_size", "0.25", "--n_eigen_modes", "3")
    g = gen.mesh_sample(0)
    A = gen.assemble_h_n0(g["nodes"], g["tets"])
    vals, vecs, _ = gen.solve_e_modes(A, 3)
    X = np.asarray(A["mesh"].p.T)
    ops, _ = geometry_operators(X, np.asarray(A["mesh"].t.T), field="E")
    rows, _ = h5_edges_to_canonical(A["edges"], len(X), ops["edges"])   # skfem order → canonical order
    free = ~np.asarray(ops["bnd_edge"])[rows]
    assert residual_indicator(A["K"], A["M"], vecs, vals, free=free).max() < 1e-6
