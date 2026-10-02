"""Cavity figures of merit (src/qoi, docs/24_CAVITY_QOI.md): closed forms, operator exactness,
FE vs analytic on gmsh pillbox / box meshes (E and H formulation), normalisation and API invariants.
The convergence study lives in scripts/qoi_validation.py (docs/24 §3)."""
import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("skfem")

from src.data.dataset_converter_3d import assemble_n0, boundary_faces, edges_from_tets, to_csr_tuple  # noqa: E402
from src.qoi import QOI_KEYS, QOI_LABELS, build_qoi_operators, cavity_qoi, qoi_from_dofs  # noqa: E402
from src.qoi import analytic as an  # noqa: E402
from src.qoi.operators import (EPS0, MU0, boundary_face_data, surface_operators,  # noqa: E402
                               surface_resistance, wall_loss_matrix)

PILLBOX = (0.10, 0.10)
BOX = (0.10, 0.08, 0.06)


# ─────────────────────────── closed forms ──────────────────────

def test_qoi_labels_contract():
    assert QOI_LABELS == ('Q0', 'G_ohm', 'R_over_Q_ohm', 'R_sh_ohm', 'T_transit', 'Epk_Eacc',
                          'Bpk_Eacc_mT_per_MVm')
    assert set(QOI_LABELS) <= set(QOI_KEYS)
    assert set(an.pillbox_tm010(*PILLBOX)) == set(QOI_KEYS)


def test_pillbox_closed_form_textbook():
    R = 0.1
    ref = an.pillbox_tm010(R, np.pi * R / an.J01)         # kL = π: T = 2/π, linac R/Q ≈ 196 Ω
    assert ref["T_transit"] == pytest.approx(2 / np.pi, rel=1e-12)
    assert ref["R_over_Q_ohm"] == pytest.approx(195.9, rel=1e-3)
    p = an.pillbox_tm010(R, R)
    assert p["f_Hz"] == pytest.approx(an.C0 * 2.404826 / (2 * np.pi * R), rel=1e-6)
    assert p["G_ohm"] == pytest.approx(an.ETA0 * an.J01 / 4, rel=1e-12)    # η j01 L / (2(R+L))
    assert p["Q0"] * p["Rs_ohm"] == pytest.approx(p["G_ohm"])
    assert p["R_sh_ohm"] == pytest.approx(p["R_over_Q_ohm"] * p["Q0"])
    assert p["Epk_Eacc"] == pytest.approx(1 / p["T_transit"])
    assert p["Bpk_Eacc_mT_per_MVm"] == pytest.approx(1e9 * an.J1MAX / (an.C0 * p["T_transit"]))
    assert p["Rs_ohm"] == pytest.approx(np.sqrt(np.pi * p["f_Hz"] * MU0 / 5.8e7))


def test_box_closed_form_matches_pozar():
    a, b, d = BOX
    ref = an.box_te101(a, b, d, Rs=0.01)
    k = np.pi * np.hypot(1 / a, 1 / b)
    # Pozar TE101 (E ∥ his b): Q = (k a' d')³ b' η / (2π² R_s (2a'³b' + 2b'd'³ + a'³d' + a'd'³)), (a', b', d') = (a, d, b)
    A, B, D = a, d, b
    q = (k * A * D) ** 3 * B * an.ETA0 / (2 * np.pi ** 2 * 0.01 * (2 * A ** 3 * B + 2 * B * D ** 3 + A ** 3 * D
                                                                  + A * D ** 3))
    assert ref["Q0"] == pytest.approx(q, rel=1e-10)
    circ = an.box_te101(a, b, d, Rs=0.01, convention="circuit")
    assert circ["R_over_Q_ohm"] == pytest.approx(ref["R_over_Q_ohm"] / 2)


# ─────────────────────────── operator exactness ────────────────

@pytest.fixture(scope="module")
def jittered_box():
    """Small skfem tensor box, interior vertices jittered, edges in skfem (unsorted) order."""
    from tests.maxwell3d_synth import box_geometry
    geom, lam, vec = box_geometry((1.0, 0.8, 0.6), n=4, jitter=0.2, scale=0.1, n_modes=3, field="H")
    return geom


def _interp(X, edges, field):
    """N0 interpolant ∫_e F·t of a field that is linear in x (2-point Gauss is exact)."""
    xa, xb = X[edges[:, 0]], X[edges[:, 1]]
    s = np.array([0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)])
    return sum(0.5 * (field(xa + si * (xb - xa)) * (xb - xa)).sum(-1) for si in s)


def test_boundary_face_data_matches_converter(jittered_box):
    X, t = jittered_box["X"].astype(float), jittered_box["tets"]
    bf = boundary_face_data(X, t)
    np.testing.assert_array_equal(bf["faces"], boundary_faces(X, t))
    assert bf["area"].sum() == pytest.approx(2 * (0.8 + 0.48 + 0.6), rel=1e-6)
    cen = X[bf["faces"]].mean(1)
    assert np.all((bf["normal"] * cen).sum(1) > 0)                  # outward (box centred at 0)


def test_wall_loss_exact_for_constant_and_rotational_fields(jittered_box):
    X, t, e = jittered_box["X"].astype(float), jittered_box["tets"], jittered_box["edges"]
    bf = boundary_face_data(X, t)
    c = np.array([0.3, -1.2, 0.7])
    u = _interp(X, e, lambda x: np.broadcast_to(c, x.shape))       # H = c (in N0 exactly)
    exact = (bf["area"] * (np.cross(bf["normal"], c) ** 2).sum(1)).sum()
    S_H = wall_loss_matrix(X, t, e, "H")
    assert u @ S_H @ u == pytest.approx(exact, rel=1e-12)
    assert abs(S_H - S_H.T).max() < 1e-14
    u2 = _interp(X, e, lambda x: np.cross(c, x))                    # curl(c × x) = 2c
    S_E = wall_loss_matrix(X, t, e, "E")
    exact2 = (bf["area"] * (np.cross(bf["normal"], 2 * c) ** 2).sum(1)).sum()
    assert u2 @ S_E @ u2 == pytest.approx(exact2, rel=1e-12)


def test_axis_and_surface_operators_exact_for_constants(jittered_box):
    g = jittered_box
    X, t, e = g["X"].astype(float), g["tets"], g["edges"]
    c = np.array([0.3, -1.2, 0.7])
    u_c = _interp(X, e, lambda x: np.broadcast_to(c, x.shape))
    u_r = _interp(X, e, lambda x: np.cross(c, x))
    bf = boundary_face_data(X, t)
    n = bf["normal"]
    for field in ("E", "H"):
        ops = build_qoi_operators(X, t, e, g["scale"], g["center"], field, n_axis=50)
        assert ops["L_axis"] == pytest.approx(0.6, rel=1e-6)        # chord of the box along z (float32 X)
        assert ops["q"].sum() == pytest.approx(ops["L_axis"]) and np.all(ops["q"] > 0)
        # Az: E primary → value_z of the constant field (c_z); H primary → curl_z of c × x (2 c_z)
        target = c[2] if field == "E" else 2 * c[2]
        np.testing.assert_allclose(ops["Az"] @ (u_c if field == "E" else u_r), target, rtol=1e-10)
        for method in ("centroid", "recovered"):
            Es, Hs = surface_operators(X, t, e, field, method)
            if field == "E":   # E = c: normal part; H ∝ curl(c × x) = 2c: tangential part
                Ev, Hv = (Es @ u_c).reshape(-1, 3), (Hs @ u_r).reshape(-1, 3)
                np.testing.assert_allclose(Ev, (n @ c)[:, None] * n, atol=1e-10)
                np.testing.assert_allclose(Hv, 2 * (c - (n @ c)[:, None] * n), atol=1e-10)
            else:              # H = c: tangential; E ∝ curl(c × x) = 2c: normal
                Ev, Hv = (Es @ u_r).reshape(-1, 3), (Hs @ u_c).reshape(-1, 3)
                np.testing.assert_allclose(Ev, 2 * (n @ c)[:, None] * n, atol=1e-10)
                np.testing.assert_allclose(Hv, c - (n @ c)[:, None] * n, atol=1e-10)


def test_build_operators_shapes_and_mass(jittered_box):
    g = jittered_box
    X, t, e = g["X"].astype(float), g["tets"], g["edges"]
    ne = len(e)
    M_ref = sp.csr_matrix((g["M"][2], g["M"][1], g["M"][0]), shape=(ne, ne))
    ops = build_qoi_operators(X, t, e, g["scale"], g["center"], "H")     # M assembled, unsorted edges
    assert abs(ops["M"] - M_ref).max() < 1e-6 * abs(M_ref).max()       # skfem M from the float64 mesh
    nf = len(ops["face_area"])
    assert ops["Esurf"].shape == ops["Hsurf"].shape == (3 * nf, ne)
    assert ops["Az"].shape == (401, ne) and ops["zeta"].shape == ops["q"].shape == (401,)
    assert ops["S"].shape == (ne, ne) and isinstance(ops["S"], sp.csr_matrix)
    assert set(ops) == {"field", "scale", "M", "S", "Az", "zeta", "q", "L_axis", "Esurf", "Hsurf", "face_area"}


# ─────────────────────────── FE vs analytic (gmsh) ─────────────

@pytest.fixture(scope="module")
def cases():
    pytest.importorskip("gmsh")
    from tests.conftest import _gmsh_available
    if not _gmsh_available():
        pytest.skip("gmsh not available")
    out = {}
    for shape, dims, h in (("pillbox", PILLBOX, 0.02), ("box", BOX, 0.01)):
        for field in ("E", "H"):
            out[shape, field] = an.reference_case(shape, dims, h, field, k=3)
    return out


# relative tolerances at the test meshes (Ne ≈ 3k); E-primary Q0 on the curved pillbox wall is
# first order (docs/24 §2.2: −6.4 % here, −2.9 % at Ne = 54k)
TOL = {"f_Hz": 0.012, "Q0": 0.02, "G_ohm": 0.02, "R_over_Q_ohm": 0.03, "R_sh_ohm": 0.03, "T_transit": 0.015,
       "Epk_Eacc": 0.03, "Bpk_Eacc_mT_per_MVm": 0.03}


@pytest.mark.gmsh
@pytest.mark.parametrize("shape", ["pillbox", "box"])
@pytest.mark.parametrize("field", ["E", "H"])
def test_fe_matches_analytic(cases, shape, field):
    geom, Y, f, ref = cases[shape, field]
    q = cavity_qoi(geom, Y[:, 0], f[0] / 1e9)
    for k, tol in TOL.items():
        if shape == "pillbox" and field == "E" and k in ("Q0", "G_ohm", "R_sh_ohm"):
            tol = 0.10
        assert q[k][0] == pytest.approx(ref[k], rel=tol), (k, q[k][0], ref[k])
    assert q["L_acc_m"][0] == pytest.approx(ref["L_acc_m"], rel=1e-9)
    assert q["U_J"][0] == 1.0


@pytest.mark.gmsh
@pytest.mark.parametrize("shape", ["pillbox", "box"])
def test_e_and_h_formulation_agree(cases, shape):
    qs = {}
    for field in ("E", "H"):
        geom, Y, f, _ = cases[shape, field]
        qs[field] = cavity_qoi(geom, Y[:, 0], f[0] / 1e9)
    for k in ("R_over_Q_ohm", "T_transit", "Epk_Eacc", "Bpk_Eacc_mT_per_MVm"):
        assert qs["E"][k][0] == pytest.approx(qs["H"][k][0], rel=0.035), k
    assert qs["E"]["Q0"][0] == pytest.approx(qs["H"]["Q0"][0], rel=0.08 if shape == "pillbox" else 0.02)


@pytest.mark.gmsh
@pytest.mark.parametrize("field", ["E", "H"])
def test_invariance_normalisation_and_conventions(cases, field):
    geom, Y, f, ref = cases["box", field]
    ops = build_qoi_operators(geom["X"], geom["tets"], geom["edges"], geom["scale"], geom["center"], field,
                              M=to_csr_tuple(geom["M"]))
    u = Y[:, 0]
    q = qoi_from_dofs(ops, u, f[0])
    for k, v in qoi_from_dofs(ops, -37.5 * u, f[0]).items():            # sign / amplitude invariance
        np.testing.assert_allclose(v, q[k], rtol=1e-12, err_msg=k)
    # U = 1 J by hand: scale u to 1 J, evaluate V and E_pk directly
    s, w = geom["scale"], 2 * np.pi * f[0]
    cU = EPS0 if field == "E" else MU0
    a = 1.0 / np.sqrt(0.5 * cU * s ** 3 * (u @ (ops["M"] @ u)))
    if field == "E":   # independent axis integral of the Whitney E_z (nedelec.eval_field, trapezoid)
        from src.viz.nedelec import eval_field
        zeta = np.linspace(ops["zeta"][0], ops["zeta"][-1], 4001)
        pts = np.column_stack([np.zeros_like(zeta), np.zeros_like(zeta), zeta])
        ez = np.nan_to_num(s * eval_field(geom["X"], geom["tets"], geom["edges"], a * u, pts)[:, 2])
        V = abs(np.trapezoid(ez * np.exp(1j * w * zeta * s / an.C0), zeta))
        assert q["V_acc_V"][0] == pytest.approx(V, rel=2e-3)
    Epk = np.linalg.norm((ops["Esurf"] @ (a * u)).reshape(-1, 3), axis=1).max()
    Epk *= 1.0 if field == "E" else 1.0 / (w * EPS0 * s)
    assert q["E_pk_Vm"][0] == pytest.approx(Epk, rel=1e-12)
    assert q["P_c_W"][0] * q["Q0"][0] == pytest.approx(w)                # ωU/P_c with U = 1 J
    assert q["R_over_Q_ohm"][0] == pytest.approx(q["V_acc_V"][0] ** 2 / w)
    # linac vs circuit: R/Q and R_sh halved, everything else unchanged
    qc = qoi_from_dofs(ops, u, f[0], convention="circuit")
    for k in QOI_KEYS:
        fac = 0.5 if k in ("R_over_Q_ohm", "R_sh_ohm") else 1.0
        np.testing.assert_allclose(qc[k], fac * q[k], rtol=1e-12, err_msg=k)
    # Rs override: Q0 = G/Rs with G unchanged; copper default = surface_resistance(f)
    assert q["Rs_ohm"][0] == pytest.approx(surface_resistance(f[0]))
    qn = qoi_from_dofs(ops, u, f[0], Rs=10e-9)
    assert qn["G_ohm"][0] == pytest.approx(q["G_ohm"][0], rel=1e-12)
    assert qn["Q0"][0] == pytest.approx(q["G_ohm"][0] / 10e-9, rel=1e-12)
    assert qn["R_sh_ohm"][0] == pytest.approx(q["R_over_Q_ohm"][0] * qn["Q0"][0], rel=1e-12)
    # L_acc override only rescales E_acc and the peak ratios
    ql = qoi_from_dofs(ops, u, f[0], L_acc=2 * q["L_acc_m"][0])
    assert ql["E_acc_Vm"][0] == pytest.approx(q["E_acc_Vm"][0] / 2)
    assert ql["Epk_Eacc"][0] == pytest.approx(2 * q["Epk_Eacc"][0])
    # β < 1 lowers the transit-time factor of this (constant E_z) mode
    assert qoi_from_dofs(ops, u, f[0], beta=0.8)["T_transit"][0] < q["T_transit"][0]


@pytest.mark.gmsh
@pytest.mark.parametrize("field", ["E", "H"])
def test_vectorised_equals_loop(cases, field):
    geom, Y, f, _ = cases["pillbox", field]
    ops = build_qoi_operators(geom["X"], geom["tets"], geom["edges"], geom["scale"], geom["center"], field,
                              M=geom["M"])
    sgn = np.array([1.0, -2.0, 0.5])
    qv = qoi_from_dofs(ops, Y * sgn, f)
    qr = qoi_from_dofs(ops, Y, f, Rs=np.array([1e-3, 2e-3, 3e-3]))
    for j in range(Y.shape[1]):
        q1 = qoi_from_dofs(ops, Y[:, j], f[j])
        q2 = qoi_from_dofs(ops, Y[:, j], f[j], Rs=(j + 1) * 1e-3)
        for k in QOI_KEYS:
            assert qv[k].shape == (3,) and q1[k].shape == (1,)
            np.testing.assert_allclose(qv[k][j], q1[k][0], rtol=1e-12, err_msg=k)
            np.testing.assert_allclose(qr[k][j], q2[k][0], rtol=1e-12, err_msg=k)
    # cavity_qoi forwards build / qoi kwargs and accepts a PKL geometry without M
    g2 = {k: v for k, v in geom.items() if k != "M"}
    qc = cavity_qoi(g2, Y, f / 1e9, n_axis=401, convention="circuit")
    np.testing.assert_allclose(qc["R_over_Q_ohm"], qv["R_over_Q_ohm"] / 2, rtol=1e-10)


@pytest.mark.gmsh
def test_sorted_and_skfem_edge_order_agree(cases):
    """Operators built on the canonical sorted edges and on a permuted DOF order give the same QoI."""
    geom, Y, f, _ = cases["box", "H"]
    perm = np.random.default_rng(0).permutation(len(geom["edges"]))
    e2 = geom["edges"][perm]
    q1 = cavity_qoi(geom, Y[:, 0], f[0] / 1e9)
    g2 = {**{k: geom[k] for k in ("X", "tets", "scale", "center", "field")}, "edges": e2}
    q2 = cavity_qoi(g2, Y[perm, 0], f[0] / 1e9)
    for k in QOI_KEYS:
        np.testing.assert_allclose(q2[k], q1[k], rtol=1e-10, err_msg=k)
    assert np.array_equal(edges_from_tets(geom["tets"]), geom["edges"])
    M = assemble_n0(geom["X"], geom["tets"])[0]
    assert abs(M - geom["M"]).max() < 1e-14
