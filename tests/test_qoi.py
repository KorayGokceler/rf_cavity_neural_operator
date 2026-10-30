"""Cavity figures of merit (src/qoi, docs/24_CAVITY_QOI.md): closed forms, the QoI contract and the rescaling of
src/qoi/figures.py, and the p2 label path (highorder.mode_qoi) against the closed forms on coarse curved pillboxes.
The tight comparison on converged meshes is scripts/validate_field_labels.py (docs/24 §3)."""
import numpy as np
import pytest
from scipy.constants import mu_0 as MU0

from src.qoi import QOI_KEYS, QOI_LABELS
from src.qoi import analytic as an
from src.qoi.figures import acceleration_length, axis_line, beam_axis, figures_of_merit, surface_resistance

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


# ─────────────────────────── figures.py: contract, rescaling, conventions ──────

def _raw(f=1.3e9, Rs=8e-3):
    """Raw functionals of a mode at an arbitrary amplitude (SI) for figures_of_merit."""
    return dict(f=[f, 2 * f], Rs=[Rs, Rs], U_raw=[2.0e-3, 5.0e-3], Pc_raw=[1.5e-4, 4.0e-4], V_raw=[40.0, 90.0],
                V0_raw=[50.0, 100.0], Epk_raw=[3.0e4, 8.0e4], Bpk_raw=[2.0e-4, 5.0e-4], L=0.1)


def test_figures_of_merit_contract_and_amplitude_invariance():
    r = _raw()
    q = figures_of_merit(**r)
    assert set(q) == set(QOI_KEYS) and all(v.shape == (2,) for v in q.values())
    np.testing.assert_array_equal(q["U_J"], 1.0)
    w = 2 * np.pi * np.asarray(r["f"])
    np.testing.assert_allclose(q["P_c_W"] * q["Q0"], w, rtol=1e-12)                 # omega U / P_c with U = 1 J
    np.testing.assert_allclose(q["R_over_Q_ohm"], q["V_acc_V"] ** 2 / w, rtol=1e-12)
    np.testing.assert_allclose(q["R_sh_ohm"], q["R_over_Q_ohm"] * q["Q0"], rtol=1e-12)
    np.testing.assert_allclose(q["G_ohm"], q["Q0"] * q["Rs_ohm"], rtol=1e-12)
    np.testing.assert_allclose(q["T_transit"], np.asarray(r["V_raw"]) / np.asarray(r["V0_raw"]), rtol=1e-12)
    # amplitude a: U ~ a^2, P ~ a^2, V ~ a, E_pk ~ a, B_pk ~ a -> every figure of merit is unchanged
    a = 37.5
    r2 = dict(r, U_raw=a ** 2 * np.asarray(r["U_raw"]), Pc_raw=a ** 2 * np.asarray(r["Pc_raw"]),
              V_raw=a * np.asarray(r["V_raw"]), V0_raw=a * np.asarray(r["V0_raw"]),
              Epk_raw=a * np.asarray(r["Epk_raw"]), Bpk_raw=a * np.asarray(r["Bpk_raw"]))
    for k, v in figures_of_merit(**r2).items():
        np.testing.assert_allclose(v, q[k], rtol=1e-12, err_msg=k)


def test_figures_of_merit_conventions():
    r = _raw()
    q, qc = figures_of_merit(**r), figures_of_merit(**r, convention="circuit")
    for k in QOI_KEYS:                                                  # circuit: R/Q and R_sh halved, rest equal
        fac = 0.5 if k in ("R_over_Q_ohm", "R_sh_ohm") else 1.0
        np.testing.assert_allclose(qc[k], fac * q[k], rtol=1e-12, err_msg=k)
    ql = figures_of_merit(**dict(r, L=0.2))                             # L_acc only rescales E_acc and the peak ratios
    np.testing.assert_allclose(ql["E_acc_Vm"], q["E_acc_Vm"] / 2, rtol=1e-12)
    np.testing.assert_allclose(ql["Epk_Eacc"], 2 * q["Epk_Eacc"], rtol=1e-12)
    np.testing.assert_allclose(ql["Q0"], q["Q0"], rtol=1e-12)
    # copper default: R_s = sqrt(pi f mu0 / sigma)
    assert surface_resistance(1.3e9) == pytest.approx(np.sqrt(np.pi * 1.3e9 * MU0 / 5.8e7), rel=1e-14)


def test_beam_axis_per_family():
    X = np.array([[0, 0, -1.0], [0, 0, 0.5], [1, 0, 0], [0, 1, 0]])
    assert beam_axis("elliptical", X, 0.1, np.zeros(3)) == {}
    ax = beam_axis("hwr", X, 0.1, np.array([0, 0, 0.2]))
    assert ax["axis_dir"] == "x" and ax["axis_point"][2] == pytest.approx(0.2 + 0.1 * (-0.25))
    # ridged_box sits in [0, a] x [0, b] x [0, d]: the axis runs along x through the middle of the gap
    p = {"a": 0.08, "b": 0.05, "d": 0.04, "ridge_h": 0.01, "double": 1.0, "gap": 0.02}
    ax = beam_axis("ridged_box", X, 1.0, np.zeros(3), p)
    assert ax["axis_dir"] == "x" and ax["axis_point"] == pytest.approx((0.0, 0.025, 0.02))
    assert beam_axis("ridged_box", X, 1.0, np.zeros(3), {**p, "double": 0.0, "ridge_h": 0.02})["axis_point"][2] == 0.01
    d, p0 = axis_line(**ax)
    np.testing.assert_allclose(d, [1, 0, 0])
    np.testing.assert_allclose(p0, ax["axis_point"])
    d, p0 = axis_line(axis_dir=(0, 0, -2.0), axis_point=(0.1, 0.2, 0.3))       # a 3-vector is normalised
    np.testing.assert_allclose(d, [0, 0, -1])


def test_acceleration_length_excludes_the_beam_pipes():
    from src.data_gen import cavity_shapes as cs
    p, halves, walls = cs.draw_elliptical(np.random.default_rng(4), n_cells=3)
    assert acceleration_length("elliptical", p) == pytest.approx(cs.elliptical_chain(halves, walls, 3)[1])
    assert acceleration_length("pillbox_pipes", {"Lc": 0.03, "Lpipe": 0.02}) == 0.03
    assert acceleration_length("composite", {"R0": 0.03}) is None and acceleration_length("elliptical", None) is None


def test_label_beam_axis_follows_the_family():
    """field_labels._beam_axis (the axis and accelerating length mode_qoi gets) on the straight-axis pillbox mesh:
    z through the origin for pillbox_pipes (L_acc = Lc), the transversal x line at mid-length for hwr (L_acc = 2 Ro)."""
    pytest.importorskip("ngsolve")
    from netgen.occ import Axes, Cylinder, OCCGeometry, Z
    from ngsolve import Mesh

    from src.data_gen import field_labels as fl
    cyl = Cylinder(Axes((0, 0, -0.05), Z), r=0.1, h=0.1)
    cyl.faces.name = "wall"
    mesh = Mesh(OCCGeometry(cyl).GenerateMesh(maxh=0.08))
    d, p0, L = fl._beam_axis("pillbox_pipes", mesh, {"Lc": 0.1})
    np.testing.assert_allclose(d, [0, 0, 1])
    np.testing.assert_allclose(p0, [0, 0, 0])
    assert L == pytest.approx(0.1)
    d, p0, L = fl._beam_axis("hwr", mesh, {"Ro": 0.1})
    np.testing.assert_allclose(d, [1, 0, 0])
    assert p0[2] == pytest.approx(0.0, abs=1e-12) and L == pytest.approx(0.2)


# ─────────────────────────── p2 label path vs closed forms ─────────────────────

def _pillbox_mode(maxh):
    """(QoI dict of the TM010 mode, mesh) of a curved (curve 3) netgen pillbox R = L = 0.1 m, p2 modes."""
    pytest.importorskip("ngsolve")
    from netgen.occ import Axes, Cylinder, OCCGeometry, Z
    from ngsolve import Mesh

    from src.data_gen import highorder as ho
    R, L = PILLBOX
    cyl = Cylinder(Axes((0, 0, -L / 2), Z), r=R, h=L)
    cyl.faces.name = "wall"
    mesh = Mesh(OCCGeometry(cyl).GenerateMesh(maxh=maxh))
    mesh.Curve(3)
    res = ho.solve_modes(mesh, 1, order=2)
    return ho.mode_qoi(mesh, res["gfs"][:1], res["f_hz"][:1], L_acc=L), mesh


def test_mode_qoi_matches_the_closed_forms_on_coarse_pillboxes():
    """highorder.mode_qoi (the label path's figures of merit, from the p2 fields) on 109 and 545 curved tets.
    Measured error vs the closed form (maxh 0.05 / 0.03): f +1e-3 / +1e-4, T -1e-3 / -1e-4, R/Q +7e-3 / -2e-4,
    Epk/Eacc +3.5e-2 / -4e-4, Q0 -5.9e-2 / -2.5e-2, Bpk/Eacc +7e-2 / +6e-2 (the wall quantities converge slower:
    they need curl E on the wall; the adaptive labels reach 1e-2 on them, scripts/validate_field_labels.py)."""
    ref = an.pillbox_tm010(*PILLBOX)
    coarse, m1 = _pillbox_mode(0.05)
    fine, m2 = _pillbox_mode(0.03)
    assert m1.ne < 200 < 400 < m2.ne < 1000
    assert set(fine) >= set(QOI_KEYS)
    err = {k: float(np.asarray(fine[k])[0] / ref[k] - 1) for k in QOI_KEYS if k not in ("U_J",)}
    np.testing.assert_array_equal(fine["U_J"], 1.0)
    assert abs(err["f_Hz"]) < 2e-3 and abs(err["T_transit"]) < 2e-3
    assert abs(err["R_over_Q_ohm"]) < 2e-3 and abs(err["Epk_Eacc"]) < 3e-3
    assert abs(err["Q0"]) < 4e-2 and abs(err["G_ohm"]) < 4e-2 and abs(err["R_sh_ohm"]) < 4e-2
    assert abs(err["Bpk_Eacc_mT_per_MVm"]) < 0.1
    assert fine["L_acc_m"][0] == pytest.approx(PILLBOX[1], rel=1e-12)
    # mesh refinement helps the wall losses (Q0 from curl E on the wall)
    assert abs(fine["Q0"][0] / ref["Q0"] - 1) < 0.6 * abs(coarse["Q0"][0] / ref["Q0"] - 1)
    # internal consistency of the dict (exact identities of figures_of_merit)
    w = 2 * np.pi * fine["f_Hz"][0]
    assert fine["P_c_W"][0] * fine["Q0"][0] == pytest.approx(w, rel=1e-12)
    assert fine["R_sh_ohm"][0] == pytest.approx(fine["R_over_Q_ohm"][0] * fine["Q0"][0], rel=1e-12)
    # the on-axis profile used for T, V and R/Q at any beta is returned with the figures of merit
    assert fine["axis_ez"].shape == (1, len(fine["axis_t"]))
