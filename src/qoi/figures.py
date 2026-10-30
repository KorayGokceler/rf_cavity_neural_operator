"""Cavity figures of merit shared by the label path (src.data_gen.highorder.mode_qoi) and the closed forms
(src.qoi.analytic): the QoI keys and conventions of docs/24 §0, the copper surface resistance, the beam axis and
accelerating length per generator family, and the rescaling of the raw functionals of a mode to U = 1 J.

The raw functionals themselves (stored energy, wall loss, transit-voltage, surface peaks) are evaluated on the
HCurl(p2) fields of the curved mesh by src.data_gen.highorder.mode_qoi.  Contract: docs/24 §0.
"""
import numpy as np
from scipy.constants import mu_0 as MU0

C0 = 299792458.0
SIGMA_CU = 5.8e7

QOI_LABELS = ('Q0', 'G_ohm', 'R_over_Q_ohm', 'R_sh_ohm', 'T_transit', 'Epk_Eacc',
              'Bpk_Eacc_mT_per_MVm')

QOI_KEYS = ('f_Hz', 'Rs_ohm', 'U_J', 'P_c_W', 'Q0', 'G_ohm', 'V_acc_V', 'T_transit', 'R_over_Q_ohm',
            'R_sh_ohm', 'L_acc_m', 'E_acc_Vm', 'E_pk_Vm', 'B_pk_T', 'Epk_Eacc', 'Bpk_Eacc_mT_per_MVm')


# ─────────────────────────── helpers ───────────────────────────

def surface_resistance(f_hz, sigma=SIGMA_CU):
    """Normal-conductor surface resistance R_s = √(π f μ0 / σ) = √(ωμ0 / 2σ) [Ω]."""
    return np.sqrt(np.pi * np.asarray(f_hz, dtype=np.float64) * MU0 / sigma)


# ─────────────────────────── beam axis ─────────────────────────

def axis_line(axis_xy=(0.0, 0.0), axis_dir=None, axis_point=None):
    """(unit direction d, physical point p) of the beam axis: default the line (x, y) = axis_xy
    along z; axis_dir 'x' | 'y' | 'z' or a 3-vector, axis_point a physical point on it [m]."""
    if axis_dir is None:
        axis_dir = "z"
    if isinstance(axis_dir, str):
        d = np.eye(3)["xyz".index(axis_dir.lower())]
    else:
        d = np.asarray(axis_dir, dtype=np.float64).reshape(3)
        d = d / np.linalg.norm(d)
    if axis_point is None:
        axis_point = (float(axis_xy[0]), float(axis_xy[1]), 0.0)
    return d, np.asarray(axis_point, dtype=np.float64).reshape(3)


# Beam axis per generator family when it is not the default line x = y = 0 along z (the families are
# built with z = beam axis, docs/24 §0.2) — hwr: the coaxial line runs along z, the beam crosses it
# transversally along x at mid-length (cavity_shapes.build_hwr).
def beam_axis(shape_type, X, scale, center, params=None):
    """{'axis_dir', 'axis_point'} kwargs (axis_line) for a generator family ({} = z through the
    origin).  params: the family's geom_params [m] (needed for ridged_box and box, whose solids start at
    the origin; without them those keep the default)."""
    st, p = str(shape_type), params or {}
    if st == "hwr":
        X = np.asarray(X, dtype=np.float64)
        zmid = 0.5 * (X[:, 2].min() + X[:, 2].max()) * float(scale) + float(np.asarray(center).reshape(3)[2])
        return {"axis_dir": "x", "axis_point": (0.0, 0.0, zmid)}
    if st == "ridged_box" and "gap" in p:       # along x through the gap (beam pipes, when drawn, lie there)
        zc = (p["ridge_h"] if p.get("double", 0.0) > 0.5 else 0.0) + p["gap"] / 2
        return {"axis_dir": "x", "axis_point": (0.0, p["b"] / 2, zc)}
    if st in ("box", "calib_box") and "a" in p:
        return {"axis_dir": "z", "axis_point": (p["a"] / 2, p["b"] / 2, 0.0)}
    return {}


def acceleration_length(shape_type, params):
    """Standard accelerating length [m] (E_acc = V / L): the cavity's inner length along the beam axis,
    beam pipes excluded — the cells for elliptical (N·βλ/2 at the design point), the cell for
    pillbox / re-entrant cells, the tank for spoke / dtl, the outer diameter for hwr.  None (→ axis
    chord) for families without a beam (composite, freeform, junction, coax_qw) or without params."""
    st, p = str(shape_type), params or {}
    if st in ("elliptical", "elliptical_long"):
        e1, e2, n, L = (p.get(k) for k in ("e1_L", "e2_L", "n_cells", "L"))
        return None if None in (e1, e2, n, L) else e1 + e2 + 2 * (n - 1) * L
    if st == "hwr":
        return 2 * p["Ro"] if "Ro" in p else None
    key = {"pillbox_pipes": "Lc", "reentrant": "Lc", "pillbox_port": "Lc", "spoke": "Lt", "dtl": "Lt",
           "ridged_box": "a", "box": "d", "calib_box": "d"}.get(st)
    return None if p.get(key) is None else float(p[key])


# ─────────────────────────── figures of merit ──────────────────

def figures_of_merit(f, Rs, U_raw, Pc_raw, V_raw, V0_raw, Epk_raw, Bpk_raw, L, convention="linac"):
    """QOI_KEYS dict [K] from the raw functionals of a mode at arbitrary amplitude (SI units):
    stored energy U_raw [J], wall loss Pc_raw [W], |V| with / without the transit phase V_raw,
    V0_raw [V], surface peaks Epk_raw [V/m], Bpk_raw [T], accelerating length L [m].  Rescaled to
    U = 1 J.  Used by src.data_gen.highorder.mode_qoi (NGSolve fields)."""
    f, Rs, U_raw, Pc_raw, V_raw, V0_raw, Epk_raw, Bpk_raw = (
        np.atleast_1d(np.asarray(x, dtype=np.float64)) for x in (f, Rs, U_raw, Pc_raw, V_raw, V0_raw,
                                                                 Epk_raw, Bpk_raw))
    K = len(U_raw)
    w = 2.0 * np.pi * f
    a = 1.0 / np.sqrt(U_raw)                                               # amplitude → U = 1 J
    fac = 1.0 if convention == "linac" else 0.5
    P_c = a ** 2 * Pc_raw
    V = a * V_raw
    L = np.broadcast_to(np.asarray(L, dtype=np.float64), (K,)).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        Q0 = w / P_c
        E_acc = V / L
        E_pk, B_pk = a * Epk_raw, a * Bpk_raw
        return {
            "f_Hz": f, "Rs_ohm": Rs, "U_J": np.ones(K), "P_c_W": P_c, "Q0": Q0, "G_ohm": Q0 * Rs,
            "V_acc_V": V, "T_transit": V_raw / V0_raw, "R_over_Q_ohm": fac * V ** 2 / w,
            "R_sh_ohm": fac * V ** 2 / P_c, "L_acc_m": L, "E_acc_Vm": E_acc, "E_pk_Vm": E_pk,
            "B_pk_T": B_pk, "Epk_Eacc": E_pk / E_acc, "Bpk_Eacc_mT_per_MVm": 1e9 * B_pk / E_acc,
        }
