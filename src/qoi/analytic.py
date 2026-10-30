"""Closed-form figures of merit of reference PEC cavities + their FE reference solutions.

All values in the conventions of docs/24 §0.2 (peak-amplitude fields, U = 1 J, β = 1 default,
'linac' R/Q = V²/(ωU)), returned with the same keys as src.qoi.operators.qoi_from_dofs.

Pillbox TM010 (radius R, length L, axis z, L < 2.03 R so TM010 is the fundamental)
-------------------------------------------------------------------------------------
    E_z = E0 J0(k r),   H_φ = −j (E0/η) J1(k r),   k = j01/R,  j01 = 2.40483,  η = √(μ0/ε0)
    U   = ½ε0 E0² L 2π ∫0^R J0²(kr) r dr = (π/2) ε0 E0² L R² J1²(j01)
          [∫0^R J0²(kr) r dr = (R²/2) J1²(j01)]
    P_c = ½ R_s (E0/η)² [ J1²(j01)·2πRL  (side wall)  +  2·2π ∫0^R J1²(kr) r dr (end caps) ]
        = π R_s (E0/η)² J1²(j01) R (L + R)
          [∫0^R J1²(kr) r dr = (R²/2) J1²(j01) since J1'(j01) = −J1(j01)/j01]
    Q0  = ωU/P_c = η j01 L / (2 R_s (R + L)),     G = Q0 R_s = η j01 L / (2 (R + L))
    V   = E0 |∫0^L e^{j k z/β} dz| = E0 L T,      T = sin(x)/x,  x = k L/(2β)
    R/Q = V²/(ωU) = 2 η L T² / (π j01 R J1²(j01))           (≈ 196 Ω for kL = π, β = 1)
    E_pk = E0 (end-cap centre, r = 0) ⇒ E_pk/E_acc = 1/T  (E_acc = V/L = E0 T)
    B_pk = μ0 (E0/η) max J1 = (E0/c) J1(j'11) = 0.58187 E0/c at r = (j'11/j01) R = 0.766 R on the
           end caps (the side wall only has J1(j01) = 0.519) ⇒ B_pk/E_acc = 0.58187/(c T)
           = 1.941/T mT/(MV/m)

Rectangular box, fundamental mode with E ∥ z (beam axis) — "TE101" in the waveguide convention
(E along the shortest side), = TM110 with respect to z.  Box [−a/2, a/2]×[−b/2, b/2]×[0, d],
d < a, b (so this is the fundamental), corner coordinates (x', y') = (x + a/2, y + b/2):
------------------------------------------------------------------------------------------------
    E_z = E0 sin(πx'/a) sin(πy'/b),   k² = (π/a)² + (π/b)²
    H   = (j/(ωμ0)) curl E:  H_x = jA (π/b) sin(πx'/a) cos(πy'/b),  H_y = −jA (π/a) cos(πx'/a) sin(πy'/b),
          A = E0/(ωμ0)
    U   = ½ε0 E0² (a/2)(b/2) d = ε0 E0² a b d / 8
    P_c = ½ R_s A² [ a b k²/2  (z = 0, d walls, both H components)
                     + π² a d/b²  (y walls: H_x)  +  π² b d/a²  (x walls: H_y) ]
    Q0  = k³ η a b d / (4 R_s [a b k²/2 + π² (a d/b² + b d/a²)])
          (= Pozar's TE101 formula (k a d)³ b η / (2π² R_s (2a³b + 2bd³ + a³d + ad³)) with his
          (a, b, d) = (our a, d, b))
    V   = E0 d T  (on the axis x' = a/2, y' = b/2, E_z = E0),   T = sin(kd/(2β))/(kd/(2β))
    R/Q = V²/(ωU) = 8 η d T² / (k a b)
    E_pk = E0 (centre of the z = 0, d walls) ⇒ E_pk/E_acc = 1/T
    H_pk = A π / min(a, b)  (the bilinear max of |H_t|² over the walls) ⇒ B_pk = π E0/(ω min(a, b))

reference_case(...) meshes these cavities with gmsh (OCC, as src/data_gen/dataset_generator_3d) and
solves them with the Whitney N0 E eigensolver of fe_modes, returning the PKL-convention geometry (normalised
mesh, sorted edges, M) and unit-M-norm DOFs — the inputs of src.qoi.cavity_qoi.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.constants import epsilon_0 as EPS0, mu_0 as MU0
from scipy.special import j1, jn_zeros

from src.qoi.operators import C0, SIGMA_CU, surface_resistance

ETA0 = float(np.sqrt(MU0 / EPS0))
J01 = float(jn_zeros(0, 1)[0])            # 2.404826
J11P = 1.8411837813406593                  # first zero of J1' (max of J1)
J1MAX = float(j1(J11P))                    # 0.581865


def transit_time_factor(k, length, beta=1.0):
    """T = sin(x)/x, x = k·length/(2β), for a constant on-axis E_z over `length`."""
    x = k * length / (2.0 * beta)
    return float(np.sinc(x / np.pi))


def _finish(f, Rs, Pc, V, L, Epk, Bpk, T, convention):
    w = 2 * np.pi * f
    fac = 1.0 if convention == "linac" else 0.5
    Q0 = w * 1.0 / Pc
    Eacc = V / L
    return {"f_Hz": f, "Rs_ohm": Rs, "U_J": 1.0, "P_c_W": Pc, "Q0": Q0, "G_ohm": Q0 * Rs, "V_acc_V": V,
            "T_transit": T, "R_over_Q_ohm": fac * V ** 2 / w, "R_sh_ohm": fac * V ** 2 / Pc, "L_acc_m": L,
            "E_acc_Vm": Eacc, "E_pk_Vm": Epk, "B_pk_T": Bpk, "Epk_Eacc": Epk / Eacc,
            "Bpk_Eacc_mT_per_MVm": 1e9 * Bpk / Eacc}


def pillbox_tm010(R, L, sigma=SIGMA_CU, Rs=None, beta=1.0, convention="linac"):
    """Closed-form TM010 figures of merit of a PEC pillbox (radius R, length L [m]); U = 1 J.
    Keys as qoi_from_dofs (floats).  Formulas: module docstring."""
    k = J01 / R
    f = C0 * k / (2 * np.pi)
    w = 2 * np.pi * f
    Rs = float(surface_resistance(f, sigma)) if Rs is None else float(Rs)
    J1 = float(j1(J01))
    E0 = np.sqrt(2.0 / (np.pi * EPS0 * L * R ** 2 * J1 ** 2))          # U = 1 J
    Pc = np.pi * Rs * (E0 / ETA0) ** 2 * J1 ** 2 * R * (L + R)
    T = transit_time_factor(w / C0, L, beta)
    V = E0 * L * T
    return _finish(f, Rs, Pc, V, L, E0, MU0 * E0 / ETA0 * J1MAX, T, convention)


def box_te101(a, b, d, sigma=SIGMA_CU, Rs=None, beta=1.0, convention="linac"):
    """Closed-form figures of merit of the E ∥ z fundamental ('TE101' / TM110 w.r.t. z) of the PEC
    box a × b × d [m] (x, y, z; beam axis = box axis, d < a, b); U = 1 J.  Formulas: module docstring."""
    k = np.pi * np.hypot(1.0 / a, 1.0 / b)
    f = C0 * k / (2 * np.pi)
    w = 2 * np.pi * f
    Rs = float(surface_resistance(f, sigma)) if Rs is None else float(Rs)
    E0 = np.sqrt(8.0 / (EPS0 * a * b * d))
    A = E0 / (w * MU0)
    Pc = 0.5 * Rs * A ** 2 * (a * b * k ** 2 / 2 + np.pi ** 2 * (a * d / b ** 2 + b * d / a ** 2))
    T = transit_time_factor(w / C0, d, beta)
    V = E0 * d * T
    return _finish(f, Rs, Pc, V, d, E0, MU0 * A * np.pi / min(a, b), T, convention)


# ─────────────────────────── FE reference solutions ────────────

def reference_mesh(shape, dims, h):
    """gmsh OCC mesh (nodes [m], tets) of 'pillbox' dims = (R, L): cylinder r ≤ R, 0 ≤ z ≤ L;
    or 'box' dims = (a, b, d): [−a/2, a/2]×[−b/2, b/2]×[0, d].  Beam axis x = y = 0 in both."""
    import gmsh
    from src.data_gen.dataset_generator_3d import _mesh_current_model
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("General.NumThreads", 1)
    try:
        gmsh.clear()
        gmsh.model.add(f"qoi_ref_{shape}")
        occ = gmsh.model.occ
        if shape == "pillbox":
            R, L = dims
            occ.addCylinder(0, 0, 0, 0, 0, L, R)
        elif shape == "box":
            a, b, d = dims
            occ.addBox(-a / 2, -b / 2, 0, a, b, d)
        else:
            raise ValueError(f"unknown reference shape {shape!r}")
        occ.synchronize()
        return _mesh_current_model(h)
    finally:
        gmsh.clear()
        if own:
            gmsh.finalize()


def _spd_factor(A):
    """SuperLU factorisation of an SPD matrix: symmetric minimum-degree ordering, no pivoting (~20x faster
    than the default for these 3D N0 matrices at identical fill)."""
    return spla.splu(sp.csc_matrix(A), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                     options=dict(SymmetricMode=True))


def _projected_shift_invert(K, M, Gp, k, tol=1e-10):
    """k lowest eigenpairs of K u = lam M u on the M-orthogonal complement of range(Gp) (full column rank):
    shift-invert with sigma < 0 (K - sigma M SPD) and the projection P = I - Gp Kp^-1 GpT M after every
    solve, so gradient fields become theta = 0 and never appear among the returned modes."""
    lam_scale = K.diagonal().mean() / M.diagonal().mean()
    sigma = -1e-2 * lam_scale
    lu = _spd_factor(K - sigma * M)
    if Gp.shape[1] == 0:                                   # no potentials (a mesh without interior vertices)
        def P(x):
            return x
    else:
        MG = (M @ Gp).tocsr()
        lup = _spd_factor(Gp.T @ MG)                       # Kp: SPD potential Laplacian

        def P(x):
            return x - Gp @ lup.solve(MG.T @ x)

    OP = spla.LinearOperator(K.shape, matvec=lambda x: P(lu.solve(x)), dtype=float)
    v0 = P(np.random.default_rng(0).standard_normal(K.shape[0]))
    vals, vecs = spla.eigsh(K, k=k, M=M, sigma=sigma, OPinv=OP, which="LM", v0=v0, tol=tol,
                            ncv=min(K.shape[0] - 1, max(2 * k + 1, 30)))
    o = np.argsort(vals)
    return vals[o], vecs[:, o], lam_scale


def fe_modes(nodes, tets, k=1, component_potentials=True, tol=1e-10):
    """Whitney N0 E-formulation reference eigensolve on the converter-normalised mesh, in the PKL
    conventions: wall edges removed (essential n x E = 0), the curl kernel (interior-vertex gradients + one
    potential per extra boundary component, dataset_converter_3d.e_potential_matrix) projected out.
    Returns (geom dict {X, tets, edges, scale, center, M, fe}, Y [Ne,k] unit M-norm in sorted-edge order
    (wall rows 0), f_hz [k]); geom['fe'] = dict(K, G, n_pot, bnd_edge, vals (normalised lam), topo,
    lam_scale) for checks.  component_potentials=False (validation only) leaves the b2 fields of isolated
    conductors in the spectrum: the zero check then raises."""
    from src.data.dataset_converter_3d import assemble_n0, e_potential_matrix, normalise
    tets = np.asarray(tets, dtype=np.int64)
    X, center, scale = normalise(nodes, tets)
    M, K, edges = assemble_n0(X, tets)
    G, bnd, n_pot, topo = e_potential_matrix(edges, len(X), tets, component_potentials=component_potentials)
    free = np.flatnonzero(~bnd)
    vals, vf, lam_scale = _projected_shift_invert(K[free][:, free].tocsr(), M[free][:, free].tocsr(),
                                                  G[free][:, :n_pot].tocsr(), k, tol)
    if not np.all(np.isfinite(vals)) or vals.min() <= 1e-6 * lam_scale:
        raise RuntimeError(f"zero / non-physical eigenvalue in {vals} (scale {lam_scale:.3g})")
    Y = np.zeros((len(edges), k))
    Y[free] = vf
    Y /= np.sqrt(np.einsum("ij,ij->j", Y, M @ Y))[None]
    f = C0 * np.sqrt(vals) / (2 * np.pi * scale)
    geom = {"X": X, "tets": tets, "edges": edges, "scale": scale, "center": center, "M": M,
            "fe": dict(K=K, G=G, n_pot=n_pot, bnd_edge=bnd, vals=vals, topo=topo, lam_scale=float(lam_scale))}
    return geom, Y, f


def reference_case(shape, dims, h, k=1):
    """(geom, Y [Ne,k], f_hz [k], analytic dict) of a reference cavity at mesh size h [m]."""
    nodes, tets = reference_mesh(shape, dims, h)
    geom, Y, f = fe_modes(nodes, tets, k)
    ref = pillbox_tm010(*dims) if shape == "pillbox" else box_te101(*dims)
    return geom, Y, f, ref
