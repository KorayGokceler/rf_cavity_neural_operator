"""3D PEC cavity eigenmode dataset generator (lowest-order Nédélec N0, E field).

Physics (docs/18 §1.4, §2.2; docs/19 "E formülasyonu"), f = c·√λ/(2π), skfem ``ElementTetN0``:

Find E ∈ H0(curl, Ω):  ∫ curl E · curl v = k² ∫ E · v  for all
v ∈ H0(curl, Ω).  n × E = 0 on the PEC wall is ESSENTIAL: the DOFs of the wall edges (edges of
boundary faces) are removed.  Kernel of curl in the discrete H0(curl) = {Gφ : φ P1, φ = 0 on the
first boundary component, constant on every other one}: the interior-vertex gradients plus one
potential per extra boundary component (the b2 harmonic Dirichlet fields of isolated inner
conductors).  Handles (b1 > 0: spoke, half-wave coax, DTL stems, tori) add NOTHING to this kernel,
so every connected, manifold PEC cavity is admissible.  Kp = GᵀMG is SPD without pinning.

Solved with the validated research recipe (scripts/research_3d/n0lib.py, ``solve_projected``):
shift-invert with σ < 0 (K − σM SPD) and the M-orthogonal projection P = I − G Kp⁻¹ GᵀM after
every solve, so gradient fields become θ = 0 (λ = ∞) and never appear among the returned modes.

H5 layout (one group ``sample_XXXX`` per geometry):
    nodes   [Nv,3] float64   vertex coordinates [m]
    tets    [Nt,4] int64     vertex indices (the skfem MeshTet.t used for the solve)
    edges   [Ne,2] int64     edges[i] = (tail, head) of the edge carrying DOF i, in
                             skfem ``ElementTetN0`` DOF order; tail < head always
                             (skfem orients each edge from the lower to the higher
                             global vertex index)
    e_edges [Ne,K] float64   E-field N0 DOFs e_edges[i,j] = ∫_{tail→head} E_j · dl on ALL
                             edges, wall-edge rows exactly 0; M-orthonormal in physical units
    freqs   [K]    float64   GHz, ascending
    attrs: shape_type, geom_params (JSON) + one float attr per parameter, n_nodes,
           n_edges, n_tets, freq_next (GHz, mode K+1: shows whether K splits a
           degenerate cluster), div_residual, mesh_h, t_mesh, t_solve, field ('E'),
           fem_element, n_bnd_components, betti1
File attr ``metadata`` (JSON): generator args, formulation, conventions, units.
"""
import argparse
import json
import logging
import os
import sys
import time
from multiprocessing import Pool, TimeoutError as MPTimeoutError, cpu_count

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import jn_zeros, jnp_zeros

if __package__ in (None, ""):            # run as a script (python src/data_gen/dataset_generator_3d.py)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.dataset_converter_3d import boundary_topology, e_potential_matrix  # noqa: E402
from src.data_gen import cavity_shapes as cs                                     # noqa: E402

logging.getLogger('skfem').setLevel(logging.ERROR)

C0 = 299792458.0  # speed of light [m/s]
# default families: realistic RF cavities + free-form solids (src/data_gen/cavity_shapes.py);
# "pillbox", "axisym_cell", "blob" stay available via --families (v1 datasets)
FAMILIES = ("elliptical", "reentrant", "pillbox_pipes", "ridged_box", "composite", "freeform",
            "hwr", "spoke", "dtl")

# Calibration geometries [m] (non-degenerate low box spectrum; L < 2.03 R → TM010 first)
CALIB_BOX = (0.10, 0.08, 0.06)
CALIB_PILLBOX = (0.04, 0.05)  # (R, L)

ARGS = None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate 3D PEC cavity eigenmode dataset (Nédélec N0, E field).")
    p.add_argument("--h5_filename", type=str, default="rf_cavity_3d_dataset.h5", help="Output H5 filename.")
    p.add_argument("--n_total", type=int, default=100, help="Number of geometries to generate.")
    p.add_argument("--mode", type=str, default="random", choices=["random", "calibration"],
                   help="random families, or calibration (PEC box / pillbox with analytic spectra).")
    p.add_argument("--families", type=str, nargs="+", default=None, choices=ALL_FAMILIES,
                   help="Geometry families drawn uniformly in random mode (default: FAMILIES).")
    p.add_argument("--n_eigen_modes", type=int, default=6, help="Number K of physical modes stored.")
    p.add_argument("--mesh_size", type=float, default=0.12,
                   help="Target tet size relative to the characteristic length V^(1/3) of the cavity.")
    p.add_argument("--mesh_size_abs", type=float, default=None,
                   help="Absolute target tet size [m]; overrides --mesh_size.")
    p.add_argument("--seed", type=int, default=0, help="Base seed; sample s uses default_rng([seed, s]).")
    p.add_argument("--ids_file", type=str, default=None,
                   help="Text file of sample ids to generate (one per line; e.g. from scripts/active_sampling.py); "
                        "overrides --start_id/--n_total.")
    p.add_argument("--sampling", type=str, default="random", choices=["random", "sobol"],
                   help="sobol: the first 64 random draws of each sample (family, shape parameters) come from one "
                        "scrambled Sobol point per sample id (low-discrepancy coverage); later draws stay random.")
    p.add_argument("--deform_prob", type=float, default=0.0,
                   help="Probability of a smooth random deformation x → x + δ(x) of the mesh (all families).")
    p.add_argument("--deform_max", type=float, default=0.5,
                   help="Max Lipschitz constant of δ (< 1 keeps the map injective, tets positively oriented).")
    p.add_argument("--start_id", type=int, default=0,
                   help="First sample id (shards: ids start_id … start_id+n_total−1, same seed → disjoint samples).")
    p.add_argument("--resume", action="store_true",
                   help="Continue an interrupted run: append to <h5_filename>.partial and skip the ids already in "
                        "it (cluster jobs killed by the wall-time limit lose nothing).")
    p.add_argument("--n_workers", type=int, default=None,
                   help="Worker processes (default: min(cpu_count, 4); field labels: cpu_count // threads).")
    p.add_argument("--sample_timeout", type=float, default=None,
                   help="Seconds before a sample is skipped (default 300; field labels 3600).")
    p.add_argument("--max_geom_tries", type=int, default=6,
                   help="Re-draws when a random geometry fails the boolean/topology checks "
                        "(field labels: also the curved-mesh / element-budget checks).")
    # high-order field labels (docs/29): NGSolve p3 on a curved mesh, projected onto the model's p2 space
    p.add_argument("--labels", type=str, default="n0", choices=["n0", "field"],
                   help="n0: Whitney N0 labels on the gmsh mesh (default); field: curved-mesh HCurl labels "
                        "(src/data_gen/field_labels.py) for the learned-field model (model.type field3d).")
    p.add_argument("--min_fillet", type=float, default=None,
                   help="Fillet radius floor as a fraction of the cavity size (cavity_shapes.MIN_FILLET; "
                        "default 0 for n0 labels — the v2 geometries — and 0.05 for field labels).")
    p.add_argument("--label_order", type=int, default=3, help="field labels: HCurl order of the labels.")
    p.add_argument("--model_order", type=int, default=2,
                   help="field labels: HCurl order of the model space the labels are projected onto "
                        "(2: field floor ~0.2 %%, frequency ~1e-4; 1: ~6x cheaper training, ~1 %% frequency floor).")
    p.add_argument("--curve", type=int, default=3, help="field labels: geometry order of the curved mesh.")
    p.add_argument("--maxh_factor", type=float, default=3.0,
                   help="field labels: netgen maxh = maxh_factor × the N0 mesh size h (curvature sets the "
                        "element size on curved faces).")
    p.add_argument("--field_max_elements", type=int, default=30000,
                   help="field labels: curved meshes with more tets are re-drawn (training cost bound).")
    p.add_argument("--max_ndof", type=int, default=900000, help="field labels: label-space DOF guard (memory).")
    p.add_argument("--threads", type=int, default=1, help="field labels: NGSolve threads per worker.")
    p.add_argument("--no_heal", action="store_true",
                   help="field labels: skip the OCC shape healing (small edges / faces, sewing) before meshing.")
    p.add_argument("--no_adapt", action="store_true",
                   help="field labels: one solve on the curvature-sized mesh instead of adaptive refinement.")
    p.add_argument("--tol_f", type=float, default=1e-6,
                   help="field labels: adaptive stop, relative frequency change between refinements.")
    p.add_argument("--tol_q", type=float, default=1e-3,
                   help="field labels: adaptive stop, relative change of Q0 / G / R/Q / surface peaks.")
    p.add_argument("--model_max_elements", type=int, default=10000,
                   help="field labels (adaptive): the model mesh is the finest refinement level with at most "
                        "this many tets; the final-level labels are L2-projected onto it (training cost).")
    p.add_argument("--no_round_edges", action="store_true",
                   help="field labels: keep re-entrant edges sharp (by default they are filleted with radius "
                        "min_fillet × equivalent-sphere radius: singular fields there make the surface peaks "
                        "mesh-dependent).")
    args = p.parse_args(argv)
    if args.families is None:
        args.families = list(FAMILIES)
    if args.min_fillet is None:
        args.min_fillet = 0.05 if args.labels == "field" else 0.0
    if args.sample_timeout is None:
        args.sample_timeout = 3600.0 if args.labels == "field" else 300.0
    args.heal = args.labels == "field" and not args.no_heal
    args.round_edges = args.labels == "field" and not args.no_round_edges
    if args.labels == "field":
        dropped = [f for f in args.families if f in DISCRETE_BUILDERS]
        args.families = [f for f in args.families if f not in DISCRETE_BUILDERS]
        if dropped:
            print(f"field labels need a CAD solid: dropping the discrete families {dropped}")
        if not args.families and args.mode != "calibration":
            p.error("--labels field: no CAD family left in --families")
    return args


def _init_worker(args):
    global ARGS
    ARGS = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if getattr(args, "labels", "n0") == "field":
        import ngsolve
        ngsolve.SetNumThreads(max(1, int(args.threads)))


def eigenvalues_to_ghz(vals):
    """k² [1/m²] → f = c·k/(2π) [GHz]."""
    return C0 * np.sqrt(np.abs(np.asarray(vals, dtype=np.float64))) / (2 * np.pi) / 1e9


# ─────────────────────────── analytic spectra ──────────────────

def box_spectrum(a, b, d, nmax=6):
    """PEC box k² = (mπ/a)² + (nπ/b)² + (pπ/d)²: multiplicity 2 if m,n,p ≥ 1
    (TE + TM), 1 if exactly one index is 0, none if two are 0. Sorted."""
    out = []
    for m_ in range(nmax):
        for n_ in range(nmax):
            for p_ in range(nmax):
                z = (m_ == 0) + (n_ == 0) + (p_ == 0)
                if z >= 2:
                    continue
                k2 = (m_ * np.pi / a) ** 2 + (n_ * np.pi / b) ** 2 + (p_ * np.pi / d) ** 2
                out += [k2] * (2 if z == 0 else 1)
    return np.sort(out)


def pillbox_spectrum(R, L, nmax=5):
    """PEC pillbox: TM_mnp k² = (j_mn/R)² + (pπ/L)², p ≥ 0; TE_mnp (j'_mn/R)² + (pπ/L)²,
    p ≥ 1; m ≥ 1 twofold degenerate. Sorted list of (k², label). TM010: k = 2.405/R."""
    out = []
    for m_ in range(nmax):
        jz = jn_zeros(m_, nmax)
        jpz = jnp_zeros(m_, nmax) if m_ > 0 else jn_zeros(1, nmax)  # J0' = −J1
        mult = 1 if m_ == 0 else 2
        for n_ in range(nmax):
            for p_ in range(nmax):
                out += [((jz[n_] / R) ** 2 + (p_ * np.pi / L) ** 2, f"TM{m_}{n_ + 1}{p_}")] * mult
                if p_ >= 1:
                    out += [((jpz[n_] / R) ** 2 + (p_ * np.pi / L) ** 2, f"TE{m_}{n_ + 1}{p_}")] * mult
    out.sort(key=lambda t: t[0])
    return out


def analytic_k2(shape_type, params, k):
    if shape_type == "calib_box":
        return box_spectrum(params["a"], params["b"], params["d"])[:k]
    if shape_type == "calib_pillbox":
        return np.array([v for v, _ in pillbox_spectrum(params["R"], params["L"])[:k]])
    raise ValueError(shape_type)


# ─────────────────────────── geometry (gmsh OCC) ───────────────

def _rand_rotation(rng):
    """Random rotation as (axis, angle) for occ.rotate."""
    ax = rng.standard_normal(3)
    return ax / np.linalg.norm(ax), rng.uniform(0, 2 * np.pi)


def build_pillbox(occ, rng):
    R, L = rng.uniform(0.03, 0.06), rng.uniform(0.02, 0.10)
    occ.addCylinder(0, 0, 0, 0, 0, L, R)
    return "pillbox", {"R": R, "L": L}


def cell_profile(rng, n_pts=13):
    """Elliptical-cell-like meridian profile r(z) on z ∈ [0, L] (flat PEC end plates at
    z = 0, L of radius r(0), r(L) > 0, so the revolved solid has no hole)."""
    L = rng.uniform(0.04, 0.10)
    r_eq = rng.uniform(0.035, 0.06)
    r_end = r_eq * rng.uniform(0.3, 0.7)
    r_end2 = r_end * rng.uniform(0.85, 1.15)
    alpha = rng.uniform(0.6, 1.6)          # bulge shape (<1: boxy, >1: pointed)
    gamma = rng.uniform(0.8, 1.25)         # z-asymmetry (equator shift)
    amp = rng.uniform(-0.06, 0.06, 2) * r_eq
    ph = rng.uniform(0, 2 * np.pi, 2)
    s = np.linspace(0, 1, n_pts)
    sg = s ** gamma
    base = np.sin(np.pi * sg) ** alpha
    ends = r_end + (r_end2 - r_end) * s
    pert = np.sin(np.pi * s) * (amp[0] * np.sin(2 * np.pi * s + ph[0]) + amp[1] * np.sin(4 * np.pi * s + ph[1]))
    r = ends + (r_eq - ends) * base + pert
    r = np.maximum(r, 0.5 * min(r_end, r_end2))
    params = {"L": L, "r_eq": r_eq, "r_end": r_end, "r_end2": r_end2, "alpha": alpha, "gamma": gamma,
              "amp1": amp[0], "amp2": amp[1]}
    return s * L, r, params


def build_axisym_cell(occ, rng):
    z, r, params = cell_profile(rng)
    pts = [occ.addPoint(ri, 0, zi) for ri, zi in zip(r, z, strict=True)]
    a0, a1 = occ.addPoint(0, 0, 0), occ.addPoint(0, 0, z[-1])
    curves = [occ.addLine(a0, pts[0]), occ.addSpline(pts), occ.addLine(pts[-1], a1), occ.addLine(a1, a0)]
    surf = occ.addPlaneSurface([occ.addCurveLoop(curves)])
    occ.revolve([(2, surf)], 0, 0, 0, 0, 0, 1, 2 * np.pi)
    return "axisym_cell", params


def _surface_point(rng, base, dims):
    """Random point on the base solid's surface (cylinder: dims=(R, L); box: (a, b, d))."""
    if base == "cylinder":
        R, L = dims
        if rng.uniform() < L / (L + R):   # side wall (∝ 2πRL vs 2·πR²)
            phi = rng.uniform(0, 2 * np.pi)
            return np.array([R * np.cos(phi), R * np.sin(phi), rng.uniform(0.15, 0.85) * L])
        rho, phi = 0.75 * R * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        return np.array([rho * np.cos(phi), rho * np.sin(phi), L * rng.integers(0, 2)])
    dims = np.asarray(dims)
    axis = rng.integers(0, 3)
    p = rng.uniform(0.15, 0.85, 3) * dims
    p[axis] = dims[axis] * rng.integers(0, 2)
    return p


def build_blob(occ, rng):
    """Base cylinder or box with 1–3 fused/cut ellipsoid or box bumps centred on its
    surface (bump size ≤ 0.4 × smallest base dimension, so a cut cannot split it)."""
    base = "cylinder" if rng.uniform() < 0.5 else "box"
    if base == "cylinder":
        dims = (rng.uniform(0.03, 0.06), rng.uniform(0.03, 0.09))
        vol = occ.addCylinder(0, 0, 0, 0, 0, dims[1], dims[0])
        dmin = min(2 * dims[0], dims[1])
    else:
        dims = tuple(rng.uniform(0.04, 0.10, 3))
        vol = occ.addBox(0, 0, 0, *dims)
        dmin = min(dims)
    params = {"base_is_box": float(base == "box"), "d0": dims[0], "d1": dims[1],
              "d2": dims[2] if base == "box" else 0.0}
    n_b = int(rng.integers(1, 4))
    ops = []
    for i in range(n_b):
        c = _surface_point(rng, base, dims)
        semi = rng.uniform(0.15, 0.4, 3) * dmin
        if rng.uniform() < 0.6:
            t = occ.addSphere(0, 0, 0, 1.0)
            occ.dilate([(3, t)], 0, 0, 0, *semi)
            kind = 0.0
        else:
            t = occ.addBox(-semi[0], -semi[1], -semi[2], *(2 * semi))
            kind = 1.0
        ax, ang = _rand_rotation(rng)
        occ.rotate([(3, t)], 0, 0, 0, *ax, ang)
        occ.translate([(3, t)], *c)
        fuse = rng.uniform() < 0.5
        ops.append((fuse, t))
        params.update({f"b{i}_box": kind, f"b{i}_fuse": float(fuse), f"b{i}_cx": c[0], f"b{i}_cy": c[1],
                       f"b{i}_cz": c[2], f"b{i}_a": semi[0], f"b{i}_b": semi[1], f"b{i}_c": semi[2]})
    for fuse, t in ops:
        out, _ = (occ.fuse if fuse else occ.cut)([(3, vol)], [(3, t)])
        vols = [tg for d, tg in out if d == 3]
        if len(vols) != 1:
            raise RuntimeError(f"boolean produced {len(vols)} volumes")
        vol = vols[0]
    params["n_bumps"] = float(n_b)
    return "blob", params


def build_calibration(occ, s_id):
    if s_id % 2 == 0:
        a, b, d = CALIB_BOX
        occ.addBox(0, 0, 0, a, b, d)
        return "calib_box", {"a": a, "b": b, "d": d}
    R, L = CALIB_PILLBOX
    occ.addCylinder(0, 0, 0, 0, 0, L, R)
    return "calib_pillbox", {"R": R, "L": L}


BUILDERS = {"pillbox": build_pillbox, "axisym_cell": build_axisym_cell, "blob": build_blob,
            "elliptical": cs.build_elliptical, "reentrant": cs.build_reentrant,
            "pillbox_pipes": cs.build_pillbox_pipes, "ridged_box": cs.build_ridged_box,
            "composite": cs.build_composite,
            # out-of-distribution test families (cs.OOD_FAMILIES), not in the default FAMILIES
            "box": cs.build_box, "coax_qw": cs.build_coax_qw, "pillbox_port": cs.build_pillbox_port,
            "elliptical_long": cs.build_elliptical_long, "junction": cs.build_junction,
            # handle families (b1 > 0; E formulation only)
            "hwr": cs.build_hwr, "spoke": cs.build_spoke, "dtl": cs.build_dtl}
DISCRETE_BUILDERS = {"freeform": cs.freeform_surface}   # closed triangulated surface → gmsh remesh
# field labels: OCC healing (small edges / sliver faces of the box booleans) before writing the CAD;
# it breaks the 1D mesh of the composite cells, so only where netgen needs it (docs/29 §2)
HEAL_FAMILIES = ("ridged_box", "dtl")


def _discrete_model(P, F):
    """gmsh volume bounded by the closed triangulation (P, F), reparametrised so the
    surface is remeshed at the target size (gmsh STL-remesh workflow)."""
    import gmsh
    s = gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(2, s, np.arange(1, len(P) + 1), np.asarray(P, float).ravel())
    gmsh.model.mesh.addElementsByType(s, 2, [], (np.asarray(F) + 1).ravel())
    gmsh.model.mesh.classifySurfaces(np.pi, True, True, np.pi)
    gmsh.model.mesh.createGeometry()
    loop = gmsh.model.geo.addSurfaceLoop([t for _, t in gmsh.model.getEntities(2)])
    gmsh.model.geo.addVolume([loop])
    gmsh.model.geo.synchronize()


ALL_FAMILIES = [*BUILDERS, *DISCRETE_BUILDERS]


def _mesh_current_model(h):
    """Mesh the current gmsh model with linear tets at size h → (nodes [Nv,3], tets [Nt,4])."""
    import gmsh
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.2 * h)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.model.mesh.generate(3)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    et, _, conn = gmsh.model.mesh.getElements(3)
    et = [int(t) for t in et]
    if et != [4]:
        raise RuntimeError(f"expected only 4-node tets, got 3D element types {et}")
    tet = np.asarray(conn[0], dtype=np.int64).reshape(-1, 4)
    xyz = np.asarray(xyz).reshape(-1, 3)
    tags = np.asarray(tags, dtype=np.int64)
    t2r = np.full(tags.max() + 1, -1, dtype=np.int64)
    t2r[tags] = np.arange(len(tags))
    tet = t2r[tet]
    if (tet < 0).any():
        raise RuntimeError("tet references a node tag missing from getNodes()")
    used = np.unique(tet)
    remap = np.full(len(tags), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return np.ascontiguousarray(xyz[used], dtype=np.float64), np.ascontiguousarray(remap[tet])


# ─────────────────────────── topology ──────────────────────────

def mesh_topology(tets, n_nodes):
    """Returns dict(euler=V−E+F−T, n_components, n_shells (= n_bnd_components), euler_boundary,
    manifold, betti1 = 1 + b2 − χ, betti2 = n_shells − 1).  A topological ball has euler=1,
    n_components=1, n_shells=1, euler_boundary=2 (⇒ b1 = b2 = 0).  Raises on a face shared by
    more than two tets.  Boundary components: connected components of the boundary faces (sharing
    an edge or a vertex) — dataset_converter_3d.boundary_topology."""
    from scipy.sparse.csgraph import connected_components
    t = np.asarray(tets, dtype=np.int64)
    bt = boundary_topology(t, n_nodes)
    A = sp.coo_matrix((np.ones(3 * len(t)), (np.repeat(t[:, 0], 3), t[:, 1:].ravel())), shape=(n_nodes,) * 2)
    n_comp = connected_components(A, directed=False)[0]
    return dict(euler=bt["euler"], n_components=int(n_comp), n_shells=bt["n_comp"],
                euler_boundary=bt["euler_boundary"], manifold=bt["manifold"], betti1=bt["betti1"],
                betti2=bt["n_comp"] - 1, n_bnd_components=bt["n_comp"])


def is_valid_e_domain(topo):
    """E-formulation certificate: a connected, manifold mesh (any b1, any number of boundary
    shells — handles and isolated conductors are handled by the E kernel)."""
    return topo["n_components"] == 1 and topo["manifold"] and topo["betti1"] >= 0


def certify(topo):
    if not is_valid_e_domain(topo):
        raise RuntimeError(f"not a connected manifold mesh: {topo}")


# ─────────────────────────── N0 solver ─────────

def assemble_n0(nodes, tets):
    """skfem N0 assembly on all edges (no essential BC). Returns dict with K, M [Ne×Ne]
    (skfem DOF order), G [Ne×Nv] discrete gradient (G[i,head]=+1, G[i,tail]=−1),
    edges [Ne,2] (tail, head) per DOF, and the skfem mesh/basis."""
    from skfem import Basis, BilinearForm, ElementTetN0, MeshTet
    from skfem.helpers import curl, dot

    @BilinearForm
    def curlcurl(u, v, w):
        return dot(curl(u), curl(v))

    @BilinearForm
    def mass(u, v, w):
        return dot(u, v)

    mesh = MeshTet(np.ascontiguousarray(nodes.T), np.ascontiguousarray(np.asarray(tets).T))
    basis = Basis(mesh, ElementTetN0())
    K = curlcurl.assemble(basis).tocsr()
    M = mass.assemble(basis).tocsr()
    dof = basis.edge_dofs[0]
    e = mesh.edges
    edges = np.empty((basis.N, 2), dtype=np.int64)
    edges[dof] = e.T                                       # skfem orients low → high vertex index
    if not np.all(edges[:, 0] < edges[:, 1]):
        raise RuntimeError("unexpected skfem edge orientation (tail ≥ head)")
    ne = e.shape[1]
    G = sp.csr_matrix((np.r_[-np.ones(ne), np.ones(ne)], (np.r_[dof, dof], np.r_[e[0], e[1]])),
                      shape=(basis.N, mesh.p.shape[1]))
    return dict(K=K, M=M, G=G, edges=edges, mesh=mesh, basis=basis)


def spd_factor(A):
    """SuperLU factorisation of an SPD matrix: symmetric minimum-degree ordering on AᵀA+A
    and no pivoting (diag_pivot_thresh=0, SymmetricMode). ~20× faster than the default
    for these 3D N0 matrices (10k DOF: 0.35 s vs 7.2 s) with identical fill."""
    return spla.splu(sp.csc_matrix(A), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                     options=dict(SymmetricMode=True))


def _projected_shift_invert(K, M, Gp, k, tol, sigma):
    """Shared recipe: k lowest eigenpairs of K u = λ M u on the M-orthogonal complement of range(Gp)
    (Gp full column rank).  Shift-invert with σ < 0, projection P = I − Gp Kp⁻¹ GpᵀM after every
    solve.  Returns (vals, vecs, sigma, lam_scale)."""
    if sigma is None:                                      # scale-aware small negative shift
        sigma = -1e-2 * (K.diagonal().mean() / M.diagonal().mean())
    lu = spd_factor(K - sigma * M)
    if Gp.shape[1] == 0:                                   # no potentials (E on a mesh without interior
        def P(x):                                          # vertices): trivial kernel, P = I
            return x
    else:
        MG = (M @ Gp).tocsr()
        lup = spd_factor(Gp.T @ MG)                        # Kp: SPD potential Laplacian

        def P(x):
            return x - Gp @ lup.solve(MG.T @ x)

    OP = spla.LinearOperator(K.shape, matvec=lambda x: P(lu.solve(x)), dtype=float)
    v0 = P(np.random.default_rng(0).standard_normal(K.shape[0]))
    vals, vecs = spla.eigsh(K, k=k, M=M, sigma=sigma, OPinv=OP, which="LM", v0=v0, tol=tol,
                            ncv=min(K.shape[0] - 1, max(2 * k + 1, 30)))
    o = np.argsort(vals)
    return vals[o], vecs[:, o], sigma, K.diagonal().mean() / M.diagonal().mean()


def solve_e_modes(A, k, tol=1e-10, sigma=None, component_potentials=True):
    """k lowest non-zero eigenpairs of the E formulation: N0 with the PEC wall edges removed
    (essential n × E = 0), gradient kernel removed by projection with the E potentials
    (dataset_converter_3d.e_potential_matrix: interior vertices + one potential per boundary
    component except the first; Kp SPD, no pinning).  Eigenvectors are returned expanded to ALL
    edges (skfem DOF order, wall rows exactly 0), M-orthonormal.
    component_potentials=False (validation only) omits the component potentials: an isolated
    conductor (b2 > 0) then leaves a spurious λ ≈ 0 mode and the zero check raises."""
    K, M = A["K"], A["M"]
    Gfull, bnd, n_pot, topo = e_potential_matrix(A["edges"], A["mesh"].p.shape[1], A["mesh"].t.T,
                                                 component_potentials=component_potentials)
    free = np.flatnonzero(~bnd)
    Kf, Mf = K[free][:, free].tocsr(), M[free][:, free].tocsr()
    Gp = Gfull[free][:, :n_pot].tocsr()
    vals, vf, sigma, lam_scale = _projected_shift_invert(Kf, Mf, Gp, k, tol, sigma)
    vecs = np.zeros((K.shape[0], k))
    vecs[free] = vf
    # ── checks: no zero / gradient modes (E-kernel potentials), M-orthonormal ──
    Mv = M @ vecs
    div = np.abs(Gfull[:, :n_pot].T @ Mv).max(axis=0, initial=0.0) / np.abs(Mv).max(axis=0)
    if not np.all(np.isfinite(vals)) or vals.min() <= 1e-6 * lam_scale:
        raise RuntimeError(f"zero / non-physical eigenvalue in {vals} (scale {lam_scale:.3g})")
    if div.max() > 1e-6:
        raise RuntimeError(f"mode not M-orthogonal to the E-kernel gradients (div residual {div.max():.2e})")
    if np.abs(vecs.T @ Mv - np.eye(k)).max() > 1e-6:
        raise RuntimeError("eigenvectors not M-orthonormal")
    return vals, vecs, dict(sigma=sigma, div_residual=float(div.max()), lam_scale=float(lam_scale),
                            n_free=int(len(free)), n_pot=n_pot, n_bnd_components=topo["n_comp"],
                            betti1=topo["betti1"])


# ─────────────────────────── sample ────────────────────────────

def _seeds(s_id):
    return np.random.default_rng([int(ARGS.seed), int(s_id)])


def _gmsh_start(restart=False):
    """(Re)initialise gmsh in this worker.  restart=True after a failed attempt: a
    3D mesher error (e.g. "PLC Error: A segment and a facet intersect") can leave
    the session broken — every later generate(3) then returns no tets while
    gmsh.clear() does not recover it — so the worker gets a fresh session."""
    import gmsh
    if restart and gmsh.isInitialized():
        gmsh.finalize()
    if not gmsh.isInitialized():
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("General.NumThreads", 1)


class SobolRNG:
    """numpy-Generator-like stream for the shape builders: the first len(u) uniforms come from one
    scrambled Sobol point (low-discrepancy across sample ids: families and parameter ranges are
    covered evenly), later draws from the fallback generator (rejection re-draws, noise fields)."""

    def __init__(self, u, fallback):
        self.u, self.i, self.rng = np.asarray(u, float), 0, fallback

    def random(self, size=None):
        n = 1 if size is None else int(np.prod(size))
        take = min(n, len(self.u) - self.i)
        v = np.r_[self.u[self.i:self.i + take], self.rng.random(n - take)]
        self.i += take
        return float(v[0]) if size is None else v.reshape(size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return low + (np.asarray(high) - low) * self.random(size)

    def integers(self, low, high=None, size=None):
        low, high = (0, low) if high is None else (low, high)
        v = np.minimum(np.floor(low + (high - low) * np.asarray(self.random(size))), high - 1).astype(np.int64)
        return int(v) if size is None else v

    def choice(self, a, size=None, p=None):
        a = np.asarray(a)
        cdf = np.cumsum(np.full(len(a), 1.0 / len(a)) if p is None else np.asarray(p, float))
        idx = np.minimum(np.searchsorted(cdf / cdf[-1], self.random(size), side="right"), len(a) - 1)
        return a[idx]

    def standard_normal(self, size=None):
        from scipy.stats import norm
        return norm.ppf(np.clip(self.random(size), 1e-12, 1 - 1e-12))

    def permutation(self, x):
        n = x if np.isscalar(x) else len(x)
        order = np.argsort(self.random(n))
        return order if np.isscalar(x) else np.asarray(x)[order]


def _sample_rng(s_id):
    base = _seeds(s_id)
    if getattr(ARGS, "sampling", "random") != "sobol":
        return base
    from scipy.stats import qmc
    sob = qmc.Sobol(d=64, scramble=True, seed=int(ARGS.seed))
    if s_id > 0:                                   # scipy: fast_forward(0) overflows
        sob.fast_forward(int(s_id))
    return SobolRNG(sob.random(1)[0], base)


def deform_map(nodes, rng, lip_max=0.5):
    """Draw the smooth map δ(x) = Σ_m a_m sin(ω_m·x + φ_m) of smooth_deform: (om [n,3], a [n,3],
    ph [n], lip).  3–8 plane waves, wavelengths 0.25–1.5 × the bounding-box diagonal;
    ‖∇δ‖₂ ≤ Σ|a_m||ω_m| is scaled to lip ∈ [0.3, 1]·lip_max < 1."""
    X = np.asarray(nodes, float)
    diag = float(np.linalg.norm(X.max(0) - X.min(0)))
    n = int(rng.integers(3, 9))
    dirs = rng.standard_normal((n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    om = dirs * (2 * np.pi / (diag * rng.uniform(0.25, 1.5, n)))[:, None]
    a = rng.standard_normal((n, 3))
    ph = rng.uniform(0, 2 * np.pi, n)
    lip = lip_max * rng.uniform(0.3, 1.0)
    a *= lip / (np.linalg.norm(a, axis=1) * np.linalg.norm(om, axis=1)).sum()
    return om, a, ph, lip


def smooth_deform(nodes, rng, lip_max=0.5, return_map=False):
    """x → x + δ(x) (deform_map).  ‖∇δ‖₂ ≤ lip < 1, so det(I + ∇δ) > 0 everywhere: the map is
    injective, every tet keeps its orientation and the topology is unchanged.  Returns (new nodes,
    params[, (om, a, ph)] with return_map — the exact map, for a high-order mesh of the same solid)."""
    X = np.asarray(nodes, float)
    diag = float(np.linalg.norm(X.max(0) - X.min(0)))
    om, a, ph, lip = deform_map(X, rng, lip_max)
    Y = X + np.sin(X @ om.T + ph) @ a
    params = {"deform_lip": float(lip), "deform_modes": float(len(ph)),
              "deform_max_disp": float(np.linalg.norm(Y - X, axis=1).max() / diag)}
    return (Y, params, (om, a, ph)) if return_map else (Y, params)


def _draw_deform(s_id, nodes):
    """(nodes, params, map) of the sample's smooth deformation (own stream: the same δ at any mesh
    size and in every geometry re-draw), or (nodes, {}, None)."""
    rng_d = np.random.default_rng([int(ARGS.seed), int(s_id), 1])
    if ARGS.mode != "calibration" and rng_d.uniform() < getattr(ARGS, "deform_prob", 0.0):
        return smooth_deform(nodes, rng_d, getattr(ARGS, "deform_max", 0.5), return_map=True)
    return nodes, {}, None


def mesh_sample(s_id, cad_path=None, post=None):
    """Geometry + mesh of sample s_id (no eigen-solve): dict(nodes [m], tets, shape_type, params,
    h, volume, t_mesh, topo, deform).  The mesh is certified a connected manifold.  deform: the
    smooth map (om, a, ph) applied to the nodes, or None.  cad_path: also write the OCC solid there
    (BREP / STEP by extension; None for the discrete freeform family) — src.data_gen.highorder.
    post(cad_path, h, deform, shape_type, params): called once the geometry passed the checks; an exception
    re-draws the geometry like a failed check (field labels: curved mesh + element budget); its
    return value is the result's 'post'.
    Used by generate_sample_data, scripts/active_sampling.py and scripts/label_benchmark.py."""
    import gmsh
    _gmsh_start()
    cs.MIN_FILLET = float(getattr(ARGS, "min_fillet", 0.0) or 0.0)
    rng = _sample_rng(s_id)
    t0 = time.perf_counter()
    fam = "calibration" if ARGS.mode == "calibration" else \
        ARGS.families[int(rng.integers(0, len(ARGS.families)))]   # fixed across re-draws: exact family balance
    for attempt in range(ARGS.max_geom_tries):
        gmsh.clear()
        gmsh.model.add(f"rf3d_{s_id}_{attempt}")
        occ = gmsh.model.occ
        try:
            if fam in DISCRETE_BUILDERS:
                P, F, params = DISCRETE_BUILDERS[fam](rng)
                shape_type, volume = fam, cs.surface_volume(P, F)
                _discrete_model(P, F)
            else:
                shape_type, params = build_calibration(occ, s_id) if fam == "calibration" \
                    else BUILDERS[fam](occ, rng)
                occ.synchronize()
                vols = gmsh.model.getEntities(3)
                if len(vols) != 1:
                    raise RuntimeError(f"{len(vols)} volumes")
                if getattr(ARGS, "round_edges", False):          # field labels: no singular edges (docs/30 §3)
                    v0 = gmsh.model.occ.getMass(3, vols[0][1])
                    r_edge = cs.MIN_FILLET * (3 * v0 / (4 * np.pi)) ** (1 / 3)
                    _, n_round, r_used, n_sharp = cs.round_reentrant_edges(occ, vols[0][1], r_edge)
                    params.update(n_rounded_edges=float(n_round), r_rounded_edges=float(r_used),
                                  n_sharp_edges=float(n_sharp))
                    vols = gmsh.model.getEntities(3)
                    if len(vols) != 1:
                        raise RuntimeError(f"{len(vols)} volumes after rounding the re-entrant edges")
                if getattr(ARGS, "heal", False) and fam in HEAL_FAMILIES:
                    occ.healShapes(sewFaces=False, makeSolids=False)   # sew / makeSolids drop revolved solids
                    occ.synchronize()
                    vols = gmsh.model.getEntities(3)
                    if len(vols) != 1:
                        raise RuntimeError(f"{len(vols)} volumes after healing")
                volume = gmsh.model.occ.getMass(3, vols[0][1])
                if cad_path:
                    gmsh.write(str(cad_path))
            vol_div = params.pop("_vol_div", 1.0)              # multi-cell: size from the per-cell volume
            h = ARGS.mesh_size_abs or ARGS.mesh_size * (volume / vol_div) ** (1.0 / 3.0)
            h_cap = params.pop("_h_cap", None)             # smallest feature (iris, nose gap, pipe)
            if h_cap and not ARGS.mesh_size_abs:
                h = max(min(h, h_cap), 0.7 * h)            # ≤ ~3× the tets of the volume rule
            nodes, tets = _mesh_current_model(h)
            topo = mesh_topology(tets, len(nodes))
            certify(topo)
            t_mesh = time.perf_counter() - t0
            nodes, dp, dmap = _draw_deform(s_id, nodes)
            extra = post(cad_path, h, dmap, shape_type, params) if post is not None else None
            break
        except Exception as e:  # re-draw the geometry (random mode only)
            _gmsh_start(restart=True)
            if ARGS.mode == "calibration" or attempt == ARGS.max_geom_tries - 1:
                raise
            print(f"sample {s_id}: geometry attempt {attempt} rejected ({e}); re-drawing")
    params = {k: float(v) for k, v in params.items()}
    params.update(dp)
    return {"nodes": nodes, "tets": tets, "shape_type": shape_type, "params": params, "h": float(h),
            "volume": float(volume), "t_mesh": t_mesh, "topo": topo, "deform": dmap,
            "cad_path": None if fam in DISCRETE_BUILDERS else cad_path, "post": extra}


def field_settings(args):
    """src.data_gen.field_labels settings from the generator args."""
    return {"label_order": int(args.label_order), "model_order": int(args.model_order), "curve": int(args.curve),
            "maxh_factor": float(args.maxh_factor), "max_elements": int(args.field_max_elements),
            "max_ndof": int(args.max_ndof), "adaptive": not getattr(args, "no_adapt", False),
            "tol_f": float(getattr(args, "tol_f", 1e-6)), "tol_q": float(getattr(args, "tol_q", 1e-3)),
            "model_max_elements": int(getattr(args, "model_max_elements", 10000))}


def generate_field_sample(s_id):
    """Field-label sample (docs/29): the generator geometry, its curved netgen mesh and the HCurl
    labels (field_labels.label_sample) — a failed curved mesh / element budget re-draws."""
    import tempfile

    from src.data_gen import field_labels as fl
    with tempfile.TemporaryDirectory() as tmp:
        cad = os.path.join(tmp, "solid.brep")
        post = lambda path, h, dmap, st, prm: fl.label_sample(  # noqa: E731
            path, h, st, ARGS.n_eigen_modes, deform=dmap, settings=field_settings(ARGS),
            sharp_edges=int(prm.get("n_sharp_edges", 0)))
        g = mesh_sample(s_id, cad_path=cad, post=post)
    lab = g["post"]
    return {"id": s_id, "labels": "field", "field": lab, "freqs": lab["freqs"], "freq_next": lab["freq_next"],
            "shape_type": g["shape_type"], "geom_params": g["params"], "mesh_h": float(g["h"]),
            "volume": float(g["volume"]), "t_mesh": g["t_mesh"] + lab["attrs"]["t_curved_mesh"],
            "t_solve": lab["attrs"]["t_labels"], "n_bnd_components": int(g["topo"]["n_bnd_components"]),
            "betti1": int(g["topo"]["betti1"]), "size": (lab["attrs"]["n_tets"], lab["attrs"]["n_dof_model"])}


def generate_sample_data(s_id):
    import gmsh
    if getattr(ARGS, "labels", "n0") == "field":
        try:
            return generate_field_sample(s_id)
        except Exception as e:
            print(f"Error generating sample {s_id}: {e}")
            return None
        finally:
            try:
                gmsh.clear()
            except Exception:
                pass
    try:
        g = mesh_sample(s_id)
        nodes, tets, shape_type, params, h, volume, t_mesh = (
            g[k] for k in ("nodes", "tets", "shape_type", "params", "h", "volume", "t_mesh"))
        t0 = time.perf_counter()
        A = assemble_n0(nodes, tets)
        k = ARGS.n_eigen_modes
        vals, vecs, info = solve_e_modes(A, k + 1)
        t_solve = time.perf_counter() - t0
        return {
            "id": s_id, "nodes": np.ascontiguousarray(A["mesh"].p.T), "tets": np.ascontiguousarray(A["mesh"].t.T),
            "edges": A["edges"], "e_edges": vecs[:, :k],
            "freqs": eigenvalues_to_ghz(vals[:k]),
            "freq_next": float(eigenvalues_to_ghz(vals[k])), "shape_type": shape_type, "geom_params": params,
            "div_residual": info["div_residual"], "mesh_h": float(h), "volume": float(volume),
            "t_mesh": t_mesh, "t_solve": t_solve, "n_bnd_components": int(g["topo"]["n_bnd_components"]),
            "betti1": int(g["topo"]["betti1"]), "size": (len(A["mesh"].p.T), len(A["edges"])),
        }
    except Exception as e:
        print(f"Error generating sample {s_id}: {e}")
        return None
    finally:
        try:
            gmsh.clear()
        except Exception:
            pass


def _write_common_attrs(g, res):
    g.attrs["shape_type"] = res["shape_type"]
    g.attrs["geom_params"] = json.dumps({k: float(v) for k, v in res["geom_params"].items()})
    for k_, v_ in res["geom_params"].items():
        g.attrs[k_] = float(v_)
    g.attrs["field"] = "E"
    for k_ in ("freq_next", "mesh_h", "volume", "t_mesh", "t_solve"):
        g.attrs[k_] = float(res[k_])
    for k_ in ("n_bnd_components", "betti1"):
        g.attrs[k_] = int(res.get(k_, 1 if k_ == "n_bnd_components" else 0))
    if res["shape_type"].startswith("calib_"):
        g.attrs["freqs_analytic"] = eigenvalues_to_ghz(analytic_k2(res["shape_type"], res["geom_params"],
                                                                   len(res["freqs"])))


def write_sample(f_h5, res):
    g = f_h5.create_group(f"sample_{res['id']:04d}")
    if res.get("labels") == "field":
        from src.data_gen.field_labels import write_field_group
        write_field_group(g, res["field"])
        _write_common_attrs(g, res)
        a = res["field"]["attrs"]
        g.attrs["fem_element"] = f"HCurl p{a['model_order']} (labels p{a['label_order']}, curve {a['curve']})"
        return g
    z = dict(compression="gzip", compression_opts=4)
    g.create_dataset("nodes", data=res["nodes"], **z)
    g.create_dataset("tets", data=res["tets"].astype(np.int64), **z)
    g.create_dataset("edges", data=res["edges"].astype(np.int64), **z)
    g.create_dataset("e_edges", data=res["e_edges"], **z)
    g.create_dataset("freqs", data=res["freqs"])
    g.attrs["shape_type"] = res["shape_type"]
    g.attrs["geom_params"] = json.dumps({k: float(v) for k, v in res["geom_params"].items()})
    for k_, v_ in res["geom_params"].items():
        g.attrs[k_] = float(v_)
    g.attrs["n_nodes"] = int(len(res["nodes"]))
    g.attrs["n_tets"] = int(len(res["tets"]))
    g.attrs["n_edges"] = int(len(res["edges"]))
    g.attrs["fem_element"] = "N0"
    g.attrs["field"] = "E"
    for k_ in ("freq_next", "div_residual", "mesh_h", "volume", "t_mesh", "t_solve"):
        g.attrs[k_] = float(res[k_])
    for k_ in ("n_bnd_components", "betti1"):
        g.attrs[k_] = int(res.get(k_, 1 if k_ == "n_bnd_components" else 0))
    if res["shape_type"].startswith("calib_"):
        ref = eigenvalues_to_ghz(analytic_k2(res["shape_type"], res["geom_params"], len(res["freqs"])))
        g.attrs["freqs_analytic"] = ref
    return g


def _run_chunk(pool, chunk_range, timeout):
    handles = [(s_id, pool.apply_async(generate_sample_data, (s_id,))) for s_id in chunk_range]
    results, hung = [], []
    for s_id, h in handles:
        try:
            results.append(h.get(timeout=timeout))
        except MPTimeoutError:
            print(f"Sample {s_id} timed out after {timeout:.0f}s, skipped.")
            hung.append(s_id)
    return results, hung


def file_metadata(args):
    if getattr(args, "labels", "n0") == "field":
        return {"generator_args": vars(args), "field": "E", "labels": "field",
                "fem_element": f"HCurl p{args.model_order} on a curved mesh (curve {args.curve}); "
                               f"labels HCurl p{args.label_order} on the same mesh, M-projected",
                "dataset": "u_model",
                "formulation": "curl-curl E = k^2 E in H0(curl), PEC wall 'wall' (NGSolve dirichlet)",
                "storage": "per sample: netgen .vol + BREP of the meshed shape (+ deformation map); "
                           "src.data_gen.field_labels.read_field_group rebuilds the curved mesh",
                "freq_unit": "GHz", "length_unit": "m", "c0": C0, "docs": "docs/29_FIELD_LABELS.md"}
    return {"generator_args": vars(args), "field": "E", "fem_element": "N0 (skfem ElementTetN0, all edges)",
            "dataset": "e_edges",
            "formulation": "curl-curl E = k^2 E in H0(curl): n x E = 0 essential (PEC wall edge DOFs removed)",
            "kernel": "gradients of P1 potentials: interior vertices + one constant per boundary component except "
                      "the first (b2 harmonic Dirichlet fields), removed by M-orthogonal projection (Kp SPD)",
            "topology": "any connected manifold domain: handles (b1 > 0) and isolated conductors (b2 > 0) allowed",
            "dof_convention": "e_edges[i] = line integral of E along edges[i,0] -> edges[i,1]; "
                              "edges[i,0] < edges[i,1]; wall-edge rows exactly 0",
            "normalisation": "e_edges columns M-orthonormal in physical units; sign arbitrary",
            "freq_unit": "GHz", "length_unit": "m", "c0": C0}


def main(argv=None):
    global ARGS
    ARGS = parse_args(argv)
    field = ARGS.labels == "field"
    n_workers = ARGS.n_workers or (max(1, cpu_count() // max(1, ARGS.threads)) if field else min(cpu_count(), 4))
    print(f"Generating {'ids from ' + ARGS.ids_file if ARGS.ids_file else ARGS.n_total} 3D samples "
          f"({ARGS.mode}, "
          f"families={ARGS.families}, sampling={ARGS.sampling}, deform_prob={ARGS.deform_prob}, labels={ARGS.labels}"
          f"{f', p{ARGS.label_order}->p{ARGS.model_order}, min_fillet={ARGS.min_fillet}' if field else ''}) "
          f"with {n_workers} workers")
    tmp_path = ARGS.h5_filename + ".partial"
    n_ok, n_fail, times, calib = 0, 0, [], []
    # maxtasksperchild: recycle workers (fresh gmsh / OCC state and memory) every 50 samples; field labels:
    # one sample per process (minutes each; gmsh + netgen OCC and NGSolve state never accumulate)
    per_child = 1 if field else 50
    new_pool = lambda: Pool(n_workers, initializer=_init_worker, initargs=(ARGS,), maxtasksperchild=per_child)  # noqa: E731
    ids = [int(x) for x in open(ARGS.ids_file).read().split()] if ARGS.ids_file else \
        list(range(ARGS.start_id, ARGS.start_id + ARGS.n_total))
    n_requested = len(ids)
    resume = bool(getattr(ARGS, "resume", False)) and os.path.exists(tmp_path)
    if resume:
        with h5py.File(tmp_path, "r") as f_h5:
            done = {int(k.split("_", 1)[1]) for k in f_h5.keys() if k.startswith("sample_")}
        n_ok = len(done)
        ids = [i for i in ids if i not in done]
        print(f"Resuming {tmp_path}: {n_ok} samples already there, {len(ids)} ids left")
    pool = new_pool()
    try:
        with h5py.File(tmp_path, "a" if resume else "w") as f_h5:
            if not resume:
                f_h5.attrs["metadata"] = json.dumps(file_metadata(ARGS))
            chunk = 4 * n_workers
            for i in range(0, len(ids), chunk):
                results, hung = _run_chunk(pool, ids[i:i + chunk], ARGS.sample_timeout)
                if hung:
                    pool.terminate()
                    pool = new_pool()
                n_fail += len(hung)
                for res in results:
                    if res is None:
                        n_fail += 1
                        continue
                    g = write_sample(f_h5, res)
                    n_ok += 1
                    times.append((res["t_mesh"], res["t_solve"]) + tuple(res["size"]))
                    if "freqs_analytic" in g.attrs:
                        calib.append((res["shape_type"], res["freqs"], g.attrs["freqs_analytic"]))
                f_h5.flush()
                print(f"  {min(i + chunk, len(ids))}/{len(ids)} done ({n_ok} ok, {n_fail} failed)")
    finally:
        pool.terminate()
    with h5py.File(tmp_path, "a") as f_h5:          # bookkeeping for cluster status reports
        f_h5.attrs["n_requested"] = n_requested
        f_h5.attrs["n_failed_last_run"] = n_fail
    if n_ok == 0:
        os.remove(tmp_path)
        raise SystemExit(f"No valid samples generated ({n_fail} failed); nothing written.")
    os.replace(tmp_path, ARGS.h5_filename)
    t = np.array(times) if times else np.zeros((1, 4))
    print(f"\nDone: {n_ok} samples ({n_fail} failed) -> {ARGS.h5_filename}")
    a, b = ("tets", "model DOF") if field else ("Nv", "Ne")
    print(f"  {a} {t[:, 2].min():.0f}-{t[:, 2].max():.0f}, {b} {t[:, 3].min():.0f}-{t[:, 3].max():.0f}; "
          f"mean t_mesh {t[:, 0].mean():.2f}s, t_solve {t[:, 1].mean():.2f}s per sample (one worker)")
    for st, f, ref in calib:
        print(f"  {st}: rel err f/f_exact-1 = {np.array2string(f / ref - 1, precision=2)}")


if __name__ == "__main__":
    main()
