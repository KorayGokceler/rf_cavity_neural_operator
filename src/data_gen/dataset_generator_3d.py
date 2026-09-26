"""3D PEC cavity eigenmode dataset generator (H-field, lowest-order Nédélec N0).

Physics (docs/18 §1.4, §2.2):  find H ∈ H(curl, Ω) with

    ∫ curl H · curl v = k² ∫ H · v        for all v ∈ H(curl, Ω),

with **no essential boundary condition** on the edge DOFs. On a PEC wall both
n·H = 0 and n × curl H = 0 (⇔ n × E = 0) are natural. Discretisation:
skfem ``ElementTetN0`` on all edges, ``K u = λ M u``, f = c·√λ/(2π).

Kernel: ker K = G·P1 = gradients of *all* P1 functions (boundary vertices
included). Constants have zero gradient, so one vertex potential is pinned
(column 0 of G dropped) to make the potential Laplacian Kp = GᵀMG SPD.
The eigensolver is the validated research recipe (scripts/research_3d/n0lib.py,
``solve_projected``): shift-invert with σ < 0 (K − σM SPD) and the M-orthogonal
projection P = I − G Kp⁻¹ GᵀM after every solve, so gradient fields become
θ = 0 (λ = ∞) and never appear among the returned modes.

**Topology restriction:** domains must be free of handles (first Betti number
b1 = 0: no through-holes, no tori, no coaxial inner conductor touching both end
plates). For b1 > 0 the H-formulation kernel additionally contains b1 harmonic
fields (curl-free, M-orthogonal to all gradients) which the projection does not
remove; they would appear as spurious λ ≈ 0 modes. Every mesh is certified
before solving: connected, Euler characteristic V−E+F−T = 1 and a single
boundary shell with χ(∂Ω) = 2 (⇒ b1 = b2 = 0, a topological ball).

H5 layout (one group ``sample_XXXX`` per geometry):
    nodes   [Nv,3] float64   vertex coordinates [m]
    tets    [Nt,4] int64     vertex indices (the skfem MeshTet.t used for the solve)
    edges   [Ne,2] int64     edges[i] = (tail, head) of the edge carrying DOF i, in
                             skfem ``ElementTetN0`` DOF order; tail < head always
                             (skfem orients each edge from the lower to the higher
                             global vertex index)
    h_edges [Ne,K] float64   H-field N0 DOFs, h_edges[i,j] = ∫_{tail→head} H_j · dl,
                             M-orthonormal in physical units (sign arbitrary)
    freqs   [K]    float64   GHz, ascending
    attrs: shape_type, geom_params (JSON) + one float attr per parameter, n_nodes,
           n_edges, n_tets, freq_next (GHz, mode K+1: shows whether K splits a
           degenerate cluster), div_residual, mesh_h, t_mesh, t_solve, field, fem_element
File attr ``metadata`` (JSON): generator args, formulation, conventions, units.
"""
import argparse
import json
import logging
import os
import time
from multiprocessing import Pool, TimeoutError as MPTimeoutError, cpu_count

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import jn_zeros, jnp_zeros

logging.getLogger('skfem').setLevel(logging.ERROR)

C0 = 299792458.0  # speed of light [m/s]
FAMILIES = ("pillbox", "axisym_cell", "blob")

# Calibration geometries [m] (non-degenerate low box spectrum; L < 2.03 R → TM010 first)
CALIB_BOX = (0.10, 0.08, 0.06)
CALIB_PILLBOX = (0.04, 0.05)  # (R, L)

ARGS = None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate 3D PEC cavity eigenmode dataset (H-field, Nédélec N0).")
    p.add_argument("--h5_filename", type=str, default="rf_cavity_3d_dataset.h5", help="Output H5 filename.")
    p.add_argument("--n_total", type=int, default=100, help="Number of geometries to generate.")
    p.add_argument("--mode", type=str, default="random", choices=["random", "calibration"],
                   help="random families, or calibration (PEC box / pillbox with analytic spectra).")
    p.add_argument("--families", type=str, nargs="+", default=list(FAMILIES), choices=list(FAMILIES),
                   help="Geometry families drawn uniformly in random mode.")
    p.add_argument("--n_eigen_modes", type=int, default=6, help="Number K of physical modes stored.")
    p.add_argument("--mesh_size", type=float, default=0.12,
                   help="Target tet size relative to the characteristic length V^(1/3) of the cavity.")
    p.add_argument("--mesh_size_abs", type=float, default=None,
                   help="Absolute target tet size [m]; overrides --mesh_size.")
    p.add_argument("--seed", type=int, default=0, help="Base seed; sample s uses default_rng([seed, s]).")
    p.add_argument("--n_workers", type=int, default=None, help="Worker processes (default: min(cpu_count, 4)).")
    p.add_argument("--sample_timeout", type=float, default=300.0, help="Seconds before a sample is skipped.")
    p.add_argument("--max_geom_tries", type=int, default=6,
                   help="Re-draws when a random geometry fails the boolean/topology checks.")
    return p.parse_args(argv)


def _init_worker(args):
    global ARGS
    ARGS = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")


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


BUILDERS = {"pillbox": build_pillbox, "axisym_cell": build_axisym_cell, "blob": build_blob}


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
    """Returns dict(euler=V−E+F−T, n_components, n_shells, euler_boundary).
    A topological ball (no handles, no voids) has euler=1, n_components=1, n_shells=1,
    euler_boundary=2."""
    from scipy.sparse.csgraph import connected_components
    t = np.sort(np.asarray(tets, dtype=np.int64), axis=1)
    edges = np.unique(t[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2), axis=0)
    faces_all = t[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3)
    faces, cnt = np.unique(faces_all, axis=0, return_counts=True)
    if cnt.max() > 2:
        raise RuntimeError("non-manifold mesh (face shared by > 2 tets)")
    bf = faces[cnt == 1]
    euler = n_nodes - len(edges) + len(faces) - len(t)
    A = sp.coo_matrix((np.ones(3 * len(t)), (np.repeat(t[:, 0], 3), t[:, 1:].ravel())), shape=(n_nodes,) * 2)
    n_comp = connected_components(A, directed=False)[0]
    bv = np.unique(bf)
    loc = np.searchsorted(bv, bf)
    be = np.unique(np.sort(loc[:, [[0, 1], [1, 2], [0, 2]]].reshape(-1, 2), axis=1), axis=0)
    Ab = sp.coo_matrix((np.ones(len(be)), (be[:, 0], be[:, 1])), shape=(len(bv),) * 2)
    n_shell = connected_components(Ab, directed=False)[0]
    return dict(euler=int(euler), n_components=int(n_comp), n_shells=int(n_shell),
                euler_boundary=int(len(bv) - len(be) + len(bf)))


def is_topological_ball(topo):
    return (topo["euler"] == 1 and topo["n_components"] == 1 and topo["n_shells"] == 1
            and topo["euler_boundary"] == 2)


# ─────────────────────────── N0 solver (H-field) ───────────────

def assemble_h_n0(nodes, tets):
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


def solve_h_modes(A, k, tol=1e-10, sigma=None):
    """k lowest non-zero eigenpairs of K u = λ M u on the full N0 space, gradient kernel
    removed by projection (research recipe n0lib.solve_projected, with the Neumann/H
    kernel: G on all vertices, one vertex pinned for the constant)."""
    K, M = A["K"], A["M"]
    Gp = A["G"][:, 1:].tocsr()                             # pin vertex 0 (constants: zero gradient)
    if sigma is None:                                      # scale-aware small negative shift
        sigma = -1e-2 * (K.diagonal().mean() / M.diagonal().mean())
    lu = spd_factor(K - sigma * M)
    MG = (M @ Gp).tocsr()
    lup = spd_factor(Gp.T @ MG)                            # Kp: P1 Neumann Laplacian, SPD after pinning

    def P(x):
        return x - Gp @ lup.solve(MG.T @ x)

    OP = spla.LinearOperator(K.shape, matvec=lambda x: P(lu.solve(x)), dtype=float)
    v0 = P(np.random.default_rng(0).standard_normal(K.shape[0]))
    vals, vecs = spla.eigsh(K, k=k, M=M, sigma=sigma, OPinv=OP, which="LM", v0=v0, tol=tol,
                            ncv=min(K.shape[0] - 1, max(2 * k + 1, 30)))
    o = np.argsort(vals)
    vals, vecs = vals[o], vecs[:, o]
    # ── checks: no zero / gradient modes, M-orthonormal ──
    lam_scale = K.diagonal().mean() / M.diagonal().mean()     # ~ 1/h² (top of the spectrum)
    Mv = M @ vecs
    div = np.abs(A["G"].T @ Mv).max(axis=0) / np.abs(Mv).max(axis=0)
    if not np.all(np.isfinite(vals)) or vals.min() <= 1e-6 * lam_scale:
        raise RuntimeError(f"zero / non-physical eigenvalue in {vals} (scale {lam_scale:.3g})")
    if div.max() > 1e-6:
        raise RuntimeError(f"mode not M-orthogonal to gradients (div residual {div.max():.2e})")
    gram = vecs.T @ Mv
    if np.abs(gram - np.eye(k)).max() > 1e-6:
        raise RuntimeError("eigenvectors not M-orthonormal")
    return vals, vecs, dict(sigma=sigma, div_residual=float(div.max()), lam_scale=float(lam_scale))


# ─────────────────────────── sample ────────────────────────────

def _seeds(s_id):
    return np.random.default_rng([int(ARGS.seed), int(s_id)])


def generate_sample_data(s_id):
    import gmsh
    try:
        if not gmsh.isInitialized():
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 1)
            gmsh.option.setNumber("General.NumThreads", 1)
        rng = _seeds(s_id)
        t0 = time.perf_counter()
        for attempt in range(ARGS.max_geom_tries):
            gmsh.clear()
            gmsh.model.add(f"rf3d_{s_id}_{attempt}")
            occ = gmsh.model.occ
            try:
                if ARGS.mode == "calibration":
                    shape_type, params = build_calibration(occ, s_id)
                else:
                    fam = ARGS.families[int(rng.integers(0, len(ARGS.families)))]
                    shape_type, params = BUILDERS[fam](occ, rng)
                occ.synchronize()
                vols = gmsh.model.getEntities(3)
                if len(vols) != 1:
                    raise RuntimeError(f"{len(vols)} volumes")
                volume = gmsh.model.occ.getMass(3, vols[0][1])
                h = ARGS.mesh_size_abs or ARGS.mesh_size * volume ** (1.0 / 3.0)
                nodes, tets = _mesh_current_model(h)
                topo = mesh_topology(tets, len(nodes))
                if not is_topological_ball(topo):
                    raise RuntimeError(f"not a topological ball: {topo}")
                break
            except Exception as e:  # re-draw the geometry (random mode only)
                if ARGS.mode == "calibration" or attempt == ARGS.max_geom_tries - 1:
                    raise
                print(f"sample {s_id}: geometry attempt {attempt} rejected ({e}); re-drawing")
        t_mesh = time.perf_counter() - t0

        t0 = time.perf_counter()
        A = assemble_h_n0(nodes, tets)
        k = ARGS.n_eigen_modes
        vals, vecs, info = solve_h_modes(A, k + 1)
        t_solve = time.perf_counter() - t0
        return {
            "id": s_id, "nodes": np.ascontiguousarray(A["mesh"].p.T), "tets": np.ascontiguousarray(A["mesh"].t.T),
            "edges": A["edges"], "h_edges": vecs[:, :k], "freqs": eigenvalues_to_ghz(vals[:k]),
            "freq_next": float(eigenvalues_to_ghz(vals[k])), "shape_type": shape_type, "geom_params": params,
            "div_residual": info["div_residual"], "mesh_h": float(h), "volume": float(volume),
            "t_mesh": t_mesh, "t_solve": t_solve,
        }
    except Exception as e:
        print(f"Error generating sample {s_id}: {e}")
        return None
    finally:
        try:
            gmsh.clear()
        except Exception:
            pass


def write_sample(f_h5, res):
    g = f_h5.create_group(f"sample_{res['id']:04d}")
    z = dict(compression="gzip", compression_opts=4)
    g.create_dataset("nodes", data=res["nodes"], **z)
    g.create_dataset("tets", data=res["tets"].astype(np.int64), **z)
    g.create_dataset("edges", data=res["edges"].astype(np.int64), **z)
    g.create_dataset("h_edges", data=res["h_edges"], **z)
    g.create_dataset("freqs", data=res["freqs"])
    g.attrs["shape_type"] = res["shape_type"]
    g.attrs["geom_params"] = json.dumps({k: float(v) for k, v in res["geom_params"].items()})
    for k_, v_ in res["geom_params"].items():
        g.attrs[k_] = float(v_)
    g.attrs["n_nodes"] = int(len(res["nodes"]))
    g.attrs["n_tets"] = int(len(res["tets"]))
    g.attrs["n_edges"] = int(len(res["edges"]))
    g.attrs["fem_element"] = "N0"
    g.attrs["field"] = "H"
    for k_ in ("freq_next", "div_residual", "mesh_h", "volume", "t_mesh", "t_solve"):
        g.attrs[k_] = float(res[k_])
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
    return {
        "generator_args": vars(args), "field": "H", "fem_element": "N0 (skfem ElementTetN0, all edges)",
        "formulation": "curl-curl H = k^2 H, no essential BC (PEC: n.H = 0, n x curl H = 0 natural)",
        "kernel": "gradients of all P1 functions, removed by M-orthogonal projection (vertex 0 pinned)",
        "topology": "topological balls only (b1 = b2 = 0): no handles / through-holes / tori / voids",
        "dof_convention": "h_edges[i] = line integral of H along edges[i,0] -> edges[i,1]; edges[i,0] < edges[i,1]",
        "normalisation": "h_edges columns M-orthonormal in physical units; sign arbitrary",
        "freq_unit": "GHz", "length_unit": "m", "c0": C0,
    }


def main(argv=None):
    global ARGS
    ARGS = parse_args(argv)
    n_workers = ARGS.n_workers or min(cpu_count(), 4)
    print(f"Generating {ARGS.n_total} 3D samples ({ARGS.mode}, families={ARGS.families}) with {n_workers} workers")
    tmp_path = ARGS.h5_filename + ".partial"
    n_ok, n_fail, times, calib = 0, 0, [], []
    pool = Pool(n_workers, initializer=_init_worker, initargs=(ARGS,))
    try:
        with h5py.File(tmp_path, "w") as f_h5:
            f_h5.attrs["metadata"] = json.dumps(file_metadata(ARGS))
            chunk = 4 * n_workers
            for i in range(0, ARGS.n_total, chunk):
                results, hung = _run_chunk(pool, range(i, min(i + chunk, ARGS.n_total)), ARGS.sample_timeout)
                if hung:
                    pool.terminate()
                    pool = Pool(n_workers, initializer=_init_worker, initargs=(ARGS,))
                n_fail += len(hung)
                for res in results:
                    if res is None:
                        n_fail += 1
                        continue
                    g = write_sample(f_h5, res)
                    n_ok += 1
                    times.append((res["t_mesh"], res["t_solve"], len(res["nodes"]), len(res["edges"])))
                    if "freqs_analytic" in g.attrs:
                        calib.append((res["shape_type"], res["freqs"], g.attrs["freqs_analytic"]))
                f_h5.flush()
                print(f"  {min(i + chunk, ARGS.n_total)}/{ARGS.n_total} done ({n_ok} ok, {n_fail} failed)")
    finally:
        pool.terminate()
    if n_ok == 0:
        os.remove(tmp_path)
        raise SystemExit(f"No valid samples generated ({n_fail} failed); nothing written.")
    os.replace(tmp_path, ARGS.h5_filename)
    t = np.array(times)
    print(f"\nDone: {n_ok} samples ({n_fail} failed) -> {ARGS.h5_filename}")
    print(f"  Nv {t[:, 2].min():.0f}-{t[:, 2].max():.0f}, Ne {t[:, 3].min():.0f}-{t[:, 3].max():.0f}; "
          f"mean t_mesh {t[:, 0].mean():.2f}s, t_solve {t[:, 1].mean():.2f}s per sample (one worker)")
    for st, f, ref in calib:
        print(f"  {st}: rel err f/f_exact-1 = {np.array2string(f / ref - 1, precision=2)}")


if __name__ == "__main__":
    main()
