import os, json, gmsh, h5py, numpy as np
import matplotlib
matplotlib.use("Agg")  # headless (worker/CI) ortamlarda display gerektirmesin
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from skfem import *
from skfem import utils
from skfem.models.poisson import laplace, mass
from multiprocessing import Pool, TimeoutError as MPTimeoutError, cpu_count
from tqdm import tqdm
import logging
logging.getLogger('skfem').setLevel(logging.ERROR)

import argparse

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate 2D RF Cavity Dataset.")
    parser.add_argument("--h5_filename", type=str, default="rf_cavity_1000_dataset.h5", help="Output H5 filename.")
    parser.add_argument("--plot_dir", type=str, default="dataset_plots", help="Directory to save plots.")
    parser.add_argument("--n_total", type=int, default=1000, help="Total number of samples to generate.")
    parser.add_argument("--n_plot", type=int, default=100, help="Number of samples to plot.")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "calibration"], help="Generation mode: random shapes or calibration shapes (square/circle).")
    # FEM Solver
    parser.add_argument("--n_eigen_modes", type=int, default=3, help="Number of eigenmodes to solve (k).")
    parser.add_argument("--eigen_sigma", type=float, default=500.0, help="Shift-invert sigma for eigen solver.")
    # Mesh control
    parser.add_argument("--mesh_size_min", type=float, default=0.0012, help="Min mesh size near boundary.")
    parser.add_argument("--mesh_size_max", type=float, default=0.005, help="Max mesh size at center.")
    parser.add_argument("--mesh_dist_min", type=float, default=0.002, help="Distance where mesh refinement starts.")
    parser.add_argument("--mesh_dist_max", type=float, default=0.03, help="Distance where mesh refinement ends.")
    # Geometry randomization
    parser.add_argument("--sharp_n_pts_range", type=int, nargs=2, default=[7, 13], metavar=("MIN", "MAX"), help="Corner count range for sharp geometries.")
    parser.add_argument("--sharp_r_range", type=float, nargs=2, default=[0.02, 0.046], metavar=("MIN", "MAX"), help="Radius range for sharp geometries.")
    parser.add_argument("--smooth_base_r", type=float, default=0.035, help="Base radius for smooth geometries.")
    parser.add_argument("--smooth_perturb", type=float, default=0.008, help="Perturbation amplitude for smooth geometries.")
    parser.add_argument("--smooth_harmonics", type=int, nargs=2, default=[2, 8], metavar=("MIN", "MAX"), help="Harmonic range for smooth geometries.")
    parser.add_argument("--hole_prob", type=float, default=0.0, help="Probability that a random geometry gets 1..max_holes elliptic holes (multiply connected).")
    parser.add_argument("--max_holes", type=int, default=2, help="Maximum number of holes per geometry.")
    # Reproducibility / robustness
    parser.add_argument("--seed", type=int, default=0, help="Base seed: sample s_id uses seed s_id*13 + seed*1000003 (seed=0 reproduces legacy datasets).")
    parser.add_argument("--n_workers", type=int, default=None, help="Worker processes (default: cpu_count()).")
    parser.add_argument("--sample_timeout", type=float, default=300.0, help="Seconds before one sample (mesh+solve) is considered hung and skipped.")
    return parser.parse_args(argv)

# Global settings placeholder (set in __main__ and in every worker via _init_worker,
# so it also works with the 'spawn' start method used by default on macOS/Windows).
ARGS = None

C0 = 299792458.0  # speed of light [m/s]

# Calibration geometry parameters (also stored as H5 attrs, read back by validate_data.py)
CALIB_SQUARE_SIDE = 0.08
CALIB_CIRCLE_RADIUS = 0.04
CALIB_ANNULUS_R_OUTER = 0.045
CALIB_ANNULUS_R_INNER = 0.02


def _init_worker(args):
    global ARGS
    ARGS = args


def eigenvalues_to_ghz(vals):
    """Dirichlet-Laplacian eigenvalue k^2 [1/m^2] -> resonant frequency f = c*k/(2*pi) [GHz]."""
    return C0 * np.sqrt(np.abs(np.asarray(vals, dtype=np.float64))) / (2 * np.pi) / 1e9


def solve_dirichlet_eigenmodes(nodes, elements, k, sigma):
    """TM (E_z) modes of a PEC cavity: -lap(u) = k^2 u in Omega, u = 0 on the boundary.

    P2 Lagrange elements. Returns (vals [k] ascending, vecs [n_p2_dofs, k]) with
    M-orthonormal eigenvectors (vecs.T @ M @ vecs = I). The first n_nodes DOFs of
    the P2 basis are the mesh vertices (checked), which the converter relies on.
    """
    m = MeshTri(np.asarray(nodes, dtype=np.float64).T, np.asarray(elements).T)
    basis = Basis(m, ElementTriP2())
    if not np.array_equal(basis.nodal_dofs[0], np.arange(m.p.shape[1])):
        raise RuntimeError("P2 vertex DOFs are not the first n_nodes DOFs; converter slicing would be wrong.")
    K, M = laplace.assemble(basis), mass.assemble(basis)
    D = basis.get_dofs(facets=m.boundary_facets())
    Kc, Mc, xc, Ic = utils.condense(K, M, D=D, expand=True)

    def _solve(sig):
        # Symmetric solver (eigsh): real eigenpairs, M-orthonormal even inside
        # (near-)degenerate pairs. skfem's default solve_eigen uses the
        # non-symmetric `eigs`, whose output is complex and not guaranteed sorted.
        vals, vecs = utils.solve_eigen(Kc, Mc, x=xc, I=Ic,
                                       solver=utils.solver_eigen_scipy_sym(k=k, sigma=sig))
        return np.real(vals), np.real(vecs)

    vals, vecs = _solve(sigma)
    # Shift-invert returns the k eigenvalues *closest to sigma*. A skipped lower
    # eigenvalue l' (0 < l' < min(vals)) would need sigma - l' >= r := max|vals - sigma|,
    # which is only possible if r < sigma. In that case re-solve at sigma=0, which
    # always gives the k lowest (K is SPD after Dirichlet condensation).
    if sigma > 0 and np.max(np.abs(vals - sigma)) < sigma:
        vals, vecs = _solve(0.0)
    order = np.argsort(vals, kind="stable")
    vals, vecs = vals[order], vecs[:, order]
    if not np.all(np.isfinite(vals)) or vals.min() <= 0:
        raise RuntimeError(f"Non-physical eigenvalues from solver: {vals}")
    return vals, vecs


def _extract_linear_triangle_mesh():
    """Current gmsh model -> (nodes [N,2] float64, elements [M,3] int32, 0-indexed).

    Maps gmsh node *tags* to array rows explicitly instead of assuming
    tags == 1..N in getNodes() order, and drops nodes no triangle uses.
    """
    tags, coords, _ = gmsh.model.mesh.getNodes()
    elem_types, _, conns = gmsh.model.mesh.getElements(2)
    elem_types = [int(t) for t in elem_types]
    if elem_types != [2]:  # gmsh type 2 = 3-node triangle
        raise RuntimeError(f"Expected only 3-node triangles, got 2D element types {elem_types}")
    tri_tags = np.asarray(conns[0], dtype=np.int64).reshape(-1, 3)
    coords = np.asarray(coords).reshape(-1, 3)[:, :2]
    tags = np.asarray(tags, dtype=np.int64)
    tag2row = np.full(tags.max() + 1, -1, dtype=np.int64)
    tag2row[tags] = np.arange(len(tags))
    tri_rows = tag2row[tri_tags]
    if (tri_rows < 0).any():
        raise RuntimeError("Triangle references a node tag missing from getNodes()")
    used = np.unique(tri_rows)
    remap = np.full(len(tags), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    nodes = np.ascontiguousarray(coords[used], dtype=np.float64)
    elements = np.ascontiguousarray(remap[tri_rows].astype(np.int32))
    return nodes, elements


def generate_sample_data(s_id):
    try:
        if not gmsh.isInitialized():
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 1)

        model_name = f"rf_{s_id}_{os.getpid()}"
        if model_name in gmsh.model.list():
            gmsh.model.remove()
        gmsh.model.add(model_name)

        np.random.seed((s_id * 13 + ARGS.seed * 1000003) % (2**32))
        cx, cy = 0.05, 0.05
        geom_params = {}

        # Build Geometry using OCC Primitives for calibration or Polygons for random
        if ARGS.mode == 'calibration':
            # Calibration: Square, Circle, or Annulus (Simit)
            shape_choice = s_id % 3
            if shape_choice == 0:
                shape_type = 'square'
                side = CALIB_SQUARE_SIDE
                geom_params = {'side': side}
                gmsh.model.occ.addRectangle(cx - side/2, cy - side/2, 0, side, side)
            elif shape_choice == 1:
                shape_type = 'circle'
                radius = CALIB_CIRCLE_RADIUS
                geom_params = {'radius': radius}
                gmsh.model.occ.addDisk(cx, cy, 0, radius, radius)
            else:
                shape_type = 'annulus'
                r_outer = CALIB_ANNULUS_R_OUTER
                r_inner = CALIB_ANNULUS_R_INNER
                geom_params = {'r_outer': r_outer, 'r_inner': r_inner}
                d1 = gmsh.model.occ.addDisk(cx, cy, 0, r_outer, r_outer)
                d2 = gmsh.model.occ.addDisk(cx, cy, 0, r_inner, r_inner)
                # Cut disk1 with disk2 to create a hole
                gmsh.model.occ.cut([(2, d1)], [(2, d2)])
        else:
            method = np.random.choice(['sharp', 'smooth'])
            if method == 'sharp':
                shape_type = 'random_sharp'
                n_pts = np.random.randint(ARGS.sharp_n_pts_range[0], ARGS.sharp_n_pts_range[1])
                delta = 2 * np.pi / n_pts
                angles = np.array([i * delta + np.random.uniform(-delta/3, delta/3) for i in range(n_pts)])
                r = np.random.uniform(ARGS.sharp_r_range[0], ARGS.sharp_r_range[1], n_pts)
                pts_c = [(cx + ri*np.cos(ai), cy + ri*np.sin(ai)) for ri, ai in zip(r, angles)]
                # Inscribed radius around (cx, cy): an edge spanning angle θ is
                # at least min(r)·cos(θ/2) from the centre.
                max_gap = np.max(np.diff(np.append(angles, angles[0] + 2*np.pi)))
                r_safe = r.min() * np.cos(max_gap / 2)
            else:
                shape_type = 'random_smooth'
                t = np.linspace(0, 2*np.pi, 100, endpoint=False)
                r_raw = ARGS.smooth_base_r + sum(np.random.uniform(-ARGS.smooth_perturb, ARGS.smooth_perturb) * np.cos(k*t + np.random.uniform(0, 2*np.pi)) for k in range(ARGS.smooth_harmonics[0], ARGS.smooth_harmonics[1]))
                # Keep r ≥ 0.015 by shrinking the perturbation (a clip made kinks).
                r_min = 0.015
                if r_raw.min() < r_min:
                    r_raw = ARGS.smooth_base_r + (r_raw - ARGS.smooth_base_r) * (ARGS.smooth_base_r - r_min) / (ARGS.smooth_base_r - r_raw.min())
                r = r_raw
                pts_c = [(cx + ri*np.cos(ti), cy + ri*np.sin(ti)) for ri, ti in zip(r, t)]
                r_safe = 0.95 * r.min()

            pts = [gmsh.model.occ.addPoint(p[0], p[1], 0) for p in pts_c]
            if method == 'smooth':
                # Periodic C2 spline through the samples: a 100-segment polyline
                # had re-entrant vertices (>200° in ~98% of "smooth" shapes →
                # corner singularities, less accurate labels).
                curves = [gmsh.model.occ.addSpline(pts + [pts[0]])]
            else:
                curves = [gmsh.model.occ.addLine(pts[i], pts[(i+1)%len(pts)]) for i in range(len(pts))]
            surf = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop(curves)])

            # Optional holes (multiply connected cavities): elliptic holes kept
            # inside the inscribed circle with a wall of ≥ 0.15·r_safe.
            # No random draw when hole_prob == 0 → legacy datasets unchanged.
            if ARGS.hole_prob > 0 and np.random.rand() < ARGS.hole_prob:
                margin, holes, placed = 0.15 * r_safe, [], []
                for _ in range(np.random.randint(1, ARGS.max_holes + 1)):
                    rh = np.random.uniform(0.15, 0.35) * r_safe
                    rho = np.random.uniform(0.0, r_safe - rh - margin)
                    phi = np.random.uniform(0, 2*np.pi)
                    hx, hy = cx + rho*np.cos(phi), cy + rho*np.sin(phi)
                    if any(np.hypot(hx - px, hy - py) < rh + pr + margin for px, py, pr in placed):
                        continue
                    placed.append((hx, hy, rh))
                    h = gmsh.model.occ.addDisk(hx, hy, 0, rh, rh * np.random.uniform(0.6, 1.0))
                    gmsh.model.occ.rotate([(2, h)], hx, hy, 0, 0, 0, 1, np.random.uniform(0, np.pi))
                    holes.append((2, h))
                gmsh.model.occ.cut([(2, surf)], holes)
                shape_type += f'_hole{len(holes)}'

        gmsh.model.occ.synchronize()

        # Automatically find boundary lines for adaptive mesh field
        surfaces = gmsh.model.getEntities(dim=2)
        all_boundary_lines = []
        for s in surfaces:
            # getBoundary returns a list of (dim, tag)
            bnd = gmsh.model.getBoundary([s], combined=True, oriented=False, recursive=False)
            all_boundary_lines.extend([abs(e[1]) for e in bnd if e[0] == 1])

        # Adaptive Mesh: Dense at boundary, balanced at center
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", all_boundary_lines)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", ARGS.mesh_size_min)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", ARGS.mesh_size_max)
        gmsh.model.mesh.field.setNumber(2, "DistMin", ARGS.mesh_dist_min)
        gmsh.model.mesh.field.setNumber(2, "DistMax", ARGS.mesh_dist_max)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(2)
        nodes, elements = _extract_linear_triangle_mesh()

        # Physics Solver (P2 Precision, Dirichlet BC = TM modes)
        vals, vecs = solve_dirichlet_eigenmodes(nodes, elements, ARGS.n_eigen_modes, ARGS.eigen_sigma)
        freqs = eigenvalues_to_ghz(vals)

        return {
            'id': s_id, 'nodes': nodes, 'elements': elements,
            'freqs': freqs, 'vecs': vecs, 'n_nodes': len(nodes),
            'shape_type': shape_type, 'geom_params': geom_params,
        }
    except Exception as e:
        print(f"Error generating sample {s_id}: {e}")
        return None
    finally:
        try:
            gmsh.model.remove()
        except Exception:
            pass

def save_sample_plot(data, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    nodes, elements = data['nodes'], data['elements']
    triang = Triangulation(nodes[:, 0], nodes[:, 1], elements)

    # Mesh Panel
    axes[0].triplot(triang, color='gray', linewidth=0.15, alpha=0.5)
    axes[0].set_title(f"ID {data['id']}: {len(elements)} Elements")

    n_show = min(3, data['vecs'].shape[1])  # --n_eigen_modes < 3 ise çökmesin
    for i in range(n_show):
        # Taking up to first n_nodes from P2 precision for visualization
        mode_v = data['vecs'][:data['n_nodes'], i]
        mode_norm = mode_v / (np.max(np.abs(mode_v)) + 1e-30)
        axes[i+1].tripcolor(triang, mode_norm, shading='gouraud', cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i+1].set_title(f"Mode {i+1}: {data['freqs'][i]:.4f} GHz")

    for ax in axes: ax.set_aspect('equal'); ax.axis('off')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def write_sample(f_h5, res):
    grp = f_h5.create_group(f"sample_{res['id']:04d}")
    grp.create_dataset("nodes", data=res['nodes'], compression="gzip", compression_opts=4)
    grp.create_dataset("elements", data=res['elements'], compression="gzip", compression_opts=4)
    grp.create_dataset("freqs", data=res['freqs'])
    # vecs: full P2 DOF vector [n_p2_dofs, K]; rows 0..n_nodes-1 are the mesh vertices.
    grp.create_dataset("vecs", data=res['vecs'], compression="gzip", compression_opts=4)
    grp.attrs['shape_type'] = res['shape_type']
    grp.attrs['n_nodes'] = int(res['n_nodes'])
    grp.attrs['fem_element'] = 'P2'
    for k_, v_ in res.get('geom_params', {}).items():
        grp.attrs[k_] = float(v_)
    return grp


def _run_chunk(pool, chunk_range, timeout):
    """Returns (results, hung_ids). A hung sample (e.g. a gmsh deadlock) is skipped
    instead of blocking the whole run forever; the caller then recycles the pool."""
    handles = [(s_id, pool.apply_async(generate_sample_data, (s_id,))) for s_id in chunk_range]
    results, hung = [], []
    for s_id, h in handles:
        try:
            results.append(h.get(timeout=timeout))
        except MPTimeoutError:
            print(f"Sample {s_id} timed out after {timeout:.0f}s, skipped.")
            hung.append(s_id)
    return results, hung


if __name__ == '__main__':
    ARGS = parse_args()

    os.makedirs(ARGS.plot_dir, exist_ok=True)
    n_workers = ARGS.n_workers or cpu_count()
    print(f"🚀 Generating {ARGS.n_total} samples using {n_workers} workers...")

    # Önce geçici dosyaya yaz, sonunda atomik olarak yeniden adlandır: yarıda kesilen
    # bir çalışma son dosya adında bozuk/yarım bir H5 bırakmaz.
    tmp_path = ARGS.h5_filename + ".partial"
    n_ok, n_fail = 0, 0
    pool = Pool(n_workers, initializer=_init_worker, initargs=(ARGS,))
    try:
        with h5py.File(tmp_path, "w") as f_h5:
            f_h5.attrs['metadata'] = json.dumps({
                'generator_args': vars(ARGS), 'bc': 'dirichlet (TM)',
                'fem_element': 'P2', 'freq_unit': 'GHz', 'length_unit': 'm'})
            # Process in chunks to manage memory
            for i in tqdm(range(0, ARGS.n_total, 50), desc="Batch Generation"):
                chunk_range = range(i, min(i + 50, ARGS.n_total))
                chunk_results, hung = _run_chunk(pool, chunk_range, ARGS.sample_timeout)
                if hung:
                    pool.terminate()
                    pool = Pool(n_workers, initializer=_init_worker, initargs=(ARGS,))
                n_fail += len(hung)

                for res in chunk_results:
                    if res is None:
                        n_fail += 1
                        continue

                    # H5 Save (Gzip compression)
                    write_sample(f_h5, res)
                    n_ok += 1

                    # Plot first N_PLOT
                    if res['id'] < ARGS.n_plot:
                        plot_path = os.path.join(ARGS.plot_dir, f"sample_{res['id']:03d}.png")
                        save_sample_plot(res, plot_path)
                f_h5.flush()
    finally:
        pool.terminate()

    if n_ok == 0:
        os.remove(tmp_path)
        raise SystemExit(f"❌ No valid samples generated ({n_fail} failed); nothing written.")
    os.replace(tmp_path, ARGS.h5_filename)
    print(f"\n✅ Completed! {n_ok} samples ({n_fail} failed) \n📦 File: {ARGS.h5_filename} \n🖼️ Plots: {ARGS.plot_dir}/")
