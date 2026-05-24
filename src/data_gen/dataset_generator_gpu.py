#!/usr/bin/env python3
"""
GPU-accelerated RF cavity dataset generator.

Stages:
  1. CPU pool:  gmsh + skfem → dense condensed (Kc, Mc, Ic) per sample
  2. GPU:       bucket-batched Cholesky-whitened eigh → k eigenpairs
  3. CPU:       expand eigenvectors + GHz conversion + HDF5 write

Key differences from dataset_generator.py:
  - Stage 1 workers do NOT call utils.solve_eigen (no ARPACK)
  - Stage 2 uses torch.linalg.eigh on GPU (batched for throughput)
  - Full spectrum diagonalization → take k smallest (physical modes)
  - --gpu_batch_size controls how many FEM systems are solved together

Usage:
    python src/data_gen/dataset_generator_gpu.py \
        --h5_filename rf_cavity_5000_dataset.h5 --n_total 5000
    python src/data_gen/dataset_generator_gpu.py --device cpu  # CPU fallback
    python src/data_gen/dataset_generator_gpu.py --gpu_batch_size 128
"""

import os
import gmsh
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from skfem import Basis, MeshTri, ElementTriP2
from skfem import utils
from skfem.models.poisson import laplace, mass
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from tqdm import tqdm
import logging
import argparse

import torch

logging.getLogger('skfem').setLevel(logging.ERROR)

# ─── Constants ────────────────────────────────────────────────────────────────
# Pad DOFs get K[pad,pad] = LARGE_EIGENVAL → eigenvalue >> physical eigenvalues
# Physical eigenvalues ≈ 1e3–1e5 for these cavity sizes
LARGE_EIGENVAL = 1e9

# Bucket boundaries (pad condensed DOF count to nearest bucket)
BUCKET_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated RF cavity dataset generator."
    )
    # Same interface as dataset_generator.py (drop-in replacement)
    parser.add_argument("--h5_filename",        type=str,   default="rf_cavity_1000_dataset.h5")
    parser.add_argument("--plot_dir",           type=str,   default="dataset_plots")
    parser.add_argument("--n_total",            type=int,   default=1000)
    parser.add_argument("--n_plot",             type=int,   default=100)
    parser.add_argument("--mode",               type=str,   default="random",
                        choices=["random", "calibration"])
    # FEM solver
    parser.add_argument("--n_eigen_modes",      type=int,   default=3)
    parser.add_argument("--eigen_sigma",        type=float, default=500.0,
                        help="Legacy: ignored by GPU solver (kept for run_pipeline compat)")
    # Mesh control
    parser.add_argument("--mesh_size_min",      type=float, default=0.0012)
    parser.add_argument("--mesh_size_max",      type=float, default=0.005)
    parser.add_argument("--mesh_dist_min",      type=float, default=0.002)
    parser.add_argument("--mesh_dist_max",      type=float, default=0.03)
    # Geometry randomization
    parser.add_argument("--sharp_n_pts_range",  type=int,   nargs=2, default=[7, 13],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--sharp_r_range",      type=float, nargs=2, default=[0.02, 0.046],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--smooth_base_r",      type=float, default=0.035)
    parser.add_argument("--smooth_perturb",     type=float, default=0.008)
    parser.add_argument("--smooth_harmonics",   type=int,   nargs=2, default=[2, 8],
                        metavar=("MIN", "MAX"))
    # GPU-specific
    parser.add_argument("--gpu_batch_size",     type=int,   default=64,
                        help="Number of FEM systems per GPU eigensolve batch")
    parser.add_argument("--device",             type=str,   default="auto",
                        choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: CPU Worker — gmsh + skfem (no eigensolve)
# ─────────────────────────────────────────────────────────────────────────────

def assemble_fem_system(args_tuple):
    """
    CPU worker: gmsh mesh generation + skfem P2 FEM assembly.
    Does NOT call utils.solve_eigen; returns raw condensed matrices.

    Args:
        args_tuple: (s_id: int, args_dict: dict)

    Returns:
        dict with keys: id, K_dense, M_dense, Ic, n_dofs_full,
                        n_nodes, nodes, elements, shape_type
        or None on failure.
    """
    s_id, args_dict = args_tuple
    # Reconstruct namespace from dict (workers can't pickle argparse.Namespace)
    args = type('Args', (), args_dict)()

    try:
        if not gmsh.isInitialized():
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 1)

        model_name = f"rf_{s_id}_{os.getpid()}"
        if model_name in gmsh.model.list():
            gmsh.model.remove()
        gmsh.model.add(model_name)

        np.random.seed(s_id * 13)
        cx, cy = 0.05, 0.05

        # ── Geometry ──────────────────────────────────────────────────────
        if args.mode == 'calibration':
            shape_choice = s_id % 3
            if shape_choice == 0:
                shape_type = 'square'
                side = 0.08
                gmsh.model.occ.addRectangle(cx - side/2, cy - side/2, 0, side, side)
            elif shape_choice == 1:
                shape_type = 'circle'
                gmsh.model.occ.addDisk(cx, cy, 0, 0.04, 0.04)
            else:
                shape_type = 'annulus'
                d1 = gmsh.model.occ.addDisk(cx, cy, 0, 0.045, 0.045)
                d2 = gmsh.model.occ.addDisk(cx, cy, 0, 0.02, 0.02)
                gmsh.model.occ.cut([(2, d1)], [(2, d2)])
        else:
            method = np.random.choice(['sharp', 'smooth'])
            if method == 'sharp':
                shape_type = 'random_sharp'
                n_pts = np.random.randint(args.sharp_n_pts_range[0],
                                          args.sharp_n_pts_range[1])
                delta  = 2 * np.pi / n_pts
                angles = np.array([i * delta + np.random.uniform(-delta/3, delta/3)
                                   for i in range(n_pts)])
                r = np.random.uniform(args.sharp_r_range[0], args.sharp_r_range[1], n_pts)
                pts_c = [(cx + ri * np.cos(ai), cy + ri * np.sin(ai))
                         for ri, ai in zip(r, angles)]
            else:
                shape_type = 'random_smooth'
                t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
                r_raw = args.smooth_base_r + sum(
                    np.random.uniform(-args.smooth_perturb, args.smooth_perturb) *
                    np.cos(k * t + np.random.uniform(0, 2 * np.pi))
                    for k in range(args.smooth_harmonics[0], args.smooth_harmonics[1])
                )
                r = np.clip(r_raw, 0.015, None)
                pts_c = [(cx + ri * np.cos(ti), cy + ri * np.sin(ti))
                         for ri, ti in zip(r, t)]

            pts   = [gmsh.model.occ.addPoint(p[0], p[1], 0) for p in pts_c]
            lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % len(pts)])
                     for i in range(len(pts))]
            gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop(lines)])

        gmsh.model.occ.synchronize()

        # ── Adaptive mesh ─────────────────────────────────────────────────
        surfaces = gmsh.model.getEntities(dim=2)
        bnd_lines = []
        for s in surfaces:
            bnd = gmsh.model.getBoundary([s], combined=True,
                                         oriented=False, recursive=False)
            bnd_lines.extend([abs(e[1]) for e in bnd if e[0] == 1])

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", bnd_lines)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField",   1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin",   args.mesh_size_min)
        gmsh.model.mesh.field.setNumber(2, "SizeMax",   args.mesh_size_max)
        gmsh.model.mesh.field.setNumber(2, "DistMin",   args.mesh_dist_min)
        gmsh.model.mesh.field.setNumber(2, "DistMax",   args.mesh_dist_max)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(2)
        _, coords, _ = gmsh.model.mesh.getNodes()
        _, _, conns  = gmsh.model.mesh.getElements(2)
        nodes    = np.ascontiguousarray(coords.reshape(-1, 3)[:, :2])
        elements = np.ascontiguousarray((conns[0].reshape(-1, 3) - 1).astype(np.int32))

        # ── P2 FEM assembly (stiffness + mass, no eigensolve) ────────────
        m_fem  = MeshTri(nodes.T, elements.T)
        basis  = Basis(m_fem, ElementTriP2())
        K_full = laplace.assemble(basis)
        M_full = mass.assemble(basis)
        D      = basis.get_dofs(facets=m_fem.boundary_facets())
        Kc, Mc, _xc, Ic = utils.condense(K_full, M_full, D=D, expand=True)

        return {
            'id':           s_id,
            'K_dense':      np.asarray(Kc.toarray(), dtype=np.float64),
            'M_dense':      np.asarray(Mc.toarray(), dtype=np.float64),
            'Ic':           np.asarray(Ic,            dtype=np.int64),
            'n_dofs_full':  int(K_full.shape[0]),
            'n_nodes':      int(len(nodes)),
            'nodes':        nodes,
            'elements':     elements,
            'shape_type':   shape_type,
        }

    except Exception as e:
        print(f"  ⚠️  Sample {s_id} assembly failed: {e}")
        return None
    finally:
        try:
            gmsh.model.remove()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: GPU Batch Eigensolve
# ─────────────────────────────────────────────────────────────────────────────

def _bucket_for(n: int) -> int:
    """Return the smallest bucket size >= n."""
    for b in BUCKET_SIZES:
        if n <= b:
            return b
    return n   # oversized: process as its own bucket


def _pad_K(K: np.ndarray, target_n: int) -> np.ndarray:
    """
    Pad K from n×n to target_n×target_n.
    Padding block: LARGE_EIGENVAL * I  → padded eigenvalues >> physical.
    """
    n   = K.shape[0]
    out = np.zeros((target_n, target_n), dtype=K.dtype)
    out[:n, :n] = K
    for i in range(n, target_n):
        out[i, i] = LARGE_EIGENVAL
    return out


def _pad_M(M: np.ndarray, target_n: int) -> np.ndarray:
    """
    Pad M from n×n to target_n×target_n.
    Padding block: I  → Cholesky stays stable, padded modes M-normalized.
    """
    n   = M.shape[0]
    out = np.zeros((target_n, target_n), dtype=M.dtype)
    out[:n, :n] = M
    for i in range(n, target_n):
        out[i, i] = 1.0
    return out


def _generalized_eigh_batched(K_t: torch.Tensor,
                               M_t: torch.Tensor,
                               device: torch.device):
    """
    Batched generalized eigenvalue: K x = λ M x   (ascending order)

    Uses Cholesky whitening:
        M = C Cᵀ  (lower-tri Cholesky)
        K_white = C⁻¹ K C⁻ᵀ  →  standard eigh
        x = C⁻ᵀ w  (back-transform)

    Args:
        K_t, M_t: [B, N, N] float64 on device
    Returns:
        vals: [B, N]  ascending eigenvalues
        vecs: [B, N, N]  corresponding eigenvectors (generalized)

    Notes:
        The FEM mass matrix M for P2 elements has entries O(h²) where h is
        the mesh size. The eps regularization must be MUCH smaller than
        min(diag(M)) to avoid corrupting eigenvalues.
        We use eps_rel = 1e-10 × mean(diag(M)) to stay negligible.
    """
    eye = torch.eye(K_t.shape[-1], device=device, dtype=K_t.dtype).unsqueeze(0)

    # Relative regularization: eps ≪ min(diag(M))
    m_diag_mean = M_t.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)  # [B, 1]
    # Broadcast: [B, 1, 1] for matrix addition
    eps_mat = (1e-10 * m_diag_mean).unsqueeze(-1) * eye   # [B, N, N]
    M_reg = M_t + eps_mat

    try:
        C      = torch.linalg.cholesky(M_reg)          # [B, N, N] lower-tri
        C_inv  = torch.linalg.inv(C)                   # [B, N, N]
        K_w    = C_inv @ K_t @ C_inv.transpose(-1, -2) # [B, N, N]
        K_w    = 0.5 * (K_w + K_w.transpose(-1, -2))   # symmetrize (numerical noise)
        vals, W = torch.linalg.eigh(K_w)               # [B,N], [B,N,N] ascending
        vecs   = C_inv.transpose(-1, -2) @ W           # [B, N, N]
    except RuntimeError as e:
        # Cholesky failed (ill-conditioned M) → fall back to standard eigh
        print(f"    ⚠️  Cholesky whitening failed ({e}), falling back to std eigh")
        vals, vecs = torch.linalg.eigh(K_t)

    return vals, vecs


def _cpu_arpack_eigensolver(fem_systems, k: int, sigma: float = 500.0):
    """
    CPU fallback: use scipy ARPACK shift-invert (same algorithm as original
    dataset_generator.py).  Fast for sparse problems with k ≪ N.

    Returns same format as the GPU path: List[(vals_k, vecs_condensed)].
    """
    from scipy.sparse import csr_matrix
    results = [None] * len(fem_systems)
    for i, sys in enumerate(fem_systems):
        if sys is None:
            continue
        try:
            Ks  = csr_matrix(sys['K_dense'])
            Ms  = csr_matrix(sys['M_dense'])
            xc  = np.zeros(sys['n_dofs_full'])
            Ic  = sys['Ic']
            vals_full, vecs_expanded = utils.solve_eigen(
                Ks, Ms, x=xc, I=Ic, k=k, sigma=sigma
            )
            # solve_eigen expands to full DOF space; extract interior rows
            vecs_condensed = np.real(vecs_expanded)[Ic, :]      # [n_cond, k]
            vals_k         = np.real(vals_full[:k])
            # Sort ascending (eigsh may not guarantee strict order)
            order          = np.argsort(vals_k)
            results[i]     = (vals_k[order], vecs_condensed[:, order])
        except Exception as e:
            print(f"  ⚠️  ARPACK failed for sample {sys['id']}: {e}")
    return results


def batch_gpu_eigensolver(fem_systems, k: int, device: torch.device,
                          arpack_sigma: float = 500.0):
    """
    Solve K x = λ M x for all FEM systems.

    Dispatch strategy:
      device == cpu  →  scipy ARPACK per sample (fast for sparse k ≪ N)
      device == cuda →  bucket-batched Cholesky-whitened torch.linalg.eigh

    Args:
        fem_systems: list of dicts from assemble_fem_system (None = failed)
        k:           number of smallest eigenpairs to return
        arpack_sigma: shift for ARPACK (only used on CPU path)

    Returns:
        List[tuple | None]:  (vals_k [k], vecs_condensed [n_cond, k]) or None.
    """
    # ── CPU path: ARPACK (same algorithm as original dataset_generator.py) ──
    if device.type == 'cpu':
        return _cpu_arpack_eigensolver(fem_systems, k, sigma=arpack_sigma)

    # ── CUDA path: bucket-batched dense eigh ─────────────────────────────────
    n_sys   = len(fem_systems)
    results = [None] * n_sys

    # Group by bucket
    buckets = defaultdict(list)   # bucket_n → [(original_index, system_dict)]
    for i, sys in enumerate(fem_systems):
        if sys is None:
            continue
        b = _bucket_for(sys['K_dense'].shape[0])
        buckets[b].append((i, sys))

    for bucket_n, items in buckets.items():
        # Stack padded matrices
        K_list, M_list = [], []
        for _, sys in items:
            K_list.append(_pad_K(sys['K_dense'], bucket_n))
            M_list.append(_pad_M(sys['M_dense'], bucket_n))

        K_t = torch.tensor(np.stack(K_list), dtype=torch.float64, device=device)
        M_t = torch.tensor(np.stack(M_list), dtype=torch.float64, device=device)

        with torch.no_grad():
            vals_t, vecs_t = _generalized_eigh_batched(K_t, M_t, device)

        vals_np = vals_t.cpu().numpy()   # [B, bucket_n]
        vecs_np = vecs_t.cpu().numpy()   # [B, bucket_n, bucket_n]

        for j, (orig_idx, sys) in enumerate(items):
            n_c = sys['K_dense'].shape[0]   # actual interior DOF count

            # Take k smallest eigenvalues/vectors (physical; pad has vals >> physical)
            vals_k = vals_np[j, :k].copy()
            vecs_k = vecs_np[j, :n_c, :k].copy()   # trim padded rows, k cols

            # Sanity: all k eigenvalues must be << LARGE_EIGENVAL
            if np.any(vals_k >= LARGE_EIGENVAL * 0.01):
                print(f"  ⚠️  Sample {sys['id']}: eigenvalue(s) suspiciously large: {vals_k}")

            results[orig_idx] = (vals_k, vecs_k)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Expand eigenvectors + unit conversion
# ─────────────────────────────────────────────────────────────────────────────

def expand_and_postprocess(system, vals_k, vecs_condensed):
    """
    Expand condensed eigenvectors to full P2 DOF space.
    Mirrors skfem.utils.solve_eigen(x=xc, I=Ic) internal expansion.

    Boundary DOFs remain zero (homogeneous Dirichlet BC satisfied).
    Frequencies: f = c * sqrt(|λ|) / (2π)  in GHz.

    Returns:
        dict with keys: id, nodes, elements, n_nodes, freqs, vecs, shape_type
    """
    Ic           = system['Ic']
    n_dofs_full  = system['n_dofs_full']
    k            = vecs_condensed.shape[1]

    vecs_full = np.zeros((n_dofs_full, k), dtype=np.float64)
    vecs_full[Ic, :] = vecs_condensed          # interior DOFs

    # Physical frequency: λ = (2πf/c)²  →  f = c sqrt(λ) / (2π)
    c_light = 299_792_458.0   # m/s
    freqs   = (c_light * np.sqrt(np.abs(np.real(vals_k)))) / (2.0 * np.pi) / 1e9

    return {
        'id':         system['id'],
        'nodes':      system['nodes'],
        'elements':   system['elements'],
        'n_nodes':    system['n_nodes'],
        'freqs':      freqs,
        'vecs':       vecs_full,
        'shape_type': system['shape_type'],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting (identical to dataset_generator.py)
# ─────────────────────────────────────────────────────────────────────────────

def save_sample_plot(data, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    nodes, elements = data['nodes'], data['elements']
    triang = Triangulation(nodes[:, 0], nodes[:, 1], elements)

    axes[0].triplot(triang, color='gray', linewidth=0.15, alpha=0.5)
    axes[0].set_title(f"ID {data['id']}: {len(elements)} elements")

    n_modes = min(3, data['vecs'].shape[1])
    for i in range(n_modes):
        mode_v    = data['vecs'][:data['n_nodes'], i]
        amp       = np.max(np.abs(mode_v)) + 1e-12
        mode_norm = mode_v / amp
        axes[i + 1].tripcolor(triang, mode_norm, shading='gouraud',
                              cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i + 1].set_title(f"Mode {i+1}: {data['freqs'][i]:.4f} GHz")

    for ax in axes:
        ax.set_aspect('equal')
        ax.axis('off')

    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()

    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    os.makedirs(args.plot_dir, exist_ok=True)
    n_cpu = cpu_count()

    print("=" * 60)
    print("🚀 GPU-Accelerated RF Cavity Dataset Generator")
    print("=" * 60)
    print(f"  Samples:       {args.n_total}")
    print(f"  CPU workers:   {n_cpu}")
    print(f"  Compute:       {device}")
    print(f"  GPU batch:     {args.gpu_batch_size}")
    print(f"  Output:        {args.h5_filename}")
    print("=" * 60)

    CHUNK  = args.gpu_batch_size
    # Convert args to dict for subprocess-safe serialization
    args_dict = vars(args)

    n_success = 0
    n_failed  = 0

    with h5py.File(args.h5_filename, 'w') as f_h5:
        with Pool(n_cpu) as pool:
            pbar = tqdm(range(0, args.n_total, CHUNK), desc="Chunks")
            for chunk_start in pbar:
                chunk_ids = list(range(chunk_start,
                                       min(chunk_start + CHUNK, args.n_total)))

                # ── Stage 1: CPU assembly (parallel) ────────────────────
                task_args   = [(s_id, args_dict) for s_id in chunk_ids]
                fem_systems = pool.map(assemble_fem_system, task_args)

                # ── Stage 2: eigensolve (GPU-batched or ARPACK) ─────────
                eigen_results = batch_gpu_eigensolver(
                    fem_systems, k=args.n_eigen_modes, device=device,
                    arpack_sigma=args.eigen_sigma
                )

                # ── Stage 3: Expand + write ──────────────────────────────
                for sys, eig in zip(fem_systems, eigen_results):
                    if sys is None or eig is None:
                        n_failed += 1
                        continue

                    vals_k, vecs_k = eig
                    data = expand_and_postprocess(sys, vals_k, vecs_k)

                    grp = f_h5.create_group(f"sample_{data['id']:04d}")
                    grp.create_dataset('nodes',    data=data['nodes'],
                                       compression='gzip', compression_opts=4)
                    grp.create_dataset('elements', data=data['elements'],
                                       compression='gzip', compression_opts=4)
                    grp.create_dataset('freqs',    data=data['freqs'])
                    grp.create_dataset('vecs',     data=data['vecs'],
                                       compression='gzip', compression_opts=4)
                    grp.attrs['shape_type'] = data['shape_type']

                    if data['id'] < args.n_plot:
                        plot_path = os.path.join(
                            args.plot_dir, f"sample_{data['id']:03d}.png"
                        )
                        save_sample_plot(data, plot_path)

                    n_success += 1

                pbar.set_postfix(ok=n_success, fail=n_failed)

    print("\n" + "=" * 60)
    print(f"✅  Completed!")
    print(f"    Success:  {n_success} / {args.n_total}")
    print(f"    Failed:   {n_failed}")
    print(f"    File:     {args.h5_filename}")
    print(f"    Plots:    {args.plot_dir}/")
    print("=" * 60)
