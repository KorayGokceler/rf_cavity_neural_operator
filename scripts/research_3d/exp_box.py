"""E1: PEC box / cube, Whitney N0 elements, kernel handling comparison."""
import time, sys
import numpy as np
from n0lib import box_mesh, assemble_n0, solve_projected, solve_naive, solve_filtered, solve_penalty, box_spectrum

K_MODES = 8
for (a, b, d) in [(1.0, 1.0, 1.0), (1.0, 0.8, 0.6)]:
    ref = box_spectrum(a, b, d)[:K_MODES]
    print(f"\n=== box {a}x{b}x{d}; analytic k²[:{K_MODES}] =", np.round(ref, 3))
    for n in (8, 12, 16, 24, 32):
        m = box_mesh(a, b, d, n)
        t0 = time.perf_counter(); A = assemble_n0(m); ta = time.perf_counter() - t0
        vals, vecs, info = solve_projected(A, K_MODES)
        rel = vals / ref - 1
        # verify discrete divergence-free: |Gᵀ M u| relative
        div = np.abs(A['G'].T @ (A['M'] @ vecs)).max() / np.abs(A['M'] @ vecs).max()
        print(f"n={n:3d} tets={m.t.shape[1]:7d} edgeDOF={A['K'].shape[0]:7d} kernel_dim={A['G'].shape[1]:6d} "
              f"asm={ta:.2f}s fac={info['t_factor']:.2f}s eig={info['t_eig']:.2f}s  "
              f"relerr λ: max={np.abs(rel).max():.2e} λ1={rel[0]:+.2e}  divres={div:.1e} LUnnz={info['lu_nnz']/1e6 if info['lu_nnz'] > 0 else float('nan'):.1f}M")
    sys.stdout.flush()

# kernel handling comparison on one mesh
a, b, d = 1.0, 0.8, 0.6
ref = box_spectrum(a, b, d)[:K_MODES]
m = box_mesh(a, b, d, 12)
A = assemble_n0(m)
print("\n=== kernel handling, box 1x0.8x0.6, n=12, DOF", A['K'].shape[0], " λ1_exact", ref[0])
v, _, info = solve_projected(A, K_MODES)
print("projected σ<0     :", np.round(v, 3), f"t={info['t_factor']+info['t_eig']:.2f}s")
v, _, t = solve_naive(A, K_MODES, sigma=0.3 * ref[0])
print("naive σ=0.3λ1      :", np.array2string(v, precision=3), f"t={t:.2f}s  (#≈0: {(np.abs(v)<1e-6*ref[0]).sum()})")
v, _, t = solve_naive(A, K_MODES, sigma=-1.0)
print("naive σ=-1         :", np.array2string(v, precision=3), f"t={t:.2f}s")
for sig in (0.6 * ref[K_MODES - 1], 0.9 * ref[0]):
    try:
        v, _, t, nz = solve_filtered(A, K_MODES, sigma=sig, extra=4)
        print(f"filtered σ={sig:6.1f}  :", np.round(v, 3), f"t={t:.2f}s  zeros_discarded={nz}")
    except Exception as e:
        print("filtered fail", e)
for s in (0.2, 1.0, 5.0):
    v, _, t, nnz = solve_penalty(A, K_MODES, s)
    print(f"penalty s={s:4.1f}      :", np.round(v, 3), f"t={t:.2f}s  nnz(K_s)/nnz(K)={nnz/A['K'].nnz:.1f}")
