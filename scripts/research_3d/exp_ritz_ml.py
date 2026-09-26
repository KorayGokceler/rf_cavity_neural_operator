"""E4: what happens to the EigenspaceOperator's Rayleigh–Ritz / span / compliance
machinery in H(curl)?  PEC box 1×0.8×0.6, Whitney N0, n = 12 and 24.

 (1) gradient contamination of a perfect span → Ritz values drop below λ (no upper bound)
 (2) one pure-gradient column → a Ritz value 0, compliance tr(G_A⁻¹G_M) → 1/ridge
 (3) Schur-complement ("divergence-free") mass  M_div = VᵀMV − BᵀKp⁻¹B,  B = GᵀMV
     restores exact Ritz values (= Ritz on span(V) ⊕ grad P1, zeros dropped)
 (4) 'ideal network' floor: exact analytic modes sampled at the VERTICES, mapped
     nodal→edge (trapezoid), vs. the canonical edge-integral interpolant
 (5) random smooth nodal vector fields (untrained-network proxy): unprojected vs projected
"""
import sys, time
import numpy as np
from n0lib import box_mesh, assemble_n0, solve_projected, box_spectrum, factor

a, b, d = 1.0, 0.8, 0.6


def analytic_modes(nmax=4):
    """(k², field function) for the PEC box, both polarisations when m,n,p ≥ 1."""
    out = []
    for m_ in range(nmax):
        for n_ in range(nmax):
            for p_ in range(nmax):
                kv = np.array([m_ * np.pi / a, n_ * np.pi / b, p_ * np.pi / d])
                z = int((kv == 0).sum())
                if z >= 2:
                    continue
                if z == 1:
                    pols = [np.eye(3)[int(np.argmin(np.abs(kv)))]]
                else:
                    u = np.cross(kv, [1.0, 0.3, 0.1]); u /= np.linalg.norm(u)
                    w = np.cross(kv, u); w /= np.linalg.norm(w)
                    pols = [u, w]
                for A in pols:
                    def f(x, A=A, kv=kv):
                        cx, sx = np.cos(kv[0] * x[0]), np.sin(kv[0] * x[0])
                        cy, sy = np.cos(kv[1] * x[1]), np.sin(kv[1] * x[1])
                        cz, sz = np.cos(kv[2] * x[2]), np.sin(kv[2] * x[2])
                        return np.stack([A[0] * cx * sy * sz, A[1] * sx * cy * sz, A[2] * sx * sy * cz])
                    out.append((float(kv @ kv), f))
    out.sort(key=lambda t: t[0])
    return out


def nodal_to_edge(mesh, basis, En):
    e = mesh.edges; t = mesh.p[:, e[1]] - mesh.p[:, e[0]]
    u = np.zeros(basis.N); u[basis.edge_dofs[0]] = 0.5 * ((En[:, e[0]] + En[:, e[1]]) * t).sum(0)
    return u


def edge_interp(mesh, basis, f, nq=4):
    e = mesh.edges; pa, pb = mesh.p[:, e[0]], mesh.p[:, e[1]]
    xq, wq = np.polynomial.legendre.leggauss(nq); xq, wq = 0.5 * (xq + 1), 0.5 * wq
    v = sum(w * (f(pa + s * (pb - pa)) * (pb - pa)).sum(0) for s, w in zip(xq, wq))
    u = np.zeros(basis.N); u[basis.edge_dofs[0]] = v
    return u


def ritz(V, K, M, ridge=0.0):
    GA, GM = V.T @ (K @ V), V.T @ (M @ V)
    GM = GM + ridge * np.trace(GM) / len(GM) * np.eye(len(GM))
    L = np.linalg.cholesky(GM)
    Li = np.linalg.inv(L)
    return np.sort(np.linalg.eigvalsh(Li @ GA @ Li.T))


def ritz_div(V, K, M, G, kp_solve):
    """Ritz with the Schur-complement mass (projection onto discrete div-free fields)."""
    B = G.T @ (M @ V)
    Z = np.column_stack([kp_solve(B[:, j]) for j in range(B.shape[1])])
    GM = V.T @ (M @ V) - B.T @ Z
    GA = V.T @ (K @ V)
    L = np.linalg.cholesky(0.5 * (GM + GM.T)); Li = np.linalg.inv(L)
    return np.sort(np.linalg.eigvalsh(Li @ GA @ Li.T)), GM


def compliance(V, K, M, ridge=1e-9):
    GA, GM = V.T @ (K @ V), V.T @ (M @ V)
    dA = 1 / np.sqrt(np.diag(GA)); GAs = GA * dA[:, None] * dA[None]; GMs = GM * dA[:, None] * dA[None]
    return np.trace(np.linalg.solve(GAs + ridge * np.eye(len(GA)), GMs))


def span_residual_M(V, T, M, GVV=None):
    GVV = V.T @ (M @ V) if GVV is None else GVV
    g = V.T @ (M @ T)
    cap = np.einsum('ij,ij->j', g, np.linalg.solve(GVV, g))
    return 1 - cap / np.einsum('ij,ij->j', T, M @ T)


if __name__ == "__main__":
    for n in (12, 24, 32):
        mesh = box_mesh(a, b, d, n); A = assemble_n0(mesh); K, M, G, I = A['K'], A['M'], A['G'], A['I']
        ref = box_spectrum(a, b, d)[:8]
        lam_h, T, _ = solve_projected(A, 8)
        T = T / np.sqrt(np.einsum('ij,ij->j', T, M @ T))
        t0 = time.perf_counter(); kp_solve, _ = factor((G.T @ M @ G).tocsr()); t_kp = time.perf_counter() - t0
        print(f"\n### box n={n}: N0 DOF={K.shape[0]}, interior vertices={G.shape[1]}, Kp factor {t_kp*1e3:.0f} ms")
        print("discrete λ_h / exact − 1:", np.array2string(lam_h / ref - 1, precision=2))
        rng = np.random.default_rng(1)
        # (1) contaminate the exact span with gradients of smooth random potentials
        xyz = mesh.p[:, A['Iv']]
        phis = np.stack([np.sin(np.pi * xyz[0] / a) * np.sin(np.pi * xyz[1] / b) * np.sin(np.pi * xyz[2] / d)
                         * (1 + 0.5 * np.cos((j + 1) * xyz[0] + 2 * xyz[1] - j * xyz[2])) for j in range(6)], 1)
        Gphi = G @ phis
        Gphi = Gphi / np.sqrt(np.einsum('ij,ij->j', Gphi, M @ Gphi))
        for eps in (0.1, 0.3, 1.0):
            V = T[:, :6] + eps * Gphi
            r_plain = ritz(V, K, M); r_div, GMd = ritz_div(V, K, M, G, kp_solve)
            rM_plain = span_residual_M(V, T[:, :6], M); rM_div = span_residual_M(V, T[:, :6], M, GMd)
            print(f"(1) ε={eps:3.1f} (gradient mass fraction {eps**2/(1+eps**2):.2f}): Ritz/λ_h−1 plain "
                  f"{np.array2string(r_plain / lam_h[:6] - 1, precision=3)}  | Schur-projected max|.| {np.abs(r_div / lam_h[:6] - 1).max():.1e}"
                  f" | span r_M plain {rM_plain.mean():.3f}, projected {rM_div.mean():.1e}")
        # (2) a pure gradient column
        V = np.column_stack([T[:, :6], Gphi[:, 0]])
        B = G.T @ (M @ V); Z = np.column_stack([kp_solve(B[:, j]) for j in range(B.shape[1])])
        GMd = V.T @ (M @ V) - B.T @ Z; dd = 1 / np.sqrt(np.diag(V.T @ (M @ V)))
        ev = np.linalg.eigvalsh(GMd * dd[:, None] * dd[None])
        print(f"(2) + 1 gradient column: plain Ritz[0:3] = {np.array2string(ritz(V, K, M, 1e-12)[:3], precision=3)}  "
              f"compliance plain = {compliance(V, K, M):.3e} (clean span: {compliance(T[:, :6], K, M):.3e}); "
              f"projected mass Gram: smallest normalised eigenvalue {ev[0]:.1e} (column annihilated → drop/ridge)")
        # (4) ideal-network floor: exact modes sampled at vertices → edges
        modes = analytic_modes()[:8]
        Vn = np.column_stack([nodal_to_edge(mesh, A['basis'], f(mesh.p))[I] for _, f in modes])
        Ve = np.column_stack([edge_interp(mesh, A['basis'], f)[I] for _, f in modes])
        exact = np.array([k2 for k2, _ in modes])
        for nm, V in (('nodal→edge', Vn), ('edge-integral', Ve)):
            rq = np.einsum('ij,ij->j', V, K @ V) / np.einsum('ij,ij->j', V, M @ V)
            rd = ritz_div(V, K, M, G, kp_solve)[0]
            print(f"(4) {nm:13s}: RQ/exact−1 (per mode) {np.array2string(rq / exact - 1, precision=2)}; "
                  f"Ritz(span8, projected)/λ_h−1 max {np.abs(rd / lam_h - 1).max():.2e}; /exact−1 max {np.abs(rd / exact - 1).max():.2e}")
        # (5) random smooth nodal fields (proxy of an untrained head), m = 16
        X = mesh.p
        W = rng.standard_normal((16, 3, 3)); ph = rng.uniform(0, 2 * np.pi, (16, 3))
        En = [np.stack([np.sin(W[j, c] @ (X * 3) + ph[j, c]) for c in range(3)]) for j in range(16)]
        V = np.column_stack([nodal_to_edge(mesh, A['basis'], e)[I] for e in En])
        Pmass = np.einsum('ij,ij->j', V, M @ V)
        rd, GMd = ritz_div(V, K, M, G, kp_solve)
        print(f"(5) random smooth fields m=16: gradient mass fraction per column median "
              f"{np.median(1 - np.diag(GMd) / Pmass):.2f}; plain Ritz[0:3]={np.array2string(ritz(V, K, M)[:3], precision=2)} "
              f"projected Ritz[0:3]={np.array2string(rd[:3], precision=2)} (λ1_h={lam_h[0]:.2f})")
        t0 = time.perf_counter(); ritz_div(V, K, M, G, kp_solve); print(f"    Schur-projected Gram for m=16: {1e3*(time.perf_counter()-t0):.0f} ms (after Kp factor)")
        sys.stdout.flush()
