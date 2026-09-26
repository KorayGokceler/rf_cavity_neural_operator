"""E5: how to impose n×ψ = 0 on a network-produced vector basis (pillbox R = L = 1).

Trial space = a fixed 'dictionary' standing in for the network head:
N_j = monomial · e_i (degree ≤ p) sampled at the vertices, mapped nodal→edge.
  A  gate+Hodge : ψ_j = w·N_j (w = 3D torsion function, −Δw = 1, w|∂Ω = 0), Ritz with the
                  Schur-complement (discrete divergence-free) mass     ← proposed
  B  gate only  : same ψ, plain Ritz (no projection)
  C  zero-bdry  : ψ_j = N_j, boundary edge DOFs set to 0, projected Ritz
Reference: N0 eigenvalues on the same mesh (projected shift-invert).
"""
import itertools, sys
import numpy as np
from skfem import Basis, ElementTetP1, condense, solve as fsolve
from skfem.models.poisson import laplace, unit_load
from n0lib import gmsh_mesh, assemble_n0, solve_projected, factor, pillbox_spectrum
from exp_ritz_ml import nodal_to_edge, ritz

R, L = 1.0, 1.0
ref = np.array([k for k, _ in pillbox_spectrum(R, L)[:6]])


def torsion(mesh):
    b = Basis(mesh, ElementTetP1())
    return fsolve(*condense(laplace.assemble(b), unit_load.assemble(b), D=b.get_dofs()))


def dictionary(X, p):
    c = X - X.mean(1, keepdims=True)
    out = []
    for a_, b_, c_ in itertools.product(range(p + 1), repeat=3):
        if a_ + b_ + c_ <= p:
            mono = c[0] ** a_ * c[1] ** b_ * c[2] ** c_
            for i in range(3):
                v = np.zeros_like(X); v[i] = mono; out.append(v)
    return out


def vertex_normal(mesh, w):
    """Unit 'wall normal' field n = −∇w/|∇w| at the vertices (volume-averaged P1 gradient
    of the torsion function); the 3D analogue of the repo's dir_bnd feature."""
    t, p = mesh.t, mesh.p
    J = np.stack([p[:, t[i]] - p[:, t[0]] for i in (1, 2, 3)], 1).transpose(2, 0, 1)   # [T,3,3] rows = edges
    dw = np.stack([w[t[i]] - w[t[0]] for i in (1, 2, 3)], 1)                           # [T,3]
    g = np.linalg.solve(J, dw[..., None])[..., 0]                                     # [T,3]
    vol = np.abs(np.linalg.det(J)) / 6
    acc = np.zeros((p.shape[1], 3)); cnt = np.zeros(p.shape[1])
    for i in range(4):
        np.add.at(acc, t[i], g * vol[:, None]); np.add.at(cnt, t[i], vol)
    gv = acc / cnt[:, None]
    return -(gv / np.linalg.norm(gv, axis=1, keepdims=True).clip(1e-12)).T, np.linalg.norm(gv, axis=1)


def ritz_div(V, K, M, G, kp_solve, tol=1e-10):
    """Rank-revealing Ritz with the Schur-complement (divergence-free) mass:
    directions whose projected mass is < tol·max (pure gradients) are dropped."""
    B = G.T @ (M @ V)
    Z = np.column_stack([kp_solve(B[:, j]) for j in range(B.shape[1])])
    GM = V.T @ (M @ V) - B.T @ Z; GM = 0.5 * (GM + GM.T)
    GA = V.T @ (K @ V)
    s, U = np.linalg.eigh(GM)
    keep = s > tol * s.max()
    W = U[:, keep] / np.sqrt(s[keep])
    return np.sort(np.linalg.eigvalsh(W.T @ GA @ W)), GM


for h in (0.18, 0.13, 0.1):
    mesh = gmsh_mesh(lambda occ: occ.addCylinder(0, 0, 0, 0, 0, L, R), h)
    A = assemble_n0(mesh); K, M, G, I = A['K'], A['M'], A['G'], A['I']
    lam_h = solve_projected(A, 6)[0]
    kp_solve, _ = factor((G.T @ M @ G).tocsr())
    w = torsion(mesh); w = w / w.max()
    nrm, _ = vertex_normal(mesh, w)
    # H-field formulation: no essential BC at all (n×curl H = 0 and n·H = 0 are natural);
    # kernel = gradients of ALL P1 functions (drop one vertex: constants have zero gradient)
    AH = dict(K=A['Kfull'].tocsr(), M=A['Mfull'].tocsr(), G=A['Gfull'][:, 1:].tocsr())
    lam_H = solve_projected(AH, 6)[0]
    kpH, _ = factor((AH['G'].T @ AH['M'] @ AH['G']).tocsr())
    print(f"\n### pillbox h={h}: H-formulation N0 DOF={AH['K'].shape[0]}; λ_H/exact−1 = {np.array2string(lam_H / ref - 1, precision=4)}")
    print(f"\n### pillbox h={h}: N0 DOF={K.shape[0]}; λ_h/exact−1 = {np.array2string(lam_h / ref - 1, precision=4)}")
    for p in (2, 3, 4):
        D = dictionary(mesh.p, p)
        VA = np.column_stack([nodal_to_edge(mesh, A['basis'], w * N)[I] for N in D])
        VC = np.column_stack([nodal_to_edge(mesh, A['basis'], N)[I] for N in D])
        # D: tangential gate  ψ = g N + (1−g)(n·N) n,  g = w (normal component left free at the wall)
        VD = np.column_stack([nodal_to_edge(mesh, A['basis'], w * N + (1 - w) * (nrm * N).sum(0) * nrm)[I] for N in D])
        # column scaling + drop numerically dependent columns (rank-revealing QR in the M-norm)
        def clean(V):
            V = V / np.sqrt(np.einsum('ij,ij->j', V, M @ V)).clip(1e-300)
            Q, Rr = np.linalg.qr(V)
            keep = np.abs(np.diag(Rr)) > 1e-8 * np.abs(np.diag(Rr)).max()
            return V[:, keep]
        VH = np.column_stack([nodal_to_edge(mesh, A['basis'], N) for N in D])      # all edges, no gate
        VA, VC, VD = clean(VA), clean(VC), clean(VD)
        VH = VH / np.sqrt(np.einsum('ij,ij->j', VH, AH['M'] @ VH))
        rA = ritz_div(VA, K, M, G, kp_solve)[0][:6]
        rB = ritz(VA, K, M, 1e-12)[:6]
        rC = ritz_div(VC, K, M, G, kp_solve)[0][:6]
        rD = ritz_div(VD, K, M, G, kp_solve)[0][:6]
        f = lambda r: np.array2string(r / lam_h - 1, precision=3, formatter={'float': lambda x: f'{x:+.2e}'})
        print(f" p={p} (m={VA.shape[1]:3d})  A gate+Hodge {f(rA)}")
        print(f"                B gate only  {f(rB)}")
        print(f"                C zero-bdry  {f(rC)}")
        print(f"                D tang.gate  {f(rD)}")
        rH = ritz_div(VH, AH['K'], AH['M'], AH['G'], kpH)[0][:6]
        print(f"                H no gate, H-field: {np.array2string(rH / lam_H - 1, precision=3, formatter={'float': lambda x: f'{x:+.2e}'})}")
    sys.stdout.flush()
