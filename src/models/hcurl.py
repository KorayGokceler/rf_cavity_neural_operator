"""H(curl) / Whitney-N0 linear algebra for the 3D Maxwell eigenspace operator
(docs/18 §3.4, docs/20).

Batching convention (maxwell3d_collate): every sparse matrix of a batch is ONE
block-diagonal torch COO tensor laid out on the PADDED index space, so sample b's
edge e is row b·Ne + e and its vertex v is column b·Nv + v:

    M, K : [B·Ne, B·Ne]   N0 mass / curl–curl
    G    : [B·Ne, B·Nv]   discrete gradient (G·1 = 0, K·G = 0),  Gt = Gᵀ
    Kp   : [B·Nv, B·Nv]   GᵀMG (P1 Neumann Laplacian, singular: constants)

Padded edges / vertices are empty rows and columns, so a padded tensor
X [B, N, m] multiplies as spmm(A, X) = (A @ X.reshape(B·N, m)).view(B, ·, m),
one sparse product for the whole batch.

The kernel of K is ∇P1.  A basis V is replaced by its M-orthogonal projection
onto discrete divergence-free fields,

    PV = V − G Z,   Z = Kp⁻¹ B,   B = GᵀMV,

without ever forming PV's Grams from scratch (K G = 0):

    (PV)ᵀK(PV) = VᵀKV,      (PV)ᵀM(PV) = VᵀMV − BᵀZ =: M_div   (Schur complement).

Kp's null space (constants on each connected mesh) needs no pinning and no
ridge: 1ᵀB = (G1)ᵀMV = 0, so Kp z = B is consistent, CG on it converges in
range(Kp), and whatever constant a solution carries is annihilated by both
uses of Z (G·const = 0 and Bᵀconst = 0); numerically Z is kept mean-free
(see _pcg) so that the rounding of those two zeros is not amplified by a
drifting constant.  (Assumes a connected mesh; a mesh with several
components would need one mean per component.)  The singular Neumann system is
also better conditioned than a pinned one (λ_min = the Fiedler value instead
of a point-constraint value O(h·λ₂)).
"""
import warnings

import torch

from src.models.spectral_no import _BroadenedEigh

SPARSE_KEYS = ('M', 'K', 'G', 'Gt', 'Kp')


def spmm(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Block-diagonal sparse A [B·R, B·C] @ padded X [B, C, m] → [B, R, m]."""
    B, C, m = X.shape
    return torch.sparse.mm(A, X.reshape(B * C, m)).view(B, -1, m)


def bgram(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Batched XᵀY: [B, N, p], [B, N, q] → [B, p, q]."""
    return torch.einsum('bnp,bnq->bpq', X, Y)


def _mean_free(x, valid):
    """Remove each (sample, column)'s mean over the valid vertices."""
    n = valid.sum(1, keepdim=True).clamp(min=1)
    return (x - (x * valid).sum(1, keepdim=True) / n) * valid


def _pcg(Kp, dinv, rhs, tol, maxiter, check_every=10):
    """Jacobi-preconditioned CG for Kp X = rhs, rhs [B, Nv, m]; every
    (sample, column) has its own step sizes.  rhs and X are kept mean-free
    over the valid vertices: the Jacobi-preconditioned iterates otherwise
    drift along the null space (constants), which is harmless in exact
    arithmetic but multiplies the rounding of 1ᵀGᵀ(·) ≈ 0 in later products.
    Columns with rhs = 0 (padding) stay 0.  Returns (X, iterations)."""
    tiny = torch.finfo(rhs.dtype).tiny
    valid = (dinv > 0).to(rhs.dtype)
    rhs = _mean_free(rhs, valid)
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    z = dinv * r
    p = z.clone()
    rz = (r * z).sum(1, keepdim=True)
    thr = tol * rhs.norm(dim=1, keepdim=True)
    it, converged = 0, False
    for it in range(1, maxiter + 1):
        Ap = spmm(Kp, p)
        pAp = (p * Ap).sum(1, keepdim=True)
        alpha = torch.where(pAp > 0, rz / pAp.clamp(min=tiny), torch.zeros_like(pAp))
        x.add_(alpha * p)
        r.sub_(alpha * Ap)
        if it % check_every == 0 and bool((r.norm(dim=1, keepdim=True) <= thr).all()):
            converged = True
            break
        z = dinv * r
        rz_new = (r * z).sum(1, keepdim=True)
        beta = torch.where(rz > 0, rz_new / rz.clamp(min=tiny), torch.zeros_like(rz))
        p = z + beta * p
        rz = rz_new
    if not converged:
        warnings.warn(f"KpSolve: CG stopped at maxiter={maxiter} before reaching tol={tol:g}; "
                      "the kernel projection (M_div) may be inaccurate.")
    return _mean_free(x, valid), it


class KpSolve(torch.autograd.Function):
    """Z = Kp⁻¹ B by batched PCG (float64, works on CPU and GPU).  Kp is
    constant and symmetric, so the backward pass is one more solve with the
    same operator: B̄ = Kp⁻¹ Z̄ (Z̄ ⟂ constants too, being a combination of B
    and Gᵀ(·) columns).  KpSolve.last_iters holds the latest forward count."""
    last_iters = 0

    @staticmethod
    def forward(ctx, rhs, Kp, dinv, tol, maxiter):
        Z, KpSolve.last_iters = _pcg(Kp, dinv, rhs, tol, maxiter)
        ctx.Kp, ctx.dinv, ctx.tol, ctx.maxiter = Kp, dinv, tol, maxiter
        return Z

    @staticmethod
    def backward(ctx, gZ):
        gB, _ = _pcg(ctx.Kp, ctx.dinv, gZ.contiguous(), ctx.tol, ctx.maxiter)
        return gB, None, None, None, None


def _sparse(batch, key, dtype):
    A = batch[key]
    return A if A.dtype == dtype else A.to(dtype)


def project_basis(V, batch, tol=1e-8, maxiter=2000):
    """Kernel-projected Grams of a padded edge basis V [B, Ne, m] (float64).

    Returns dict:
        A_V   [B, m, m]  VᵀKV = (PV)ᵀK(PV)      (curl–curl Gram)
        M_V   [B, m, m]  VᵀMV                     (unprojected mass Gram)
        M_div [B, m, m]  VᵀMV − BᵀKp⁻¹B = (PV)ᵀM(PV)
        Z     [B, Nv, m] Kp⁻¹GᵀMV  (PV = V − G Z)
    """
    V = V.double()          # float32 CG does not converge (M_div off by ~79%, docs/21 B3)
    dt = V.dtype
    M, K, Gt, Kp = (_sparse(batch, k, dt) for k in ('M', 'K', 'Gt', 'Kp'))
    diag = batch['Kp_diag'].to(dt)
    dinv = torch.where(diag > 0, 1.0 / diag.clamp(min=1e-300), torch.zeros_like(diag)).unsqueeze(-1)
    MV = spmm(M, V)
    Bm = spmm(Gt, MV)                                        # GᵀMV [B, Nv, m]
    Z = KpSolve.apply(Bm, Kp, dinv, tol, maxiter)
    M_V = bgram(V, MV)
    M_div = M_V - bgram(Bm, Z)
    A_V = bgram(V, spmm(K, V))
    sym = lambda A: 0.5 * (A + A.transpose(-1, -2))          # noqa: E731
    return {'A_V': sym(A_V), 'M_V': sym(M_V), 'M_div': sym(M_div), 'Z': Z}


def projected(V, Z, batch):
    """PV = V − G Z  [B, Ne, m]."""
    return V - spmm(_sparse(batch, 'G', V.dtype), Z)


def projected_eigh(M_V, M_div, A_V, K, mass_ridge=1e-8, drop_tol=1e-6, eig_broadening=1e-4):
    """K lowest pairs of  A_V c = θ M_div c  with gradient directions DROPPED.

    M_div is singular exactly on the directions c with PVc = 0 (span(V) ∩ ∇P1),
    where A_V vanishes too: 0/0.  A ridge would give them θ = ridge_A/ridge_M,
    an arbitrary value inside the spectrum (_ritz: the mean Rayleigh quotient),
    and the span/compliance losses never see gradient content, so nothing
    would keep training away from it.  Instead (rank-revealing):
      1. whiten by the UNPROJECTED mass: D M_V D + ridge·I = C Cᵀ (D Jacobi);
      2. μ, Q = eig(C⁻¹ D M_div D C⁻ᵀ): μ ∈ [0, 1] is the divergence-free mass
         fraction of each direction;
      3. keep μ > drop_tol, W = C⁻ᵀ Q μ^{-1/2}; dropped directions get θ = ∞
         (a large finite value, sorted last);
      4. θ, y = eig(Wᵀ D A_V D W), c = D W y  (M_div-orthonormal).
    Every kept θ is the Rayleigh quotient of a nonzero divergence-free field,
    so θ_k ≥ λ_h,k (min–max): no θ = 0, no Ritz value pulled below λ_h.
    float64; both eighs use the broadened backward.  Returns θ [B,K], c [B,m,K].
    """
    diag = torch.diagonal(M_V, dim1=-2, dim2=-1)
    d = torch.where(diag > 0, diag.clamp(min=1e-300).rsqrt(), torch.zeros_like(diag))
    Dm = lambda A: A * d.unsqueeze(-1) * d.unsqueeze(-2)     # noqa: E731
    eye = torch.eye(M_V.shape[-1], dtype=M_V.dtype, device=M_V.device)
    C = torch.linalg.cholesky(Dm(M_V) + mass_ridge * eye)
    Ci = torch.linalg.solve_triangular(C, eye.expand_as(C), upper=False)       # C⁻¹
    S = Ci @ Dm(M_div) @ Ci.transpose(-1, -2)
    mu, Q = _BroadenedEigh.apply(0.5 * (S + S.transpose(-1, -2)), eig_broadening)
    keep = mu > drop_tol
    W = Ci.transpose(-1, -2) @ Q * torch.where(keep, mu.clamp(min=drop_tol).rsqrt(), torch.zeros_like(mu)
                                              ).unsqueeze(-2)
    Aw = W.transpose(-1, -2) @ Dm(A_V) @ W
    Aw = 0.5 * (Aw + Aw.transpose(-1, -2))
    # 'θ = ∞' for dropped directions: far above every kept θ, and a fixed floor
    # (1e12 ≫ any normalised-mesh eigenvalue, ~1/h²) for samples that keep no
    # direction — there Aw = 0 and A_V = 0 too, so no V-based scale exists
    # (was 1.0, below the physical spectrum; docs/21 B1).  No gradient flows.
    big = (1e3 * torch.diagonal(Aw, dim1=-2, dim2=-1).abs().amax(-1, keepdim=True) + 1e12).detach()
    Aw = Aw + torch.diag_embed(torch.where(keep, torch.zeros_like(mu), big))
    theta, y = _BroadenedEigh.apply(Aw, eig_broadening)
    return theta[:, :K], d.unsqueeze(-1) * (W @ y[:, :, :K])


def hcurl_ritz(V, batch, K, mass_ridge=1e-8, drop_tol=1e-6, eig_broadening=1e-4,
               tol=1e-8, maxiter=2000):
    """Rayleigh–Ritz on the divergence-free projection of span(V) → K lowest.

    V [B, Ne, m] float64.  Returns (θ [B, K], fields [B, Ne, K] = PV·c with
    unit M-norm, proj dict of project_basis).
    """
    proj = project_basis(V, batch, tol, maxiter)
    PV = projected(V, proj['Z'], batch)
    lam, c = projected_eigh(proj['M_V'], proj['M_div'], proj['A_V'], K,
                            mass_ridge, drop_tol, eig_broadening)
    nrm2 = torch.einsum('bmk,bml,blk->bk', c, proj['M_div'], c)
    modes = torch.einsum('bem,bmk->bek', PV, c) / nrm2.clamp(min=1e-300).sqrt().unsqueeze(1)
    return lam, modes, proj


def hcurl_grams(V, T, proj, batch):
    """Mass / curl–curl Grams of [PV | T] for the span losses, float64.

    V [B, Ne, m] raw basis, T [B, Ne, n] target edge DOFs (all stored modes),
    proj from project_basis (reuses its M_div, A_V, Z: no second Kp solve).
    The cross block is exact for any T:  (PV)ᵀMT = VᵀMT − Zᵀ(GᵀMT).
    Returns G_M, G_A [B, m+n, m+n] and MT [B, Ne, n].
    """
    dt = torch.float64
    V, T = V.to(dt), T.to(dt)
    M, K, Gt = (_sparse(batch, k, dt) for k in ('M', 'K', 'Gt'))
    MT, KT = spmm(M, T), spmm(K, T)
    C_M = bgram(V, MT) - bgram(proj['Z'], spmm(Gt, MT))
    C_A = bgram(V, KT)

    def block(VV, VT, TT):
        return torch.cat([torch.cat([VV, VT], -1),
                          torch.cat([VT.transpose(-1, -2), TT], -1)], -2)
    G_M = block(proj['M_div'], C_M, bgram(T, MT))
    G_A = block(proj['A_V'], C_A, bgram(T, KT))
    return G_M, G_A, MT


def mode_rel_l2(F, T, MT, clusters):
    """Per-mode M-norm rel-L2 of predicted edge fields F [Ne, K] vs targets
    T [Ne, K] (one sample, MT = M·T): sign-agnostic for isolated modes
    (min_s ‖f − s t̂‖_M), subspace error ‖t − Π_F t‖_M/‖t‖_M inside each
    near-degenerate cluster (list of index lists).  F must be M-orthonormal,
    which the projected Ritz output is by construction.  float64 [K]."""
    F, T, MT = F.double(), T.double(), MT.double()
    C = F.T @ MT                                            # f_iᵀ M t_j
    tt = (T * MT).sum(0).clamp(min=1e-300)
    out = torch.zeros(T.shape[1], dtype=F.dtype, device=F.device)
    for cl in clusters:
        idx = torch.as_tensor(cl, device=F.device)
        Cc = C[idx][:, idx]
        if len(cl) == 1:
            cos = Cc[0, 0].abs() / tt[idx[0]].sqrt()
            out[idx[0]] = (2.0 - 2.0 * cos.clamp(max=1.0)).clamp(min=0).sqrt()
        else:
            cap = (Cc ** 2).sum(0)                           # ‖Π_F t‖² = Σ_i (f_iᵀMt)²
            out[idx] = (1.0 - cap / tt[idx]).clamp(min=0).sqrt()
    return out
