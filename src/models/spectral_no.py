"""Spectral Neural Operator (SpectralNO) via *physical* Galerkin reduction.

Architecture overview
---------------------
1. **Local geometry encoder** – RFF coordinates + node features → per-node
   embeddings [B, N, embed_dim].
2. **Basis network** – pointwise MLP → M basis function values ψ_m(x) per
   node, multiplied by a soft Dirichlet gate so ψ_m = 0 on ∂Ω.
3. **Physical Galerkin matrices** – the stiffness (L) and mass (M) matrices are
   *assembled from the basis itself* by numerical quadrature over the nodes,
   exactly like a finite-element Galerkin projection of the Dirichlet Laplacian:

       M_mn = ∫_Ω ψ_m ψ_n dΩ            ≈ Σ_i w_i ψ_m(x_i) ψ_n(x_i)
       L_mn = ∫_Ω ∇ψ_m · ∇ψ_n dΩ        ≈ Σ_i w_i ∇ψ_m(x_i) · ∇ψ_n(x_i)

   where the quadrature weights w_i are the per-node mesh areas (input feature
   column 5).  ∇ψ is obtained by autograd because the basis network is
   *pointwise*, so a single summed backward yields every node's gradient.
   For ∇ψ to be the *total* spatial derivative, every input that is a known
   function of x must depend on the autograd leaf: the (x, y) feature columns
   are replaced by the differentiable coordinates, and dist_bnd (which drives
   the Dirichlet gate) is linearised with its exact gradient ∇d = −dir_bnd.
   The remaining per-node features (areas, principal angles, curvature, …)
   have no available spatial derivative and act as frozen conditioning.

   This is the key change vs. the previous design: L and M are **no longer
   predicted by free per-geometry MLP heads** (which mapped a 128-d global
   "fingerprint" → an arbitrary 32×32 SPD matrix and thereby memorised the
   training set).  The only learned object is now the *function space* ψ(x);
   the operator is the true Laplacian projected onto it, so it generalises.
4. **Generalized eigendecomposition** – solve L u_k = λ_k M u_k via Cholesky
   whitening + `eigh`, in float64 with autocast disabled; eigenvalues are
   returned sorted ascending, so no OT matching is needed at training or
   inference time.  The eigh backward uses Lorentzian broadening of the
   1/(λ_j − λ_i) terms so (near-)degenerate spectra give finite gradients.
5. **Eigenfunction reconstruction** – φ_k(x) = Σ_m u_{k,m} · ψ_m(x), then
   rescaled so its signed peak over valid nodes is +1 — the same max-abs
   convention as the FEM targets (dataset_converter divides by max|Y|).
   The raw M-normalised fields have Σ w φ² = 1, a fixed amplitude that a
   max-normalised target cannot match (irreducible rel-L2 floor).
6. **Frequency transform** – small *monotone* MLP on log λ maps Galerkin
   (Laplacian) eigenvalues → z-scored physical frequencies, so the frequency
   order always equals the eigenvalue (= field) order.

Key properties
--------------
* Ordering is **structural** (eigh returns sorted eigenvalues; the frequency
  head is strictly increasing).
* M-orthogonality of eigenfunctions is **structural** (the peak rescaling is
  per-column, so orthogonality is preserved).
* Dirichlet BC (φ = 0 on ∂Ω) is built into the basis via the boundary gate.
* Sign / degenerate-subspace ambiguity is handled by the sign-agnostic /
  Grassmannian field loss (see lightning_module).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.gnot import (RandomFourierFeatures, _normalize_peak, _check_val_dim,
                             _physics_freq_z)


# ─────────────────────────────────────────────────────────────────────────────
#  Numerical helpers
# ─────────────────────────────────────────────────────────────────────────────

class _BroadenedEigh(torch.autograd.Function):
    """Symmetric ``eigh`` whose backward is finite for degenerate spectra.

    The exact eigenvector gradient is  gA = V (F ∘ VᵀḡV + diag(ḡλ)) Vᵀ  with
    F_ij = 1 / (λ_j − λ_i), which is ±inf/NaN for (near-)degenerate pairs —
    the normal case for RF cavities (dipole/quadrupole pairs).  We use the
    Lorentzian-broadened  F_ij = Δ / (Δ² + ε),  Δ = λ_j − λ_i, which is exact
    for |Δ| ≫ √ε and bounded by 1/(2√ε).  Subspace-invariant losses (the
    Grassmannian loss on degenerate clusters) have zero gradient along those
    directions anyway, so the broadening only removes numerical garbage.
    """

    @staticmethod
    def forward(ctx, A, eps):
        vals, vecs = torch.linalg.eigh(A)
        ctx.save_for_backward(vals, vecs)
        ctx.eps = eps
        return vals, vecs

    @staticmethod
    def backward(ctx, g_vals, g_vecs):
        vals, vecs = ctx.saved_tensors
        vecs_t = vecs.transpose(-1, -2)
        inner = torch.zeros_like(vecs)
        if g_vecs is not None:
            diff = vals.unsqueeze(-2) - vals.unsqueeze(-1)       # [.., i, j] = λ_j − λ_i
            inner = diff / (diff * diff + ctx.eps) * (vecs_t @ g_vecs)
        if g_vals is not None:
            inner = inner + torch.diag_embed(g_vals)
        g_A = vecs @ inner @ vecs_t
        return 0.5 * (g_A + g_A.transpose(-1, -2)), None


def _p1_galerkin(basis: torch.Tensor, X: torch.Tensor, elems: torch.Tensor):
    """Exact P1 mass / stiffness Gram matrices of nodal basis values.

    ψ_m is read as the piecewise-linear interpolant of basis[:, :, m] on the
    mesh triangles, so per triangle T with vertices (0, 1, 2):
        ∇ψ = g / (2A_s),  g = Σ_i ψ_i · R(p_{i+2} − p_{i+1})   (R = rot −90°)
        L_T = g gᵀ / (4|A|),    M_T = |A|/12 · (Σ_i ψ_i ψ_iᵀ + s sᵀ),  s = Σ_i ψ_i
    (consistent mass).  Padded triangles (0, 0, 0) have zero area and add 0.

    Args:
        basis [B, N, M], X [B, N, 2], elems [B, T, 3] (node indices).
    Returns:
        M_mat, L_mat [B, M, M].
    """
    b = torch.arange(basis.shape[0], device=basis.device)[:, None, None]
    idx = elems.long()
    P = X[b, idx]                                     # [B, T, 3, 2]
    Psi = basis[b, idx]                               # [B, T, 3, M]
    Pj, Pk = P.roll(-1, dims=2), P.roll(-2, dims=2)
    G = torch.stack([Pj[..., 1] - Pk[..., 1], Pk[..., 0] - Pj[..., 0]], dim=-1)  # [B,T,3,2]
    e1, e2 = P[:, :, 1] - P[:, :, 0], P[:, :, 2] - P[:, :, 0]
    area = 0.5 * (e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]).abs()      # [B, T]
    g = torch.einsum('btim,btid->btmd', Psi, G)       # [B, T, M, 2]
    inv4a = torch.where(area > 0, 0.25 / area.clamp(min=1e-30), torch.zeros_like(area))
    L_mat = torch.einsum('btmd,btld,bt->bml', g, g, inv4a)
    S = Psi.sum(dim=2)                                # [B, T, M]
    M_mat = (torch.einsum('btim,btil,bt->bml', Psi, Psi, area)
             + torch.einsum('btm,btl,bt->bml', S, S, area)) / 12.0
    return M_mat, L_mat


def _generalized_eigh(L_mat, M_mat, eig_broadening):
    """Batched  L u = λ M u  via Cholesky whitening (see
    SpectralNO._generalized_eigh_safe).  float64 in; ascending λ, M-orthonormal u."""
    eye = torch.eye(M_mat.shape[-1], device=L_mat.device, dtype=L_mat.dtype).unsqueeze(0)
    scale = torch.diagonal(M_mat, dim1=-2, dim2=-1).mean(-1).clamp(min=1e-30)
    scale = scale[:, None, None].detach()
    try:
        C = torch.linalg.cholesky(M_mat + 1e-10 * scale * eye)
    except RuntimeError:
        C = torch.linalg.cholesky(M_mat + 1e-4 * scale * eye)
    A = torch.linalg.solve_triangular(C, L_mat, upper=False)                  # C⁻¹ L
    L_white = torch.linalg.solve_triangular(C, A.transpose(-1, -2), upper=False)
    L_white = 0.5 * (L_white + L_white.transpose(-1, -2))
    vals, vecs_w = _BroadenedEigh.apply(L_white, eig_broadening)
    vecs = torch.linalg.solve_triangular(C.transpose(-1, -2), vecs_w, upper=True)
    return vals, vecs


def _ritz(basis64, M_mat, L_mat, K, mass_ridge=1e-4, stiff_ridge=1e-4, eig_broadening=1e-4):
    """Rayleigh–Ritz on the span of the basis columns: ridges + generalized
    eigh → K lowest (λ [B, K], fields [B, N, K], u [B, M, K], M, L).  float64."""
    # Ridges relative to each matrix's own mean diagonal: invariant to
    # rescaling the basis, and a (near-)null direction of a collinear
    # basis gets λ ≈ (stiff/mass)·mean Rayleigh quotient of the ψ_m
    # (each ≥ λ₁ for ψ ∈ H¹₀) instead of a spurious λ ≈ stiff/mass ≈ 1
    # below the true fundamental — keeps λ₁ a Ritz upper bound.
    eye = torch.eye(M_mat.shape[-1], device=M_mat.device, dtype=M_mat.dtype).unsqueeze(0)
    ridge_M = torch.diagonal(M_mat, dim1=-2, dim2=-1).mean(-1).clamp(min=1e-12)
    ridge_L = torch.diagonal(L_mat, dim1=-2, dim2=-1).mean(-1).clamp(min=1e-12)
    M_mat = M_mat + mass_ridge * ridge_M.detach()[:, None, None] * eye
    L_mat = L_mat + stiff_ridge * ridge_L.detach()[:, None, None] * eye
    # Symmetrize (guards against tiny einsum asymmetry).
    M_mat = 0.5 * (M_mat + M_mat.transpose(-1, -2))
    L_mat = 0.5 * (L_mat + L_mat.transpose(-1, -2))
    vals, vecs = _generalized_eigh(L_mat, M_mat, eig_broadening)
    u_K = vecs[:, :, :K]
    return vals[:, :K], torch.einsum('bnm,bmk->bnk', basis64, u_K), u_K, M_mat, L_mat


class _MonotoneFreqHead(nn.Module):
    """Strictly increasing scalar map  log λ → z-scored frequency.

    Positive (softplus) weights + tanh keep f(λ) monotone, so sorted
    eigenvalues always give sorted frequencies (the targets are sorted too).
    log λ puts the widely-ranged Laplacian eigenvalues on an O(1) scale; the
    hidden units are initialised with centres spread over log λ ∈ [−2, 10].
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.inp = nn.Linear(1, hidden)
        self.out = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.inp.weight.fill_(math.log(math.e - 1.0))        # softplus → 1
            self.inp.bias.copy_(-torch.linspace(-2.0, 10.0, hidden))
            self.out.weight.fill_(-3.0)                          # softplus ≈ 0.05
            self.out.bias.zero_()

    def forward(self, lam: torch.Tensor) -> torch.Tensor:
        s = torch.log(lam.clamp(min=1e-8))
        h = torch.tanh(F.linear(s, F.softplus(self.inp.weight), self.inp.bias))
        return F.linear(h, F.softplus(self.out.weight), self.out.bias)


# ─────────────────────────────────────────────────────────────────────────────
#  SpectralNO
# ─────────────────────────────────────────────────────────────────────────────

class SpectralNO(nn.Module):
    """Spectral Neural Operator: predict RF cavity eigenmodes via a *physical*
    Galerkin projection of the Dirichlet Laplacian onto a learned basis.

    Args:
        val_dim: Number of input node features (e.g. 8 or 12).
        grid_dim: Spatial coordinate dimension (2 for 2-D cavities).
        embed_dim: Internal embedding width.
        n_basis: Reduced-basis size M.
        num_field_modes: K — number of eigenpairs to predict.
        rff_dim: Output dimension of RandomFourierFeatures encoder.
        rff_length_scale: Length scale for the Gaussian RFF kernel.
        n_heads: Unused (kept for config/back-compat; the global pooler that
            fed the removed matrix heads is gone).
        dropout: Dropout rate (applied to basis_net and local_encoder).
        use_checkpoint: Placeholder (not wired).
        bc_scale: Smoothness of the soft Dirichlet boundary gate
            (gate = 2·sigmoid(dist_bnd / bc_scale) − 1; → 0 at ∂Ω, → 1 interior).
        area_feature_idx: Input-feature column holding the per-node mesh area
            used as the quadrature weight (default 5 = node_area).
        mass_ridge / stiff_ridge: diagonal ridges added to M / L for SPD
            conditioning of the generalized eigenproblem, each relative to
            the mean diagonal of its own matrix (basis-scale invariant, and
            no spurious λ below the fundamental from a collinear basis).
        coord_feature_idx: Input-feature columns that duplicate X (x_norm,
            y_norm); replaced by the differentiable coordinates so ∇ψ sees
            them.  ``None`` disables (e.g. ablations without these columns).
        dist_feature_idx / dir_feature_idx: columns of dist_bnd and the unit
            vector to the nearest boundary node; dist_bnd is linearised with
            ∇d = −dir so the Dirichlet gate's gradient enters L.
        eig_broadening: ε of the Lorentzian-broadened eigh backward (units of
            λ²; gaps |Δλ| ≲ √ε are treated as degenerate).
        physics_freq: frequency straight from the eigenvalue,
            f = c·√λ / (2π·scale) [GHz], z-scored with ``freq_stats`` — no
            learned head (λ(sΩ) = λ(Ω)/s² is exact; a head that only sees the
            scale-free λ cannot know the cavity size).  Needs batch['Scale']
            (newer converter output) and ``freq_stats`` (set by GNOTLightning).
        assembly: 'nodal' — M/L from nodal quadrature of autograd ∇ψ;
            'p1' — ψ is read as the P1 interpolant of its nodal values and
            M (consistent mass) / L are assembled exactly per mesh triangle
            (needs batch['Elements']).  The gate makes ψ = 0 on boundary
            nodes ⇒ ψ ∈ H¹₀ ⇒ true Rayleigh–Ritz upper bounds; no autograd,
            so it is also much faster.  Incompatible with node sub-sampling.
        torsion_feature_idx: (assembly='p1') column of the torsion feature
            w/max w; used as the Dirichlet factor ψ = w·N(x) instead of the
            sigmoid gate — smooth and exactly 0 on ∂Ω, not a thin
            bc_scale-wide boundary layer.  None → sigmoid gate.
    """

    def __init__(
        self,
        val_dim: int = 12,
        grid_dim: int = 2,
        embed_dim: int = 128,
        n_basis: int = 16,
        num_field_modes: int = 3,
        rff_dim: int = 64,
        rff_length_scale: float = 0.1,
        n_heads: int = 4,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
        bc_scale: float = 0.02,
        area_feature_idx: int = 5,
        mass_ridge: float = 1e-4,
        stiff_ridge: float = 1e-4,
        coord_feature_idx=(0, 1),
        dist_feature_idx: int = 2,
        dir_feature_idx=(3, 4),
        eig_broadening: float = 1e-4,
        physics_freq: bool = False,
        assembly: str = 'nodal',
        torsion_feature_idx=None,
    ):
        super().__init__()
        self.val_dim = val_dim
        self.grid_dim = grid_dim
        self.K = num_field_modes
        self.M = n_basis
        if self.K > self.M:
            raise ValueError(f"num_field_modes ({self.K}) must be <= n_basis ({self.M}).")
        self.bc_scale = bc_scale
        self.use_checkpoint = use_checkpoint
        self.area_feature_idx = area_feature_idx
        self.mass_ridge = mass_ridge
        self.stiff_ridge = stiff_ridge
        self.coord_feature_idx = tuple(coord_feature_idx) if coord_feature_idx is not None else None
        self.dist_feature_idx = dist_feature_idx
        self.dir_feature_idx = tuple(dir_feature_idx) if dir_feature_idx is not None else None
        self.eig_broadening = eig_broadening
        self.physics_freq = physics_freq
        if assembly not in ('nodal', 'p1'):
            raise ValueError(f"assembly must be 'nodal' or 'p1', got {assembly!r}")
        self.assembly = assembly
        self.torsion_feature_idx = torsion_feature_idx
        self.freq_stats = None   # {'mean','std'} [GHz]; set by GNOTLightning

        # ── 1. Spatial encoder (fixed random Fourier features) ──────────────
        self.spatial_encoder = RandomFourierFeatures(
            grid_dim, rff_dim, length_scale=rff_length_scale
        )

        # ── 2. Local geometry encoder (pointwise) ────────────────────────────
        input_dim = rff_dim + val_dim
        self.local_encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # ── 3. Basis function network (pointwise) ─────────────────────────────
        self.basis_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, n_basis),
        )

        # ── 4. Frequency transform (eigenvalue → z-scored physical frequency) ─
        # Monotone so the frequency order is the eigenvalue order.
        self.freq_transform = None if physics_freq else _MonotoneFreqHead(hidden=32)

        # NOTE: L_chol_head / M_chol_head / AttentionPool pooler are intentionally
        # removed — the Galerkin matrices are now assembled from the basis, not
        # predicted from a per-geometry global fingerprint.

    # ──────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _basis_gradient(self, basis: torch.Tensor, X: torch.Tensor,
                        create_graph: bool) -> torch.Tensor:
        """Spatial gradient ∇ψ_m(x_i) for every basis function and node.

        The basis network is *pointwise* (per-node MLP, no cross-node mixing),
        so ψ_m(x_i) depends only on x_i.  Therefore ∂(Σ_{b,i} ψ_m(x_{b,i})) /
        ∂x_{b,i} = ∂ψ_m(x_{b,i})/∂x_{b,i}, i.e. one summed backward per basis
        index recovers the exact per-node gradient.

        Args:
            basis: [B, N, M] basis values (must be part of an autograd graph
                rooted at ``X``).
            X: [B, N, grid_dim] coordinates with ``requires_grad=True``.
            create_graph: keep the graph so the stiffness term is itself
                differentiable w.r.t. model parameters (needed during training).
        Returns:
            grad_basis: [B, N, M, grid_dim].
        """
        # All M vector-Jacobian products in one vmapped backward (one-hot
        # grad_outputs) instead of M sequential backward passes: identical
        # result, ~3-4x faster forward+backward at M=16 on CPU.
        B, N, M = basis.shape
        eye = torch.eye(M, device=basis.device, dtype=basis.dtype)
        grad_outputs = eye.view(M, 1, 1, M).expand(M, B, N, M)
        g = torch.autograd.grad(
            basis, X, grad_outputs=grad_outputs, is_grads_batched=True,
            create_graph=create_graph, retain_graph=True,
        )[0]                                       # [M, B, N, grid_dim]
        return g.permute(1, 2, 0, 3)               # [B, N, M, grid_dim]

    def _generalized_eigh_safe(self, L_mat: torch.Tensor, M_mat: torch.Tensor):
        """Stable batched generalized symmetric eigendecomposition.

        Solves  L u_k = λ_k M u_k  via Cholesky whitening:
            M = C C^T  (Cholesky)
            L_white = C^{-1} L C^{-T}      (symmetric, triangular solves)
            eigh(L_white) → (λ_k, v_k)     — sorted ascending
            u_k = C^{-T} v_k               (back-transform, M-orthonormal)

        Call with float64 inputs (see ``forward``).  The jitter is relative to
        the mean diagonal of M.  If Cholesky still fails, it is retried with a
        much larger (still relative) jitter — the problem stays a generalized
        one, unlike the former silent fallback to ``eigh(L)`` (M = I), which
        returned eigenvectors that are not M-orthonormal.

        Args:
            L_mat: [B, M, M] SPD stiffness.
            M_mat: [B, M, M] SPD mass.
        Returns:
            vals: [B, M] ascending; vecs: [B, M, M] M-orthonormal columns.
        """
        return _generalized_eigh(L_mat, M_mat, self.eig_broadening)

    def _differentiable_inputs(self, X_req: torch.Tensor, Y: torch.Tensor):
        """Make the position-derived feature columns depend on ``X_req``.

        Returns (Y_d, dist_bnd [B, N, 1] or None): Y with the coordinate
        columns replaced by ``X_req`` and the dist_bnd column linearised
        around each node,
            d(x) ≈ d_i − dir_i · (x − x_i),
        which equals d_i in value and has the exact gradient ∇d = −dir of the
        distance function (dir points from the node to its nearest boundary
        node).  Without this, ∂ψ/∂x only saw the RFF path and ignored the
        Dirichlet gate, whose gradient 1/(2·bc_scale) at ∂Ω dominates L.
        Values are unchanged, so the forward output is identical.
        """
        V = Y.shape[-1]
        cols = list(Y.unbind(-1))
        if self.coord_feature_idx is not None and max(self.coord_feature_idx) < V:
            for d, c in enumerate(self.coord_feature_idx):
                cols[c] = X_req[..., d]

        dist = None
        if self.dist_feature_idx is not None and self.dist_feature_idx < V:
            dist = Y[..., self.dist_feature_idx]                                   # [B, N]
            if self.dir_feature_idx is not None and max(self.dir_feature_idx) < V:
                direction = Y[..., list(self.dir_feature_idx)]                     # [B, N, 2]
                # On boundary nodes the converter's dir is the zero vector (the
                # nearest boundary node is the node itself).  There d = 0 ⇒
                # gate = 0 ⇒ ∇ψ = net·gate'(0)·n, and ∇ψ_m·∇ψ_n does not depend
                # on the direction of the unit normal n: any unit vector is exact.
                unit = torch.zeros_like(direction)
                unit[..., 0] = 1.0
                degenerate = (direction * direction).sum(dim=-1, keepdim=True) < 0.25
                direction = torch.where(degenerate, unit, direction)
                dx = X_req - X_req.detach()                                        # value 0, grad I
                dist = dist - (direction * dx).sum(dim=-1)
            cols[self.dist_feature_idx] = dist
            # Value-only clamp: clamp()'s gradient is 0 at exactly d = 0, which
            # would drop the gate slope on every boundary node.
            dist = dist.unsqueeze(-1)
            dist = dist + (dist.clamp(min=0.0) - dist).detach()
        return torch.stack(cols, dim=-1), dist

    def _nodal_galerkin(self, X, Y, mask, solve_dtype):
        """M/L by nodal quadrature of the autograd basis gradient (assembly='nodal')."""
        B, N, _ = X.shape
        device, dtype = X.device, X.dtype
        # ── Quadrature weights (per-node mesh area), masked + normalized ──────
        if self.area_feature_idx is not None and Y.shape[-1] > self.area_feature_idx:
            w = Y[:, :, self.area_feature_idx].clamp(min=0.0)   # [B, N]
        else:
            w = torch.ones(B, N, device=device, dtype=dtype)
        if mask is not None:
            w = w * mask.to(dtype)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)             # [B, N], Σ_i w_i = 1

        # ── Basis + spatial gradients (autograd; needs grad even at eval) ─────
        # The stiffness only needs a differentiable graph when the caller
        # wants parameter gradients (not tied to self.training: an eval-mode
        # backward must still see the L-path).  inference_mode(False) + clone
        # lets this run under Lightning's validation/test inference_mode,
        # where enable_grad() alone does not record a graph.
        create_graph = torch.is_grad_enabled()
        with torch.inference_mode(False), torch.enable_grad():
            X_req = X.detach().clone().requires_grad_(True)
            Y_in = Y.clone() if Y.is_inference() else Y
            Y_d, dist_bnd = self._differentiable_inputs(X_req, Y_in)
            x_rff = self.spatial_encoder(X_req)                 # [B, N, rff_dim]
            h = self.local_encoder(torch.cat([x_rff, Y_d], dim=-1))
            basis = self.basis_net(h)                           # [B, N, M]

            # Soft Dirichlet gate: genuinely 0 at the boundary, → 1 in interior.
            if dist_bnd is not None:
                bc_gate = 2.0 * torch.sigmoid(dist_bnd / self.bc_scale) - 1.0
                basis = basis * bc_gate                         # [B, N, M]

            grad_basis = self._basis_gradient(basis, X_req, create_graph)  # [B,N,M,2]
            if not create_graph:
                basis, grad_basis = basis.detach(), grad_basis.detach()

        # ── Physical Galerkin assembly + generalized eigensolve ──────────────
        #   M_mn = Σ_i w_i ψ_m ψ_n ;  L_mn = Σ_i w_i ∇ψ_m · ∇ψ_n
        # float64 with autocast off: fp16/bf16 (AMP) or TF32 matmuls
        # (train.py sets matmul precision 'medium') would corrupt the small
        # Gram matrices, and linalg kernels do not support half precision.
        with torch.autocast(device_type=device.type, enabled=False):
            basis64 = basis.to(solve_dtype)
            w64 = w.to(solve_dtype)
            M_mat = torch.einsum('bnm,bnl,bn->bml', basis64, basis64, w64)
            L_mat = torch.einsum('bnmd,bnld,bn->bml', grad_basis.to(solve_dtype),
                                 grad_basis.to(solve_dtype), w64)
        return M_mat, L_mat, basis64

    # ──────────────────────────────────────────────────────────────────────────
    #  Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> dict:
        """Predict K eigenpairs from a geometry batch.

        Args:
            batch: dict with keys
                'X'           [B, N, 2]         grid coordinates
                'Input_funcs' [B, N, val_dim]    node features
                'Mask'        [B, N]  bool       valid-node mask (optional)

        Returns:
            dict with 'field' [B, N, K], 'freq' [B, K], 'eigenvalues' [B, K],
            'L_mat' [B, M, M], 'M_mat' [B, M, M], 'u_K' [B, M, K].
        """
        X = batch['X']               # [B, N, 2]
        Y = batch['Input_funcs']     # [B, N, val_dim]
        mask = batch.get('Mask', None)
        _check_val_dim(Y, self.val_dim, 'SpectralNO')
        B, N, _ = X.shape
        device, dtype = X.device, X.dtype
        solve_dtype = torch.float64

        if self.assembly == 'p1':
            elems = batch.get('Elements', None)
            if elems is None:
                raise ValueError(
                    "SpectralNO(assembly='p1') needs batch['Elements'] (mesh triangles): "
                    "use a PKL/H5 from the current converter and dataset.max_nodes: null.")
            dist_bnd = None
            if self.dist_feature_idx is not None and self.dist_feature_idx < Y.shape[-1]:
                dist_bnd = Y[..., self.dist_feature_idx].clamp(min=0.0).unsqueeze(-1)
            h = self.local_encoder(torch.cat([self.spatial_encoder(X), Y], dim=-1))
            basis = self.basis_net(h)                               # [B, N, M]
            if self.torsion_feature_idx is not None and self.torsion_feature_idx < Y.shape[-1]:
                basis = basis * Y[..., self.torsion_feature_idx:self.torsion_feature_idx + 1].clamp(min=0.0)
            elif dist_bnd is not None:
                basis = basis * (2.0 * torch.sigmoid(dist_bnd / self.bc_scale) - 1.0)
            if mask is not None:
                basis = basis * mask.unsqueeze(-1).to(dtype)
            with torch.autocast(device_type=device.type, enabled=False):
                basis64 = basis.to(solve_dtype)
                M_mat, L_mat = _p1_galerkin(basis64, X.to(solve_dtype), elems)
        else:
            M_mat, L_mat, basis64 = self._nodal_galerkin(X, Y, mask, solve_dtype)

        with torch.autocast(device_type=device.type, enabled=False):
            lambda_K, fields, u_K, M_mat, L_mat = _ritz(
                basis64, M_mat, L_mat, self.K, self.mass_ridge, self.stiff_ridge,
                self.eig_broadening)                                    # [B,K], [B,N,K]
        fields = fields.to(dtype)
        if mask is not None:
            fields = fields * mask.unsqueeze(-1).to(dtype)
        # Same amplitude convention as the targets (max|Y| = 1), sign fixed
        # so the peak is positive.  Per-column → M-orthogonality preserved.
        fields = _normalize_peak(fields, mask)

        # ── Frequency transform ───────────────────────────────────────────────
        lambda_K = lambda_K.to(dtype)
        if self.physics_freq:
            log_k = 0.5 * torch.log(lambda_K.clamp(min=1e-12))
            freq = _physics_freq_z(log_k, batch, self.freq_stats, 'SpectralNO')
        else:
            freq = self.freq_transform(lambda_K.unsqueeze(-1)).squeeze(-1)  # [B, K]

        L_mat, M_mat, u_K = L_mat.to(dtype), M_mat.to(dtype), u_K.to(dtype)
        return {
            'field':       fields,
            'freq':        freq,
            'eigenvalues': lambda_K,
            'L_mat':       L_mat,
            'M_mat':       M_mat,
            'u_K':         u_K,
        }
