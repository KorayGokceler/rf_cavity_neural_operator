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

   This is the key change vs. the previous design: L and M are **no longer
   predicted by free per-geometry MLP heads** (which mapped a 128-d global
   "fingerprint" → an arbitrary 32×32 SPD matrix and thereby memorised the
   training set).  The only learned object is now the *function space* ψ(x);
   the operator is the true Laplacian projected onto it, so it generalises.
4. **Generalized eigendecomposition** – solve L u_k = λ_k M u_k via Cholesky
   whitening + `torch.linalg.eigh`; eigenvalues are returned sorted ascending,
   so no OT matching is needed at training or inference time.
5. **Eigenfunction reconstruction** – φ_k(x) = Σ_m u_{k,m} · ψ_m(x).
6. **Frequency transform** – lightweight MLP maps Galerkin (Laplacian)
   eigenvalues → z-scored physical frequencies.

Key properties
--------------
* Ordering is **structural** (eigh returns sorted eigenvalues).
* M-orthogonality of eigenfunctions is **structural**.
* Dirichlet BC (φ = 0 on ∂Ω) is built into the basis via the boundary gate.
* Sign / degenerate-subspace ambiguity is handled by the sign-agnostic /
  Grassmannian field loss (see lightning_module).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.gnot import RandomFourierFeatures


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
            conditioning of the generalized eigenproblem.
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
    ):
        super().__init__()
        self.K = num_field_modes
        self.M = n_basis
        self.bc_scale = bc_scale
        self.use_checkpoint = use_checkpoint
        self.area_feature_idx = area_feature_idx
        self.mass_ridge = mass_ridge
        self.stiff_ridge = stiff_ridge

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
        self.freq_transform = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

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
        grads = []
        for m in range(self.M):
            gm = torch.autograd.grad(
                basis[:, :, m].sum(), X,
                create_graph=create_graph, retain_graph=True,
            )[0]                                   # [B, N, grid_dim]
            grads.append(gm)
        return torch.stack(grads, dim=2)           # [B, N, M, grid_dim]

    def _generalized_eigh_safe(self, L_mat: torch.Tensor, M_mat: torch.Tensor):
        """Stable batched generalized symmetric eigendecomposition.

        Solves  L u_k = λ_k M u_k  via Cholesky whitening:
            M = C C^T  (Cholesky)
            L_white = C^{-1} L C^{-T}      (symmetric)
            eigh(L_white) → (λ_k, v_k)     — sorted ascending
            u_k = C^{-T} v_k               (back-transform, M-orthonormal)

        Falls back to standard ``eigh(L_mat)`` (M = I) if Cholesky fails.

        Args:
            L_mat: [B, M, M] SPD stiffness.
            M_mat: [B, M, M] SPD mass.
        Returns:
            vals: [B, M] ascending; vecs: [B, M, M] M-orthonormal columns.
        """
        eps = 1e-5
        eye = torch.eye(self.M, device=L_mat.device, dtype=L_mat.dtype)
        M_reg = M_mat + eps * eye.unsqueeze(0)
        try:
            C = torch.linalg.cholesky(M_reg)
            C_inv = torch.linalg.inv(C)
            L_white = C_inv @ L_mat @ C_inv.transpose(-1, -2)
            L_white = 0.5 * (L_white + L_white.transpose(-1, -2))
            vals, vecs_w = torch.linalg.eigh(L_white)
            vecs = C_inv.transpose(-1, -2) @ vecs_w
        except RuntimeError:
            vals, vecs = torch.linalg.eigh(L_mat)
        return vals, vecs

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
        B, N, _ = X.shape
        device, dtype = X.device, X.dtype

        # ── Quadrature weights (per-node mesh area), masked + normalized ──────
        if Y.shape[-1] > self.area_feature_idx:
            w = Y[:, :, self.area_feature_idx].clamp(min=0.0)   # [B, N]
        else:
            w = torch.ones(B, N, device=device, dtype=dtype)
        if mask is not None:
            w = w * mask.to(dtype)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)             # [B, N], Σ_i w_i = 1

        # ── Basis + spatial gradients (autograd; needs grad even at eval) ─────
        create_graph = self.training
        with torch.enable_grad():
            X_req = X.detach().requires_grad_(True)
            x_rff = self.spatial_encoder(X_req)                 # [B, N, rff_dim]
            h = self.local_encoder(torch.cat([x_rff, Y], dim=-1))
            basis = self.basis_net(h)                           # [B, N, M]

            # Soft Dirichlet gate: genuinely 0 at the boundary, → 1 in interior.
            dist_bnd = Y[:, :, 2:3].clamp(min=0.0)              # [B, N, 1]
            bc_gate = 2.0 * torch.sigmoid(dist_bnd / self.bc_scale) - 1.0
            basis = basis * bc_gate                             # [B, N, M]

            grad_basis = self._basis_gradient(basis, X_req, create_graph)  # [B,N,M,2]

        # ── Physical Galerkin assembly (geometry-derived, no free params) ─────
        #   M_mn = Σ_i w_i ψ_m ψ_n ;  L_mn = Σ_i w_i ∇ψ_m · ∇ψ_n
        M_mat = torch.einsum('bnm,bnl,bn->bml', basis, basis, w)
        L_mat = torch.einsum('bnmd,bnld,bn->bml', grad_basis, grad_basis, w)
        eye = torch.eye(self.M, device=device, dtype=dtype).unsqueeze(0)
        M_mat = M_mat + self.mass_ridge * eye
        L_mat = L_mat + self.stiff_ridge * eye
        # Symmetrize (guards against tiny einsum asymmetry).
        M_mat = 0.5 * (M_mat + M_mat.transpose(-1, -2))
        L_mat = 0.5 * (L_mat + L_mat.transpose(-1, -2))

        # ── Generalized eigendecomposition ───────────────────────────────────
        vals, vecs = self._generalized_eigh_safe(L_mat, M_mat)
        u_K      = vecs[:, :, :self.K]              # [B, M, K]
        lambda_K = vals[:, :self.K]                 # [B, K]

        # ── Eigenfunction reconstruction ──────────────────────────────────────
        fields = torch.einsum('bnm,bmk->bnk', basis, u_K)       # [B, N, K]
        if mask is not None:
            fields = fields * mask.unsqueeze(-1).to(dtype)

        # ── Frequency transform ───────────────────────────────────────────────
        freq = self.freq_transform(lambda_K.unsqueeze(-1)).squeeze(-1)  # [B, K]

        return {
            'field':       fields,
            'freq':        freq,
            'eigenvalues': lambda_K,
            'L_mat':       L_mat,
            'M_mat':       M_mat,
            'u_K':         u_K,
        }
