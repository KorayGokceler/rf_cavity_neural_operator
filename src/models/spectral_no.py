"""Spectral Neural Operator (SpectralNO) via Galerkin reduction.

Architecture overview
---------------------
1. **Local geometry encoder** – RFF coordinates + node features → per-node
   embeddings [B, N, embed_dim].
2. **Global context** – AttentionPool → [B, embed_dim] geometry summary.
3. **Basis network** – pointwise MLP → M basis function values ψ_m(x) per
   node, multiplied by a soft Dirichlet gate so ψ_m ≈ 0 on ∂Ω.
4. **Galerkin matrix heads** – two heads predict the lower-triangular Cholesky
   factors of a stiffness-like matrix L and a mass-like matrix M (both SPD by
   construction: A = L_chol @ L_chol^T + ε I).
5. **Generalized eigendecomposition** – solve L u_k = λ_k M u_k via Cholesky
   whitening + `torch.linalg.eigh`; eigenvalues are returned sorted ascending
   by PyTorch, so no OT matching is needed at training or inference time.
6. **Eigenfunction reconstruction** – φ_k(x) = Σ_m u_{k,m} · ψ_m(x).
7. **Frequency transform** – lightweight MLP maps normalized Galerkin
   eigenvalues → z-scored physical frequencies for regression against
   dataset targets.

Key properties vs. GNOT
------------------------
* Ordering is **structural** (eigh returns sorted eigenvalues).
* M-orthogonality of eigenvectors is **structural** (eigenvectors of a
  symmetric eigenvalue problem are orthogonal with respect to M).
* Sign ambiguity is intrinsic to the eigenproblem — handled in the loss by
  the same sign-agnostic Grassmannian/rel-L2 used for GNOT.
* Near-degenerate subspace ambiguity: eigenvectors inside a degenerate block
  can rotate freely — the Grassmannian loss handles this for free (no OT
  needed).
* No slot-collapse penalty needed (M-orthogonality prevents collapse).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.gnot import RandomFourierFeatures, AttentionPool


# ─────────────────────────────────────────────────────────────────────────────
#  SpectralNO
# ─────────────────────────────────────────────────────────────────────────────

class SpectralNO(nn.Module):
    """Spectral Neural Operator: predict RF cavity eigenmodes via Galerkin
    reduction over a learned M-dimensional basis.

    Args:
        val_dim: Number of input node features (e.g. 8 or 12).
        grid_dim: Spatial coordinate dimension (2 for 2-D cavities).
        embed_dim: Internal embedding width.
        n_basis: Reduced-basis size M (Risk 2 from design doc: default 32).
        num_field_modes: K — number of eigenpairs to predict.
        rff_dim: Output dimension of RandomFourierFeatures encoder.
        rff_length_scale: Length scale for the Gaussian RFF kernel.
        n_heads: Number of attention heads for AttentionPool.
        dropout: Dropout rate (applied to basis_net and local_encoder).
        use_checkpoint: Gradient checkpointing (not yet wired, placeholder).
        bc_scale: Smoothness of the soft Dirichlet boundary gate
            (gate = sigmoid(dist_bnd / bc_scale); smaller → sharper).
    """

    def __init__(
        self,
        val_dim: int = 12,
        grid_dim: int = 2,
        embed_dim: int = 128,
        n_basis: int = 32,
        num_field_modes: int = 3,
        rff_dim: int = 64,
        rff_length_scale: float = 0.1,
        n_heads: int = 4,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
        bc_scale: float = 0.02,
    ):
        super().__init__()
        self.K = num_field_modes
        self.M = n_basis
        self.bc_scale = bc_scale
        self.use_checkpoint = use_checkpoint

        # ── 1. Spatial encoder (fixed random Fourier features) ──────────────
        self.spatial_encoder = RandomFourierFeatures(
            grid_dim, rff_dim, length_scale=rff_length_scale
        )

        # ── 2. Local geometry encoder ────────────────────────────────────────
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

        # ── 3. Global geometry pooler ─────────────────────────────────────────
        self.pooler = AttentionPool(embed_dim, n_heads)

        # ── 4. Basis function network (pointwise) ─────────────────────────────
        self.basis_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, n_basis),
        )

        # ── 5. Galerkin matrix heads (lower-triangular Cholesky factor) ───────
        tri_size = n_basis * (n_basis + 1) // 2
        self.L_chol_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, tri_size),
        )
        self.M_chol_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, tri_size),
        )

        # ── 6. Frequency transform (eigenvalue → z-scored physical frequency) ─
        # Simple 2-layer MLP per-mode; shared across K modes.
        self.freq_transform = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Pre-register triangular indices and diagonal indices.
        # tril_indices returns (row_indices, col_indices) for lower triangle.
        tril = torch.tril_indices(n_basis, n_basis, offset=0)
        self.register_buffer('_tril_r', tril[0])   # row indices
        self.register_buffer('_tril_c', tril[1])   # col indices
        self.register_buffer('_diag_idx', torch.arange(n_basis))

    # ──────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _vec_to_spd(self, v: torch.Tensor) -> torch.Tensor:
        """Construct a batch of SPD matrices from flattened Cholesky vectors.

        Converts [B, tri_size] → [B, M, M] SPD via:
            A_chol[lower-tri] = v
            A_chol[diag] = softplus(v[diag]) + 1e-3  (strict positivity)
            SPD = A_chol @ A_chol^T + 1e-4 * I

        The 1e-4 ridge ensures eigenvalue conditioning.

        Args:
            v: [B, tri_size] flattened lower-triangular values.
        Returns:
            spd: [B, M, M] symmetric positive-definite matrix.
        """
        B = v.shape[0]
        device, dtype = v.device, v.dtype
        A = torch.zeros(B, self.M, self.M, device=device, dtype=dtype)
        A[:, self._tril_r, self._tril_c] = v
        # Enforce strict positive diagonal
        A[:, self._diag_idx, self._diag_idx] = (
            F.softplus(A[:, self._diag_idx, self._diag_idx]) + 1e-3
        )
        # SPD: A A^T + ε I
        eye = torch.eye(self.M, device=device, dtype=dtype).unsqueeze(0)
        return A @ A.transpose(-1, -2) + 1e-4 * eye

    def _generalized_eigh_safe(
        self, L_mat: torch.Tensor, M_mat: torch.Tensor
    ):
        """Stable batched generalized symmetric eigendecomposition.

        Solves  L u_k = λ_k M u_k  via Cholesky whitening:
            M = C C^T  (Cholesky)
            L_white = C^{-1} L C^{-T}   (whitened, still symmetric SPD)
            standard eigh(L_white) → (λ_k, v_k)  — sorted ascending
            u_k = C^{-T} v_k            (back-transform)

        Fallback to standard ``eigh(L_mat)`` (assumes M=I) if Cholesky
        decomposition fails (e.g. conditioning loss near-degenerate batch).

        Risk 1 mitigation: try/except around Cholesky, symmetrize L_white
        before eigh, ridge on M_mat.

        Args:
            L_mat: [B, M, M] SPD stiffness approximation.
            M_mat: [B, M, M] SPD mass approximation.
        Returns:
            vals: [B, M] eigenvalues sorted ascending.
            vecs: [B, M, M] eigenvectors as columns (M-orthonormal).
        """
        eps = 1e-5
        eye = torch.eye(self.M, device=L_mat.device, dtype=L_mat.dtype)
        M_reg = M_mat + eps * eye.unsqueeze(0)

        try:
            C = torch.linalg.cholesky(M_reg)           # [B, M, M] lower-tri
            C_inv = torch.linalg.inv(C)                # [B, M, M]
            L_white = C_inv @ L_mat @ C_inv.transpose(-1, -2)  # [B, M, M]
            # Symmetrize for numerical stability
            L_white = 0.5 * (L_white + L_white.transpose(-1, -2))
            vals, vecs_w = torch.linalg.eigh(L_white)  # sorted ascending
            vecs = C_inv.transpose(-1, -2) @ vecs_w    # back-transform
        except RuntimeError:
            # Fallback: standard eigh (treats M = I)
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
            dict with:
                'field'       [B, N, K]  eigenfunction values at nodes
                'freq'        [B, K]     z-scored predicted frequencies
                'eigenvalues' [B, K]     raw Galerkin eigenvalues (for loss)
                'L_mat'       [B, M, M]  predicted stiffness matrix
                'M_mat'       [B, M, M]  predicted mass matrix
                'u_K'         [B, M, K]  Galerkin coefficients
        """
        X = batch['X']               # [B, N, 2]
        Y = batch['Input_funcs']     # [B, N, val_dim]
        mask = batch.get('Mask', None)   # [B, N] bool or None
        B, N, _ = X.shape

        # ── 1. RFF spatial encoding ──────────────────────────────────────────
        x_rff = self.spatial_encoder(X)             # [B, N, rff_dim]

        # ── 2. Local geometry encoding ────────────────────────────────────────
        h = torch.cat([x_rff, Y], dim=-1)           # [B, N, rff_dim+val_dim]
        h = self.local_encoder(h)                   # [B, N, embed_dim]

        # ── 3. Global context via attention pool ──────────────────────────────
        # AttentionPool expects a boolean [B, N] mask (True = valid node).
        g = self.pooler(h, mask)               # [B, embed_dim]

        # ── 4. Basis function values ──────────────────────────────────────────
        basis = self.basis_net(h)                   # [B, N, M]

        # Soft Dirichlet boundary condition gate:
        # dist_to_boundary is feature column index 2
        dist_bnd = Y[:, :, 2:3].clamp(min=0.0)     # [B, N, 1]  non-negative
        bc_gate = torch.sigmoid(dist_bnd / self.bc_scale)   # → 0 at boundary
        basis = basis * bc_gate                     # [B, N, M]

        # ── 5. Galerkin matrix prediction ─────────────────────────────────────
        L_vec = self.L_chol_head(g)                 # [B, tri_size]
        M_vec = self.M_chol_head(g)                 # [B, tri_size]
        L_mat = self._vec_to_spd(L_vec)             # [B, M, M] SPD
        M_mat = self._vec_to_spd(M_vec)             # [B, M, M] SPD

        # ── 6. Generalized eigendecomposition ─────────────────────────────────
        vals, vecs = self._generalized_eigh_safe(L_mat, M_mat)
        # vals: [B, M] ascending;  vecs: [B, M, M] columns

        # Take the K smallest modes
        u_K      = vecs[:, :, :self.K]              # [B, M, K]
        lambda_K = vals[:, :self.K]                 # [B, K]

        # ── 7. Eigenfunction reconstruction ───────────────────────────────────
        # φ_k(x) = Σ_m u_{k,m} · ψ_m(x)
        fields = torch.einsum('bnm,bmk->bnk', basis, u_K)   # [B, N, K]

        # Apply valid-node mask
        if mask is not None:
            fields = fields * mask.unsqueeze(-1).float()

        # ── 8. Frequency transform ─────────────────────────────────────────────
        # Map raw Galerkin eigenvalues → z-scored physical frequencies.
        # Input [B, K] → unsqueeze last dim → [B, K, 1] → freq_transform →
        # [B, K, 1] → squeeze → [B, K].
        freq = self.freq_transform(
            lambda_K.unsqueeze(-1)
        ).squeeze(-1)                               # [B, K]

        return {
            'field':       fields,      # [B, N, K]
            'freq':        freq,        # [B, K] — sorted ascending (λ sorted)
            'eigenvalues': lambda_K,    # [B, K] — raw Galerkin eigenvalues
            'L_mat':       L_mat,       # [B, M, M]
            'M_mat':       M_mat,       # [B, M, M]
            'u_K':         u_K,         # [B, M, K]
        }
