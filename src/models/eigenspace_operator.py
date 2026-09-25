"""EigenspaceOperator: a NEO-style learned eigenspace with exact Rayleigh–Ritz.

After NEO (Yang, Du & Liu, SIGGRAPH 2026, "Learning Laplacian Eigenspace with
Mass-Aware Neural Operators on Point Clouds"), adapted to the Dirichlet
Laplacian  −Δu = λu in Ω, u = 0 on ∂Ω  on a P1 mesh:

1. Encoder: RFF(X) ‖ Input_funcs → MLP → D.
2. Trunk: pre-LN residual blocks of mass-aware linear attention + FFN.  The
   attention's key/value sums carry the node areas w (Σ w = 1 over valid
   nodes), so it is a quadrature of a global integral operator. That makes
   it resolution-invariant, and every basis function sees the whole cavity
   (SpectralNO's basis is pointwise).
3. Head: one shared linear map D → m ≥ K basis functions. It has no per-mode
   branches, so it imposes no mode identity (GNOT has per-mode branches).
   ψ is multiplied by a Dirichlet factor, so ψ = 0 on ∂Ω, ψ ∈ H¹₀, and the
   Ritz values are true upper bounds.
4. Rayleigh–Ritz with the mesh's exact P1 Grams:
   ψᵀLψ c = λ ψᵀMψ c  →  K lowest pairs,  f = c·√λ / (2π·scale).
   Only the span of ψ matters. The basis ordering, sign and rotation within
   degenerate clusters are all gauge.
"""
import torch
import torch.nn as nn

from src.models.gnot import (LinearAttention, MLPEncoder, RandomFourierFeatures,
                             _check_val_dim, _normalize_peak, _physics_freq_z)
from src.models.spectral_no import _p1_galerkin, _ritz


def mass_weights(batch: dict, mask: torch.Tensor) -> torch.Tensor:
    """Quadrature weights [B, N]: node area normalised to Σ w = 1 over the
    valid nodes (uniform over them without batch['Area']); padding gets 0."""
    m = mask.to(batch['X'].dtype)
    area = batch.get('Area', None)
    w = m if area is None else area.to(m.dtype).clamp(min=0.0) * m
    return w / w.sum(dim=1, keepdim=True).clamp(min=1e-30)


class MassAwareBlock(nn.Module):
    """Pre-LN residual block: x += A_w(LN x);  x += FFN(LN x)."""

    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, n_heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask, weights):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, mask=mask, weights=weights)
        return (x + self.ffn(self.norm2(x))) * mask.unsqueeze(-1).to(x.dtype)


class EigenspaceOperator(nn.Module):
    """Geometry → m-dimensional trial space → K Ritz eigenpairs.

    forward(batch) returns
        'basis'       [B, N, m]  Dirichlet-gated, masked raw basis (pre-Ritz)
        'field'       [B, N, K]  Ritz modes, peak-normalised, masked
        'eigenvalues' [B, K]     Ritz values λ (normalised coordinates)
        'freq'        [B, K]     z-scored GHz of f = c·√λ / (2π·scale)
        'M_mat', 'L_mat' [B, m, m]  ridged P1 mass / stiffness Grams of ψ
    Needs batch['Elements'], batch['Dist_bnd'] and batch['Scale']; self.freq_stats
    is set by GNOTLightning.
    """

    def __init__(self, val_dim, grid_dim=2, embed_dim=128, n_layers=4, n_heads=4,
                 n_basis=16, num_field_modes=3, rff_dim=64, rff_length_scale=0.1,
                 dropout=0.0, bc_scale=0.02, torsion_feature_idx=None,
                 physics_freq=True, mass_ridge=1e-4, stiff_ridge=1e-4,
                 eig_broadening=1e-4):
        super().__init__()
        if not physics_freq:
            raise ValueError("EigenspaceOperator needs physics_freq=True: f comes from "
                             "the Ritz λ, f = c·√λ / (2π·scale).")
        if n_basis < num_field_modes:
            raise ValueError(f"EigenspaceOperator: n_basis={n_basis} < num_field_modes="
                             f"{num_field_modes}; Rayleigh–Ritz needs m ≥ K.")
        self.val_dim = val_dim
        self.num_field_modes = num_field_modes
        self.n_basis = n_basis
        self.bc_scale = bc_scale
        self.torsion_feature_idx = torsion_feature_idx
        self.physics_freq = physics_freq
        self.mass_ridge, self.stiff_ridge = mass_ridge, stiff_ridge
        self.eig_broadening = eig_broadening
        self.freq_stats = None   # {'mean', 'std'} [GHz]; set by GNOTLightning

        self.rff = RandomFourierFeatures(grid_dim, rff_dim, length_scale=rff_length_scale)
        self.encoder = MLPEncoder(rff_dim + val_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [MassAwareBlock(embed_dim, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_basis)

    def embed(self, batch: dict) -> torch.Tensor:
        """Trunk output [B, N, D] (padded nodes = 0)."""
        X, Y = batch['X'], batch['Input_funcs']
        _check_val_dim(Y, self.val_dim, 'EigenspaceOperator')
        mask = batch.get('Mask', None)
        if mask is None:
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=X.device)
        w = mass_weights(batch, mask)
        h = self.encoder(torch.cat([self.rff(X), Y], dim=-1)) * mask.unsqueeze(-1).to(X.dtype)
        for blk in self.blocks:
            h = blk(h, mask, w)
        return self.norm(h)

    def dirichlet_factor(self, batch: dict, mask: torch.Tensor) -> torch.Tensor:
        """[B, N, 1] factor that is exactly 0 on ∂Ω (Dist_bnd ≤ 1e-9) and on padding."""
        dist = batch['Dist_bnd'].to(batch['X'].dtype)
        Y = batch['Input_funcs']
        if self.torsion_feature_idx is not None:
            g = Y[..., self.torsion_feature_idx].clamp(min=0.0)
        else:
            g = 2.0 * torch.sigmoid(dist.clamp(min=0.0) / self.bc_scale) - 1.0
        return (g * ((dist > 1e-9) & mask)).unsqueeze(-1)

    def forward(self, batch: dict) -> dict:
        for key in ('Elements', 'Dist_bnd', 'Scale'):
            if batch.get(key, None) is None:
                raise ValueError(
                    f"EigenspaceOperator needs batch['{key}']: use current converter "
                    "output (elements, scale) and dataset.max_nodes: null.")
        X = batch['X']
        mask = batch.get('Mask', None)
        if mask is None:
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=X.device)
        mask = mask.bool()

        basis = self.head(self.embed(batch)).float() * self.dirichlet_factor(batch, mask).float()

        with torch.autocast(device_type=X.device.type, enabled=False):
            basis64 = basis.double()
            M_mat, L_mat = _p1_galerkin(basis64, X.double(), batch['Elements'])
            lam, modes, _, M_mat, L_mat = _ritz(
                basis64, M_mat, L_mat, self.num_field_modes,
                self.mass_ridge, self.stiff_ridge, self.eig_broadening)
        lam = lam.float()
        field = _normalize_peak(modes.float() * mask.unsqueeze(-1), mask)
        freq = _physics_freq_z(0.5 * torch.log(lam.clamp(min=1e-12)), batch,
                               self.freq_stats, 'EigenspaceOperator')
        return {'basis': basis, 'field': field, 'eigenvalues': lam, 'freq': freq,
                'M_mat': M_mat.float(), 'L_mat': L_mat.float()}
