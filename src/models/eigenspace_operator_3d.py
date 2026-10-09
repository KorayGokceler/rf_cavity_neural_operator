"""EigenspaceOperator3D: learned H(curl) trial space + kernel-projected Rayleigh–Ritz for 3D
Maxwell cavity eigenmodes (E field, Whitney N0 edge DOFs).

    curl curl E = k² E in Ω,  n × E = 0 on the PEC wall (essential)
    discrete:   K y = λ M y on the non-wall edges, wall DOFs 0,
                ker K|H0(curl) = span(G) (interior-vertex + boundary-component potentials;
                handles add no kernel).

1. Trunk: RFF(xyz) ‖ vertex features → MLP → n_layers mass-aware linear-attention blocks (key /
   value sums weighted by the node volumes batch['Area'], NEO).
2. Edge basis: for edge e = (a → b) (low → high vertex index) a vector MLP evaluates m vector
   fields at the midpoint from ½(h_a + h_b) ‖ RFF(x_mid) and dots them with t_e = x_b − x_a:
       V[e, j] = ψ_j(x_mid) · t_e  ≈ ∫_e ψ_j · t   (midpoint rule).
   Orientation is structural: flipping (a, b) flips t_e and nothing else. Wall edges and padding
   are zeroed, so E = 0 on the PEC wall by construction.
3. Rayleigh–Ritz on the divergence-free projection PV = V − G Kp⁻¹GᵀMV (hcurl.hcurl_ritz):
   A_V = VᵀKV, M_div = VᵀMV − BᵀKp⁻¹B, float64 rank-revealing generalized eigh (pure-gradient
   directions are dropped, not ridged), fields PV·c with unit M-norm, f = c·√λ/(2π·scale).
"""
import torch
import torch.nn as nn

from src.models.hcurl import hcurl_ritz
from src.models.layers import MassAwareBlock, MLPEncoder, RandomFourierFeatures, mass_weights, physics_freq_z

REQUIRED_KEYS = ('Edges', 'EdgeMask', 'BndEdge', 'M', 'K', 'G', 'Gt', 'Kp', 'Kp_diag', 'Scale')


class EigenspaceOperator3D(nn.Module):
    """Geometry → m edge basis functions → K projected Ritz eigenpairs.

    forward(batch) returns
        'basis'       [B, Ne, m]  raw edge basis V (pre-projection, masked)
        'field'       [B, Ne, K]  projected Ritz modes, unit M-norm (sign arbitrary)
        'eigenvalues' [B, K]      Ritz values λ (normalised mesh)
        'freq'        [B, K]      z-scored GHz of f = c·√λ / (2π·scale)
        'M_mat', 'L_mat' [B, m, m]  projected mass M_div / curl–curl A_V Grams (float32)
        'M_div', 'A_V', 'M_V' [B, m, m] (float64) and 'Z' [B, Nv, m] = Kp⁻¹GᵀMV, reused by the
            losses (hcurl_grams) without a second Kp solve.
    freq_stats ({'mean', 'std'} GHz) is set by the Lightning module.
    """

    def __init__(self, val_dim, embed_dim=128, n_layers=4, n_heads=4, n_basis=16,
                 num_field_modes=6, rff_dim=64, rff_length_scale=0.1, dropout=0.0,
                 edge_rff=True, mass_ridge=1e-8, drop_tol=1e-6, eig_broadening=1e-4,
                 kp_tol=1e-8, kp_maxiter=2000):
        super().__init__()
        if n_basis < num_field_modes:
            raise ValueError(f"n_basis={n_basis} < num_field_modes={num_field_modes}: "
                             "Rayleigh–Ritz needs m ≥ K.")
        self.val_dim, self.num_field_modes, self.n_basis = val_dim, num_field_modes, n_basis
        self.mass_ridge, self.drop_tol, self.eig_broadening = mass_ridge, drop_tol, eig_broadening
        self.kp_tol, self.kp_maxiter = kp_tol, kp_maxiter
        self.edge_rff = bool(edge_rff)
        self.freq_stats = None

        self.rff = RandomFourierFeatures(3, rff_dim, length_scale=rff_length_scale)
        self.encoder = MLPEncoder(rff_dim + val_dim, embed_dim)
        self.blocks = nn.ModuleList([MassAwareBlock(embed_dim, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        d_in = embed_dim + (rff_dim if self.edge_rff else 0)
        self.edge_head = nn.Sequential(nn.Linear(d_in, embed_dim), nn.GELU(), nn.Linear(embed_dim, 3 * n_basis))

    def embed(self, batch: dict) -> torch.Tensor:
        """Trunk output [B, Nv, D] (padded vertices = 0)."""
        X, Y = batch['X'], batch['Input_funcs']
        if Y.shape[-1] != self.val_dim:
            raise ValueError(f"batch['Input_funcs'] has {Y.shape[-1]} features, the model was built with "
                             f"val_dim={self.val_dim} (feature_indices of the checkpoint?).")
        mask = batch.get('Mask', None)
        if mask is None:
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=X.device)
        w = mass_weights(batch, mask)
        h = self.encoder(torch.cat([self.rff(X), Y], dim=-1)) * mask.unsqueeze(-1).to(X.dtype)
        for blk in self.blocks:
            h = blk(h, mask, w)
        return self.norm(h)

    def edge_basis(self, batch: dict) -> torch.Tensor:
        """V [B, Ne, m] = ψ(x_mid) · t_e, 0 on padded and PEC-wall edges."""
        X, E = batch['X'], batch['Edges'].long()
        h = self.embed(batch)
        b = torch.arange(X.shape[0], device=X.device)[:, None]
        ia, ib = E[..., 0], E[..., 1]
        xa, xb = X[b, ia], X[b, ib]
        z = 0.5 * (h[b, ia] + h[b, ib])
        if self.edge_rff:
            z = torch.cat([z, self.rff(0.5 * (xa + xb))], dim=-1)
        psi = self.edge_head(z).view(*E.shape[:2], self.n_basis, 3)
        V = torch.einsum('bemc,bec->bem', psi, xb - xa)
        keep = batch['EdgeMask'] & ~batch['BndEdge'].bool()
        return V * keep.unsqueeze(-1).to(V.dtype)

    def forward(self, batch: dict) -> dict:
        missing = [k for k in REQUIRED_KEYS if batch.get(k, None) is None]
        if missing:
            raise ValueError(f"EigenspaceOperator3D needs batch keys {missing}: use Maxwell3DDataset + "
                             "maxwell3d_collate.")
        basis = self.edge_basis(batch).float()
        with torch.autocast(device_type=basis.device.type, enabled=False):
            lam, modes, proj = hcurl_ritz(basis.double(), batch, self.num_field_modes, self.mass_ridge,
                                          self.drop_tol, self.eig_broadening, self.kp_tol, self.kp_maxiter)
        lam = lam.float()
        freq = physics_freq_z(0.5 * torch.log(lam.clamp(min=1e-12)), batch, self.freq_stats,
                              'EigenspaceOperator3D')
        return {'basis': basis, 'field': modes.float(), 'eigenvalues': lam, 'freq': freq,
                'M_mat': proj['M_div'].float(), 'L_mat': proj['A_V'].float(),
                'M_div': proj['M_div'], 'A_V': proj['A_V'], 'M_V': proj['M_V'], 'Z': proj['Z']}
