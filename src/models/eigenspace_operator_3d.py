"""EigenspaceOperator3D: learned H(curl) trial space + kernel-projected
Rayleigh–Ritz for 3D Maxwell cavity eigenmodes (H field, Whitney N0 DOFs).

    curl curl H = k² H in Ω,  n·H = 0, n×curl H = 0 on PEC ∂Ω  (both natural)
    discrete:   K y = λ M y  on ALL edges,  ker K = G·P1  (docs/18 §1.4, §3.4)

1. Trunk (reused from the 2D EigenspaceOperator): RFF(xyz) ‖ vertex features
   → MLP → n_layers mass-aware linear-attention blocks, key/value sums
   weighted by the node volumes (batch['Area']).
2. Edge basis: for edge e = (a → b) (low → high vertex index), a vector MLP
   evaluates m vector fields at the midpoint from the symmetric input
   ½(h_a + h_b) ‖ RFF(x_mid) and dots them with t_e = x_b − x_a:
       V[e, j] = ψ_j(x_mid) · t_e  ≈ ∫_e ψ_j · t   (midpoint rule, docs/18 §3.3).
   Orientation is structural: flipping (a, b) flips t_e and nothing else, so
   V[e] changes sign exactly as the target DOFs do.  No boundary gate (both
   PEC conditions are natural for H); padded edges give 0.
3. Rayleigh–Ritz on the divergence-free projection PV = V − G Kp⁻¹GᵀMV
   (hcurl.hcurl_ritz): A_V = VᵀKV, M_div = VᵀMV − BᵀKp⁻¹B, float64
   rank-revealing generalized eigh (hcurl.projected_eigh: directions that are
   pure gradients are dropped, not ridged), fields PV·c with unit M-norm,
   f = c·√λ/(2π·scale).  Gradient content in span(V) can neither pull θ
   below λ_h nor create θ = 0.
"""
import torch
import torch.nn as nn

from src.models.eigenspace_operator import EigenspaceOperator
from src.models.gnot import _physics_freq_z
from src.models.hcurl import hcurl_ritz

REQUIRED_KEYS = ('Edges', 'EdgeMask', 'M', 'K', 'G', 'Gt', 'Kp', 'Kp_diag', 'Scale')


class EigenspaceOperator3D(EigenspaceOperator):
    """Geometry → m edge basis functions → K projected Ritz eigenpairs.

    forward(batch) returns
        'basis'       [B, Ne, m]  raw edge basis V (pre-projection, masked)
        'field'       [B, Ne, K]  projected Ritz modes, unit M-norm (sign arbitrary)
        'eigenvalues' [B, K]      Ritz values λ (normalised mesh)
        'freq'        [B, K]      z-scored GHz of f = c·√λ / (2π·scale)
        'M_mat', 'L_mat' [B, m, m]  projected mass M_div / curl–curl A_V Grams
        'M_div', 'A_V', 'M_V' [B, m, m] (float64: projected mass, curl–curl,
            unprojected mass) and 'Z' [B, Nv, m] = Kp⁻¹GᵀMV, reused by the
            losses (hcurl_grams) without a second Kp solve.
    """

    def __init__(self, val_dim, embed_dim=128, n_layers=4, n_heads=4, n_basis=16,
                 num_field_modes=6, rff_dim=64, rff_length_scale=0.1, dropout=0.0,
                 edge_rff=True, physics_freq=True, mass_ridge=1e-8, drop_tol=1e-6,
                 eig_broadening=1e-4, kp_tol=1e-8, kp_maxiter=2000):
        super().__init__(val_dim, grid_dim=3, embed_dim=embed_dim, n_layers=n_layers,
                         n_heads=n_heads, n_basis=n_basis, num_field_modes=num_field_modes,
                         rff_dim=rff_dim, rff_length_scale=rff_length_scale, dropout=dropout,
                         physics_freq=physics_freq, mass_ridge=mass_ridge,
                         eig_broadening=eig_broadening)
        del self.head                                         # scalar nodal head → edge head
        self.edge_rff = bool(edge_rff)
        self.drop_tol, self.kp_tol, self.kp_maxiter = drop_tol, kp_tol, kp_maxiter
        d_in = embed_dim + (rff_dim if self.edge_rff else 0)
        self.edge_head = nn.Sequential(nn.Linear(d_in, embed_dim), nn.GELU(),
                                       nn.Linear(embed_dim, 3 * n_basis))

    def edge_basis(self, batch: dict) -> torch.Tensor:
        """V [B, Ne, m] = ψ(x_mid) · t_e, 0 on padded edges."""
        X, E = batch['X'], batch['Edges'].long()
        h = self.embed(batch)                                 # [B, Nv, D]
        b = torch.arange(X.shape[0], device=X.device)[:, None]
        ia, ib = E[..., 0], E[..., 1]
        xa, xb = X[b, ia], X[b, ib]
        z = 0.5 * (h[b, ia] + h[b, ib])
        if self.edge_rff:
            z = torch.cat([z, self.rff(0.5 * (xa + xb))], dim=-1)
        psi = self.edge_head(z).view(*E.shape[:2], self.n_basis, 3)
        V = torch.einsum('bemc,bec->bem', psi, xb - xa)
        return V * batch['EdgeMask'].unsqueeze(-1).to(V.dtype)

    def forward(self, batch: dict) -> dict:
        missing = [k for k in REQUIRED_KEYS if batch.get(k, None) is None]
        if missing:
            raise ValueError(f"EigenspaceOperator3D needs batch keys {missing}: "
                             "use Maxwell3DDataset + maxwell3d_collate.")
        basis = self.edge_basis(batch).float()
        with torch.autocast(device_type=basis.device.type, enabled=False):
            lam, modes, proj = hcurl_ritz(
                basis.double(), batch, self.num_field_modes, self.mass_ridge,
                self.drop_tol, self.eig_broadening, self.kp_tol, self.kp_maxiter)
        lam = lam.float()
        freq = _physics_freq_z(0.5 * torch.log(lam.clamp(min=1e-12)), batch,
                               self.freq_stats, 'EigenspaceOperator3D')
        return {'basis': basis, 'field': modes.float(), 'eigenvalues': lam, 'freq': freq,
                'M_mat': proj['M_div'].float(), 'L_mat': proj['A_V'].float(),
                'M_div': proj['M_div'], 'A_V': proj['A_V'], 'M_V': proj['M_V'], 'Z': proj['Z']}
