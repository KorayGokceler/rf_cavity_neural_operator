"""EigenspaceOperatorField (docs/28): learned H(curl) trial space on a curved mesh + kernel-projected
Rayleigh–Ritz for 3D Maxwell cavity eigenmodes (E field).

    curl curl E = k² E in Ω,  n × E = 0 on the PEC wall (essential)
    discrete:   K y = λ M y on the non-wall DOFs of HCurl(p), wall DOFs 0,
                ker K|H0(curl) = span(G) (gradients of the potential space).

1. Trunk: RFF(x) ‖ token features → MLP → n_layers mass-aware linear-attention blocks (key / value
   sums weighted by the token weights batch['Area'], NEO).  The tokens are the nq quadrature points
   of every (curved) element (features from highorder.point_features, mass weights = quadrature
   weights: the attention integrals are the element quadrature).
   grad_checkpoint: in training, keep only the inputs of each block's attention and FFN halves and
   recompute them in the backward (one extra trunk forward for ~÷2 training memory; docs/31 §3.6).
2. Field basis: m learned vector fields ψ_j(x_q) = head(h_q ‖ RFF(x_q)) at the quadrature points,
   reduced to the DOFs of HCurl(p) by a fixed linear map (src.data_gen.highorder.field_operators):

       ψ_j(x_q)  →  per element: L2(p) coefficients  c = P_t ψ      (|J|-weighted least squares)
                 →  HCurl(p) DOFs  V = C · c                         (projection-based interpolation)

   ψ is a geometric field, so the DOFs transform correctly under any mesh renumbering; wall DOFs and
   padding are zeroed (n × E = 0 by construction).
3. Rayleigh–Ritz on the divergence-free projection PV = V − G Kp⁻¹GᵀMV (hcurl.hcurl_ritz):
   A_V = VᵀKV, M_div = VᵀMV − BᵀKp⁻¹B, float64 rank-revealing generalized eigh (pure-gradient
   directions are dropped, not ridged), fields PV·c with unit M-norm, f = c·√λ/(2π·scale).
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.models.hcurl import hcurl_ritz, spmm
from src.models.layers import MassAwareBlock, MLPEncoder, RandomFourierFeatures, mass_weights, physics_freq_z

REQUIRED_KEYS = ('X', 'Input_funcs', 'QX', 'QP', 'C', 'DofMask', 'BndDof', 'M', 'K', 'G', 'Gt', 'Kp',
                 'Kp_diag', 'Scale')


class EigenspaceOperatorField(nn.Module):
    """Geometry → m learned vector fields → HCurl(p) DOFs → K projected Ritz eigenpairs.

    forward(batch) returns
        'basis'       [B, Ndof, m]  raw DOF basis V (pre-projection, masked)
        'field'       [B, Ndof, K]  projected Ritz modes, unit M-norm (sign arbitrary)
        'eigenvalues' [B, K]        Ritz values λ (normalised mesh)
        'freq'        [B, K]        z-scored GHz of f = c·√λ / (2π·scale)
        'M_mat', 'L_mat' [B, m, m]  projected mass M_div / curl–curl A_V Grams (float32)
        'M_div', 'A_V', 'M_V' [B, m, m] (float64) and 'Z' [B, Np, m] = Kp⁻¹GᵀMV, reused by the
            losses (hcurl_grams) without a second Kp solve.
    freq_stats ({'mean', 'std'} GHz) is set by the Lightning module.
    """

    def __init__(self, val_dim, embed_dim=128, n_layers=4, n_heads=4, n_basis=16,
                 num_field_modes=6, rff_dim=64, rff_length_scale=0.1, dropout=0.0,
                 edge_rff=True, mass_ridge=1e-8, drop_tol=1e-6, eig_broadening=1e-4,
                 kp_tol=1e-8, kp_maxiter=2000, grad_checkpoint=False):
        super().__init__()
        if n_basis < num_field_modes:
            raise ValueError(f"n_basis={n_basis} < num_field_modes={num_field_modes}: "
                             "Rayleigh–Ritz needs m ≥ K.")
        self.val_dim, self.num_field_modes, self.n_basis = val_dim, num_field_modes, n_basis
        self.mass_ridge, self.drop_tol, self.eig_broadening = mass_ridge, drop_tol, eig_broadening
        self.kp_tol, self.kp_maxiter = kp_tol, kp_maxiter
        self.edge_rff = bool(edge_rff)          # RFF(x_q) enters the field head (name kept: checkpoints)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.freq_stats = None

        # module names are part of the checkpoint state dict: keep them
        self.rff = RandomFourierFeatures(3, rff_dim, length_scale=rff_length_scale)
        self.encoder = MLPEncoder(rff_dim + val_dim, embed_dim)
        self.blocks = nn.ModuleList([MassAwareBlock(embed_dim, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        d_in = embed_dim + (rff_dim if self.edge_rff else 0)
        self.edge_head = nn.Sequential(nn.Linear(d_in, embed_dim), nn.GELU(), nn.Linear(embed_dim, 3 * n_basis))

    def embed(self, batch: dict) -> torch.Tensor:
        """Trunk output [B, N, D] over the tokens (padded tokens = 0)."""
        X, Y = batch['X'], batch['Input_funcs']
        if Y.shape[-1] != self.val_dim:
            raise ValueError(f"batch['Input_funcs'] has {Y.shape[-1]} features, the model was built with "
                             f"val_dim={self.val_dim} (feature_indices of the checkpoint?).")
        mask = batch.get('Mask', None)
        if mask is None:
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=X.device)
        w = mass_weights(batch, mask)
        ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        inp = torch.cat([self.rff(X), Y], dim=-1)
        h = (checkpoint(self.encoder, inp, use_reentrant=False) if ckpt else self.encoder(inp)) \
            * mask.unsqueeze(-1).to(X.dtype)
        for blk in self.blocks:
            if ckpt:        # keep the half-block inputs; the backward recomputes one half at a time
                h = checkpoint(blk.attn_part, h, mask, w, use_reentrant=False)
                h = checkpoint(blk.ffn_part, h, mask, use_reentrant=False)
            else:
                h = blk(h, mask, w)
        return self.norm(h)

    def field_basis(self, batch: dict) -> torch.Tensor:
        """V [B, Ndof, m]: ψ at the quadrature points → HCurl DOFs, 0 on the wall and padding.  The
        trunk's tokens ARE the quadrature points (batch X / Input_funcs / Area = weights)."""
        qx = batch['QX']                                                 # [B, Nt, nq, 3]
        B, Nt, nq, _ = qx.shape
        z = self.embed(batch).view(B, Nt, nq, -1)                        # [B, Nt, nq, D]
        if self.edge_rff:
            z = torch.cat([z, self.rff(qx.to(z.dtype))], dim=-1)
        psi = self.edge_head(z).view(B, Nt, nq, self.n_basis, 3)
        coef = torch.einsum('btlq,btqmc->btclm', batch['QP'].to(z.dtype), psi)   # [B, Nt, 3, nl, m]
        V = spmm(batch['C'], coef.reshape(B, -1, self.n_basis).double(), batch.get('Ct'))
        keep = batch['DofMask'] & ~batch['BndDof'].bool()
        return V * keep.unsqueeze(-1).to(V.dtype)

    def forward(self, batch: dict) -> dict:
        missing = [k for k in REQUIRED_KEYS if batch.get(k, None) is None]
        if missing:
            raise ValueError(f"EigenspaceOperatorField needs batch keys {missing}: use the field dataset "
                             "(src.data.field_dataset) and its collate.")
        basis = self.field_basis(batch).float()
        with torch.autocast(device_type=basis.device.type, enabled=False):
            lam, modes, proj = hcurl_ritz(basis.double(), batch, self.num_field_modes, self.mass_ridge,
                                          self.drop_tol, self.eig_broadening, self.kp_tol, self.kp_maxiter)
        lam = lam.float()
        freq = physics_freq_z(0.5 * torch.log(lam.clamp(min=1e-12)), batch, self.freq_stats,
                              'EigenspaceOperatorField')
        return {'basis': basis, 'field': modes.float(), 'eigenvalues': lam, 'freq': freq,
                'M_mat': proj['M_div'].float(), 'L_mat': proj['A_V'].float(),
                'M_div': proj['M_div'], 'A_V': proj['A_V'], 'M_V': proj['M_V'], 'Z': proj['Z']}
