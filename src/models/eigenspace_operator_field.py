"""EigenspaceOperatorField (model option B, docs/28): the same trunk and Ritz layer as
EigenspaceOperator3D, but the learned basis lives in a HIGH-ORDER H(curl) space on a curved mesh,
so neither the Ritz frequencies nor the fields have the N0 floor of the edge model.

The edge model already evaluates m learned vector fields ψ_j at one point per edge (the midpoint)
and reduces them to Whitney DOFs, V[e, j] = ψ_j(x_mid)·t_e.  Here the same ψ_j are evaluated at
the nq quadrature points of every (curved) element and reduced to the DOFs of HCurl(p) by a fixed
linear map (src.data_gen.highorder.field_operators):

    ψ_j(x_q)  →  per element: L2(p) coefficients  c = P_t ψ      (|J|-weighted least squares)
              →  HCurl(p) DOFs  V = C · c                         (projection-based interpolation)

The trunk's tokens are the quadrature points themselves (features from highorder.point_features,
mass weights = quadrature weights: the attention integrals are the element quadrature), and
ψ(x_q) = head(h_q ‖ RFF(x_q)) — a geometric field, so the DOFs transform correctly under any mesh
renumbering; wall DOFs and padding are zeroed (n × E = 0).  The Rayleigh–Ritz with the kernel projection (hcurl_ritz) is unchanged: it only needs
the space's K, M, G, Kp.
"""
import torch

from src.models.eigenspace_operator_3d import EigenspaceOperator3D
from src.models.hcurl import hcurl_ritz, spmm
from src.models.layers import physics_freq_z

REQUIRED_KEYS = ('X', 'Input_funcs', 'QX', 'QP', 'C', 'DofMask', 'BndDof', 'M', 'K', 'G', 'Gt', 'Kp',
                 'Kp_diag', 'Scale')


class EigenspaceOperatorField(EigenspaceOperator3D):
    """Geometry → m learned vector fields → HCurl(p) DOFs → K projected Ritz eigenpairs.

    forward(batch) returns the keys of EigenspaceOperator3D.forward; 'basis' / 'field' are DOF
    vectors [B, Ndof, ·] of the batch's HCurl(p) space."""

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
        V = spmm(batch['C'], coef.reshape(B, -1, self.n_basis).double())
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
