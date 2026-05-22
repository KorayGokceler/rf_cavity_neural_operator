"""Physics-driven losses for Helmholtz eigenproblem on RF cavities.

Implements the Rayleigh-quotient + Gram-Schmidt recipe from
Rowan et al. (arXiv:2506.04375), "Solving Engineering Eigenvalue Problems with
Neural Networks Using the Rayleigh Quotient", adapted to GNOT-style mesh-based
mode prediction.

Provided primitives:
    rayleigh_quotient        — R[u] = ∫|∇u|²/∫u² with autograd OR FEM K,M path
    gram_schmidt_modes       — area-weighted analytic orthogonalization
    eigenvalue_ordering_loss — hinge penalty enforcing λ_k ≤ λ_{k+1}
    parametric_expected_rayleigh — Rayleigh averaged over batch (geometry dist.)
    PhysicsCurriculum        — epoch→weights scheduler for hybrid training
"""
from __future__ import annotations

from typing import Optional, Sequence, Literal

import torch
import torch.nn.functional as F


# ── Rayleigh quotient ──────────────────────────────────────────────────────

def _autograd_grad_field(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Per-mode spatial gradient of `field` w.r.t. `coords` via autograd.

    Args:
        field:  [B, N, K] mode field (must depend on coords through the model).
        coords: [B, N, D] node coordinates (D=2 for 2D), requires_grad=True.
    Returns:
        grad: [B, N, K, D] — ∂field_k/∂x_d for each mode and coord direction.
    """
    B, N, K = field.shape
    D = coords.shape[-1]
    grads = []
    for k in range(K):
        gk = torch.autograd.grad(
            outputs=field[..., k].sum(),
            inputs=coords,
            create_graph=field.requires_grad,
            retain_graph=True,
        )[0]                                    # [B, N, D]
        grads.append(gk)
    return torch.stack(grads, dim=-2)           # [B, N, K, D]


def rayleigh_quotient(
    field: torch.Tensor,
    coords: Optional[torch.Tensor] = None,
    area: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    mode: Literal["autograd", "fem"] = "autograd",
    K_sparse: Optional[Sequence[torch.Tensor]] = None,
    M_sparse: Optional[Sequence[torch.Tensor]] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Rayleigh quotient R[u_k] = ∫|∇u_k|² / ∫u_k² for each predicted mode k.

    Two evaluation modes:
      * ``mode="autograd"``: differentiate the field w.r.t. ``coords`` and
        integrate ‖∇E_k‖² with area weights. Requires ``coords.requires_grad_``
        before the model forward pass.
      * ``mode="fem"``: use precomputed sparse stiffness K and mass M matrices
        per sample. R_k = (E_k^T K E_k) / (E_k^T M E_k). Much cheaper but
        requires the FEM-cache dataset.

    Args:
        field:  [B, N, K] predicted mode fields at mesh nodes.
        coords: [B, N, D] mesh coordinates with requires_grad (autograd mode).
        area:   [B, N] node-area weights (quadrature). If None and mode=autograd,
                uses uniform 1/N weights.
        mask:   [B, N] boolean — True = valid (non-padding) node.
        K_sparse / M_sparse: list-of-sparse-tensors (per-sample, fem mode).
        eps:    denominator floor.
    Returns:
        R: [B, K] Rayleigh quotient per sample per mode.
    """
    B, N, K = field.shape
    device = field.device

    if mode == "fem":
        if K_sparse is None or M_sparse is None:
            raise ValueError("mode='fem' requires K_sparse and M_sparse lists.")
        R = torch.zeros(B, K, device=device, dtype=field.dtype)
        for b in range(B):
            Kb = K_sparse[b].to(device)
            Mb = M_sparse[b].to(device)
            n_b = Kb.shape[0]
            E_b = field[b, :n_b, :]             # [n_b, K]
            for k in range(K):
                e = E_b[:, k:k + 1]              # [n_b, 1]
                num = (e * torch.sparse.mm(Kb, e)).sum()
                den = (e * torch.sparse.mm(Mb, e)).sum().clamp_min(eps)
                R[b, k] = num / den
        return R

    # autograd path
    if coords is None:
        raise ValueError("mode='autograd' requires coords.")
    if not coords.requires_grad:
        raise RuntimeError(
            "Rayleigh autograd path requires coords.requires_grad=True before "
            "the model forward call. Set batch['X'].requires_grad_(True) "
            "before calling self.model(batch).")
    grad = _autograd_grad_field(field, coords)     # [B, N, K, D]
    grad_sq = (grad ** 2).sum(dim=-1)              # [B, N, K]

    if area is None:
        w = torch.ones(B, N, device=device, dtype=field.dtype) / float(N)
    else:
        w = area.to(field.dtype)
    if mask is not None:
        w = w * mask.to(field.dtype)
    w = w.unsqueeze(-1)                            # [B, N, 1]

    num = (grad_sq * w).sum(dim=1)                 # [B, K]
    den = ((field ** 2) * w).sum(dim=1).clamp_min(eps)
    return num / den


# ── Gram-Schmidt orthogonalization ─────────────────────────────────────────

def gram_schmidt_modes(
    field: torch.Tensor,
    area: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Area-weighted modified Gram-Schmidt over the K mode columns.

    Mode k is replaced by its component orthogonal to modes 0..k-1 under the
    discrete L² inner product ⟨u, v⟩ = Σ_i u_i v_i · area_i · mask_i.

    Args:
        field: [B, N, K] mode fields.
        area:  [B, N] quadrature weights (defaults to uniform).
        mask:  [B, N] validity mask.
        eps:   norm floor.
    Returns:
        field_ortho: [B, N, K] with columns mutually orthogonal in the
                     area-weighted inner product (per batch sample).
    """
    B, N, K = field.shape
    device = field.device

    if area is None:
        w = torch.ones(B, N, device=device, dtype=field.dtype) / float(N)
    else:
        w = area.to(field.dtype)
    if mask is not None:
        w = w * mask.to(field.dtype)
    w = w.unsqueeze(-1)                            # [B, N, 1]

    # Build orthogonalized columns iteratively. Out-of-place to keep grads clean.
    ortho_cols = [field[..., 0:1]]
    for k in range(1, K):
        u_k = field[..., k:k + 1]                  # [B, N, 1]
        for j, u_j in enumerate(ortho_cols):
            inner = (u_k * u_j * w).sum(dim=1, keepdim=True)        # [B, 1, 1]
            denom = ((u_j ** 2) * w).sum(dim=1, keepdim=True).clamp_min(eps)
            u_k = u_k - (inner / denom) * u_j
        ortho_cols.append(u_k)
    return torch.cat(ortho_cols, dim=-1)            # [B, N, K]


# ── Ordering hinge ─────────────────────────────────────────────────────────

def eigenvalue_ordering_loss(
    freq_pred: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Hinge on λ_k > λ_{k+1}.  Returns mean(relu(λ_k − λ_{k+1} + margin)).

    Acts on whatever scale ``freq_pred`` is on (z-score or GHz).  Margin is
    in the same units. Returns scalar.
    """
    if freq_pred.dim() == 1:
        freq_pred = freq_pred.unsqueeze(0)
    diff = freq_pred[..., :-1] - freq_pred[..., 1:] + margin
    return F.relu(diff).mean()


# ── Parametric expected Rayleigh ───────────────────────────────────────────

def parametric_expected_rayleigh(
    field: torch.Tensor,
    coords: Optional[torch.Tensor] = None,
    area: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    geom_weights: Optional[torch.Tensor] = None,
    mode: Literal["autograd", "fem"] = "autograd",
    K_sparse: Optional[Sequence[torch.Tensor]] = None,
    M_sparse: Optional[Sequence[torch.Tensor]] = None,
) -> torch.Tensor:
    """E_a[R_k(a)] — Rayleigh averaged over the batch (parameter distribution).

    Equivalent to the expected Rayleigh quotient minimisation in Rowan et al.
    Section 5.1 (Eq. 16) and 5.2 (Eq. 18), evaluated empirically over batch
    samples that index the parameter space.

    Returns scalar: weighted mean over batch of mean-over-modes Rayleigh.
    """
    R = rayleigh_quotient(field, coords=coords, area=area, mask=mask, mode=mode,
                          K_sparse=K_sparse, M_sparse=M_sparse)  # [B, K]
    R_per_sample = R.mean(dim=-1)                                # [B]
    if geom_weights is None:
        return R_per_sample.mean()
    w = geom_weights.to(R_per_sample.dtype)
    w = w / (w.sum() + 1e-12)
    return (R_per_sample * w).sum()


# ── Curriculum scheduler ───────────────────────────────────────────────────

class PhysicsCurriculum:
    """Three-phase scheduler mixing supervised and physics losses.

    Phase A (epoch < e1):   pure physics warmup
    Phase B (e1..e2):       linear ramp blending in supervised terms
    Phase C (epoch >= e2):  supervised-dominant with small physics anchors

    Returns a dict of weights consumed by lightning_module._compute_loss:
        field, freq, smooth, slot_ortho, rayleigh, order, param
    """

    def __init__(
        self,
        e1: int = 20,
        e2: int = 80,
        # Phase A / B physics anchors
        w_phys_rayleigh: float = 1.0,
        w_phys_order:    float = 0.1,
        w_phys_param:    float = 0.2,
        # Phase C target supervised weights
        w_field:         float = 1.0,
        w_freq:          float = 0.5,
        w_smooth:        float = 0.0,
        w_slot_ortho:    float = 0.0,
        # Phase C residual physics weights
        w_rayleigh_anchor: float = 0.05,
        w_order_anchor:    float = 0.01,
        w_param_anchor:    float = 0.0,
        enabled: bool = True,
    ):
        assert e2 >= e1 >= 0
        self.e1 = e1
        self.e2 = e2
        self.w_phys_rayleigh = w_phys_rayleigh
        self.w_phys_order = w_phys_order
        self.w_phys_param = w_phys_param
        self.w_field = w_field
        self.w_freq = w_freq
        self.w_smooth = w_smooth
        self.w_slot_ortho = w_slot_ortho
        self.w_rayleigh_anchor = w_rayleigh_anchor
        self.w_order_anchor = w_order_anchor
        self.w_param_anchor = w_param_anchor
        self.enabled = enabled

    def weights(self, epoch: int) -> dict:
        if not self.enabled:
            # Identity: caller uses its own fixed weights.
            return {
                'field': 1.0, 'freq': 1.0, 'smooth': 1.0, 'slot_ortho': 1.0,
                'rayleigh': 1.0, 'order': 1.0, 'param': 1.0,
            }
        if epoch < self.e1:
            t = 0.0
        elif epoch >= self.e2:
            t = 1.0
        else:
            span = max(self.e2 - self.e1, 1)
            t = (epoch - self.e1) / span

        # Linear blend between (Phase A: physics only) and (Phase C: supervised dominant)
        return {
            'field':     t * self.w_field,
            'freq':      t * self.w_freq,
            'smooth':    t * self.w_smooth,
            'slot_ortho': t * self.w_slot_ortho,
            'rayleigh':  (1.0 - t) * self.w_phys_rayleigh + t * self.w_rayleigh_anchor,
            'order':     (1.0 - t) * self.w_phys_order    + t * self.w_order_anchor,
            'param':     (1.0 - t) * self.w_phys_param    + t * self.w_param_anchor,
        }
