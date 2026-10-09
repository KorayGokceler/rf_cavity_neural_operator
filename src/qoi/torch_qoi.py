"""Differentiable cavity figures of merit (docs/24_CAVITY_QOI.md §0.5) on a
maxwell3d_collate batch.

    qoi_torch(batch, F, f_ghz, Rs=None, sigma=5.8e7, beta=1.0, convention='linac',
              peak_p=None) -> dict[str, Tensor [B, K]]

Same keys and values as the numpy reference src/qoi/operators.qoi_from_dofs,
evaluated for every (sample, mode) of a padded batch at once with the
per-geometry sparse operators the collate stacks block-diagonally on the padded
index space (§0.4; sample b's edge e ↔ column b·Ne_max + e, exactly like M):

    QoI_S      [B·Ne × B·Ne]     wall-loss quadratic form S
    QoI_Az     [B·P  × B·Ne]     axis evaluation (Ẽ along the beam axis)
    QoI_zeta, QoI_q [B, P]       axial coordinate / quadrature weight (q = 0 pad)
    QoI_Esurf, QoI_Hsurf [B·3Nf × B·Ne]   Cartesian rows 3i..3i+2 of surface point i
    QoI_SurfMask [B, Nf] bool,   QoI_scale [B] (s [m]),  QoI_Laxis [B] (normalised)
    M          [B·Ne × B·Ne]     volume mass (the batch's own)

Formulas (§0.2, peak-amplitude convention, every value normalised to U = 1 J;
u = F[b, :, k] any amplitude / sign, ω = 2π f, φ_p = ω s ζ_p/(βc)):

    U        ½ ε0 s³ uᵀMu
    P_c      ½ R_s/(ωμ0)² · uᵀSu
    V        s·|Σ q a e^{jφ}|,  a = A_z u
    E_pk     max|E_surf u|
    B_pk     max|H_surf u|/(ω s)
  then (amplitude 1/√U):  Q0 = ωU/P_c,  G = Q0·R_s,  R/Q = V²/(ωU),
  R_sh = V²/P_c ('circuit': both halved),  T = |Σ q a e^{jφ}| / Σ q|a|,
  E_acc = V/L_acc (L_acc given, else s·L_axis),  Epk/Eacc,  Bpk/Eacc [mT/(MV/m)].

Everything is float64 internally (the surface / axis operators are evaluated on
float32 Ritz fields).  Padding is inert: padded edges are empty columns, padded
axis points have q = 0, padded surface points are masked out of the peaks.
The complex voltage is formed from its real / imaginary parts (cos / sin
sums); |·| uses √(x² + tiny) so the gradient is finite at V = 0 (bounded by 1).
Peaks: exact max (piecewise smooth, gradient through the arg-max point) when
peak_p is None, else the p-norm (Σ_i |x_i|^p)^{1/p} over the surface points
(an upper bound of the max, → max as p → ∞, gradient spread over the near-peak
points), computed scale-stably as m·(Σ (|x_i|/m)^p)^{1/p} with m = the detached
max (that expression does not depend on m, so detaching m is exact).
"""
import math

import torch
from scipy import constants as _const

from src.models.hcurl import spmm

MU0 = float(_const.mu_0)
EPS0 = float(_const.epsilon_0)
C0 = float(_const.c)

QOI_KEYS = ('f_Hz', 'Rs_ohm', 'U_J', 'P_c_W', 'Q0', 'G_ohm', 'V_acc_V', 'T_transit',
            'R_over_Q_ohm', 'R_sh_ohm', 'L_acc_m', 'E_acc_Vm', 'E_pk_Vm', 'B_pk_T',
            'Epk_Eacc', 'Bpk_Eacc_mT_per_MVm')
# What the PKL stores per sample (docs/24 §0.3 QOI_LABELS); duplicated here so
# training does not depend on the numpy module being importable.
QOI_LABELS = ('Q0', 'G_ohm', 'R_over_Q_ohm', 'R_sh_ohm', 'T_transit', 'Epk_Eacc',
              'Bpk_Eacc_mT_per_MVm')
# Quantities that involve the on-axis voltage (ill-defined for modes that do
# not accelerate: V ≈ 0 → log R/Q, Epk/Eacc ... blow up).
VOLTAGE_KEYS = ('V_acc_V', 'T_transit', 'R_over_Q_ohm', 'R_sh_ohm', 'E_acc_Vm', 'Epk_Eacc',
                'Bpk_Eacc_mT_per_MVm')
QOI_BATCH_KEYS = ('QoI_S', 'QoI_Az', 'QoI_zeta', 'QoI_q', 'QoI_Esurf', 'QoI_Hsurf',
                  'QoI_SurfMask', 'QoI_Laxis')
CONVENTIONS = ('linac', 'circuit')

_TINY = 1e-300


def has_qoi_ops(batch) -> bool:
    """True when the batch carries the per-geometry QoI operators (§0.4)."""
    return all(batch.get(k, None) is not None for k in QOI_BATCH_KEYS)


def _sp(batch, key, dtype, device):
    A = batch[key]
    if A.dtype != dtype:
        A = A.to(dtype)
    return A if A.device == device else A.to(device)


def _peak(A, F, mask, p):
    """max_i |(A F)_i| over the valid surface points; A [B·3Nf, B·Ne], F [B, Ne, K],
    mask [B, Nf] → [B, K] (exact max, or the p-norm soft max)."""
    Y = spmm(A, F)                                                    # [B, 3Nf, K]
    B, R, K = Y.shape
    mag2 = Y.view(B, R // 3, 3, K).pow(2).sum(2)                      # [B, Nf, K]
    mag2 = torch.where(mask.unsqueeze(-1), mag2, torch.zeros_like(mag2))
    mx = mag2.amax(1)                                                 # [B, K]
    if p is None:
        return torch.sqrt(mx + _TINY)
    p = float(p)
    if p < 1.0:
        raise ValueError(f"peak_p must be >= 1 (p-norm), got {p}")
    m2 = mx.detach().clamp(min=_TINY).unsqueeze(1)                    # [B, 1, K]
    s = (mag2 / m2).pow(0.5 * p).sum(1)                               # ≥ 1 (the max term)
    return m2.squeeze(1).sqrt() * s.pow(1.0 / p)


def qoi_torch(batch, F, f_ghz, Rs=None, sigma=5.8e7, beta=1.0, convention='linac',
              peak_p=None, L_acc=None):
    """CST-style figures of merit of the fields F [B, Ne_max, K] (N0 DOFs on the
    normalised meshes, any amplitude / sign) at frequencies f_ghz [B, K] (or
    [B] / scalar, broadcast).  Rs: fixed surface resistance [Ω] (scalar or [B, K]),
    default copper R_s(f) = √(π f μ0/σ) at each mode's own frequency.
    L_acc: accelerating length [m] (scalar or [B]); default s·L_axis.
    Returns {key: float64 [B, K]} for QOI_KEYS (see module docstring)."""
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")
    missing = [k for k in QOI_BATCH_KEYS + ('M',) if batch.get(k, None) is None]
    if missing:
        raise ValueError(f"qoi_torch needs batch keys {missing}: build the dataset with "
                         "Maxwell3DDataset(..., qoi_ops=True).")
    dt = torch.float64
    F = F.to(dt)
    dev = F.device
    B, _, K = F.shape
    sp = lambda k: _sp(batch, k, dt, dev)                            # noqa: E731
    vec = lambda k: batch[k].to(device=dev, dtype=dt)                # noqa: E731

    s = (batch['QoI_scale'] if batch.get('QoI_scale', None) is not None
         else batch['Scale']).to(device=dev, dtype=dt).reshape(B, 1)  # [B, 1] metres
    f_hz = torch.as_tensor(f_ghz, dtype=dt, device=dev) * 1e9
    if f_hz.dim() == 1:                                              # [B]: one frequency per sample
        f_hz = f_hz.reshape(B, 1)
    f_hz = torch.broadcast_to(f_hz, (B, K))
    omega = 2.0 * math.pi * f_hz
    if Rs is None:
        Rs = torch.sqrt(math.pi * f_hz * MU0 / float(sigma))
    else:
        Rs = torch.broadcast_to(torch.as_tensor(Rs, dtype=dt, device=dev), (B, K))

    # quadratic forms
    uMu = (F * spmm(sp('M'), F)).sum(1)                              # [B, K]
    uSu = (F * spmm(sp('QoI_S'), F)).sum(1)
    U = (0.5 * EPS0 * s ** 3 * uMu).clamp(min=_TINY)
    P = (0.5 * Rs / (omega * MU0) ** 2 * uSu).clamp(min=_TINY)

    # on-axis voltage: complex sum via cos / sin
    a = spmm(sp('QoI_Az'), F)                                        # [B, P, K]
    q = vec('QoI_q').unsqueeze(-1)                                   # [B, P, 1]
    zeta = vec('QoI_zeta').unsqueeze(-1)
    phase = zeta * (omega * s / (float(beta) * C0)).unsqueeze(1)     # [B, P, K]
    qa = q * a
    re, im = (qa * torch.cos(phase)).sum(1), (qa * torch.sin(phase)).sum(1)
    vabs = torch.sqrt(re * re + im * im + _TINY)                     # |Σ q a e^{jφ}|
    T = vabs / (q * a.abs()).sum(1).clamp(min=_TINY)
    V = s * vabs

    # surface peaks
    mask = batch['QoI_SurfMask'].to(dev).bool()
    e_pk = _peak(sp('QoI_Esurf'), F, mask, peak_p)
    h_pk = _peak(sp('QoI_Hsurf'), F, mask, peak_p)
    E_pk, B_pk = e_pk, h_pk / (omega * s)

    # U = 1 J normalisation (field amplitude 1/√U)
    rU = torch.sqrt(U)
    V_n, P_n, E_pk, B_pk = V / rU, P / U, E_pk / rU, B_pk / rU
    half = 0.5 if convention == 'circuit' else 1.0
    Q0 = omega / P_n
    if L_acc is None:
        L = s * vec('QoI_Laxis').reshape(B, 1)
    else:
        L = torch.as_tensor(L_acc, dtype=dt, device=dev).reshape(-1, 1)
    L = torch.broadcast_to(L, (B, K))
    E_acc = V_n / L.clamp(min=_TINY)
    E_acc_safe = E_acc.clamp(min=_TINY)
    return {
        'f_Hz': f_hz, 'Rs_ohm': Rs, 'U_J': torch.ones_like(U), 'P_c_W': P_n,
        'Q0': Q0, 'G_ohm': Q0 * Rs, 'V_acc_V': V_n, 'T_transit': T,
        'R_over_Q_ohm': half * V_n ** 2 / omega, 'R_sh_ohm': half * V_n ** 2 / P_n,
        'L_acc_m': L, 'E_acc_Vm': E_acc, 'E_pk_Vm': E_pk, 'B_pk_T': B_pk,
        'Epk_Eacc': E_pk / E_acc_safe, 'Bpk_Eacc_mT_per_MVm': B_pk * 1e9 / E_acc_safe,
    }
