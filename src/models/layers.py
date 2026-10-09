"""Building blocks of the eigenspace operator: coordinate encoding, mass-aware linear attention,
the physical frequency map and a degeneracy-safe symmetric eigensolver."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

C0 = 299792458.0  # speed of light [m/s] (same constant as the data generator)


class RandomFourierFeatures(nn.Module):
    """Random Fourier Features (Rahimi & Recht 2007) of the Gaussian kernel
    k(x, y) = exp(−‖x − y‖² / (2 ℓ²)):  φ(x) = √(2/D) · [cos(Bx), sin(Bx)],  B_ij ~ N(0, 1/ℓ²).
    B is sampled once and frozen (a buffer): the kernel approximation is unbiased only for
    fixed random frequencies (Bochner)."""

    def __init__(self, in_dim, out_dim, length_scale=0.1):
        super().__init__()
        if out_dim % 2 != 0:
            raise ValueError(f"out_dim must be even (split into sin/cos), got {out_dim}.")
        self.in_dim, self.out_dim, self.length_scale = in_dim, out_dim, length_scale
        self.register_buffer('B', torch.randn(in_dim, out_dim // 2) / length_scale)
        self.register_buffer('scale', torch.tensor((2.0 / out_dim) ** 0.5))

    def forward(self, x):
        proj = x @ self.B
        return self.scale * torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
                                 nn.Linear(out_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    """Linear attention with the ELU+1 kernel. Optional key weights w [B, N_k] make it mass-aware
    (NEO): kv = Σ w φ(k) vᵀ, z = Σ w φ(k), a quadrature of a global integral operator; with Σ w = 1
    over the valid nodes it does not depend on how the domain is discretised."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-6

    def forward(self, query, key, value, mask=None, weights=None):
        b, n_q, d = query.shape
        n_k = key.shape[1]
        q = self.q_proj(query).view(b, n_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).view(b, n_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).view(b, n_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = F.elu(q) + 1.0, F.elu(k) + 1.0
        if mask is not None:
            k_mask = mask.view(b, 1, -1, 1).to(k.dtype)
            k, v = k * k_mask, v * k_mask
        if weights is not None:
            k = k * weights.view(b, 1, -1, 1).to(k.dtype)          # enters kv and Σ k alike
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        z = torch.einsum('bhnd,bhde->bhne', q, kv)
        z_norm = torch.einsum('bhnd,bhmd->bhn', q, k.sum(dim=2, keepdim=True)).unsqueeze(-1)
        out = (z / (z_norm + self.eps)).permute(0, 2, 1, 3).contiguous().reshape(b, n_q, d)
        return self.dropout(self.out_proj(out))


def mass_weights(batch: dict, mask: torch.Tensor) -> torch.Tensor:
    """Quadrature weights [B, N]: node volume normalised to Σ w = 1 over the valid nodes (uniform
    without batch['Area']); padding gets 0."""
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


def physics_freq_z(log_k, batch, freq_stats, who):
    """z-scored GHz of f = c·k / (2π·scale), k = exp(log_k) = √λ in the converter's normalised
    (scale-free) coordinates. λ(sΩ) = λ(Ω)/s² is exact, so the size enters only via batch['Scale']."""
    scale = batch.get('Scale', None)
    if scale is None or not freq_stats:
        raise ValueError(f"{who} needs batch['Scale'] (converter output) and freq_stats (set from the "
                         "training dataset by the Lightning module).")
    f_ghz = C0 * torch.exp(log_k) / (2.0 * math.pi * scale.to(log_k.dtype).unsqueeze(-1)) / 1e9
    return (f_ghz - freq_stats['mean']) / freq_stats['std']


class BroadenedEigh(torch.autograd.Function):
    """Symmetric eigh whose backward is finite for degenerate spectra.

    The exact eigenvector gradient gA = V (F ∘ VᵀḡV + diag(ḡλ)) Vᵀ has F_ij = 1/(λ_j − λ_i), ±inf
    for (near-)degenerate pairs — the normal case for cavities (dipole / quadrupole pairs). The
    Lorentzian-broadened F_ij = Δ / (Δ² + ε) is exact for |Δ| ≫ √ε and bounded by 1/(2√ε);
    subspace-invariant losses have zero gradient along those directions anyway."""

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
