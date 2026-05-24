import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.eps = 1e-6

    def forward(self, query, key, value, mask=None):
        b, n_q, d = query.shape
        b, n_k, _ = key.shape

        q = self.q_proj(query).view(b, n_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).view(b, n_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).view(b, n_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Use ELU+1 kernel for more stable linear attention
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        if mask is not None:
            # Mask out invalid key/values
            k_mask = mask.view(b, 1, -1, 1).to(k.dtype)
            k = k * k_mask
            v = v * k_mask

        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        z = torch.einsum('bhnd,bhde->bhne', q, kv)

        k_sum = k.sum(dim=2, keepdim=True)
        z_norm = torch.einsum('bhnd,bhmd->bhn', q, k_sum).unsqueeze(-1)

        output = z / (z_norm + self.eps)
        output = output.permute(0, 2, 1, 3).contiguous().reshape(b, n_q, d)

        return self.out_proj(output)

class AttentionPool(nn.Module):
    """Learned summary token for global pooling."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # x: [B, N, D]
        b = x.shape[0]
        q = self.query.expand(b, -1, -1)
        
        # PyTorch MultiheadAttention expects key_padding_mask to be True for elements to ignore
        pytorch_mask = ~mask if mask is not None else None
        
        out, _ = self.attn(q, x, x, key_padding_mask=pytorch_mask)
        return self.norm(out).squeeze(1)

class GeometricGatingFFN(nn.Module):
    def __init__(self, embed_dim, num_experts=4, dropout=0.1):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x, gate_weights):
        # gate_weights: [B, N, num_experts]
        
        # Weighted sum of ALL experts for spatial continuity
        # Since num_experts is small (4), full dense evaluation is very stable and fast.
        final_output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            w = gate_weights[..., i:i+1] # [B, N, 1]
            final_output = final_output + w * expert(x)
                
        return final_output

class GNOTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, coords_dim, num_experts=4, dropout=0.0):
        super().__init__()
        self.cross_attn = LinearAttention(embed_dim, num_heads, dropout)
        self.norm1_q  = nn.LayerNorm(embed_dim)
        self.norm1_kv = nn.LayerNorm(embed_dim)
        self.self_attn = LinearAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = GeometricGatingFFN(embed_dim, num_experts, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)


        self.local_gating = nn.Sequential(
            nn.Linear(coords_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, num_experts)
        )
        self.register_buffer('_expert_calls', torch.zeros(num_experts, dtype=torch.long))

    def forward(self, x, condition_emb, mode_idx, pos, mask=None, condition_mask=None, global_context=None):
        attn_out = self.cross_attn(
            query=self.norm1_q(x),
            key=self.norm1_kv(condition_emb),
            value=self.norm1_kv(condition_emb),
            mask=condition_mask
        )
        x = x + attn_out

        # Global Context Injection: Kavitenin "Büyük Resmini" her noktaya aşıla.
        # Bu, yön-duyarlı (dipol vb.) fiziklerin asimetrisini anlamak için kritiktir.
        if global_context is not None:
            # global_context: [B, D] -> [B, 1, D]
            x = x + global_context.unsqueeze(1)

        attn_out = self.self_attn(
            query=self.norm2(x),
            key=self.norm2(x),
            value=self.norm2(x),
            mask=mask
        )
        x = x + attn_out
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        # Local Smooth Gating Logic - Uses weighted sum of all experts for continuity
        gate_logits = self.local_gating(pos)
        # Temperature 0.5 to soften transitions and prevent patchy fields
        gate_weights = F.softmax(gate_logits / 0.5, dim=-1) # [B, N, num_experts]

        # Use full weighted sum for spatial continuity in physics fields
        ffn_out = self.ffn(self.norm3(x), gate_weights)
        x = x + ffn_out



        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x





class RandomFourierFeatures(nn.Module):
    """Random Fourier Features for Gaussian-kernel coordinate encoding.

    Rahimi & Recht (2007). Approximates k(x, y) = exp(-||x - y||^2 / (2 * length_scale^2))
    via a finite-dimensional embedding:
        phi(x) = sqrt(2 / D) * [cos(B x), sin(B x)]
    where B_ij ~ N(0, 1 / length_scale^2) and D is the output dimension.
    Frequencies are fixed (not learned): per Bochner's theorem the kernel
    approximation is unbiased only when B is sampled and frozen.
    """
    def __init__(self, in_dim, out_dim, length_scale=0.1):
        super().__init__()
        if out_dim % 2 != 0:
            raise ValueError(f"out_dim must be even (split into sin/cos), got {out_dim}.")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.length_scale = length_scale
        sigma = 1.0 / length_scale
        B = torch.randn(in_dim, out_dim // 2) * sigma
        self.register_buffer('B', B)
        self.register_buffer('scale', torch.tensor((2.0 / out_dim) ** 0.5))

    def forward(self, x):
        # x: [..., in_dim] -> [..., out_dim]
        proj = x @ self.B
        return self.scale * torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

class MLPEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def _apply_gram_schmidt(field: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Apply masked Gram-Schmidt orthonormalization to [B, N, K] field columns.

    For each batch element independently, orthonormalizes the K field columns
    with respect to the L2 inner product over valid (unmasked) nodes.  After
    orthonormalization the columns satisfy Q^T Q = I (over valid nodes).

    This is an optional post-processing step: it enforces column orthonormality
    structurally, removing the need for the slot_ortho_weight penalty.

    Args:
        field: [B, N, K] predicted field values.
        mask:  [B, N] boolean (True = valid node) or None.
    Returns:
        [B, N, K] with orthonormal columns (over valid nodes).
    """
    B, N, K = field.shape
    out = field.clone()
    m = mask.float().unsqueeze(-1) if mask is not None else None  # [B, N, 1]

    for k in range(K):
        v = out[:, :, k]          # [B, N]
        for j in range(k):
            u = out[:, :, j]      # [B, N]
            if m is not None:
                # masked inner products: [B]
                dot_vu = (v * u * m.squeeze(-1)).sum(dim=1)
                dot_uu = (u * u * m.squeeze(-1)).sum(dim=1).clamp(min=1e-8)
            else:
                dot_vu = (v * u).sum(dim=1)
                dot_uu = (u * u).sum(dim=1).clamp(min=1e-8)
            proj = dot_vu / dot_uu   # [B]
            v = v - proj.unsqueeze(1) * u
        # Normalize v
        if m is not None:
            nrm = (v * v * m.squeeze(-1)).sum(dim=1).clamp(min=1e-8).sqrt()
        else:
            nrm = v.norm(dim=1).clamp(min=1e-8)
        out[:, :, k] = v / nrm.unsqueeze(1)

    return out


class GNOTModel(nn.Module):
    def __init__(self, val_dim=8, grid_dim=2, embed_dim=256,
                 n_shared_layers=6, n_mode_layers=1, n_field_head_layers=3,
                 n_heads=8, num_experts=4, num_field_modes=3,
                 use_checkpoint=False, predict_frequency=True,
                 dropout=0.0,
                 rff_dim=64, rff_length_scale=0.1,
                 n_basis=16,
                 orthonormalize_output=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_field_modes = num_field_modes

        # 1. Coordinate Encoding — Random Fourier Features (Gaussian kernel)
        self.spatial_encoder = RandomFourierFeatures(grid_dim, rff_dim, length_scale=rff_length_scale)
        router_dim = rff_dim
        input_dim_q = rff_dim
        input_dim_f = val_dim + grid_dim

        # Query points
        self.query_encoder = MLPEncoder(input_dim_q, embed_dim)
        # Input features + Geometry Context
        self.input_func_encoder = MLPEncoder(input_dim_f, embed_dim)

        # ── Frequency-first slot decoding ────────────────────────────────────
        # There is no more `mode_idx` input.  The model predicts the full set of
        # K eigenfrequencies from a global geometry context, then decodes one
        # field per slot conditioned on its (predicted) frequency.
        self.freq_embed = nn.Sequential(
            nn.Linear(1, embed_dim // 4), nn.SiLU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )
        self.slot_queries = nn.Parameter(torch.randn(num_field_modes, embed_dim))
        nn.init.normal_(self.slot_queries, mean=0.0, std=0.02)
        self.freq_head_global = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.SiLU(),
            nn.Linear(embed_dim // 2, num_field_modes)
        )

        # Mode-specific field branches
        self.mode_field_blocks = nn.ModuleList([
            nn.ModuleList([
                 GNOTBlock(embed_dim, n_heads, num_experts=num_experts, coords_dim=router_dim, dropout=dropout)
                 for _ in range(n_mode_layers)
            ]) for _ in range(num_field_modes)
        ])
        
        self.pooler = AttentionPool(embed_dim, n_heads)       # global context + freq

        # ── Per-mode direct field head ────────────────────────────────────────
        # No DeepONet factorization (no Φ basis / α coefficients).  Each mode
        # has its own pointwise MLP that maps the per-mode node representation
        # x_m [B, N, D] directly to a scalar field value [B, N, 1].
        # Depth is controlled by n_field_head_layers.  Heads are independent
        # across modes; near-degenerate handling lives entirely in the loss.
        def _make_field_head(dim, n_layers):
            layers = [nn.LayerNorm(dim)]
            curr = dim
            if n_layers > 2:
                layers += [nn.Linear(curr, dim * 2), nn.LayerNorm(dim * 2), nn.GELU()]
                curr = dim * 2
                for _ in range(n_layers - 3):
                    layers += [nn.Linear(curr, curr), nn.LayerNorm(curr), nn.GELU()]
            if n_layers >= 2:
                layers += [nn.Linear(curr, dim), nn.LayerNorm(dim), nn.GELU()]
                curr = dim
            layers.append(nn.Linear(curr, 1))
            return nn.Sequential(*layers)

        self.field_heads = nn.ModuleList([
            _make_field_head(embed_dim, n_field_head_layers)
            for _ in range(num_field_modes)
        ])

        # Frequency is now predicted by a single global head (`freq_head_global`)
        # that emits all K eigenfrequencies at once from the pooled geometry
        # context.  Per-mode frequency heads are gone.
        self.predict_frequency = predict_frequency
        self.orthonormalize_output = orthonormalize_output

        # Initialize heads with small weights to prevent early training explosion
        self._init_weights()

    def _init_weights(self):
        # field_heads: moderate init; final layer small → field ≈ 0 at start
        # (prevents early training explosion).
        for head in self.field_heads:
            head_linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
            for m in head_linears:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            nn.init.trunc_normal_(head_linears[-1].weight, std=0.01)

        for m in self.freq_head_global.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        X = batch['X']
        inputs = batch['Input_funcs']
        mask = batch.get('Mask', None)
        B, N, _ = X.shape

        # Random Fourier Features for coordinates — used both for query embedding and FFN router
        x_rff = self.spatial_encoder(X)
        pos_enhanced = x_rff
        enhanced_inputs = torch.cat([inputs, X], dim=-1)

        # 2. Embedding Layers — fully mode-agnostic (no mode_idx input anymore)
        x_emb = self.query_encoder(x_rff)
        y_emb = self.input_func_encoder(enhanced_inputs)

        condition_emb = y_emb
        condition_mask = mask if mask is not None else None

        # Global Context Extraction
        global_context = self.pooler(condition_emb, condition_mask)

        # `mode_idx` argument of GNOTBlock is kept for signature compatibility
        # but is unused inside the block; pass a zero placeholder.
        mode_placeholder = torch.zeros(B, dtype=torch.long, device=X.device)

        # --- Frequency-first decoding -------------------------------------
        # Predict all K eigenfrequencies from the global geometry context,
        # then always sort them ascending so the slot order is canonical.
        c = self.pooler(x_emb, condition_mask)            # [B, D] global context
        f_pred = self.freq_head_global(c)                 # [B, K]
        f_pred = torch.sort(f_pred, dim=-1).values        # always sorted ascending

        # --- Per-slot decoder: each mode owns its GNOT blocks + field head ---
        # All modes start from the same raw x_emb (shared encoder output).
        # Per-mode blocks handle cross-attention with geometry independently.
        # No DeepONet factorization: the field is regressed directly from the
        # per-mode node representation by field_heads[k] (pointwise MLP).
        fields = []
        for k in range(self.num_field_modes):
            # Slot query conditioned on this slot's predicted frequency.
            f_k = f_pred[:, k:k + 1]                        # [B, 1]
            q_k = self.slot_queries[k].unsqueeze(0) + self.freq_embed(f_k)  # [B, D]

            # Inject slot/frequency bias and run mode-specific attention blocks.
            x_m = x_emb + q_k.unsqueeze(1)                  # [B, N, D]

            for m_block in self.mode_field_blocks[k]:
                if self.use_checkpoint and self.training:
                    x_m = torch.utils.checkpoint.checkpoint(
                        m_block, x_m, condition_emb, mode_placeholder, pos_enhanced,
                        mask, condition_mask, global_context, use_reentrant=False)
                else:
                    x_m = m_block(x_m, condition_emb, mode_placeholder, pos_enhanced,
                                  mask, condition_mask, global_context)

            # Direct field head: [B, N, D] -> [B, N, 1]. The head starts with
            # its own LayerNorm, so no separate final_ln is needed.
            field_k = self.field_heads[k](x_m)                # [B, N, 1]
            fields.append(field_k)

        field = torch.cat(fields, dim=-1).float()             # [B, N, K]
        if mask is not None:
            field = field * mask.unsqueeze(-1)                # padding = 0

        # Optional Gram-Schmidt orthonormalization on field outputs
        if self.orthonormalize_output:
            field = _apply_gram_schmidt(field, mask)

        return {'field': field, 'freq': f_pred}              # [B,N,K], [B,K]

    def reset_expert_calls(self):
        for m in self.modules():
            if hasattr(m, '_expert_calls'):
                m._expert_calls.zero_()

    def get_expert_calls_per_block(self):
        calls_dict = {}
        for name, m in self.named_modules():
            if hasattr(m, '_expert_calls'):
                # Format name safely to use as TensorBoard tag (e.g. "shared_blocks.0" -> "shared_blocks_0")
                safe_name = name.replace('.', '_')
                calls_dict[safe_name] = m._expert_calls.clone()
        return calls_dict


# Public alias — the model takes (geometry) and predicts all K modes at once.
GNOT = GNOTModel
