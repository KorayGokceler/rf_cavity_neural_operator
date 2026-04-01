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

    def forward(self, query, key, value):
        b, n_q, d = query.shape
        b, n_k, _ = key.shape

        q = self.q_proj(query).view(b, n_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).view(b, n_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).view(b, n_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Use ELU+1 kernel for more stable linear attention
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

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

    def forward(self, x):
        # x: [B, N, D]
        b = x.shape[0]
        q = self.query.expand(b, -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out).squeeze(1)

class GeometricGatingFFN(nn.Module):
    def __init__(self, embed_dim, coords_dim, num_experts=4, dropout=0.1):
        super().__init__()
        self.gating_net = nn.Sequential(
            nn.Linear(coords_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_experts)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x, coords):
        gate_logits = self.gating_net(coords)
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, N, num_experts]

        # Top-2 sparse routing: only compute the 2 highest-weight experts
        top2_weights, top2_idx = gate_weights.topk(2, dim=-1)  # [B, N, 2]
        # Re-normalize so the two weights sum to 1
        top2_weights = top2_weights / (top2_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Find which experts are actually needed (avoid computing unused ones)
        active_experts = set(top2_idx.unique().tolist())

        # Pre-compute only active expert outputs
        expert_outputs = {}
        for i in active_experts:
            expert_outputs[i] = self.experts[i](x)

        final_output = torch.zeros_like(x)
        for k in range(2):
            idx = top2_idx[..., k]   # [B, N] — which expert each token picks
            w   = top2_weights[..., k].unsqueeze(-1)  # [B, N, 1]
            for i in active_experts:
                mask = (idx == i).float().unsqueeze(-1)  # [B, N, 1]
                if mask.any():
                    final_output = final_output + w * mask * expert_outputs[i]
        return final_output

class GNOTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, coords_dim=2, num_experts=4, dropout=0.0):
        super().__init__()
        self.cross_attn = LinearAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = LinearAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = GeometricGatingFFN(embed_dim, coords_dim, num_experts, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, condition_emb, coords):
        # Pre-norm: normalize inputs BEFORE attention/FFN (more stable for deep nets)
        attn_out = self.cross_attn(
            query=self.norm1(x),
            key=self.norm1(condition_emb),
            value=self.norm1(condition_emb)
        )
        x = x + attn_out

        attn_out = self.self_attn(
            query=self.norm2(x),
            key=self.norm2(x),
            value=self.norm2(x)
        )
        x = x + attn_out

        ffn_out = self.ffn(self.norm3(x), coords)
        x = x + ffn_out
        return x

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

class GNOTModel(nn.Module):
    def __init__(self, val_dim=6, grid_dim=2, theta_dim=1, embed_dim=128, n_layers=6, n_heads=4, num_experts=4, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.query_encoder = MLPEncoder(grid_dim, embed_dim)
        self.input_func_encoder = MLPEncoder(val_dim, embed_dim)
        self.theta_encoder = MLPEncoder(theta_dim, embed_dim)

        # Architecture division into Shared -> [Field Branch, Freq Branch]
        # Total n_layers is distributed as: shared, task_field, task_freq.
        # Minimal set: 1 shared, 1 field, 1 freq.
        shared_layers = max(1, n_layers // 3)
        remaining = n_layers - shared_layers
        field_field_layers = max(1, remaining // 2)
        field_freq_layers = max(1, remaining - field_field_layers)
        
        self.shared_blocks = nn.ModuleList([
            GNOTBlock(embed_dim, n_heads, grid_dim, num_experts)
            for _ in range(shared_layers)
        ])
        
        self.field_blocks = nn.ModuleList([
            GNOTBlock(embed_dim, n_heads, grid_dim, num_experts)
            for _ in range(field_field_layers)
        ])
        
        self.freq_blocks = nn.ModuleList([
            GNOTBlock(embed_dim, n_heads, grid_dim, num_experts)
            for _ in range(field_freq_layers)
        ])

        self.pooler = AttentionPool(embed_dim, n_heads)

        self.field_decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )
        self.freq_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, batch):
        X = batch['X']
        inputs = batch['Input_funcs']
        theta = batch['Theta_in']

        x_emb = self.query_encoder(X)
        y_emb = self.input_func_encoder(inputs)
        theta_emb = self.theta_encoder(theta).unsqueeze(1)

        condition_emb = torch.cat([y_emb, theta_emb], dim=1)

        # Shared processing with checkpointing option
        for block in self.shared_blocks:
            if self.use_checkpoint and self.training:
                x_emb = torch.utils.checkpoint.checkpoint(block, x_emb, condition_emb, X, use_reentrant=False)
            else:
                x_emb = block(x_emb, condition_emb, X)

        # Task-specific branching
        x_field = x_emb
        for block in self.field_blocks:
            if self.use_checkpoint and self.training:
                x_field = torch.utils.checkpoint.checkpoint(block, x_field, condition_emb, X, use_reentrant=False)
            else:
                x_field = block(x_field, condition_emb, X)
            
        x_freq = x_emb
        for block in self.freq_blocks:
            if self.use_checkpoint and self.training:
                x_freq = torch.utils.checkpoint.checkpoint(block, x_freq, condition_emb, X, use_reentrant=False)
            else:
                x_freq = block(x_freq, condition_emb, X)

        field_pred = self.field_decoder(x_field)
        global_feat = self.pooler(x_freq)
        freq_pred = self.freq_decoder(global_feat)

        return {'field': field_pred, 'freq': freq_pred}
