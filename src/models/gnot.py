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

        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-1)

        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        z = torch.einsum('bhnd,bhde->bhne', q, kv)

        k_sum = k.sum(dim=2, keepdim=True)
        z_norm = torch.einsum('bhnd,bhmd->bhn', q, k_sum).unsqueeze(-1)

        output = z / (z_norm + self.eps)
        output = output.permute(0, 2, 1, 3).contiguous().reshape(b, n_q, d)

        return self.out_proj(output)

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
        gate_weights = F.softmax(gate_logits, dim=-1)

        final_output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            weight = gate_weights[:, :, i].unsqueeze(-1)
            final_output += weight * expert_out
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
        attn_out = self.cross_attn(query=x, key=condition_emb, value=condition_emb)
        x = self.norm1(x + attn_out)
        attn_out = self.self_attn(query=x, key=x, value=x)
        x = self.norm2(x + attn_out)
        ffn_out = self.ffn(x, coords)
        x = self.norm3(x + ffn_out)
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
    def __init__(self, val_dim=6, grid_dim=2, theta_dim=1, embed_dim=128, n_layers=4, n_heads=4, num_experts=4):
        super().__init__()
        self.query_encoder = MLPEncoder(grid_dim, embed_dim)
        self.input_func_encoder = MLPEncoder(val_dim, embed_dim)
        self.theta_encoder = MLPEncoder(theta_dim, embed_dim)

        self.blocks = nn.ModuleList([
            GNOTBlock(embed_dim, n_heads, grid_dim, num_experts)
            for _ in range(n_layers)
        ])

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

        for block in self.blocks:
            x_emb = block(x_emb, condition_emb, X)

        field_pred = self.field_decoder(x_emb)
        global_feat = x_emb.mean(dim=1)
        freq_pred = self.freq_decoder(global_feat)

        return {'field': field_pred, 'freq': freq_pred}
