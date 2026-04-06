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
    def __init__(self, embed_dim, num_heads, coords_dim=2, num_experts=4, dropout=0.0, num_modes=20):
        super().__init__()
        self.cross_attn = LinearAttention(embed_dim, num_heads, dropout)
        self.norm1_q  = nn.LayerNorm(embed_dim)
        self.norm1_kv = nn.LayerNorm(embed_dim)
        self.self_attn = LinearAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = GeometricGatingFFN(embed_dim, coords_dim, num_experts, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        # Per-block FiLM: her blok sonunda mode sinyalini yenile.
        # DiT (Diffusion Transformer) yaklaşımı — mode bilgisi katmanlar boyunca erimez.
        self.block_film = FiLMConditioner(embed_dim, num_modes)

    def forward(self, x, condition_emb, coords, mode_idx, mask=None, condition_mask=None):
        attn_out = self.cross_attn(
            query=self.norm1_q(x),
            key=self.norm1_kv(condition_emb),
            value=self.norm1_kv(condition_emb),
            mask=condition_mask
        )
        x = x + attn_out

        attn_out = self.self_attn(
            query=self.norm2(x),
            key=self.norm2(x),
            value=self.norm2(x),
            mask=mask
        )
        x = x + attn_out
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        ffn_out = self.ffn(self.norm3(x), coords)
        x = x + ffn_out

        # Mode conditioning: her blok sonunda mode sinyalini tazele
        x = self.block_film(x, mode_idx)

        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x

class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation: mode embedding'i LayerNorm-proof şekilde uygular.
    
    Additive injection (+) yerine affine transform (gamma * x + beta) kullanır.
    LayerNorm mean/variance'ı sıfırladığında additive bias kaybolur,
    ama multiplicative gamma sinyali korunur.
    """
    def __init__(self, embed_dim, num_modes=20):
        super().__init__()
        # Her mod için ayrı gamma ve beta üretiyor (2 * embed_dim)
        self.emb = nn.Embedding(num_modes, embed_dim * 2)
        # FIX: zeros yerine küçük random init — modlar baştan birbirinden ayrışıyor.
        # zeros ile tüm modlar aynı başlangıç noktasından geldiği için model Mode 0'a kilitleniyor.
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, x, mode_idx):
        # mode_idx: [B] veya [B, 1] — her ikisini de destekle
        if mode_idx.dim() > 1:
            mode_idx = mode_idx.squeeze(-1)          # [B]
        params = self.emb(mode_idx)                  # [B, 2*D]
        gamma, beta = params.chunk(2, dim=-1)        # her biri [B, D]
        gamma = gamma.unsqueeze(1)                   # [B, 1, D] — broadcast over nodes
        beta  = beta.unsqueeze(1)
        # 1 + gamma: başlangıçta identity (gamma=0 init)
        return x * (1.0 + gamma) + beta

class RandomFourierFeatures(nn.Module):
    def __init__(self, in_dim, out_dim, scale=1.0):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim for RandomFourierFeatures must be even."
        # Fixed random frequencies 
        self.B = nn.Parameter(torch.randn(in_dim, out_dim // 2) * scale, requires_grad=False)
        
    def forward(self, x):
        # x shape: [B, N, in_dim]
        # x_proj shape: [B, N, out_dim // 2]
        x_proj = 2 * torch.pi * (x @ self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

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
    def __init__(self, val_dim=6, grid_dim=2, theta_dim=1, embed_dim=128, n_layers=6, n_heads=4, num_experts=4, num_field_modes=3, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_field_modes = num_field_modes

        # --- Random Fourier Features for high spatial frequency encoding ---
        self.rff_dim = 64
        self.rff = RandomFourierFeatures(in_dim=grid_dim, out_dim=self.rff_dim, scale=1.0)

        # Query points (represented in raw space + Fourier space)
        self.query_encoder = MLPEncoder(grid_dim + self.rff_dim, embed_dim)
        # Input features + Query point Raw Context + Query point RFF context
        self.input_func_encoder = MLPEncoder(val_dim + grid_dim + self.rff_dim, embed_dim)
        
        # FiLM conditioner — giriş seviyesinde mode modülasyonu
        self.film_query    = FiLMConditioner(embed_dim, num_modes=20)  # encoder girişi
        self.film_cond     = FiLMConditioner(embed_dim, num_modes=20)  # condition girişi

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
        
        # Mode-specific field blocks: her mod kendi GNOTBlock'ından geçiyor.
        # Shared field_blocks yerine — gradient izolasyonu blok seviyesinde.
        self.mode_field_blocks = nn.ModuleList([
            GNOTBlock(embed_dim, n_heads, grid_dim, num_experts)
            for _ in range(num_field_modes)
        ])
        
        self.freq_blocks = nn.ModuleList([
            GNOTBlock(embed_dim, n_heads, grid_dim, num_experts)
            for _ in range(field_freq_layers)
        ])

        self.pooler = AttentionPool(embed_dim, n_heads)

        # Mode-specific field heads: her mod kendi decoder'ından geçiyor.
        # Gradient çakışması yok — Mode 0'ın gradyanı Head 1'e dokunmuyor.
        self.field_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1)
            ) for _ in range(num_field_modes)
        ])
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
        mask = batch.get('Mask', None)

        # Inject Geometric Fourier features alongside Raw Grid
        x_fourier = self.rff(X)
        x_enhanced = torch.cat([X, x_fourier], dim=-1)
        x_emb = self.query_encoder(x_enhanced)
        
        # Give functional inputs local RFF and explicit coordinates awareness
        enhanced_inputs = torch.cat([inputs, X, x_fourier], dim=-1)
        y_emb = self.input_func_encoder(enhanced_inputs)
        
        # FIX 2: FiLM ile mode conditioning — LayerNorm mode sinyalini silemiyor
        theta_int = theta.squeeze(-1)  # [B, 1] -> [B]
        x_emb = self.film_query(x_emb, theta_int)   # affine: x * (1 + gamma) + beta
        y_emb = self.film_cond(y_emb,  theta_int)

        # The condition targets are simply the y_emb now
        condition_emb = y_emb

        if mask is not None:
            # Mask remains the same shape as y_emb
            condition_mask = mask
        else:
            condition_mask = None

        # Shared processing with checkpointing option
        for block in self.shared_blocks:
            if self.use_checkpoint and self.training:
                x_emb = torch.utils.checkpoint.checkpoint(block, x_emb, condition_emb, X, theta_int, mask, condition_mask, use_reentrant=False)
            else:
                x_emb = block(x_emb, condition_emb, X, theta_int, mask, condition_mask)

        # Mode-specific field branch: her mod kendi blok + head'ından geçiyor
        B, N, D = x_emb.shape
        field_pred = torch.zeros(B, N, 1, device=x_emb.device, dtype=x_emb.dtype)

        # Freq branch: shared freq blocks (moddan bağımsız frekans tahmini)
        x_freq = x_emb
        for block in self.freq_blocks:
            if self.use_checkpoint and self.training:
                x_freq = torch.utils.checkpoint.checkpoint(block, x_freq, condition_emb, X, theta_int, mask, condition_mask, use_reentrant=False)
            else:
                x_freq = block(x_freq, condition_emb, X, theta_int, mask, condition_mask)
        for mode_val in range(self.num_field_modes):
            mode_mask = (theta_int == mode_val)  # [B] boolean
            if mode_mask.any():
                # Bu modun sample'ını seç
                x_mode = x_emb[mode_mask]
                cond_mode = condition_emb[mode_mask]
                X_mode = X[mode_mask]
                mask_mode = mask[mode_mask] if mask is not None else None
                cmask_mode = condition_mask[mode_mask] if condition_mask is not None else None
                theta_mode = theta_int[mode_mask]
                
                # Mode-specific GNOTBlock
                x_mode = self.mode_field_blocks[mode_val](
                    x_mode, cond_mode, X_mode, theta_mode, mask_mode, cmask_mode
                )
                # Mode-specific head
                field_pred[mode_mask] = self.field_heads[mode_val](x_mode)

        global_feat = self.pooler(x_freq, mask)
        freq_pred = self.freq_decoder(global_feat)

        return {'field': field_pred, 'freq': freq_pred}
