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

    def forward(self, x, gate_info):
        # gate_info: (top2_weights, top2_idx)
        top2_weights, top2_idx = gate_info

        # Pre-compute all expert outputs (no CPU syncs)
        # Since num_experts is small (4), full dense evaluation is faster than D2H halting.
        expert_outputs = [expert(x) for expert in self.experts]

        final_output = torch.zeros_like(x)
        for k in range(2):
            idx = top2_idx[..., k]   # [B, N]
            w   = top2_weights[..., k].unsqueeze(-1)  # [B, N, 1]
            for i in range(len(self.experts)):
                mask = (idx == i).float().unsqueeze(-1)  # [B, N, 1]
                # Multiply directly without using .any() to avoid GPU-CPU sync blocks
                final_output = final_output + w * mask * expert_outputs[i]
                
        return final_output

class GNOTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, coords_dim, num_experts=4, dropout=0.0, num_modes=20, use_film=True):
        super().__init__()
        self.use_film = use_film
        self.cross_attn = LinearAttention(embed_dim, num_heads, dropout)
        self.norm1_q  = nn.LayerNorm(embed_dim)
        self.norm1_kv = nn.LayerNorm(embed_dim)
        self.self_attn = LinearAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = GeometricGatingFFN(embed_dim, num_experts, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        # Per-block FiLM: her blok sonunda mode sinyalini yenile.
        # Mode-specific bloklar için zorunlu, shared bloklar için mode-blind olmalı.
        if use_film:
            self.block_film = FiLMConditioner(embed_dim, num_modes)
        else:
            self.block_film = None

        self.local_gating = nn.Sequential(
            nn.Linear(coords_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_experts)
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

        # Local Sparse Gating Logic
        gate_logits = self.local_gating(pos)
        gate_weights = F.softmax(gate_logits, dim=-1)
        top2_weights, top2_idx = gate_weights.topk(2, dim=-1)
        top2_weights = top2_weights / (top2_weights.sum(dim=-1, keepdim=True) + 1e-8)
        gate_info = (top2_weights, top2_idx)

        # Track usage during validation
        if not self.training:
            with torch.no_grad():
                counts = torch.bincount(top2_idx.flatten(), minlength=gate_logits.shape[-1])
                self._expert_calls += counts

        ffn_out = self.ffn(self.norm3(x), gate_info)
        x = x + ffn_out

        # Mode conditioning: sadece fizik öğrenen şubelerde aktif
        if self.block_film is not None:
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
    def __init__(self, val_dim=6, grid_dim=2, theta_dim=1, embed_dim=128, 
                 n_shared_layers=2, n_mode_layers=2, n_field_head_layers=2,
                 n_heads=4, num_experts=4, num_field_modes=3,
                 use_checkpoint=False, predict_frequency=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_field_modes = num_field_modes
        # Query points (represented in raw space)
        self.query_encoder = MLPEncoder(grid_dim, embed_dim)
        # Input features + Query point Raw Context
        self.input_func_encoder = MLPEncoder(val_dim + grid_dim, embed_dim)
        
        # Entrance Mode Embedding: Query'yi (X) ve Condition'ı (Y) en başta mode-aware yapar.
        self.mode_emb_entrance = nn.Embedding(num_field_modes, embed_dim)
        nn.init.normal_(self.mode_emb_entrance.weight, mean=0.0, std=0.02)

        # Minimal architecture: Shared -> Mode-Specific (Deep)
        shared_layers = n_shared_layers
        mode_layers   = n_mode_layers  # Her modun özel fizik derinliği
        
        # Bloc-specific coordinate dimension: always raw (x, y)
        block_coords_dim = grid_dim

        self.shared_blocks = nn.ModuleList([
            GNOTBlock(embed_dim, n_heads, coords_dim=block_coords_dim, num_experts=num_experts, use_film=True)
            for _ in range(shared_layers)
        ])
        
        # Mode-specific field branches
        self.mode_field_blocks = nn.ModuleList([
            nn.ModuleList([
                 GNOTBlock(embed_dim, n_heads, coords_dim=block_coords_dim, num_experts=num_experts, use_film=True)
                 for _ in range(mode_layers)
            ]) for _ in range(num_field_modes)
        ])
        
        self.pooler = AttentionPool(embed_dim, n_heads)

        # Dynamic Mode-specific field heads
        self.field_heads = nn.ModuleList()
        for _ in range(num_field_modes):
            layers = []
            curr_dim = embed_dim
            
            # If depth > 2, expand first
            if n_field_head_layers > 2:
                layers.extend([
                    nn.Linear(curr_dim, embed_dim * 2),
                    nn.LayerNorm(embed_dim * 2),
                    nn.GELU()
                ])
                curr_dim = embed_dim * 2
                
                # Intermediate layers
                for _ in range(n_field_head_layers - 3):
                    layers.extend([
                        nn.Linear(curr_dim, curr_dim),
                        nn.LayerNorm(curr_dim),
                        nn.GELU()
                    ])
            
            # Penultimate layer: shrink back to embed_dim
            if n_field_head_layers >= 2:
                layers.extend([
                    nn.Linear(curr_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU()
                ])
                curr_dim = embed_dim
            
            # Final output layer
            layers.append(nn.Linear(curr_dim, 1))
            self.field_heads.append(nn.Sequential(*layers))

        # Per-mode frequency heads: her mod kendi frekansını tahmin eder.
        # Fiziksel olarak doğru — her eigenmode'un kendi rezonans frekansı vardır.
        if predict_frequency:
            self.freq_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, 1)
                ) for _ in range(num_field_modes)
            ])
        else:
            self.freq_heads = None

        # Initialize heads with small weights to prevent early training explosion
        self._init_weights()

    def _init_weights(self):
        for m in self.field_heads.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if self.freq_heads is not None:
            for m in self.freq_heads.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        X = batch['X']
        inputs = batch['Input_funcs']
        theta_in = batch['Theta_in']  # [B, 1] — mode index per sample
        mask = batch.get('Mask', None)
        B = X.shape[0]

        # Use Raw Grid
        x_enhanced = X
        enhanced_inputs = torch.cat([inputs, X], dim=-1)

        x_emb = self.query_encoder(x_enhanced)
        y_emb = self.input_func_encoder(enhanced_inputs)

        # Entrance Mode Injection: Embedding'leri daha en başta mod bilgisiyle harmanlıyoruz.
        mode_indices = theta_in[:, 0]  # [B]
        m_emb_init = self.mode_emb_entrance(mode_indices).unsqueeze(1) # [B, 1, D]
        x_emb = x_emb + m_emb_init
        y_emb = y_emb + m_emb_init
        
        condition_emb = y_emb
        condition_mask = mask if mask is not None else None
        
        # Spatial features for gating
        x_f_pass = X

        # Global Context Extraction
        global_context = self.pooler(condition_emb, condition_mask)

        # Trunk (Shared processing) — mode-aware
        for block in self.shared_blocks:
            if self.use_checkpoint and self.training:
                x_emb = torch.utils.checkpoint.checkpoint(block, x_emb, condition_emb, mode_indices, x_f_pass, mask, condition_mask, global_context, use_reentrant=False)
            else:
                x_emb = block(x_emb, condition_emb, mode_indices, x_f_pass, mask, condition_mask, global_context)

        # --- Dynamic Routing: route each sample to its own mode branch ---
        
        # Initialize output tensors
        N = X.shape[1]
        field_pred = torch.zeros(B, N, 1, device=X.device)
        freq_pred = torch.zeros(B, 1, device=X.device) if self.freq_heads is not None else None
        
        for mode_val in range(self.num_field_modes):
            # Find which samples in this batch belong to this mode
            mode_mask = (mode_indices == mode_val)  # [B] bool
            if not mode_mask.any():
                continue
            
            # Extract subset
            x_m = x_emb[mode_mask]          # [B_m, N, D]
            c_m = condition_emb[mode_mask]   # [B_m, N, D]
            pos_m = x_f_pass[mode_mask]      # [B_m, N, pos_dim]
            m_mask = mask[mode_mask] if mask is not None else None
            c_mask = condition_mask[mode_mask] if condition_mask is not None else None
            gc_m = global_context[mode_mask]  # [B_m, D]
            B_m = x_m.shape[0]
            th_m = torch.full((B_m,), mode_val, dtype=torch.long, device=X.device)
            
            for m_block in self.mode_field_blocks[mode_val]:
                if self.use_checkpoint and self.training:
                    x_m = torch.utils.checkpoint.checkpoint(m_block, x_m, c_m, th_m, pos_m, m_mask, c_mask, gc_m, use_reentrant=False)
                else:
                    x_m = m_block(x_m, c_m, th_m, pos_m, m_mask, c_mask, gc_m)
            
            # Field prediction
            field_pred[mode_mask] = self.field_heads[mode_val](x_m)  # [B_m, N, 1]

            # Frequency prediction
            if self.freq_heads is not None:
                mode_global = self.pooler(x_m, m_mask)
                freq_pred[mode_mask] = self.freq_heads[mode_val](mode_global)  # [B_m, 1]

        return {'field': field_pred, 'freq': freq_pred}

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
