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
    def __init__(self, val_dim=6, grid_dim=2, theta_dim=1, embed_dim=128, 
                 n_shared_layers=2, n_mode_layers=2, n_freq_layers=2,
                 n_heads=4, num_experts=4, num_field_modes=3, rff_scale=1.0, 
                 use_rff=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_field_modes = num_field_modes
        self.use_rff = use_rff

        # --- Random Fourier Features for high spatial frequency encoding ---
        if use_rff:
            self.rff_dim = 64
            self.rff = RandomFourierFeatures(in_dim=grid_dim, out_dim=self.rff_dim, scale=rff_scale)
        else:
            self.rff_dim = 0
            self.rff = None

        # Query points (represented in raw space + Fourier space)
        self.query_encoder = MLPEncoder(grid_dim + self.rff_dim, embed_dim)
        # Input features + Query point Raw Context + Query point RFF context
        self.input_func_encoder = MLPEncoder(val_dim + grid_dim + self.rff_dim, embed_dim)
        
        # NOTE: Entrance FiLM (film_query/film_cond) kaldırıldı.
        # Trunk'ı başlangıçta mode-blind olmaya zorluyoruz; sadece geometriyi temsil etmeli.

        # Minimal architecture: Shared -> [Mode-Specific (Deep), Freq (Deep)]
        shared_layers = n_shared_layers
        mode_layers   = n_mode_layers  # Her modun özel fizik derinliği
        freq_layers   = n_freq_layers
        
        # Bloc-specific coordinate dimension: RFF kapalıysa raw (x, y) kullanılır.
        block_coords_dim = self.rff_dim if use_rff else grid_dim

        self.shared_blocks = nn.ModuleList([
            GNOTBlock(embed_dim, n_heads, coords_dim=block_coords_dim, num_experts=num_experts, use_film=False)
            for _ in range(shared_layers)
        ])
        
        # Mode-specific field branches
        self.mode_field_blocks = nn.ModuleList([
            nn.ModuleList([
                 GNOTBlock(embed_dim, n_heads, coords_dim=block_coords_dim, num_experts=num_experts, use_film=True)
                 for _ in range(mode_layers)
            ]) for _ in range(num_field_modes)
        ])
        
        self.freq_blocks = nn.ModuleList([
            GNOTBlock(embed_dim, n_heads, coords_dim=block_coords_dim, num_experts=num_experts, use_film=True)
            for _ in range(freq_layers)
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

        # Initialize heads with small weights to prevent early training explosion
        self._init_weights()

    def _init_weights(self):
        for m in self.field_heads.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.freq_decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        X = batch['X']
        inputs = batch['Input_funcs']
        theta = batch['Theta_in']
        mask = batch.get('Mask', None)

        # Inject Geometric Fourier features alongside Raw Grid (Optional)
        if self.use_rff and self.rff is not None:
            x_fourier = self.rff(X)
            # Query embedding uses [Raw, RFF]
            x_enhanced = torch.cat([X, x_fourier], dim=-1)
            # Functional inputs use [Raw_Features, Raw_Coords, RFF]
            enhanced_inputs = torch.cat([inputs, X, x_fourier], dim=-1)
        else:
            x_fourier = None
            x_enhanced = X
            enhanced_inputs = torch.cat([inputs, X], dim=-1)

        x_emb = self.query_encoder(x_enhanced)
        y_emb = self.input_func_encoder(enhanced_inputs)
        
        # NOTE: Giriş seviyesinde mode conditioning (film_query/cond) kaldırıldı.
        # Shared trunk sadece geometriye (X, RFF) ve query contextine odaklanır.
        theta_int = theta.squeeze(-1)  # [B]
        
        # The condition targets are simply the y_emb now
        condition_emb = y_emb

        condition_mask = mask if mask is not None else None
        
        # If RFF is off, we still want blocks to have some spatial awareness for gating.
        # Use Raw X as the 'fourier' input to blocks if RFF is disabled.
        x_f_pass = x_fourier if x_fourier is not None else X

        # Global Context Extraction: Kavitenin "Büyük Resmini" bir kere çıkarıp tüm bloklara dağıtacağız.
        global_context = self.pooler(condition_emb, condition_mask)

        # Shared processing with checkpointing option
        for block in self.shared_blocks:
            if self.use_checkpoint and self.training:
                x_emb = torch.utils.checkpoint.checkpoint(block, x_emb, condition_emb, theta_int, x_f_pass, mask, condition_mask, global_context, use_reentrant=False)
            else:
                x_emb = block(x_emb, condition_emb, theta_int, x_f_pass, mask, condition_mask, global_context)

        # Freq branch
        x_freq = x_emb
        for block in self.freq_blocks:
            if self.use_checkpoint and self.training:
                x_freq = torch.utils.checkpoint.checkpoint(block, x_freq, condition_emb, theta_int, x_f_pass, mask, condition_mask, global_context, use_reentrant=False)
            else:
                x_freq = block(x_freq, condition_emb, theta_int, x_f_pass, mask, condition_mask, global_context)

        # Sample'ları mode'a göre sırala
        sort_idx = theta_int.argsort()
        unsort_idx = sort_idx.argsort()

        x_sorted = x_emb[sort_idx]
        cond_sorted = condition_emb[sort_idx]
        X_sorted = X[sort_idx]
        theta_sorted = theta_int[sort_idx]
        global_context_sorted = global_context[sort_idx]
        mask_sorted = mask[sort_idx] if mask is not None else None
        cmask_sorted = condition_mask[sort_idx] if condition_mask is not None else None
        
        # Koordinat bilgisini de (x_f_pass) sırala
        x_f_pass_sorted = x_f_pass[sort_idx]

        # Her modun kaç sample'ı var
        mode_counts = [(theta_int == m).sum().item() for m in range(self.num_field_modes)]

        # Sıralı işle ve sonuçları topla
        field_parts = []
        start = 0
        for mode_val in range(self.num_field_modes):
            count = mode_counts[mode_val]
            if count == 0:
                continue
            end = start + count
            x_m = x_sorted[start:end]
            c_m = cond_sorted[start:end]
            th_m = theta_sorted[start:end]
            mk_m = mask_sorted[start:end] if mask_sorted is not None else None
            cm_m = cmask_sorted[start:end] if cmask_sorted is not None else None
            g_m = global_context_sorted[start:end]
            
            # Bu modun koordinatlarını dilimle
            pos_m = x_f_pass_sorted[start:end]
            
            for m_block in self.mode_field_blocks[mode_val]:
                if self.use_checkpoint and self.training:
                    x_m = torch.utils.checkpoint.checkpoint(m_block, x_m, c_m, th_m, pos_m, mk_m, cm_m, g_m, use_reentrant=False)
                else:
                    x_m = m_block(x_m, c_m, th_m, pos_m, mk_m, cm_m, g_m)
            
            field_parts.append(self.field_heads[mode_val](x_m))
            start = end

        # Birleştir ve orijinal sıraya geri dön — NO in-place ops
        field_pred_sorted = torch.cat(field_parts, dim=0)  # [B, N, 1]
        field_pred = field_pred_sorted[unsort_idx]          # orijinal batch sırası

        global_feat = self.pooler(x_freq, mask)
        freq_pred = self.freq_decoder(global_feat)

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
