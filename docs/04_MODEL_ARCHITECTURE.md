# 04 — Model Mimarisi (GNOT)

> **Dosya:** `src/models/gnot.py`  
> **Model:** `GNOTModel`

---

## 🎯 Genel Bakış

GNOT, **neural operator** ailesinden bir Transformer modelidir. Klasik NN'lerden farkı, **fonksiyondan fonksiyona** bir eşleme öğrenmesidir:

$$\mathcal{G}_\theta: (\text{Geometri}, \text{Mod İndeksi}) \mapsto \text{Alan Dağılımı } E(x, y)$$

Model iki çıktı üretir:
1. **Field Prediction:** Her node için alan değeri ($E_i$) → `[B, N, 1]`
2. **Frequency Prediction:** Rezonans frekansı ($f$) → `[B, 1]`

---

## 🏗️ Mimari Diyagramı

```
               ┌─────────────────────────────────┐
               │          GİRDİLER                │
               │  X: [B, N, 2]    (koordinatlar)  │
               │  Input_funcs: [B, N, 8] (features)│
               │  Theta: [B, 1]   (mod indeksi)   │
               └──────────┬───────────┬───────────┘
                          │           │
               ┌──────────▼──┐  ┌─────▼──────────────────┐
               │ RFF(X)       │  │ [Input_funcs, X] concat │
               │ [B, N, 64]   │  │       [B, N, 10]        │
               └──────────┬──┘  └─────┬──────────────────┘
                          │            │
               ┌──────────▼────────────▼──────────┐
               │  query_encoder(RFF) → x_emb       │
               │  input_func_encoder([IF,X]) → y_emb│
               │         → [B, N, 256]              │
               └──────────────────┬────────────────┘
                                  │
               ┌──────────────────▼──────────────┐
               │  + mode_emb_entrance(theta)      │  ← Mod enjeksiyonu
               │    (her ikisine de eklenir)       │
               └──────────────────┬──────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │  AttentionPool(y_emb)    │
                     │  → global_context [B,256] │
                     └────────────┬─────────────┘
                                  │
               ┌──────────────────▼──────────────────┐
               │         SHARED TRUNK                 │
               │    6 × GNOTBlock (paylaşımlı)        │
               └──────────┬──────────────────────────┘
                          │
           ┌──────────────▼──────────────────┐
           │     MODE-SPECIFIC BRANCHES       │
           │                                  │
           │  Mode 0: 1×Block → field_head[0] → alan tahmini
           │                  → pool → freq_head[0] → frekans_0
           │                                  │
           │  Mode 1: 1×Block → field_head[1] → alan tahmini
           │                  → pool → freq_head[1] → frekans_1
           │                                  │
           │  Mode 2: 1×Block → field_head[2] → alan tahmini
           │                  → pool → freq_head[2] → frekans_2
           └──────────────────────────────────┘
```

---

## 🧩 Bileşenler — Detaylı Açıklamaları

### 1. Random Fourier Features (RFF)

```python
class RandomFourierFeatures:
    # B ~ N(0, 1/length_scale²) — sabit, öğrenilmez
    B = randn(2, 32) * (1.0 / length_scale)   # [2, 32]

    def forward(x):          # x: [B, N, 2]
        proj = x @ B         # [B, N, 32]
        scale = sqrt(2 / 64) # Bochner normalizasyonu
        return scale * [cos(proj), sin(proj)]  # [B, N, 64]
```

**Matematiksel arka plan (Rahimi & Recht, 2007):**

$$\phi(x) = \sqrt{\frac{2}{D}} \begin{bmatrix} \cos(Bx) \\ \sin(Bx) \end{bmatrix}, \quad B_{ij} \sim \mathcal{N}\!\left(0, \frac{1}{\ell^2}\right)$$

Bu encoding, Gaussian kernel'i yaklaşık olarak ifade eder:
$$k(x, y) \approx \phi(x)^\top \phi(y) \approx \exp\!\left(-\frac{\|x-y\|^2}{2\ell^2}\right)$$

`length_scale` ($\ell$) parametresi kernel bant genişliğini kontrol eder:
- Küçük `length_scale` (0.05) → yüksek frekans, ince detaylar
- Büyük `length_scale` (0.2) → düşük frekans, pürüzsüz

**Önemli:** `B` matrisi öğrenilmez (`register_buffer`). Bochner teoremi gereği, kernel yaklaşımının yansız olması için frekansların sabit tutulması gerekir. `sqrt(2/D)` faktörü, `k(x,x) = 1` normalizasyonunu sağlar.

**Config parametreleri:** `model.rff_dim` (çıktı boyutu, çift sayı olmalı), `model.rff_length_scale`

### 2. Linear Attention

```python
class LinearAttention:
    q = ELU(Wq @ query) + 1.0   # Kernel: φ(x) = elu(x) + 1
    k = ELU(Wk @ key) + 1.0
    v = Wv @ value

    # O(N²) yerine O(N·d²) karmaşıklık:
    kv = einsum('bhnd,bhne->bhde', k, v)      # K^T V önce [d,d]
    output = einsum('bhnd,bhde->bhne', q, kv) # Q ile çarp
    output = output / normalization
```

| | Standart Softmax | Linear (ELU+1) |
|---|---|---|
| Karmaşıklık | $O(N^2 d)$ | $O(N d^2)$ |
| 1024 node | 1M işlem | 67K işlem |
| Bellek | $O(N^2)$ | $O(Nd)$ |

RF kavitelerinde mesh boyutları 500–2000 node arası. Linear attention bu boyutlarda hem hızlı hem de bellek dostu.

**ELU+1 kernel:** Negatif değerleri yumuşatır ama sıfırlamaz. `+1` tüm değerlerin pozitif olmasını sağlar — kernel trick'in geçerli olması için gereklidir.

### 3. GNOTBlock (Temel Yapı Taşı)

Her blokta sırasıyla şunlar devreye girer:

```
Input x
   │
   ├──▶ CrossAttention(Q=x, KV=condition_emb)  → "Geometri bilgisini sorgula"
   │    + Residual
   │
   ├──▶ + global_context.unsqueeze(1)           → "Kavitenin büyük resmini enjekte et"
   │
   ├──▶ SelfAttention(Q=K=V=x)                  → "Node'lar birbirleriyle konuşsun"
   │    + Residual + Mask
   │
   └──▶ GeometricGatingFFN(x, gate_from_pos)    → "Bölgeye özel dönüşüm"
        + Residual + Mask
```

**Global context injection:** Her blokta `AttentionPool(condition_emb)` sonucu `[B, D]` olarak her node'a eklenir. Bu, dipol modların asimetrisini anlamak için kritik — kavitenin "büyük resmi" tüm noktalara aşılanır.

### 4. GeometricGatingFFN (Mixture of Experts)

```python
class GeometricGatingFFN:
    experts = [FFN_0, FFN_1, FFN_2, FFN_3]   # 4 uzman (256→1024→256)

    def forward(x, pos):
        gate_logits = local_gating(pos)        # RFF → gate logits [B, N, 4]
        gate_weights = softmax(logits / 0.5)   # Temperature=0.5 → biraz keskin

        # DENSE: Tüm expertler çalışır, ağırlıklı toplanır
        output = Σ_i  gate_weights[i] * experts[i](x)
```

**Neden dense (hepsi çalışır)?**
Physics field tahminlerinde **uzamsal süreklilik** kritiktir. Top-k (keskin seçim) komşu node'lar arasında görsel artifact yaratır — `(x=0.5, y=0.3)` bir expert'ten, `(x=0.51, y=0.3)` başka bir expert'ten gelebilir. Dense weighted sum bu süreksizliği önler.

**Temperature=0.5:** Softmax'ı biraz keskinleştirir (bir expert daha baskın hale gelir) ama sıfırlamaz. Geçişler hâlâ pürüzsüz.

**Fiziksel motivasyon:** Kavite içinde farklı bölgeler farklı fizik sergiler:
- Sınır yakını: hızlı düşen alan → keskin gradyan
- Merkez: düzgün dağılım
- Köşeler: singularity-benzeri davranış

Her expert farklı bir bölge tipine uzmanlaşır.

### 5. AttentionPool (Learned Global Pooling)

```python
class AttentionPool:
    query = nn.Parameter(randn(1, 1, D))   # Öğrenilebilir özet token
    attn = MultiheadAttention(D, heads)

    def forward(x, mask):
        out = attn(query, x, x, key_padding_mask=~mask)
        return out.squeeze(1)  # [B, D]
```

Tüm node bilgisini tek bir vektöre sıkıştırır. Standart `mean pooling`'den farklı olarak bazı node'lara daha fazla "dikkat" verebilir.

**Kullanım yerleri:**
1. Her GNOTBlock'ta `global_context` üretmek için (`condition_emb`'den pool)
2. Frekans tahmini için mode-specific global özet (`x_m`'den pool → freq_head)

### 6. Mode Conditioning

Mode indeksi modele iki kanaldan girer:

```python
# Giriş (entrance) enjeksiyonu — hem query hem KV'ye eklenir
m_emb_init = mode_emb_entrance(mode_indices).unsqueeze(1)  # [B, 1, D]
x_emb = x_emb + m_emb_init
y_emb = y_emb + m_emb_init
```

`mode_emb_entrance`: `nn.Embedding(num_field_modes=3, embed_dim)`, `std=0.02` ile başlatılmış.

**Neden addition?** LayerNorm sonrası addition, FiLM'e kıyasla daha sade ama bu mimaride yeterli — embedding vektörü trunk boyunca gradyan ile optimize edilir.

---

## 🔀 Mode-Specific Branching

Forward pass'te model sample'ları mod maskesiyle ayırır ve her modu kendi branch'inden geçirir:

```python
for mode_val in range(num_field_modes):
    mode_mask = (mode_indices == mode_val)   # [B] bool
    if not mode_mask.any():
        continue
    x_m = x_emb[mode_mask]                  # [B_m, N, D]
    for block in mode_field_blocks[mode_val]:
        x_m = block(x_m, ...)
    field_pred[mode_mask] = field_heads[mode_val](x_m)
    if freq_heads:
        freq_pred[mode_mask] = freq_heads[mode_val](pool(x_m))
```

**Neden gerekli?** Mode 0 (monopol, radyal simetrik) ve Mode 1 (dipol, asimetrik) çok farklı fizik. Aynı parametreler ikisini öğrenmeye çalışırsa gradyan çakışması olur. Mode-specific branching ile her modun gradyanı sadece kendi parametrelerine etki eder.

---

## 📊 Parametre Sayısı Tahmini

(`embed_dim=256`, `n_shared=6`, `n_mode=1`, `n_field_head_layers=3`, `num_experts=4`)

| Bileşen | Yaklaşık Parametre |
|---------|-------------------|
| RFF (B matrisi, frozen) | ~128 (öğrenilmez) |
| query_encoder (64→256) | ~50K |
| input_func_encoder (10→256) | ~70K |
| mode_emb_entrance (3×256) | ~800 |
| Shared blocks (6 × GNOTBlock) | ~6 × 2.2M = 13.2M |
| Mode field blocks (3×1×) | ~3 × 2.2M = 6.6M |
| Field heads (3×, 3-layer) | ~3 × 400K = 1.2M |
| Freq heads (3×) | ~3 × 130K = 390K |
| AttentionPool | ~200K |
| **TOPLAM** | **~22M parametre** |

> GNOTBlock başına: 2×LinearAttn (~0.5M) + GeometricGatingFFN (4 uzman×(256×1024+1024×256) = 2M) + gate_network (64→128→4, ~35K) + LayerNorms ≈ 2.2M

---

## ⚠️ Geliştirme Önerileri

### Mimari
1. **Soft Top-2 Experts:** Mevcut dense MoE yerine top-2 + renormalize yaklaşımı RAM'i yarıya indirir. Dipol modlardaki spatial continuity için yeterince smooth.
2. **Flash Attention:** Linear attention yerine `flash_attn` kütüphanesi denenebilir. Küçük-orta mesh boyutlarında daha iyi kalite verebilir.
3. **Deeper Shared Trunk:** `n_shared_layers=6` → 8'e çıkarılabilir. Geometrik temsil güçlenir.
4. **Learnable RFF:** `B` matrisi `nn.Parameter` yapılabilir. Frekans spektrumuna adaptif hale gelir ama Bochner garantisini kaybeder.

### Fizik-Bilinçli Geliştirmeler
5. **Helmholtz Residual Loss:** `∇²E + k²E = 0` denkleminin residualı ek loss terimi olarak.
6. **Sobolev Loss:** Alan yanı sıra gradyan `∇E` de hedeflenirse fiziksel pürüzsüzlük zorunlu hale gelir.

---

## 🔗 Bağlantılar

- Bu modeli besleyen veri: [[03_DATASET_LOADER]]
- Eğitim detayları: [[05_TRAINING_SYSTEM]]
- Config'te mimari parametreleri: [[08_CONFIG_REFERENCE]]
- Fizik arka planı: [[09_PHYSICS_BACKGROUND]]

#model #transformer #attention #moe #rff #gnot
