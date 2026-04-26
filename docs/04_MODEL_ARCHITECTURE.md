# 04 — Model Mimarisi (GNOT)

> **Dosya:** `src/models/gnot.py`  
> **Model:** `GNOTModel`  
> **Referans Paper:** GNOT (General Neural Operator Transformer) — `examples/2302.14376v3.pdf`

---

## 🎯 Genel Bakış

GNOT, **neural operator** ailesinden bir Transformer modelidir. Klasik NN'lerden farkı, **fonksiyondan fonksiyona** bir eşleme öğrenmesidir:

$$\mathcal{G}_\theta: (\text{Geometri}, \text{Mod İndeksi}) \mapsto \text{Alan Dağılımı } E(x, y)$$

Model bu projeyle 2 çıktı üretir:
1. **Field Prediction:** Her node için alan değeri ($E_i$) → `[B, N, 1]`
2. **Frequency Prediction:** Rezonans frekansı ($f$) → `[B, 1]`

---

## 🏗️ Mimari Diyagramı

```
               ┌─────────────────────────────────┐
               │          GİRDİLER                 │
               │                                   │
               │  X: [B, N, 2]    (koordinatlar)   │
               │  Input_funcs: [B, N, 8] (features)│
               │  Theta: [B, 1]   (mod indeksi)    │
               └──────────┬───────────┬────────────┘
                          │           │
                    ┌─────▼─────┐  ┌──▼──────────────┐
                    │ RFF(X)     │  │                  │
                    │ [B,N,64]   │  │                  │
                    └─────┬─────┘  │                  │
                          │        │                  │
               ┌──────────▼────────▼──────────────────┐
               │  query_encoder([X, RFF])              │
               │  input_func_encoder([IF, X, RFF])     │
               │         → x_emb, y_emb [B, N, 256]   │
               └──────────────────┬────────────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │  AttentionPool(y_emb)     │
                     │  → global_context [B,256] │
                     └────────────┬──────────────┘
                                  │
               ┌──────────────────▼──────────────────┐
               │         SHARED TRUNK                 │
               │    4 × GNOTBlock (mode-blind)        │
               │    FiLM: KAPALI                      │
               └──────────┬──────────────────────────┘
                          │
           ┌──────────────▼──────────────────┐
           │     MODE-SPECIFIC BRANCHES       │
           │                                  │
           │  Mode 0: 2×Block → field_head[0] → alan tahmini
           │                   → pool → freq_head[0] → frekans_0
           │                                  │
           │  Mode 1: 2×Block → field_head[1] → alan tahmini
           │                   → pool → freq_head[1] → frekans_1
           │                                  │
           │  Mode 2: 2×Block → field_head[2] → alan tahmini
           │                   → pool → freq_head[2] → frekans_2
           │                                  │
           └──────────────────────────────────┘
```

---

## 🧩 Bileşenler — Detaylı Açıklamaları

### 1. Random Fourier Features (RFF)

```python
class RandomFourierFeatures:
    B = randn(2, 32) * scale       # Sabit rastgele frekanslar (learnable DEĞİL)
    def forward(x):                 # x: [B, N, 2]
        proj = 2π * (x @ B)        # [B, N, 32]
        return [sin(proj), cos(proj)]  # [B, N, 64]
```

**Matematiksel Arka Plan:**  
Standart NN'ler düşük frekanslı fonksiyonları öğrenmeyi tercih eder ("spectral bias"). RF kavitelerinde modlar yüksek uzamsal frekanslar içerir. RFF, koordinatları yüksek boyutlu uzaya taşıyarak bu bias'ı kırar.

Bu, Rahimi & Recht (2008) "Random Features for Large-Scale Kernel Machines" çalışmasına dayanır:

$$\phi(x) = \begin{bmatrix} \sin(2\pi B^T x) \\ \cos(2\pi B^T x) \end{bmatrix}$$

`B` matrisi **öğrenilmez** — sabit tutulur. Bu, eğitim kararlılığını artırır.

`rff_scale` parametresi, `B`'nin genliğini kontrol eder:
- `scale=1.0` → Orta frekansları yakalar
- `scale > 1` → Daha yüksek frekansları temsil edebilir
- `scale < 1` → Daha düşük, pürüzsüz özellikler

### 2. Linear Attention

```python
class LinearAttention:
    q = ELU(Wq @ query) + 1.0     # Kernel: φ(x) = elu(x) + 1
    k = ELU(Wk @ key) + 1.0
    v = Wv @ value

    # O(N²) yerine O(N) karmaşıklık:
    kv = einsum('bhnd,bhne->bhde', k, v)     # K'V' matrisini önce hesapla
    output = einsum('bhnd,bhde->bhne', q, kv) # Sonra Q ile çarp
    output = output / normalization
```

**Standart Attention vs Linear Attention:**

| | Standart (Softmax) | Linear |
|---|---|---|
| Karmaşıklık | $O(N^2 \cdot d)$ | $O(N \cdot d^2)$ |
| 5000 node | 25M işlem | 327K işlem |
| Avantaj | Daha ifade gücü | Bellek dostu |

RF kavitelerinde mesh boyutları 1000–5000 node arası. Standart attention bu boyutlarda bellek patlatır, linear attention ise ölçeklenebilir.

**ELU + 1 Kernel:** Negatif değerleri yumuşatır ama sıfırlamaz. `+1` eklenmesi tüm değerlerin pozitif olmasını sağlar — bu, kernel trick'in geçerli olması için gereklidir.

### 3. GNOTBlock (Temel Yapı Taşı)

Her blokta sırasıyla şunlar devreye girer:

```
Input x
   │
   ├──▶ Cross-Attention(Q=x, KV=condition_emb)    → "Geometri bilgisini sorgula"
   │    + Residual
   │
   ├──▶ Global Context Injection (+ pooler output) → "Kavitenin büyük resmini enjekte et"
   │
   ├──▶ Self-Attention(Q=K=V=x)                   → "Node'lar birbirleriyle konuşsun"
   │    + Residual + Mask
   │
   ├──▶ GeometricGatingFFN(x, gate_from_pos)       → "Uzman seçimi"
   │    + Residual
   │
   └──▶ FiLM(x, mode_idx)                         → "Mod bilgisini aşıla" (opsiyonel)
         + Mask
```

### 4. GeometricGatingFFN (Mixture of Experts)

```python
class GeometricGatingFFN:
    experts = [MLP_0, MLP_1, MLP_2, MLP_3]  # 4 uzman

    def forward(x, gate_info):
        gate_logits = local_gating(pos)       # pos → gate kararı
        top2_weights, top2_idx = softmax(gate_logits).topk(2)

        for expert_i in selected_experts:
            output += weight_i * expert_i(x)
```

**Fiziksel Motivasyon:**  
Kavite içinde farklı bölgeler farklı fizik sergiler:
- **Sınır yakını:** Alan hızla düşer → keskin gradyan
- **Merkez:** Düzgün dağılım
- **Köşeler:** Singularity-benzeri davranış

Her "uzman" bu bölgelerden birini öğrenmeye uzmanlaşır. `pos` (koordinat/RFF) bilgisi hangi uzmanın seçileceğini belirler.

**Top-2 Seçimi:** Her nokta için en iyi 2 uzman seçilir ve ağırlıklarına göre birleştirilir. Bu, saf MoE'dan daha yumuşak bir geçiş sağlar.

### 5. FiLM (Feature-wise Linear Modulation)

```python
class FiLMConditioner:
    emb = Embedding(num_modes=20, dim=2*D)  # Her mod için gamma ve beta

    def forward(x, mode_idx):
        params = emb(mode_idx)               # [B, 2*D]
        gamma, beta = params.chunk(2)        # Her biri [B, D]
        return x * (1 + gamma) + beta         # Affine transform
```

**Neden FiLM?** Mode indeksinin modele enjekte edilmesinin birçok yolu var:
- ❌ **Concatenation:** LayerNorm tarafından yıkanır
- ❌ **Addition:** LayerNorm mean-shift'i sıfırlar
- ✅ **FiLM (Affine):** `gamma * x` çarpımsal olduğu için LayerNorm'dan sağ çıkar

**Başlatma:** `std=0.02` ile küçük normal dağılım. Bu, başlangıçta `gamma ≈ 0, beta ≈ 0` → model identity ile başlar. Sıfır ile başlatıldığında tüm modlar aynı davranır ve ayrışamaz.

### 6. AttentionPool (Learned Global Pooling)

```python
class AttentionPool:
    query = nn.Parameter(randn(1, 1, D))  # Öğrenilebilir "özet" token'ı
    attn = MultiheadAttention(D, heads)

    def forward(x, mask):
        out = attn(query, x, x, key_padding_mask=~mask)
        return out.squeeze(1)  # [B, D]
```

Tüm node bilgisini tek bir vektöre sıkıştırır. Standart `mean pooling`'den farklı olarak:
- Bazı node'lara daha fazla "dikkat" verebilir
- Öğrenilebilir bir özet çıkarır

**Kullanım yerleri:** Frekans branch'inde (global bilgi) ve her blokta (global context injection).

---

## 🔀 Mode-Specific Branching Mekanizması

Forward pass'te model sample'ları mod indeksine göre sıralar ve her modu kendi branch'inden geçirir:

```python
sort_idx = theta_int.argsort()        # Batch'i mod sırasına koy
unsort_idx = sort_idx.argsort()        # Geri dönüş indeksi

for mode_val in [0, 1, 2]:
    x_m = x_sorted[start:end]          # Bu modun sample'ları
    for block in mode_field_blocks[mode_val]:
        x_m = block(x_m, ...)
    field_parts.append(field_heads[mode_val](x_m))

field_pred = cat(field_parts)[unsort_idx]  # Orijinal sıraya geri dön
```

**Neden gerekli?**  
Mode 0 (monopol: radyal simetrik) ve Mode 1 (dipol: asimetrik) çok farklı fizik sergiler. Eğer aynı parametreler ikisini de öğrenmeye çalışırsa → **gradyan çakışması.** Mode 0'ın gradyanı, Mode 1'in parametrelerini bozar ve tersi.

Mode-specific branching ile her modun gradyanı sadece kendi parametrelerine etki eder.

---

## 📊 Parametre Sayısı Tahmini

| Bileşen | Yaklaşık Parametre |
|---------|-------------------|
| RFF (B matrisi, frozen) | ~128 (öğrenilmez) |
| query_encoder | ~200K |
| input_func_encoder | ~200K |
| Shared blocks (4×) | ~4×1.6M = 6.4M |
| Mode field blocks (3×2×) | ~6×1.6M = 9.6M |
| Field heads (3×) | ~100K |
| Freq heads (3×, per-mode) | ~300K |
| AttentionPool | ~200K |
| FiLM conditioners | ~50K |
| **TOPLAM** | **~17M parametre** |

> **Not:** Ayrı freq_blocks (eski ~3.2M) kaldırıldı. Per-mode freq_heads çok daha hafif (~300K). Net parametre tasarrufu: ~2.9M.

---

## ⚠️ Geliştirme Önerileri

### Mimari
1. **Graph Neural Network Entegrasyonu:** Şu an mesh bağlantıları (`elements`) modelde kullanılmıyor. GNN katmanları (Message Passing) eklenebilir → komşuluk bilgisi doğrudan öğrenilir.
2. **Positional Encoding Alternatifleri:** RFF yerine **Sinusoidal PE** veya **Learnable Fourier Features** denenebilir.
3. **Flash Attention:** Linear Attention yerine `flash_attn` kütüphanesi ile donanım hızlandırmalı softmax attention denenebilir. Küçük-orta mesh boyutlarında daha iyi kalite verebilir.
4. **Deeper Shared Trunk:** `n_shared_layers=4` → 6'ya çıkarılabilir. Geometrik temsil daha güçlenir.

### Fizik-Bilinçli Geliştirmeler
5. **Helmholtz Residual Loss:** Tahmin edilen alanın $\nabla^2 E + k^2 E = 0$ denklemini ne kadar sağladığı ek bir loss terimi olarak eklenebilir.
6. **Orthogonality Constraint:** Modlar birbirine dik olmalı: $\int E_m \cdot E_n \, dA = \delta_{mn}$. Bu, ek bir regularization olarak uygulanabilir.
7. **Adaptive Mode Count:** Sabit 3 mod yerine, kavite şekline göre dinamik mod sayısı.

### Performans
8. **Mixed Precision (AMP):** `torch.cuda.amp` ile float16 kullanarak ~2x hızlanma + bellek tasarrufu.
9. **Gradient Checkpointing:** Zaten `use_checkpoint=True` desteği var. Aktifleştirildiğinde RAM tasarrufu sağlar ama eğitim ~%20 yavaşlar.

---

## 🔗 Bağlantılar

- Bu modeli besleyen veri: [[03_DATASET_LOADER]]
- Eğitim detayları: [[05_TRAINING_SYSTEM]]
- Config'te mimari parametreleri: [[08_CONFIG_REFERENCE]]
- Fizik arka planı: [[09_PHYSICS_BACKGROUND]]

#model #transformer #attention #moe #film #gnot
