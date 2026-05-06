# 05 — Eğitim Sistemi (Training)

> **Dosyalar:** `src/training/lightning_module.py`, `src/training/callbacks.py`, `train.py`  
> **Framework:** PyTorch Lightning  
> **Sınıf:** `GNOTLightning(pl.LightningModule)`

---

## 🎯 Ne Yapıyor?

GNOT modelinin eğitim, doğrulama ve test döngülerini yönetir. Fizik bilgili kayıp fonksiyonu (PINN boundary constraint), akıllı işaret düzeltme, permütasyon-invaryant dipol kaybı, mode-specific ağırlıklandırma ve çoklu LR scheduler desteği içerir.

---

## 📐 Kayıp Fonksiyonu (Loss Function) — Tam Matematiksel Formülasyon

Toplam loss 3 bileşenden oluşur:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{field}} + \alpha \cdot \mathcal{L}_{\text{freq}} + \lambda \cdot \mathcal{L}_{\text{bnd}}$$

Varsayılan değerler: $\alpha = 0.5$ (`freq_weight`), $\lambda = 0.0$ (`smoothness_weight` — varsayılan kapalı)

### 1. Field Loss (Alan Kaybı) — Relative L2 + L1 Hybrid

Her mod için ayrı hesaplanır, sonra mode ağırlıkları uygulanır:

$$\mathcal{L}_{\text{field}} = \frac{1}{\sum w_m} \sum_{m=0}^{2} w_m \cdot \mathcal{L}_m$$

Her modun loss'u:

$$\mathcal{L}_m = \underbrace{\frac{\sum_i (p_i - t_i)^2 \cdot \text{mask}_i}{\sum_i t_i^2 \cdot \text{mask}_i + \varepsilon}}_{\text{Relative L2}} + \underbrace{0.1 \cdot \frac{\sum_i |p_i - t_i| \cdot \text{mask}_i}{N_{\text{valid}}}}_{\text{L1 regularizer}}$$

**Relative L2** doğrudan validation metriğini (`val/field_rel_l2`) optimize eder.  
**L1 terimi** dipol/kuadropol modlarının düğüm noktalarındaki (nodal lines) düşük genlikli bölgelere de düzgün gradyan sağlar — rel L2 bu bölgeleri görmezden gelebilir.

### 2. Boundary Loss (PINN Sınır Cezası)

```python
dist_bnd = batch['Input_funcs'][:, :, 2]   # dist_to_boundary feature
bnd_mask = (dist_bnd < 1e-4)                # Sınırdaki noktalar

loss_bnd = mean(pred_field[bnd_mask] ** 2)  # Sınırda tahmin → 0 olmalı
```

**Fizik:** Dirichlet BC'den $E(x_{\text{bnd}}) = 0$. Bu, ağa fizik bilgisi aşılar.

**Hard Constraint (Kırpma):**
```python
pred_field = pred_field * (~bnd_mask).float()  # Sınırdaki tahminleri 0'a zorla
```

Bu, loss'tan bağımsız olarak çıktıyı 0'a keser. `loss_bnd` ise sınırda sıfır çıkmayı öğrenmesi için ek gradyan sağlar. `smoothness_weight=0.0` (varsayılan) ile bu terim loss'a katkıda bulunmaz, ancak hard constraint hâlâ aktiftir.

### 3. Frequency Loss

$$\mathcal{L}_{\text{freq}} = \text{MSE}(\hat{f}, f)$$

Frekanslar z-score normalize edilmiştir. İnference'ta GHz'e geri dönüştürülür.

**Not:** `predict_frequency: false` ise bu branch hesaplanmaz ve total loss'a eklenmez.

---

## 🔄 Sign Realignment (İşaret Hizalama)

**Problem:** FEM çözücü eigenfunction'ı $+E$ veya $-E$ olarak dönebilir.

**Çözüm — Eğitim sırasında:**
```python
with torch.no_grad():
    diff_pos = MSE(pred, target)          # Normal yönde hata
    diff_neg = MSE(pred, -target)         # Ters yönde hata
    signs = where(diff_pos <= diff_neg, +1, -1)

pred_field = pred_field * signs           # Daha yakın tarafı seç
```

**Kritik:** `torch.no_grad()` içinde yapılır — signs üzerinden gradyan akışı yok. Aksi takdirde model "hedefi ters çevirmeyi" öğrenebilir (degenerate çözüm).

---

## 🔀 Permütasyon-İnvaryant Dipol Kaybı

**Problem:** Mode 1 ve Mode 2 dipoler oluşturur. Yakın frekanslı geometrilerde FEM çözücü bu iki modu keyfi yönde döndürerek döndürebilir. Eğer dönüştürme (canonical rotation) mükemmel çalışmazsa, Model 1 tahminini hedef Mode 2 ile eşleştirilirse daha düşük hata elde edilebilir.

**Çözüm (`permutation_invariant_dipole: true`):**

Her batch'te, aynı geometrinin her iki modu da varsa, iki atama denenır:
- Normal: pred_1 ↔ target_1, pred_2 ↔ target_2  
- Takas: pred_1 ↔ target_2, pred_2 ↔ target_1  

Toplam kaybı düşüren atama seçilir.

```python
if l_swp_1 + l_swp_2 < original_loss_1 + original_loss_2:
    use_swapped_assignment()
```

Bu, tek başına **canonical rotation'ın** yanında ek bir güvenlik ağı işlevi görür. Geometriler batch'e geometri bazlı split ile girdiğinden, aynı geometrinin her iki modu sık sık aynı batch'te bulunur.

---

## 📊 Metrikler

| Metrik | Formül | Kullanım |
|--------|--------|----------|
| **Relative L2** | $\frac{\|p - t\|_2}{\|t\|_2}$ | Primary metric (val/field_rel_l2) |
| **R² Score** | $1 - \frac{\sum(p_i - t_i)^2}{\sum(t_i - \bar{t})^2}$ | Genel kalite |
| **MAE** | $\frac{1}{N}\sum|p_i - t_i|$ | Node seviyesi hata |
| **Freq MAE (GHz)** | $|f_{pred} - f_{true}|$ GHz cinsinden | Frekans tahmini kalitesi |
| **Mode-specific Rel L2** | Her mod için ayrı Relative L2 | Mode-level izleme |

**EarlyStopping monitörü:** `val/field_rel_l2` (düşük = daha iyi)

---

## ⚙️ Optimizer & Scheduler

### AdamW & Özel Öğrenme Oranları (Per-Mode LR)

Model mimarisindeki dallanmalar farklı düzeyde zorluklara sahip olduğu için, mod-specific branch'ler (`lr_mode_specific`) ve frekans headleri (`lr_freq_heads`) kendi özel öğrenme oranlarına sahip olabilir. Varsayılan olarak her ikisi de `null` — tüm parametreler `base_lr` ile eğitilir.

```python
optimizer = AdamW([
    {'params': shared_params, 'lr': base_lr},
    {'params': mode_0_params, 'lr': lr_mode_specific[0]},  # null → base_lr
    {'params': freq_heads,    'lr': lr_freq_heads},         # null → base_lr
], weight_decay=0.0)
```

**Not:** `weight_decay=0.0` bilinçli bir karar. Ağırlık çürümesi modelin genlik (amplitude) öğrenme kapasitesini baskılayabilir.

### Scheduler Seçenekleri

#### 1. custom_cosine (Varsayılan)

```python
# Epoch 0-9 (warmup): LR = base_lr (sabit)
# Epoch 10+: Kosinüs ile cosine_eta_min'e düşer
def lr_lambda(epoch):
    if epoch < 10:
        return 1.0
    progress = (epoch - 10) / (max_epochs - 10)
    cosine_factor = 0.5 * (1 + cos(π * progress))
    return (eta_min + (base_lr - eta_min) * cosine_factor) / base_lr
```

10 epoch boyunca tam LR ile stabil bir başlangıç sağlar, ardından kosinüs eğrisi ile yavaşça düşer. 300 epoch eğitim için idealdir.

#### 2. ReduceLROnPlateau

```python
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', 
    factor=cfg.reducelr_factor,     # Config'ten (örn 0.5)
    patience=cfg.reducelr_patience, # Config'ten (örn 10)
    min_lr=1e-7
)
```

Belirlenen süre boyunca val loss iyileşmezse LR küçültülür. En güvenli adaptif seçenek.

#### 3. OneCycleLR

```python
scheduler = OneCycleLR(
    max_lr=base_lr,
    pct_start=0.3,          # İlk %30'da warmup
    div_factor=25,           # Başlangıç LR: max_lr / 25
    final_div_factor=1e4     # Son LR: max_lr / 1e4
)
```

Agresif bir scheduler. Kısa eğitimler için ideal.

#### 4. CosineAnnealing

```python
scheduler = CosineAnnealingLR(T_max=max_epochs, eta_min=cosine_eta_min)
```

Warmup olmadan direkt kosinüs. `custom_cosine`'nin daha basit versiyonu.

---

## 🖼️ Görselleştirme Callback'i

`FieldVisualizationCallback` her N epoch'ta (varsayılan: 10) bir batch'in:
- Ground truth alan dağılımını
- Modelin tahminini
- Hata haritasını (pred - truth)

TensorBoard'a görüntü olarak loglar.

---

## 🔧 Expert Load Balancing Monitoring

Her validation epoch'unun sonunda, her bloktaki expert kullanım oranları TensorBoard'a loglanır:
```
Experts_shared_blocks_0/E0: %35
Experts_shared_blocks_0/E1: %25
Experts_shared_blocks_0/E2: %22
Experts_shared_blocks_0/E3: %18
```

Expert'ler arasında dengeli dağılım istenir. Bir expert %80+ kullanılıyorsa → MoE çalışmıyor, diğer expert'ler ölü.

---

## 🖥️ Multi-GPU (DDP) Desteği

- `strategy: "ddp"` ile birden çok GPU'da eğitim yapılabilir
- `find_unused_parameters=False` → Tüm parametreler kullanıldığı için DDP overhead azaltılır
- `sync_dist=True` → Tüm GPU'lar arası metrikler senkronize
- Rank-0 guard: Yazdırma/loglama sadece ana GPU'da

---

## ⚠️ Geliştirme Önerileri

### Loss Fonksiyonu
1. **Sobolev Loss:** Sadece alanı değil, alanın gradyanını da ($\nabla E$) hedef olarak kullanma.
2. **Helmholtz Residual:** $\mathcal{L}_{\text{phys}} = \|\nabla^2 E + k^2 E\|^2$ terimi eklenerek fizik denklemi zorunluluğu artırılabilir.

### Eğitim Stratejisi
3. **Curriculum Learning:** Önce basit geometrilerle (daire, kare), sonra karmaşık geometrilerle eğit.
4. **EMA (Exponential Moving Average):** Model ağırlıklarının hareketli ortalamasını tutarak daha kararlı inference.
5. **Stochastic Weight Averaging (SWA):** Son epochlardaki ağırlıkların ortalaması → daha iyi generalization.

---

## 🔗 Bağlantılar

- Model mimarisi: [[04_MODEL_ARCHITECTURE]]
- Config ayarları: [[08_CONFIG_REFERENCE]]
- Inference: [[06_INFERENCE]]
- Fizik arka planı: [[09_PHYSICS_BACKGROUND]]

#training #loss #pinn #optimizer #scheduler #ddp
