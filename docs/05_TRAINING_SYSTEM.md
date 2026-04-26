# 05 — Eğitim Sistemi (Training)

> **Dosyalar:** `src/training/lightning_module.py`, `src/training/callbacks.py`, `train.py`  
> **Framework:** PyTorch Lightning  
> **Sınıf:** `GNOTLightning(pl.LightningModule)`

---

## 🎯 Ne Yapıyor?

GNOT modelinin eğitim, doğrulama ve test döngülerini yönetir. Fizik bilgili kayıp fonksiyonu (PINN boundary constraint), akıllı işaret düzeltme, mode-specific ağırlıklandırma ve çoklu LR scheduler desteği içerir.

---

## 📐 Kayıp Fonksiyonu (Loss Function) — Tam Matematiksel Formülasyon

Toplam loss 3 bileşenden oluşur:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{field}} + \alpha \cdot \mathcal{L}_{\text{freq}} + \lambda \cdot \mathcal{L}_{\text{bnd}}$$

Varsayılan değerler: $\alpha = 0.5$ (freq_weight), $\lambda = 1.0$

### 1. Field Loss (Alan Kaybı) — Peak-Weighted MSE + L1

Her mod için ayrı hesaplanır:

$$\mathcal{L}_{\text{field}} = \frac{1}{\sum w_m} \sum_{m=0}^{2} w_m \cdot \mathcal{L}_m$$

Her modun loss'u:

$$\mathcal{L}_m = \underbrace{\frac{1}{N_{\text{valid}}} \sum_i \left[(p_i - t_i)^2 \cdot (1 + 5|t_i|)\right] \cdot m_i}_{\text{Peak-Weighted MSE}} + \underbrace{0.1 \cdot \frac{1}{N_{\text{valid}}} \sum_i |p_i - t_i| \cdot m_i}_{\text{L1 regularizer}}$$

**Peak Weighting açıklaması:** `(1 + 5|t|)` terimi, hedef alanın yüksek genlikli bölgelerine daha fazla önem verir:
- $|t| = 0$ (düz bölge) → ağırlık = 1
- $|t| = 1$ (tepe noktası) → ağırlık = 6

Bu, modelin modların "tepelerini" doğru yakalamasını sağlar. Tepe bölgeleri fiziksel olarak en önemli yerlerdir (alan yoğunluğu → enerji depolanması).

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

Bu, loss'tan bağımsız olarak çıktıyı 0'a keser. Ama `loss_bnd` hâlâ hesaplanır — sınırda zaten 0 çıkmasını "öğrenmesi" için gradyan sağlar.

### 3. Frequency Loss

$$\mathcal{L}_{\text{freq}} = \text{MSE}(\hat{f}, f)$$

Frekanslar z-score normalize edilmiştir. İnference'ta GHz'e geri dönüştürülür.

**Not:** `predict_frequency: false` ise bu branch tamamen devre dışıdır. Config'te frekans tahmini kapalı ayarlı — modelin alan tahminine odaklanması hedefleniyor.

---

## 🔄 Sign Realignment (İşaret Hizalama)

**Problem:** FEM çözücü eigenfunction'ı $+E$ veya $-E$ olarak dönebilir. Peak-sign normalization bunu büyük ölçüde çözer ama bazı edge case'ler kalabilir.

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

### AdamW
```python
optimizer = AdamW(params, lr=2e-4, weight_decay=0.0)
```

**Not:** `weight_decay=0.0` bilinçli bir karar. Ağırlık çürümesi modelin genlik (amplitude) öğrenme kapasitesini baskılayabilir. Alan dağılımlarının tepeleri tam ölçekte öğrenilmelidir.

### Scheduler Seçenekleri

#### 1. ReduceLROnPlateau (Varsayılan)
```python
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5,
    patience=10, min_lr=1e-7
)
```
Val loss 10 epoch iyileşmezse LR yarıya düşer. En güvenli seçenek.

#### 2. OneCycleLR
```python
scheduler = OneCycleLR(
    max_lr=2e-4,
    pct_start=0.3,          # İlk %30'da warmup
    div_factor=25,           # Başlangıç LR: max_lr / 25 = 8e-6
    final_div_factor=1e4     # Son LR: max_lr / 1e4 = 2e-8
)
```
Agresif bir scheduler. Tek bir "dağ" profili çizer. Kısa eğitimler için ideal.

#### 3. CosineAnnealing
```python
scheduler = CosineAnnealingLR(T_max=max_epochs, eta_min=1e-6)
```
LR kosinüs eğrisi ile yavaşça düşer. Uzun eğitimler için uygun.

---

## 🖼️ Görselleştirme Callback'i

`FieldVisualizationCallback` her N epoch'ta (varsayılan: 10) bir batch'in:
- Ground truth alan dağılımını
- Modelin tahminini

TensorBoard'a görüntü olarak loglar. Bu, modelin eğitim sırasında görsel olarak takibini sağlar.

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
1. **Sobolev Loss:** Sadece alanı değil, alanın gradyanını da ($\nabla E$) hedef olarak kullanma. Bu, modelin "fiziksel pürüzsüzlüğü" öğrenmesini sağlar.
2. **Helmholtz Residual:** $\mathcal{L}_{\text{phys}} = \|\nabla^2 E + k^2 E\|^2$ terimi eklenerek fizik denklemi zorunluluğu artırılabilir.
3. **Adversarial Training:** Alan tahminlerinin "gerçekçi" olup olmadığını değerlendiren bir discriminator ağı. GAN-benzeri yaklaşım.

### Eğitim Stratejisi
4. **Curriculum Learning:** Önce basit geometrilerle (daire, kare), sonra karmaşık geometrilerle eğit.
5. **Progressive Resizing:** Önce küçük meshlerle (hızlı iterasyon), sonra büyük meshlerle (hassasiyet).
6. **EMA (Exponential Moving Average):** Model ağırlıklarının hareketli ortalamasını tutarak daha kararlı inference.
7. **Stochastic Weight Averaging (SWA):** Son epochlardaki ağırlıkların ortalaması → daha iyi generalization.

---

## 🔗 Bağlantılar

- Model mimarisi: [[04_MODEL_ARCHITECTURE]]
- Config ayarları: [[08_CONFIG_REFERENCE]]
- Inference: [[06_INFERENCE]]
- Fizik arka planı: [[09_PHYSICS_BACKGROUND]]

#training #loss #pinn #optimizer #scheduler #ddp
