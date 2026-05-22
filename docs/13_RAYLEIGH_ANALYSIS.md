# 🔬 Rayleigh Quotient Analizi — Senin Sisteme Uygulanabilirlik

> **Güncelleme (2026-05):** Rowan et al. (arXiv:2506.04375) — *Solving Engineering
> Eigenvalue Problems with Neural Networks Using the Rayleigh Quotient* — bu
> yaklaşımı kâğıt-doğrulanmış hâliyle entegre ettik. Bu doküman tarihsel
> analizi (önceki "ertelendi" kararı) ve **mevcut hibrit implementasyonun**
> nasıl çalıştığını birlikte özetler.
>
> **Bağlam:** Bir kaynak multi-head Rayleigh Quotient + Orthogonality yaklaşımını önermiş.
> Bu analiz, GNOT mimarisine uygunluğunu ve alternatiflerini değerlendirir.

---

## 📌 TL;DR — Kısa Değerlendirme

| Önerilen Bileşen                 | Uygulanabilir mi?             | Neden?                                                         |
| -------------------------------- | ----------------------------- | -------------------------------------------------------------- |
| **Orthogonality Loss**           | ✅ Kesinlikle evet             | Fiziksel olarak doğru, direkt uygulanabilir                    |
| **Full Rayleigh Quotient**       | ⚠️ Kısmen, ama zorlukları var | Senin ortamında operatör $\mathcal{A}$'yı hesaplamak maliyetli |
| **Subspace Learning (Brockett)** | ❌ Şu an gereksiz              | Senin modların zaten sıralı (Mode 0 < 1 < 2)                   |
| **L2 Normalization**             | ✅ Basit ama etkili            | Peak-sign normalization zaten var, güçlendirilebilir           |
|                                  |                               |                                                                |

---

## 1. Rayleigh Quotient Nedir ve Senin Sistemde Nasıl Görünür?

Rayleigh Quotient:
$$R[u] = \frac{\langle u, \mathcal{A} u \rangle}{\langle u, u \rangle} = \frac{\int_\Omega |\nabla u|^2 \, dA}{\int_\Omega |u|^2 \, dA}$$

Bu, Helmholtz denklemi ($\nabla^2 u + k^2 u = 0$) için:
- **Pay** = Stiffness (gradyan enerjisi)
- **Payda** = Norm (fonksiyonun büyüklüğü)
- **Minimum** = $k_0^2$ (ilk eigenvalue = en düşük rezonans frekansı)

### Senin modelin bunu DOĞRUDAN kullanabilir mi?

**Sorun:** Rayleigh Quotient'ı hesaplamak için $\nabla u$ (gradyan) lazım. Senin modelinin çıktısı $u(x_i)$ — düğüm noktalarındaki ayrık değerler. Gradyanı hesaplamak için:

| Yöntem | Zorluk | Maliyet |
|--------|--------|---------|
| **FEM Stiffness matrisi** | Mesh bağlantıları (`elements`) gerekli | Batch'te farklı meshler var → matrisler farklı boyutlarda |
| **Autograd** | `torch.autograd.grad(u, x)` | Model çıktısının x'e göre türevi → zaten mümkün ama ek graph oluşturur |
| **Finite Difference** | Komşu node'lar arası fark | Düzensiz meshte hata büyük |

**Sonuç:** Full Rayleigh Quotient senin sisteme uygulanabilir ama **operasyonel maliyeti yüksek.** Her mesh farklı boyutta olduğu için stiffness matrisini batch'lemek zor.

---

## 2. Senin Sistemine En Uygun Strateji

Kaynak 3 bileşen öneriyor. Bunları senin sisteme uyarlayalım:

### ✅ A. Orthogonality Loss — KESİNLİKLE UYGULANMALI

Bu en kolay ve en etkili parça. Fiziksel zorunluluk:
$$\int_\Omega E_m(x) \cdot E_n(x) \, dA = 0 \quad (m \neq n)$$

**Senin sisteminde:**
```python
# Aynı geometrinin 3 modunu bul
# pred_m, pred_n: [N, 1] boyutunda tahminler
# node_area: [N] boyutunda (Input_funcs[:, 5])

dot = (pred_m * pred_n * node_area).sum()
norm_m = (pred_m**2 * node_area).sum().sqrt()
norm_n = (pred_n**2 * node_area).sum().sqrt()
loss_ortho += (dot / (norm_m * norm_n + eps)) ** 2
```

**Neden `node_area` ağırlığı?** Düz bir iç çarpım, sık mesh bölgelerini (sınır yakını) fazla ağırlıklandırır. `node_area` ile ağırlıklandırma, integralin doğru bir ayrık yaklaşımıdır.

### ⚠️ B. Energy Terimi — KISMI UYGULAMA

Full Rayleigh yerine, **Helmholtz Residual** olarak basitleştirilmiş versiyonu daha pratik:

$$\mathcal{L}_{\text{phys}} = \sum_i \left| \nabla^2 E(x_i) + k^2 E(x_i) \right|^2$$

Bu, Rayleigh'ın minimize ettiği şeyin "residual" versiyonu — aynı fiziği zorlar ama operatör matrisine ihtiyaç duymaz.

**Ama:** Bu bile $\nabla^2$ hesaplamayı gerektiriyor. Şu an için bunu atlamak ve sadece orthogonality + supervised loss ile gitmek daha pragmatik.

### ❌ C. Subspace Learning (Brockett) — GEREKSIZ

Brockett Cost Function, mod sıralaması bilinmediğinde kullanılır. **Ama senin veride sıralama belli:**
- Mode 0 = en düşük frekans (monopol, $f_0$)
- Mode 1, 2 = dipol modları ($f_1 \leq f_2$)

Head'ler zaten bu sıralamayla eşleşmiş (Head 0 → Mode 0, vb.). Subspace öğrenmesine gerek yok.

---

## 3. Kaynak Analizi — Ne Doğru, Ne Yanlış?

### ✅ Doğru Söylenenler

1. **"Her head bir eigenfunction adayı"** — Tam olarak senin `field_heads[i]` yapın. ✓
2. **"Shared trunk + private heads kritik"** — Senin `shared_blocks` (geometri) + `mode_field_blocks` (mod-spesifik) yapısı. ✓
3. **"Orthogonality collapse'ı önler"** — Eğer modlar dik olmazsa, model tüm head'lerden aynı çözümü üretir. ✓
4. **"Integration (quadrature) gerekli"** — `node_area` ağırlığı bunu sağlar. ✓
5. **"L2 normalization stabilize eder"** — Peak-sign normalization zaten benzeri bir şey yapıyor. ✓

### ⚠️ Kısmen Uygulanabilir

6. **"Rayleigh Quotient doğrudan loss olarak"** — Prensipte doğru ama **senin sisteminde $\nabla u$ hesaplamak pratik değil.** Bunun yerine supervised loss (FEM ground truth'a karşı MSE) zaten bunu dolaylı olarak yapıyor: eğer model doğru eigenfunction'ı öğrenirse, Rayleigh oranı otomatik olarak minimize olmuş olur.

7. **"Fixed eigenvalue ordering loss ($L_1 < L_2 < L_3$)"** — Gereksiz çünkü frekanslar zaten veriden biliniyor. Supervised frekans loss'u zaten sıralamayı öğretiyor.

### ❌ Senin Sisteme Uymayan

8. **"Transformer'ın Attention mekanizması zaten global korelasyonları yakalıyor"** — Doğru ama **senin Linear Attention kullanıyorsun.** Full softmax attention'dan daha düşük ifade gücüne sahip. Bu, Rayleigh-based self-supervised öğrenme kalitesini etkileyebilir.

9. **"Deep Ritz mantığıyla ikinci türevden kaçınabilirsin"** — Deep Ritz yöntemi, $\int |\nabla u|^2$ integrali için autograd kullanır. Ama **senin modelinde girdi koordinatları (`X`) bir parametre DEĞİL, bir tensor.** Yani `torch.autograd.grad(output, X)` çalışması için X'in `requires_grad=True` olması lazım. Bu, bellek kullanımını 2-3x artırır.

---

## 4. Önerilen Hibrit Strateji

Tüm bu analiz ışığında, senin sistemine en uygun yaklaşım:

```
┌──────────────────────────────────────────────────────┐
│                  TOPLAM LOSS                          │
│                                                       │
│  L_total = L_field + α·L_freq + λ·L_bnd + β·L_ortho │
│                                                       │
│  L_field:  MSE (peak-weighted) — MEVCUT ✓             │
│  L_freq:   MSE (per-mode)     — YENİ EKLENDİ ✓       │
│  L_bnd:    PINN boundary      — MEVCUT ✓              │
│  L_ortho:  Orthogonality      — EKLENECEK ←           │
│                                                       │
│  Rayleigh: ŞİMDİLİK EKLENMİYOR                       │
│  (supervised loss zaten eigenfunction'ları öğretiyor) │
└──────────────────────────────────────────────────────┘
```

### Neden Full Rayleigh Eklemiyoruz?

1. **Zaten supervised eğitim yapıyoruz.** FEM çözücüsü ground truth veriyor. Model doğru alan şeklini öğrenirse, Rayleigh oranı zaten minimum olur.
2. **Rayleigh, self-supervised durumda gerekli.** Eğer ground truth yoksa, Rayleigh tek rehber olur. Ama senede FEM verileri var.
3. **$\nabla u$ hesaplama maliyeti yüksek.** Düzensiz mesh üzerinde autograd veya FEM matrisleri gerektirir.
4. **Orthogonality loss + supervised loss = Rayleigh'ın sağladığı fizik bilgisinin %90'ı.** Geri kalan %10 için 3x bellek maliyeti karşılanamaz.

### Ne Zaman Rayleigh Ekleriz?

- Eğer **veri az** ve model generalize edemiyorsa → Rayleigh ek bir fizik rehberi olur
- Eğer **self-supervised** (unsupervised) eğitime geçilirse → Rayleigh zorunlu olur
- Eğer **transfer learning** yapılacaksa (farklı kaviteye adapt et) → Rayleigh az veriyle çalışmayı sağlar

---

## 5. Sonuç ve Aksiyon Planı

| Adım | Ne Yapılacak | Durum |
|------|-------------|-------|
| 1 | **Orthogonality Loss ekle** (Soft Constraint, cosine similarity) | ✅ Tamamlandı |
| 2 | **Per-mode freq prediction** | ✅ Tamamlandı |
| 3 | **Full Rayleigh Quotient** (autograd VE FEM K,M iki yol) | ✅ Tamamlandı |
| 4 | **Hard Gram-Schmidt** forward içinde | ✅ Tamamlandı |
| 5 | **Curriculum** (fizik warmup → supervised geçişi) | ✅ Tamamlandı |
| 6 | **Parametric expected Rayleigh** | ✅ Tamamlandı |

---

## 6. Mevcut Hibrit Implementation (Rowan 2506.04375 entegrasyonu)

### Eklenen modüller

| Bileşen | Dosya | Rol |
|---------|-------|-----|
| Rayleigh + GS + ordering + parametric | `src/training/physics_losses.py` | Fizik kayıpları |
| Forward-içi sert GS | `src/models/gnot.py` (`use_gram_schmidt`) | Slot ortogonalitesi garantisi |
| Curriculum scheduler | `PhysicsCurriculum` (physics_losses.py) | Faz A/B/C ağırlıkları |
| P1 FEM K, M önbelleği | `src/data/dataset_converter.py` (`_assemble_p1_K_M`) | autograd alternatifi |
| Dataset yükleme | `src/data/dataset.py` | Sparse K, M tensörlerini batch'e taşır |
| LightningModule entegrasyonu | `src/training/lightning_module.py` `_compute_loss` | Toplam kaybı oluşturur |

### Curriculum çizelgesi

```
Faz A (epoch < e1=20):    pure physics warmup
Faz B (e1..e2=80):        linear ramp — supervised devreye girer
Faz C (epoch >= e2):      supervised dominant + küçük fizik anchor
```

`enable_curriculum: false` ile devre dışı bırakılabilir; o zaman fixed-weight
ablation çalıştırılır.

### Toplam kayıp

```
L = w_field    · L_field           # Grassmannian / soft-Procrustes
  + w_freq     · L_freq            # frekans MSE
  + w_smooth   · L_bnd             # PINN sınır cezası
  + w_slot_o   · L_ortho           # soft cosine (GS açıkken w_slot_o=0)
  + w_rayleigh · L_rayleigh        # R[E_pred] ↔ (2πf_pred/c)² tutarlılığı
  + w_order    · L_order           # hinge λ_k ≤ λ_{k+1}
  + w_param    · L_param           # mini-batch Rayleigh beklentisi
```

Tüm w'ler curriculum tarafından epoch-bağımlı olarak ölçeklendirilir.

### Rayleigh hesabı — iki yol

| Mode | Maliyet | Gereksinim |
|------|---------|------------|
| `autograd` | 2-3× bellek (kâğıttaki gibi spatial autograd) | `coords.requires_grad=True` |
| `fem` | sparse `K @ E` + `M @ E` — autograd yok | Yeni dataset (K, M önbelleği) |

### Birim/birim kalibrasyonu

Tahmin edilen frekans GHz cinsinden, geometri normalize edildi (`L=0.1 m`).
Tutarlılık kaybı:
```
f_GHz   = f_norm · σ + μ                       (denormalize)
k²_phys = (2π · f_GHz · 1e9 / c)²              (1/m²)
k²_norm = k²_phys · L²                         (dimensionless)
L_rayleigh = mean( ((R - k²_norm) / k²_norm)² )
```

### Çalıştırma

```bash
# Hibrit (curriculum + GS + Rayleigh) açık
python train.py --config configs/physics_rayleigh.yaml

# Baseline (saf supervised) — default davranış
python train.py --config configs/default.yaml
```

### Doğrulama

- `tests/test_physics_loss.py` — 11 birim test (1D Fourier eigenvalue,
  FEM yolu, GS ortogonalite, ordering hinge, curriculum fazları).
- `tests/test_dataset_K_M_cache.py` — P1 K, M assembly sin(πx)sin(πy)
  eigenfunction için 2π² eigenvalue'sunu %2 hata ile döndürüyor.
- `tests/test_physics_lightning_smoke.py` — Lightning modülü Faz A ve Faz
  C'de backprop yapabiliyor; forward GS sonrası alanlar area-orthogonal.

---

## 🔗 Bağlantılar

- Orthogonality opsiyonları: [[11_ORTHOGONALITY_ANALYSIS]]
- Model mimarisi: [[04_MODEL_ARCHITECTURE]]
- Fizik: [[09_PHYSICS_BACKGROUND]]

#rayleigh #eigenvalue #orthogonality #fizik-bilincli
