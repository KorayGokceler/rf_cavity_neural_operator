# 🧪 Analiz: Orthogonality Constraint & Derinlik Optimizasyonu

> **Branch:** `feature/orthogonality-and-depth`  
> **Durum:** TASARIM AŞAMASI — Onay bekleniyor

---

## 1. Orthogonality (Diklik) Neden Gerekli?

[[09_PHYSICS_BACKGROUND]] notundan:

> Helmholtz eigenvalue probleminde modlar **fiziksel olarak birbirine diktir:**
> $$\int_\Omega E_m(x,y) \cdot E_n(x,y) \, dA = 0 \quad (m \neq n)$$

Şu an model bunu **öğrenmeye zorlanmıyor.** Mode 0'ın tahmini, Mode 1'e benzeyebilir — özellikle eğitimin başlarında ve karmaşık geometrilerde. Orthogonality constraint, modelin fiziksel gerçekliğe uygun tahminler üretmesini sağlar.

---

## 2. Orthogonality — 3 Opsiyon

### Opsiyon A: Soft Constraint (Loss Penalty) ✅ ÖNERİLEN

Loss fonksiyonuna ek bir terim olarak orthogonality cezası eklenir:

$$\mathcal{L}_{\text{ortho}} = \sum_{m < n} \left( \frac{\sum_i E_m^{(i)} \cdot E_n^{(i)} \cdot A_i}{\|E_m\| \cdot \|E_n\|} \right)^2$$

Burada $A_i$ = node alanı (mesh yoğunluğu ağırlığı), $E_m$ ve $E_n$ = farklı modların tahminleri.

```python
# Pseudocode
for (m, n) in [(0,1), (0,2), (1,2)]:
    dot = (pred_m * pred_n * node_area).sum()         # alan-ağırlıklı iç çarpım
    norm_m = (pred_m**2 * node_area).sum().sqrt()
    norm_n = (pred_n**2 * node_area).sum().sqrt()
    cos_sim = dot / (norm_m * norm_n + eps)
    loss_ortho += cos_sim ** 2
```

| Avantaj | Dezavantaj |
|---------|-----------|
| Basit implementasyon | Ağırlık (`λ`) tuning gerekir |
| Mevcut pipeline'a kolayca entegre | Sert ortogonalite garanti etmez |
| Gradyanlar düzgün akar | |

**Zorluk:** Aynı geometrinin 3 modunun aynı batch'te olması gerekir. Mevcut DataLoader bunu garanti etmiyor → collate düzenlemesi lazım.

---

### Opsiyon B: Gram-Schmidt Projection (Hard Constraint)

Model çıktıları üzerinde deterministik olarak ortogonalleştirme uygula:

```python
# Forward pass sonrası:
e0 = pred_mode0                                    # Olduğu gibi
e1 = pred_mode1 - proj(pred_mode1, e0)             # Mode 0'a dik bileşeni al
e2 = pred_mode2 - proj(pred_mode2, e0) - proj(pred_mode2, e1)  # İkisine de dik
```

| Avantaj | Dezavantaj |
|---------|-----------|
| Ortogonalite %100 garanti | Mode sıralamasına bağımlı (Mode 0 "ayrıcalıklı") |
| Ek hiperparametre yok | Gradyan akışı karmaşıklaşır |
| | Forward pass'te ek maliyet |

**Risk:** Mode 0 her zaman ham çıktıyı alır, Mode 2 en çok "düzeltme" görür. Bu, modların eşit şekilde öğrenmemesine neden olabilir.

---

### Opsiyon C: Regularized Inner Product (Yumuşak + Fiziksel)

Cosine similarity yerine, modların iç çarpımını (fiziksel ortogonalite) doğrudan cezala ama node alanı ile ağırlıkla:

$$\mathcal{L}_{\text{ortho}} = \sum_{m < n} \left| \sum_i E_m^{(i)} \cdot E_n^{(i)} \cdot A_i \right|$$

Bu, L1 ceza kullanır (L2 yerine). Avantajı: tamamen sıfır olmaya zorlar ama "neredeyse dik" durumda aşırı cezalandırmaz.

| Avantaj | Dezavantaj |
|---------|-----------|
| Fiziksel olarak en doğru (alan integrali) | `node_area` feature'ı gerektirir |
| L1 daha agresif sıfıra iter | Opsiyon A'dan biraz yavaş |

---

### 📌 ÖNERİM: Opsiyon A (Soft Constraint)

Nedenleri:
1. En basit ve en güvenli
2. `λ` parametresiyle kontrol edilebilir (küçük başla: `λ=0.01`)
3. Mevcut eğitim pipeline'ını minimal bozar
4. Yanlış giderse kapatmak kolay

---

## 3. Shared vs Mode-Specific Derinlik — Analiz

[[04_MODEL_ARCHITECTURE]] notundan mevcut yapı:

```
Shared Trunk:    4 × GNOTBlock  (FiLM KAPALI, ~6.4M param)
Mode Branches:   3 × 2 × GNOTBlock  (FiLM AÇIK, ~9.6M param)
```

### Sorduğun Soru: "2 mode-specific blok fazla mı? Shared mi daha derin olmalı?"

**Kısa cevap:** Şu anki denge (4 shared + 2 mode) makul ama **shared daha derin olmalı.**

### Neden?

| Argüman | Shared Daha Derin | Mode-Specific Daha Derin |
|---------|------------------|------------------------|
| **Fizik** | Tüm modlar aynı geometri üzerinde yaşar. Geometri anlama = paylaşılan bilgi. | Her mod farklı fizik: monopol (radyal) vs dipol (kutbisel). |
| **Parametre Verimliliği** | 1 shared blok, 3 modun hepsine hizmet eder | Her mod bloğu sadece 1 moda hizmet eder (3x maliyet) |
| **Veri Miktarı** | Shared bloklar 3x daha fazla veri görür (her mod geçer) | Mode bloklar sadece kendi modunun verisini görür (1/3) |
| **Generalization** | Shared derinlik genellemeyi güçlendirir | Mode derinlik overfit riskini artırır |

### Sonuç

```
EŞİT PARAMETRE BÜTÇESİNDE:
  6 shared + 1 mode   >   4 shared + 2 mode   >   2 shared + 3 mode
       ↑                        ↑                        ↑
  En iyi generalization   Mevcut yapı         Overfit riski yüksek
```

**Ancak:** Modlar arası fark büyükse (monopol vs dipol) en az 1 mode-specific blok şart. Tamamen kaldırmak olmaz — FiLM tek başına yetmez.

### Pratik Optimizasyon Stratejisi

Bu soruyu kesin cevaplamak için en iyi yol: **3 deneme çalıştırıp karşılaştırmak.**

| Deney | Shared | Mode | Toplam Blok | Parametre Tahmini |
|-------|--------|------|------------|-------------------|
| **A** | 6 | 1 | 6 + 3×1 = 9 | ~16M |
| **B** | 4 | 2 | 4 + 3×2 = 10 | ~20M (mevcut) |
| **C** | 5 | 1 | 5 + 3×1 = 8 | ~14M |

Bu deneyleri config override ile hızlıca çalıştırabilirsin:
```bash
python train.py --override model.n_shared_layers=6 model.n_mode_layers=1
python train.py --override model.n_shared_layers=5 model.n_mode_layers=1
```

---

## 4. Implementasyon Planı

Onay verirsen şu değişiklikleri yapacağım:

### Adım 1: Orthogonality Loss (Opsiyon A)
- `lightning_module.py` → `_compute_loss` fonksiyonuna `loss_ortho` terimi ekle
- Aynı geometrinin farklı modlarını batch içinde eşle (`geom_id` ile)
- Config'e `ortho_weight` parametresi ekle (varsayılan: `0.01`)

### Adım 2: Config Güncelleme
- `default.yaml` → `training.ortho_weight: 0.01` ekle
- `training.ortho_weight: 0.0` yapılarak devre dışı bırakılabilir

### Adım 3: Derinlik Deneyleri İçin Config
- 3 deney config'i oluştur (`configs/depth_experiment/`)

### Adım 4: Docs Güncelleme
- [[05_TRAINING_SYSTEM]] → Yeni loss terimi
- [[04_MODEL_ARCHITECTURE]] → Derinlik analizi
- [[10_IMPROVEMENT_IDEAS]] → Orthogonality durumunu güncelle

---

## 🗳️ Onay Bekleniyor

1. **Orthogonality:** Opsiyon A (Soft Constraint) ile devam edeyim mi? Yoksa B veya C'yi mi tercih edersin?
2. **Derinlik:** İlk olarak `6 shared + 1 mode` config'ini denemek ister misin?

#orthogonality #derinlik #tasarim #analiz
