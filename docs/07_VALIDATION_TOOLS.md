# 07 — Doğrulama & Debug Araçları

> **Dosyalar:** `validate_data.py`, `visualize_features.py`, `plot_splits.py`, `run_ablation.py`, `scripts/check_mode_data.py`, `scripts/debug_modes.py`

---

## 🧰 Araç Listesi

| Araç | Dosya | Amaç |
|------|-------|------|
| **Fiziksel Validasyon** | `validate_data.py` | Analitik çözümlerle karşılaştırma |
| **Feature Görselleştirme** | `visualize_features.py` | 8 feature'un harita ve histogramları |
| **Split Önizleme** | `plot_splits.py` | Train/Val ayrımını görsel olarak kontrol |
| **Ablation Study** | `run_ablation.py` | Feature encoding deney çerçevesi |
| **Mode Veri Kontrolü** | `scripts/check_mode_data.py` | Her modun Y_field istatistiği |
| **Mode Debug** | `scripts/debug_modes.py` | Gradyan izolasyonu ve sort-unsort testi |

---

## 1. Fiziksel Validasyon (`validate_data.py`)

Bilinen geometrilerin (kare ve daire) FEM çözümlerini analitik formüllerle karşılaştırır.

### Analitik Frekans Formülleri

**Dikdörtgen Kavite (Kare, a=0.08m):**
$$f_{mn} = \frac{c}{2} \sqrt{\left(\frac{m}{a}\right)^2 + \left(\frac{n}{a}\right)^2}$$

| Mod | (m, n) | Analitik Frekans (GHz) |
|-----|--------|----------------------|
| 0 | (1, 1) | ~2.652 |
| 1 | (1, 2) | ~4.192 |
| 2 | (2, 1) | ~4.192 (degenerate) |

**Dairesel Kavite (R=0.04m):**
$$f_{mn} = \frac{c \cdot j_{mn}}{2\pi R}$$

| Mod | $j_{mn}$ | Analitik Frekans (GHz) |
|-----|----------|----------------------|
| 0 | 2.40482 | ~2.868 |
| 1 | 3.83171 | ~4.570 |
| 2 | 3.83171 | ~4.570 (degenerate) |

**Beklenen hata:** <%1 (P2 elemanlar çok hassas).

### Mesh Kalitesi Kontrolü
Her üçgenin `max_edge / min_edge` oranı hesaplanır:
- `< 2.0` → ✅ İyi mesh
- `> 2.0` → ⚠️ Çarpık üçgenler var

---

## 2. Feature Görselleştirme (`visualize_features.py`)

3 tür görselleştirme üretir:

1. **Geometrik Feature Haritaları:** 8 feature'un her birini kavite üzerinde renk haritası olarak gösterir
2. **Mod Şekli + Mesh Overlay:** Alanın mesh ile birlikte görünümü
3. **Feature Dağılımları:** Her feature'un histogram + istatistikleri (μ, σ, medyan)
4. **Dataset-Wide İstatistikler:** Frekans dağılımları, mesh boyutları, feature korelasyonları

> **Not:** Bu script'te feature isimleri eski (6 feature) olarak kodlanmış. Mevcut 8-feature sistemiyle tam uyumlu olmayabilir. Güncellenmesi gerekebilir.

---

## 3. Ablation Study (`run_ablation.py`)

4 deney yapılarak hangi feature'ların önemli olduğu test edilir:

| Deney | Features | RFF | Amaç |
|-------|----------|-----|------|
| **A** | Sadece (x, y) | ❌ | Baseline: koordinat yeterli mi? |
| **B** | (x, y) + boundary | ❌ | Sınır bilgisi ne kadar katkı sağlıyor? |
| **C** | Tüm 8 feature | ❌ | RFF olmadan tam feature seti |
| **D** | Tüm 8 feature | ✅ | **Referans** — tam güçlü model |

**Karşılaştırma:** TensorBoard'da `val/field_rel_l2` metriği ile yapılır.

---

## 4. Mode Debug (`scripts/debug_modes.py`)

3 kritik test yapar:

### TEST 1: Forward Pass
Her modun tahmin istatistikleri (mean, std, min, max, Rel L2) kontrol edilir. Tüm modlar benzer istatistik gösteriyorsa → model modları ayırt edemiyor.

### TEST 2: Backward — Gradyan İzolasyonu
Tek bir modun loss'unu backward'layıp diğer modların parametrelerinin gradyan alıp almadığını kontrol eder:
```
Mode 0 backward →
  ✓ shared_trunk: gradyan ALIR (beklenen)
  ✓ mode_field_blocks[0]: gradyan ALIR (beklenen)
  ✓ mode_field_blocks[1]: gradyan ALMAZ (izolasyon sağlandı!)
  ✓ field_heads[0]: gradyan ALIR
  ✓ field_heads[1]: gradyan ALMAZ
```

Eğer başka modun bloğu da gradyan alıyorsa → **gradyan sızıntısı** var demektir.

### TEST 3: Sort-Unsort Doğruluğu
Mod sıralaması ve geri dönüşün tutarlılığı test edilir:
```python
theta_recovered = theta[sort_idx][unsort_idx]
assert (theta == theta_recovered).all()  # Hep True olmalı
```

---

## 5. Check Mode Data (`scripts/check_mode_data.py`)

Basit veri doğrulama: her modun Y_field'ının gerçekten farklı olup olmadığını kontrol eder.

Eğer tüm modların istatistikleri neredeyse aynıysa → veri üretiminde hata var (aynı mod 3 kez kaydedilmiş olabilir).

---

## ⚠️ Geliştirme Önerileri

1. **Otomatik CI Pipeline:** Bu test scriptlerini GitHub Actions ile her push'ta çalıştırma.
2. **Regression Test:** En iyi modelin Rel L2 değerini kaydet, yeni modeller daha kötüyse alarm ver.
3. **Feature Importance (SHAP/Gradient):** Her feature'un modele katkısını SHAP veya integrated gradients ile ölç.

---

## 🔗 Bağlantılar

- Feature tanımları: [[02_FEATURE_ENGINEERING]]
- Model mimarisi: [[04_MODEL_ARCHITECTURE]]
- Fizik: [[09_PHYSICS_BACKGROUND]]

#debug #validasyon #ablation #test
