# 06 — Inference (Tahmin)

> **Dosya:** `infer.py`  
> **Girdi:** Eğitilmiş checkpoint (`.ckpt`) + Dataset  
> **Çıktı:** Her geometri için karşılaştırmalı görselleştirme (`.png`)

---

## 🎯 Ne Yapıyor?

Eğitilmiş modeli yükleyip, test/val verisinde tahmin yapıyor. Her geometri için tüm modların (0, 1, 2) Ground Truth ve Prediction karşılaştırmasını tek bir görselde sunuyor.

---

## 🔄 Tahmin Akışı

```
1. Checkpoint yükle           → GNOTLightning.load_from_checkpoint(path)
2. Dataset yükle               → GNOTDataset(split='val')
3. Batch'ler üzerinde iterasyon
4. Her sample için:
   a. Forward pass → field_pred, freq_pred
   b. Sign-agnostic karşılaştırma
   c. Sonuçları geometri bazlı grupla
5. Her geometri için 3-modlu comparison plot oluştur
```

---

## 🔄 Sign-Agnostic Evaluation

Inference'ta da işaret belirsizliği ele alınır:

```python
# iki yönde de hata hesapla
rel_pos = ||pred - target|| / ||target||        # Normal
rel_neg = ||pred + target|| / ||target||        # Ters

if rel_neg < rel_pos:
    # Model ters faz öğrenmiş → görselleştirmede ters çevir
    final_pred = -pred
    sign_info = "*"           # Plotlarda yıldız ile işaretle
```

Bu şekilde model doğru alan şeklini öğrendiği sürece, faz (+/−) farklılıkları hata olarak sayılmaz.

---

## 📊 Çıktı Formatı

Her geometri için tek bir `.png`:
```
┌────────────────────────────────────┐
│        Geometry ID: 0042            │
├──────────────┬─────────────────────┤
│  Mode 0 GT   │  Mode 0 Prediction  │
│  Freq: 3.21  │  Freq: 3.18 (0.032) │
├──────────────┼─────────────────────┤
│  Mode 1 GT   │  Mode 1 Prediction* │
│  Freq: 5.67  │  Freq: 5.71 (0.089) │
├──────────────┼─────────────────────┤
│  Mode 2 GT   │  Mode 2 Prediction  │
│  Freq: 7.84  │  Freq: 7.79 (0.045) │
└──────────────┴─────────────────────┘

* = sign flip uygulandı
(0.089) = Relative L2 Error
```

---

## 💻 Kullanım

```bash
python infer.py \
    --checkpoint training_logs/gnot_v1/best-epoch=42-val_rel_l2=0.0512.ckpt \
    --data_path data/gnot_dataset.pkl \
    --split test \
    --num_samples 10 \
    --output_dir inference_plots
```

**Opsiyonel:** Frekans istatistikleri override edilebilir:
```bash
python infer.py --checkpoint ... --freq_mean 5.2 --freq_std 3.1
```

---

## ⚠️ Geliştirme Önerileri

1. **Nicel Rapor:** Sadece görsel değil, CSV/JSON formatında sayısal sonuçlar (tüm geometrilerin Rel L2, frekans hatası).
2. **Hata Haritası:** `|pred - target|` farkını ayrı bir panel olarak çizme → nerede hata yapıyor?
3. **Uncertainty Estimation:** MC-Dropout veya ensemble ile her tahminin güven aralığı.
4. **Yeni Geometri Tahmini:** Eğitim setinde olmayan, kullanıcının verdiği geometri için tahmin yapabilme (mesh → features → inference pipeline).
5. **Real-Time Demo:** Basit bir web arayüzü ile kavite geometrisi çizilip anlık tahmin yapılabilir.

---

## 🔗 Bağlantılar

- Eğitim: [[05_TRAINING_SYSTEM]]
- Modelin çıktı formatı: [[04_MODEL_ARCHITECTURE]]

#inference #tahmin #gorsellesirme
