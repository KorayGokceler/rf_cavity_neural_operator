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

**Checkpoint dizini / config ile:**
```bash
# Dizin verilirse en düşük val_rel_l2'li best-*.ckpt (yoksa last.ckpt) seçilir
python infer.py --checkpoint training_logs/spectral_no_v1 --data_path data/gnot_dataset_5k_v12.pkl
# Config: dataset.data_path + inference.{checkpoint_path,output_dir,num_visualize}
python infer.py --config configs/spectral_no.yaml
```

**Tutarlılık notları (train ↔ infer):**
- Split oranları, seed, `feature_indices` ve gauge-feature sıfırlama checkpoint
  `hparams['data_cfg']` içinden okunur (`build_dataset`) → eğitimdeki val/test
  split'i ve input feature'ları birebir aynı. `max_nodes` bilinçli olarak
  kullanılmaz (plot için tam mesh gerekir).
- `--deg_threshold` verilmezse dejenere küme kuralı eğitimdekiyle aynıdır
  (z-score frekans ekseninde mutlak eşik = `model.near_deg_threshold`).
  Verilirse eski davranış: fiziksel frekansta göreli eşik.
- `scale_invariant_field` ile eğitilmiş modellerde (SpectralNO) tahmin kolonları
  hedef normuna ölçeklenir; raporlanan rel-L2 eğitim metriğiyle aynıdır.
- `scripts/infer_val_all.py` ve `scripts/diagnose_data_floor.py` aynı
  `load_model` / `build_dataset` yardımcılarını kullanır.

---

## 🔧 FEM İyileştirme (`--refine N`)

`src/fem_refine.py`: modelin düğüm alanları bir deneme alt uzayı olarak alınır, P2'ye
(kenar orta noktaları = uç ortalaması) taşınır ve mesh'in kendi P2 matrisleriyle
(etiketlerin çözüldüğü ayrıklaştırma) `N` adım blok ters iterasyon + Rayleigh–Ritz uygulanır:
$S = \mathrm{span}\{V, A^{-1}MV\}$, $S^\top A S\,c = \lambda\,S^\top M S\,c$.

| Başlangıç hatası (alan, %5) | 1 adım | 2 adım |
|---|---|---|
| düzgün (ağ tipi) | ~4e-4 | ~3e-5 |
| gürültü | ~3e-5 | ~2e-6 |

(disk, 545 düğüm, ~25 ms; docs/17 ile tutarlı.) Sonrasında modlar özdeğer sırasındadır (OT gerekmez)
ve konsol `mean |Δf| model → refined` özetini yazar. Gereken: güncel `convert.py` çıktısı
(`scale` + `elements`).

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
