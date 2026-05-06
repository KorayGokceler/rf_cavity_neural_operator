# 🏠 RF Cavity Neural Operator — Dashboard

> **Son Güncelleme:** 2026-05-06  
> **Amaç:** 2D RF kavitelerin rezonans frekanslarını ve alan dağılımlarını GNOT ile tahmin etmek.

---

## 🗺️ Proje Haritası

```
rf_cavity_neural_operator/
│
├── 📂 src/
│   ├── 📂 data_gen/
│   │   └── dataset_generator.py      → [[01_DATA_GENERATION]]
│   ├── 📂 data/
│   │   ├── dataset_converter.py      → [[02_FEATURE_ENGINEERING]]
│   │   └── dataset.py                → [[03_DATASET_LOADER]]
│   ├── 📂 models/
│   │   └── gnot.py                   → [[04_MODEL_ARCHITECTURE]]
│   └── 📂 training/
│       ├── lightning_module.py        → [[05_TRAINING_SYSTEM]]
│       └── callbacks.py              → [[05_TRAINING_SYSTEM]]
│
├── run_pipeline.py                   → Tam pipeline (gen → convert → train)
├── train.py                          → [[05_TRAINING_SYSTEM]]
├── infer.py                          → [[06_INFERENCE]]
├── convert.py                        → [[02_FEATURE_ENGINEERING]]
├── validate_data.py                  → [[07_VALIDATION_TOOLS]]
├── visualize_features.py             → [[07_VALIDATION_TOOLS]]
├── run_ablation.py                   → [[07_VALIDATION_TOOLS]]
├── plot_splits.py                    → [[07_VALIDATION_TOOLS]]
│
├── 📂 configs/
│   ├── default.yaml                  → [[08_CONFIG_REFERENCE]]
│   ├── kaggle_2gpu.yaml
│   ├── mode1_isolated.yaml
│   └── 📂 ablation/
│
├── 📂 scripts/                       → [[07_VALIDATION_TOOLS]]
│   ├── check_mode_data.py
│   └── debug_modes.py
│
├── 📂 examples/
│   ├── GNOT_RF_Cavity_Colab.ipynb
│   ├── rf_cavity.ipynb
│   └── 2302.14376v3.pdf              → GNOT Orijinal Paper
│
└── 📂 docs/                          → (Bu klasör)
```

---

## 🔗 Hızlı Navigasyon

| Modül                    | Dosya                          | Not                        |
| ------------------------ | ------------------------------ | -------------------------- |
| **Veri Üretimi**         | `dataset_generator.py`         | [[01_DATA_GENERATION]]     |
| **Feature Mühendisliği** | `dataset_converter.py`         | [[02_FEATURE_ENGINEERING]] |
| **Dataset & DataLoader** | `dataset.py`                   | [[03_DATASET_LOADER]]      |
| **GNOT Model**           | `gnot.py`                      | [[04_MODEL_ARCHITECTURE]]  |
| **Eğitim Sistemi**       | `lightning_module.py`          | [[05_TRAINING_SYSTEM]]     |
| **Tahmin (Inference)**   | `infer.py`                     | [[06_INFERENCE]]           |
| **Debug & Araçlar**      | `scripts/`, `validate_data.py` | [[07_VALIDATION_TOOLS]]    |
| **Konfigürasyon**        | `default.yaml`                 | [[08_CONFIG_REFERENCE]]    |
| **Fizik Arka Planı**     | —                              | [[09_PHYSICS_BACKGROUND]]  |
| **Geliştirme Fikirleri** | —                              | [[10_IMPROVEMENT_IDEAS]]   |

---

## 📊 Pipeline Akışı

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  1. GEOMETRI     │     │  2. FEATURE       │     │  3. EGITIM       │
│  URETIMI         │────▶│  ENGINEERING      │────▶│                  │
│                  │     │                   │     │                  │
│  gmsh + skfem    │     │  8-dim features   │     │  PyTorch         │
│  → .h5 dosyasi   │     │  → .pkl dosyasi   │     │  Lightning       │
│                  │     │                   │     │                  │
│ dataset_         │     │ dataset_          │     │ train.py +       │
│ generator.py     │     │ converter.py      │     │ lightning_       │
│                  │     │ + convert.py      │     │ module.py        │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
         ▲ Tüm adımlar run_pipeline.py ile otomatikleştirilebilir ▲
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  4. TAHMIN       │
                                                 │  (INFERENCE)     │
                                                 │                  │
                                                 │  infer.py        │
                                                 │  → .png plotlar  │
                                                 └─────────────────┘
```

---

## 🧪 Mevcut Durum

- [x] Veri üretim pipeline'ı çalışır durumda
- [x] Feature engineering (8-dim) tamamlandı
- [x] GNOT modeli kodlanmış ve çalışıyor
- [x] Multi-GPU (DDP) desteği mevcut
- [x] Ablation study altyapısı hazır
- [x] RFF (Random Fourier Features) spatial encoder implemente edildi
- [x] GNN kaldırıldı — RAM optimize edildi (~120 MB tasarruf)
- [x] Config parametreleri tamamen wired (dead param yok)
- [x] Permütasyon-invaryant dipol kaybı eklendi (mode 1/2 robustness)
- [x] 5000 geometri dataseti hedefleniyor

---

## 📝 Etiketler (Tags)

#proje #rf-kavite #gnot #neural-operator #fizik #fem #deep-learning
