# 🏠 RF Cavity Neural Operator — Dashboard

> **Son Güncelleme:** 2026-09-24  
> **Amaç:** 2D RF kavitelerin rezonans frekanslarını ve alan dağılımlarını neural operator'lerle (GNOT, SpectralNO) tahmin etmek.

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
│   │   ├── gnot.py                   → [[04_MODEL_ARCHITECTURE]]
│   │   └── spectral_no.py            → SpectralNO (fiziksel Galerkin + eigh)
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
├── analyze_freq_separation.py        → [[07_VALIDATION_TOOLS]]
│
├── 📂 configs/
│   ├── default.yaml                  → [[08_CONFIG_REFERENCE]] (GNOT)
│   ├── spectral_no.yaml              → SpectralNO (`model_type: spectral_no`)
│   ├── kaggle_2gpu.yaml
│   ├── mode1_isolated.yaml
│   └── 📂 ablation/
│
├── 📂 scripts/                       → [[07_VALIDATION_TOOLS]]
│   ├── infer_val_all.py              → tüm split üzerinde metrik (CSV/JSON)
│   ├── analyze_val_errors.py
│   ├── diagnose_data_floor.py
│   └── check_mode_data.py
│
├── 📂 tests/                         → pytest (`python -m pytest -q`)
├── 📂 .github/workflows/ci.yml       → CI: ruff + --help smoke + pytest
├── requirements.txt / requirements-dev.txt
├── pyproject.toml                    → yalnızca pytest / ruff ayarları
│
├── 📂 examples/
│   ├── GNOT_RF_Cavity_Colab.ipynb
│   ├── rf_cavity.ipynb
│   ├── rf_cavity_ipynb_adlı_not_defterinin_kopyası.ipynb  → rf_cavity.ipynb kopyası
│   └── 2302.14376v3 (2).pdf          → GNOT Orijinal Paper (4.3 MB)
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
| **SpectralNO Model**     | `spectral_no.py`               | `configs/spectral_no.yaml` |
| **Eğitim Sistemi**       | `lightning_module.py`          | [[05_TRAINING_SYSTEM]]     |
| **Tahmin (Inference)**   | `infer.py`                     | [[06_INFERENCE]]           |
| **Debug & Araçlar**      | `scripts/`, `validate_data.py` | [[07_VALIDATION_TOOLS]]    |
| **Konfigürasyon**        | `default.yaml`, `spectral_no.yaml` | [[08_CONFIG_REFERENCE]] |
| **Test & CI**            | `tests/`, `.github/workflows/` | `python -m pytest -q`      |
| **Fizik Arka Planı**     | —                              | [[09_PHYSICS_BACKGROUND]]  |
| **Geliştirme Fikirleri** | —                              | [[10_IMPROVEMENT_IDEAS]]   |
| **Matematiksel İyileştirmeler** | —                       | [[14_MATHEMATICAL_IMPROVEMENTS]] |
| **SciML Literatür Taraması** | —                          | [[15_SCIML_LITERATURE_REVIEW]] |
| **Spektral Geometri Analizi** | —                         | [[16_SPECTRAL_GEOMETRY_ANALYSIS]] |
| **Sayısal Analiz İncelemesi** | —                         | [[17_NUMERICAL_ANALYSIS_REVIEW]] |
| **Colab Notebook**       | `RF_Cavity_Colab.ipynb`        | README → Kullanım          |

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
- [x] OT slot↔mod eşleştirmesi + Grassmann (subspace) kaybı (GNOT)
- [x] SpectralNO: bazdan kurulan fiziksel Galerkin L/M + `eigh` (yapısal sıralama)
- [x] pytest altyapısı (`pyproject.toml`) + GitHub Actions CI
- [x] `infer.py` çift `main`/`__main__` bloğu temizlendi
- [x] `scripts/debug_modes.py` (eski API) silindi; gradyan izolasyonu testlerde

---

## 📝 Etiketler (Tags)

#proje #rf-kavite #gnot #neural-operator #fizik #fem #deep-learning
