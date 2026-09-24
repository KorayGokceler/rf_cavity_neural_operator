# RF Cavity Neural Operator

2D RF kavitelerin rezonans mod şekillerini (alan dağılımı $E(x,y)$) ve rezonans frekanslarını,
geometriden doğrudan tahmin eden neural operator'ler. Model, PEC sınır koşullu Helmholtz
özdeğer problemini öğrenir; inference sırasında FEM çözümüne gerek kalmaz.

$$\nabla^2 E + k^2 E = 0, \qquad E|_{\partial\Omega} = 0$$

Bir geometri için ilk $K$ mod (varsayılan $K=3$) ve her modun frekansı (GHz) tahmin edilir.

---

## Modeller

Model seçimi config'teki üst seviye `model_type` anahtarıyla yapılır (`gnot` varsayılan, `spectral_no`).

### 1. GNOT — `configs/default.yaml` (`model_type` yok ⇒ `gnot`)

Geometric Neural Operator Transformer (`src/models/gnot.py`).

- **Random Fourier Features** ile koordinat kodlama (Gauss kernel yaklaşımı)
- **LinearAttention** (ELU+1 kernel, $O(Nd^2)$) + **CrossAttention** ile geometri sorgusu
- **GeometricGatingFFN**: 4 uzmanlı dense MoE
- Paylaşılan gövde + moda özel dallar; her dal kendi alanını ve frekansını tahmin eder
- Çıkış sırası serbest olduğu için eğitimde **OT (optimal transport) slot↔mod eşleştirmesi**
  ve yakın-dejenere modlar için **Grassmann / subspace kaybı** kullanılır
- **Ritz başlığı** (`model.ritz_basis: 2`, bu config'te açık): her slot 2 baz fonksiyonu verir,
  $K\cdot 2$ alanlık alt uzayda mesh'in tam P1 matrisleriyle Rayleigh–Ritz → modlar özdeğer
  sırasında (OT gerekmez), dejenere çiftler yapısal olarak alt uzay, $f = c\sqrt{\lambda}/(2\pi\,\text{scale})$.
  `ritz_basis: 0` eski doğrudan çıkış (o zaman `max_nodes: 1024` kullanılabilir).

### 2. SpectralNO — `configs/spectral_no.yaml` (`model_type: spectral_no`)

Fiziksel Galerkin indirgemesiyle özdeğer çözen spektral neural operator (`src/models/spectral_no.py`).

1. RFF + düğüm özellikleri → düğüm başına gömme
2. Noktasal MLP → $M$ adet baz fonksiyonu $\psi_m(x)$ (`n_basis`, varsayılan 16);
   yumuşak Dirichlet kapısı ile sınırda $\psi_m = 0$
3. Rijitlik ve kütle matrisleri bazdan kurulur: $L_{mn} = \int \nabla\psi_m\cdot\nabla\psi_n$, $M_{mn} = \int \psi_m\psi_n$.
   `model.spectral.assembly: p1` (varsayılan config): $\psi$ düğüm değerlerinin P1 interpolantı,
   integraller mesh üçgenleri üzerinde **tam** (consistent mass) → $\psi\in H^1_0$, gerçek
   Rayleigh–Ritz üst sınırı, autograd yok (hızlı). `assembly: nodal`: eski düğüm kuadratürü + autograd.
4. $L u = \lambda M u$ → Cholesky + `torch.linalg.eigh`; özdeğerler **yapısal olarak sıralı**
   (OT eşleştirmesi gerekmez), M-ortogonallik yapısal
5. $\phi_k = \sum_m u_{k,m}\psi_m$; frekans fizikten: `model.physics_freq: true` ⇒
   $f = c\sqrt{\lambda}/(2\pi\,\text{scale})$ (öğrenilen kafa yok; `scale` PKL'de saklanır).
   GNOT'ta aynı ayar: kafa boyutsuz log √λ tahmin eder, boyut `scale`'den gelir.

Eğitim kaybı `training.area_weighted_field: true` ile düğüm alanıyla ağırlıklıdır (≈ $\int_\Omega$).
Bu seçimlerin gerekçesi: [docs/14](docs/14_MATHEMATICAL_IMPROVEMENTS.md)–[17](docs/17_NUMERICAL_ANALYSIS_REVIEW.md).

---

## Veri Hattı

```
dataset_generator.py ──► H5 ──► convert.py ──► PKL/H5 (özellikli) ──► train.py
   (gmsh + scikit-fem, P2 FEM)     (geometrik özellik çıkarımı, normalizasyon)
```

Geometri tipleri: **sharp** (7–12 köşeli rastgele poligon), **smooth** (Fourier pertürbasyonlu disk),
**calibration** (kare / daire / halka — analitik olarak doğrulanabilir, `--mode calibration`).
Düğüm özellikleri: `x_norm, y_norm, dist_boundary, dir_bnd_x, dir_bnd_y, node_area,
cos_principal, sin_principal`, `dist_2nd/3rd_boundary, curvature, convexity` ve `torsion`
(burulma fonksiyonu $w/\max w$; `val_dim: 13`). `max w` geometri başına saklanır: GNOT frekansı
$\sqrt{\lambda_1} \approx j_{01}/(2\sqrt{\max w})$ ön bilgisinden başlar, SpectralNO bazı $\psi = w\cdot N(x)$ ile kurar.

---

## Kurulum

```bash
git clone https://github.com/KorayGokceler/rf_cavity_neural_operator
cd rf_cavity_neural_operator

# gmsh için sistem kütüphaneleri (Debian/Ubuntu/Colab)
sudo apt-get install -y libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1

pip install -r requirements.txt        # çalışma bağımlılıkları
pip install -r requirements-dev.txt    # + pytest, ruff
```

Sadece CPU için `pip install torch` (PyPI) yeterlidir; GPU için pytorch.org'daki CUDA wheel'ini kullanın.

---

## Kullanım

### Colab notebook (en kolay yol)

[`RF_Cavity_Colab.ipynb`](RF_Cavity_Colab.ipynb): tek ayar hücresi (`MODE="smoke"` / `"full"`,
model config'i, isteğe bağlı Google Drive) → *Runtime → Run all*. Üretim → dönüştürme →
eğitim → inference → görseller sırayla çalışır; `smoke` modu CPU'da bile ~1 dakikadır.

### Uçtan uca pipeline (üretim → dönüştürme → eğitim)

```bash
python run_pipeline.py --config configs/default.yaml       # GNOT
python run_pipeline.py --config configs/spectral_no.yaml   # SpectralNO

# Adım atlama: --skip-gen, --skip-convert, --skip-train
# Bilinmeyen argümanlar train.py'ye aktarılır:
python run_pipeline.py --config configs/default.yaml --skip-gen --skip-convert \
    --override training.num_workers=2
```

Parametreler config'in `data_gen` ve `data_convert` bölümlerinden okunur.

### Adım adım

```bash
# 1) FEM veri seti (H5)
python src/data_gen/dataset_generator.py --h5_filename rf_cavity_5000_dataset.h5 --n_total 5000

# 2) Özellik çıkarımı (H5 → PKL)
python convert.py --h5_filepath rf_cavity_5000_dataset.h5 \
    --output_path data/gnot_dataset_5k.pkl --modes 0 1 2

# 3) Eğitim
python train.py --config configs/default.yaml
python train.py --config configs/spectral_no.yaml

# Config değerlerini CLI'dan ezmek (key=value, listeler JSON)
python train.py --config configs/default.yaml \
    --override model.embed_dim=128 training.batch_size=32 "training.mode_loss_weights=[1,2,2]"

# Checkpoint'ten devam
python train.py --config configs/default.yaml --resume path/to.ckpt
```

Loglar ve checkpoint'ler `training_logs/<exp_name>/` altına yazılır (TensorBoard: `tensorboard --logdir training_logs`).
Checkpoint dosya adı metrik adındaki `/` yüzünden bir alt klasör içerir:
`training_logs/<exp_name>/best-epoch=XX-val/field_rel_l2=0.XXXX.ckpt`.

### Inference

```bash
# Birkaç geometri için GT | Tahmin | Hata görselleri (PNG); eğitim klasörü verilirse best/last ckpt seçilir
python infer.py --checkpoint training_logs/gnot_5k_v1 \
    --data_path data/gnot_dataset_5k.pkl --split val --num_samples 5 --output_dir inference_plots

# + FEM iyileştirme: tahmin → P2 mesh üzerinde 2 adım ters iterasyon + Rayleigh–Ritz
#   (src/fem_refine.py; frekans hatası etiket doğruluğuna iner, geometri başına ~25–70 ms)
python infer.py --checkpoint training_logs/gnot_5k_v1 --data_path data/gnot_dataset_5k.pkl --refine 2

# Tüm split üzerinde metrikler (CSV/JSON, isteğe bağlı her geometri için grafik)
python scripts/infer_val_all.py --checkpoint path/to.ckpt --split val \
    --csv val_metrics.csv --json val_metrics.json
```

`--deg_threshold` (varsayılan 0.05) yakın-dejenere mod eşiğini belirler; bu modlar subspace hatasıyla raporlanır.
Hem GNOT hem SpectralNO checkpoint'leri aynı komutlarla yüklenir (`model_type` checkpoint hiperparametrelerinde saklıdır).

### Smoke test (birkaç dakika, CPU)

Tüm hattın çalıştığını küçük bir veri setiyle doğrulamak için:

```bash
python src/data_gen/dataset_generator.py --h5_filename smoke.h5 --n_total 12 --n_plot 0
python convert.py --h5_filepath smoke.h5 --output_path data/smoke.pkl --modes 0 1 2
python train.py --config configs/spectral_no.yaml --fast_dev_run \
    --override dataset.data_path=data/smoke.pkl training.num_workers=0
```

`--fast_dev_run` tek bir train/val batch'i çalıştırır ve checkpoint yazmaz.

---

## Testler ve CI

```bash
python -m pytest -q              # tüm testler (tests/, ayarlar pyproject.toml'da)
python -m pytest -q -m "not slow"
ruff check .                     # sözdizimi hatası / tanımsız isim kontrolü
```

Testler: RFF kernel özellikleri, geometri bazlı split (sızıntı yok), kayıp fonksiyonları
(işaret hizalama, OT / Grassmann), model forward şekilleri ve gradyan akışı, config sistemi,
entegrasyon. `@pytest.mark.gmsh` ile işaretli testler gmsh yüklenemezse otomatik atlanır.

GitHub Actions (`.github/workflows/ci.yml`): CPU torch + gmsh sistem kütüphaneleri kurulur,
`ruff check`, tüm giriş script'lerinin `--help` smoke testi ve `pytest` çalıştırılır.

---

## Proje Yapısı

```
├── configs/
│   ├── default.yaml          # GNOT (ana config)
│   ├── spectral_no.yaml      # SpectralNO
│   ├── kaggle_2gpu.yaml      # çoklu GPU (DDP)
│   ├── mode1_isolated.yaml   # tek mod deneyi
│   └── ablation/             # özellik ablation config'leri
├── src/
│   ├── models/gnot.py, spectral_no.py
│   ├── data/dataset.py, dataset_converter.py
│   ├── data_gen/dataset_generator.py
│   ├── training/lightning_module.py, callbacks.py
│   └── config.py             # YAML + dotted-key override
├── scripts/                  # infer_val_all, analyze_val_errors, diagnose_data_floor, ...
├── tests/                    # pytest
├── docs/                     # ayrıntılı dokümantasyon (Türkçe)
├── run_pipeline.py  train.py  infer.py  convert.py
└── validate_data.py  visualize_features.py  plot_splits.py  analyze_freq_separation.py  run_ablation.py
```

---

## Dokümantasyon

[`docs/00_DASHBOARD.md`](docs/00_DASHBOARD.md) ile başlayın. Öne çıkanlar:
[veri üretimi](docs/01_DATA_GENERATION.md), [özellikler](docs/02_FEATURE_ENGINEERING.md),
[model mimarisi](docs/04_MODEL_ARCHITECTURE.md), [eğitim](docs/05_TRAINING_SYSTEM.md),
[inference](docs/06_INFERENCE.md), [doğrulama araçları](docs/07_VALIDATION_TOOLS.md),
[config referansı](docs/08_CONFIG_REFERENCE.md), [fizik](docs/09_PHYSICS_BACKGROUND.md).
Matematiksel incelemeler: [14 özet](docs/14_MATHEMATICAL_IMPROVEMENTS.md),
[15 SciML literatürü](docs/15_SCIML_LITERATURE_REVIEW.md),
[16 spektral geometri](docs/16_SPECTRAL_GEOMETRY_ANALYSIS.md),
[17 sayısal analiz](docs/17_NUMERICAL_ANALYSIS_REVIEW.md).

---

## Referanslar

- [GNOT: A General Neural Operator Transformer for Operator Learning](https://arxiv.org/abs/2302.14376) — Hao vd., 2023
- [Random Features for Large-Scale Kernel Machines](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) — Rahimi & Recht, 2007
