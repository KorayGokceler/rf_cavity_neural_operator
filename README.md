# RF Cavity Neural Operator

3B RF kavitelerin rezonans modlarını (frekans + elektrik/manyetik alan) ve kavite büyüklüklerini
(Q0, G, R/Q, R_sh, T, Epk/Eacc, Bpk/Eacc) geometriden doğrudan tahmin eden bir neural operator.
Model, PEC duvarlı Maxwell özdeğer problemini öğrenir; tahmin sırasında FEM çözümü gerekmez.

$$\nabla\times\nabla\times\mathbf E = k^2\,\mathbf E \ \text{ in } \Omega, \qquad \mathbf n\times\mathbf E = 0 \ \text{ on } \partial\Omega$$

- **Formülasyon:** E alanı (CST / HFSS gibi), en düşük mertebe Nédélec (Whitney N0) kenar DOF'ları.
  PEC şartı yapısal: duvar kenarlarının DOF'u tam 0. Kulplu kaviteler (spoke, HWR, DTL) dahil her
  kapalı PEC kavite çalışır.
- **Model:** `EigenspaceOperator3D` — kütle-farkında lineer attention ile her kenarda `m` baz alanı
  üretir, gradyan çekirdeği projekte edilmiş Rayleigh–Ritz ile `K` mod (frekans + alan) çıkarır;
  $f = c\sqrt{\lambda}/(2\pi\,\text{scale})$. Ayrıntılar: [docs/20](docs/20_3D_MODEL.md).

## Akış

```
src/data_gen/dataset_generator_3d.py ──► H5 parçaları ──► convert_3d.py ──► PKL ──► train.py ──► checkpoint
  (gmsh geometri + tet mesh,               (özellikler, N0 operatörleri,                     │
   skfem N0 özdeğer çözümü)                  QoI etiketleri)                                   ▼
                                                     scripts/eval_3d.py · eval_qoi.py · predict_geometry.py · web arayüzü
```

- **Geometri aileleri** (`src/data_gen/cavity_shapes.py`): `elliptical` (TESLA tipi, 1–5 hücre), `reentrant`,
  `pillbox_pipes`, `ridged_box`, `composite`, `freeform`, `hwr`, `spoke`, `dtl`; dağılım dışı test için
  `box`, `coax_qw`, `pillbox_port`, `elliptical_long`, `junction`.
- **Düğüm özellikleri:** `x, y, z, dist_to_boundary, dir_bnd_x/y/z, node_volume, torsion`.
- **Etiketler:** her geometri için ilk `K` modun frekansı, N0 alan DOF'ları ve kavite büyüklükleri
  (bakır duvar, β = 1, U = 1 J; [docs/24](docs/24_CAVITY_QOI.md)).

## Nerede çalıştırılır

| ortam | dosya | ne yapar |
|---|---|---|
| **Colab, tek hücre** | [`Colab_WebUI.ipynb`](Colab_WebUI.ipynb) | Drive + kurulum + web arayüzü (telefondan da): dataset üretimi, eğitim, tahmin, inceleme |
| Colab, adım adım | [`Colab_Maxwell3D.ipynb`](Colab_Maxwell3D.ipynb) | üretim → PKL → eğitim → değerlendirme → görselleştirme hücreleri |
| TRUBA (SLURM) | [`cluster/truba/`](cluster/truba/README.md) | aile başına dataset üretimi, birleştirme, eğitim, değerlendirme |
| kendi makinen | aşağıdaki komutlar | |

## Kurulum

```bash
git clone https://github.com/KorayGokceler/rf_cavity_neural_operator
cd rf_cavity_neural_operator
sudo apt-get install -y libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1   # gmsh için
pip install -r requirements.txt            # + requirements-web.txt (arayüz), requirements-dev.txt (testler)
```

Yalnız CPU için `pip install torch` yeterlidir; GPU için pytorch.org'daki CUDA wheel'ini kullanın.

## Komutlar

```bash
# 1) FEM verisi (H5); mesh 0.10 eğitim çözünürlüğü
python src/data_gen/dataset_generator_3d.py --n_total 200 --mesh_size 0.10 --n_eigen_modes 10 \
    --families elliptical reentrant --sampling sobol --deform_prob 0.5 --h5_filename data/part00.h5

# 2) PKL (özellikler + operatörler + QoI etiketleri; --no_operators: hafif PKL)
python convert_3d.py --h5_filepath data/part*.h5 --output_path data/train.pkl --no_operators

# 3) Eğitim (checkpoint'ler training_logs/<exp_name>/, ilerleme progress.json)
python train.py --config configs/eigenspace_3d.yaml --override dataset.data_path=data/train.pkl \
    dataset.cache_operators=false training.exp_name=small
python train.py ... --resume training_logs/small/last.ckpt          # devam

# 4) Değerlendirme
python scripts/eval_3d.py  --checkpoint training_logs/small --data_path data/train.pkl --split test
python scripts/eval_qoi.py --checkpoint training_logs/small --data_path data/train.pkl --split test
python scripts/plot_3d.py  --checkpoint training_logs/small --data_path data/train.pkl   # kesitler + .vtu

# 5) Yeni bir kavite (STEP / mesh / üretici ailesi) için tahmin, isteğe bağlı FE karşılaştırması
python scripts/predict_geometry.py --checkpoint training_logs/small --step cavity.step --unit mm --fe

# Web arayüzü (yerel ağ ya da --tunnel ile her yerden)
python scripts/serve_web.py --checkpoint training_logs/small --data data --gen_root data/ui --runs_root training_logs
```

## Repo haritası

```
src/data_gen/    geometri aileleri (cavity_shapes.py) + N0 E-alanı özdeğer üreticisi (dataset_generator_3d.py)
src/data/        H5 → PKL dönüştürücü (dataset_converter_3d.py), eğitim dataset'i + collate (dataset_3d.py)
src/models/      EigenspaceOperator3D, H(curl) cebiri (hcurl.py), yapı taşları (layers.py)
src/training/    Lightning modülü (kayıplar, metrikler), checkpoint bulma, progress.json
src/qoi/         kavite büyüklükleri: numpy referansı, türevlenebilir torch sürümü, analitik referanslar
src/viz/         alan rekonstrüksiyonu, kesit figürleri, ParaView .vtu, notebook görüntüleyici
src/service/     web API (FastAPI): tahmin, dataset gezgini, üretim ve eğitim işleri
web/             arayüz (React + vtk.js)
scripts/         değerlendirme, tahmin, aktif örnekleme, çözünürlük çalışması, sunucu başlatma
cluster/truba/   SLURM betikleri
configs/         eigenspace_3d.yaml
tests/           pytest (sentetik veri + küçük gmsh örnekleri)
```

## Doğruluk (etiketler)

Etiketler mesh 0.10'da N0 E-alanı çözümüdür: frekans hatası ~%0.2–0.5 (pillbox TM010 −%0.24, TESLA hücresi
TM010 +%0.15), eğri duvarlı kavitelerde Q0/G'de birkaç % (birinci mertebe duvar kaybı). Ölçümler:
[docs/19 §6.3](docs/19_3D_DATA_PIPELINE.md). Yüksek mertebe (NGSolve p=3, eğri eleman) referans etiketleri
yol haritasında: [docs/23](docs/23_ROADMAP.md).

## Belgeler

[docs/README.md](docs/README.md): hangi belge neyi anlatır, hangileri tarihsel araştırma kaydıdır.

## Testler

```bash
pip install -r requirements-dev.txt -r requirements-web.txt
python -m pytest -q          # gmsh testleri sistem kütüphanelerini ister
ruff check .
```

2D model (GNOT, SpectralNO) ve H-alanı formülasyonu çıkarıldı; eski hâlleri `legacy-2d` (2D main) ve `legacy-h` (H dahil son 3D sürüm) dallarında duruyor.
