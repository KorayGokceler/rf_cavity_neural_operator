# RF Cavity Neural Operator

3B RF kavitelerin rezonans modlarını (frekans + elektrik/manyetik alan) ve kavite büyüklüklerini
(Q0, G, R/Q, R_sh, T, Epk/Eacc, Bpk/Eacc) geometriden doğrudan tahmin eden bir neural operator.
Model, PEC duvarlı Maxwell özdeğer problemini öğrenir; tahmin sırasında FEM çözümü gerekmez.

$$\nabla\times\nabla\times\mathbf E = k^2\,\mathbf E \ \text{ in } \Omega, \qquad \mathbf n\times\mathbf E = 0 \ \text{ on } \partial\Omega$$

- **Formülasyon:** E alanı (CST / HFSS gibi), eğri tet mesh üzerinde yüksek mertebe H(curl) (NGSolve
  HCurl p2) DOF'ları. PEC şartı yapısal: duvar DOF'ları tam 0. Kulplu kaviteler (spoke, HWR, DTL) dahil
  her kapalı PEC kavite çalışır.
- **Model:** `EigenspaceOperatorField` (model_type `field3d`) — kütle-farkında lineer attention, eleman
  quadrature noktalarında `m` vektör alanı üretir; sabit bir lineer harita bunları HCurl(p2) DOF'larına
  çevirir, gradyan çekirdeği projekte edilmiş Rayleigh–Ritz `K` mod (frekans + alan) çıkarır;
  $f = c\sqrt{\lambda}/(2\pi\,\text{scale})$. Ayrıntılar: [docs/28](docs/28_FIELD_MODEL.md),
  etiketler: [docs/29](docs/29_FIELD_LABELS.md).

## Akış

```
src/data_gen/dataset_generator_3d.py ──────────────► H5 parçaları ──► train.py ──► checkpoint
  (gmsh CAD → eğri netgen mesh, NGSolve p3 etiketleri,     (FieldDataset operatörleri      │
   uyarlamalı inceltme, p2 model uzayına izdüşüm)            öğe başına yeniden kurar)       ▼
                                                              scripts/eval_fidelity.py · web arayüzü
```

Alan etiketleri tek etiket türüdür; üreticinin eski `--labels n0` (Whitney N0) seçeneği kaldırıldı, `--labels`
gizli ve etkisizdir (yalnız `field` kabul eder). Gösterim ve QoI katmanındaki N0 kodu da kaldırıldı (docs/31 madde 15):
alan doğrudan HCurl p2 olarak çizilir (`src/viz/hcurl_display.py`), kavite büyüklükleri p2 alanlarından hesaplanır
(`highorder.mode_qoi`, `src/qoi/figures.py`).

- **Geometri aileleri** (`src/data_gen/cavity_shapes.py`): `elliptical` (TESLA tipi, 1–5 hücre), `reentrant`,
  `pillbox_pipes`, `ridged_box`, `composite`, `freeform`, `hwr`, `spoke`, `dtl`; dağılım dışı test için
  `box`, `coax_qw`, `pillbox_port`, `elliptical_long`, `junction`.
- **Düğüm özellikleri:** `x, y, z, dist_to_boundary, dir_bnd_x/y/z, node_volume, torsion`.
- **Etiketler:** her geometri için ilk `K` modun frekansı ve kavite büyüklükleri (p3 etiket mesh'inden;
  bakır duvar, β = 1, U = 1 J; [docs/24](docs/24_CAVITY_QOI.md)) ve model mesh'indeki p2 alan DOF'ları.

## Nerede çalıştırılır

| ortam | dosya | ne yapar |
|---|---|---|
| **Colab, tek hücre** | [`Colab_WebUI.ipynb`](Colab_WebUI.ipynb) | Drive + kurulum + web arayüzü (telefondan da): dataset üretimi, eğitim, tahmin, inceleme |
| TRUBA (SLURM) | [`cluster/truba/`](cluster/truba/README.md) | aile başına dataset üretimi ve eğitim (`run.sh`, `LABELS=field`) |
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
# 1) Alan etiketli FEM verisi (H5; docs/29)
python src/data_gen/dataset_generator_3d.py --n_total 100 --families elliptical reentrant \
    --h5_filename data/field/s0.h5
python scripts/validate_field_labels.py --max_ndof 300000     # analitik pillbox kontrolü (PASS yazmalı)

# 2) Eğitim (varsayılan config configs/field_3d.yaml; checkpoint'ler training_logs/<exp_name>/, ilerleme progress.json)
python train.py --override "dataset.data_path=data/field/*.h5" training.exp_name=small
python train.py ... --resume training_logs/small/last.ckpt          # devam

# 3) Değerlendirme (LF/HF sadakat, docs/34)
python scripts/eval_fidelity.py --help

# Web arayüzü (yerel ağ ya da --tunnel ile her yerden)
python scripts/serve_web.py --checkpoint training_logs/small --data data --gen_root data/ui --runs_root training_logs
```

## Repo haritası

```
src/data_gen/    geometri aileleri (cavity_shapes.py), üretici (dataset_generator_3d.py), eğri mesh + p3 çözücü
                 (highorder.py), uyarlamalı inceltme (adaptive.py), alan etiketleri (field_labels.py)
src/data/        field_dataset.py (field3d öğeleri + collate), sparse.py (blok-köşegen operatörler),
                 mesh_features.py (mesh yardımcıları, sınır topolojisi, modelin girdi öznitelikleri)
src/models/      EigenspaceOperatorField, H(curl) cebiri (hcurl.py), yapı taşları (layers.py)
src/training/    Lightning modülü (kayıplar, metrikler), checkpoint bulma, progress.json
src/qoi/         kavite büyüklükleri: sözleşme ve ölçekleme (figures.py), analitik referanslar
src/viz/         HCurl p2 alanını doğrudan çizme (hcurl_display.py), mesh örnekleme (meshsample.py), ParaView .vtu, yüzey geometrisi
src/service/     web API (FastAPI): tahmin, dataset gezgini, üretim ve eğitim işleri
web/             arayüz (React + vtk.js)
scripts/         doğrulama, değerlendirme, tahmin, sunucu başlatma
cluster/truba/   SLURM betikleri
configs/         field_3d.yaml
tests/           pytest (sentetik veri + küçük gmsh örnekleri)
```

## Doğruluk (etiketler)

Etiketler eğri mesh üzerinde uyarlamalı NGSolve HCurl p3 çözümüdür; frekans ve QoI bu etiket mesh'inden,
eğitim alanı p2 model uzayına izdüşümdür. Ölçümler ve maliyet: [docs/29](docs/29_FIELD_LABELS.md),
[docs/30](docs/30_ADAPTIVE_LABELS.md). Eski N0 etiketlerinin doğruluğu (tarihsel): [docs/19 §6.3](docs/19_3D_DATA_PIPELINE.md).

## Belgeler

[docs/README.md](docs/README.md): hangi belge neyi anlatır, hangileri tarihsel araştırma kaydıdır.

## Testler

```bash
pip install -r requirements-dev.txt -r requirements-web.txt
python -m pytest -q          # gmsh testleri sistem kütüphanelerini ister
ruff check .
vulture                      # ölü kod (ayarlar pyproject.toml [tool.vulture])
```

2D model (GNOT, SpectralNO), H-alanı formülasyonu ve N0 kenar modeli (`eigenspace3d`, PKL hattı) çıkarıldı;
ilk ikisinin eski hâlleri `legacy-2d` (2D main) ve `legacy-h` (H dahil son 3D sürüm) dallarında, N0 modeli
git geçmişinde duruyor.
