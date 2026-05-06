# 2D RF Cavity Neural Operator Prediction

Bu repo, 2D RF Cavity için GNOT (Geometric Neural Operator Transformer) modelinin eğitim ve tahmin kodağacını içerir.

## Google Colab Üzerinde Kullanım
Google Colab'de kullanırken hücrelere `!` (ünlem) ekleyerek terminal komutlarını çalıştırabilirsiniz.

**1. Gereksinimlerin Kurulması:**
```bash
!pip install -r requirements.txt
```

**2. Tam Pipeline (Veri Üretimi → Dönüşüm → Eğitim):**
```bash
!python run_pipeline.py --config configs/default.yaml
```

Ya da adımları ayrı ayrı çalıştırabilirsiniz:

**2a. Veri Üretimi (H5 dosyası):**
```bash
!python src/data_gen/dataset_generator.py
```
*(Elinizde hazır bir H5 dosyası varsa bu adımı atlayabilirsiniz)*

**2b. H5 Verisini GNOT İçin İşleme (PKL Dönüşümü):**
```bash
!python convert.py --h5_filepath rf_cavity_5000_dataset.h5 --output_path data/gnot_dataset_5k.pkl
```

**3. Eğitim:**
```bash
!python train.py --config configs/default.yaml
```
*(Hızlı test için `configs/default.yaml` içinde `training.fast_dev_run: true` yapın)*
