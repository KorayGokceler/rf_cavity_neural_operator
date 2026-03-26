# 2D RF Cavity Neural Operator Prediction

Bu repo, 2D RF Cavity için GNOT (Geometric Neural Operator Transformer) modelinin eğitim ve tahmin kodağacını içerir.

## Google Colab Üzerinde Kullanım
Google Colab'de kullanırken hücrelere `!` (ünlem) ekleyerek terminal komutlarını çalıştırabilirsiniz.

**1. Gereksinimlerin Kurulması:**
```bash
!pip install -r requirements.txt
```

**2. Veri Üretimi (H5 dosyası):**
(Eğer elinizde `rf_cavity_1000_dataset.h5` hazırsa bu adımı atlayabilirsiniz)
```bash
!python src/data_gen/dataset_generator.py
```

**3. H5 Verisini GNOT İçin İşleme (PKL Dönüşümü):**
```bash
!python convert.py --h5_filepath rf_cavity_1000_dataset.h5 --output_pkl data/gnot_dataset.pkl
```

**4. Eğitim:**
```bash
!python train.py
```
*(Tam eğitime geçmek için `train.py` içerisindeki `fast_dev_run=True` kısmını `False` olarak değiştirmeyi unutmayın!)*

