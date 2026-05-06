# 08 — Konfigürasyon Referansı

> **Dosyalar:** `configs/default.yaml`, `src/config.py`

---

## 🎯 Sistem

Config sistemi `src/config.py` tarafından yönetilir. `ConfigDict` sınıfı, YAML dosyasını attribute-style erişime dönüştürür:

```python
cfg = load_config("configs/default.yaml")
cfg.model.embed_dim      # → 256
cfg.training.batch_size  # → 32
```

CLI'dan override edebilirsin:
```bash
python train.py --config configs/default.yaml --override model.embed_dim=128 training.learning_rate=0.001
```

---

## 📋 Tüm Parametreler

### 1. Data Generation (`data_gen:`)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `h5_filename` | `rf_cavity_5000_dataset.h5` | Çıktı dosya adı |
| `n_total` | 5000 | Üretilecek toplam geometri sayısı |
| `n_plot` | 100 | Görselleştirilecek örnek sayısı |
| `generation_mode` | `random` | `random` veya `calibration` |
| `n_eigen_modes` | 3 | Çözülecek eigenmode sayısı (k) |
| `eigen_sigma` | 500.0 | Shift-invert sigma |
| `mesh_size_min` | 0.0012 | Sınır yakını mesh boyutu |
| `mesh_size_max` | 0.005 | Merkez mesh boyutu |
| `mesh_dist_min` | 0.002 | Mesh sıklaştırma başlangıcı |
| `mesh_dist_max` | 0.03 | Mesh sıklaştırma bitişi |
| `sharp_n_pts_range` | [7, 13] | Köşeli geometri köşe sayısı |
| `sharp_r_range` | [0.02, 0.046] | Köşeli geometri yarıçap aralığı |
| `smooth_base_r` | 0.035 | Yumuşak geometri baz yarıçap |
| `smooth_perturb` | 0.008 | Pertürbasyon genliği |
| `smooth_harmonics` | [2, 8] | Fourier harmonik aralığı |

> **Not:** FEM element tipi P2 olarak sabit bırakılmıştır. P1'e geçmek veri formatını kırar.

### 2. Data Conversion (`data_convert:`)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `h5_filepath` | `rf_cavity_5000_dataset.h5` | Girdi H5 dosyası |
| `output_path` | `data/gnot_dataset_5k.pkl` | Çıktı dosya yolu |
| `output_format` | `pkl` | `pkl` veya `h5` |
| `mode_indices` | [0, 1, 2] | Hangi modlar dahil? |
| `max_samples` | null | Sınırlama (null = tümü) |

### 3. Dataset (`dataset:`)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `data_path` | `data/gnot_dataset_5k.pkl` | İşlenmiş veri yolu |
| `train_ratio` | 0.8 | Eğitim oranı |
| `val_ratio` | 0.1 | Doğrulama oranı (test = 0.1) |
| `random_seed` | 42 | Geometri split seed'i |
| `max_nodes` | 1024 | Sub-sampling limiti (null = devre dışı) |
| `active_mode_index` | null | null = tüm modlar; 0/1/2 = tek mod eğitimi |

### 4. Model (`model:`)

| Parametre | Varsayılan | Açıklama | Etki |
|-----------|-----------|----------|------|
| `val_dim` | 8 | Feature boyutu | Feature sayısı değişirse güncelle |
| `grid_dim` | 2 | Koordinat boyutu (2D) | 3D için 3 yapılır |
| `embed_dim` | 256 | Hidden dimension | ↑ = daha güçlü ama yavaş |
| `n_shared_layers` | 6 | Shared trunk derinliği | ↑ = daha iyi geometri temsili |
| `n_mode_layers` | 1 | Mode-specific derinlik | ↑ = daha iyi mod ayrımı |
| `n_field_head_layers` | 3 | Tahmin kafası derinliği | ↑ = daha güçlü decoder |
| `n_heads` | 8 | Attention head sayısı | embed_dim / n_heads tamsayı olmalı |
| `num_experts` | 4 | MoE expert sayısı | 4-8 arası önerilir |
| `num_field_modes` | 3 | Toplam mod sayısı | Veriyle uyumlu olmalı |
| `predict_frequency` | true | Frekans tahmini | false = sadece alana odaklan |
| `rff_dim` | 64 | RFF çıktı boyutu (çift olmalı) | ↑ = daha zengin koordinat temsili |
| `rff_length_scale` | 0.1 | Gaussian kernel uzunluk ölçeği | Normalize koordinatlar için 0.05–0.2 |
| `dropout` | 0.0 | Regularization | 0.1-0.3 denenebilir |
| `use_checkpoint` | false | Gradient checkpointing | true = VRAM↓, hız↓ |

### 5. Training (`training:`)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `learning_rate` | 1e-4 | Başlangıç öğrenme oranı |
| `weight_decay` | 0.0 | L2 regularization (bilinçli olarak 0) |
| `batch_size` | 32 | Batch boyutu |
| `max_epochs` | 300 | Maksimum epoch |
| `gradient_clip_val` | 1.0 | Gradyan kırpma |
| `freq_weight` | 0.5 | Frekans loss ağırlığı (α) |
| `smoothness_weight` | 0.0 | Boundary loss ağırlığı (λ); 0 = kapalı |
| `mode_loss_weights` | [1.0, 2.0, 2.0] | Mode-specific ağırlıklar |
| `lr_mode_specific` | null | Modlara özel öğrenme oranları (null = base_lr kullan) |
| `lr_freq_heads` | null | Frekans başlıklarına özel LR (null = base_lr) |
| `scheduler` | `custom_cosine` | LR scheduler tipi |
| `cosine_eta_min` | 1e-6 | custom_cosine / cosine'de minimum LR |
| `patience` | 50 | Early stopping sabırlılığı |
| `num_workers` | 4 | DataLoader worker sayısı |
| `strategy` | `auto` | `auto` veya `ddp` |
| `exp_name` | `gnot_5k_v1` | Deney ismi |
| `viz_every_n_epochs` | 10 | Görselleştirme sıklığı |
| `permutation_invariant_dipole` | true | Mode 1/2 permütasyonu denenirse loss düşüyor mu? |

> **Scheduler seçenekleri:** `custom_cosine` (varsayılan), `cosine`, `onecycle`, `reducelr`

### 6. Inference (`inference:`)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `checkpoint_path` | null | Checkpoint yolu |
| `output_dir` | `inference_results` | Çıktı dizini |
| `num_visualize` | 5 | Görselleştirilecek örnek |

---

## 📁 Diğer Config Dosyaları

| Dosya | Amaç |
|-------|------|
| `configs/kaggle_2gpu.yaml` | Kaggle'da 2 GPU ile eğitim |
| `configs/mode1_isolated.yaml` | Sadece Mode 1'i eğitmek için |
| `configs/ablation/*.yaml` | Feature ablation deneyleri |

---

## ⚡ Hızlı Deneyler İçin Override Örnekleri

```bash
# Hızlı test (1 iterasyon)
python train.py --fast_dev_run

# Küçük model
python train.py --override model.embed_dim=64 model.n_shared_layers=2

# Sadece Mode 0 eğitimi
python train.py --config configs/mode1_isolated.yaml

# Farklı scheduler
python train.py --override training.scheduler=cosine training.cosine_eta_min=1e-7
```

---

## 🔗 Bağlantılar

- Model parametreleri: [[04_MODEL_ARCHITECTURE]]
- Eğitim sistemi: [[05_TRAINING_SYSTEM]]

#config #yaml #hiperparametre
