# 03 — Dataset & DataLoader

> **Dosya:** `src/data/dataset.py`  
> **Sınıf:** `GNOTDataset(Dataset)`, `gnot_collate_fn`  
> **Destek:** Hem `.pkl` hem `.h5` formatlarını okuyabilir

---

## 🎯 Ne Yapıyor?

İşlenmiş veriyi (`.pkl` veya `.h5`) PyTorch `Dataset` formatına dönüştürüyor. Geometri bazlı train/val/test split yapıyor. Farklı boyutlardaki meshleri aynı batch'e sığdırmak için padding ve maskeleme sağlıyor.

---

## 🔑 Kritik Tasarım Kararları

### 1. Geometri Bazlı Split (Data Leakage Koruması)

```python
# Geometrileri (sample'ları DEĞİL) shuffle et
perm_geoms = np.random.permutation(unique_geoms)

train_geoms = perm_geoms[:n_train_geoms]      # %80
val_geoms   = perm_geoms[n_train:n_train+n_val] # %10
test_geoms  = perm_geoms[n_train+n_val:]        # %10
```

**NEDEN ÖNEMLİ:** Her geometrinin 3 modu var. Eğer aynı geometrinin Mode 0'ı train'de, Mode 1'i val'de olursa → **data leakage!** Çünkü aynı geometri bilgisi (`Input_funcs`) iki set arasında paylaşılır.

Bu yapı, geometrileri atomik birim olarak ayırarak sızıntıyı önler:
- Geometri 42'nin tüm modları (Mode 0, 1, 2) → train
- Geometri 77'nin tüm modları → val
- Hiçbir geometri iki farklı split'te bulunmaz

### 2. Node Sub-Sampling (VRAM Optimizasyonu)

```python
if self.max_nodes is not None and n_nodes > self.max_nodes:
    rand_idx = torch.randperm(n_nodes)[:self.max_nodes]
    x = x[rand_idx]
    input_features = input_features[rand_idx]
    y_field = y_field[rand_idx]
```

Config'te `max_nodes: 2048` ayarlı. Bazı meshler 4000+ node içerebilir. Bu, padding yüzünden VRAM patlamasına neden olur:
- 3000 node'lu 1 mesh + 1500 node'lu 15 mesh = 16×3000 = 48000 boyutlu tensor
- Sub-sampling ile: 16×2048 = 32768 → *%32 VRAM tasarrufu*

**RNG Dikkat:** `torch.randperm` kullanılıyor (`np.random` değil). Çünkü DataLoader worker'ları aynı numpy seed'i paylaşabilir, ama PyTorch her worker'a ayrı seed atar.

### 3. Frekans Normalizasyonu

```python
norm_freq = (raw_freq - stats['mean']) / stats['std']  # z-score
```

Frekanslar GHz cinsinden (~2-15 GHz arası). Loss fonksiyonunda MSE kullanılacağı için, büyük değerler baskın hale gelir. Z-score ile tüm frekanslar ~(-2, +2) aralığına çekilir.

### 4. Feature Ablation Desteği

```python
if self.feature_indices is not None:
    input_features = input_features[:, self.feature_indices]
```

Ablation study'de sadece belirli feature'ları kullanmak için. Örneğin `feature_indices=[0,1]` sadece `(x, y)` kullanır.

---

## 📦 Collate Function (Batching)

Farklı boyutlardaki meshler tek bir batch'te birleştirilirken:

```python
def gnot_collate_fn(batch):
    X_padded = pad_sequence(batch_x, batch_first=True, padding_value=0.0)
    # ...
    # Mask oluştur: gerçek noktalar True, padding False
    for i, l in enumerate(lengths):
        mask[i, :l] = True
```

**Çıktı:**
```
{
    'X':          [B, max_N, 2]     → Padded koordinatlar
    'Input_funcs':[B, max_N, 8]     → Padded features
    'Y_field':    [B, max_N, 1]     → Padded hedef alan
    'Y_freq':     [B, 1]            → Normalize frekans
    'Theta_in':   [B, 1]            → Mode indeksi (0, 1, 2)
    'Mask':       [B, max_N]        → Boolean padding maskesi
    'elements':   list of [M_i, 3]  → Her mesh'in bağlantıları (pad yapılmaz)
}
```

Mask'ın kullanım yerleri:
- **Attention:** Padding token'larına attend etmeyi engeller
- **Loss:** Sadece gerçek node'lardaki hata hesaplanır
- **Metrics:** Relative L2 sadece valid node'larla hesaplanır

---

## 📂 H5 Format Desteği

`.h5` formatı büyük datasetler için bellek dostu bir alternatif sunar: veri ihtiyaç oldukça diskten okunur, tamamı RAM'e yüklenmez.

```python
def _get_h5_handle(self):
    worker_id = worker_info.id if worker_info else 'main'
    attr = f'_h5_handle_{worker_id}'
    if not hasattr(self, attr):
        setattr(self, attr, h5py.File(self.data_path, 'r', swmr=True))
    return getattr(self, attr)
```

**SWMR (Single-Writer Multiple-Reader):** Birden fazla DataLoader worker'ı aynı anda dosyayı güvenle okuyabilir.

Her worker kendi H5 handle'ını tutar — thread-safety sağlar.

---

## ⚠️ Geliştirme Önerileri

1. **Dynamic Batching:** Benzer mesh boyutlarını aynı batch'e koyarak padding israfını minimize etme. → `BucketBatchSampler` kullanılabilir.
2. **Chunk-Based Loading:** Çok büyük datasetlerde `mmap_mode` veya chunk tabanlı lazy loading.
3. **Augmentation:** Geometrilere dönüşümler (rotasyon, aynalama) uygulayarak dataset çeşitliliğini artırma. **Dikkat:** PCA-based features rotasyonla bozulur, onlar da yeniden hesaplanmalı.
4. **Weighted Sampling:** Nadir geometri tiplerini (çok küçük, çok büyük) daha sık örnekleyerek mode collapse'i önleme.

---

## 🔗 Bağlantılar

- Önceki adım: [[02_FEATURE_ENGINEERING]]
- Sonraki adım: [[04_MODEL_ARCHITECTURE]]
- Config ayarları: [[08_CONFIG_REFERENCE]]

#dataset #dataloader #padding #split #vram
