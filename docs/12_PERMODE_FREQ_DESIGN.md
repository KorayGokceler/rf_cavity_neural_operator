# 🔧 Per-Mode Frequency Prediction — Tasarım

> **Branch:** `feature/orthogonality-and-depth`
> **Durum:** Uygulama aşaması

---

## Sorun

Mevcut yapıda frekans tahmini **ayrı bir branch** ile yapılıyor:
```
Shared Trunk → freq_blocks (2 ayrı blok) → AttentionPool → freq_decoder → tek frekans
```

Bu, **mode-blind** bir tahmin: model hangi modu tahmin ettiğini bilmeden frekans çıkarıyor.
Ama fiziksel olarak her modun frekansı farklı ($f_0 < f_1 \leq f_2$).

## Çözüm

Frekansı **mode-specific branch'ten** çıkar. Her mod branch'inin çıktısını pooling'den geçirip kendi freq_head'inden decode et:

```
Shared Trunk
     │
     ├── Mode 0 Branch (2 blok) → field_head[0] → alan
     │                           → pool → freq_head[0] → frekans_0
     │
     ├── Mode 1 Branch (2 blok) → field_head[1] → alan
     │                           → pool → freq_head[1] → frekans_1
     │
     └── Mode 2 Branch (2 blok) → field_head[2] → alan
                                 → pool → freq_head[2] → frekans_2
```

## Değişiklikler

1. **`gnot.py`:** `freq_blocks` kaldır, `freq_heads` (per-mode) ekle, forward'da her mod branch'inden frekans çıkar
2. **`lightning_module.py`:** Frekans loss'u aynı kalır (zaten per-sample)
3. **`default.yaml`:** `predict_frequency: true` yap, `n_freq_layers` kaldırılabilir
4. **`train.py`:** `n_freq_layers` parametresini kaldır
