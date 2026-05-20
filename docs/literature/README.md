# Literatür Notları — RF Cavity Eigenmode Neural Operator

Bu klasör, projenin teknik bileşenleriyle doğrudan ilişkili **20 seçili makalenin**
proje-özel okuma notlarını içerir. Her not; makalenin çekirdek fikri + **bizim
kodumuzun hangi parçasına nasıl uygulanabileceği** + risk/uyarı şeklinde yazıldı.

> **Kod referansları hakkında:** Notlardaki `ot_match`, `soft_procrustes_loss`,
> `grassmannian_loss`, per-mode `field_heads`/`mode_field_blocks` gibi semboller
> deneysel loss/mimari branch'lerine aittir
> (`claude/ot-soft-grassmannian-loss`, `claude/no-shared-blocks-per-mode-decoder`).
> Notlar mimari-niyet seviyesinde yazıldı; branch fark etmez.

## Klasör yapısı

| Dosya | Tema | Makale sayısı |
|---|---|---|
| [`01-architecture.md`](01-architecture.md) | Neural operator omurgası (GNOT, Transolver, FNO, DeepONet) | 4 |
| [`02-multimode-eigenvalue.md`](02-multimode-eigenvalue.md) | Multi-eigenmode tahmini, ortogonalite, EM eigenproblem | 5 |
| [`03-degeneracy-gauge-subspace.md`](03-degeneracy-gauge-subspace.md) | Degeneracy / O(d) gauge / subspace loss + stabil SVD | 6 |
| [`04-set-prediction-ot-qroadmap.md`](04-set-prediction-ot-qroadmap.md) | Set prediction, OT matching, Q-faktör yol haritası | 5 |

## Projenin bileşeni → ilgili makale haritası

| Proje bileşeni (dosya) | İlgili notlar |
|---|---|
| GNOT omurga, encoder, per-mode decoder (`src/models/gnot.py`) | 01 (tümü), 02 (FieldTNN, PhC bands) |
| Mode collapse / ortogonalite guard (`loss_ortho`, `_compute_loss`) | 02 (PINN-eig ortho-loss, SpectralNet) |
| `ot_match` — Hungarian matching (`src/training/lightning_module.py`) | 04 (DETR, Sinkhorn, aMCL, PIT) |
| `soft_procrustes_loss` / SVD stabilizasyonu | 03 (Robust SVD, BP-Eigendecomp, SVD-rotation) |
| `grassmannian_loss` / projector P=QQᵀ | 03 (BasisNet, Subspace Regression, GrNet) |
| Uzun vade: PEC→lossy, Q/G tahmini | 04 (High-Q photonic), 02 (Helmholtz/FieldTNN) |

## Öncelikli okuma sırası (sınırlı zaman)

1. **GNOT** `2302.14376` — mimari temel → `01`
2. **BasisNet** `2202.13013` — degeneracy'yi mimaride çöz → `03`
3. **Subspace Regression** `2509.23249` — projector-loss teorik gerekçe → `03`
4. **Robust Differentiable SVD** `2104.03821` + **BP-Friendly Eigendecomp** `1906.09023` — `soft_procrustes_loss` sertleştir → `03`
5. **DETR** `2005.12872` + **aMCL** `2407.15580` — OT matching'i annealed soft→hard → `04`
6. **2D Photonic Crystal bands** `2411.06063` — en yakın multi-mode EM analog → `02`
7. **High-Q Photonic DL** `2105.03001` — Q yol haritası → `04`

## En somut 5 aksiyon

1. `soft_procrustes_loss` içindeki SVD gradyanını near-degenerate'te stabilize et
   (Robust SVD / BP-Eigendecomp — `03`).
2. `ot_match`'i annealed soft→hard yap (Sinkhorn başlangıç → aMCL schedule — `04`).
3. Mimari O(d)-invariance düşün (BasisNet projector head — `03`).
4. Ortogonalite auxiliary loss güçlendir (PINN ortho-loss / SpectralNet — `02`).
5. Bağımsız gauge-invariant validasyon metriği ekle (DeepH tarzı — `03`).

> Kaynak: 4 paralel literatür-tarama ajanının ~55 makalelik taramasından
> seçilen en ilişkili 20 makale. Doğrulanamayan arXiv ID'leri ilgili notta
> ⚠️ ile işaretlidir.
