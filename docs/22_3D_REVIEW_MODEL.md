# 22 — 3B İnceleme: Model, Eğitim ve Entegrasyon

> Kapsam: `src/models/eigenspace_operator_3d.py`, `src/data/dataset_3d.py`, `src/training/lightning_module.py`
> (3B yolu, `_span_terms` yeniden düzenlemesi, `_jacobi_cholesky`), `train.py`, `scripts/eval_3d.py`,
> `configs/eigenspace_3d.yaml`, `Colab_Maxwell3D.ipynb`, `tests/test_eigenspace_3d.py`, ve model/kayıpların
> kullandığı `hcurl.py` arayüzleri. `hcurl.py`'nin içindeki fizik matematiği fizik incelemesine aittir (docs/21).
> İncelenen commit: `50ce20e` (`claude/3d-maxwell`), temel: `4bcd0e4`. Dal: `review-3d-model`.
> Veri: `smoke3d.pkl` (30 geometri, mesh 0.10, Ne 6.8–9.5k, Nv 1.1–1.7k, 6 mod; pillbox / axisym_cell / blob).
> Donanım: CPU, 4 çekirdek, torch 2.14, Lightning 2.6.6.

## 1. Özet tablo

Satır numaraları `50ce20e`'ye göredir.

| # | Önem | Yer | Bulgu | Durum |
|---|---|---|---|---|
| M1 | **major** | `lightning_module.py:975-991` (`_compute_loss_3d`) | Tüm girdileri NaN olan (bölünmüş küme) bir batch `val/field_rel_l2 = NaN` logluyordu. Epoch ortalaması NaN olur, EarlyStopping eğitimi 1. epoch'ta durdurur, ModelCheckpoint hiç iyileşmez. Mod başına anahtarlar koşullu loglandığı için DDP'de rank'ler arasında anahtar kümesi farklılaşabiliyordu. | **Düzeltildi** + regresyon testi |
| m1 | minor | `lightning_module.py:357` (`_jacobi_cholesky`) | "Ölü kolon" eşiği (köşegen ≤ 1e-12·max) 2D'yi de etkiler. Normu en büyük kolonun < 1e-6'sı olan geçerli bir kolon artık atılıyor (ölçek değişmezliği bozuluyor). Gerçek 2D batch'lerinde sonuç bit düzeyinde aynı. | Rapor |
| m2 | minor | `lightning_module.py:398` (`ritz_compliance`), 3B kullanımı | Kompliyans, $A_V$'nin tam çekirdek yönlerinde (iki kolonun diverjanssız kısımları aynı) ridge'e duyarlı. `kp_tol=1e-8` ve kaba gradyanlarla kurulmuş bir örnekte %10.6 hata ölçüldü. Genel ağ çıktısında tetiklenmez. | Rapor |
| m3 | minor | `scripts/eval_3d.py:69,87` ile `lightning_module.py` | `eval_3d` `rel_l2`'yi satır bazında nanmean alıp satırların ortalaması olarak hesaplıyor; Lightning metriği (düzeltmeden sonra) geçerli tüm (geometri, mod) girdilerinin havuz ortalaması. Aynı test kümesinde 0.685 ile 0.730 çıktı. | Rapor |
| m4 | minor | `configs/eigenspace_3d.yaml:88`, `_clusters_3d` | `near_deg_rel_threshold: 0.02` zincirleme kümelemeyle modların çoğunu kümeye sokuyor (smoke train: 8/9 geometri, 34/54 mod). K=6'da girdilerin %12'si (22/180) "bölünmüş" diye metrikten ve checkpoint monitöründen çıkıyor. K ≤ 2'de tüm satırın NaN olduğu geometriler var (K=1: 6/30). | Rapor (tasarım) |
| m5 | minor | `hcurl.py:45` (`spmm`), `dataset_3d.py:193` | Blok-köşegen operatörler COO. CPU'da CSR SpMM, $K_p$ için 3.3×, $M$ için (dönüşüm dahil) 7.9× daha hızlı. CUDA'da COO `sparse.mm` her çağrıda COO→CSR dönüşümü yapar (adım başına ~200 CG çarpımı). `collate`'teki `.coalesce()` gereksiz (bloklar ayrık ve CSR sıralı): B=8'de 365 ms. | Rapor (performans) |
| m6 | minor | `dataset_3d.py:82`, `train.py:113` | train/val/test veri kümelerinin her biri PKL'in tamamını (operatörler dahil tüm `geometry_pool`) belleğe yüklüyor: 3× RAM. `STORE_OPERATORS` sınırında (8 GB) bu 24 GB demek. | Rapor |
| m7 | minor | `Colab_Maxwell3D.ipynb` hücre 2 | `STORE_OPERATORS = N_TOTAL·4.7e-3 < 8` mesh boyutunu hesaba katmıyor. 4.7 MB/geometri, mesh 0.10 için geçerli. 0.07'de (docs/19'un hedefi) ≈ (0.10/0.07)³ ≈ 2.9× büyük olur. Örneğin N=1500'de tahmin 7 GB, gerçekte ≈ 20 GB (×3 veri kümesi). | Rapor |
| m8 | minor | `dataset_3d.py:136`, config `cache_operators: true` | `--no_operators` PKL ile `cache_operators=true` olunca, kalıcı her DataLoader işçisi zamanla bütün train operatörlerini kendi kopyasında tutar (ölçüldü: işçi RSS'i epoch başına büyüyor). N_workers × 4.7 MB × N_geo. Notebook bunu doğru biçimde `false` yapıyor; config yorumu da uyarıyor. | Rapor |
| m9 | minor | `lightning_module.py:953`, config `freq_weight: 0.1` | Başlangıçta frekans terimi (z-MSE ≈ 100) toplam kaybın ≈ %65'i. Gradyan normunda: freq 405, span_M 0.85, selfsup 2.1. `gradient_clip_val=1.0` ile ilk adımlarda span gradyanı neredeyse tamamen bastırılıyor. | Rapor (tasarım) |
| n1 | nit | `dataset_3d.py:107` | Bilinmeyen `split` adı sessizce test kümesine düşüyor. | Rapor |
| n2 | nit | `train.py:160` | `feature_indices` çıktısı 3B için 2D özellik adlarını (`GNOTDataset.FEATURE_NAMES`) yazıyor. | Rapor |
| n3 | nit | `lightning_module.py:962` | Metrik döngüsü batch üzerinde Python ile dönüyor; örnek başına `detect_clusters(...).tolist()` GPU'da bir D2H senkronu demek (B=8'de adım başına 8 senkron). | Rapor |
| n4 | nit | `hcurl.py:97` | `KpSolve.last_iters` sınıf düzeyinde global; ileri ve geri çözümü ayırmıyor, çoklu modelde karışır. | Rapor |

Fizik incelemesinin `hcurl.py` bulgularının (B1–B4) model/eğitim yolunda **tetiklenmediği** doğrulandı (§4.6).

## 2. Bulgular (ayrıntı)

### M1 — NaN, logları, checkpoint'i ve EarlyStopping'i zehirliyordu (major, düzeltildi)

**Sorun.** `_compute_loss_3d`, bölünmüş kümelerin girdilerini NaN yapıp `rl.nanmean()` logluyordu, `batch_size=B` ile. Bir batch'te geçerli girdi hiç yoksa değer NaN oluyordu. Lightning'in epoch ortalaması $\sum v_b n_b/\sum n_b$ olduğundan tek bir NaN adım bütün epoch'u NaN yapar. Ardından:
- `EarlyStopping(check_finite=True)` "not finite" deyip eğitimi durdurur;
- `ModelCheckpoint` NaN'ı +∞ sayar, en iyi skor hiç güncellenmez ve dosya adı `val_rel_l2=nan` olur;
- `scheduler: reducelr` seçilirse ReduceLROnPlateau her `patience` epoch'ta LR'yi düşürür.

Mod başına anahtarlar yalnız "tümü NaN değilse" loglanıyordu. `sync_dist=True` ile DDP'de rank'ler farklı anahtar kümeleri toplayabilir, bu da kilitlenme riski demek.

**Kanıt** (`num_field_modes=1`, `batch_size=1`, smoke verisi, `max_epochs=3`):
```
GLOBAL VAL │ Rel L2: nan          (epoch 0)
→ fit epoch 0'dan sonra bitti; last.ckpt: epoch 0,
  EarlyStopping {'stopped_epoch': 0, 'best_score': inf}, ModelCheckpoint {'best_model_score': inf}
  dosya: best-epoch=00-val_rel_l2=nan.ckpt
```
Smoke verisinde K=1'de 6/30, K=2'de 3/30 geometrinin bütün çıktıları bölünmüş bir kümede. K=6'da tamamı NaN satır yok, ama bölünmüş girdi oranı %12.

**Düzeltme** (`lightning_module.py`, `_compute_loss_3d`): NaN girdiler değerle değil **ağırlıkla** dışlanıyor. Her anahtar her adımda loglanıyor (DDP'de tutarlı). Değer, geçerli girdilerin ortalaması; `batch_size` ise geçerli girdi sayısı (0 ise katkı yok). Böylece epoch değeri, geçerli tüm (geometri, mod) girdilerinin havuz ortalaması olur. Düzeltmeden sonra aynı koşu 3 epoch'u tamamlıyor: val 1.0600 → 1.0122 → 0.9876, `best-epoch=02-val_rel_l2=0.9876.ckpt`.

**Regresyon testi:** `tests/test_eigenspace_3d.py::test_split_clusters_do_not_poison_logged_metrics`. Eski kodda başarısız oluyor (`batch_size` = B, geçerli sayı değil; tümü NaN batch'te değer NaN).

### m1 — `_jacobi_cholesky` "ölü kolon" eşiği 2D'yi de değiştiriyor (minor)

Eşik `diag > 1e-12·max(diag)`. Kolon normu oranı olarak bu 1e-6'ya karşılık gelir. Eski kod Jacobi ölçeklemesiyle bu kolonları da kullanıyordu; `span_residual` kolon ölçeğinden tamamen bağımsızdı.
```
kolon ölçeği   span_residual(t0)  eski       yeni
1e-3 / 1e-5                        1.0e-09    1.0e-09
1e-6                               1.0e-09    9.9e-01
1e-7                               1.0e-09    9.8e-01
```
Gerçek 2D batch'lerinde (dataset_40, 4 geometri; eigenspace×3 varyant, spectral_no, spectral+span, gnot) kayıp, gradyan ve bütün loglar **bit düzeyinde aynı** (§4.3). Ağ çıktısında kolon normları 1e6 kat ayrışmadığı için pratikte tetiklenmiyor. Öneri: eşiği yalnız 3B'ye uygulamak ya da (daha doğrusu) göreli kütleye göre tanımlamak, yani $\mu_j=M_{div,jj}/M_{V,jj}$.

### m2 — Kompliyansın ridge duyarlılığı (minor)

`ritz_compliance(M_div, A_V)`, $A_V$'yi Jacobi ölçekleyip `ridge=1e-9` ekliyor. Kurulan örnek: $V=[t_0+g,\ t_0-g,\ t_1..t_5]$, $g$ saf gradyan, $\|g\|_M=30$. Bu durumda $A_V$ tam tekildir. O yönde $M_{div}$ yalnız CG hatasını taşır ve katkısı ≈ hata/ridge olur. Ölçüm (temiz değer 0.598144):
```
kp_tol=1e-8  kaba φ: 0.534975 (−%10.6)   düzgün φ: 0.598174
kp_tol=1e-12 kaba φ: 0.598174            düzgün φ: 0.598114
```
Ağın çıktısı genel konumda olduğundan bu durum tetiklenmez: başlangıçta ve 5 epoch sonra min μ 5.9e-3 / 1.1e-2, bütün yönler tutuluyor (§4.6). Öneri: kompliyansı `projected_eigh`'in tuttuğu yönler üzerinde hesaplamak, ya da ridge'i $M_V$ ile beyazlatılmış pencil'e koymak.

### m3 — `eval_3d` ile Lightning'in `field_rel_l2` tanımları farklı (minor)

Aynı checkpoint ve test kümesi (3 geometri, NaN desenleri `[5]`, `[3,4,5]`, `[5]`): `eval_3d` satır ortalamalarının ortalaması 0.6853, havuz ortalaması 0.7300 (Lightning `test/field_rel_l2`). Öneri: `summarize` içinde `rel_l2`'yi bütün `rel_l2_k` girdilerinin nanmean'i olarak hesaplamak.

### m4 — Kümeleme eşiği ve dışlanan girdiler (minor, tasarım)

`near_deg_rel_threshold=0.02` ardışık ≤ %2 aralıkları zincirliyor. Smoke verisinde, K'ye göre bölünmüş küme içeren geometri / dışlanan girdi / tümü NaN satır:

| K | bölünmüş kümeli geometri | dışlanan girdi | tümü NaN satır |
|---|---|---|---|
| 6 | 14/30 | 22/180 | 0 |
| 4 | 13/30 | 13/120 | 0 |
| 2 | 19/30 | 22/60 | 3 |
| 1 | 6/30 | 6/30 | 6 |

Checkpoint monitörü, en zor girdileri (son moda kadar süren yakın-dejenere çiftler) sistematik olarak görmüyor. Öneri: monitöre ek olarak `span_rel_l2`'yi (NaN'sız, bütün modlar) izlemek, ya da bölünmüş kümeyi `FreqNext`'e kadar alt uzay hatası olarak saymak.

### m5 — Seyrek çarpımlar COO (minor, performans)

B=8, Ne=7948, Nv=1393; nnz(M)=836k, nnz(Kp)=124k:
```
K_p spmm (24 kolon):  COO 5.41 ms   CSR 1.65 ms
M   spmm (24 kolon):  COO 40.1 ms   CSR (dönüşüm dahil) 5.09 ms
collate (B=8): 365 ms, çoğu .coalesce()
```
Öneri: `maxwell3d_collate` içinde `is_coalesced=True` ile kurup `to_sparse_csr()`'ye çevirmek (torch 2.x'te CSR @ dense, dense argümana göre autograd destekliyor). Alternatif: `hcurl` içinde her batch için tek dönüşüm yapmak.

### m6, m7, m8 — CPU RAM (minor)

- smoke PKL 140 MB → tek `Maxwell3DDataset` süreci 692 MB RSS (torch dahil), `--no_operators` ile 570 MB. `train.py` üç veri kümesi kurduğu için PKL üç kez açılıyor. Öneri: PKL'i bir kez yükleyip `geometry_pool`'u split'e göre filtreleyerek paylaşmak.
- Notebook'taki heuristik `N_TOTAL·4.7e-3·(0.10/MESH_SIZE)**3 < 8` biçiminde olmalı, ideal olarak 3 veri kümesi de hesaba katılmalı.
- `cache_operators=true` + `--no_operators` + 2 kalıcı işçi (30 geometri): işçi RSS'i 495 → 529 MB (4 epoch). Önbellek her işçide ayrı tutuluyor. Önbellek zorunluysa ana süreçte, fork'tan önce kurulmalı.

### m9 — Başlangıçta frekans terimi baskın (minor, tasarım)

Başlangıçta ($d=32$, m=12) min θ₀/λ₀ = 9, yani f yaklaşık 3× büyük. $d=32$, m=16'da freq z-MSE ≈ 100. Terim başına gradyan normu:
```
span_M 0.85   span_A 0.14   selfsup 2.1   ortho 0.28   freq 405
```
`freq_weight·freq` toplam kaybın ≈ 10/15.4'ü. Clip (1.0) sonrası ilk adımlar neredeyse yalnız frekansı düşürüyor. Eğitim yine de ilerliyor (§4.4). Öneri: log-frekans MSE kullanmak, ya da `freq_weight`'i ilk birkaç epoch 0'dan artırmak (warmup).

## 3. Doğrulanan (doğru bulunan) noktalar

| Kontrol | Sonuç |
|---|---|
| Hedefler $M$-ortonormal | max $\lvert T^\top MT-I\rvert$ = 3.8e-9 (30 geometri) |
| Hedefler gradyanlara $M$-dik | $\max \lVert G^\top Mt\rVert/\lVert Mt\rVert$ = 5.9e-8 (float32 saklama düzeyi) |
| Hedefler özvektör | max göreli $\lVert Kt-\lambda Mt\rVert$ = 1.1e-5 |
| Fizik frekans formülü | $c\sqrt{\lambda_{RQ}}/(2\pi\,\mathrm{scale})$ saklı GHz'e max göreli 1.4e-7 ile uyuyor |
| O(3) artırma: operatörler | Döndürülmüş mesh'te yeniden kurulan M, K, G, Kp: göreli fark ≤ 2.4e-8 (float32 X); yansımada tam 0. $Y$ döndürülmüş operatörlerin de özvektörü (artık 2.5e-6). Yani $Y$'nin değişmemesi doğru; yansımadaki işaret, işaretten bağımsız kayıplarca yok sayılıyor. |
| O(3) artırma: özellikler | ‖X‖, ‖dir‖, dir·X korunuyor (≤ 1.8e-7); `X == F[:, :3]`; dist, hacim ve torsion sütunları değişmiyor |
| Kenar yönü eşdeğerliği (gerçek veri) | Kenarların 1/3'ü ters çevrildi: basis tam $\pm$ (fark 0.0), λ farkı 0, kayıp aynı (13.319497) |
| Heterojen batch = tek tek | Nv 1161/1393/1403, Ne 6806/7948/7914: basis 3.8e-7, λ 1.2e-7, alan 1.0e-6, Z 5.1e-8 göreli fark; kayıp 15.404573 ile 15.404574; dolgu satırları tam 0 |
| Örnekler arası sızıntı | Örnek 2'nin girdisi değiştirildi; örnek 0 ve 1'in çıktısı bit düzeyinde aynı |
| Gradyan akışı | 5 terimin her biri 44/44 parametreye sonlu gradyan veriyor, sessiz detach yok |
| Uçtan uca türev (FD) | Genel ve tamamen diverjanssız (μ=1) tabanlarda span ve selfsup 5e-9, θ 1.3e-9 / 2.4e-3 (broadening) |
| `inference_mode` + özel `KpSolve` | val/test `inference_mode=True` ile sorunsuz |
| bf16 autocast | basis float32, M_div float64, sonlu; hcurl autocast dışında |
| ModelCheckpoint/EarlyStopping anahtarları | `val/field_rel_l2` 3B'de loglanıyor; checkpoint adı ve `resolve_checkpoint` çalışıyor |
| `freq_stats` akışı | train.py → `GNOTLightning.freq_stats` → model; eval PKL istatistiğini kullanıyor. Frekans λ'dan hesaplandığı için z-skoru tutarlı. |
| Determinizm | Aynı seed, 2 epoch, `num_workers=2`, artırma açık, `--no_operators`: iki koşunun ağırlıkları arasında max fark **0.0** |
| Saklı ve yeniden kurulan operatörler | Aynı seed'de val eğrileri özdeş (1.0801, 1.0727) |

## 4. Koşulan testler ve koşular

### 4.1 Test takımı ve lint
`python -m pytest -q`: temelde 226 geçti; düzeltmeden sonra 227 geçti. `ruff check .`: temiz.

### 4.2 Eğitim döngüsü (3B)
- `train.py --config configs/eigenspace_3d.yaml --fast_dev_run` (config olduğu gibi, 128×4, m=24): 16 s, sorunsuz.
- 3 epoch (d=32, L=2, m=12, `num_workers=2`): 51 s; test/field_rel_l2 0.80.
- `last.ckpt`'den devam, `max_epochs=5`: bütün durum geri yükleniyor, val 1.0713 → 1.0665 → 1.0569.
- `cache_operators` true/false × saklı/`--no_operators` × `num_workers` 0/2: hepsi çalışıyor. Seyrek COO tensörleri işçi kuyruğundan geçiyor (yalnız "sparse invariant checks" uyarısı çıkıyor).
- Öğrenme (d=64, L=2, m=24, lr 1e-3, 25 epoch, 24 train geometrisi): bkz. §4.4.

### 4.3 2D regresyonları (`4bcd0e4` ile karşılaştırma)
Aynı seed ve batch (dataset_40, ilk 4 train geometrisi) üzerinde ileri geçiş + kayıp + geri geçiş:

| Yapılandırma | $\lvert\Delta$kayıp$\rvert$ | max $\lvert\Delta$grad$\rvert$ | loglar |
|---|---|---|---|
| eigenspace, span both, compliance | 0 | 0 | aynı |
| eigenspace, mass, logdet, ritz_field 0.5 | 0 | 0 | aynı |
| eigenspace, energy, span_root=False | 0 | 0 | aynı |
| spectral_no | 0 | 0 | aynı |
| spectral_no + span_loss | 0 | 0 | aynı |
| gnot | 0 | 0 | aynı |

`train.py --fast_dev_run`, `configs/default.yaml`, `spectral_no.yaml` ve `eigenspace.yaml` ile (gerçek 2D veri): üçü de "Verification complete".

### 4.4 Öğrenme kontrolü
d=64, L=2, m=24, lr 1e-3, 25 epoch (24 train / 3 val), CPU'da 3 dk 49 s:
- val/field_rel_l2: 1.13 (sanity) → 0.96 → 0.62 (ep 4) → 0.54 (ep 8) → 0.50 (ep 15) → 0.49 (ep 24).
- Test (3 geometri, hepsi axisym_cell): field_rel_l2 0.065, freq MAE 0.090 GHz (%2.2), span_rel_l2 0.085 (enerji normunda 0.225), grad_frac 0.58 (başlangıçta 0.88).

Pipeline öğreniyor; çok küçük veri olduğu için sayılar yalnız sağlama amaçlı.

### 4.5 Notebook ve eval
`Colab_Maxwell3D.ipynb` geçerli JSON (nbformat 4). Hücreler 2 ve 4–8, smoke modunda **birebir** koşuldu. Atlananlar: hücre 1 (git clone/checkout), 3 (apt/pip) ve 9 (tensorboard); `USE_DRIVE=False`, `WORK` scratch dizini.
- Hücre 5: 12 geometri, mesh 0.12, 12/12 başarılı (Ne 4140–5665).
- Hücre 6: PKL (`STORE_OPERATORS=True`, `_noops` soneki yok, adlandırma tutarlı), Rayleigh hatası 3.2e-15.
- Hücre 7: 2 epoch, en iyi checkpoint'le test.
- Hücre 8: `eval_3d.py --checkpoint <eğitim dizini>` en iyi checkpoint'i buluyor, CSV yazıyor, şekil tipi kırılımını basıyor.
- Hücre 7'nin yeniden çalıştırılması `--resume last.ckpt` ile devam ediyor.

Bayrakların hepsi (`--n_total --n_eigen_modes --mesh_size --families --seed --n_workers --h5_filename`, `--h5_filepath --output_path --no_operators`, `dataset.cache_operators`, `model.eigenspace.n_layers`) argparse ve config'te mevcut.

### 4.6 Fizik incelemesinin B1–B4 bulgularının erişilebilirliği
- **B1** (tüm yönler atılırsa θ=1): model yolunda tetiklenmiyor. Başlangıçta min μ = 5.9e-3, 5 epoch sonra 1.1e-2; `drop_tol=1e-6`'nın çok üstünde. Her örnekte 12/12 yön tutuluyor. Dolgu kolon üretmediği için de tetiklenmez. Yalnız bir örnekte $V\equiv0$ olursa oluşur, ki bu gözlenmedi.
- **B2** (Schur formu, birinci mertebe hata): $M_{div}$ köşegeni CG'nin Galerkin dikliğiyle ikinci mertebe. Saf gradyan kolonlarında `kp_tol=1e-8` ile $M_{div}/M_V$ ≤ 1.5e-13 ölçüldü; köşegen dışı terimler birinci mertebe. Kayıplarda gözle görülür bir etki yok.
- **B3** (float32 girdi): model `basis.double()` veriyor, `hcurl_grams` float64'e çeviriyor, bf16 autocast altında da M_div float64. Tetiklenmiyor.
- **B4** (maxiter'da sessiz dönüş): smoke setinin tamamında CG 100–110 iterasyon (≪ 2000); `*/kp_cg_iters` loglanıyor.

## 5. Profil (CPU, 4 çekirdek; önerilen genişlik 128×4, m=24, K=6)

B=8 (Ne 7948, Nv 1393), bir eğitim adımı:

| Parça | Süre |
|---|---|
| ileri + kayıp | 1.71 s |
| geri | 1.49 s |
| `KpSolve` ileri (100 iter) | 0.86 s |
| `KpSolve` geri (100 iter) | 0.87 s → **CG adımın %54'ü** |
| gövde (mass-aware attention + FFN, 4 blok) ileri | 0.55 s |
| kenar başı (edge MLP, B·Ne=64k kenar) ileri | ≈ 0.50 s |
| `project_basis` (CG dahil) | 0.875 s |
| `projected_eigh` (iki 24×24 eigh) | 4.6 ms |
| `hcurl_grams` | 78 ms |
| collate (işçide) | 365 ms |
| `__getitem__` (saklı operatör) | 3.7 ms/geometri |
| operatörü yeniden kurma (`--no_operators`, cache yok) | ≈ 83 ms/geometri (DataLoader, collate dahil; saklıyken 28 ms) |

CG iterasyonları toleransa zayıf bağlı: 1e-6 → 90, 1e-8 → 100, 1e-10 → 110. Tolerans gevşetmek kazandırmaz; CSR'ye geçmek (m5) kazandırır.

Bellek (geri geçiş için saklanan aktivasyonlar, 128×4, m=24):

| B | Ne | aktivasyon | batch tensörleri |
|---|---|---|---|
| 8 | 7948 | 0.67 GB | 52 MB |
| 16 | 8469 | 1.44 GB | 108 MB |

Parametre sayısı 854k (3.4 MB).

## 6. Önerilen ayarların sağlaması (3000 geometri, mesh 0.10, batch 8, 128×4, m=24, 95 GB GPU)

- **GPU belleği:** B=8'de aktivasyonlar ≈ 0.7 GB; CUDA çalışma alanıyla tepe ≈ 1.5–2 GB. Bu, 95 GB'ın %2'si. Aktivasyonlar B ve Ne ile doğrusal büyüyor. B=32'de ≈ 3 GB; mesh 0.07'de (Ne ≈ 19–24k) B=16'da ≈ 4–5 GB. GPU kısıt değil. Öneriler: `batch_size` 16–32 (lr'yi √-ölçekle, ör. 3e-4–4e-4) ya da doğruluk için docs/19'un hedefi olan mesh 0.07.
- **CPU RAM (asıl kısıt):** 3000×4.7 MB = 14 GB > 8 GB olduğu için notebook `--no_operators` seçiyor (1.6 GB PKL; ×3 veri kümesi ≈ 4.8 GB) ve `cache_operators=false` ile doğru biçimde önbellek kapalı. Mesh 0.07 seçilirse m7'deki heuristik düzeltilmeli.
- **Veri yükleyici hızı:** `--no_operators` ile geometri başına ≈ 83 ms CPU. 2400 train geometrisi / 8 işçi ≈ 25 s/epoch; 150 epoch ≈ 1 saat yükleyici CPU'su. Bu, GPU adım süresiyle aynı mertebede olabilir. B büyütülürse yükleyici darboğaz olabilir; `NUM_WORKERS`'ı çekirdek sayısına çekmek veya önbelleği fork'tan önce kurmak (m8) yardımcı olur.
- **K=6, m=24:** `n_basis ≥ K` sağlanıyor; bütün yönler tutuluyor (μ ≫ drop_tol). K ≤ 2 seçilirse M1 düzeltmesi gerekli (artık mevcut).
- **Süre (tahmini, GPU):** Adım başına ~200 CG SpMM + eigh + gövde. B=8 için ~0.1 s/adım, 300 adım/epoch ile ≈ 30 s/epoch; 150 epoch ≈ 1.5 saat. CSR'ye geçiş (m5) CG kısmını belirgin kısaltır.

---

## ✅ İnceleme sonrası uygulanan düzeltmeler

| Bulgu | Durum |
|---|---|
| **M1** (split-cluster NaN → EarlyStopping epoch 0'da durdu) | Düzeltildi (`cdbc457`), regresyon testi var |
| **docs/21 B1** (tüm yönler düşerse θ = 1.0) | Düzeltildi: düşürülen yönler için sabit taban 1e12 (`hcurl.projected_eigh`), test: `test_all_gradient_sample_gets_theta_above_spectrum` |
| **docs/21 B3** (float32 girdide CG yakınsamıyor) | Düzeltildi: `project_basis` girdiyi float64'e çevirir |
| **docs/21 B4** (CG maxiter'da sessiz dönüş) | Düzeltildi: `UserWarning` |
| **m5** (COO seyrek çarpım yavaş) | Düzeltildi: blok-diyagonal operatörler CSR; aynı kayıp (52.322559), adım ~1.7× hızlı (CPU, B=8), `num_workers=2` ile doğrulandı |
| **m7** (notebook `STORE_OPERATORS` mesh boyutunu yok sayıyor) | Düzeltildi: tahmin ∝ (0.10 / MESH_SIZE)³ |
| Diğer minor/nit (m1–m4, m6, m8, m9; docs/21 B2, B5–B15) | Açık — raporlandı |
