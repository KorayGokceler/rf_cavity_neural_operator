# 25 · Uçtan uca pipeline: veri üretiminden modelin son tahminine (TRUBA)

> **Historical:** describes the removed N0 model (`eigenspace3d`, `EigenspaceOperator3D`, PKL / `convert_3d.py` pipeline); removed in the `claude/field-model-ui` PR. The current model is `field3d` (docs/28, docs/29). Exception: §7 (prediction of a new cavity) was rewritten for `field3d` and is current.

> TRUBA'da sırayla izlenecek akış. Her aşama: **ne yapar → komut → ne üretir → nasıl kontrol
> edilir**. Bütün komutlar repo kökünden çalışır. Ayarların tamamı `cluster/truba/config.sh` içinde.
> Her değişken komut başında ezilebilir (`MODEL=large EXP=l1 cluster/truba/submit_train.sh`).

## 0. Genel resim

```
 geometri ailesi (families.tsv)                               ┌───────────── TRUBA CPU (orfoz) ─┐
   │  1. üretim: gmsh şekil + tet mesh → N0 FEM özdeğer çözümü  │ aile başına SLURM iş dizisi     │
   ▼                                                           │ parça = 1024 geometri            │
 h5/<aile>/<aile>_s00000.h5   (asıl kayıt: mesh, E alanı, f)   └──────────────────────────────────┘
   │  2. dönüştürme: normalize mesh, 9 düğüm özelliği, M/K/G/Kp, QoI etiketleri (Q0, R/Q, …)
   ▼
 pkl/<aile>.pkl  ── 3. karışım ──►  pkl/mix_train.pkl   pkl/mix_ood.pkl
   │                                     │                     │
   │            4. eğitim (GPU) ◄────────┘                     │
   ▼                                                           │
 runs/<EXP>/best-….ckpt, last.ckpt                              │
   │  5. değerlendirme: test (ID) + OOD ◄───────────────────────┘
   ▼        frekans, alan, Q0/G/R/Q/R_sh/Epk/Bpk, kesitler, ParaView
 runs/<EXP>/eval/
   │  6. tahmin: yeni kavite (STEP / mesh) → model → frekans + alan + kavite değerleri (FEM yok)
   ▼
 runs/<EXP>/predict/<ad>/prediction.json, modes.vtu
```

Klasör yapısı: `$DATA_ROOT/$TAG/` (varsayılan `/arf/scratch/$USER/rfcav3d/E_ms0.1_k10_v2/`):
`h5/ pkl/ runs/ logs/`. TAG fizik ayarlarından türetilir; ayar değişince yeni klasör açılır,
böylece farklı ayarlarla üretilmiş veriler birbirine karışmaz.

## 1. Bir kerelik kurulum (login düğümü)

```bash
cd $HOME
git clone https://github.com/KorayGokceler/rf_cavity_neural_operator.git
cd rf_cavity_neural_operator && git checkout main
bash cluster/truba/setup_env.sh          # Apptainer imajı (debug düğümünde) + 2 geometrilik kontrol
```

- **Kontrol:** son satır `Environment OK`.
- `config.sh`'de gözden geçir:
  - `PARTITION`: CPU kuyruğu, varsayılan `orfoz`, 56 çekirdek.
  - `GPU_PARTITION`: varsayılan `akya-cuda` (herkese açık; `palamut-cuda` kısıtlı, docs/36 §2).
  - `GPU_CPUS`: GPU başına çekirdek kuralı kuyruğa göre (akya 10, barbun 20, palamut 16).
  - `ACCOUNT`: proje kodu gerekiyorsa.
  - `DATA_ROOT`.
- **Eğitim ortamı:** imaj CUDA 12.4 torch içerir (GPU için `--nv`, konteyner dışında kurulum yok). Başka CUDA
  sürümü için `TORCH_INDEX=… FORCE=1 bash cluster/truba/setup_env.sh`. TRUBA `/arf` altına conda / pip kurulumunu
  yasaklar (docs/36 §5); bu yüzden Miniforge kullanılmıyor.

## 2. Deneme koşusu (aile başına 112 geometri, `<TAG>_test`)

```bash
TEST=1 cluster/truba/submit_all.sh all       # 14 aile: üretim + aile PKL'i
TEST=1 cluster/truba/status.sh               # hepsi 112/112, pkl sütunu dolu olmalı
```

Kontroller:
- `logs/gen_<aile>_0_*.out` dosyasının sonunda `Done: N samples (k failed)` ve `mean t_mesh … t_solve …` satırları olmalı.
  - Başarısız oranı %5'in üstündeyse o aileye bak.
  - Bu satırlar gerçek süreyi verir, `TIME`'ı buna göre ayarla.
- `logs/conv_<aile>_*.out` içinde `Rayleigh max rel err ~1e-14` ve `QoI labels (0 geometries failed)` olmalı.
- Görsel kontrol (isteğe bağlı, yerelde ya da Colab'da): `scripts/show_geometries.py` ve `scripts/dataset_stats_3d.py`.

İsteğe bağlı olarak deneme verisiyle kısa bir uçtan uca koşu yapabilirsin. Bu, GPU ortamını ve iş akışını sınar; sonuçları anlamlı değildir:
```bash
TEST=1 cluster/truba/merge.sh train elliptical:all pillbox_pipes:all hwr:all
TEST=1 cluster/truba/merge.sh ood box:all coax_qw:all
TEST=1 EXP=smoke MODEL=small EPOCHS=3 cluster/truba/submit_train.sh --then-eval
```

## 3. Asıl veri üretimi

```bash
cluster/truba/submit_all.sh train          # 9 aile × 10 parça × 1024 = 92 160 geometri
cluster/truba/submit_all.sh ood            # 5 OOD ailesi × 1024 (yalnız test)
cluster/truba/status.sh                    # ilerleme
```

**Her aile ayrı üretilir.** Üretim adımı için:
- Geometri kimliği `id = blok·2^19 + i`. Aynı kimlik her zaman aynı geometriyi verir.
- Örnekleme Sobol ile yapılır; geometrilerin %50'sine düzgün deformasyon uygulanır.
- Mesh boyu 0.10 (hücre başına). Her geometride 11 mod çözülür, 10'u saklanır.

**Kesilen ya da başarısız parça:** aynı `submit_family.sh <aile>` komutunu tekrar çalıştır. Yalnız
eksik parçalar gider, yarım parça `.partial` dosyasından devam eder.

**Ne kadar sürer:** yaklaşık 50–100 çekirdek-saat. Asıl bekleme kuyrukta olur.

**Disk:** H5 ~100 GB, lean PKL ~130 GB.

## 4. Eğitim verisinin hazırlanması

Aile PKL'leri `pkl/<aile>.pkl` dosyalarıdır. Eğitim için bunlardan bir karışım hazırlanır. Dataset PKL'in tamamını
RAM'e aldığı için ilk eğitimlerde alt küme kullan (aile başına 2 parça ≈ 18k geometri ≈ 25 GB):

```bash
cluster/truba/merge.sh train elliptical:2 reentrant:2 pillbox_pipes:2 ridged_box:2 \
    composite:2 freeform:2 hwr:2 spoke:2 dtl:2                 # → pkl/mix_train.pkl
cluster/truba/merge.sh ood box:all coax_qw:all pillbox_port:all elliptical_long:all junction:all
                                                               # → pkl/mix_ood.pkl
```

- **Bölme:** `mix_train.pkl` geometri bazında %80 train, %10 val, %10 test olarak bölünür (seed 42).
  Test bölmesi aynı ailelerden ama hiç görülmemiş geometrilerden oluşur (ID test). `mix_ood.pkl`'in
  tamamı OOD testidir.
- **Etiketler:** her geometride 10 FE modu (alan + frekans), 11. modun frekansı (`freq_next`) ve
  her mod için 7 kavite değeri (bakır duvar, β = 1, linac R/Q).

## 5. Eğitim (GPU)

```bash
EXP=base  MODEL=base  cluster/truba/submit_train.sh --then-eval     # 2.8 M parametre, ilk ciddi koşu
EXP=large MODEL=large cluster/truba/submit_train.sh --then-eval     # 6.5 M, önerilen ana koşu
```

### Model nedir, ne öğrenir
- **Girdi:** tet mesh ve her düğümde 9 özellik: konum, duvara uzaklık ve yönü, düğüm hacmi, torsiyon fonksiyonu.
- **Ağ:** kütle-farkında lineer attention katmanları N0 kenarlarında `n_basis` tane baz vektörü üretir.
- **Ritz katmanı:** bazları alır, gradyan kısmını projeksiyonla çıkarır, Ku = λMu problemini bu alt uzayda çözer.
- **Çıktı:** en alttaki K = 6 modun alanları (M-ortonormal) ve frekansları, f = c√λ/(2π·scale).
- **Frekans garantisi:** Rayleigh–Ritz, aynı mesh'teki FE çözümünün altına inemez.

### Loss
- **Span loss:** 10 FE modunun, tahmin edilen alt uzaya ne kadar uzak olduğu.
- **Küçük ek terimler:** self-supervised ve ortogonallik terimleri.
- **Frekans terimi:** `freq_weight · MSE`.
- **İsteğe bağlı:** `QOI_WEIGHT=0.1` ile kavite değerleri loss'u.

### Model boyutları
`MODEL` ile seçilir (`config.sh: model_overrides`):

| MODEL | genişlik | katman | baz | parametre |
|---|---|---|---|---|
| small | 128 | 4 | 24 | 0.85 M |
| base | 192 | 6 | 32 | 2.8 M |
| large | 256 | 8 | 48 | 6.5 M |
| xl | 384 | 12 | 64 | 21.7 M |

### Önemli ayarlar
- `EPOCHS` (150), `BATCH` (4), `LR` (3e-4) ve `NUM_WORKERS` (6): her worker PKL'in kopyasını görür, RAM'i izle.
- `GPUS=4`: tek düğümde DDP. Diğer ayarlar için: `TRAIN_EXTRA="training.patience=30 …"`.

### Kesilen eğitim
Aynı `EXP` ile tekrar gönder; `last.ckpt`'tan devam eder.

### İzleme
`logs/train_<EXP>_*.out` dosyasını izle. TensorBoard için `runs/<EXP>/version_*` klasörünü bilgisayarına kopyala ya da port yönlendirme yap.

Değerlere bak:
- `val/field_rel_l2`: en iyi checkpoint bununla seçilir, düşmeli.
- `val/freq_mae_ghz`, `val/freq_rel_err`.
- `val/mode_k_rel_l2`: hangi modun zor olduğunu gösterir.

Teşhis:
- Eğitim ve doğrulama eğrileri birlikte düz kalıyorsa: model küçük, bir üst `MODEL`'e geç.
- Eğitim hatası doğrulamanın çok altındaysa: veri az; `merge.sh` ile daha fazla parça al.

**GPU seçimi:** Ritz katmanı ve CG float64 çalışıyor; FP64'ü güçlü GPU'lar (A100 / V100 / H100) şart.

## 6. Değerlendirme

`--then-eval` verdiysen eğitimden sonra kendiliğinden başlar. Elle başlatmak için: `EXP=large cluster/truba/submit_eval.sh`.

`runs/<EXP>/eval/` içinde, `test` (ID) ve `ood` için ayrı ayrı:

| dosya | içerik |
|---|---|
| `*_eval.txt / .csv` | geometri başına alan rel-L2 (mod mod), frekans hatası, span hatası; aile kırılımı |
| `*_qoi.txt / .csv / _summary.csv` | Q0, G, R/Q, R_sh, T, Epk/Eacc, Bpk/Eacc: model ve FE (aynı operatörler), göreli hata; **hızlandırıcı mod** ayrı |
| `*_viz/` | kesit PNG'leri (gerçek, tahmin, hata) ve ParaView `.vtu` dosyaları |

Nasıl okunur:
- ID test sonuçlarını OOD sonuçlarıyla karşılaştır. Büyük fark varsa genelleme sınırlı demektir.
- Aile kırılımında en kötü aile, sıradaki veri artışının hedefidir.
- QoI hatası: dejenere modlar hariç tutulur. Hızlandırıcı mod (R/Q'su en büyük mod) asıl ölçüt.
- Referans: mesh 0.10'da FE'nin kendi hatası frekansta ~%0.1–0.4. E formülasyonunda eğri duvarlarda Q0 ve G etiketleri ~%3–6 düşük (docs/24 §3).

**Etkileşimli görüntüleyici** (CST benzeri faz kaydırıcısı): `Colab_Maxwell3D.ipynb`'deki
`ModeViewer` hücresi. Bunun için checkpoint'i ve PKL'i Drive'a kopyala.

**Çözünürlük genellemesi** (isteğe bağlı): aynı geometrileri daha ince mesh'te üret ve
`scripts/resolution_study.py` ile karşılaştır.

## 7. Son tahmin: yeni bir kavite

```bash
# CAD dosyası (mm), 9 hücreli bir kavite: mesh yoğunluğunu eğitimle eşlemek için --vol_div 9
EXP=field_base NAME=tesla cluster/truba/predict.sh --step $HOME/cad/tesla.step --unit mm --vol_div 9 --fe
# üretilmiş bir geometri (aile + kimlik; alan datasetlerindeki geometrinin aynısı)
EXP=field_base NAME=e1 cluster/truba/predict.sh --family elliptical --id 524300 --fe
# alan-etiketi H5'indeki bir geometri (saklanan eğri mesh; p3 etiketleri de yazılır)
EXP=field_base NAME=h1 cluster/truba/predict.sh --h5 $DATA/F_…/h5/elliptical/elliptical_s00000.h5 --sample 0
```

Akış (`scripts/predict_geometry.py`, web arayüzünün Tahmin'iyle aynı kod: docs/26 §1b):
1. **Geometri:** katının eğri netgen mesh'i (`field_labels.start_mesh`, maxh = 3 h, `h = mesh_size·(V/vol_div)^(1/3)`).
   Mesh dosyası kabul edilmez: field3d modeli CAD katısının eğri geometrisinde çalışır.
2. **Ön kontrol:** topoloji kontrolü (bağlı, manifold mesh).
3. **Model girdisi:** `highorder.field_operators` (HCurl p2, kuadratür noktası belirteçleri), eğitimdekiyle aynı.
4. **Tahmin:** Ritz frekansları ve p2 alanları (FEM çözümü yok).
5. **Kavite değerleri:** tahmin edilen p2 alanlarından `highorder.mode_qoi` ile Q0, G, R/Q, R_sh, T, Epk/Eacc ve
   Bpk/Eacc (U = 1 J, bakır, β = 1; etiketlerle aynı eksen ve hızlanma boyu).
6. `--fe` verilirse aynı eğri mesh'te p2 FE de çözülür; Ritz frekansları bu ayrık özdeğerlerin üst sınırıdır.
   `--h5` ile saklanan p3 etiket frekansları da yazılır.
7. **Çıktılar:** `prediction.json` ve `modes.vtu` (tahmin edilen her modun E ve H alanı, düğümlerde); ParaView'de aç.
   Checkpoint verilmezse eğitilmemiş bir model kullanılır (yalnız boru hattı denemesi).

Kavite için koşullar:
- **Kapalı olmalı:** PEC duvar, beam pipe uçları kapalı.
- **Işın ekseni:** CAD dosyasında z ekseni, x = y = 0; aile geometrilerinde ailenin ekseni (`src.qoi.operators.beam_axis`, HWR: x). Eksen seçeneği (`--axis_dir/--axis_point`) field3d CLI'ında henüz yok.
- **Vakum hacmi:** STEP içinde kavitenin metal kabuğu değil, vakum hacmi (iç boşluk) olmalı.

## 8. Önerilen ilk hafta takvimi

| gün | iş | çıktı |
|---|---|---|
| 1 | kurulum + deneme koşusu (§1–2) | ortam çalışıyor, gerçek süreler |
| 1–2 | asıl üretim, train + ood (§3) | 92k + 5k geometri |
| 2 | karışımlar (§4), `EXP=base` eğitimi | ilk referans sonuç |
| 3–5 | `EXP=large` (ve gerekirse `xl`), değerlendirme (§6) | veri mi model mi kararı (docs/23 §1.2) |
| 5 | gerçek kavite tahmini `--fe` ile (§7) | CST karşılaştırmasına hazır sayılar |

## 9. Bilinen sınırlar

- **Dataset RAM:** PKL'in tamamı belleğe yükleniyor. 92k geometrinin hepsiyle eğitim için parça bazlı tembel yükleme gerekiyor (yol haritası §3).
- **Q0/G mesh yanlılığı:** E formülasyonunda eğri duvarlarda birinci mertebeden; bir düzeltme yöntemi hazır (docs/24 §2.2).
- **Işın ekseni olmayan şekiller:** ridged_box, composite ve freeform için R/Q nominal z eksenine göre tanımlı; hızlandırıcı anlamı yok.
- **Model sınırı:** model, etiketi üreten mesh'in FE çözümünden daha doğru olamaz. CST'ye karşı mutlak doğruluk için daha ince mesh ya da yüksek mertebeden etiket gerekir.
