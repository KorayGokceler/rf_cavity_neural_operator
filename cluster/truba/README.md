# TRUBA'da 3D dataset üretimi: her geometri ailesi ayrı

Her aile (elliptical, reentrant, …, dtl ve OOD test aileleri) **ayrı ayrı üretilir ve saklanır**. Aileler birbirini beklemez. Bir aileyi büyütmek ya da yeni aile eklemek, var olanlara
dokunmaz.

```
$DATA_ROOT/$TAG/                         (LABELS=field: /arf/scratch/$USER/rfcav3d/F_p3p2_ms0.10_k10_mf0.05_v2)
├── h5/<aile>/<aile>_s00000.h5 …         FEM etiketleri (asıl kayıt, eğitim doğrudan bunları okur): parça = 1024 geometri
├── runs/<EXP>/                          eğitim koşuları (last.ckpt, best-*.ckpt)
└── logs/                                gen_<aile>_<parça>_<job>.out, train_<EXP>_<job>.out
```

## Dosyalar

| dosya | ne yapar |
|---|---|
| `config.sh` | **tek ayar dosyası**: yollar, kuyruk, çekirdek, süre, fizik (MESH_SIZE, N_STORE, …), TAG |
| `families.tsv` | aile → blok numarası, istenen geometri sayısı, parça boyu, grup (train / ood) |
| `setup_env.sh` | bir kere: `rfcav.def`'ten Apptainer imajını (`$CONTAINER`, tek `.sif`) bir `debug` düğümünde kurar, 2 geometrilik kontrol üretimi yapar. TRUBA `/arf` altına conda / pip kurulumunu yasaklar (docs/36 §5) |
| `rfcav.def` | imajın tanımı: python:3.11-slim + CUDA 12.4 torch + NGSolve + gmsh + scikit-fem + h5py (`requirements.txt`). Repo kodu imaja girmez, çalışırken bağlanır |
| `submit_family.sh <aile>` | ailenin **eksik** parçalarını bir SLURM iş dizisi olarak gönderir |
| `submit_all.sh [train\|ood\|all]` | gruptaki her aile için `submit_family.sh` |
| `status.sh` | aile başına ilerleme tablosu + `squeue` |
| `submit_train.sh` | GPU eğitimi, `LABELS=field` (`EXP`, `MODEL=small\|base\|large\|xl`, …); aynı EXP → `last.ckpt`'tan devam |
| `predict.sh …` | yeni bir kavitede saf field3d model tahmini (STEP / üretilmiş geometri / alan-etiketi H5; docs/25 §7) |
| `gen_shard.sbatch`, `train.sbatch` | işlerin kendisi (elle çağırmaya gerek yok) |

Model `field3d`'dir (docs/28–29); eski N0 modeli (`eigenspace3d`, PKL dönüştürme, `merge.sh`,
`eval.sbatch`) kaldırıldı.

## Tek satır: `run.sh` (arayüzün TRUBA sekmesi bu komutları üretir)

```bash
bash cluster/truba/run.sh all LABELS=field FAMILIES=train N_PER_FAMILY=1024 EXP=field_base --dry-run   # önce bak
bash cluster/truba/run.sh all LABELS=field FAMILIES=train N_PER_FAMILY=1024 EXP=field_base             # gönder
bash cluster/truba/run.sh status LABELS=field
```

- İşlemler:
  - `setup`: conda ortamı; CUDA torch ve NGSolve dahil.
  - `dataset`: ailelerin eksik parçaları.
  - `train`: GPU eğitimi.
  - `all`: veri, ardından eğitim; eğitim `--dependency=afterany` ile veri işlerini bekler.
  - `status`: aile başına ilerleme tablosu.
- `KEY=VALUE`: `config.sh`'deki her değişken (PARTITION, CPUS, TIME, ACCOUNT, MESH_SIZE, MODEL, EPOCHS, …), ayrıca:
  - `FAMILIES`: `train` | `ood` | `all` | virgüllü liste;
  - `N_PER_FAMILY`;
  - `TRAIN_FAMILIES`.
- `--dry-run`: her `sbatch` komutunu bağımlılıklarıyla yazdırır, hiçbir iş göndermez.
- İmaj (`CONTAINER`, varsayılan `~/containers/rfcav.sif`) yoksa ilk çağrı `setup_env.sh`'yi çalıştırır: imaj bir `debug`
  düğümünde `srun` altında kurulur (~15–25 dk, terminal açık kalmalı; `tmux` kullan). Tüm Python işleri imajın içinde koşar
  (`config.sh` `container_prefix`: `apptainer exec [--nv] --bind <repo>,<DATA_ROOT>[,CACHE_DIR,EXTRA_BIND] …`).
  `requirements.txt` değişince `FORCE=1 bash cluster/truba/setup_env.sh` ile yeniden kur.
- `LABELS=field` (varsayılan ve tek etiket türü; docs/29):
  - NGSolve p3 etiketleri eğri mesh üzerinde üretilir, p2 model uzayına izdüşürülür; dönüştürme adımı yoktur.
  - Eğitim, train grubundaki ailelerin H5 parça klasörlerinden yapılır (OOD aileleri eğitime girmez).
  - Ayrı TAG kullanılır: `F_p3p2_ms0.10_k10_mf0.05_v2`.
  - İş başına `CPUS / THREADS` örnek aynı anda çalışır (`THREADS=8` → 7 örnek, örnek başına ~5 GB).
- Eski `LABELS=n0` (N0 etiketleri) kaldırıldı: `config.sh` `LABELS=field` dışındaki her değerde hata verir.

## Adımlar

```bash
# 0) login düğümünde, bir kere
cd $HOME && git clone <repo> rf_cavity_neural_operator && cd rf_cavity_neural_operator
git checkout main
bash cluster/truba/setup_env.sh               # ~15-25 dk (debug düğümünde Apptainer imajı); sonunda "Environment OK"
nano cluster/truba/config.sh                  # PARTITION / ACCOUNT / TIME / DATA_ROOT'u kontrol et

# 1) deneme: aile başına 112 geometri, ayrı klasörde (<TAG>_test)
TEST=1 LABELS=field cluster/truba/submit_all.sh all
TEST=1 LABELS=field cluster/truba/status.sh   # bitince: tüm aileler 112/112
#    loglara bak: logs/gen_*.out içinde "mean t_mesh … t_solve …" → gerçek süre

# 2) asıl üretim: her aile ayrı iş dizisi
LABELS=field cluster/truba/submit_all.sh train   # train grubundaki aileler × 1024'lük parçalar
LABELS=field cluster/truba/submit_all.sh ood     # OOD test aileleri
LABELS=field cluster/truba/status.sh

# 3) yarım kalan / başarısız parçalar: aynı komutu tekrar çalıştır. Yalnız eksikler gider; yarım
#    parça .partial dosyasından kaldığı yerden devam eder (--resume).
LABELS=field cluster/truba/submit_family.sh spoke

# 4) eğitim: train grubundaki ailelerin H5 parça klasörleri
LABELS=field EXP=field_base MODEL=base cluster/truba/submit_train.sh
```

Ayarları komut başına da değiştirebilirsin: `CPUS=112 TIME=1-00:00:00 cluster/truba/submit_family.sh dtl`.
`config.sh`'deki her değişken bu şekilde ezilebilir.

## Bir aileyi büyütmek, yeni aile eklemek

- **Büyütmek:** `families.tsv`'de `n_total`'ı artır (ör. 10240 → 20480) ve `submit_family.sh <aile>`'yi
  çalıştır. Var olan parçalar atlanır, yalnız yeniler üretilir.
- **Yeni aile:** `families.tsv`'ye yeni bir satır ekle. Blok numarası **yeni** olmalı (≤ 31).
  Aile, `src/data_gen/cavity_shapes.py` ile `dataset_generator_3d.BUILDERS`'ta tanımlı olmalı.
- **Asla** bir ailenin `block` ya da `shard_size` değerini sonradan değiştirme: geometri kimlikleri
  bunlardan hesaplanıyor. Fizik ayarlarını (MESH_SIZE, N_STORE, deformasyon) değiştirdiğinde
  yeni bir `TAG` kullan; farklı ayarlar aynı klasöre karışmasın.

## Kimlikler ve tekrarlanabilirlik

- Geometri kimliği `id = block · 2^19 + i`. Her ailenin kimlik aralığı ayrı, dolayısıyla farklı
  ailelerin parçaları aynı eğitimde birlikte kullanılabilir.
- Kimlik aynı zamanda rastgele sayı akışı (`default_rng([SEED, id])`) ve Sobol indisi, yani aynı
  kimlik her zaman aynı geometriyi verir. Bir parçayı silip yeniden üretirsen birebir aynısı gelir.
- Parça boyu 1024 (2'nin kuvveti): Sobol blokları dengeli.
- Kimlikler < 2^24 kalır (float32'de tam).

## Maliyet (TARİHSEL: eski N0 etiketlerinin ölçümü, artık üretilmiyor — mesh 0.10, 10 mod, 1 çekirdek, aile başına 4 geometri; güncel field etiket maliyetleri: docs/29–30)

| aile | s / geometri | H5 MB / geometri |
|---|---|---|
| elliptical | 2.2 | 1.3 |
| reentrant | 1.0 | 0.8 |
| pillbox_pipes | 1.2 | 0.7 |
| ridged_box | 1.0 | 0.8 |
| composite | 0.9 | 0.5 |
| freeform | 2.3 | 0.6 |
| hwr | 1.8 | 1.1 |
| spoke | 2.6 | 1.4 |
| dtl | 3.9 | 1.6 |

- **Süre:** 9 aile × 10240 geometri ≈ 50 çekirdek-saat (yeniden çekme ve başarısız deneme payıyla
  belki 2 katı). 56 çekirdekte 1024 geometrilik bir parça yalnız ~1–3 dk sürer; bir ailenin tamamı
  ~10–20 dk. Darboğaz hesap değil, **kuyrukta bekleme** ve **disk**. Daha ince mesh rahatça
  mümkün: MESH_SIZE 0.07 ile kenar sayısı ~3×, süre ~4–5× olur. İşler çok kısa kalırsa
  `families.tsv`'de yeni bir aile için daha büyük `shard_size` (2048 / 4096) seçebilirsin. Ölçüm 4
  çekirdekli bir makinede yapıldı; TRUBA'da gerçek süreyi deneme koşusunun loglarından oku.
- **Disk:** eski N0 H5'i ~1 MB/geometri idi (tarihsel); field etiketli H5 için docs/29 §3'e bak. Scratch kotanı kontrol et.

## Kuyruk notları

- GPU: varsayılan `GPU_PARTITION=akya-cuda` (4 × V100 16 GB, GPU başına en az 10 çekirdek, herkese açık), `GPU_CPUS=10`.
  `barbun-cuda` (2 × P100) için `GPU_CPUS=20` şart; `palamut-cuda` / `kolyoz-cuda` yalnız araştırma merkezi / sözleşmeli
  projelere açık (docs/36 §2). Çok GPU: `GPUS=4` (bir akya düğümü, DDP).
- Varsayılan `PARTITION=orfoz`, `CPUS=56`. orfoz düğümü 112 çekirdekli ve işler 56'nın katı çekirdek
  ister. Güncel sınırları `sinfo` ve TRUBA dokümanlarındaki "Kuyruk Bilgisi" sayfasından kontrol et.
- Hesabın bir proje kodu gerektiriyorsa `ACCOUNT=...` ayarla.
- `MAX_PARALLEL=10`: bir ailenin en fazla 10 parçası aynı anda koşar. Aileler bağımsız iş
  dizileridir.
