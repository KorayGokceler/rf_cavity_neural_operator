# TRUBA'da 3D dataset üretimi: her geometri ailesi ayrı

Her aile (elliptical, reentrant, …, dtl ve OOD test aileleri) **ayrı ayrı üretilir, dönüştürülür ve
saklanır**. Aileler birbirini beklemez. Bir aileyi büyütmek ya da yeni aile eklemek, var olanlara
dokunmaz.

```
$DATA_ROOT/$TAG/                         (varsayılan /arf/scratch/$USER/rfcav3d/E_ms0.10_k10_v1)
├── h5/<aile>/<aile>_s00000.h5 …         ham FEM çıktısı (asıl kayıt): parça = 1024 geometri
├── pkl/<aile>.pkl                       ailenin eğitim PKL'i (N0 + QoI etiketleri, lean)
├── pkl/mix_<ad>.pkl                     isteğe bağlı karışık eğitim PKL'i (merge.sh)
└── logs/                                gen_<aile>_<parça>_<job>.out, conv_<aile>_<job>.out
```

## Dosyalar

| dosya | ne yapar |
|---|---|
| `config.sh` | **tek ayar dosyası**: yollar, kuyruk, çekirdek, süre, fizik (FIELD, MESH_SIZE, N_STORE, …), TAG |
| `families.tsv` | aile → blok numarası, istenen geometri sayısı, parça boyu, grup (train / ood) |
| `setup_env.sh` | bir kere: Miniforge + `rfcav` ortamı (conda-forge gmsh), kısa kontrol üretimi |
| `submit_family.sh <aile>` | ailenin **eksik** parçalarını bir SLURM iş dizisi olarak gönderir, ardından dönüştürmeyi gönderir |
| `submit_all.sh [train\|ood\|all]` | gruptaki her aile için `submit_family.sh` |
| `status.sh` | aile başına ilerleme tablosu + `squeue` |
| `merge.sh <ad> aile:n …` | ailelerin ilk n parçasından karışık PKL |
| `submit_train.sh [--then-eval]` | GPU eğitimi (`EXP`, `MODEL=small\|base\|large\|xl`, …); aynı EXP → `last.ckpt`'tan devam |
| `submit_eval.sh` | test (ID) + OOD değerlendirmesi: frekans/alan, Q0/R/Q…, kesit grafikleri, ParaView |
| `predict.sh …` | yeni bir kavitede saf model tahmini (STEP / mesh / üretilmiş geometri) |
| `gen_shard.sbatch`, `convert_family.sbatch`, `merge.sbatch`, `train.sbatch`, `eval.sbatch` | işlerin kendisi (elle çağırmaya gerek yok) |

Uçtan uca akış (veri → eğitim → değerlendirme → tahmin): **docs/25_PIPELINE.md**.

## Adımlar

```bash
# 0) login düğümünde, bir kere
cd $HOME && git clone <repo> rf_cavity_neural_operator && cd rf_cavity_neural_operator
git checkout claude/truba-datagen
bash cluster/truba/setup_env.sh               # ~10 dk; sonunda "Environment OK"
nano cluster/truba/config.sh                  # PARTITION / ACCOUNT / TIME / DATA_ROOT'u kontrol et

# 1) deneme: aile başına 112 geometri, ayrı klasörde (<TAG>_test)
TEST=1 cluster/truba/submit_all.sh all
TEST=1 cluster/truba/status.sh                # bitince: tüm aileler 112/112, pkl dolu
#    loglara bak: logs/gen_*.out içinde "mean t_mesh … t_solve …" → gerçek süre

# 2) asıl üretim: her aile ayrı iş dizisi
cluster/truba/submit_all.sh train             # 9 aile × 10 parça × 1024
cluster/truba/submit_all.sh ood               # 5 OOD test ailesi × 1024
cluster/truba/status.sh

# 3) yarım kalan / başarısız parçalar: aynı komutu tekrar çalıştır. Yalnız eksikler gider; yarım
#    parça .partial dosyasından kaldığı yerden devam eder (--resume).
cluster/truba/submit_family.sh spoke

# 4) eğitim için karışık PKL (aile PKL'leri yerinde kalır)
cluster/truba/merge.sh mix9x2 elliptical:2 reentrant:2 pillbox_pipes:2 ridged_box:2 \
    composite:2 freeform:2 hwr:2 spoke:2 dtl:2
```

Ayarları komut başına da değiştirebilirsin: `CPUS=112 TIME=1-00:00:00 cluster/truba/submit_family.sh dtl`.
`config.sh`'deki her değişken bu şekilde ezilebilir.

## Bir aileyi büyütmek, yeni aile eklemek

- **Büyütmek:** `families.tsv`'de `n_total`'ı artır (ör. 10240 → 20480) ve `submit_family.sh <aile>`'yi
  çalıştır. Var olan parçalar atlanır, yalnız yeniler üretilir, sonra PKL yeniden yazılır.
- **Yeni aile:** `families.tsv`'ye yeni bir satır ekle. Blok numarası **yeni** olmalı (≤ 31).
  Aile, `src/data_gen/cavity_shapes.py` ile `dataset_generator_3d.BUILDERS`'ta tanımlı olmalı.
- **Asla** bir ailenin `block` ya da `shard_size` değerini sonradan değiştirme: geometri kimlikleri
  bunlardan hesaplanıyor. Fizik ayarlarını (FIELD, MESH_SIZE, N_STORE, deformasyon) değiştirdiğinde
  yeni bir `TAG` kullan; farklı ayarlar aynı klasöre karışmasın.

## Kimlikler ve tekrarlanabilirlik

- Geometri kimliği `id = block · 2^19 + i`. Her ailenin kimlik aralığı ayrı, dolayısıyla farklı
  ailelerin parçaları tek PKL'de birleşebilir (converter tekrarlanan kimliği reddeder).
- Kimlik aynı zamanda rastgele sayı akışı (`default_rng([SEED, id])`) ve Sobol indisi, yani aynı
  kimlik her zaman aynı geometriyi verir. Bir parçayı silip yeniden üretirsen birebir aynısı gelir.
- Parça boyu 1024 (2'nin kuvveti): Sobol blokları dengeli.
- Kimlikler < 2^24 kalır: PKL'deki float32 `Theta` sütununda tam saklanır.

## Maliyet (ölçüm: mesh 0.10, 10 mod, 1 çekirdek, aile başına 4 geometri)

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
- **Disk:** H5 ~1 MB/geometri → ~100 GB. Lean PKL ~1.4 MB/geometri → ~130 GB. Tam PKL (operatörlü)
  ~9× H5 olurdu, bu yüzden `PKL_LEAN=1` varsayılan. Scratch kotanı kontrol et. Gerekirse PKL'leri
  yalnız lazım olan aileler için üret; H5'ler asıl kayıt.
- **Eğitim RAM'i:** dataset PKL'in tamamını belleğe alıyor. Hepsi birden (~130 GB) tek süreçte
  fazla, bu yüzden ilk eğitimler için `merge.sh` ile alt küme kullan (ör. aile başına 2 parça ≈
  18k geometri ≈ 25 GB). Lean PKL ile eğitimde `dataset.cache_operators: false` kullan; operatörler
  her örnekte yeniden kurulur, DataLoader `num_workers` ile paralelleşir. Tüm veriyle eğitim için
  sıradaki iş parça bazlı tembel yükleme (docs/23_ROADMAP.md §3).

## Kuyruk notları

- Varsayılan `PARTITION=orfoz`, `CPUS=56`. orfoz düğümü 112 çekirdekli ve işler 56'nın katı çekirdek
  ister. Güncel sınırları `sinfo` ve TRUBA dokümanlarındaki "Kuyruk Bilgisi" sayfasından kontrol et.
- Hesabın bir proje kodu gerektiriyorsa `ACCOUNT=...` ayarla.
- `MAX_PARALLEL=10`: bir ailenin en fazla 10 parçası aynı anda koşar. Aileler bağımsız iş
  dizileridir.
- Dönüştürme işi tek süreçlidir. 10240 geometri, QoI etiketleriyle birlikte ~30–60 dk sürer.
  Çekirdek isteği RAM içindir; daha küçük bir kuyruk varsa `CONVERT_PARTITION` ve `CONVERT_CPUS`
  ile ayarla.
- Dönüştürme, üretim işleri bittikten sonra (başarısız olsalar bile, `afterany`) başlar ve bitmiş
  parçaları dönüştürür. Eksik parça kaldıysa logda `WARNING` görürsün; `submit_family.sh`'yi tekrar
  çalıştırmak hem eksikleri hem dönüştürmeyi yeniler.
