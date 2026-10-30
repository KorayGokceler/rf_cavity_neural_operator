# 36 — TRUBA referansı: kurallar, kuyruklar, kotalar, depolama, yazılım ve bizim ayarlarla karşılaştırma

> **Durum (2026-10-30).** Kaynak: `docs.truba.gov.tr` sayfalarının **tam metni** (32 sayfa
> [`research/truba/pages/`](research/truba/pages/) altında; nasıl alındığı:
> [`research/truba/README.md`](research/truba/README.md)). Her bölümün başında ilgili sayfa ve son değişiklik
> tarihi var. Sayfalar birbiriyle çelişirse tarihi yeni olan geçerlidir. Sınırlar zamanla değişir; koşudan önce
> TRUBA'da `lssrv`, `sinfo`, `scontrol show partition=<kuyruk>` ile doğrulayın. Bu belge `truba-ops` ajanının ve
> `/truba-run` komutunun platform bilgisi kaynağıdır; betikler: `cluster/truba/README.md`, docs/29 §5.

## 0. Bizi doğrudan ilgilendiren bulgular (önce bunlar)

1. **`/arf` üzerine conda / pip kurulumu yasak** (Python kılavuzu 2025-12, depolama 2026-04). `setup_env.sh` bugün
   `$HOME/miniforge3` altına Miniforge + `rfcav` ortamı + `pip install` yapıyor; bu TRUBA kuralına aykırı (yüz
   binlerce küçük dosya, 500K inode kotası). Doğru yol: **Apptainer konteyneri** (tek `.sif` dosyası, §5). Açık iş:
   docs/31 madde 22.
2. **Eşzamanlı çekirdek sınırı: standart kullanıcı 160, yüksek lisans 40, lisans 4** (SSS 16). Bizim varsayılanımız
   `CPUS=56` × `MAX_PARALLEL=10` = 560 çekirdek ister; aşan işler `AssocGrpCpuLimit` ile kuyrukta bekler. 160 sınırında
   en fazla **2 shard işi** (112 çekirdek) aynı anda çalışır. Gerçek sınırı `ACCOUNT`'un sahibi bilir.
3. **Scratch dosyaları en çok 30 gün yaşar, otomatik silinir; yedekleme yok** (depolama 2026-04). `DATA_ROOT`
   scratch'te; H5 etiketleri ve checkpoint'ler iş bittikçe başka yere kopyalanmalı.
4. **`GPU_PARTITION=palamut-cuda` varsayılanı çoğu hesap için erişilemez**: `palamut-cuda` ve `kolyoz-cuda` yalnız
   araştırma merkezlerinin altyapı projelerine ve TÜBİTAK ULAKBİM sözleşmeli projelerine açık. Herkese açık GPU
   kuyrukları `barbun-cuda` (P100) ve `akya-cuda` (V100).
5. **Başvuruda belirtilen uygulamadan başka iş çalıştırılmaz** ve **TRUBA uygulama geliştirme için kullanılamaz**
   (kullanım politikası): derlenmiş / hazır uygulama taşınır. Yeni uygulama için `trubadestek@tubitak.gov.tr`'e
   yazılır.

## 1. Kullanım politikası ve hesap

*(`0-truba/kullanim_kurallari`, `uyelik_rehberi`, `kullanici_basvurulari`, SSS 13, 38)*

- Hesap kişiye özeldir; paylaşılırsa kapatılır. Yayınlarda TRUBA'ya teşekkür etmek **zorunludur**
  (`0-truba/TRUBAtesekkur`'te örnek metinler).
- Başvuru: `portal.truba.gov.tr` (politikayı onayla, e-postayı doğrula). Lisans öğrencisi: danışmanın proje hesabı
  gerekir. Yüksek lisans: danışman onayı. Doktora ve akademisyen: YÖKSİS'e göre doğrudan.
- Her kullanıcıya **eşzamanlı çekirdek sayısı, çekirdek-saat kotası ve disk kotası** tanımlanır. Çekirdek-saat
  kotası bitince iş kuyruğa girer ama başlamaz (`AssocGrpCPUMinutesLimit`); artırım: portalda "Yayınlar"a TRUBA
  kullanan yayınları ekle, sonra `trubadestek@tubitak.gov.tr`'e ek kaynak talebi yaz. Çekirdek sınırı aşılınca
  `AssocGrpCpuLimit`.
- **Login düğümlerinde (`arf-ui*`, `cuda-ui`) ağır iş çalıştırılmaz**: yalnız derleme ve kısa işler; aşırı yük
  yaratan iş sonlandırılır, ısrar edenin hesabı askıya alınır. Derleme, konteyner kurma ve interaktif iş için
  `debug` kuyruğundan `srun --pty` kullanılır (§3).
- Destek: yalnız `trubadestek@tubitak.gov.tr` (2026-04-13'ten beri; eski `grid-teknik@ulakbim.gov.tr` değil).
  Şablon (`0-truba/truba_iletisim`): kullanıcı adı, proje, Job ID, yazılım + sürüm, modüller, dizin, betik adı,
  NodeList, hata kayıtları, adımlar. Betikte `echo "SLURM_JOB_ID $SLURM_JOB_ID"; echo "SLURM_NODELIST
  $SLURM_NODELIST"` ve sonradan `sacct -j <id> --format="JobID,JobName%30,Partition,NodeList,Account,Start,AllocCPUS,State,ExitCode"`.
- Ücretli / öncelikli kullanım (HSB / VSB) ve proje hesapları: `4-proje_hesaplari_ucretli_kullanim/…` (fiyatlar
  yıllık değişir; bu belgeye yazılmadı).
- Duyuru (2026-06-30): eski `/truba/home` verileri `/arf/truba_old/home` altındaydı ve **30 Haziran 2026'da kalıcı olarak silinecekti**; her şey `/arf/home` altında olmalı.

## 2. Kuyruklar

*(`1-kaynaklar/arf/arf_kuyruk_bilgisi` 2025-12, `2-temel_bilgiler/hesaplama_kumeleri` 2025-09,
`1-kaynaklar/arf_acc/arf_acc_kuyruk_bilgisi` 2025-08, `2-temel_bilgiler/gpu` 2025-08)*

| kuyruk | sunucu | çekirdek / sunucu | bellek / sunucu | GPU | en az | varsayılan / en çok bellek (çekirdek başına) | erişim |
|---|---|---|---|---|---|---|---|
| `orfoz` | 504 × 2 Xeon Platinum 8480+ | 112 | 256 GB | – | 56 çekirdek | 2000 / 2295 MB | herkes |
| `hamsi` | 144 × 2 Xeon 6338 | 56 | 192 GB | – | 28 | 3400 / 3400 MB | herkes |
| `barbun` | 119 × 2 Xeon Gold 6248R | 40 | 384 GB | – | 20 | 8500 / 9500 MB | herkes |
| `barbun-cuda` | 24 | 40 | 384 GB | 2 × P100 16 GB | 20 çekirdek + 1 GPU (**20 / GPU**) | 8500 / 9500 MB | herkes |
| `akya-cuda` | 24 | 40 | 384 GB | 4 × V100 16 GB NVLink | 10 çekirdek + 1 GPU (**10 / GPU**) | 8500 / 9500 MB | herkes; `/tmp` altında 1,4 TB NVMe |
| `palamut-cuda` | 9 × AMD EPYC 7742 | 128 | 1 TB | 8 × A100 80 GB | 16 çekirdek + 1 GPU (**16 / GPU**) | 7500 / 16384 MB | **kısıtlı**; `/localscratch` 12 TB NVMe |
| `kolyoz-cuda` | 24 H100 + 48 H200 sunucu | 64 | 1 TB | 4 × H100 / H200 | 16 + 1 GPU (**16 / GPU**), `-C H100` / `-C H200` | 14–16 GB | **kısıtlı**; `/tmp` 7 TB NVMe |
| `smp` | 1 (orkinos) | 224 | 4 TB | – | 4 | 17000 / 18400 MB | özel |
| `debug` | orfoz, hamsi, barbun, barbun-cuda, akya-cuda | – | – | – | 1 | kuyruğa göre | herkes; **4 saat**, iş başına 1 düğüm |

- Çoğu kuyrukta en uzun süre **3 gün**. GPU kuyruklarında yalnız GPU kullanan iş çalışır (`--gres=gpu:N`, N = sunucu
  başına GPU).
- Çok GPU / çok düğüm örneği (palamut / kolyoz): `sbatch -N 2 -n 8 -c 16 --gres=gpu:4`. Çekirdekler GPU başına
  kuyruğun kuralıyla çarpılır.
- **Süre vermeyen iş 1–2 dakikada sonlandırılır** (sayfalar "bir dakika" ve "2 dakika" diyor); `--time` zorunlu.
  Süreyi gereğinden uzun vermeyin: backfill kısa işleri öne alır (SSS 32).
- Bellek: `--mem` ya da `--mem-per-core` ile artırılır, en çok çekirdek × MaxMemPerCore. Verilmezse çekirdek × varsayılan.
- Sunucu ailesini `debug`'ta `#SBATCH --constraint=<aile>` (örn. `orfoz`) ile seçersiniz.
- Tüm sunucularda `/tmp` NVMe diski; yüksek I/O'lu geçici dosyalar için önerilir.
- Tek sunucuya sığan paralel işi tek sunucuda çalıştırmak genelde daha hızlı (SSS 26); OpenMP sunucu içinde MPI'dan
  genelde iyidir (SSS 27).

## 3. SLURM betiği kuralları

*(`2-temel_bilgiler/slurm-betik-ozellik`, `slurm_komutlari_ve_dosyalari`, `interaktif-is-calistirma`, SSS 22)*

- `-p` kuyruk, `-A` TRUBA hesap adı (grup / proje çalışmasında kullanıcı adından farklı olabilir; ARDEB TBAG
  projelerinde `tbag` hesabı), `-J` ad, `-n` görev, `-c` görev başına çekirdek (varsayılan 1; OpenMP / iş parçacığı için
  `OMP_NUM_THREADS` ile aynı), `-N` düğüm, `--time`, `--mem`, `--gres=gpu:N`, `--mail-type/--mail-user`.
- **`--output`, `--error` ve `--workdir` scratch'te olmak zorunda** (verilmezse `slurm-JOBID.out` iş gönderilen dizine).
- Donanım arızasında iş otomatik **yeniden kuyruğa girer** ve baştan başlar; istenmiyorsa `--no-requeue`. Bizim
  üreticimiz `--resume` ile `.partial`'dan devam eder, yani yeniden kuyruğa girmek zararsız.
- Gövde: `module purge`, sonra `module load …`. Hazır örnek betikler: `/arf/sw/scripts/`.
- İnteraktif / derleme: `srun -p debug [-C orfoz] -N 1 -n 1 -c 55 -A <hesap> -J test --time=0:30:00 --pty /usr/bin/bash -i`
  (orfoz 55, hamsi 54, barbun 20, barbun-cuda 20 + `--gres=gpu:1`, akya-cuda 10 + `--gres=gpu:1`).
- Komutlar: `lssrv` (kuyruk durumu), `sinfo -o "%P %C"`, `squeue -u $USER`, `sacct`, `sstat -j`, `scancel`,
  `scontrol show partition=<ad>`, `avci <ad>` (modül arama).
- Zararsız uyarı: `task/cgroup: plugin not compiled with hwloc support` (SSS 14). `squeue: Socket timed out`:
  yoğunlukta geçici, birkaç dakika sonra tekrar (SSS 15).

## 4. Depolama

*(`1-kaynaklar/arf/arf_depolama_kaynaklari` 2026-04; ARF-ACC aynı sistemi kullanır)*

| yol | amaç | kota | ömür |
|---|---|---|---|
| `/arf/home/$USER` | betikler, küçük kurulumlar, küçük çıktılar | **2 TB ve 500K dosya (inode)**, her iki alan için ayrı | kullanıcı kontrolünde; **merkezi yedek yok** |
| `/arf/scratch/$USER` | hesaplama, büyük veri, paralel I/O | aynı kota | **en çok 30 gün**, otomatik temizlenir |
| `/tmp` (düğüm), `/localscratch` (palamut) | düğüm içi NVMe geçici alan | – | iş bitince gider |

- Kota bilgisi `arf-ui1`'e girince görülür. Eski sayfalar 1000 GB / 100 GB / 1 TB der; güncel olan 2 TB.
- Kota ya da inode dolarsa yeni dosya oluşmaz, iş başlamaz, veri kaybolabilir. Küçük dosyaları `tar` ile birleştirin.
- Ev dizini izinleri yalnız kullanıcıya açıktır (`chmod 700 $HOME`); paylaşım için ACL (`setfacl`, SSS 35).
- Geçici dosyalar: `export TMPDIR=/arf/scratch/$USER/tmp` ve işte `trap 'rm -rf $TMPDIR' EXIT`.
- Dosya aktarımı: `rsync -avP` (paralelde en çok 4–8 bağlantı); daha hızlı aktarım için `arf-ui4` / `arf-ui5` önerilir.

## 5. Yazılım: conda yasak, modül ve konteyner var

*(`2-temel_bilgiler/uygulamalar/Python/*` 2025-12, `konteyner_kullanimi` 2026-04, `1-kaynaklar/arf_ai_ent/*`)*

- **`/arf` altına Anaconda, Miniconda, conda ya da pip ile paket kurulmaz.** Hazır olanlar:
  - Standart Python: `module load comp/python/3.12.2`; conda: `module load miniconda3` (ya da `miniconda-intelpython3`);
    `/arf/sw/lib` altında geniş conda kurulumları var.
  - Yapay zekâ yığını (XGBoost, TensorFlow, Keras, **PyTorch**, Lightning, scikit-learn, transformers, deepspeed…):
    `module purge; module load apps/truba-ai/cpu-2024.0` (CPU) ya da `…/gpu-2024.0` (GPU). Örnek GPU betiği
    `comp/python/ai-tools-kolyoz-1.0` yükler. `module av` / `avci` ile güncel adları görün.
  - Sayfalarda NGSolve, netgen, scikit-fem modülü geçmiyor (gmsh yalnız CFD eğitim sayfalarında anılıyor); `avci ngsolve` ile
    TRUBA'da doğrulayın. Yoksa bizim yığın için konteyner gerekir. Diğer modüller: `lib/cuda/12.4`, `lib/hdf5/1.14.3-oneapi-2023.0`,
    `lib/greasy/greasy` (çok sayıda küçük işi tek tahsisatta koşturma), `comp/cmake/3.31.1`, `comp/gcc/14.1.0`.
- **Apptainer** (singularity): kendi konteynerinizi yerelde ya da `debug` düğümünde (login düğümünde değil) kurup tek
  `.sif` dosyası olarak taşırsınız; kopyala → sandbox aç → yaz → kapat → kullan. GPU için `--nv`. Önbellek:
  `APPTAINER_CACHEDIR`, `APPTAINER_TMPDIR`. Hazır taban: `/arf/sw/containers/miniconda3/miniconda3-container.sif`;
  NVIDIA AI Enterprise imajları (PyTorch 26.03, TensorRT, NeMo…): `/arf/ai-ent/apptainer/sif/nvidia` (ARF-AI, `cuda-ui`).
  Eğitim deposu: `github.com/TRUBA-HPC/Intro2ContainerUsageHPC` (tek GPU, MPI, çok GPU, `.def` örnekleri).
- Konteyner kullanımı: `apptainer exec --nv <imaj>.sif python train.py`; MPI için `mpirun … apptainer exec …`.
- Kod tarafında dosya sayısını azaltmak için tek `.sif` en iyisidir: 500K inode kotasını conda ortamıyla (yüz binlerce
  dosya) tüketmeyiz.
- Uyarı: eğitim bölümündeki PyTorch kurulum sayfası (2025-07) Anaconda kurmayı öğretir ve eskidir; uymayın.

## 6. Erişim

*(`1-kaynaklar/arf/arf_baglanti`, `arf_acc/arf_acc_baglanti`, `openvpn_installation/*`, SSS 1–6)*

- Önce **OpenVPN** (kullanıcı adı ve parolayla; en az OpenVPN 2.4; aynı kimlikle iki yerden bağlanılmaz), sonra SSH ile
  kullanıcı arayüzü: ARF için `arf-ui1…arf-ui5` (ya da Open OnDemand web arayüzü), ARF-ACC / ARF-AI (GPU, palamut,
  kolyoz) için `cuda-ui`. İç ağ adresleri sayfalarda; repoya yazılmaz.
- Diğer sunuculara doğrudan girilmez; işler yalnız `sbatch` / `srun` ile gider (SSS 12).
- Çok düğümlü paralel iş için düğümler arası parolasız SSH gerekebilir (SSS 30).

## 7. Bizim ayarlarla karşılaştırma (`cluster/truba/config.sh`, `setup_env.sh`)

| ayar | bizde | TRUBA kuralı | durum |
|---|---|---|---|
| ortam kurulumu | Miniforge + conda + pip, `$HOME` (`/arf/home`) | `/arf` altında conda / pip yasak (§5) | **kural ihlali**; konteynere taşınmalı |
| `PARTITION`, `CPUS` | `orfoz`, 56 | orfoz en az 56, 56'nın katları | uyumlu |
| `MAX_PARALLEL` | 10 shard işi × 56 = 560 çekirdek | eşzamanlı 160 (standart), 40 (YL), 4 (lisans) | **aşar**; ≤ 2 (160 / 56) ya da hesap sınırı |
| `TIME` | 12 saat | üst sınır 3 gün, `--time` zorunlu | uyumlu |
| `GPU_PARTITION` | `palamut-cuda` | kısıtlı kuyruk | **hesap erişimi doğrulanmadı**; yoksa `akya-cuda` (V100) |
| `GPU_CPUS` | 16 / GPU | palamut 16, kolyoz 16, akya 10, **barbun 20** | `barbun-cuda` için 20 şart |
| `GPU_TIME` | 3 gün | 3 gün üst sınır | uyumlu (sınırda) |
| FP64 | Ritz ve Kp CG float64 | P100, V100, A100, H100 gerçek FP64 verir | tüm herkese açık GPU'lar uygun |
| `--mem` | verilmiyor | orfoz 2000 MB × 56 = 112 GB varsayılan, düğüm 256 GB | 7 işçi × ≤ 12 GB (docs/29) sığar; `CPUS` düşerse `--mem` ekle |
| `DATA_ROOT` | `/arf/scratch/$USER/rfcav3d` | scratch 30 gün | **yedekle**; checkpoint ve H5'i `/arf/home`'a (kota 2 TB) ya da yerele kopyala |
| `ACCOUNT` | boş | `-A` TRUBA hesabı (örneklerde zorunlu görünüyor) | sahibi verir |
| log yolu | `$OUT_DIR/logs` (scratch) | `--output/--error` scratch'te olmalı | uyumlu |
| `setup_env.sh` doğrulama koşusu | login düğümünde 2 geometri | login'de ağır iş yok; "birkaç saniye" kabul edilebilir ama sınırda | `debug` `srun` içine taşınmalı |
| `CACHE_DIR` | geometri başına ~0,5 GB | inode 500K | az dosya, büyük dosya: sorun değil |
| kullanım izni | – | başvuruda belirtilen uygulama dışında iş yok | sahibi, başvuruda bu uygulamayı belirtmiş mi? |

## 8. Açık sorular (sahibe)

1. Hesabın tipi (lisansüstü / akademik) ve gerçek eşzamanlı çekirdek ile çekirdek-saat kotası ne? `ACCOUNT`?
2. `palamut-cuda` / `kolyoz-cuda` erişimi var mı (sözleşmeli proje)? Yoksa eğitim `akya-cuda` (4 × V100) ya da
   `barbun-cuda` üzerinde; `GPU_PARTITION` ve `GPU_CPUS` varsayılanları ve `src/service/truba.py` +
   `tests/test_service_truba.py` birlikte değişir.
3. TRUBA başvurusunda hangi uygulama beyan edildi? NGSolve + PyTorch tabanlı bu iş akışı kapsanıyor mu?
4. `setup_env.sh`'i Apptainer'a çevirelim mi (konteyner `debug` düğümünde kurulur, tek `.sif`)? docs/31 madde 22.
5. `lssrv` ve `sinfo` çıktısını yapıştırırsanız §2 tablosunu güncel değerle doğrularız.
