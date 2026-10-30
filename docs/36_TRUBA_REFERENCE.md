# 36 — TRUBA referansı: kuyruklar, depolama, erişim, yazılım ve bizim ayarlarla karşılaştırma

> **Durum (2026-10-30).** Kaynak: `docs.truba.gov.tr` (TRUBA Kullanıcı Dokümanları). Sayfalar sandbox'tan
> **doğrudan okunamadı** (§1); aşağıdaki bilgiler arama özetlerinden geliyor ve çoğu sayfada tutarsızlık var.
> Kuyruk sınırlarını, kotaları ve erişim kurallarını bir koşuyu planlamadan önce TRUBA'da `sinfo`,
> `scontrol show partition=<kuyruk>` ve kota komutlarıyla doğrulayın. Bu belge `truba-ops` ajanının ve
> `/truba-run` komutunun platform bilgisi kaynağıdır; betikler için bkz. `cluster/truba/README.md`, docs/29 §5.

## 1. Siteye nasıl bakılır (ve neden doğrudan okunamadı)

- `docs.truba.gov.tr:443`'e proxy üzerinden tünel kuruluyor, ancak TLS doğrulaması **başarısız**: sunucu yalnız
  yaprak sertifikayı (`*.truba.gov.tr`, TÜBİTAK ULAKBİM) gönderiyor; imzalayan ara sertifika (`TUBITAK Kamu SM SSL
  Sertifika Hizmet Saglayicisi - Surum 2`) eksik ve sandbox'ın güven deposunda yok. `curl` `error 60 / 21`, `WebFetch`
  `503` veriyor.
- **TLS doğrulamasını kapatmayın**, internetten bulunan bir CA'ya doğrulamasız güvenmeyin (AGENTS.md ve proxy
  kuralları). Sertifika zincirini sahibi doğrulanmış biçimde (PEM + SHA-256 parmak izi) verirse `--cacert` ile
  yalnız bu çağrılar için kullanılabilir; ya da sayfalar tarayıcıdan PDF / Markdown olarak
  `docs/research/truba/` altına konabilir.
- Çalışan yol: `WebSearch` (`allowed_domains: ["docs.truba.gov.tr"]`) sayfa başlıklarını, URL'leri ve özetleri
  veriyor. Tam metin değil; sayı vermeden önce ilgili URL'yi sahibe doğrulatın. `www.truba.gov.tr` bu oturumda
  engelli (403).

Gördüğümüz sayfalar (özet olarak okundu, tam metin okunmadı):

| konu | URL (`https://docs.truba.gov.tr/…`) |
|---|---|
| kümeler, kuyruk tablosu | `2-temel_bilgiler/hesaplama_kumeleri.html`, `1-kaynaklar/arf/arf_kuyruk_bilgisi.html` |
| SLURM betik özellikleri | `2-temel_bilgiler/slurm-betik-ozellik.html` |
| GPU kılavuzu | `2-temel_bilgiler/gpu.html`, `1-kaynaklar/arf_acc/index.html` |
| depolama, kota | `1-kaynaklar/arf/arf_depolama_kaynaklari.html` |
| bağlantı | `1-kaynaklar/arf_acc/arf_acc_baglanti.html`, `2-temel_bilgiler/open_ondemand.html` |
| Python, conda, modüller | `2-temel_bilgiler/uygulamalar/Python/python.html`, `2-temel_bilgiler/uygulamalar/Python/jupyter.html` |
| PyTorch | `3-egitim_seminerler/pytorch/intro_installation.html`, `…/pytorch/dist.html` (çok düğümlü, çok GPU) |
| konteyner | `2-temel_bilgiler/konteyner_kullanimi.html` (okunmadı) |
| hesap, proje, kota politikası | `0-truba/kullanici_basvurulari.html`, `0-truba/proje_basvurulari.html`, `0-truba/uyelik_rehberi.html`, `2-temel_bilgiler/ybh-akis.html` |
| ücretli / öncelikli kullanım | `4-proje_hesaplari_ucretli_kullanim/…` (HSB / VSB), `…/kullanim-maliyeti/kullanim-maliyet.html` |
| SSS, destek | `5-sikca_sorulan_sorular/index.html`, `0-truba/truba_iletisim.html` |
| MareNostrum 5 (BSC) kaynağı | `1-kaynaklar/mn5/…` (okunmadı; ARF'ten ayrı bir sistem) |

## 2. Kuyruklar (ARF kümesi)

Çoğu kuyrukta en uzun süre **3 gün**; `debug` kuyruğu **4 saat, iş başına 1 düğüm**, test içindir (orfoz, hamsi,
barbun, barbun-cuda, akya-cuda üzerinde). **Betikte süre verilmeyen iş 2 dakikada sonlandırılır** (bir sayfa);
bizim betiklerimiz her zaman `-t` verir.

| kuyruk | tür | kural (sayfalara göre) | not |
|---|---|---|---|
| `orfoz` | CPU | çekirdek 56'nın katları; ~2000 MB/çekirdek; 256 GB RAM / düğüm; `--constraint=orfoz` kullanılabilir | düğüm 112 çekirdek (ARF tablosu) ya da 56×2 CPU (kümeler sayfası): tutarsız |
| `hamsi` | CPU | 56×n çekirdek (MATLAB sayfası) ya da en az 28 (kümeler sayfası); ~3400 MB/çekirdek | tutarsız |
| `barbun` | CPU | 40×n çekirdek (MATLAB sayfası) ya da en az 20; ~8500 MB/çekirdek | tutarsız |
| `barbun-cuda` | GPU | düğümde 40 çekirdek, 2 GPU; en az 20 çekirdek + 1 GPU; **20 çekirdek / GPU** | tüm kullanıcılara açık; GPU modeli sayfalarda tutarsız (biri P100) |
| `akya-cuda` | GPU | düğümde 40 çekirdek, 4 GPU; en az 10 çekirdek + 1 GPU; **10 çekirdek / GPU** | tüm kullanıcılara açık; GPU modeli doğrulanmadı |
| `palamut-cuda` | GPU | düğüm başına 8 × A100; **16 çekirdek / GPU** | **yalnız belirli araştırma merkezlerine / sözleşmeli projelere açık** (ARF-ACC sayfası) |
| `kolyoz-cuda` | GPU | değerler bulunamadı | palamut-cuda gibi kısıtlı |

- GPU kuyruklarında **yalnız GPU kullanan işler** çalışır; `--gres=gpu:N` zorunlu.
- Güncel değer için: `sinfo`, `scontrol show partition=<ad>`.
- Her kullanıcıya tanımlı **eşzamanlı çekirdek sınırı**, **çekirdek-saat kotası** ve **disk kotası** var. Çekirdek-saat
  kotası bitince işler kuyruğa girer ama başlamaz; artırım için TRUBA'ya başvurulur. Ücret karşılığı öncelikli kuyruk
  (HSB / VSB) ayrı bir hizmettir.

## 3. Depolama

- `/arf/home/$USER` (küçük dosyalar, kod, ortam) ve `/arf/scratch/$USER` (hesaplama, büyük veri; docs'a göre
  hesaplar scratch altında yapılmalı).
- Kota (tutarsız): güncel görünen sayfa **2 TB ve 500K dosya (inode)** / kullanıcı, her iki alan için; eski MATLAB
  sayfası ev 100 GB, scratch 1 TB; SSS 1000 GB. Kota dolunca yeni dosya açılamaz, iş başlamayabilir.
- **Scratch saklama süresi tutarsız** (bir sayfa en çok ~30 gün, bir başkası iş bitince silinebilir diyor).
  Eğitilmiş checkpoint ve önemli H5'ler yalnız scratch'te tutulmamalı.

## 4. Erişim

- Önce **OpenVPN**, sonra SSH ile kullanıcı arayüzü: ARF için `arf-ui1…arf-ui5`; ARF-ACC (GPU) için ayrı sunucu
  `cuda-ui`. Open OnDemand arayüzü de var. Hostname ve hesap bilgileri repoya yazılmaz (AGENTS.md); sahibi verir.
- Hesap: portalda başvuru, TRUBA Kullanım Politikası onayı, e-posta doğrulaması, ekip onayı. Yüksek lisans
  öğrencisi için danışman onayı gerekir.
- İşler yalnız kullanıcı arayüzü sunucularından `sbatch` ile gönderilir (login düğümünde ağır iş çalıştırılmaz).
- Örneklerde `#SBATCH -A kullanici_adi` biçimi kullanılıyor: hesap adı çoğu zaman kullanıcı adı ya da proje
  adıdır; `config.sh`'te `ACCOUNT` boş, sahibi verir.

## 5. Yazılım

- Modüller: `module load miniconda3` (ya da `miniconda-intelpython3`); yapay zekâ yığını:
  `module purge && module load apps/truba-ai/gpu-2024.0` (PyTorch, Lightning, TensorFlow, scikit-learn, …;
  güncel modül adı için `module av`).
- SLURM betiğinde ortam: `module purge`, sonra `source …/bin/activate` ya da
  `eval "$(<conda>/bin/conda shell.bash hook)"` ve `conda activate`.
- PyTorch kurulum sayfasındaki `pip` komutu eski (torch 1.9 / cu111): güncel komutu pytorch.org'dan alın.
- Çok GPU / çok düğüm eğitimi için TRUBA'nın PyTorch dağıtık eğitim sayfası var (`dist.html`).
- Konteyner kullanımı sayfası var (okunmadı).

## 6. Bizim ayarlarla karşılaştırma (`cluster/truba/config.sh`)

| ayar | değerimiz | TRUBA bilgisi | durum |
|---|---|---|---|
| `PARTITION` / `CPUS` | `orfoz` / 56 | orfoz en az 56, 56'nın katları | uyumlu |
| `TIME` | 12 saat | çoğu kuyrukta üst sınır 3 gün | uyumlu |
| `GPU_TIME` | 3 gün | üst sınır tam 3 gün; bazı kuyrukta daha kısaysa iş reddedilir | `scontrol show partition` ile doğrula |
| `GPU_PARTITION` | **`palamut-cuda`** | yalnız belirli merkezlere / sözleşmeli projelere açık | **sahibin hesabı bu kuyruğa erişebiliyor mu? Yoksa `akya-cuda` ya da `barbun-cuda`** |
| `GPU_CPUS` | 16 çekirdek / GPU | palamut 16, akya 10, **barbun 20** | `barbun-cuda` seçilirse `GPU_CPUS=20`; akya için ≥ 10 yeterli |
| `--mem` | verilmiyor | orfoz varsayılanı ~2000 MB × çekirdek ≈ 112 GB (56 çekirdek), düğüm 256 GB | 7 işçi × ≤ 12 GB (docs/29) sığıyor; `CPUS` düşürülürse varsayılan bellek de düşer, o zaman `--mem` ekle |
| `DATA_ROOT` | `/arf/scratch/$USER/rfcav3d` | scratch önerilen yer; saklama süresi belirsiz | koşu bitince sonuçları scratch dışına yedekle |
| dosya sayısı | parça başına 1024 geometri / H5 | 500K inode sınırı | H5 hattında sorun yok; `CACHE_DIR` (geometri başına çok dosya) açılacaksa inode say |
| FP64 | Ritz ve Kp CG float64 (AGENTS.md) | palamut: A100; akya / barbun-cuda GPU modeli doğrulanmadı | GPU kuyruğunu seçmeden önce FP64 hızını kontrol et |
| test | `TEST=1` küçük koşu | `debug` kuyruğu 4 saat, 1 düğüm | küçük denemeler `PARTITION=debug` olabilir (CPU) |

## 7. Açık sorular (sahibe)

1. Hesap `palamut-cuda`'ya erişebiliyor mu? Değilse varsayılan `GPU_PARTITION` ve `GPU_CPUS` değişmeli (config.sh,
   `src/service/truba.py` varsayılanları ve `tests/test_service_truba.py` birlikte güncellenir).
2. Hesabın çekirdek-saat, eşzamanlı çekirdek ve disk kotası ne? (`ACCOUNT` değeri de.)
3. `/arf/scratch` temizleme politikası ne? (30 gün sınırı doğru mu?)
4. `sinfo` / `scontrol show partition` çıktılarını bir kez yapıştırın: §2 tablosunu bu çıktıyla değiştirip
   "tutarsız" notlarını kaldırırız.
5. Sertifika zinciri ya da sayfaların PDF / Markdown kopyası verilirse bu belge tam metinle yeniden yazılır.
