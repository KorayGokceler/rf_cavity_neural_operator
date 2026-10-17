# 31 — Oturum devri: durum, ölçümler, kararlar, açık işler

> Bu belge her çalışma oturumunun sonunda güncellenir. Bir sonraki ajan (hangi model ya da araç olursa
> olsun) sohbet geçmişinden değil buradan başlar. Önce `AGENTS.md`'yi okuyun.
> Son güncelleme: 2026-10-17, dal `claude/field-model`.

## 1. Durum

| alan | durum |
|---|---|
| Etiket çözücüsü | Adaptif p3, eğri mesh (docs/30). Analitik pillbox kontrolü geçiyor: 157k DOF'ta f 6e-7, Q0 3e-5, R/Q 1e-5, Epk 2e-4, Bpk 4e-4. |
| Alan modeli (`field3d`) | Kod hazır; tek örnek ezberleme testinde frekans hatası 1e-6 – 9e-6 (docs/28). **Ölçekli eğitim yapılmadı.** |
| Model uzayı tabanı | p2, ≤ 10k tetlik model mesh'inde ölçülen frekans tabanı %0,04–0,9 (docs/30). Model bunun altına inemez. |
| Web arayüzü | Çalışma alanı alanı tarayıcıda hesaplıyor (docs/26). "Model" kaynağı hâlâ N0 modeli; alan modeli arayüze bağlanmadı. |
| TRUBA | Alan etiketi koşusu henüz yapılmadı (`TEST=1` referans koşuları açık iş). |
| README | Model bölümü N0 modelini anlatıyor; alan modeline göre güncellenmeli. |

Son commitler: `866e4a8` (çalışma alanı istemci tarafında), `971834e` (kesit kaydırıcısı, 3B hacim görünümü,
oklar), `e36d3fd` (pahı mesh'lenemeyen çekilişte keskin kenarlı yedek), `584abc2` (alan etiketi üretimi
arayüzde; onarılmış mesh'lerin model mesh'i), `2249fbd` (adaptif etiketler).

## 2. Yöntem tek bakışta

```
CAD → eğri mesh → token'lar: tet başına 14 kuadratür noktası, 9 özellik
    → gövde: RFF + MLP + 4 hacim ağırlıklı lineer dikkat bloğu
    → başlık: her noktada m = 24 vektör ψ_j
    → sabit doğrusal harita: V = C·P·ψ  (HCurl p2 katsayıları, duvar satırları 0)
    → projeksiyon: Z = Kp⁻¹ GᵀMV (CG, float64),  PV = V − G Z
    → Ritz: VᵀKV c = θ M_div c,  M_div = VᵀMV − (GᵀMV)ᵀ Z
    → f_k = c₀ √θ_k / (2π·ölçek),  E_k = PV c_k,  H = −curl E / (ω μ₀)
```

- Min-max teoremi: θ_k ≥ λ_h,k (aynı uzaydaki FE özdeğeri). Ağın işi, gerçek modları span(V) içine sokmak.
- Projeksiyonun sebebi: `curl ∇φ = 0`. Gradyan karışan bir alanın Rayleigh bölümü yapay olarak düşer;
  saf gradyan θ = 0 verir.
- Kayıp: `1,0·ort √r_k` (span, kütle + enerji normu) + `0,1·frekans MSE` + `0,01·(−log Σ 1/θ)` +
  `0,01·korelasyon`. Hedef alanlar p3 etiketin model mesh'indeki p2'ye izdüşümü; frekanslar p3'ten.
  `qoi_weight = 0` (QoI kayıp operatörleri N0'a göre yazılmış).

## 3. Bu oturumdaki ölçümler

### 3.1 Web arayüzü

Tablo docs/26 §2'de. Özet: mod, faz, E/H ve kesit değişimleri artık ağa gitmiyor. 10 adımlık hacim
faz taraması 6,2 s → 0,65 s.

### 3.2 Bir eğitim adımı (CPU, 4 iş parçacığı)

Örnek: pillbox_pipes, 13 052 tet, 182 728 token, 141 654 DOF, 4 mod. Model `field_3d.yaml` boyutlarında.

| parça | süre |
|---|---|
| gövde (4 dikkat bloğu), ileri | 8–9 s, adımın ~%75'i |
| Kp CG, ileri + geri | 1,5 + 1,5 s, 110 iterasyon |
| başlık + ψ → DOF | ~0,5 s |
| 24×24 eig + kayıp Gram'ları | < 0,1 s |
| **toplam** | **~24 s / örnek** |

### 3.3 Veri hazırlama (örnek başına, her epoch)

| parça | süre |
|---|---|
| eğri mesh'i yeniden kurma | 0,2 s |
| p2 operatörleri (K, M, G, Kp, C) | 5,5–6,5 s |
| token özellikleri (duvar kd-ağacı + torsiyon çözümü) | 2,8 s |
| collate | 1,6 s |

Operatörler örnek başına 26–30M sıfır olmayan eleman, yaklaşık 0,3 GB. GPU'da gövde hızlanınca darboğazın
bu CPU işi olması beklenir. Bu bir tahmin; GPU'da ölçülmedi.

### 3.4 Etiket çözücüsü: doğrudan çözüm ve LOBPCG

Bakır pillbox, p3, 4 iş parçacığı.

| DOF | yöntem | süre | tepe bellek |
|---|---|---|---|
| 118k | sparsecholesky + ARPACK (bugünkü) | 32 s | 2,2 GB |
| 118k | LOBPCG + BDDC, soğuk, 80 iterasyon | 73 s | 1,6 GB |
| 358k | bugünkü | 175 s | 8,4 GB |
| 358k | LOBPCG + BDDC, soğuk, 80 iterasyon | 286 s | 4,5 GB |

Sıcak başlatma (önceki uyarlamalı seviyenin modlarıyla, 358k DOF):

| başlangıç | iterasyon | süre | özdeğer hatası |
|---|---|---|---|
| sıcak | 10 | 36 s | 1,9e-8 |
| sıcak | 20 | 65 s | 2,3e-11 |
| soğuk | 20 | 64 s | 8,3e-4 |
| soğuk | 40 | 123 s | 1,7e-8 |

Kurulum maliyeti +26 s. Modlar yeni mesh'e NGSolve'un mesh'ler arası `Set` işlemiyle aktarılırken 68 s
sürdü; sıcak başlatmanın kazancı için bu aktarımın hızlı yazılması gerekiyor.

İki noktadan kaba uzatma: doğrudan çözücünün belleği 900k DOF'ta ~25 GB'a çıkabilir (docs/30 ~12 GB
diyor). TRUBA'da ölçülmeli.

### 3.5 Kp ve CG deneyleri

Örnek: 13 052 tet; Kp 48 925 potansiyel (1 194'ü köşe), 1,88M sıfır olmayan eleman.

- **Jacobi ön koşullu CG (bugünkü):** 101 iterasyon; eig(D⁻¹Kp) ∈ [0,0267; 4,5], koşul sayısı ~170.
- **İki seviyeli ön koşullayıcı** (köşe potansiyellerinde tam çözüm + Jacobi): 98 iterasyon; kazanç yok.
  Kötü koşullu yönler yüksek mertebe potansiyellerde.
- **CG'siz ceza yöntemi** `K + s·MG D⁻¹ GᵀM`, s = 2λ_max/c₁ = 1964. Hedef özdeğerlere göre en büyük
  göreli hata:

| alandaki gradyan kütle payı | projeksiyon (bugünkü) | ceza |
|---|---|---|
| 0 | 5e-6 | 4e-6 |
| 1e-6 | 6e-6 | 6,6e-4 |
| 1e-4 | 5e-6 | 6,7e-2 |
| 1e-2 | 5e-6 | 6,7 |

  Ceza hiçbir zaman projeksiyonun altına inmiyor (garanti korunuyor), ama ağın çıktısında gradyan payı
  ~1e-8'in altına inmedikçe frekansları bozuyor. **Reddedildi.**
- **Kp'nin tam LU'su (SciPy SuperLU):** 15 dakikada bitmedi; gerçek bir Cholesky kütüphanesi gerekiyor
  (CHOLMOD ya da NGSolve `sparsecholesky`).

## 4. Kararlar ve gerekçeleri

- **E formülasyonu kalıyor.** H formülasyonu CG'yi kaldırmıyor (Neumann Laplace'ı yine gerekli).
  Kulplu ailelerde (spoke, HWR, DTL) gradyan projeksiyonunun silemediği harmonik alanlar üretiyor.
  Tek avantajı duvardaki H'nin (Q0, Bpk) doğrudan çıkması; karşılığında R/Q ve Epk zayıflıyor.
- **Projeksiyon kalıyor;** yerine ceza yöntemi konmuyor (§3.5).
- **`E = curl A` (vektör potansiyel) yolu:** HCurl'deki A'nın curl'ü HDiv'de olduğu için FE uyumlu değil.
  Duvar koşulu yapıya gömülemiyor. Yalnızca bir araştırma yönü (sinir ağı tabanlı Ritz), garanti kaybıyla.
- **İki mesh (etiket / model):** doğruluk ile eğitim maliyeti tek mesh'te birlikte sağlanamıyor (docs/30 §5).
- **Açık kaynak çözücü araştırması:**
  - NGSolve çekirdek olarak kalıyor (Python, OCC, eğri yüksek mertebe, mevcut uyarlamalı döngü).
  - Bağımsız referans için Palace (AWS, Apache-2.0, 0.18.1 – Eylül 2026): yüksek mertebe, eğri mesh,
    empedans sınır koşuluyla doğrudan Q, AMR. Her aileden küçük bir alt kümeyi çapraz doğrulamak için.
  - MFEM/PyMFEM (HypreAME) yeniden yazım gerektiriyor. dolfinx'in avantajı yok. ACE3P açık kaynak değil.
    FDTD (openEMS, MEEP) Q0 ve R/Q için yeterince hassas değil.

## 5. Stratejik değerlendirme (tartışmaya açık)

- **Yapısal üst sınır:** model, model mesh'indeki p2 FE çözümünden daha iyi olamaz. Mesh'leme ve
  operatör kurma da modelde aynen kaldığı için o çözümden ancak birkaç kat hızlı olabilir. "CST
  doğruluğu + çok daha hızlı" hedefiyle bu kurgu arasında bir gerilim var.
- **Önerilen yönler:**
  - **hedefi netleştirmek:** genel CAD çözücüsü mü, aile içi tasarım vekili mi;
  - **hibrit mod:** modelin alt uzayını tam bir LOBPCG çözümüne sıcak başlangıç olarak vermek; doğruluk
    çözücüden garanti, hız modelden;
  - **büyük üretimden önce küçük bir pilot eğitim;**
  - **etiket toleranslarını ikiye ayırmak:** eğitim seti için gevşek, test seti ve CST kıyası için sıkı;
  - **kayba duvar terimi** (duvarda H ve E) ve **log-frekans kaybı** eklemek;
  - **eksenel simetrik aileler** için 2B çözücüyle ucuz etiket.

## 6. Açık işler (öncelik sırasıyla)

Bir işe başlayan ajan sahip ve dal sütunlarını doldurur; bitince durumu günceller.

| # | iş | sahip | dal | durum |
|---|---|---|---|---|
| 1 | Pilot eğitim: 1–2 aile, birkaç yüz örnek. Öğrenme eğrisi ve aileler arası genelleme. | — | — | açık |
| 2 | Veri hattı: token özelliklerini veri üretilirken H5'e yazmak (örnek başına −2,8 s); operatörleri yerel diske önbelleklemek ya da yüklenen örnekleri birkaç adım yeniden kullanmak. | — | — | açık |
| 3 | CG'yi eğitim grafiğinden çıkarmak: Kp'yi örnek başına bir kez Cholesky ile ayrıştırıp üçgensel çözüm. Alternatifler: p-multigrid ön koşullayıcı, tol 1e-6 denemesi. | — | — | açık |
| 4 | TRUBA referans koşuları (`TEST=1`, `MAX_NDOF` 900k) ve örnek başına belleğin ölçümü (§3.4). | — | — | açık |
| 5 | Alan modelini web arayüzüne bağlamak; README'yi alan modeline göre güncellemek. | — | — | açık |
| 6 | Hibrit mod prototipi (model alt uzayı → LOBPCG). | — | — | açık |
| 7 | Görev E: gövdede token alt örnekleme. | — | — | açık |
| 8 | Google Drive bağlayıcısı oturumda "Forbidden" döndü; kullanıcının araç izinlerini açması gerekiyor. | sahip | — | kullanıcıda |

1, 2 ve 5 birbirine dokunmuyor; paralel yürüyebilir. 3, 2 ile aynı veri hattına dokunuyor; sıraya konmalı.

## 7. Ölçümleri yeniden üretme

Ölçüm betikleri oturumun geçici alanında kaldı; yöntemleri:

- **§3.2:** `FieldDataset` ile bir örnek yükle; `CavityLightning(model_type="field3d", hidden_dim=128,
  n_heads=4, n_basis=24, eigenspace_kwargs={n_layers: 4, kp_maxiter: 3000})`. `_compute_loss_3d`,
  `backward` ve `hcurl._pcg` etrafına zamanlayıcı koy.
- **§3.3:** `field_labels.read_field_group`, `highorder.field_operators` ve `point_features` sürelerini ayrı ölç.
- **§3.4:** `scripts/validate_field_labels.py`'deki pillbox, `ad.mesh_remesher(..., curvaturesafety=1.5)`
  ve bir düzgün inceltme. `highorder.solve_modes` ile `ngsolve.solvers.LOBPCG` + `Preconditioner(a, "bddc")`
  + gradyan projeksiyonu karşılaştırması.
- **§3.5:** Kp, D⁻¹Kp'nin uç özdeğerleri (CG'nin Lanczos katsayılarından), ceza Gram'ı
  `VᵀKV + s·(GᵀMV)ᵀ D⁻¹ (GᵀMV)` ile Ritz.

## 8. Çalışma düzeni: birden çok model, birden çok ajan

Ayrıntılar `docs/32_AGENT_WORKFLOW.md`'de: kurulum, roller ve modeller, worktree ile paralel iş,
görev yaşam döngüsü, mesaj şablonları. Özet:
- tek doğruluk kaynağı repo (`AGENTS.md`, bu belge, `docs/`);
- ajan başına bir dal ve worktree (`scripts/agent_worktree.sh`);
- inceleme farklı bir modelden; hakem testler;
- Claude Code'da `/task`, `/verify`, `/handoff` ve dört alt ajan;
- her oturumun sonunda bu belgenin §1, §3, §4 ve §6'sı güncellenir.
