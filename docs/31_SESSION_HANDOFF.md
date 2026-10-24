# 31 — Oturum devri: durum, ölçümler, kararlar, açık işler

> Bu belge her çalışma oturumunun sonunda güncellenir. Bir sonraki ajan (hangi model ya da araç olursa
> olsun) sohbet geçmişinden değil buradan başlar. Önce `AGENTS.md`'yi okuyun.
> Son güncelleme: 2026-10-24, dal `claude/field-model`.

## 1. Durum

| alan | durum |
|---|---|
| Etiket çözücüsü | Adaptif p3, eğri mesh (docs/30). Analitik pillbox kontrolü geçiyor: 157k DOF'ta f 6e-7, Q0 3e-5, R/Q 1e-5, Epk 2e-4, Bpk 4e-4. |
| Alan modeli (`field3d`) | Kod hazır; tek örnek ezberleme testinde frekans hatası 1e-6 – 9e-6 (docs/28). **Ölçekli eğitim yapılmadı.** |
| Model uzayı tabanı | p2, ≤ 10k tetlik model mesh'inde ölçülen frekans tabanı %0,04–0,9 (docs/30). Model bunun altına inemez. |
| Web arayüzü | Çalışma alanı alanı tarayıcıda hesaplıyor (docs/26). "Model" kaynağı hâlâ N0 modeli; alan modeli arayüze bağlanmadı. |
| TRUBA | Alan etiketi koşusu henüz yapılmadı (`TEST=1` referans koşuları açık iş). |
| Geometri aileleri | Literatürle karşılaştırıldı ve düzeltmeler yapıldı (docs/33 §12): ridged_box / box ekseni, borusuz standart E_acc uzunluğu, eksen E_z'nin saklanması (her β için), eliptik yarım hücre çözücüsü (tolerans, çok başlangıç, re-entrant duvar), çapalar, aralıklar. Veri kümesi etiketi `_v2`. Yeni aileler ayrı iş. |
| README | Model bölümü N0 modelini anlatıyor; alan modeline göre güncellenmeli. |

Son commitler:
- bu oturumun son commit'i: geometri incelemesi (docs/33, docs/research/2026-10_geometry);
- `6ccb65f` inceleme düzeltmeleri (burulma CG yedeği, paylaşılmayan batch değerleri, int64 ofsetler);
- `843146f` gövdede gradyan kontrol noktası (eğitim belleği 9,9 → 3,8 GB);
- `d832941` blok köşegen CSR'yi doğrudan kuran collate;
- `6e17f3d` 2,3 kat hızlı alan operatörleri (bit düzeyinde aynı çıktı) ve aşama profil betiği;
- `99adcee`, `7e7db32` ajan düzeni (AGENTS.md, CLAUDE.md, alt ajanlar, docs/31–32).

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

### 3.6 Eğitim ve çıkarım maliyeti: ölçüm ve optimizasyon

Ölçüm betiği: `python scripts/profile_field_step.py <H5> --sample 1 [--device cuda] [--cprofile]`.
Örnek §3.2'deki geometri. CPU, 4 iş parçacığı, 4 mod, `field_3d.yaml` boyutları, 0,854 M parametre.

**Temel durum** (optimizasyondan önce):

| aşama | süre | not |
|---|---|---|
| mesh (üretici, CAD'den eğri mesh) | 8,3 s | yalnız çıkarımda; eğitimde `.vol`'dan 0,24 s |
| `field_operators` | 8,8–10,3 s (6 koşu) | |
| `field_item` / `field_collate` | 0,13 / 2,4 s | |
| çıkarım ileri geçişi (gradyansız) | 8,1–9,3 s | CG 1,8 s (110 iterasyon); tepe RSS 3,1 GB |
| eğitim adımı | ileri 8,8–20,6 s, geri 13–15 s | CG 1,5–1,8 s + 1,5–1,8 s; **tepe RSS 10,1 GB** |
| ağın yoğun FLOP'u (ileri) | 309 GFLOP | GPU'da ~0,1 s mertebesi (ölçülmedi) |

Sonuç: ağ küçük. Maliyet CPU'daki operatör kurulumunda, batch'leme dönüşümlerinde ve aktivasyon belleğinde.

**Faz 1a — `field_operators`** (`highorder._tosp`, `field_operators`, `point_features`):

- NGSolve matrisleri kendi CSR dizilerinden kopyalanıyor. COO üzerinden yeniden sıralama kalktı
  (K: 0,66 s → 1 ms).
- K ve M ölçeklemesi yerinde yapılıyor.
- `ConvertOperator` `TaskManager` içinde çalışıyor (1,73 s → 0,57 s, aynı sonuç).
- C'nin sütun permütasyonu indeks yeniden etiketlemesiyle yapılıyor.
- Duvar kd-ağacı sorgusu paralel (1,29 s → 0,4 s).
- Torsiyon özelliği seyrek Cholesky yerine Jacobi-CG ile çözülüyor, tol 1e-12 (1,35 s → 0,55 s).
  Float32 özelliklerde 182 728 girdinin hiçbiri değişmedi. İncelemede 95k tetlik mesh'te 1,33M girdinin 7'si
  ≤ 6e-8 farklı çıktı; bit eşitliği mesh'e bağlı, fark zararsız. CG `maxiter`'da sessizce durursa seyrek
  Cholesky'ye dönülüyor (`test_torsion_feature_falls_back_to_cholesky_when_cg_stalls`).
- **Sonuç:** 6 koşuda 3,5–6,3 s, ortanca ~4,2 s (önce ortanca ~9,5 s).
- **Doğruluk:** bu örnekte K, M, G, Kp, C, P, qx, qw ve token özellikleri bit düzeyinde aynı. İnceleme, iki duvar
  kabuklu 9,2k tetlik pillbox'ta da aynı sonucu buldu (6,81 s → 2,92 s).
- Denenip alınmayanlar:
  - `Gᵀ(MG)` sırası 0,27 s kazandırıyor ama Kp'yi 4e-17 değiştiriyor; bit eşitliği için alınmadı.
  - Statik yoğunlaştırmalı torsiyon daha yavaş (1,3 s).

**Ölçüm notu:** bu bulut sanal makinesinde belleğe ilk dokunuş ~0,2 GB/s (800 MB: 3,8 s), sonra ~4,8 GB/s.
Bir sürecin ilk birkaç çağrısı bu yüzden 2–10 kat yavaş görünüyor. Tablolardaki "kararlı" sayılar ısınmadan
sonraki koşular; gerçek donanımı onlar temsil ediyor. Eğitimin ilk adımındaki 20,6 s de bu etkiden.

**Faz 1b — `field_collate`** (`dataset_3d._block_diag`, CSR yolu):

- Bloklar SciPy'nin COO üzerinden `block_diag`'i yerine doğrudan birleştiriliyor; her çıktı dizisi için
  tek bellek ayırma yapılıyor.
- **Sonuç (kararlı durum):** B=1 0,80 s → 0,34 s; B=2 1,58 s → 0,61 s. Tek bloklu batch değerleri kopyalıyor;
  veri kümesinin dizileriyle bellek paylaşmıyor (inceleme bulgusu, kopyasız 0,25 s idi).
- **Doğruluk:** batch tensörleri eskisiyle bit düzeyinde aynı (20 anahtar).
- Yeni birim test `test_block_diag_csr_matches_scipy_block_diag`: sırasız, tekrarlı, açık sıfırlı, dolgulu
  girdiler.

**Faz 2a — aktivasyon kontrol noktası** (`model.eigenspace.grad_checkpoint`, `field_3d.yaml`'da açık):

- Eğitimde encoder ile her bloğun dikkat ve FFN yarılarının yalnız girdileri tutuluyor; backward'da yarılar
  tek tek yeniden hesaplanıyor.
- Sıralı denemeler (3 tekrar, tepe RSS):

| ayar | tepe RSS | kararlı adım (ileri + geri) |
|---|---|---|
| kapalı | 9,89 GB | 21,5–24,9 s |
| blok başına | 4,56 GB | 24,4–28,6 s |
| yarım blok başına | 4,07 GB | 21,8–24,0 s |
| yarım blok + encoder (**seçilen**) | **3,79 GB** | 21,4–22,5 s |

- **Doğruluk:** gradyanlar bit düzeyinde aynı
  (`test_grad_checkpoint_gives_the_same_outputs_and_gradients`).
- **Uçtan uca:** `train.py --fast_dev_run` ile train, val ve test adımları geçiyor.
- Yeniden hesaplamanın maliyeti CPU'da ölçüm gürültüsünde kaldı. GPU'da ölçülmeli.
- Aynı koşudaki çıkarım ileri geçişi 5,5–6,6 s çıktı (önce 8,1–9,3 s). Çıkarım kodu değişmedi; bu fark
  makine değişkenliği.

**Faz 0–2a özeti** (kararlı durum, örnek başına):

| | önce | sonra |
|---|---|---|
| veri hazırlama (operatörler + collate) | ~10,3 s | ~4,5 s |
| eğitim tepe belleği | 9,9 GB | 3,8 GB |

**Faz 1c — operatör önbelleği:** mevcut `dataset.cache_dir` (geometri başına pickle) hedefi zaten
karşılıyor, kod değişmedi.

- Önbellekten yükleme 0,16–0,19 s (sayfa önbelleği sıcak); ilk kurulum + yazma 5,9–7,0 s.
- Boyut: geometri başına 332–382 MB. 2000 geometri ≈ 0,7 TB.
- İçerik: K, M 97 MB, C 135 MB, Kp 22 MB, P 14 MB; indeksler int32.
- K ile M'nin deseni aynı; paylaşılırsa ~%9 kazanç. Değmediği için yapılmadı.
- Disk kotası yetmiyorsa önbelleksiz kurulum örnek başına ~4,5 s.

**Faz 3a ön ölçümü — Kp'nin seyrek Cholesky'si** (NGSolve `sparsecholesky`, aynı Kp, CPU, 24 sağ taraf):

| yöntem | süre |
|---|---|
| ayrıştırma | 0,92–0,96 s |
| çözüm | 0,54–0,58 s |
| torch CG, tol 1e-8 | 1,66–1,77 s (130 iterasyon) |

- Çözümler 6e-10 içinde aynı.
- Eğitim adımı başına (ileri + geri) CG ~3,4 s, Cholesky ~2,0 s. Adımın %6–10'u kadar kazanç.
- GPU'da CG büyük olasılıkla bunun çok altında. CPU'daki bir ayrıştırma GPU'daki eğitimi yavaşlatır;
  NGSolve nesnesi DataLoader işçilerinden de taşınamaz.
- **Karar:** GPU ölçümünden önce yapılmıyor.

**Sıradaki adımlar:**
- **3a:** yalnız GPU ölçümü CG'yi pahalı gösterirse.
- **2b — bf16 karışık hassasiyet** (GPU).
- **GPU ölçümü:** Colab'da `profile_field_step.py --device cuda`.

### 3.7 Eğitim veri yolu: her epoch / adımda tekrarlanan işler (açık iş 12)

Ölçüm düzeni. Örnekler: `pillbox_pipes` (8 931 tet, 125 034 token, 92 751 DOF; C 8,0M, K ve M 5,7M sıfır olmayan
eleman) ve `elliptical` (5 719 tet, 80 066 token, 63 090 DOF). Makine üç ajanla paylaşıldı (yük ortalaması
5–8, 4 çekirdek): duvar saati ölçümleri ±%30 oynuyor. Güvenilir olanlar **tek iş parçacığında CPU süresi**
(`time.process_time`; tekrarlar arası ±%1; eski ve yeni ağaç ardışık koşuldu) ve aynı süreçte sıralı karşılaştırma.
Eşitlik: eğitim adımının kaybı, girdi tensörleri ve 76 parametre gradyanının tamamı eski koddan (commit
öncesi ağacın `archive` kopyası) **bit düzeyinde aynı**, iki örnekte de.

**Neyin tekrarlandığı** (`field_operators`, pillbox, 2 iş parçacığı, sıcak, en iyi): token özellikleri
(duvar kd-ağacı + torsiyon CG) 1,3–2,0 s, çağrının %35–45'i; `ConvertOperator` 0,55–0,9 s; K ve M montajı
~0,55 s; kalanı (P, Kp, G, tet listeleri) ~0,5 s. Hepsi yalnız mesh'e bağlı, her epoch aynı sonucu veriyor.
Eğitim adımında ise `torch.sparse.mm`'in geri geçişi CSR matrisi **her çağrıda** sıralayarak devriyor
(`aten::sort` + `index_select`): C, M ve K için 0,5–0,7 s'er, adım başına ~2 s (63k DOF).

**Yapılanlar:**

| değişiklik | kazanç | eşitlik |
|---|---|---|
| Token özellikleri (`qfeat_wall`, 5 sütun) H5'e yazılıyor, yükleyici okuyor (`highorder.wall_features`, `field_labels.write_wall_features / add_wall_features`, `scripts/add_field_features.py`) | `FieldDataset.load` pillbox 4,24 → 2,67 s (1 iş parçacığı, CPU); 5,10 → 2,85 s CPU, 3,06 → 2,15 s duvar (2 iş parçacığı); elliptical 3,12 → 1,83 s CPU | `qfeat`, K, M, G, Kp, C, P, qx, qw bit düzeyinde aynı; sütunsuz eski H5 yüklenmeye devam ediyor |
| `hcurl.spmm` geri geçişinde hazır devrik: `Cᵀ` item'da (yalnız eğitim bölmesi), K ve M tam simetrik (`SymmetricMK`), G ve Gt batch'te zaten var | devrik başına 0,54–0,69 s → 0,03 s (63k DOF); item +0,2 s, collate +0,08 s | gradyanlar bit düzeyinde aynı (`test_precomputed_transposes_give_bitwise_equal_gradients`) |
| CG vektör güncellemeleri `out=` ile hazır tamponlara (`hcurl._pcg`) | CG süresi −%15–20 (24 sağ taraf, yüklü makinede 4,43 → 3,54 s) | X ve iterasyon sayısı bit düzeyinde aynı (`test_pcg_buffers_do_not_change_the_solution`) |
| Dikkatte `(k·maske)·w` yerine `k·(w·maske)` (maske ∈ {0,1}): anahtar üzerinde tek geçiş | ≤ 0,2 s / adım (ölçüm gürültüsünde) | bit düzeyinde aynı |
| DataLoader işçilerinde NGSolve 1 iş parçacığı (`loader_worker_init`, `train.py`) | 3 işçi / 4 çekirdek: 2,1–2,5 → 1,6 s / batch | operatörler 1, 2, 4 iş parçacığında bit düzeyinde aynı |

**Sonuç** (tek iş parçacığı, CPU saniyesi):

| | önce | sonra |
|---|---|---|
| pillbox: item + Cᵀ + collate | 4,40 s | 3,0–3,2 s |
| pillbox: eğitim adımı (ileri + geri) | 40,7 s (11,9 + 28,8) | **32,4 s** (10,4 + 22,0), −%20 |
| elliptical: item + collate | 2,97 s | 2,2 s |
| elliptical: eğitim adımı | 20,1 s (5,9 + 14,1) | 19,0 s (5,5 + 13,6), −%5,5 |

Kazanç token ve sıfır olmayan eleman sayısıyla büyüyor (pillbox elliptical'dan %20'ye karşı %5,5); 16k tetlik
geometride daha fazla olması beklenir (ölçülmedi).

**Ölçülüp alınmayanlar:**
- **Daha küçük operatör önbelleği.** Pillbox'ta pickle 263 MB (K 65, M 65, C 92, Kp 17, P 9,5 MB). K ile M'nin deseni
  aynı. Indeksler fark kodlanıp zlib-1 ile sıkışınca 5–40× küçülüyor (K 21,7 → 2,8; C 30,7 → 0,8 MB); K değerleri
  4,4× (43 → 9,9 MB), C değerleri yalnız 1,5× (61 → 41 MB). Kayıpsız toplam ~78 MB, açma ~0,8 s. Yeniden kurma
  özellikler okununca ~2 s. Geometri başına 78 MB'a karşılık ~1,2 s: değmiyor. float32 değerler gradyanı değiştirir
  (K, M, C float64 kullanılıyor). Mevcut `dataset.cache_dir` aynen duruyor.
- **Kp'nin geometri başına Cholesky'si** (NGSolve `sparsecholesky`, aynı Kp, 24 sağ taraf, tek iş parçacığı):
  ayrıştırma 1,5–1,8 s, 24 çözüm 1,3 s; CG 2,6 s (130 iterasyon). Adım başına (ileri + geri) CG 5,2 s, Cholesky
  ~4,2 s: %5. Yalnız CPU, NGSolve nesnesi ana süreçte yaşamalı. §3.6'daki karar geçerli.
- **Geri geçiş CG'sini ileri çözümden başlatmak:** sağ taraf farklı (B̄ = Kp⁻¹Z̄, Z̄ ≠ B), ileri çözüm iyi bir başlangıç değil.
- **CG toleransı:** 1e-8 → 1e-6: 130 → 90 iterasyon (−%30 CG), ama Z'nin göreli hatası 5e-8 → 8e-6; gradyanlar
  değişir. Varsayılan değişmedi.
- **Etiketten gelen sabit terimler** (`MT = M T`, `K T`, `GᵀMT`): adım başına ~50 ms, taşımaya değmez.

**Açık kalan:** gövde (dikkat ve FFN) adımın ~%75'i; `copy_` (adımın ~%10'u) einsum'un yeniden yerleşimlerinden
geliyor, düzeni değiştirmek bit eşitliğini bozar. GPU'da ölçülmedi.

### 3.8 Etiket üretiminin hızı: iki kez yapılan işler (açık iş 11)

Ölçüm: örnek başına aşama tablosu, 2 iş parçacığı, tohum 7, `--threads 2`. Geçici zamanlayıcılar commit
edilmedi (betik: scratchpad `datagen-speed/prof_gen.py`). Makine paylaşımlı (yük 3–5); tablolarda iki koşu var.

**Önce (pillbox_pipes, `--max_ndof 450000`, 664 s, 2 seviye 235k + 430k DOF):**

| aşama | süre |
|---|---|
| CAD + yuvarlama + gmsh hacim mesh'i + topoloji sertifikası + eğri netgen mesh | 0,2 + 0,3 + 1,3 s |
| `solve_modes` seviye 0 (235k DOF): montaj 11,5 / ayrıştırma 19,9 / ARPACK 64,5 | 98 s |
| `mode_qoi` + `indicator` + `size_field` + yeniden mesh (seviye 0) | 1,3 + 11,4 + 0,1 + 1,6 s |
| `solve_modes` seviye 1 (430k DOF): montaj 26,3 / ayrıştırma 106 / ARPACK 141 | 277 s |
| `mode_qoi` + `indicator` (seviye 1) | 1,9 + 17,5 s |
| **`transfer_to_model`** (16,6k tet etiket → 8,9k tet model mesh'i) | **253,5 s** |
| `_mesh_bytes`, `_check_reload`, BREP, H5 yazımı | < 0,3 s toplam |

**Bulgular.**
- `transfer_to_model` model mesh'inin her entegrasyon noktasını (tet başına 216, 1,9M nokta) her form ve her mod için
  etiket mesh'inde yeniden arıyordu. Noktalar bir kez aranıyor, değerler `IntegrationRuleSpace`'e konuyor:
  **253,5 → 29,4 s** (kalan: arama ~6 µs/nokta ≈ 9 s, 6 mod değerlendirmesi ≈ 7 s; arama tek iş parçacıklı).
  Eşitlik: aynı alanlarda eski ve yeni dönüşüm 227,8 s ve 25,6 s; U farkı 3,5e-14 (göreli), rq 2,8e-15, proj_err 9e-12.
  Birim test eski formülle karşılaştırıyor. Tepe bellek +0,85 GB (8k tet model mesh'i, 1,7M nokta).
- `Kp = GᵀMG` scipy üçlü çarpımı ve `tolist()` ile NGSolve'a çevriliyordu: 4,4–6 s her çözümde. H1(p+1) üzerinde
  `∇φ·∇ψ` formu aynı matrisi veriyor (serbest blokta göreli 9e-16) ve 0,34 s: montaj 8,5 → 4,9 s (158k), 20,5/11,2 → 6,0/7,6 s (235k).
  `_diag_mean`: `np.add.at` kalktı.
- Reddedilen eliptik çizim (`max_elements`): cs 1,5 ağı 36 353 tet (> 30 000) idi, yine de 3 bozulma onarım turu (9 s)
  ve cs 2 ile 10,6 s'lik ikinci bir mesh (53 502 tet) üretiliyordu. Artık üretimden hemen sonra reddediliyor ve daha ince cs
  denenmiyor (daha ince cs daha çok tet verir): reddedilen çizim 29 → 12 s.
- `project_to_model` (aynı mesh): etiket modları `solve_modes`'tan aynı kuralla birim normlu geliyor, norm integralleri
  atlandı: 11,1–11,3 → 6,9–7,4 s (235k), 5,0 → 3,3 s (158k).

**Sonra:**

| örnek | önce | sonra |
|---|---|---|
| pillbox_pipes `--max_ndof 450000` | 664 s | 433 s (transfer 253→29, montaj 11,5+26,3 → 6,7+15,3) |
| pillbox_pipes `--max_ndof 150000` (tek seviye) | 150 s / 121 s | 112 s / 115 s (montaj 20,5/11,2 → 6,0/7,6; proje 11,2 → 7,1) |
| elliptical `--max_ndof 150000` (ilk çizim reddediliyor) | 104 s / 112 s | 75,5 s (reddedilen çizim 29 → 12; montaj 11,3 → 4,5; proje 5,5 → 3,3) |

Ayrıştırma ve ARPACK aşamaları değişmedi; koşular arasındaki oynama (örn. ARPACK 59–74 s) makine yükünden.

**Eşitlik.** Üç örnekte de `vol` ve `brep` baytları aynı. pillbox_pipes 150k: frekans 1,6e-15, Q0/G 7e-14, R/Q ve tepe değerleri
≤ 7e-10, `rq_model` 3e-14, `u_model` küme alt uzayı sin θ ≤ 1,3e-7 (float32). Aynı kodun iki koşusu arasındaki fark da aynı büyüklükte
(7e-10, 1,4e-7). 450k: frekans 2e-15, Q0 8e-13, R/Q 6,7e-9, `u_model` alt uzayı 8e-6 (ARPACK tolerance 1e-10 özvektörü ~1e-5'e kadar
belirler; dönüşüm tek başına 3,5e-14). `validate_field_labels.py --max_ndof 300000`: PASS.

**Denenip alınmayanlar (ölçümle).**
- ARPACK: `tol` 1e-10 → 1e-8: 78 → 74 iterasyon, → 1e-6: 60. `ncv` 40: 89. `n_extra` 4 → 1: 72. σ = −0,03·λ₁: 61 iterasyon (−%22) ama λ₁ ilk seviyede bilinmiyor; ~%10 kazanç, alınmadı.
  `M·x` çarpımı çok iş parçacıklı değil (30 ms, TaskManager fark etmiyor). Bir ARPACK adımı 0,42 s: Ainv 0,19 + projeksiyon 0,10 + 3 M çarpımı 0,09.
- İndikatör (9–18 s/seviye): duvar terimi `ds("wall")` ile 0,65 → 0,03 s olur ama kural farklı (toplam 3,5e-5 sapma) → boyut alanı, dolayısıyla mesh değişirdi. Tek integralde birleştirme ve `Compile()` kazandırmadı. Alınmadı.
- gmsh hacim mesh'i + topoloji sertifikası: 0,1–0,9 s/çizim (< %1). Deformasyon haritası düğümlerin sınır kutusundan çiziliyor, o yüzden kalktığında deforme örnekler değişir. Bırakıldı.
- `_check_reload`: 0,1–0,3 s; yüklenen mesh artık token özellikleri için kullanılıyor (§3.7). Bırakıldı. `_mesh_bytes`: 0,03 s.
- Sıcak başlatmalı LOBPCG (§3.4): mesh'ler arası aktarım artık ucuz (yukarıdaki yöntemle), ama yeni mesh'te L2 izdüşümü 216 nokta × 16,6k tet yeniden arama + 11 mod
  değerlendirmesi ≈ 60 s ve BDDC kurulumu 26 s; kazanç en iyi durumda son seviyenin ARPACK'ının yarısı (~%10 örnek başına), özvektör kümeleri ve gradyan çekirdeği için ayrı iş. Yapılmadı.
- MKL Pardiso (`pip install mkl`, NGSolve kendi sarmalayıcısı ya da pypardiso, SPD, METIS): ayrıştırma NGSolve'un sparsecholesky'sinden 2–3,4× hızlı
  (158k: 3,2 → 1,5 s; 235k: 16,1 → 6,0 s; 312k: 26,6 → 8,75 s; 326k: 35,9 → 10,5 s, 2 iş parçacığı) ama çözüm 0,11–0,49 s yerine 0,19–0,75 s (pypardiso her çağrıda A'yı hashliyor).
  ARPACK'ta 78 çözüm var: ≤ 330k DOF'ta net kazanç ≈ 0. 900k DOF'ta (sparsecholesky ayrıştırması bu boyutta dakikalar) yerel ctypes sarmalayıcısıyla kazanç olabilir;
  bu sanal makinede (15 GB) 900k çalıştırılamadığı için ölçülemedi. TRUBA'da denenmeli (açık iş). NGSolve'un kendi `pardiso` sarmalayıcısı sekiz iş parçacığını sabitliyor ve MDO sırası kullanıyor: 158k'da 6,2 s / 0,67 s (daha kötü); `pardisospd` hata −1 verdi.

**Açık.** Çözücü artık örneğin %90'ı. Sıradakiler: (1) TRUBA'da MAX_NDOF 900k ile ayrıştırma/ARPACK dağılımını ölçmek ve yerel MKL Pardiso denemesi;
(2) transfer içindeki nokta aramasını çok iş parçacıklı yapmak (~9 s); (3) deforme mesh'lerde `mesh(x,y,z)` noktaların ~%11'ini bulamıyor (SetDeformation
ile; 1 mm kaydırmada), eski ve yeni dönüşümde aynı davranış: deformasyonlu (`--deform_prob > 0`) alan etiketlerinde `proj_err` ölçülmeli.

### 3.9 Çoklu doğruluk (açık iş 13, ayrıntı docs/34)

- `pillbox_pipes` (pervaz tabanı 0,12), aynı 4 geometri: LF (p2, tek çözüm) 31 s, HF (p3, `max_ndof 150000`) 157 s
  → örnek başına ~5 kat. Frekans farkı LF/HF − 1: +0,004 – +0,04 %; R/Q ≤ 0,04 %; Q0 −0,8 – −1,4 %;
  E_pk −4 – −28 %, B_pk +5 – +8 %. Mesh boyutu `curvaturesafety` / `maxh_factor` ile küçülmüyor.
- LF ile eğitilen (14 geometri, 30 epoch) küçük model HF geometrilerde: frekans hatası %0,6–3,0 (hep üst
  sınır), alan rel-L2 0,03–0,07; hata p2 tabanından (≤ 0,033 %) değil modelden. Aynı ağırlıklarla p3 ve 8 kat ince
  mesh'te çıkarım çalışıyor (Ritz üst sınırı korunuyor) ama doğruluk 0,2–0,6 puan kötüleşiyor.
- 2 HF örnekle ince ayar, aynı epoch sayısında yalnız LF kontrolüne göre kazanç vermedi.
- Birleşik kodla `train.py --fast_dev_run` (`data_path` = 19 LF geometri, `train_extra` = 4 HF, `hf_repeat 2`,
  `fidelity_weights [1,2]`) train/val/test adımlarını çalıştırıyor. 15 GB'lık makinede `num_workers 8` yükleyici
  işçilerini bellek yetmezliğinden öldürüyor; 1 işçiyle sorunsuz. TRUBA'da işçi sayısını düğüm belleğine göre seçin.

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
| 2 | Veri hattı: token özelliklerini veri üretilirken H5'e yazmak (örnek başına −2,8 s); operatörleri yerel diske önbelleklemek ya da yüklenen örnekleri birkaç adım yeniden kullanmak. | ajan "train-data-speed" | worktree → `claude/field-model` | özellikler yapıldı, önbellek değmedi: iş 12, §3.7 |
| 3 | CG'yi eğitim grafiğinden çıkarmak: Kp'yi örnek başına bir kez Cholesky ile ayrıştırıp üçgensel çözüm. Alternatifler: p-multigrid ön koşullayıcı, tol 1e-6 denemesi. | — | — | açık |
| 4 | TRUBA referans koşuları (`TEST=1`, `MAX_NDOF` 900k) ve örnek başına belleğin ölçümü (§3.4). | — | — | açık |
| 5 | Alan modelini web arayüzüne bağlamak; README'yi alan modeline göre güncellemek. | — | — | açık |
| 6 | Hibrit mod prototipi (model alt uzayı → LOBPCG). | — | — | açık |
| 7 | Görev E: gövdede token alt örnekleme. | — | — | açık |
| 8 | Google Drive bağlayıcısı oturumda "Forbidden" döndü; kullanıcının araç izinlerini açması gerekiyor. | sahip | — | kullanıcıda |
| 9 | Eğitim ve çıkarım maliyeti (§3.6). Sıra: Faz 0 ölçüm betiği, 1a operatörler, 1b collate, 2a aktivasyon kontrol noktası; sonra 1c önbellek, 3a Kp Cholesky, 2b bf16; en son 2c token azaltma, 1d özellikleri H5'e yazma (üretici: sahibin onayı gerekli). | Claude Code oturumu | `claude/field-model` | sürüyor: 0, 1a, 1b, 2a bitti; 1c ölçüldü (kod gerekmiyor); 3a GPU ölçümüne kadar ertelendi; sırada GPU ölçümü (Colab) ve 2b |
| 10 | Geometri ve etiket düzeltmeleri (docs/33 §12). Kalan: yeni aileler (qwr, rfd, dtl_cell, gun, büyük açıklıklı eliptik), spoke uç duvarı ve dtl yarım tüpleri, composite'in dönüşümü, doğrulama seti (§8). TRUBA `TEST=1` koşuları artık `_v2` etiketleriyle. | Claude Code oturumu | `claude/field-model` | düzeltmeler bitti; yeni aileler sahibin sıra kararını bekliyor |

| 11 | Etiket üretiminin hızı: iki kez yapılan pahalı işler (uyarlamalı seviyeler, mesh'ler, yeniden yüklemeler); etiketler aynı kalmalı. | ajan "datagen-speed" | worktree → `claude/field-model` | bitti (§3.8): transfer 253→29 s, Kp formu, erken ret; kalan: çözücü (TRUBA ölçümü, Pardiso) |
| 12 | Eğitim veri yolunun hızı: her epoch'ta yeniden kurulan operatörler ve token özellikleri (açık iş 2'nin devamı). | ajan "train-data-speed" | worktree → `claude/field-model` | bitti (§3.7): özellikler H5'te, hazır devrikler, CG tamponları, işçi iş parçacığı; operatör önbelleği ve Kp Cholesky ölçüldü, alınmadı |
| 13 | Çoklu doğruluk: düşük doğruluklu etiketle (p2, tek çözüm) eğitim, yüksek doğruluklu (p3 uyarlamalı) veride çıkarım. | ajan "multi-fidelity" | worktree → `claude/field-model` | kod + ölçüm bitti (docs/34); hibrit mod denenmedi; ölçekli LF+HF eğitim açık |

1, 2 ve 5 birbirine dokunmuyor; paralel yürüyebilir. 3, 2 ile aynı veri hattına dokunuyor; sıraya konmalı.
10'un etiket kısmı 4'ten (TRUBA referans koşuları) önce yapılırsa referans koşuları yeni tanımlarla ölçülür.

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
