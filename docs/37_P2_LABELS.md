# 37 — p2 etiketleri: varsayılan eğitim etiketi, model mesh kuralı, p4 referans aracı

> 2026-11-02: p2 mesh kuralı değişti (§3: tet tavanı yok, hata sınırı `tol_f_p2`); 5k–20k sürümü tarihçe (§3.1).
> Durum: kod hazır, **commit edilmemiş** (`db1ba24` üzerine; dal `claude/gallant-goodall-bnd4lo`). Karar: sahip, 2026-11-01.
> Ölçekli eğitim ve TRUBA koşusu yapılmadı. Bağlam: docs/29 (p3 etiketleri), docs/30 (uyarlamalı döngü),
> docs/34 §8 (çoklu doğruluk), docs/31 §3.13 ve §6.

## 1. Karar

Eğitim etiketi artık **p2**: model mesh'inin kendisinde tek bir HCurl p2 çözümü.

| | önceki varsayılan | yeni varsayılan |
|---|---|---|
| etiket çözümü | uyarlamalı p3, ayrı etiket mesh'i | p2, model mesh'i (tek çözüm) |
| alan hedefi | p3 alanının model mesh'indeki p2'ye L2 izdüşümü | p2 çözümün kendisi |
| frekans, QoI | p3 etiket mesh'inden | p2 alanlarından (tepeler p3'e göre daha kaba) |
| model mesh'i | ≤ 10k tet | başlangıç ~10k (≥ 5000), `est_rel_f ≤ tol_f_p2` olana dek tekdüze inceltme, tet tavanı yok (§3; 2026-11-02) |
| etiket türü / TAG | `F_p3p2_ms0.10_k10_mf0.05_v2` | `F_p2p2_ms0.10_k10_mf0.05_v2` (`_1s` eki hiç yok) |

p3 yolu duruyor: `--label_order 3` (uyarlamalı; `--no_adapt` tek çözüm). Test, doğrulama ve CST kümeleri için (§7).
`label_order == model_order == 2`; `model_order > label_order` reddedilir, eşit olabilir.

## 2. Gerekçe

1. **Model tabanı.** Modelin Ritz katmanı kendi mesh'inde HCurl p2'de çalışır; min-max gereği bu uzayın özdeğerlerinin altına
   inemez. Frekans tabanını tahmin edilen mesh belirler, etiket doğruluğu değil. p2 etiketi tam bu tabandır (docs/31 §5, "yapısal üst sınır").
2. **Ölçülen eşdeğerlik (docs/34 §8, eşit etiket maliyeti, tek aile `pillbox_pipes`, tek koşu).**

   | etiket | \|f\| hatası | alan rel-L2 |
   |---|---|---|
   | HF (p3), 8 geometri | %1,66 | 0,050 |
   | LF (p2), 24 geometri | %1,55 | 0,046 |
   | LF (p2), 40 geometri | %1,37 | 0,045 |

   Aynı mesh'te LF–HF alan farkı %0,08–0,32 (M-normu); HF/LF maliyeti 4,3×. Geometri sayısı, etiket doğruluğundan daha çok işe yarıyor.
   Kapsam: bir aile, tek koşu, CPU, küçük model (docs/34 başlığı).
3. **Model hatası (~%1) taban hatasının 100 katı** (§4); etiketi p3'e çıkarmak bu aşamada darboğazı değiştirmez.
4. **Uyarlamalı p3 döngüsünün kaba başlangıç mesh'inde durması** (docs/31 §3.13, iş 14): p3 yolu varsayılan olmaktan çıktı, düzeltmesi ayrı iş.

Olumsuz taban çizgileri değişmedi: GP ve kaba FE, ≤ 10k tet ve ≥ 24 geometride modelden iyi (docs/35 §3). p2 etiket bunu değiştirmez.

## 3. Model mesh kuralı (2026-11-02): başlangıç mesh'i + hata sınırına kadar inceltme

> Sahip kararı 2026-11-02. **Önceki sürüm (5k–20k tet tavanı, 2026-11-01) yerine geçildi:** tavan curvaturesafety'yi 1,5 / 1'e indirip katlanmış elemanlara, yuvarlatmasız (keskin) geri dönüşlere ve yeniden çekmelere yol açıyordu. Önceki sürümün sayıları §3.1'de tarihçe olarak duruyor.

**Kural (p2 etiketleri):**
1. Başlangıç mesh'i: `field_labels.model_sized_mesh` — ≥ 5000 tet, 10000'e en yakın, `curvaturesafety` 2; katlanma olursa 3, 4'e **çıkarılır, hiç düşürülmez**; tet tavanı yok.
2. p2 çözümü ve hata tahmini `est_rel_f`.
3. `est_rel_f > tol_f_p2` ise tekdüze inceltme, `r = (tol/est)^(1/q)·0,9`; 1. turda q = 4, 2. turdan itibaren **gözlenen** hız q = log(est_önceki/est) / log((ne/ne_önceki)^(1/3)), [1, 4] aralığına kırpılır (ölçülen: hwr ~3,5, keskin elliptical ~2,35; yuvarlatma / kenarlar h⁴'ten yavaş yakınsıyor): `maxh_factor` → `f·r` ve `curvaturesafety` → `cs/r` birlikte (yalnız maxh eğrilik-sınırlı şekilleri inceltmiyor; hwr yoklaması: cs 2'de maxh 3 → 2: 6674 → 7660 tet; cs 2 → 3: 16628 tet).
4. Yeniden kurulan mesh %15'ten az büyürse r², r³ ile yeniden denenir (cs ≤ 6, DOF ≤ `max_ndof`).
   - Bütçe: adım, `max_ndof`'a sığan en büyük adıma kırpılır (`r_fit = (ne·11/max_ndof)^(1/3)·1,02`); `budget` yalnız `r_fit` > 0,95 ise.
   - `curvaturesafety` `p2_max_cs` = 6'da kırpılır, maxh küçülmeye devam eder.
   - Onarımdan sonra sınırı aşan mesh katlanma değil `budget` sayılır.
5. En çok 3 inceltme turu. Durma nedeni `p2_stop`: `converged` | `budget` (`max_ndof` 900000) | `stalled` | `max_rounds` | `mesh_failed`.
6. Saklanan öznitelikler: `p2_rounds`, `p2_stop`, `p2_tets`, `p2_est`, `p2_cs`, `tol_f_p2` (`est_max` takma adı). CLI: `--tol_f_p2` (varsayılan 1e-3); `label_ok = est_rel_f ≤ tol_f_p2`.
7. %15 eşiği ve r-kuvveti yeniden denemeleri sezgiseldir.

**Neden 1e-3 (dürüst gerekçe).**
- Modelin hatası **bilinmiyor**. Eldeki tek sayı (%1,37–1,66, docs/34 §8) eğitilmiş bir model değil, küçük bir LF / HF karşılaştırma deneyi (tek aile, 8–40 geometri, 600 adım). Ölçekli eğitim yok; tek örnek ezberlemede frekans hatası 1e-6 – 9e-6 (docs/28).
- Dayanak: docs/33 §10 tarama hedefi ≤ %2; 1e-3 bunun 20× altı. Tahmin edici gerçek hatayı 1–3× fazla tahmin ediyor (§4), yani gerçek ≈ 3e-4 – 1e-3.
- Pilot eğitimden sonra (iş 1) yeniden bakılacak: model ~%0,1'e inerse sınır 2–3e-4'e sıkılaştırılır.

**Ölçümler** (4 iş parçacığı, `mesh_size` 0,10, 10 mod):

| örnek | tet | tahmin | tur / süre / bellek |
|---|---|---|---|
| pillbox | 10658 | 4,3e-5 | 1 çözüm, 44 s, 1,9 GB |
| elliptical, rastgele id 0 | 17318 | 5,9e-4 | 68 s, 1,8 GB |
| hwr 3670016 (sobol) | 9411 → 11664 (önceki sürüm: 11588) | 1,08e-3 → 8,4e-4 (önceki: 8,46e-4) | 62 s (önceki: cs 2 → 2,57, 66 s, 1,1 GB) |
| elliptical, rastgele id 1 | 23239 → 34954 | 1,19e-3 → 8,65e-4 | yuvarlatılmış geometri cs 2, 3, 4'te katlandı (det-J −52,6) → keskin kenarla etiketlendi (`round_fallback` 1, tepeler NaN); 428 s, 4,1 GB (366k DOF) |
| blob id 0 (gözlenen hızlı döngü) | 14294 → 30282 → 42408 | 1,01e-2 → 1,28e-3 → 6,6e-4 | cs 2 → 3,51 → 4,14; 3 çözüm, 428 s, 4,1 GB, 447576 DOF; `label_ok` 1 |

- Blob artık `budget` ile durmuyor (üstteki satır; önceki sürümde 14294 tet, 1,01e-2, `budget`, `label_ok` 0 idi), ama pahalı; sahibin tercihiyle eğitimde şimdilik kullanılmıyor (zaten `families.tsv`'de yok). `label_ok` 0 kalan örnekler işaretli, `require_label_ok` onları düşürür.
- **Tolerans ortalamadır:** `est_rel_f` K moddaki bağıl frekans hatası tahminlerinin ortalaması (0,5·Σ ev / K). Bir mod `tol_f_p2`'nin üstünde kalırken `label_ok` = 1 olabilir; arayüz etiketi "modların ortalaması" der.
- Maliyet: `FIELD_P2_CORE_S` → 200 çekirdek-saniye (arayüz ajanı ayarlıyor); 100k geometri için tahmin ≈ 4,3–8 bin çekirdek-saat (pay dahil). Diskte örnek başına 5,6–12 MB (ortalama ~8,5 MB) → 100k için 0,6–1,2 TB; scratch 30 günde siliniyor; bu ölçekte `CACHE_DIR` açılmamalı (geometri başına ~330 MB).

**Tahmin (sahip kararı: seçenek a).** Tahmin başlangıç mesh'inde yapılır (~10k, inceltme yok). Bedeli: eğitimde inceltilen şekillerde (hwr, bazı elliptical) tahmin mesh'i eğitim mesh'inden kaba; oradaki taban 1e-3'ü aşabilir. Seçenek (b) — modelin kendi tahmin ettiği alanla aynı göstergeyi sürmek — açık iş (31 §6 iş 30).

### 3.1 Önceki kural (5k–20k, 2026-11-01; yerine geçildi, tarihçe)

Üretici **ve** tahmin (`src/service/geometry.curved_from_cad`, `scripts/predict_geometry.py`) aynı kuralı kullanır; eğitim ve tahmin mesh'leri aynı kurala uyar.

- (Eski) eğri netgen mesh, **5k–20k tet, 10k civarı** (`model_min_elements` 5000, `model_target_elements` 10000, `model_max_elements` 20000).
- `maxh = f·h`, `f` aranır (tet sayısı ∝ f⁻³ ama netgen sıçrıyor: pillbox'ta iki maxh arasında %3 farkla 6,1k → 10,1k tet).
- (Eski) Sayı küçük detayların eğriliğiyle sınırlıysa `curvaturesafety` 2 → 1,5 → 1 (yeni kuralda cs hiç düşürülmez).
- `curvaturesafety` ikiye katlama denemeleri yok; `single_level` ile yeniden kurulur.
- Aralıkta mesh bulunmazsa `RuntimeError`; sonrası aşağıdaki yedekleme kuralıyla (yuvarlatılmışsa keskin kenarla etiketleme, keskinse yeniden çekme).
- p3 yolunun model mesh tavanı 20000 (`model_max_elements` varsayılanı); p2 yolunda tavan yok.

Aile başına ölçüm, **eski 5k–20k kuralıyla** (sobol, `mesh_size` 0,10, aile başına bir çizim; tet, curvaturesafety, mesh'leme süresi):

| aile | tet | cs | süre | min det-J oranı | qoi_ok |
|---|---|---|---|---|---|
| pillbox | 10434 | 2 | 0,8 s | 0,91 | 1 |
| axisym_cell | 10710 | 2 | 1,7 s | 0,72 | 1 |
| blob | 8279 | 2 | 8 s | 0,25 | 0 |
| elliptical | 8601 | 2 | 19 s | 0,50 | 1 |
| pillbox_pipes | 18710 | 2 | 6 s (eğrilik sınırlı) | 0,46 | 1 |
| reentrant | 17217 | 1,5 | 32 s (cs 2'de maxh'den bağımsız 23,7k) | 0,23 | 0 |
| box | 10960 | 2 | 1 s | 1,0 | 1 |
| composite | 9232 | 2 | 11 s | 0,213 | 0 |
| coax_qw | 9083 | 2 | 6 s | 0,372 | 1 |
| pillbox_port | 11742 | 2 | 3 s | 0,661 | 1 |
| hwr | 9411 | 2 | 2 s | 0,388 | 1 |
| spoke | 8431 | 2 | 6 s | 0,388 | 1 |
| junction | 16320 | 1,5 | 16 s | 0,214 | 0 |
| ridged_box | 17830 | 1,5 | 73 s | 0,215 | 0 |
| elliptical_long | 18610 | 1,5 | 34 s | 0,323 | 1 |
| dtl | 18807 | 1,5 | 51 s | 0,184 | 0 |

**`qoi_ok`:** model mesh'inin en küçük det-J oranı ≥ `qoi_min_jacobian` (DEFAULTS 0,3) ise 1. `netgen_mesh` docstring'i: oran 0,05'te Q0 10× kayıyor, 0,07'de tepe değerleri 3×; ≥ 0,4 sorunsuz. `label_ok` ve QoI değerleri değişmedi; QoI eğitimde kullanılmıyor.
Tablodaki 16 çizimden **6'sında `qoi_ok` = 0** (blob, reentrant, ridged_box, composite, junction, dtl). QoI kalitesi yalnız det-J ile ölçüldü, QoI hatasıyla değil (§9).

`ridged_box`, `elliptical_long` ve `dtl`'de **ilk çizim reddedildi** (cs 2'de sayı > 20k; cs 1,5 / 1'de katlanmış elemanlar, det-J oranı −25 … −6700) ve yeniden çekildi. `model_sized_mesh` başarısız olunca üretici (kod yolları değişmedi; yeni kuralda yalnız başlangıç mesh'i cs 2, 3, 4'te de katlanırsa çalışır; inceltmedeki `mesh_failed` bunu tetiklemez, son çözülen mesh tutulur ve `label_ok = 0` olur; eski 30k başlangıç tavanına göre 20k tavanında daha sık çalışması bekleniyor, ölçülmedi):
- **yuvarlatılmış çizim** (kenar yuvarlatma uygulanmış): aynı parametrelerle **keskin kenarlarla** yeniden etiketlenir (`round_fallback = 1`, `n_sharp_edges > 0`, tepe QoI NaN); yeniden çekilmez;
- **keskin çizim** başarısız olursa en çok `max_geom_tries` = 6 kez yeniden çekilir (aile sabit; sonraki denemeler Sobol noktasının kalanını ve sonra `default_rng([seed, s_id])` kullanır: bu örneklerde düşük sapmalı kapsama kaybolur); 6'dan sonra örnek eksik kalır.

Büyük ve ince detaylı şekillerin veride az temsil edilme riski var; oranı bilinmiyor (§9, iş 24; yeni kuralda neden artık tavan değil, yuvarlatılmış elliptical'ın katlanması).

## 4. p2 doğruluğu mesh boyutuna göre

p4 referansı aynı mesh'te; yöntem doğrulandı: p4, analitik kutu spektrumuna 3e-8; p3–p4 pillbox'ta ~6e-7. 10 mod üzerinden en büyük / medyan bağıl frekans hatası; parantezde hata tahmini.

| aile | tet | en büyük | medyan | tahmin |
|---|---|---|---|---|
| pillbox | 1590 | 7,9e-4 | 4,6e-4 | 1,2e-3 |
| pillbox | 4591 | 1,4e-4 | 6,7e-5 | 1,3e-4 |
| pillbox | 6060 | 1,1e-4 | 5,8e-5 | 1,2e-4 |
| box | 1839 | 7,0e-4 | 4,4e-4 | — |
| box | 4772 | 1,2e-4 | 8,0e-5 | — |
| box | 6693 | 1,0e-4 | 6,6e-5 | 1,1e-4 |
| axisym_cell | 1250 | 1,2e-3 | 5,9e-4 | — |
| axisym_cell | 4574 | 2,2e-4 | 1,3e-4 | — |
| axisym_cell | 5662 | 2,4e-4 | 1,4e-4 | 2,5e-4 |
| elliptical | 8500 | 7,7e-4 | 3,8e-4 | 1,0e-3 |
| blob | 7448 | 8,3e-3 | 3,5e-3 | 1,2e-2 |

- blob: 0,7 mm'lik küçük kenar yuvarlatmaları; eski hattın blob model tabanı da %0,56'ydı.
- Hata tahmini / gerçek: 0,8–3 (box), ~1,0 (axisym_cell, pillbox).
- Duman testleri: pillbox 8506 tet, tahmin 5,7e-5; box 8025 tet, tahmin 1,0e-4. `tests/test_p2_labels.py`: pillbox 1777 tet TM010 hatası 3,9e-5, tahmin 1,6e-4, proj_err 3,2e-8.
- Model hatası (~%1) bu tabanın 100 katı üstünde (docs/34 §8: |f| hatası %1,4–1,7).

## 5. Hata tahmini ve `label_ok`

Her p2 örneği saklar: `est_rel_f`, `est_rel_wall`, `label_ok` (`est_rel_f ≤ est_max`, `est_max = 1e-3`), `est_max`, `model_meshes`,
`maxh_factor_used`, `curvaturesafety_used`. `label_ok = est_rel_f ≤ tol_f_p2` (`est_max` takma ad); `est_rel_f` modların **ortalamasıdır** (§3), tek bir mod sınırın üstünde olabilir. Örnekler üreticide **atılmaz**. Eğitimde süzme için `dataset.require_label_ok` (`configs/field_3d.yaml`, `FieldDataset(require_label_ok=False)`) var: true olursa eğitim bölmesinde `label_ok == 0` örnekler düşer, aile başına sayı yazılır. **Varsayılan false** (süzme kapalı). Dataset sayfası `GET /api/datasets/{id}/quality`
ile `label_ok` payını, `est` medyan / en büyük değerini ve tet aralığını gösterir.

## 6. Neden uyarlamalı p2 yok (A: kural mesh'i, B: p2 uyarlamalı)

B (boyut alanı hedef tet sayısına ölçeklenir = zorunlu büyüme) aynı tet sayısında yaklaşık 1,5× daha doğruydu:

| aile | tet | en büyük hata |
|---|---|---|
| pillbox | 6757 | 7,4e-5 |
| box | 7870 | 6,4e-5 |
| axisym_cell | 5841 | 1,5e-4 |

Kullanılmıyor, çünkü:
- tahmin zamanında çözüm yok, mesh'i süren bir şey yok: eğitim / tahmin mesh'i uyumsuzluğu olur (docs/34: 8× ince mesh'te çıkarım 0,2–0,7 puan kaybettirdi);
- örnek başına 2–3 çözüm;
- model hatası (~%1) tabanın 100 katı.

Sonra: kontrollü deney (`/experiment`, iş 26) ya da tahmin zamanında modelin kendi alanından süren bir gösterge.

## 7. p3 yolunun yeri

`--label_order 3`: test, doğrulama ve CST kümeleri (docs/29, docs/30). `scripts/cst_calibration_set.py` artık `--label_order 3` geçiriyor (CST kümeleri p3 uyarlamalı kalır). Uyarlamalı döngünün kaba başlangıçta durma sorunu hâlâ açık: docs/31 §6 iş 14, `docs/30` sonundaki "Known issue".

## 8. p4 referans aracı ve arayüz

- `scripts/reference_check.py` (mantık `src/data_gen/reference.py`): H5 dosyası ya da klasör; varsayılan aile başına bir örnek; saklanan model mesh'ini kurar, orada p4 çözer.
  Seçenekler: `--order`, `--max_ndof 700000`, `--fallback_order`, `--cross_check` (|p3−p4| kendi hatası), `--allow_other_mesh`. Çıktı JSON + tablo.
  - Etiket frekansları başka (daha ince) bir etiket mesh'inden gelen örnekler (uyarlamalı p3: `n_tets_label != n_tets` ya da `adapt_iters`) atlanır ("label frequencies from a different (finer) label mesh"); `--allow_other_mesh` ile açılır.
  - `per_family` dosyalar (parçalar) boyunca sayılır; arayüz işi dataset'in dosyalarını verir.
  - `ref_self_err`, order-1 etiket mertebesine eşitse null. "Gerçek hata" iki çözümde ortak olan eğri-3 geometri hatasını içermez (docs/29: pillbox'ta eğri 2 ile 3 arası 1,4e-4).
- Maliyet: 6k tetlik mesh'in p4'ü ≈ 330k DOF, 4 iş parçacığında ~100 s, ~5 GB. Sahip çalıştırır.
- Arayüz: Dataset sayfasında "p4 ile doğrula" (arka plan işi, `POST /api/jobs/reference`, sonuç tablosu, bellek uyarısı; `GET /api/jobs/{id}/reference`, `GET /api/jobs/reference_plan`).
- Üretim ve TRUBA sayfalarında etiket türü seçimi: p2 / p3 uyarlamalı / p3 tek çözüm (varsayılan p2). `config.sh`: `LABEL_ORDER=2`; arayüzde p2 için tolerans alanı `tol_f_p2` (varsayılan 1e-3).
- İşçi belleği p2 için model uzayından: `jobs.label_ndof` p2'de `max_ndof` döndürür (işçi belleği `field_mem_gb(max_ndof)` ≈ 13 GB @ 900k, tutucu; ölçülen ≤ 4,1 GB, 366k–448k DOF). Dataset kavite değerleri kaynağı p2 datasetlerde `p2_labels`.

## 9. Açık işler

| iş | durum |
|---|---|
| Yuvarlatılmış elliptical geometrilerin katlanması (cs 2, 3, 4; keskin kenarla etiketleme): oran bilinmiyor, TRUBA `TEST=1` ile `round_fallback` payı ölç. Artık tavan nedeni değil. Sahibin kararı. | açık (docs/31 §6 iş 24) |
| Blob yakınsıyor ama pahalı (428 s, 4,1 GB); eğitimde kullanılmıyor (sahibin tercihi). | açık |
| **Eğitim belleği:** p2 eğitim mesh'lerinde tavan yok; `MAX_NDOF` 900k'da ~82k tete kadar (`max_ndof` / 11). `BATCH` (2) ve 16 GB V100 ~20k tet için boyutlandırılmıştı; eğitimden önce TRUBA `TEST=1` shard'ının en büyük örneğinde bir batch denenecek. | açık (iş 31) |
| Tahmin seçeneği (b): modelin kendi alanıyla inceltme; bugün başlangıç mesh'inde tahmin (seçenek a). | açık (iş 30) |
| %15 durma eşiği ve r-kuvveti yeniden denemeleri sezgisel; `tol_f_p2` pilot eğitimden sonra gözden geçirilecek. | açık (iş 30) |
| `FIELD_P2_CORE_S`: 150 çekirdek-saniye ölçüldü (50–408, ortalama ~155, 20k kuralıyla; docs/31 §3.13 e); inceltmeli kural için 200 (§3). TRUBA `THREADS` p2 için ayarlanmadı. | kısmen (iş 27) |
| p3 uyarlamalı düzeltme: zorunlu büyüme, `converged` koşulu, mutlak R/Q tabanı. | açık (iş 14) |
| Kontrollü A/B mesh deneyi (kural mesh'i ile uyarlamalı p2). | açık (iş 26) |
| Reddedilen ilk çizimler için ayrıca: `round_fallback` payı ve yeniden çekme sayısı aile başına TRUBA `TEST=1` ile ölçülecek (§3). | açık (iş 24) |
| p2 etiketlerin QoI kalitesi düşük kaliteli mesh'lerde (6/16 çizimde `qoi_ok` = 0): yalnız det-J ile ölçüldü, QoI hatasıyla değil. | açık (iş 29) |
| `require_label_ok` açık eğitim ile kapalının karşılaştırması yapılmadı. | açık |
| Ölçekli eğitim p2 etiketiyle yapılmadı. | açık (iş 1) |
