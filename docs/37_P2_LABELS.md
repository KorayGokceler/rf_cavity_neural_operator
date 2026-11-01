# 37 — p2 etiketleri: varsayılan eğitim etiketi, model mesh kuralı, p4 referans aracı

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
| model mesh'i | ≤ 10k tet | 5k–20k tet, 10k civarı (§3) |
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

## 3. Model mesh kuralı: `field_labels.model_sized_mesh`

Üretici **ve** tahmin (`src/service/geometry.curved_from_cad`, `scripts/predict_geometry.py`) aynı kuralı kullanır; eğitim ve tahmin mesh'leri aynı kurala uyar.

- Eğri netgen mesh, **5k–20k tet, 10k civarı** (`model_min_elements` 5000, `model_target_elements` 10000, `model_max_elements` 20000).
- `maxh = f·h`, `f` aranır (tet sayısı ∝ f⁻³ ama netgen sıçrıyor: pillbox'ta iki maxh arasında %3 farkla 6,1k → 10,1k tet).
- Sayı küçük detayların eğriliğiyle sınırlıysa (kaba maxh sayıyı düşürmüyorsa) `curvaturesafety` 2 → 1,5 → 1.
- `curvaturesafety` ikiye katlama denemeleri yok; `single_level` ile yeniden kurulur.
- Aralıkta mesh bulunmazsa `RuntimeError`; sonrası aşağıdaki yedekleme kuralıyla (yuvarlatılmışsa keskin kenarla etiketleme, keskinse yeniden çekme).
- p3 yolunun model mesh tavanı da artık 20000 (`model_max_elements` varsayılanı).

Aile başına ölçüm (sobol, `mesh_size` 0,10, aile başına bir çizim; tet, curvaturesafety, mesh'leme süresi):

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

`ridged_box`, `elliptical_long` ve `dtl`'de **ilk çizim reddedildi** (cs 2'de sayı > 20k; cs 1,5 / 1'de katlanmış elemanlar, det-J oranı −25 … −6700) ve yeniden çekildi. `model_sized_mesh` başarısız olunca üretici (kod yolları değişmedi; eski 30k başlangıç tavanına göre 20k tavanında daha sık çalışması bekleniyor, ölçülmedi):
- **yuvarlatılmış çizim** (kenar yuvarlatma uygulanmış): aynı parametrelerle **keskin kenarlarla** yeniden etiketlenir (`round_fallback = 1`, `n_sharp_edges > 0`, tepe QoI NaN); yeniden çekilmez;
- **keskin çizim** başarısız olursa en çok `max_geom_tries` = 6 kez yeniden çekilir (aile sabit; sonraki denemeler Sobol noktasının kalanını ve sonra `default_rng([seed, s_id])` kullanır: bu örneklerde düşük sapmalı kapsama kaybolur); 6'dan sonra örnek eksik kalır.

Büyük ve ince detaylı şekillerin veride az temsil edilme riski var; oranı bilinmiyor (§8, iş 24).

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
`maxh_factor_used`, `curvaturesafety_used`. Örnekler üreticide **atılmaz**. Eğitimde süzme için `dataset.require_label_ok` (`configs/field_3d.yaml`, `FieldDataset(require_label_ok=False)`) var: true olursa eğitim bölmesinde `label_ok == 0` örnekler düşer, aile başına sayı yazılır. **Varsayılan false** (süzme kapalı). Dataset sayfası `GET /api/datasets/{id}/quality`
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
- Üretim ve TRUBA sayfalarında etiket türü seçimi: p2 / p3 uyarlamalı / p3 tek çözüm (varsayılan p2). `config.sh`: `LABEL_ORDER=2`, `MODEL_MAX_ELEMENTS=20000`.
- İşçi belleği p2 için model uzayından: `jobs.label_ndof` = 11 DOF/tet × tavan. Dataset kavite değerleri kaynağı p2 datasetlerde `p2_labels`.

## 9. Açık işler

| iş | durum |
|---|---|
| Reddedilen ilk çizimler (3 aile) ve oranı: TRUBA `TEST=1` ile ölç; seçenekler: bu aileler için tavanı gevşet, daha güçlü onarım. Tavan 20k kalıyor. Sahibin kararı. | açık (docs/31 §6 iş 24) |
| `FIELD_P2_CORE_S` = 150 çekirdek-saniye (ölçüm: 50–408, ortalama ~155, tepe bellek ≤ 1,8 GB; docs/31 §3.13 e). TRUBA `THREADS` p2 için ayarlanmadı. | kısmen (iş 27) |
| p3 uyarlamalı düzeltme: zorunlu büyüme, `converged` koşulu, mutlak R/Q tabanı. | açık (iş 14) |
| Kontrollü A/B mesh deneyi (kural mesh'i ile uyarlamalı p2). | açık (iş 26) |
| Reddedilen ilk çizimler için ayrıca: `round_fallback` payı ve yeniden çekme sayısı aile başına TRUBA `TEST=1` ile ölçülecek (§3). | açık (iş 24) |
| p2 etiketlerin QoI kalitesi düşük kaliteli mesh'lerde (6/16 çizimde `qoi_ok` = 0): yalnız det-J ile ölçüldü, QoI hatasıyla değil. | açık (iş 29) |
| `require_label_ok` açık eğitim ile kapalının karşılaştırması yapılmadı. | açık |
| Ölçekli eğitim p2 etiketiyle yapılmadı. | açık (iş 1) |
