# 38 — Projeksiyonsuz eğitim ve doğrulama (`training.projection: false`)

> 2026-11-03, dal `claude/train-no-projection` (`50f8f82` üzerine), **commit edilmemiş**. Karar: sahip. Varsayılan değişmedi
> (`projection: true`; çıktılar, kayıp, gradyanlar, doğrulama kaybı bit düzeyinde aynı, §5). Seçenek **eğitim çalışması olarak denenmedi**:
> `val/span_loss` ile seçilen bir modelin test kalitesi (`test/field_rel_l2`, frekans hatası, `test/grad_frac`) projeksiyonlu eğitimle
> karşılaştırılmadı (§7). Bağlam: docs/18 §3.4–3.6 (Schur-projekteli Gram), docs/31 §3.5–3.6, §4 (ceza yöntemi reddi), docs/21 B3.

## 1. Ne yapar

`training.projection: false` ile **eğitim ve doğrulama** (train ve val bölmeleri) çekirdek projeksiyonu ve Rayleigh–Ritz olmadan çalışır.
İkisi **aynı fonksiyonla** hesaplanır (`CavityLightning._compute_loss_3d_plain(batch, prefix)`; aynı terimler, ağırlıklar, `span_norm`,
`span_root`, `span_ridge`, ölü sütun kuralı, gradyan-kör ortho; kopyalanmış formül yok), bu yüzden `train/…` ve `val/…` değerleri
birebir karşılaştırılabilir. **Test** (eğitim sonunda, en iyi checkpoint) projekte kalır: gerçek frekans ve alan kalitesi orada ölçülür.

| | `projection: true` (varsayılan) | `projection: false` |
|---|---|---|
| train / val ileri geçişi | gövde → V → `PV = V − G Kp⁻¹ GᵀMV` (CG) → `projected_eigh` | gövde → V (`forward_basis`), burada biter |
| Kp CG (ileri + geri) | var (örnek başına ~6 s, 23k tet) | **yok** (train ve val) |
| kütle Gram'ı `G_M` | `[[M_div, VᵀMT − Zᵀ(GᵀMT)], [·, TᵀMT]]` | `[[VᵀMV, VᵀMT], [·, TᵀMT]]`, V sütunları birim kütle normuna ölçeklenir, çökmüş sütunlar ölü sayılır (§2.6) |
| enerji Gram'ı `G_A` | `[[VᵀKV, VᵀKT], [·, TᵀKT]]` | aynı (bit düzeyinde; yalnız sütun ölçeği) |
| kayıp | span + 0,01 selfsup + 0,01 ortho + 0,1 freq | `span_weight · span + ortho_weight · ortho` |
| ortho | kütle Gram'ı `M_div` üzerinde | **enerji (curl–curl) Gram'ı** üzerinde, gradyan-kör (§2.5) |
| train / val batch'inde `G, Gt, Kp, Kp_diag` | var | **yok** (veri kümesi kurmaz, collate koymaz) |
| test | projeksiyon + Ritz | **aynı**; ek olarak `test/loss_plain` (§1.2) |
| model seçimi (ModelCheckpoint, EarlyStopping, `reducelr`) | `val/field_rel_l2` | **`val/span_loss`** (§1.1) |
| tahmin | projeksiyon + Ritz | aynı (değişmedi) |

| bölme | `projection: true` | `projection: false` |
|---|---|---|
| train | `build_potential=True` | `build_potential=False` (önbellek eki `.nopot`) |
| val | `build_potential=True` | `build_potential=False` (aynı) |
| test | `build_potential=True` | `build_potential=True` |

`selfsup_weight > 0` ya da `freq_weight > 0` ile `projection: false` birlikte **`ValueError`** verir (`CavityLightning` kurulurken):
selfsup terimi projeksiyonsuz gradyan içeriğini ödüllendirir (§3), frekans terimi Ritz değerleri ister. Örnek: `configs/field_3d_noproj.yaml`.

**Loglananlar (train ve val, aynı anahtarlar, her adımda, her DDP sırasında).** `{train,val}/loss`, `…/span_loss`, `…/ortho_loss`,
`…/span_rel_l2` (kütle normu, en iyi yaklaşım hatası, modların ortalaması), `…/span_rel_l2_energy`, `…/span_mode_k_rel_l2`, `…/mass_diag_ratio`, `…/grad_frac_lb`, `…/grad_frac_lb_max` (gradyan kütle payının kesin **alt sınırı**, §3.5);
yalnız train: `lr`. **Yoktur** (train, val): `field_rel_l2`, `mode_k_rel_l2`, `freq_*`, `r2`, `mae`, `grad_frac`, `kp_cg_iters`, `selfsup_loss`
(Ritz alanı, frekans ya da sınırsız bir terim gerektirirler); bunlar yalnız testte vardır. Test: bugünkü bütün test metrikleri
(`test/field_rel_l2`, `test/mode_k_rel_l2`, `test/freq_*`, `test/grad_frac`, `test/kp_cg_iters`, `test/span_*`, `test/r2`, `test/mae`, `test/loss`) ve `test/loss_plain`.
Epoch sonu konsol tablosu (`on_validation_epoch_end`) bu modda mod başına `span_mode_k_rel_l2` (en iyi yaklaşım hatası) ve `val/span_loss` (seçilen), `val/loss`, `val/span_rel_l2`, `val/mass_diag_ratio`, `val/grad_frac_lb` yazar.

### 1.1 Model seçimi: `val/span_loss`

`val/span_loss` (`monitor_metric(False)`), kayıp fonksiyonunun **yalnız span terimidir** (`mean √(½(r_M + r_A) + 1e-8)`, doğrulama kümesinde, projeksiyonsuz). Kütle normundaki
bileşeni doğrulama kümesinin **projekteli span artığının üst sınırıdır** (§2.3, `r_düz ≥ r_proj`; enerji normunda eşit). `val/loss = span_weight · span + 0,01 · ortho` ise
ortho'yu da içerir; ortho (∈ [0, 1], düzenleyici) sütunların doğrusal bağımlılığını ölçer, bir kalite ölçüsü değildir ve seçimi gürültüyle bozar. Bu yüzden **seçim `val/span_loss` ile yapılır**;
`val/loss` train/loss ile birebir karşılaştırılabilir olsun diye yine loglanır (grafiklerde "Kayıp" eğrisi). (Önceki sürüm `val/loss` ile seçiyordu; o dosya adlarını `resolve_checkpoint` hâlâ okur.)
En iyi checkpoint dosya adı `best-epoch=NN-val_span_loss=0.xxxx.ckpt` (varsayılan: `…-val_rel_l2=0.xxxx.ckpt`);
`src/training/checkpoint.py:resolve_checkpoint` `val_rel_l2=`, `val_span_loss=` ve `val_loss=` etiketlerini okur; birden çok çeşit içeren bir dizinde en son yazılan çeşit içinde en düşüğü
seçer, çünkü skorlar karşılaştırılamaz. `progress.json`'daki `best.monitor` `val/span_loss` olur; web arayüzü bunu gösterir (§4).
**`val/span_loss` Ritz alan hatasının vekilidir, ölçüsü değildir**: kütle normunda span hatası gradyan içeriğiyle şişer ve aynı değer farklı frekans/alan kalitelerine karşılık gelebilir (§3). Seçilen checkpoint'in gerçek kalitesi
yalnız `trainer.test` çıktısındadır.

### 1.2 Test: projekte, ve `test/loss_plain`

`test_step` değişmedi (`_compute_loss_3d(batch, "test")`: projeksiyon + Ritz, bütün test metrikleri, `test/loss` = projekteli kayıp). `projection: false` iken ek olarak
`test/loss_plain` loglanır: **eğitim kaybı** (aynı fonksiyon) test kümesinde. `test/loss` ile `test/loss_plain` arasındaki fark projekteli ile projeksiyonsuz kaybın
boşluğunu gösterir. İki sayı tam olarak aynı büyüklük değildir: `selfsup_weight = freq_weight = 0` olduğundan `test/loss = span + 0,01·ortho(kütle Gram'ı, projekteli)`,
`test/loss_plain = span(projeksiyonsuz) + 0,01·ortho(enerji Gram'ı)`. Span kısmı için `r_düz ≥ r_proj` beklenir (§2.3). `test_step` gövdeyi **bir kez** çalıştırır: `loss_plain` projekteli geçişin bazını (`out['basis']`, `_compute_loss_3d(…, basis_out=)`) yeniden kullanır (test: `field_basis` çağrı sayısı 1).

## 2. Matematik

Gösterim: `K` curl–curl, `M` kütle, `G` çekirdek potansiyellerinin gradyanı (`K G = 0`), `Kp = GᵀMG`, `P = I − G Kp⁻¹ GᵀM` (`M`-ortogonal
izdüşüm, `range(G)` üzerinden dik tümleyene), `T` saklanan ayrık özmodlar (`K t = λ M t`, `λ > 0`).

**2.1 Hedefler gradyanlara `M`-diktir.** `Gᵀ K t = 0` (çünkü `K G = 0` ve `K` simetrik) ve `Gᵀ K t = λ Gᵀ M t` olduğundan `λ > 0` için
`Gᵀ M t = 0`. Yani `P t = t`. Bu yalnız `t`, **modelin kendi uzayının** özvektörüyse ya da onun iç içe bir uzaydan (aynı mesh'te p3 → p2) tam `M`-izdüşümüyse tam geçerlidir (§3.2).

**2.2 Çapraz bloklar değişmez.** `P`, `M`-ortogonal olduğundan `Pᵀ M = M P` ve `(PV)ᵀ M T = Vᵀ M P T = VᵀMT`. Enerji tarafı: `(PV)ᵀK(PV) = VᵀKV`,
`(PV)ᵀKT = VᵀKT` (`K G = 0`, bu her `T` için, bu yüzden `G_A` projeksiyonlu ve projeksiyonsuz bit düzeyinde aynı). Tek fark `VV` kütle
bloğu: `VᵀMV − M_div = Bᵀ Kp⁻¹ B ⪰ 0` (`B = GᵀMV`), yani projeksiyonsuz blok `V`'nin gradyan kütlesi kadar **büyüktür**.

**2.3 `r_düz ≥ r_proj`.** Kütle normunda `r_k = 1 − ‖Π_V t_k‖² / ‖t_k‖²` (`span_residual`), `span(V)` içindeki en iyi yaklaşım hatasının
karesidir. `t ∈ range(P)` ve `c` herhangi bir katsayı vektörü için
`‖t − Vc‖²_M = ‖t − PVc‖²_M + ‖(I − P)Vc‖²_M ≥ ‖t − PVc‖²_M`; `c` üzerinden minimum alınırsa `r_düz ≥ r_proj`. Yorumlar:

- Projeksiyonsuz span terimi, projekteli span teriminin bir **üst sınırıdır** (etiketler `M`-dikse, §3.2). Bu yüzden `val/span_*` ve `val/span_loss`, aynı doğrulama kümesinin
  projekteli span hatasından küçük olamaz; test ve tahmin projeksiyonla çalıştığından orada ölçülen span hatası, eğitimde ve doğrulamada izlenen büyüklükten küçük ya da eşittir.
  Tahmin zamanındaki projeksiyon span hatasını yalnız düşürebilir.
- Fark `V`'nin gradyan içeriğidir: gradyan kütlesi `VV` bloğunu şişirir ve `r_M`'yi artırır. Yani kayıp, gradyan içeriğini **yumuşak** bir
  ceza gibi cezalandırır (docs/18 §3.4: "ceza gibi davranır ama kesin değildir"; E4: %50 gradyan payında `r_M` = 0,174). Sıfırlanmasını
  garanti etmez. Gradyan payının kendisi eğitimde/doğrulamada ölçülmez (`grad_frac` yalnız testte vardır); onun yerine kesin bir **alt sınırı** (`grad_frac_lb`) ve dolaylı göstergeler §3.5'te.
- `span_norm: energy` ile `projection: false` **`ValueError`** verir: hem enerji normundaki span hem enerji-Gram ortho'su gradyan içeriğini görmez (`K G = 0`), yani V'yi gradyandan uzak tutacak hiçbir terim kalmaz (sıfır baskı). `mass` ve `both` (varsayılan) geçerlidir.
- Enerji normu gradyanları hiç görmez (`‖curl e‖`); `span_norm: energy` projeksiyonlu ve projeksiyonsuz aynıdır.

**2.4 Hesap.** `hcurl_grams_plain(V, T, batch)`: yalnız `MV, KV, MT, KT` seyrek çarpımları ve üç Gram. Kp çözümü, `G`, `Kp`, `Kp_diag` gerekmez.
Birim testi `Mᵀ`, `Kᵀ` önceden hazırlanmış geri geçişinin (`SymmetricMK`) torch'un kendi geri geçişiyle bit düzeyinde aynı gradyanı verdiğini ve
Gram'ların scipy ile elle kurulan Gram'larla `1e-10` içinde uyuştuğunu denetler.

**2.5 ortho gradyan-kör olmalı.** Projekteli yolda ortho, `M_div` bloğunun (`PV`'nin kütle Gram'ı) sütun korelasyonlarıdır. Projeksiyonsuz `VᵀMV` üzerinde aynısı
yapılırsa, `V_i = c_i + g_i` (`c_i = PV_i` rot kısmı, `g_i = (I−P)V_i` gradyan kısmı) için `M_mn = (c_m, c_n) + (g_m, g_n)`: bağımsız gradyan kısımları köşegeni büyütür,
çapraz terimi büyütmez, korelasyon **sabit aralıkta (span) düşer**. Yani aynı doğrusal bağımlı sütunlar gradyan eklenerek "dik" görünür: ortho gradyan içeriğini ödüllendirir. Bu yüzden
projeksiyonsuz yolda ortho **enerji Gram'ı `VᵀKV`** üzerinden hesaplanır (`K G = 0`: gradyan kısmını hiç görmez; projekteli bazın enerji Gram'ıyla aynı). Saf gradyan sütunlarının
enerji köşegeni sıfıra yakındır; köşegen, örnek içindeki en büyük köşegenin `1e-12`'si ile alttan sınırlanır (`_basis_conditioning_loss(rel_floor=1e-12)`), böylece sıfır-rot sütunu ne
patlar ne de diğerleriyle doğrusal bağımlı sayılır. Test: gradyan `G·φ` eklenmiş kopya sütunlarda enerji-ortho değişmez (float32 baz, `< 1e-5`) ve doğrusal bağımlılığı yine yakalar
(değer `> 0,1`), kütle-Gram ortho'su ise aynı kurguda `< %30`'a düşer (ödülün kanıtı); saf gradyan sütunu ortho'yu artırmaz.

**2.6 Ölü sütun kuralı ölçekten bağımsız olmalı; çökmüş sütun ölü olmalı.** `span_residual` her sütunun ölçeğine karşı değişmezdir, ama `_jacobi_cholesky`'nin ölü sütun kuralı
(`diag ≤ 1e-12 · en büyük diag` ⇒ sütun 0) değildir. Projekteli yolda gradyan sütunları `M_div`'de de ölüdür ve kural zararsızdır. Projeksiyonsuz `VᵀMV`'de ise gradyan-ağır bir
sütunun normu çok büyükse **temiz sütunlar "ölü" sayılır** (hakem sondajı: 3 temiz sütun + 2 saf gradyan sütunu ×10⁶⁻⁷ ⇒ mod 0 için `r_M = 1`). Çözüm: Gram'lar `hcurl_grams_plain`
çıktısından sonra, V sütunları birim kütle normuna ölçeklenerek (`_normalise_columns`: `diag VᵀMV = 1`; sıfır sütun 0 kalır) hem `G_M` hem `G_A` için kullanılır. Birim kütle normunda ölü sınaması
kütle Gram'ında hiçbir canlı sütunu ölü yapmaz; enerji Gram'ında ölü olan, gerçekten rot içeriği ~0 olan (Rayleigh bölümü `≈ 0`) sütundur: ölçekten değil içerikten kaynaklanır.
Projekteli yol bit düzeyinde aynı: `_normalise_columns` yalnız projeksiyonsuz yolda çağrılır.

*Çökmüş sütun (hakem bulgusu).* Kayıp her sütunun ölçeğine karşı değişmez olduğundan, birim ölçekleme tek başına sıfıra çöken bir sütunu korumaz: kayıbın V üzerindeki gradyanı `∝ 1/‖V_i‖` büyür
(hakem sondajı `probe3.py`: göreli genlik 10⁻⁹'da ~2,4·10⁶ kat, 10⁻²⁰'de ~10¹⁶ kat; `gradient_clip_val = 1` ile tek sütun bütün güncellemeyi küçültür). **İstenen "sınırlı ölçekleme"
`s = 1/√max(d, ε·d_max)` bu gradyanı sınırlamıyor** (denendi: span_residual kendi Jacobi ölçeklemesiyle yeniden normalleştiğinden gradyan yine `1/genlik`). Uygulanan çözüm:
`d_i ≤ COLUMN_FLOOR · d_ref` (`COLUMN_FLOOR = 1e-6`, yani genlik < 10⁻³) olan sütun **ölüdür** (ölçek 0: span ve ortho gradyanı yok; projekteli yoldaki Jacobi kuralının eşdeğeri), canlı bir sütunun gradyanı
referans genlikteki bir sütunun en çok ~10³ katıdır. Referans `d_ref` = **rot taşıyan** sütunların en büyük `VᵀMV` köşegeni (Rayleigh bölümü `RQ_i ≥ 0,1 λ₁`, `λ₁` = saklanan hedeflerin en küçük Rayleigh bölümü; hiçbiri değilse bütün sütunlar):
gradyan-ağır bir sütun (RQ ≈ 0) referansı belirlemez, bu yüzden 10⁷× gradyan sütununun yanındaki temiz sütunlar canlı kalır (eski sondaj testi geçiyor).
Ölçüm (`gtab.py`, depoda değil; pillbox, 5 sütun: 2 hedef + 2 rastgele alan + göreli genliği `amp` olan 1 rastgele sütun, `r_M`/`r_A` + ortho kaybı; sütunun `‖dL/dV_i‖`'si, float64):

| göreli genlik | 1 | 10⁻² | 10⁻³ | 10⁻⁴ | 10⁻⁶ | 10⁻⁹ | 10⁻¹² | 10⁻²⁰ |
|---|---|---|---|---|---|---|---|---|
| eski (tam birim ölçek) | 1,2·10⁻² | 1,2 | 1,2·10¹ | 1,2·10² | 1,2·10⁴ | 1,2·10⁷ | 1,2·10¹⁰ | 1,2·10¹⁸ |
| yeni (`COLUMN_FLOOR = 1e-6`) | 1,2·10⁻² | 1,2 | 0 (eşikte, ölü) | 0 | 0 | 0 | 0 | 0 |

Eşik d_i ≤ 10⁻⁶ d_ref'tir (10⁻³ genlik); testte canlı taraf 5·10⁻³ ile denenir (gradyan ≤ 1,2/amp). (Hakemin 2,4·10⁶ sayısı bundan farklı bir kurgudan geldi; sıra büyüklüğü ve `1/genlik` yasası aynı.)
Bedel: referansın 10⁻³'ünden zayıf genlikli bir sütun kayıpta **yok sayılır** (ölü) ve gradyan almadığından kendiliğinden geri dönemez (yalnız ağırlık sönümü/diğer sütunlarla paylaşılan gövde ağırlıkları onu değiştirir);
projekteli yolda eşik 10⁻⁶ genliktir (daha gevşek). Bu yüzden sütun ölçeğine karşı değişmezlik testi artık 10^±1,2 aralığında (genlik oranı ≤ ~250); 10⁻⁶ … 10⁶ aralığı bilerek kaldırıldı (genlik oranı 10¹² olan sütun ölü).
Eşik `COLUMN_FLOOR` tek yerde; gradyan-ağır sütunun referans dışı bırakılması float32'de `K` köşegeni gürültüsüne dayanır (RQ ≥ 0,1 λ₁ ölçütü; `λ_max/λ₁ · 10⁻⁷ ≳ 0,1`, yani çok ince mesh'lerde, gradyan sütununu "rot taşıyan" gösterebilir: o durumda referans şişer ve temiz sütunlar ölü olabilir; ölçülmedi).
Uç durum (değişmedi): gradyan kütlesi rot kütlesinin ~10⁶ katını aşan bir sütunun enerji köşegeni `1e-12` sınırının altına düşer ve enerji Gram'ında ölü sayılır; bu `r_A`'yı yalnız büyütür (üst sınır korunur).
Sütun ölçeği kayıpta görünmez; ölçek dağılımı `mass_diag_ratio` ile izlenir (§3.5).

## 3. Kaybedilenler ve kesinlik sınırları

1. **Frekans ve selfsup terimleri eğitimde yok.** Frekans terimi Ritz değerleri ister. Selfsup terimi (`−log Σ 1/θ`, Ky Fan) projeksiyonsuz Gram'da
   sınırsızdır (docs/18 §3.4 E4: `tr(G_A⁻¹ G_M) = 6·10¹²`, saf gradyan sütunu `θ₁ ≈ 10⁻¹³`) ve gradyan üretmeyi ödüllendirir. Frekans bilgisi
   eğitime yalnız enerji normundaki span hatası üzerinden girer (Ritz özdeğer hatası bu hatayla orantılıdır). Frekans doğruluğuna etkisi
   **ölçülmedi** (§7).
2. **Etiketler `M`-dik olmalıdır; etiket türüne göre tam, yaklaşık ya da değil.** §2.1 yalnız model uzayının kendi ayrık özvektörlerinde ve onların iç içe uzaydan tam izdüşümünde tam:
   - **p2 etiket, etiket mesh'i = model mesh'i** (`label_order == model_order`): `T`, model uzayının özvektörü. `GᵀMT` çözücü toleransı ve float32 saklama düzeyinde (§6.1: ~2·10⁻⁸ kütle normu gradyan payı).
   - **p3 etiket, aynı mesh'te p2'ye izdüşüm** (`label_order 3`, `n_tets_label == n_tets`): izdüşüm `field_labels.project_to_model`, **tam** `M`-ortogonal (L²) izdüşümdür ve p2 ⊂ p3 iç içe uzaydır.
     Her p2 gradyanı `g` aynı zamanda p3 alanıdır, dolayısıyla `(g, P e₃) = (g, e₃) = 0` (p3 özvektörü p3 gradyanlarına dik). **Tam eşitlik**; geriye yalnız model `M`'sinin integral kuralı (`dx₂`) ile etiketin kuralı (`dx₃`) arasındaki fark ve float32 saklama kalır
     (ölçüm §6.1: çift duyarlıkta `‖GᵀMt‖/‖Mt‖ ≤ 3·10⁻¹⁰`, mesh inceldikçe düşüyor). Uyarı verilmez.
   - **Başka mesh'ten aktarılmış etiketler** (`n_tets_label ≠ n_tets`, `field_labels.transfer_to_model`: ince p3 mesh'inden kaba model mesh'ine L² aktarım) ya da `label_order < model_order`:
     `T`, model uzayının özvektörü değil, ona yakın bir alan; kaba mesh'in p2 gradyanları ince mesh'in uzayında iç içe değildir. `GᵀMT ≠ 0`; §2.2'deki eşitlikler **yaklaşıktır** ve `r_düz`, `r_proj`'tan farklı bir hedefe bakar.
     Küçük bir kalibrasyon kümesinde (46–57 tet'lik model mesh'i, izdüşüm hatası %8–12) gradyan payı ≤ 6·10⁻⁴ çıktı (§6.1): büyüklük olarak küçük, ama garanti değil.
     `FieldDataset(build_potential=False)` bu durumda yükleme sırasında **uyarı** verir (train ve val bölmeleri için birer kez; `labels_are_model_eigenvectors`).
3. **Eğitim/test uyumsuzluğu.** Eğitim `span(V)`'nin tamamını (m sütun), tahmin `span(V)` içinden Ritz ile seçilen K modu kullanır; bu fark
   projeksiyonlu eğitimde de vardı (span kaybı Ritz öncesi). Yeni olan, eğitimde ve doğrulamada Ritz seçiminin ve çekirdek temizliğinin hiç görülmemesidir.
4. **Gradyan sızıntısı riski.** `V` gradyan içeriği öğrenirse tahmin zamanındaki projeksiyon bunu atar (güvenli), ama kapasite boşa gider, `M_div` küçülür ve `drop_tol` ile atılan yön sayısı artabilir.
   Eğitim bunu yalnız §2.3'teki yumuşak ceza (kütle normu) ile engeller.
5. **Eğitim sırasında görünmeyenler ve yerine izlenecekler.** Eğitim ve doğrulama sırasında **görünmez**: Ritz frekans hataları (`freq_*`), Ritz alan hatası (`field_rel_l2`, `mode_k_rel_l2`), `grad_frac`, `kp_cg_iters`, Ritz marjı.
   Yerine izlenecekler (hepsi train ve val'da, TensorBoard / `progress.json` / arayüz eğrilerinde):
   - `train/grad_frac_lb`, `val/grad_frac_lb` (sütunlar üzerinde ortalama) ve `…_max` (en kötü sütun): **gradyan kütle payının kesin alt sınırı**, her adımda, yalnız projeksiyonsuz Gram'lardan
     (`_grad_frac_lower_bound`). `RQ_i = (VᵀKV)_ii/(VᵀMV)_ii` sütunun Rayleigh bölümü, `λ₁` = saklanan hedeflerin en küçük Rayleigh bölümü (`(TᵀKT)_kk/(TᵀMT)_kk`'nin min'i) olmak üzere `grad_frac_lb_i = max(0, 1 − RQ_i/λ₁)`.
     Türetme: `V_i = c + g`, `c ∈ range(P)`, `g = (I−P)V_i`; `K g = 0` ve `(c,g)_M = 0` olduğundan `RQ_i = cᵀKc/(cᵀMc + gᵀMg)`, ve bölünmüş-uzayda `cᵀKc ≥ λ₁ cᵀMc` (en küçük sıfırdan farklı özdeğer),
     dolayısıyla `gᵀMg/‖V_i‖² ≥ 1 − RQ_i/λ₁`. **Hakemin "V = c + a·g için tam" iddiası yalnız `c` en düşük özmodlar (RQ_c = λ₁) için doğrudur**; genel `c` için eşitsizliktir:
     yüksek frekanslı rot içeriği olan bir sütun (RQ_c ≫ λ₁) istediği kadar gradyan kütlesini saklayabilir, yani bu bir **alt sınırdır, ölçü değildir** (gerçek payı küçümser; "0" gradyan olmadığını göstermez). Testte (pillbox): `V_i = T_k + a g` için
     en düşük modda alt sınır `a²/(1+a²)`'ye 10⁻⁴ içinde eşit, üst modlarda kesin küçük (a = 2'de > 0,01 eksik), ve rastgele karışık tabanda projekteli tabanın gerçek payı `1 − diag(M_div)/diag(M_V)`'yi hiçbir sütunda aşmaz. Sıfır sütun 0 sayılır.
     `λ₁` p2 etiket + aynı mesh için tam (en düşük ayrık özdeğer); p3→p2 izdüşümlü etiketlerde izdüşümün Rayleigh bölümüdür (p2 özdeğerinden büyük olabilir, farkı ölçülmedi) ve sınır o oranda gevşer. DDP: her adımda ve her sırada loglanır
     (`…_max` `reduce_fx='max'`). Arayüzde "gradyan kütle payı (alt sınır)" eğrisi (yoksa gizli).
   - `train/mass_diag_ratio`, `val/mass_diag_ratio`: örnek içinde `max/min diag(VᵀMV)` (sütun kütle normlarının karesi; sıfır sütun için tavan 10³⁰), batch ortalaması. Gradyan-ağır ya da ölçeği kaçmış sütunlar bunu büyütür. **Zayıf, dolaylı bir göstergedir**:
     eşit normlu ama gradyan-ağır sütunları göstermez, ve sağlıklı değer aralığı bilinmiyor (küçük fixture'da başlangıçta 8–13). Eğilime bakın: ani/sürekli artış alarm sayılır.
   - `val/span_rel_l2` ve `val/span_mode_k_rel_l2` (kütle normunda mod başına en iyi yaklaşım hatası) ile `val/span_rel_l2_energy` (enerji normunda): gradyan içeriği yalnız kütle normunu bozar (enerji normu gradyanı görmez),
     bu yüzden iki hata arasındaki oranın eğitim boyunca kayması bir ipucudur (güvenilir bir ölçü değil: iki normun oranı gradyandan bağımsız olarak da modlara ve mesh'e bağlıdır). `train/span_*` ile `val/span_*`
     aynı fonksiyondur, doğrudan karşılaştırılır (genelleme açığı).
   - **Ara test:** en iyi/`last.ckpt` checkpoint'ine eğitim sırasında ayrıca projekteli değerlendirme (`scripts/eval_fidelity.py --spaces native`, ya da `trainer.test`) çalıştırmak, `grad_frac` ve gerçek frekans hatasını gösteren tek yoldur. Eğitim hiçbir zaman bu sayıları otomatik görmez.

## 4. Kullanım

```bash
python train.py --config configs/field_3d_noproj.yaml --override "dataset.data_path=data/field/*.h5"
# ya da
python train.py --config configs/field_3d.yaml --override "dataset.data_path=data/field/*.h5" \
    training.projection=false training.selfsup_weight=0 training.freq_weight=0
```

- `configs/field_3d_noproj.yaml`, `field_3d.yaml` ile aynıdır (`tests/test_projection_free.py` bunu denetler); yalnız `training.projection: false`,
  `selfsup_weight: 0`, `freq_weight: 0`, `exp_name: field3d_noproj_v1`.
- `training.projection` yalnız `true` / `false` olabilir (başka bir değer `ValueError`). Eski checkpoint'lerde anahtar yok: `projection=true` olarak yüklenir.
- **İzlenen metrik:** `val/span_loss` (ModelCheckpoint, EarlyStopping, `reducelr` zamanlayıcısı; `val/loss` yalnız loglanır, §1.1); `projection: true` iken `val/field_rel_l2`. Tek yer: `lightning_module.monitor_metric(projection)`; `train.py:_selection_callbacks`
  callback'leri kurar. `--fast_dev_run` checkpoint yazmaz (callback seçimi testte doğrudan denetlenir).
- **`--resume`:** checkpoint'in `projection` değeri config'ten farklıysa `train.py` **`ValueError`** verir (kayıp ve izlenen metrik değişir; ModelCheckpoint/EarlyStopping başka metriğin en iyi skorunu geri yüklerdi).
  Geçiş için `--init_from <ckpt>` ile yeni koşu başlatın (yalnız ağırlıklar). Kontrol her sırada çalışır (DDP'de tek sırada hata diğerlerini asılı bırakırdı). Anahtarı olmayan eski checkpoint `projection=true` sayılır.
- **TRUBA:** `TRAIN_EXTRA="training.projection=false training.selfsup_weight=0 training.freq_weight=0"` (cluster/truba/README.md; `run.sh train/all` ve `submit_train.sh` aynı değişkeni alır). Arayüzün TRUBA sekmesi
  `train_extra` alanıyla aynısını geçirir; arayüzün yerel Eğitim sayfası bu anahtarı göndermez (varsayılan: projeksiyon açık, kod değişmedi).
- **Web arayüzü / `src/service`:** `progress.json` okuyucusu (`src/service/training.py`) `best.monitor`'u olduğu gibi taşır; Eğitim sayfası en iyi skoru `val/span_loss` (ve eski `val/loss`) için `sci`, `val/field_rel_l2` için yüzde olarak yazar ve
  anahtarı gösterir; eğrilerde olmayan anahtarların grafiği çizilmez (`field_rel_l2`, `freq_rel_err` yoksa gizlenir, `field_rel_l2` yoksa span hatası grafiği, `mass_diag_ratio` varsa oran grafiği gösterilir). `CURVE_KEYS` genişledi:
  `val/span_loss`, `train/span_rel_l2`, `train/mass_diag_ratio`, `val/mass_diag_ratio`, `train/grad_frac_lb`, `val/grad_frac_lb`, `val/grad_frac`. `val_field_rel_l2` yoksa `null` döner. Arayüz derlendi (`npm run build`, tsc + vite temiz; `node_modules` ana depodan salt-okunur symlink ile geçici dizinde).
- Tahmin yolları (`src/service/model.py`, `scripts/predict_geometry.py`, TRUBA `predict.sh`, `scripts/eval_fidelity.py`) değişmedi; checkpoint'te
  `projection=false` bulunması onları etkilemez (yalnız `training_step`, `validation_step`, `test_step` bakar). Checkpoint seçimi (`resolve_checkpoint`) `val_rel_l2=`, `val_span_loss=` ve `val_loss=` etiketlerini okur.
- Ölçüm: `python scripts/profile_field_step.py <H5> --sample 0 --grad_checkpoint --no_projection` (eğitim adımı projeksiyonsuz; çıkarım geçişi ayrıca projekteli ölçülür).

## 5. Uygulama

| dosya | değişiklik |
|---|---|
| `src/models/hcurl.py` | `hcurl_grams_plain` (projeksiyonsuz Gram'lar) |
| `src/models/eigenspace_operator_field.py` | `forward_basis` (gövde + DOF eşlemesi; `BASIS_KEYS`), `forward` aynı (anahtar denetimi `_require`'a taşındı) |
| `src/training/lightning_module.py` | `projection` argümanı (+ `ValueError`), `_compute_loss_3d_plain` (train ve val), `_normalise_columns` (çökmüş sütun ölü), `_label_lambda1`, `_grad_frac_lower_bound`, `_span_terms(prefix=None, selfsup, ortho)`, `_basis_conditioning_loss(rel_floor)`, `monitor_metric`, `training_step` / `validation_step` dalı, `test_step` (`test/loss_plain`), epoch sonu tablosu |
| `src/training/checkpoint.py` | `resolve_checkpoint`: `val_rel_l2=`, `val_span_loss=` ve `val_loss=` etiketleri |
| `src/data_gen/highorder.py` | `field_operators(potential=True)`: `False` iken `G`, `Kp` kurulmaz (`n_pot = None`) |
| `src/data/field_dataset.py` | `FieldDataset(build_potential=True)`, `field_item` / `field_collate` G/Kp yokken çalışır, önbellek (aşağıda), `labels_are_model_eigenvectors` + uyarı (train, val) |
| `train.py` | `training.projection` okunur ve denetlenir, train ve val bölmeleri `build_potential=projection`, test her zaman `True`, `_selection_callbacks`, `_checkpoint_filename`, `--resume` hatası (`_check_projection_matches`) |
| `src/service/training.py`, `web/src/components/TrainPage.tsx`, `web/src/i18n.ts` | `CURVE_KEYS`; monitöre göre skor biçimi; olmayan anahtarların grafiği gizlenir; span ve oran grafikleri |
| `scripts/profile_field_step.py` | `--no_projection` |
| `src/data/field_dataset.py`, `configs/*.yaml`, `cluster/truba/train.sbatch` | `dataset.max_tets` (§8); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

**Model ileri geçişinde bayrak yerine ayrı yöntem.** `forward` hiç değişmedi (aynı satırlar), yeni `forward_basis` onun ilk adımıdır. Böylece
varsayılan yolun bit düzeyinde aynı kalması kod okunarak da görülür, ve projeksiyonsuz batch'te (G/Kp yok) yanlışlıkla Ritz çağırmak
`ValueError` ("needs batch keys ['G', ...]") verir, sessizce yanlış sonuç vermez. Test adımı G/Kp'siz batch alırsa aynı hatayı verir.

**Önbellek (`dataset.cache_dir`).** Anahtar eki seçildi, "her zaman tam sözlüğü yaz, G/Kp'yi yüklemeden sonra at" seçilmedi:
`{tag}__{key}.nopot.ops2.pkl` yalnız `build_potential=False` tarafından (train ve val) okunur ve yazılır; tam varyant `{tag}__{key}.ops2.pkl` dosyasını kullanır (dosya biçimi ve boyutlar: §9).
Tam bir pikl varsa projeksiyonsuz varyant onu okuyup G/Kp'yi atar (yeniden kurulum yok). Böylece (i) G/Kp'siz bir önbellek, G/Kp isteyen bir bölmeye (test)
hiçbir koşulda sunulmaz (elle kopyalanmış olsa bile `ValueError`), (ii) train ve val bölmelerinin ilk epoch'u da G/Kp kurmaz ve yazmaz (~0,5 s/geometri, docs/31 §3.7), (iii) önbellek diski küçülür.
Bedel: aynı geometri iki varyantla önbelleğe yazılırsa iki dosya (test bölmesi ya da farklı projeksiyon ayarlı koşular).

**Bit düzeyinde aynılık (`projection: true`).** Aynı sabit pillbox + kutu batch'i, aynı tohum, `grad_checkpoint` açık: değişmemiş kodla (`50f8f82`) ve değişmiş kodla
eğitim kaybı, 40 log değeri, 28 parametre gradyanı, özdeğerler, alanlar, baz ve doğrulama kaybı `torch.equal` ile aynı (bu oturumda, val/test/izleme değişikliklerinden sonra yeniden çalıştırıldı:
kayıp, `preds`, `targets`, log sözlüğü, gradyanlar, özdeğerler, alan, baz, val kaybı: hepsi `True`). (Etiketler NGSolve özdeğer çözücüsünden gelir;
süreçler arasında etiket işaretleri değişebilir, bu yüzden `preds` / `targets` mutlak değerle karşılaştırılır.) Karşılaştırma bir betikle yapıldı
(depoda değil, testte değil); testte `test_default_projection_loss_is_composed_as_before` bileşimi, `test_monitor_follows_the_projection_option` varsayılan izlenen metriği ve dosya adını denetler.
İnceleme düzeltmelerinden sonra yeniden çalıştırıldı (`bitid.py`, aynı sabit batch): kayıp, 40 log değeri, özdeğerler, taban, alan, val kaybı `torch.equal`; 28 gradyan, aynı kodla değişmemiş depo arasında da süreçten sürece 3·10⁻⁸ (float32) oynıyor (değişmemiş kod iki süreçte 18 gradyanda ayrıştı, sonraki iki süreçte aynıydı): değişmiş kod, değişmemiş kodun ikinci/üçüncü süreciyle 28/28 bit düzeyinde aynı. Süreçler arası gürültü eşzamanlı çalıştırma/iş parçacığı sırasından gelir, koddan değil.

## 6. Ölçümler

Tek koşu, tek örnek; birbirinden bağımsız tekrar yok. CPU, 4 çekirdek, `torch.get_num_threads() = 4`, NGSolve 4 iş parçacığı, eşzamanlı iş yok (ölçüm sırasında başka süreç çalıştırılmadı).

### 6.1 Etiketlerin gradyan içeriği

`‖GᵀMt‖ / ‖Mt‖` (Öklid, istenen) ve `‖P_G t‖_M / ‖t‖_M = √((GᵀMt)ᵀ Kp⁻¹ (GᵀMt) / tᵀMt)` (kütle normunda gradyan payı, fiziksel olan), her saklanan mod için
(`u_model`, float32 saklama, wall DOF'ları 0). Kp çözümü: Jacobi-CG, tol 1e-12.

| örnek | etiket | mesh | `‖GᵀMt‖/‖Mt‖` (min–maks, 10 mod) | gradyan payı (min–maks) | `‖GᵀKt‖/‖Kt‖` |
|---|---|---|---|---|---|
| pillbox_0 (`adapt_meas/pillbox_0_adaptive.h5`) | p2, `n_tets_label == n_tets` = 10434 | model mesh = etiket mesh | 4,7e-8 – 5,1e-8 | 1,8e-8 – 1,9e-8 | ≤ 1,5e-14 |
| hwr_3670021 (`adapt_meas/hwr_3670021_adaptive.h5`) | p2, 21596 tet | aynı | 4,0e-8 – 5,2e-8 | 1,8e-8 – 2,0e-8 | ≤ 2,8e-14 |
| kalibrasyon kutusu (test fixture'ı, 4 mod) | p3 → p2, **aktarılmış**: `n_tets_label` 192 ≠ `n_tets` 46, izdüşüm hatası 12 % | farklı | 6,9e-6 – 5,9e-5 | 3,4e-5 – 2,0e-4 | |
| kalibrasyon pillbox (aynı) | p3 → p2, **aktarılmış**: 163 ≠ 57, izdüşüm hatası 8 % | farklı | 1,1e-5 – 1,8e-4 | 4,7e-5 – 6,3e-4 | |

p2 etiketlerde gradyan payı float32 saklama düzeyinde (~2·10⁻⁸, kare olarak ~4·10⁻¹⁶): §2.2'deki eşitlikler orada pratikte tam. **Aktarılmış** p3 → p2 etiketlerde 10⁻⁵–10⁻³,
yani 3–4 mertebe büyük ama küçük. Tek küçük fixture ve iki gerçek örnek; aile bazında yaygın ölçüm yapılmadı.

**Aynı mesh'te p3 → p2 (iç içe uzay; sahibin karara dayanak olarak verdiği ve burada ölçülen durum).** Çift duyarlıkta, mesh'i doğrudan kurup (`solve_modes` p3, `project_to_model` p2, model operatörleri `field_operators`,
`dx₂`), 4 mod, depoya konmayan betikle: `e = ‖GᵀMt‖/‖Mt‖`, `f` = kütle normunda gradyan payı:

| geometri | tet / p2 DOF | p2 çözümü kendisi: `e`, `f` | p3 → p2, aynı mesh: `e` (en büyük) | `f` (en büyük) | izdüşüm hatası |
|---|---|---|---|---|---|
| pillbox, maxh 0,05 | 109 / 1452 | ≤ 2,5·10⁻¹⁶, ≤ 4,3·10⁻¹⁶ | 3,2·10⁻¹⁰ | 2,8·10⁻⁹ | %0,8–3,2 |
| pillbox, maxh 0,03 | 532 / 6195 | ≤ 1,1·10⁻¹⁵, ≤ 5,2·10⁻¹⁶ | 1,8·10⁻¹² | 1,3·10⁻¹¹ | %0,18–0,75 |
| kutu, maxh 0,25 | 150 / 1872 | ≤ 7,2·10⁻¹⁶, ≤ 5,1·10⁻¹⁶ | 4,9·10⁻¹⁵ | 2,3·10⁻¹⁴ | %0,7–2,7 |

Yani iç içe izdüşümde gradyan payı, izdüşüm hatası %1 olsa bile, ≤ 3·10⁻⁹ (çift duyarlık; diskte float32 saklama ~2·10⁻⁸ tabanı ekler) ve mesh inceldikçe düşüyor (`dx₂`/`dx₃` fark terimi). Bu, aktarılmış etiketlerden (10⁻⁵–10⁻³) 4–6 mertebe küçük.
`tests/test_projection_free.py::test_p3_labels_projected_onto_p2_of_the_same_mesh_are_m_orthogonal_to_the_gradients` pillbox için `e < 10⁻⁷` denetler. Gerçek bir üretici çıktısında (adaptif p3, `n_tets_label == n_tets`) ölçülmedi; geometri/kural farkı olan örnekler için aile bazında ölçüm yok.


### 6.2 Eğitim adımı maliyeti

`scripts/profile_field_step.py comp_adaptive.h5 --sample 0 --reps 3 --threads 4 --grad_checkpoint` (kompozit geometri, 23 155 tet, 234 738 DOF, 96 042 potansiyel,
324 170 token, küçük model 0,854 M parametre, 6 mod; CAD'den mesh yeniden kurulumu ayrıca 336 s). `--no_projection` aynı komut. Tepe RSS işlem geneli, `ru_maxrss`.

Ayrı süreçlerde (her biri `--reps 3`, tekrar sırası 0, 1, 2; #0 soğuk bellek etkisi taşır, docs/31 §3.6 ölçüm notu):

| | `projection: true` | `projection: false` |
|---|---|---|
| eğitim ileri + kayıp, #0 / #1 / #2 | 12,96 / 11,36 / 12,50 s | 9,18 / 9,53 / 7,42 s |
| eğitim geri, #0 / #1 / #2 | 27,63 / 23,60 / 23,22 s | 20,36 / 21,50 / 19,37 s |
| Kp CG (ileri + geri), #1 | 2,84 + 3,10 s (2 çağrı, 130 iterasyon) | 0 çağrı |
| tepe RSS (eğitim adımları) | 6,09 – 6,19 GB | 5,90 – 5,94 GB |
| eğitim batch'i `G`, `Kp` | var (nnz 0,14 M, 4,21 M) | yok |

Ayrı süreçler arası farkın bir kısmı makine değişkenliğidir: aynı koda sahip çıkarım geçişi ikinci süreçte 11,2–12,3 s yerine 9,3–9,7 s sürdü. Bu yüzden
karşılaştırma **aynı süreçte, sırayla değiştirilerek** tekrarlandı (depoda olmayan geçici betik `ab.py`, `profile_field_step.py`'nin işlevlerini kullanır: iki batch, aynı model ağırlıkları, 6 tur, tur başına sıra
çevrilerek; ilk tur dışarıda, 5 tur):

| eğitim adımı (ileri + kayıp + geri) | ortalama | en küçük – en büyük |
|---|---|---|
| `projection: true` | 10,66 + 23,20 = **33,86 s** | 31,33 – 35,20 s |
| `projection: false` | 7,20 + 19,76 = **26,96 s** | 24,78 – 28,12 s |
| fark | −3,46 + −3,44 = **−6,9 s (−%20)** | |

Fark, ölçülen CG süresine (2,8–3,1 + 3,1–3,3 s) uyuyor: kazancın ~6 s'si Kp CG'dir, kalan ~1 s Ritz (m = 24, 24×24 özdeğer problemi) ve Gram farkına karşılık gelir ama gürültü içindedir.
Adımın geri kalanı (gövde, ~27 s) değişmiyor. Bu, docs/31 §3.6'daki "ağ küçük, maliyet gövdede" bulgusuyla uyumlu: 23k tet'te beklenen kazanç ~%20, 10× değil.
Tepe bellek yalnız ~%4 düşüyor (gövde etkinleştirmeleri ve 324k token baskın, G/Kp küçük): bellek için bu seçenek bir çözüm değil.
Veri hazırlığı (örnek başına, bir süreçte dönüşümlü 4 tur, ilk tur dışarıda; `ops_ab.py`): `field_operators` G/Kp'li 6,04 s, G/Kp'siz 5,11 s (≈ −0,9 s);
`field_item` + collate G/Kp'li 7,18 s (6,52–8,15), G/Kp'siz 6,80 s (4,53–10,38): tur-içi dağılım farktan büyük, **ayrılamadı** (ayrı süreçlerdeki ilk ölçümler
de soğuk bellek etkisiyle bu kadar dalgalı). docs/31 §3.7'deki ~0,5 s/geometri (13k tet) ile uyumlu büyüklükte.
Tek örnek (kompozit, 23k tet), tek makine; başka boyut ve aile için ölçülmedi.

### 6.3 Testler

- Tam takım (inceleme düzeltmelerinden sonra): `python -m pytest -q tests/` → **290 geçti** (11 dk 33 s; 4 çekirdek, makinede başka iş görülmedi; `ruff check .` ve `vulture` temiz). Önceki sürüm: 283, daha önceki (yalnız eğitim projeksiyonsuz): 274, değişiklik öncesi (`50f8f82`): 259.
  İnceleme düzeltmelerinin testleri: çökmüş sütun ölü + canlı sütun gradyanı sınırlı, `grad_frac_lb` alt sınır / en düşük modda tam / loglama, `span_norm: energy` hatası, `--resume` projeksiyon uyuşmazlığı hatası, `test_step` gövdeyi bir kez çalıştırır, `resolve_checkpoint` üç etiket.
  Yeni/değişen testler: `tests/test_projection_free.py` (20) ve `tests/test_field_labels.py` (`test_noproj_fast_dev_run_train_py`, `test_noproj_train_py_trains_selects_by_val_loss_and_tests_projected`, `test_dataset_without_potentials_cache_and_label_warning`), `tests/test_service_training.py` (`test_projection_free_run_is_listed_without_ritz_metrics`).
- Denetlenenler: G/Kp'siz operatörler = tam operatörler − {G, Kp}; Gram'lar projeksiyonlu olanlarla (enerji bit düzeyinde, çapraz blok 1e-6 içinde, `VV` = `M_V`, `M_V − M_div ⪰ 0`, `r_düz ≥ r_proj`) ve scipy ile bağımsız kurulmuş Gram'larla uyumlu;
  kayıp = `span + 0,01·ortho(enerji)` (scipy Gram'ından, bağımsız); **val kaybı = aynı batch'te train adımı kaybı** (eval modu, aynı ağırlıklar, `torch.equal`, 9 anahtarın hepsi eşit) ve `test/loss_plain` aynı değer;
  train ve val adımında 0 `_pcg` / `KpSolve.apply` çağrısı, val batch'inde G/Kp yok, test adımında CG > 0 ve `test/field_rel_l2` vb. loglanır, G/Kp'siz batch ile test `ValueError`;
  izlenen metrik: `monitor_metric`, `_selection_callbacks` (ModelCheckpoint + EarlyStopping izleme, dosya adı, `reducelr` zamanlayıcısı) iki modda; `resolve_checkpoint` iki etiketi okur;
  ortho gradyan-kör (gradyan eklenince değişmez) ve doğrusal bağımlılığı yakalar; sütun ölçeğinden bağımsızlık (10⁻⁶ … 10⁶); hakem sondajı (ölü sütun, saf gradyan sütunları ×10⁷);
  aynı mesh'te p3 → p2 etiketleri `M`-dik (`e < 10⁻⁷`); `ValueError` koruması; anahtarsız checkpoint `projection=True` yüklenir; config farkı yalnız dört anahtar;
  `train.py --fast_dev_run` iki modda (projeksiyonsuzda train **ve val** batch'inde G/Kp yok, CG yalnız testte); `train.py` 2 epoch uçtan uca (`val_loss=` dosya adı, `progress.json` `best.monitor = val/loss`, yalnız ortak anahtarlar, test projekte + `test/loss_plain`);
  önbellek varyantları karışmaz; aktarılmış-etiket uyarısı (train ve val); arayüz servisi `val/field_rel_l2`'siz bir koşuyu listeler (`monitor`, `CURVE_KEYS`).
- Bit düzeyinde aynılık (`projection: true`): §5. Arayüz: `npm run build` (tsc + vite) temiz (çalışma ağacında `node_modules` yoktu; kaynaklar ve ana depodaki `node_modules` kopyası geçici dizinde derlendi).

## 7. Yapılmayanlar ve açık sorular

- **Hiçbir eğitim çalışması yapılmadı.** Soru: aynı veri, aynı bölme (geometri kimliğine göre), aynı model boyutu, aynı adım sayısı ve ≥ 3 tohumla
  `projection: false` (`val/span_loss` ile seçilmiş checkpoint) testte (`test/field_rel_l2`, frekans bağıl hatası, `test/grad_frac`, Ritz marjı) `projection: true`'dan (`val/field_rel_l2` ile seçilmiş) kötü mü, eşit mi?
  "Eşit maliyet" iki türlü okunabilir (eşit adım: `false` adım başına daha ucuz; eşit duvar saati: `false` daha çok adım atar); ikisi ayrı raporlanmalı. Tek tohum "tek koşu" sayılır.
- `val/span_loss` ile `test/field_rel_l2` arasındaki ilişki bilinmiyor: `val/span_loss` seçimi projekteli en iyi checkpoint'ten farklı bir epoch seçebilir.
- Frekans doğruluğu: frekans ve selfsup terimleri olmadan Ritz frekansları (özellikle `drop_tol` ile atılan yönlerin sayısı) ne yapar, bilinmiyor.
- Aktarılmış p3 etiketlerle (HF ince ayar, docs/34 §4.3; adaptif etiket mesh'i model mesh'inden farklı) kullanım: uyarı verir, etkisi ölçülmedi.
- Karma kullanım: aynı koşuda LF/HF `Fidelity` ağırlıkları span terimine uygulanır (`fidelity_freq_scale` etkisizdir, frekans terimi yok).
- Doğrulama adımının maliyet kazancı (CG ileri geçişi yok) ölçülmedi; §6.2 yalnız eğitim adımıdır.
- `--resume` ile `projection` değiştirmek **hata**dır (§4); geçiş `--init_from` ile. `val/loss` ile seçen önceki (yayımlanmamış) sürümün checkpoint'lerinden `val/span_loss` ile devam etmek desteklenmez (aynı hata değil, ama ModelCheckpoint eski metriğin skorunu geri yükler).
- `grad_frac_lb` bir alt sınırdır (§3.5); gerçek gradyan payı yalnız test/ara testte ölçülür. Çökmüş-sütun eşiği (`COLUMN_FLOOR = 1e-6`) ve float32 `K` gürültüsüne duyarlılığı gerçek eğitimde denenmedi.

## 8. GPU belleği koruması: `dataset.max_tets`

`dataset.max_tets` (varsayılan `null` = sınır yok; `FieldDataset(max_tets=)`), model mesh'i bu kadar tetten fazla olan örnekleri eler (H5 özniteliği `n_tets`; sınır dahil, `n_tets ≤ max_tets` kalır).
- **Bölmeyi değiştirir.** Eleme **tüm bölmelerde, tohumlu train/val/test bölmesinden önce** yapılır (`require_label_ok` ile aynı yer): `max_tets` koyunca kalan örneklerin permütasyonu değişir ve geometrilerin hangi bölmeye düştüğü, `max_tets = null` koşusundakinden farklıdır.
  İki koşuyu (ör. projeksiyonlu ve projeksiyonsuz) karşılaştırırken ikisinde de aynı `max_tets` kullanılmalı. Test kümesi de küçülür: büyük geometriler testte de yoktur (rapor edilirken belirtilmeli).
- Eleme sayısı aile bazında **bir kez, train bölmesi oluşturulurken** yazılır ("`max_tets=N dropped K samples (…) before the train/val/test split`"); sayı tüm bölmeler için aynıdır, val/test için tekrar yazılmaz. Hepsi elenirse `ValueError`.
- Bellek: bir eğitim adımının belleği tet sayısıyla yaklaşık doğrusal artar. Aşağıdaki GPU ölçümleri TRUBA'dan; yerel CPU ölçümü (§6.2) yalnız tepe RSS'dir.

**TRUBA ölçümü (2026-11-03, akya-cuda, Tesla V100-SXM2-16GB, küçük model 0,854 M, 6 mod; `scripts/profile_field_step.py --device cuda --threads 10 --reps 2 --grad_checkpoint --no_projection`; tek koşu, tek örnek/boyut):**

| örnek | tet | DOF | token | eğitim adımı (projeksiyonsuz), tepe GPU tahsisi | adım ileri + geri (ısınmış; ilk) | çıkarım (projekte), tepe | çıkarım süresi (CG) | `field_operators` (G/Kp'li) | .vol'den yeniden kurma | item / collate |
|---|---|---|---|---|---|---|---|---|---|---|
| elliptical sample_524388 | 23 711 | 251 892 | 331 954 | **4,74 GB** | 0,23 + 0,51 s (1,29 + 1,01 s) | 3,06 GB | 0,39 s (CG 0,17 s, 120 it) | 4,08 s (4,90 s) | 2,70 s | 0,39 / 0,52 s |
| ridged_box sample_2097174 | 51 038 | 542 820 | 714 532 | **10,18 GB** | 0,49 + 1,08 s | 6,56 GB | 0,82 s (CG 0,35 s) | 8,79 s (10,23 s) | 3,99 s | 0,89 / 1,01 s |
| hwr sample_3670100 | 81 530 | | | **CUDA OOM** eğitim adımında (15,77 GiB toplam; 11,88 tahsisli + 3,26 ayrılmış-tahsissiz) | | | | | | |

Bellek ≈ 0,2 GB / 1000 tet (doğrusal). Üç noktadan ikisi arasında doğrusal enterpolasyon; 81k tet'in 16 GB'a sığmadığı ölçüldü. OOM'daki 3,26 GiB parçalanmadır: `cluster/truba/train.sbatch` artık `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
dışa aktarır (`container_prefix` `--cleanenv` kullanmaz, `apptainer exec` ana makine ortamını geçirir; doğrulandı: `cluster/truba/config.sh`). Etkisi ölçülmedi.
v3 `TEST=1` train-grubu verisi (878 örnek): > 40k tet 170 örnek, > 50k tet 95, > 60k tet 44. Sahibin koşusu `dataset.max_tets=60000` kullanır (≈ 12 GB tepe, 0,2 GB/1000 tet ile; 60k'nın üstündeki 44 örnek, 878'in %5'i, hem train hem val/test'ten çıkar).

## 9. Operatör önbelleği: biçim ve boyut (`src/data/ops_cache.py`)

> 2026-11-03, aynı dal, **commit edilmemiş**. Sahibin isteği: aynı içerik, ~3× küçük disk; boyut üst sınırı yok. Davranış (kayıp, gradyan, tam varyantın içeriği) değişmedi; yalnız
> diskteki biçim ve train/val değerlerinin duyarlığı (float32, aşağıda ölçüldü).

**Biçim.** Geometri başına tek dosya, `pickle` içinde `format = 2` sürümlü sözlük (`ops_cache.pack` / `unpack` / `write` / `read`):

| öğe | eski (`.ops.pkl`) | yeni (`.ops2.pkl`) |
|---|---|---|
| K, M | tam CSR, float64 + int32 | **üst üçgen** (köşegen dahil); yükte tam CSR yeniden kurulur (COO → CSR sayım sıralaması, değerler yalnız taşınır, toplanmaz; açık sıfırlar yerinde kalır) |
| indis dizileri | int32 | int32 (aralık denetlenir, sığmazsa int64: `index_dtypes`), **farkları** saklanır (CSR satırları ardışık sütun koşularıdır) |
| tüm diziler | ham | ilk 1 MiB'lik deneme zlib-1 ile ≥ %10 kazandırıyorsa zlib, yoksa ham (sıkıştırılamayan değerler açma maliyeti ödemez) |
| K, M, C değerleri | float64 | **projeksiyonsuz (`.nopot`) varyantta float32**, tam varyantta float64 |
| P, qx, qw | float64 | nopot'ta float32 (kayıpsız: `field_item` zaten float32'ye çevirir), tam'da float64 |
| `u_model` (etiketler) | float64 | float32 (H5'te zaten float32: kayıpsız; öyle değilse float64 kalır) |
| G, Kp, bnd, qfeat, center | ham | aynı değerler (G, Kp float64; Kp = GᵀMG bit düzeyinde simetrik değil, tam saklanır) |

Yüklemede her dizi özgün dtype'ına (float64, int32 …) döner; aşağı akış kodu (`field_item`, `field_collate`, `block_diag`, `SymmetricMK`) değişmedi.

**Simetri denetimi.** Yazarken `pack`, üst üçgenden yeniden kurulan K / M'yi özgün matrisle **bit bit** karşılaştırır (indptr, indices, değerlerin bit deseni; `-0,0 ≠ +0,0`). Eşit değilse (ya da matris kare değil / kanonik değil)
o matris tam saklanır (`sym: false`). Ölçülen: K ve M iki örnekte de bit düzeyinde simetrik (NGSolve simetrik depolamadan, tek üçgenden kurulur), Kp değil. float32'ye yuvarlama simetriyi bozmaz: yuvarlanan üst üçgen yansıtılır, K, M yüklemede **tam simetrik** (`(K != K.T).nnz == 0`, testte), yani `SymmetricMK = True` varsayımı geçerli kalır.
**K'nın %76'sı açık sıfırdır** (pillbox: 5,09M / 6,65M; gradyan DOF'larının rot'u); yapı korunur (aynı CSR), zlib bunları ~4× küçültür.

**Neden nopot float32, tam float64.** Projeksiyonsuz train/val kaybı yalnız `VᵀMV, VᵀKV, VᵀMT, VᵀKT` çarpımlarını kullanır (§2.4); göreli 6·10⁻⁸ yuvarlama modelin zaten float32 çalışan baz çıktısının (V) hata düzeyindedir, ve etiketler (`Y_field`) zaten float32'ydir.
Tam varyant test bölmesi içindir: Kp = GᵀMG, M ve G ile **float64'te tutarlı** kalmalı (çekirdek projeksiyonu ve Kp CG float64 değişmezi, AGENTS.md); float32'ye yuvarlanmış Kp, GᵀMG'den ~10⁻⁷ sapardı (etkisi **ölçülmedi**); sahibin kararı ve AGENTS.md değişmezi gereği riske girilmedi. Bu yüzden tam varyantta hiçbir değer daralmaz (bit bit aynı, test + ölçüm).
float32'nin tek sınırı: |a| < 1,2·10⁻³⁸ olan girdiler duyarlık kaybeder (montaj gürültüsü ~10⁻⁴⁰; K, M için anlamsız), > 3,4·10³⁸ olan girdi `OverflowError` verir.

**Sürüm ve eski dosyalar.** Yeni dosya adı `….ops2.pkl` / `….nopot.ops2.pkl`; eski `….ops.pkl` / `….nopot.ops.pkl` dosyaları **okunmaz** (yok sayılır, operatörler yeniden kurulur, eski dosyaya dokunulmaz: elle silinmeli, yoksa disk boşuna dolar).
Dosyanın içindeki `format` sayısı da denetlenir: farklı sürüm ya da bozuk/yarım dosya `UserWarning` ile yeniden kurulur ve üzerine yazılır. Tam önbellek nopot'u hâlâ besler (G/Kp atılır; değerler float64 kalır, yeniden yuvarlanmaz: bir çalışmada bir geometri hep aynı dosyadan gelir).
Önbelleğe ilk yazan çağrı (ilk epoch) **diskteki biçimin açılmış hâlini** döndürür: nopot'ta epoch 1 ile sonraki epoch'lar aynı (float32-yuvarlanmış) değerleri görür; `cache_dir` yoksa taze float64 kullanılır (fark ~10⁻⁷).

**Atomik yazma.** `write`: aynı dizinde `{ad}.{pid}.{rastgele}.tmp`, `pickle.dump`, `os.replace`; hata olursa `.tmp` silinir ve hedef dosya değişmez. DDP sıraları / DataLoader işçileri aynı dosyayı aynı anda yazarsa her biri tam dosya yazar, sonuncusu kazanır (testte 4 iş parçacığı × 3 yazma). Öldürülen bir yazıcıdan kalan `.tmp` dosyaları elle silinir (eskisi gibi).

### 9.1 Boyut ve süre (ölçüldü)

Tek koşu, tek örnek; CPU 4 çekirdek, `torch.get_num_threads() = 4`, `ngsolve.GetNumThreads()` = 1 (yükleme NGSolve kullanmaz), eşzamanlı iş yok (yükleme süreleri sıcak sayfa önbelleğinde; ağ dosya sisteminde okuma süresi ölçülmedi).
Yükleme = dosyayı oku + dizileri aç + K/M kur (`pickle.load` / `ops_cache.unpack(read())`), 6 tekrarın **medyanı** (aralık: nopot 23k tet 2,4–3,3 s, tam 3,7–10 s; en büyük değerler soğuk disk önbelleği / bellek baskısı). Eski sayılar aynı makinede `7797382` kodundan. Bu ortamda scipy indisleri zaten int32'ydi (eski dosya 12 B/girdi; "16 B/girdi" varsayımı bu makinede geçerli değil), kazanç int32'den değil üst üçgen + float32 + zlib'den.

| örnek | varyant | eski | yeni | oran | yeni yazma ek süresi (pack+yaz+aç) | yükleme eski → yeni |
|---|---|---|---|---|---|---|
| pillbox, 10 434 tet, 108 291 DOF | tam | 326,4 MB | 127,2 MB | 2,57× | ~3,2 s | 0,19 s → 2,07 s |
| | nopot | 304,1 MB | 58,1 MB | 5,24× | ~5,7 s | 0,20 s → 1,31 s |
| comp_adaptive, 23 155 tet, 234 738 DOF | tam | 725,3 MB | 284,1 MB | 2,55× | ~15 s | 0,54 s → 4,15 s |
| | nopot | 671,8 MB | 128,4 MB | 5,23× | ~7,4 s | 0,38 s → 2,52 s |

(Yazma ek süresi = yeni "kur + paketle + yaz + aç" − eski "kur + yaz" farkı; her geometri için bir kez.) Yükleme **6–10× yavaşlar** (zlib açma ~1 s, simetrik K/M kurulumu ~0,3–0,5 s/matris, tek iş parçacığı), ama taze kurulumdan (12–15 s) hâlâ ~5× hızlı ve CPU adımından (≈24 s/örnek) küçük.
V100'de 23,7k tet'lik adım ileri+geri ~0,75 s + item/collate ~0,9 s (§8 tablosu): 14 işçiyle yükleme (~2,5 s/işçi ⇒ ~5 örnek/s) adımı (~0,6 örnek/s) beslemeye yeter, ama **işçi sayısı azsa (≲ 4) yükleme darboğaz olur**; eğitim çalışmasıyla ölçülmedi. Disk okuma süresi (Lustre) yeni biçimde ~2,5–5× daha az bayt okur, ölçülmedi.
Dosya içeriği, nopot, 23k tet: C değerleri 43 MB, M 34 MB (sıkıştırılamaz float32), K 13 MB, P 11 MB, qfeat 11,7 MB, `u_model` 7,8 MB, indisler toplam ~12 MB.
Hedef "~3×" nopot'ta (train + val, ~750 örnek ≈ 0,75 TB → ~0,1 TB) aşıldı (5,2×), tam varyantta (yalnız test bölmesi) sağlanmadı (2,55×; C ve M float64 değerleri sıkıştırılamaz).

### 9.2 Sayısal etki (ölçüldü)

Aynı makinede, tek koşu, model ağırlıkları aynı (`torch.manual_seed(0)`, küçük model: `embed_dim 64, n_basis 12, num_field_modes 4, rff_dim 32, n_layers 2`), 1 örnek, `batch_size 1`; "taze" = `cache_dir` yok (float64 kurulum), "önbellekli" = `.nopot.ops2.pkl`'den okunan öğe.
Betik depoda değil (`scratchpad/cache/effect.py`).

| ölçü | pillbox 10 434 tet | comp_adaptive 23 155 tet |
|---|---|---|
| nopot: K, M, C'de `max‖ΔA‖/max‖A‖` (taze − önbellekli) | 3,9·10⁻⁸ / 3,9·10⁻⁸ / 4,3·10⁻⁸ | 2,7·10⁻⁸ / 3,4·10⁻⁸ / 3,1·10⁻⁸ |
| nopot: `X, Input_funcs, Area, QX, QP, Y_field, Y_freq, BndDof` | `torch.equal` (hepsi aynı) | hepsi aynı |
| nopot: `hcurl_grams_plain`, aynı V: `G_M`, `G_A`, `MT` göreli en büyük fark | 9,9·10⁻¹⁰, 7,1·10⁻⁸, 4,8·10⁻⁸ | 7,4·10⁻¹⁰, 1,2·10⁻⁷, 4,6·10⁻⁸ |
| nopot: eğitim kaybı `_compute_loss_3d_plain` (float32 skaler) ve bütün log değerleri | tam eşit (fark 0) | tam eşit |
| nopot: 44 parametrenin gradyanı: global `‖Δg‖/‖g‖`; `max|Δg|/max|g|`; tensör başına en kötü `‖Δg‖/‖g‖` | 9,5·10⁻⁸; 1,0·10⁻⁷; 5,4·10⁻⁶ | 1,1·10⁻⁷; 1,5·10⁻⁷; 5,2·10⁻⁶ |
| tam: tüm öğe girdileri (CSR indptr/indices/değer baytları, dtype, tensörler) | bit bit aynı (yazan çağrı ve dosyadan okuma) | bit bit aynı |
| tam: projekteli kayıp `_compute_loss_3d` + 44 gradyan | kayıp eşit, gradyanlar `torch.equal` | ölçülemedi (süreç bellek yetersizliğiyle öldü; öğeler bit bit aynı olduğundan aşağı akış girdisi aynı) |

Yani beklenen ~10⁻⁷ (float32 yarım-ulp 6·10⁻⁸) çıktı; gradyanın tensör başına en kötü farkı 5·10⁻⁶ en küçük normlu tensörde (norm oranı), global fark 10⁻⁷. Bu fark tek örnek, tek ağırlık kümesi için;
eğitim eğrisine etkisi (kayıp azalması, `val/span_loss`) **ölçülmedi**: float32 önbelleğin eğitim sonucunu değiştirmediği iddia edilmez, yalnız adım düzeyinde farkın 10⁻⁷ olduğu ölçüldü.

### 9.3 Testler

`tests/test_operator_cache.py` (6): tam varyant bit bit gidiş-dönüş (K, M `sym`, Kp ve C tam), nopot float32 (tam float32 yuvarlaması, simetri, P/qx/qw `field_item` girdisi aynı, düz pikle göre > 2,5× küçük), üst üçgenin
bit-bit doğrulaması (açık sıfırlar, eksik köşegen, 1 ulp asimetri, `-0,0`/`+0,0`, eksik yansıma, dikdörtgen, sırasız sütun → tam saklanır), tamsayı aralığı koruması (`index_dtypes`, fark taşması, float32 taşması), eski/yanlış sürüm/yarım dosya reddi,
atomik ve eşzamanlı yazma. `tests/test_field_labels.py::test_dataset_without_potentials_cache_and_label_warning` güncellendi: önbellekli öğe = taze öğe (tam bit bit, nopot float32), eski `.ops.pkl` okunmaz ve dokunulmaz, farklı sürüm/yarım dosya uyarıyla yeniden kurulur, tam önbellek nopot'u besler.
