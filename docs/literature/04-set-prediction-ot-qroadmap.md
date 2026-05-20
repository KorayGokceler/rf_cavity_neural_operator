# 04 — Set Prediction, OT Matching, Q-Faktör Yol Haritası

> Sıralamadan bağımsız K-mod set tahmini ve Hungarian eşleme: `ot_match`,
> `_compute_loss`. Ayrıca uzun vade: PEC→lossy, Q/G tahmini (gelecek head).
>
> ⚠️ Bu ortamda web egress kapalı (403); notlar yerleşik/arama bilgisinden.

### 1. End-to-End Object Detection with Transformers (DETR)
**Künye:** Carion, Massa, Synnaeve, Usunier, Kirillov, Zagoruyko, 2020, ECCV 2020, arXiv 2005.12872
**Çekirdek fikir:** DETR, object detection'ı doğrudan bir *set prediction* problemi olarak kurar. Sabit sayıda "object query"den paralel tahmin üretilir; ground-truth ile tahminler arasında scipy `linear_sum_assignment`'a denk gelen bir bipartite Hungarian matching yapılır. Kritik nokta: matching argmin'i *detach* edilir (gradient akmaz), sonra yalnızca eşlenmiş çiftler üzerinden differentiable bir kayıp (sınıf + box) hesaplanır. NMS / anchor gibi elle tasarlanmış bileşenleri kaldırır.
**Projeye uygulama:** Bu, bizim `ot_match` + `_compute_loss` şemamızın birebir kaynağı: K=3 slot ⇄ hedef mod eşlemesi tam DETR'in detached-matching/differentiable-matched-loss desenidir. Frekans+field maliyeti DETR'in sınıf+box maliyetinin analoğu. "No-object" sınıfı yerine bizde sabit K var; ileride değişken mod sayısına geçilirse DETR'in ∅ slot fikri doğrudan uygulanır. Q-G head eklendiğinde Q terimi maliyet matrisine ek bir DETR-tarzı bileşen olur.
**Risk/uyarı:** DETR matching'i de yakın-eşdeğer hedeflerde kararsızdır — bizim degeneracy flip sorunumuzu tek başına çözmez.

### 2. Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances
**Künye:** Marco Cuturi, 2013, NeurIPS (NIPS) 2013, arXiv 1306.0895
**Çekirdek fikir:** Klasik OT lineer programına bir entropik düzenleme terimi (−(1/λ)H(P)) eklenir; problem kesin konveks hale gelir ve çözümü Sinkhorn-Knopp matris ölçekleme iterasyonlarıyla (satır/sütun normalizasyonu) elde edilir. Sonuç: hard permütasyon yerine yumuşak, *differentiable* bir taşıma planı ve transport çözücülerden mertebelerce hızlı hesaplama. λ→∞ iken plan hard assignment'a yakınsar.
**Projeye uygulama:** `ot_match` içindeki `linear_sum_assignment` (hard, detached, gradient-kesik) yerine entropik Sinkhorn planı P koyulabilir: aynı maliyet matrisi C = `freq_match_weight·Δf² + cost_field` üzerinden soft, differentiable bir slot↔mod eşlemesi. Matched-loss, Σ_ij P_ij·C_diff_ij olarak yazılır; degeneracy'de hard flip yerine kütle iki yakın mod arasında bölünür, gradient pürüzsüzleşir. `_compute_loss` döngüsünde per-sample uygulanır.
**Risk/uyarı:** Düşük λ'da plan fazla bulanıklaşıp mod collapse'i (duplicate slot) körükleyebilir; `slot_ortho` cezası korunmalı.

### 3. Annealed Multiple Choice Learning (aMCL)
**Künye:** Letzelter, Perera, Rommel, Richard, Pérez, Cord, Gallinari, 2024, NeurIPS 2024, arXiv 2407.15580
**Çekirdek fikir:** MCL'in Winner-Takes-All (WTA) şeması greedy argmin olduğu için keyfi kötü lokal minimuma sıkışır. aMCL, WTA argmin'ini sıcaklık T'li bir Gibbs/softmin atamasıyla değiştirir ve T'yi yüksekten düşüğe annealing yapar (deterministic annealing / rate-distortion). Eğitim yörüngesi rate-distortion eğrisini tırmanır; kritik sıcaklıklarda hipotezler faz-geçişi gibi aniden "ayrışır", böylece mod collapse engellenir.
**Projeye uygulama:** Sinkhorn fikrini bir takvime bağlar. `ot_match`'i sıcaklıklı soft assignment yapıp T'yi (Sinkhorn'da 1/λ) eğitim boyunca soft→hard annealing yapmak, yakın-dejenere frekanslarda epoch'tan epoch'a Hungarian eşlemesinin "flip" etmesini durdurur: başta yumuşak keşif, sonda kararlı hard eşleme. Takvim `GNOTLightning` `__init__`'e bir `match_temp_schedule` hparam'ı ve `_compute_loss`'ta `self.current_epoch`'a bağlı T olarak girer.
**Risk/uyarı:** Kritik-sıcaklık geçişi K=3'te ani olabilir; annealing çok hızlıysa avantaj kaybolur, takvim ablation ister.

### 4. Permutation Invariant Training (PIT / uPIT)
**Künye:** Yu, Kolbæk, Tan, Jensen, 2016/2017, ICASSP 2017, arXiv 1607.00325
**Çekirdek fikir:** Çok-konuşmacılı kaynak ayrıştırmada "label permutation problem"i çözer: tüm çıktı↔hedef permütasyonları için pairwise MSE maliyeti hesaplanır, minimum maliyetli permütasyon seçilip yalnızca onun hatası geri yayılır (K küçükken tüm K! permütasyon, Hungarian'a denk). uPIT ise permütasyonu tüm utterance boyunca *kilitler* — frame'ler arası tutarlılığı zorlayarak permütasyon-kaçaklarını (swap) engeller.
**Projeye uygulama:** `ot_match`'imiz aslında PIT'in Hungarian genellemesi. uPIT'in asıl katkısı bizde "permutation-locking stabilizer" olarak slot'lanır: yakın-dejenere bir geometri için per-epoch yeniden eşleme yerine, ilk kararlı eşlemeyi (ör. EMA / cached perm) sabitleyip bir uPIT-tarzı consistency cezası eklemek, `degeneracy_mode` yolundaki flip-brittleness'i azaltır. aMCL annealing'i bittikten sonra hard fazda lock devreye girer.
**Risk/uyarı:** Erken yanlış kilitlenen permütasyon kalıcı bias yaratır; lock sadece düşük matching-entropisinde aktive edilmeli.

### 5. A Universal Deep Learning Strategy for Designing High-Quality-Factor Photonic Resonances
**Künye:** (RIDL) — Chen, Liu, Yu ve ark., 2021, Nature-family (photonics), arXiv 2105.03001 **[fetch başarısız — search bilgisinden]**
**Çekirdek fikir:** Naif uçtan-uca DL, yüksek-Q rezonanslarda başarısız: yüksek Q = spektrumda neredeyse delta-fonksiyonu keskinlikte tepe; ham spektrum regresyonu bu ultra-dar piki kaçırır (kayıp yüzeyi düz, örnekleme bu ölçeği yakalayamaz). RIDL bunun yerine spektrumu *rezonans-bilgili* parametrelere (Fano/Lorentzian: merkez frekans + linewidth/leakage rate) ayrıştırır; Q sonsuza giden BIC'lerde bile sonlu, iyi-koşullu nicelikler üzerinden öğrenir.
**Projeye uygulama:** Q-G head yol haritasının doğrudan gerekçesi. Q'yu doğrudan regrese etmek yerine G = Q·R_s ayrıştırması + log-ölçek, RIDL'in "leakage rate / linewidth üzerinden öğren" ilkesinin RF-kavite karşılığı: Q lossless limitte →∞ (bizde PEC, Q=∞) olduğundan ham hedef patlar; log G sonlu ve iyi-koşulludur. Yeni head, `_compute_loss`'ta field/freq ile birlikte OT-matched (aynı `perm`) ek bir log G MSE terimi olarak girer; maliyet matrisine de bir Δ(log G)² bileşeni eklenebilir.
**Risk/uyarı:** R_s geometri-bağımlı; G ayrıştırması ancak R_s tutarlı normalize edilirse iyi-koşullu kalır.
