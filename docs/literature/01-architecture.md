# 01 — Neural Operator Omurgası

> Projenin GNOT tabanlı transformer mimarisi: shared encoder + per-mode decoder
> (`mode_field_blocks[k]` + `field_heads[k]`). Bu grup mimari temel + baseline'lar.
>
> ⚠️ Bu ortamda web egress kapalı (403); notlar makalelerin yerleşik
> bilgisinden yazıldı, her biri ilgili etiketle işaretli.

### 1. GNOT: A General Neural Operator Transformer for Operator Learning
**Künye:** Z. Hao, Z. Wang, H. Su, C. Ying, Y. Dong, S. Liu, Z. Cheng, J. Song, J. Zhu — 2023, ICML 2023, arXiv:2302.14376
**[fetch başarısız — search bilgisinden]**
**Çekirdek fikir:** GNOT, irregular mesh ve çoklu input function destekleyen bir transformer neural operator. Üç ana bileşeni var: (i) *Heterogeneous Normalized (linear) cross-Attention* — query noktaları ile farklı kaynaklar (geometri, input functions, parametreler) arasında ayrı encoder'lar üzerinden normalize linear attention; (ii) *Geometric Gating MoE FFN* — uzaysal koordinata göre soft-gate edilen mixture-of-experts FFN, çok-ölçekli/çok-bölgeli fiziği modellemek için; (iii) lineer kompleksite sayesinde büyük mesh'lere ölçeklenme. Çoklu PDE benchmark'ında SOTA bildiriliyor.
**Projeye uygulama:** Bu mimarinin doğrudan iskeletimiz: `GNOTBlock` zaten cross-attn (`LinearAttention`, ELU+1 kernel) + self-attn + `GeometricGatingFFN` (`local_gating` koordinattan softmax/T=0.5 ile expert ağırlığı) içeriyor; `mode_field_blocks[k]` bu bloğun per-mode kopyası. Borçlanılacak ek mekanizma: GNOT'un *çoklu input-encoder* tasarımı — şu an tek `input_func_encoder` var; boundary feature'ları ayrı bir encoder'a ayırıp ayrı cross-attention condition stream'i olarak `condition_emb`'e eklemek heterojen koşullamayı güçlendirir.
**Risk/uyarı:** Per-mode K kopya blok GNOT'taki parametre paylaşımını kırar; küçük veride aşırı parametreleşme riski.

### 2. Transolver: A Fast Transformer Solver for PDEs on General Geometries
**Künye:** H. Wu, H. Luo, H. Wang, J. Wang, M. Long — 2024, ICML 2024, arXiv:2402.02366
**[fetch başarısız — search bilgisinden]**
**Çekirdek fikir:** Transolver, mesh noktaları üzerinde doğrudan attention yerine *Physics-Attention* önerir. Her nokta, öğrenilen bir soft-assignment (softmax slice weights) ile az sayıda *slice* (physical-state token, ör. M=32-64) içine dağıtılır; her slice'ın özelliği noktaların ağırlıklı toplamıdır. Standart attention bu küçük slice token kümesi üzerinde uygulanır, sonra slice çıktıları aynı assignment ağırlıkları ile noktalara geri yayılır. Böylece nokta sayısında lineer kompleksite ve geometri-agnostik, fiziksel olarak anlamlı tokenizasyon elde edilir.
**Projeye uygulama:** `GNOTBlock.self_attn` içindeki node-üstü `LinearAttention` yerine Physics-Attention slice katmanı konabilir: koordinat/RFF'ten (`pos_enhanced`) slice logit'leri üretip noktaları M slice'a aggregate et, slice'lar arası dense attention, geri scatter. Bu, `local_gating` ile aynı "koordinattan soft routing" felsefesini paylaşır; slice'lar cavity'nin mode-bölgelerini (anti-node bölgeleri) doğal olarak yakalayabilir. AttentionPool'un yerine de slice-havuzu global context verebilir.
**Risk/uyarı:** Çok az slice eigenfield'in ince uzaysal yapısını (yüksek mode düğümlerini) düzleştirip bulanıklaştırabilir.

### 3. Fourier Neural Operator for Parametric PDEs (FNO)
**Künye:** Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar — 2021, ICLR 2021 (arXiv 2020), arXiv:2010.08895
**[fetch başarısız — search bilgisinden]**
**Çekirdek fikir:** FNO, integral kernel operatörünü Fourier uzayında parametrize eder. Her katmanda girdi FFT ile dönüştürülür, yalnızca ilk birkaç düşük frekans modu tutulur (truncation), bu modlara öğrenilen kompleks ağırlık matrisi çarpılır, inverse FFT ile geri dönülür ve yerel lineer dönüşüm + nonlinearite eklenir. Sonuç: global receptive field, çözünürlükten bağımsız (resolution-invariant) operatör öğrenimi, türbülans/akış PDE'lerinde güçlü.
**Projeye uygulama:** Çoğunlukla *baseline/atıf*. Helmholtz/Maxwell eigenmode düşük-frekans dalga yapılarıdır; FNO'nun ruhu mimarimizde zaten `RandomFourierFeatures` ile spektral koordinat kodlaması olarak mevcut. Doğrudan spectral-conv branch eklemek için noktaları regular grid'e resample edip bir FNO branch'inin çıktısını `field_heads[k]` öncesi `x_m`'e füzyon olarak vermek düşünülebilir; pratikte irregular boundary nedeniyle düşük öncelikli.
**Risk/uyarı:** FNO FFT için uniform regular grid varsayar; bizim irregular point-cloud + boundary maskemizle doğrudan uyumsuz (resampling artefaktı).

### 4. DeepONet: Learning Nonlinear Operators via Deep Operator Networks
**Künye:** L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis — 2021, Nature Machine Intelligence (arXiv 2019), arXiv:1910.03193
**[fetch başarısız — search bilgisinden]**
**Çekirdek fikir:** DeepONet, operatör için universal approximation teoremine dayanır. İki alt-ağ: *branch net* input function'ı sabit sensör noktalarındaki değerlerinden kodlayıp katsayı vektörü üretir; *trunk net* sorgu koordinatını basis fonksiyon vektörüne kodlar. Çıktı bu ikisinin nokta-çarpımıdır (Φ basis × α coefficients), opsiyonel bias ile. Stacked (çoklu bağımsız branch) ve unstacked (tek branch) varyantları var.
**Projeye uygulama:** Bu proje DeepONet faktörizasyonunu *bilinçli olarak kaldırdı* (kod yorumu: "No DeepONet factorization, no Φ basis / α coefficients"); `field_heads[k]` artık `[B,N,D]→[B,N,1]` doğrudan pointwise regresyon. Yani DeepONet burada *karşılaştırma/atıf* rolünde. Yeniden borçlanılırsa: trunk = `query_encoder`(RFF), branch = `pooler` global context; mode başına α_k katsayıları üretip ortak Φ basis ile dot-product, parametre-verimli ve degenerate-mode'larda daha kararlı bir alternatif olur.
**Risk/uyarı:** Sabit Φ basis sayısı yüksek eigenmode'ların alan detayını kısıtlar; mevcut doğrudan head daha esnek ama parametrece pahalı.
