# 02 — Multi-Eigenmode Tahmini, Ortogonalite, EM Eigenproblem

> K=3 modun ayrı ve ortogonal kalması (mode collapse / duplicate-mode önleme),
> EM eigenvalue probleminin doğru kurulması. İlgili kod: `_compute_loss`,
> `loss_ortho`, `field_heads`, `mode_field_blocks` (shared encoder + per-mode).
>
> ⚠️ Bu ortamda web egress kapalı (403); notlar yerleşik bilgiden yazıldı.

### 1. Physics-Informed Neural Networks for Quantum Eigenvalue Problems
**Künye:** Henry Jin, Marios Mattheakis, Pavlos Protopapas, 2022, IJCNN 2022 / arXiv 2203.00451 **[fetch başarısız — search bilgisinden]**
**Çekirdek fikir:** Schrödinger özdeğer problemini PINN ile çözer; özdeğer E, ağ girişindeki bir affine (trainable) katman olarak öğrenilir, yani solver hem özfonksiyonu hem skaler özdeğeri eşzamanlı optimize eder. Trivial sıfır çözümünü engellemek için norm-loss (normalizasyon) ve farklı uyarılmış durumları (excited states) ayrık tutmak için ortho-loss kullanılır; bir scanning mekanizması keyfi sayıda ve dejenere durumu ortogonalize edilmiş halde bulur.
**Projeye uygulama:** Bu, projemizin merkezindeki sorunun doğrudan kaynağıdır. `_compute_loss` içinde zaten `loss_ortho` (W-ağırlıklı slot-collapse guard) ve `soft_procrustes_loss` var; makalenin norm-loss'u eksik — `field_heads[k]` çıktısı için maskeli L2 normunu sabit referansa çeken bir auxiliary terim eklenebilir, böylece slotlar büyüklük olarak da çökmez. Scanning fikri ileride K'yı dinamikleştirmek için decoder döngüsüne uyarlanabilir.
**Risk/uyarı:** Onların ortho-loss'u tek geometride global; bizde per-sample OT eşlemesi sonrası uygulanmalı, yoksa dejenere blokları cezalandırır.

### 2. SpectralNet: Spectral Clustering using Deep Neural Networks
**Künye:** Uri Shaham, Kelly Stanton, Henri Li, Boaz Nadler, Ronen Basri, Yuval Kluger, 2018, ICLR 2018 / arXiv 1801.01587 **[fetch başarısız — search bilgisinden]**
**Çekirdek fikir:** Graf Laplacian'ının özuzayına bir gömme öğrenen derin ağ; ölçeklenebilirlik için stochastic optimization, çıktının ortonormalliğini garanti etmek için özel bir output katmanı kullanır — minibatch üzerinde Cholesky/QR ile çıktıları ortogonalize eden (geri-yayılabilir) bir lineer dönüşüm. Bu, öğrenilmiş spektral gömmenin görülmemiş noktalara genelleşmesini sağlar (out-of-sample extension).
**Projeye uygulama:** Cholesky/QR ortogonalizasyon katmanı bizim için en somut transferdir. `src/training/lightning_module.py`'deki `_masked_orthonormalize` (QR) zaten kayıp tarafında var; aynı işlemi modele, `field_heads[k]` çıktılarından sonra **differentiable orthonormalization layer** olarak gömmek mode collapse'i mimari düzeyde (cezayla değil yapıyla) engeller — `GNOTModel`'de field birleştirildikten sonra maskeli QR uygulanır.
**Risk/uyarı:** Maskeli QR per-sample değişken N ile batched yapıldığında pahalı/kararsız olabilir; dejenere kolonlarda ridge gerekir.

### 3. Laplacian Eigenfunction-Based Neural Operator (LE-NO)
**Künye:** Jindong Wang, Wenrui Hao, 2025, arXiv 2502.05571 (v2 Eylül 2025) **[fetch başarısız — search bilgisinden]**
**Çekirdek fikir:** Reaction-diffusion denklemlerinde sağ taraftaki nonlineer operatörü, Laplacian özfonksiyonlarını sabit bir spektral baz olarak kullanarak öğrenir. Laplacian matrisinin doğrudan tersi alınarak hesaplama karmaşıklığı düşürülür, az veriyle ve küçük ağla çalışılabilir; öğrenilen dinamik yorumlanabilir ve farklı sınır koşullarına genelleşir.
**Projeye uygulama:** Saf node-uzayında field regresyonu yerine, geometriye özel Laplacian özfonksiyon bazına projeksiyon yaklaşımı uygulanabilir: `field_heads[k]` doğrudan node değerleri yerine spektral katsayılar üretir, alan bazda yeniden kurulur. Bu, PEC sınır koşulunu bazın kendisine gömerek `loss_bnd`'yi gereksiz kılar ve modlar farklı baz vektörlerine düşeceğinden mode collapse'i doğal olarak azaltır.
**Risk/uyarı:** Bizde geometri örnekten örneğe değişiyor; her geometri için Laplacian özbazını önceden hesaplamak pahalı ve operator-learning hızını öldürebilir.

### 4. FieldTNN-based ML method for Maxwell Eigenvalue Problems
**Künye:** Jiantao Jiang, Yanli Wang, Yifan Wang, Hehu Xie, 2024, J. Comput. Phys. (2025) / arXiv 2411.15828 **[fetch başarısız — search bilgisinden]**
**Çekirdek fikir:** Tensor Neural Network (TNN) ansatzını Maxwell özdeğer problemine genişletir; tensör olmayan (non-tensor) hesaplama bölgelerini de ele alır. Kritik katkı: divergence-free koşulunu optimizasyon hedefine dahil ederek spurious (sahte, divergence taşıyan) özçiftleri otomatik filtreler — Maxwell özproblemlerinin klasik belası.
**Projeye uygulama:** Bizde tahmin edilen `field`'in fiziksel geçerliliği denetlenmiyor; `_compute_loss`'a, tahmin edilen mod alanının ayrıklaştırılmış divergence'ının (∇·E) maskeli L2 normu şeklinde bir **divergence-free auxiliary term** eklenebilir (loss_bnd'nin yanında, smoothness_weight benzeri bir ağırlıkla). Bu hem field kalitesini artırır hem de spurious/duplicate modları doğal olarak bastırarak `loss_ortho`'yu destekler.
**Risk/uyarı:** Düzensiz FEM mesh'inde ∇· için güvenilir diferansiyel operatör gerekir; gürültülü tahmin çözümü kararsızlaştırabilir.

### 5. Predicting band structures for 2D Photonic Crystals via Deep Learning
**Künye:** Yueqi Wang, Richard Craster, Guanglian Li, 2024, arXiv 2411.06063 (math.NA) **[fetch başarısız — search bilgisinden]**
**Çekirdek fikir:** Brillouin bölgesi boyunca dispersiyon ilişkilerini U-Net ile supervised olarak öngörür; transfer learning ve super-resolution ile düşük çözünürlüklü veriden yüksek çözünürlüklü band yapısı üretir, ince mesh ihtiyacını kaldırır. U-Net'in tek geçişte birden çok band fonksiyonunu eşzamanlı tahmin etmesi, her bandı bağımsız ele alan yöntemlere göre doğruluğu artırır.
**Projeye uygulama:** "Tek shared gövde + eşzamanlı çok-band çıkışı" tezi bizim shared encoder + `mode_field_blocks[k]` tasarımımızı doğrudan destekler ve modlar arası tutarlılığın neden collapse'i azalttığına dair gerekçe verir. Super-resolution/transfer learning şeması, kaba mesh FEM verisinde önce eğitip ince mesh'e fine-tune ederek veri maliyetini düşürmek için pipeline'a uygulanabilir; ayrıca ileride Q-prediction için ek band-benzeri head ısıtmaya örnek.
**Risk/uyarı:** U-Net düzenli grid varsayar; bizim düzensiz FEM node bulutuna doğrudan taşınamaz, yalnızca multi-head/transfer stratejisi aktarılabilir.
