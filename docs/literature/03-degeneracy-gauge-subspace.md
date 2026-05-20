# 03 — Degeneracy / O(d) Gauge / Subspace Loss + Stabil SVD

> Projenin en özgün teknik katmanı. Dejenere eigenvalue'larda eigenvektörler
> yalnızca eigenuzay projeksiyonuna göre (`P=QQᵀ`) tanımlı — bireysel vektörler
> O(d) gauge serbest. İlgili kod: `soft_procrustes_loss`, `grassmannian_loss`,
> `ot_match` (`src/training/lightning_module.py`).
>
> ⚠️ Web egress kapalı (403); notlar yerleşik/arama bilgisinden yazıldı.

### 1. Sign and Basis Invariant Networks (SignNet / BasisNet)
**Künye:** Derek Lim, Joshua Robinson, Lingxiao Zhao, Tess Smidt, Suvrit Sra, Haggai Maron, Stefanie Jegelka; 2022/2023, ICLR 2023; arXiv 2202.13013
**Çekirdek fikir:** Eigenvektörlerin iki simetrisini mimari düzeyde garanti eder: işaret simetrisi (v ve −v) ve dejenere özuzaylardaki O(d) baz simetrisi. SignNet `ρ(Σᵢ[φ(vᵢ)+φ(−vᵢ)])` ile işaret-değişmez; BasisNet ise her özuzayı projeksiyon matrisi `VₗVₗᵀ` üzerinden işleyerek (IGN ile) tam O(d) baz değişmezliği sağlar. Süreklilik koşulları altında universal olduğu kanıtlanır; çoklu (multiplicity) özuzaylar projektör formuyla doğal olarak ele alınır.
**Projeye uygulama:** Kayıp tarafına değil ağ tarafına öneri: GNOT field head'in çıktısını doğrudan eigenvektör olarak regrese etmek yerine, hedefi BasisNet-tarzı projektör başlığıyla `P = QQᵀ` olarak temsil etmek. Bu, `ot_match` içindeki sign-agnostic maliyeti ve `soft_procrustes_loss` içindeki detached SVD rotasyonunu gereksizleştirir çünkü O(d) gauge mimaride çözülür. `grassmannian_loss` zaten `P=QQᵀ` mesafesini kullanıyor; BasisNet bunu tahmin tarafına taşıyarak dejenere blokta tutarlı kılar. SignNet'in `φ(v)+φ(−v)` simetrizasyonu singleton slotlarda ± belirsizliğini kaynakta kaldırır.
**Risk/uyarı:** BasisNet'in IGN bileşeni hesapça ağır; N büyük node sayılı mesh'lerde projektör N×N olur — düşük-rank/Nyström gerekebilir.

### 2. Deep Learning for Subspace Regression
**Künye:** Vladimir Fanaskov, Vladislav Trifonov, Alexander Rudikov, Ekaterina Muravleva, Ivan Oseledets; 2025; arXiv 2509.23249 ⚠️(ID doğrulanamadı — spot-check et)
**Çekirdek fikir:** Grassmannian üzerinde regresyon problemini biçimsel olarak tanımlar. İki kayıp önerir: L1 = ortogonal projektör farkı `‖P_pred − P_tgt‖²_F` (bizim Grassmann mesafemizle birebir aynı), L2 = projektörü least-squares artığıyla değiştirip Hutchinson stochastic trace estimation kullanarak normal denklemler + randomized NLA araçları ile hesaplama. Teorik katkı: elliptik özproblemler için altuzay haritasının (projektör) eigenvektör haritasından daha pürüzsüz (smoother) olduğunun kanıtı.
**Projeye uygulama:** `grassmannian_loss = n − ‖QₕᵀQₜ‖²_F` formülümüzü gerekçelendirir (L1 formuyla aynı geometri). L2 formülasyonu büyük K'da tehlikeli SVD/QR'ı Hutchinson trace estimator + least-squares artığıyla baypas etmeyi önerir. En önemli teorik katkı: "projector haritası eigenvektör haritasından pürüzsüz" teoremi — `soft_procrustes_loss`'un `P=QQᵀ` üzerinden çalışmasının degrade gradyan bölgelerini neden azalttığının matematiksel gerekçesi.
**Risk/uyarı:** K=3 küçük olduğunda tam projektör hesabı zaten ucuz; Hutchinson avantajı ölçekli K'da (K≥10+) belirginleşir.

### 3. Building Deep Networks on Grassmann Manifolds (GrNet)
**Künye:** Zhiwu Huang, Jiqing Wu, Luc Van Gool; 2018, AAAI 2018; arXiv 1611.05742
**Çekirdek fikir:** Grassmann verisi üzerinde derin ağ katmanları tanımlar: FRMap (full-rank lineer dönüşüm), ReOrth (QR ile yeniden ortonormalleştirme, Stiefel'de kalmak için), ProjMap (özuzayı `P=YYᵀ` projektörüne taşıyıp Öklid forma geçer) ve ProjPooling. Eğitim manifold üstünde SGD ve QR/projektör katmanları için yapılandırılmış matris backpropagation ile yapılır.
**Projeye uygulama:** `_masked_orthonormalize` (reduced QR) tam olarak GrNet'in ReOrth katmanı; `grassmannian_loss`'taki `P=QQᵀ` mesafesi GrNet'in ProjMap'i. Somut güçlendirme: GrNet'in kapalı-form Stiefel gradyanını (ReOrth backprop türevi) `torch.linalg.qr`'ın varsayılan gradyanı yerine kullanmak, near-collinear (dejenere) sütunlarda daha kararlı. ProjPooling fikri: K dejenere slot'unu tek blok-projektöre indirgeyen alternatif bir `loss_ortho` slot-collapse guard'ı.
**Risk/uyarı:** GrNet'in QR backprop'u da R köşegeni sıfıra giderken patlar; near-degenerate'te yine ridge/epsilon gerekir.

### 4. Backpropagation-Friendly Eigendecomposition
**Künye:** Wei Wang, Zheng Dang, Yinlin Hu, Pascal Fua, Mathieu Salzmann; 2019, NeurIPS 2019; arXiv 1906.09023
**Çekirdek fikir:** ED/SVD gradyanı özdeğerler birbirine yaklaştığında `1/(λᵢ − λⱼ)` terimleri yüzünden sayısal olarak patlar. Yazarlar forward'da analitik ED kullanıp, backward'da gradyanı Power Iteration (PI) ile yaklaştırarak hesaplar; PI türevi bu kötü-koşullu farkı içermez, böylece dejenere/yakın-dejenere özdeğerlerde kararlı ve büyük matrislere ölçeklenir (ZCA whitening, PCA denoising uygulamaları).
**Projeye uygulama:** `soft_procrustes_loss` içindeki `U,_,Vh = torch.linalg.svd(C)` şu an `no_grad` altında, dolayısıyla grad akmıyor — güvenli ama Procrustes'i kısmen diferansiyellenebilir yapmak istenirse bu çalışmanın PI-tabanlı backward'ı zorunlu. Ayrıca `grassmannian_loss`'taki QR'ı ED tabanlı projektörle değiştirip backward'ı PI ile stabilize etmek, dejenere σ kümelenmesindeki `1/(σᵢ²−σⱼ²)` patlamasını sınırlar. Dejenere σᵢ≈σⱼ tam senaryomuzun tanımıdır.
**Risk/uyarı:** PI yaklaşımı iterasyon sayısına duyarlı; çok yakın özdeğerlerde deflation gerekebilir.

### 5. Robust Differentiable SVD
**Künye:** Wei Wang, Zheng Dang, Yinlin Hu, Pascal Fua, Mathieu Salzmann; 2021, ICCV 2021; arXiv 2104.03821
**Çekirdek fikir:** [1906.09023]'ün devamı. SVD gradyanındaki `Kᵢⱼ = 1/(σᵢ² − σⱼ²)` terimini, geometrik/Taylor serisi açılımıyla `Σₖ (σⱼ²/σᵢ²)ᵏ` biçiminde değiştirir; K. dereceli Taylor açılımının K+1 power iteration gradyanına teorik olarak eşdeğer olduğunu kanıtlar. Sonuç: iteratif süreç olmadan sınırlı (bounded) gradyan, kontrollü yaklaşım hatası.
**Projeye uygulama:** En somut reçete: `soft_procrustes_loss`'taki SVD'yi gradyanlı hale getirmek istenirse, `torch.linalg.svd`'nin autograd gradyanını bu makalenin §3'teki Taylor-truncated `K̃ᵢⱼ` formülüyle override et (custom `torch.autograd.Function`) — near-degenerate σ kümelenmesinde patlamayı sınırlar. Aynı yama `_masked_orthonormalize`'a alternatif ED tabanlı orthonormalizasyon için de geçerli. Pratikte detached `R` yerine kısmen-diferansiyellenebilir Procrustes denenecekse zorunlu.
**Risk/uyarı:** Taylor kesme derecesi K küçükse tam-dejenere (σᵢ=σⱼ) limitte bias kalır; derece/tolerans ablation ister.

### 6. An Analysis of SVD for Deep Rotation Estimation
**Künye:** Jake Levinson, Carlos Esteves, Kefan Chen, Noah Snavely, Angjoo Kanazawa, Afshin Rostamizadeh, Ameesh Makadia; 2020, NeurIPS 2020; arXiv 2006.14616
**Çekirdek fikir:** Ağ çıktısı M'yi rotasyona projeksiyon için simetrik orthogonalization (SVD) analiz eder. `SVD⁺(M)`: `M=UΣVᵀ` → `R = U·diag(1,…,1,det(UVᵀ))·Vᵀ`, son tekil yönün işaretini çevirerek `det=+1` zorlar (SO(d), salt O(d) değil). Gram-Schmidt ve quaternion temsillerine karşı SVD orthogonalization'ın gradyanının daha iyi koşullu ve yan etkisiz olduğunu gösterir.
**Projeye uygulama:** Doğrudan `soft_procrustes_loss`'taki `R = Vh.T @ U.T` satırını ilgilendirir. Şu an determinant düzeltmesi YOK — `R ∈ O(K)`, yansımalar (det=−1) serbest. Eigenvektör gauge'i gerçekten O(d) olduğundan (yansıma fiziksel olarak geçerli baz), bizim O(K) Procrustes'imiz doğrudur ve `det=+1` zorlanmamalıdır. Bu makale tam tersine SO(d) ister — bizim `diag(1,…,det)` düzeltmesini eklememiz HATA olur. Öte yandan makalenin SVD backward σ-fark patlaması uyarısı ve bunun paper 4-5 ile bağlantısı önemli referans.
**Risk/uyarı:** SO(d)↔O(d) ayrımı kritik: makaledeki determinant-flip reçetesini körlemesine almak dejenere eigenuzay yansıma simetrisini kırar — **uygulamadan kaçın**.
