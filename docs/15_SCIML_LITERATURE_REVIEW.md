# 15 — SciML Literatür Taraması: Nöral Özçözücüler, Geometri-Farkında Operatörler, Simetri ve Teori

> **Kapsam:** Araştırma notu. Model/eğitim kodunda değişiklik yok, repo eğitim koşusu yok. Yalnız küçük CPU deneyleri yapıldı (§5).
> **Soru:** Ω ↦ {(λ_k, u_k)}_{k=1}^{3} (Dirichlet Laplace, P2 FEM etiketleri, f = c√λ/2π) problemini 2026 itibarıyla literatür nasıl çözüyor? [[14_MATHEMATICAL_IMPROVEMENTS]]'teki öneriler literatürle ne kadar uyumlu?
> **Referans commit:** `fd2d27c`. İncelenen dosyalar: `src/models/gnot.py`, `src/models/spectral_no.py`, `src/training/lightning_module.py`, `src/data/dataset_converter.py`, `src/data/dataset.py`, `configs/*.yaml`.
> **Deney scriptleri:** `scratchpad/math_sciml/` (repo'ya dahil değil): `toy_data.py`, `toy_train.py`, `e_energy.py`, `e_p1floor.py`.

---

## 🧭 Özet

**Beş ana bulgu:**

1. **Literatürün yönü nettir: tekil özvektör değil, *alt uzay* öğren; özçiftleri gerçek operatörle *Rayleigh–Ritz (RR)* ile geri kazan.** Bu tasarımı en güncel üç çalışma bağımsız olarak kullanıyor: NEO (Yang, Du & Liu, SIGGRAPH 2026; nokta bulutlarında Laplace–Beltrami), Deep Eigenspace Network (Li, Sun & Zhang 2025; parametrik öz-eşlenik olmayan problemler) ve NN tabanlı alt uzay yöntemi (Dai, Fan & Sheng 2024). NEO gerekli olandan *fazla* (redundant) baz fonksiyonu tahmin ediyor, bunları kütle matrisiyle ortonormalize ediyor ve gerçek matrisle küçük bir özdeğer problemi çözüyor. Bizim oyuncak deneyimizde (§5.2) bu şema doğrudan özvektör regresyonuna göre $P_{\le3}$ alt uzay hatasını **~6–7×**, yakın-dejenere ikinci mod hatasını **~400×** düşürdü.
2. **$L^2$ alan hatası özdeğer hatasını kontrol etmez.** Repo'nun kendi meshlerinde (§5.1) *aynı* $L^2$ alt uzay hatasında ($\sin^2\theta=10^{-3}$) RR özdeğer hatası pürüzsüz hata için **6·10⁻⁴**, duvar bandındaki düğüm gürültüsü için **0.93**. Aradaki fark üç mertebedir. Bu, Babuška–Osborn'un "özdeğer hatası ≈ enerji normunda alan hatasının karesi" sonucunun doğrudan karşılığıdır. Doc 14'ün $L^2$-flag kaybı bu yüzden gerekli ama yeterli değildir.
3. **Bir blok ters iterasyon (mesh'in kendi matrisleriyle), yüksek frekanslı NN hatasını neredeyse tamamen siler** (0.93 → 3.7·10⁻⁶). Pürüzsüz hatayı ise yalnız λ_k/λ_{K+1} oranında azaltır. Doğru iş bölümü şudur: ağ düşük frekanslı alt uzayı öğrenir, "cilayı" ucuz lineer cebir yapar. Doc 14'ün 5. önerisi (hibrit) böylece hem literatürle (NEO, NOWS) hem deneyle güçlü biçimde destekleniyor.
4. **Geometri kodlaması:** Mesh-noktası operatörleri (GNOT, Transolver/Transolver++, GINO, GAOT, RIGNO, PCNO) standart geometri-değişken benchmark'larda düşük boyutlu parametrizasyon *gerektirmeden* %0.5–2 rel-L2 hatasına ulaşıyor. Teori ise tarafını açıkça seçiyor. Şekle *holomorf* bağlı haritalar için boyuttan bağımsız yaklaşım oranları var (Harbrecht & Schwab 2026, afin-parametrik şekil kodlaması; Schwab & Zech 2019). Yalnız Lipschitz olan genel operatör sınıflarında ise "parametrik karmaşıklık laneti" ve "veri karmaşıklığı laneti" geçerli (Lanthaler & Stuart 2023; Kovachki, Lanthaler & Mhaskar 2024). Basit özdeğerler ve *ayrık kümelerin* projektörleri holomorftur. Sıralı özdeğerler kesişimlerde yalnız Lipschitz'tir. Pratik sonuç: global, düşük boyutlu bir şekil kodu (r(θ) Fourier katsayıları) mesh girdisinin **yanına** eklenmeli ve hedefler holomorf nesneler (log-λ, küme projektörleri) seçilmelidir.
5. **Simetri:** SignNet/BasisNet sınıfı yöntemler özvektörleri *girdi* olarak kullanan ağlar içindir. Bizim problemde özvektörler *çıktıdır* ve doğru araç alt uzay kayıplarıdır. Repo'daki PCA-tabanlı `cos/sin_principal` özellikleri sezgisel bir kanonikleştirmedir. Neredeyse izotrop şekillerde kararsızdır; bu, Szwagier & Pennec'in "izotropi laneti"nin aynısıdır. Önerimiz frame averaging (Puny et al. 2022) ya da bu kolonları hiç kullanmamak (dönme augmentasyonu zaten var).

### Öncelik sıralı, repo'ya özgü öneriler

| # | Öneri | Etki | Efor | Dosyalar | Dayanak |
|---|---|---|---|---|---|
| **1** | **"Redundant alt uzay + gerçek-matrisle RR" çıktı katmanı** (her iki model için). Ağ m = 2K = 6 nodal alan üretir. Batch'in kendi mesh'inden P1 eleman rijitlik/kütle formları ile $\Phi^\top K_h\Phi$, $\Phi^\top M_h\Phi$ kurulur, ardından `eigh`. Kayıp: $K-\|Q^\top M_h U_K\|_F^2$ ($Q$: $M_h$-ortonormal span). OT eşleştirme, `detect_clusters`, soft-Procrustes ve monoton frekans başlığı gereksizleşir. | **Çok yüksek.** Oyuncak (2 seed): $P_{\le3}$ hatası 3.5–4.0e-3 → 5.5–6.2e-4; yakın-dejenere mod-2 0.20–0.29 → 4.8e-4. 1 ters iterasyon sonrası λ: 5.5–6.2e-5 / 5.4–6.6e-4 / 1.1e-3 → 0.9–1.0e-5 / 5.4–6.4e-5 / 1.8–1.9e-4. | Orta | `spectral_no.py` (autograd-∇ψ + düğüm quadratürü yerine eleman-bazlı P1 montaj), `gnot.py` (3 slot yerine m slot), `lightning_module.py` (yeni `subspace_capture_loss`, `ritz_pairs`), `dataset.py`/`gnot_collate_fn` (`elements`, eleman alanları/gradyanları) | NEO 2026, DEN 2025, Dai+ 2024, Knyazev (blok boyu > K); §5.2 |
| **2** | **Özdeğerleri L²-eğitilmiş span'den doğrudan okuma.** Validasyon/çıkarımda 1 blok ters iterasyon + RR (mümkünse **P2** matrisleriyle); frekans $f=c\sqrt{\lambda}/(2\pi s)$. Eğitimde ya enerji-farkında bir terim (Ritz değeri) ya da ters iterasyonlu hedef. | **Çok yüksek.** Aynı L² hatasında λ hatası 6e-4 ↔ 0.93; ters iterasyon sonrası ≤2.5e-4 (tüm hata tipleri). | Düşük–Orta | `infer.py` (`--refine`), `lightning_module.py` (val metriği), `spectral_no.py` (`_MonotoneFreqHead` kaldır) | Babuška–Osborn; NEO; §5.1 |
| **3** | **Kütle-farkında (quadratür-ağırlıklı) attention/pooling (GNOT).** `LinearAttention` şu an $\sum_n k_n v_n^\top$ biçiminde **ağırlıksız** düğüm toplamı yapıyor. Doğru integral operatör yaklaşımı $\sum_n w_n k_n v_n^\top$ ve normalizasyon $\sum_n w_n k_n$'dir. `AttentionPool`'a da $\log w_n$ bias'ı eklenmeli. | Orta–Yüksek: mesh-yoğunluğu bağımlılığını kaldırır. Graded mesh'te sınır bandı ~5× fazla ağırlıklanıyor (doc 14 E9). | Düşük | `gnot.py` (`LinearAttention`, `AttentionPool`), `dataset.py` (w = kolon 5) | NEO (mass-aware attention), Cao 2021 (Galerkin attention), GINO/PCNO (discretization invariance) |
| **4** | **Doğrudan regresyon kalacaksa** (matrissiz çıkarım): per-mode işaret-invaryant kayıp yerine gap-ağırlıklı **flag kaybı**, $M_h$-ağırlıklı iç çarpımlar, ≥5 mod sakla. Sert eşikli küme kuralını kaldır. | Orta: oyuncakta $P_{\le3}$ 3.5–4.0e-3 → 2.4–2.6e-3, λ(1 inv. it.) ~%35 daha iyi. Ancak Öneri 1'in çok gerisinde. | Düşük–Orta | `lightning_module.py`, `dataset_generator.py` (`n_eigen_modes=5`) | Szwagier & Pennec 2025 (flag), doc 14 §1.3; §5.2 |
| **5** | **Global düşük boyutlu şekil kodu:** r(θ) Fourier katsayıları (K=16–32, doc 14 E4) ya da sınır-SDF örnekleri, GNOT'a global token olarak, SpectralNO'ya ise ψ'nin $x$'ten bağımsız koşulu olarak verilsin. Uzun vadede tüm şekiller için ortak referans mesh (polar harita / mesh morfolama). | Yüksek (veri verimliliği, teori). Doğrudan kıyas benchmark'ı yok (§4.3). | Orta (token) / Yüksek (referans domain) | `dataset_converter.py`, `gnot.py`, `spectral_no.py`, `data_gen/*` | DIMON 2024, Geo-FNO, Harbrecht & Schwab 2026, Weder+ 2024 |
| **6** | **Dönme simetrisi:** `cos/sin_principal`'ı (kolon 6–7) ya kaldır ya da 4 PCA çerçevesi üzerinde frame averaging uygula. Augmentasyonu tam O(2) (yansıma dahil) yap. | Orta | Düşük | `dataset_converter.py`, `dataset.py` | Puny+ 2022, Kaba+ 2023, Ma+ 2023, Szwagier & Pennec 2024 |
| **7** | **P2 sorgu noktaları:** GNOT sorgu-tabanlı olduğundan kenar-ortası noktalarında da değerlendirilebilir. Böylece gerçek P2 alanı üretilir ve RR, P1 tabanının (+%0.25–0.67) altına inebilir. | Düşük–Orta (ters iterasyon zaten telafi ediyor) | Orta | `dataset_converter.py`, `gnot.py` | §5.3 |
| **8** | **Etiketsiz Ky Fan / NeuralSVD-tipi yardımcı eğitim:** Öneri 1'in altyapısıyla etiketsiz şekillerde $\sum_{k\le K}\hat\lambda_k^{RR}$ minimize edilir. | Düşük–Orta: oyuncakta sıfır etiketle $P_{\le3}$ 8.9e-3, λ(RR) %2.8/%2.1/%3.2. Etiket maliyeti zaten gmsh'e bağlı. | Düşük (1'den sonra) | `lightning_module.py` | Pfau+ 2019, Ryu+ 2024, E & Yu 2018; §5.2 |

**Stratejik not (doc 14 ile uyumlu, literatürle teyitli):** NEO'nun 88× hızlanması 512k noktada ölçülüyor. Bizim ~1.5k düğümlük meshlerde P2 montaj + `eigsh` medyan **0.19 s**, P1 **0.08 s** (§5.3). Saf bir NN bu rejimde FEM'i doğrulukta da hızda da anlamlı biçimde geçemez. Repo'nun katma değeri toplu (batched) GPU değerlendirmesi, türevlenebilir tasarım döngüsü ve FEM için sıcak başlangıçtır. Öneri 1–2 tam da bu kullanımı hedefler.

---

## 1. Problemin literatürdeki yeri

Repo'nun problemi üç literatürün kesişiminde durur:

- **Nöral özçözücüler** tek bir operatörün özfonksiyonlarını öğrenir: SpIN, NeuralEF, NeuralSVD, Deep Ritz, PINN-eigen.
- **Geometri-farkında nöral operatörler** şekil → alan haritası öğrenir: Geo-FNO, GINO, GNOT, Transolver, DIMON.
- **Parametrik özdeğer / indirgenmiş baz teorisi** şekle bağlı spektral veri için düzenlilik ve yaklaşım oranlarını inceler: holomorfi, RB, POD-NN.

İlk grup operatörü *bilir*, genelleme yapmaz. İkinci grup genelleme yapar ama çıktının tekil olmadığını (işaret/dönme belirsizliği, mod kesişimi) genelde görmezden gelir. 2024–2026'da bu iki grubu birleştiren küçük ama net bir çizgi oluştu: Shape Space Spectra, Operator Inference for Elliptic EVP, DEN, NEO. Bu çizginin ortak kararı şudur: **çıktı, alt uzay (invariant subspace) olmalı; özçiftler gerçek operatörle RR ile çıkarılmalı.**

---

## 2. Literatür haritası

### 2.1 Tablo

| Yöntem | Yıl | Ne öğrenir | Geometri | Dejenerelik / işaret | Buradaki önemi |
|---|---|---|---|---|---|
| **SpIN** (Pfau et al.) | 2019 | Tek operatörün ilk k özfonksiyonu (NN), stokastik; bilevel optimizasyon | Sabit (tek operatör) | Ky Fan izi + Cholesky "maskeli gradyan" ile sıralı özvektör | Ky Fan / etiketsiz kayıp fikri (Öneri 8) |
| **NeuralEF** (Deng, Shi & Zhu) | 2022 | Kernel integral operatörünün özfonksiyonları; EigenGame'in fonksiyon uzayına genellemesi | Sabit | Sıralı, asimetrik (hiyerarşik) amaçla | Sıralı özfonksiyon kaybı; dejenerede kararsız |
| **Neural eigenfunctions → representation** (Deng et al.) | 2025 | Aynı çerçeve, temsil öğrenmede | — | "Simetri kırma" ile önem sırası | Kavramsal |
| **NeuralSVD** (Ryu et al.) | 2024 | Operatör SVD/EVD; iç içe (nested) düşük-rank amaç | Sabit | Nesting ile sıralı; ortogonallik örtük | Flag kaybının NN karşılığı; **nested = flag** |
| **Deep Ritz** (E & Yu) | 2018 | Varyasyonel PDE ve özdeğer (Rayleigh bölümü) | Sabit | Tek mod / deflasyon | Rayleigh–Ritz enerjisi = doğru kayıp ölçüsü |
| **DMC-benzeri** (Han, Lu & Zhou) | 2020 | Yüksek boyutlu özdeğer, Feynman–Kac sabit nokta | Sabit, yüksek boyut | Temel durum | Düşük önem (2D'de FEM üstün) |
| **Lu & Lu, a priori** | 2022 | 2 katmanlı NN, Schrödinger temel durumu | Hiperküp | Temel durum (Krein–Rutman) | Boyuttan bağımsız genelleme hatası (spektral Barron) |
| **Rayleigh bölümü + Gram–Schmidt** (Rowan et al.) | 2025 | Mühendislik özdeğerleri, NN ayrıklaştırma | Sabit | GS ile sıralı | Basit baseline |
| **PINN eigen** (Bonder & Salort; Banderwaar & Gupta) | 2025 | Eliptik özdeğerler; bikonveks yeniden formülasyon | Sabit | Tek tek / keyfi indeks | Düşük: tek geometri |
| **STNet** (Wang et al.) | 2025 | Deflasyon + spektral filtre dönüşümü | Sabit | Deflasyon projeksiyonu | Filtre fikri; çok-şekil yok |
| **NN alt uzay yöntemi** (Dai, Fan & Sheng) | 2024 | NN'den ortogonal baz → **Galerkin projeksiyonu** + küçük özproblem | Sabit | RR ile | SpectralNO'nun ruh ikizi (tek geometri) |
| **Shape Space Spectra** (Chang et al.) | 2025 | Sürekli parametreli şekil ailesi üzerinde özfonksiyon neural field'ları | İçerde/dışarıda göstergesi; şekil parametresi girdi | Nested varyasyonel ilke; **kesişimde özdeğere göre dinamik yeniden sıralama**, "causal gradient filtering" | Doğrudan analog; düşük boyutlu şekil parametresi |
| **Operator Inference for Elliptic EVP** (Li, Sun & Zhang) | 2025 | Piksel domain → ilk 20 Dirichlet λ (CNN) + özfonksiyonlar (FNO) | Piksel görüntü; **ölçek + ana-eksen hizalama** ön işlemesi | Ana eksen hizalamayla kanonikleştirme | Neredeyse aynı problem; kanonikleştirme dersleri |
| **DEN** (Li, Sun & Zhang) | 2025 | Parametrik öz-eşlenik olmayan (Steklov) problemde **özuzay haritası** | Yapısız mesh; geometri-adaptif POD bazları + FNO | Tekil fonksiyon yerine invariant alt uzay; alt uzayın parametreye göre **Lipschitz**liği kanıtlı | Öneri 1'in teorik dayanağı |
| **NEO** (Yang, Du & Liu) | 2026 | Nokta bulutu → düşük frekanslı LBO **özuzayı**; redundant baz + $M$-ortonormalizasyon + **RR** | Nokta bulutu, mass-aware attention | İşaret/dönme belirsizliği alt uzay hedefiyle tamamen ortadan kalkar | Öneri 1 ve 3'ün doğrudan şablonu |
| **WE-FNO metamalzeme** (Zhang et al.) | 2026 | Keyfi metamalzeme geometrilerinde çoklu elastik özmod | Dalgacık kodlaması + FNO | "Tek girdi–çıktı eşlemesi" varsayımının özproblemde bozulduğunu vurgular | Aynı zorluk, farklı fizik |
| **Can a NN hear the shape of a drum?** (Zhao & Fogler) | 2022 | Ters problem: 100 λ → poligon | Görüntü | — | Ağ ölçek yasasını ve dönme serbestliğini kendiliğinden öğreniyor; latent ≈ Weyl parametreleri |
| **Learning orthonormal bases** (Kamkari, Nabizadeh & Solomon) | 2026 | Fonksiyon uzayında ortonormal baz, Lie-grubu ODE'si | — | Ortogonallik yapısal | Uzun vade: yapısal ortonormal çıktı |
| **FEENet** (Li & Salahshoor) | 2026 | Geometriye içkin FEM özfonksiyon bazında PDE katsayıları | FEM özbazı (tek sefer) | — | "Özbaz = geometri kodu" fikri |
| **Galerkin POD-NN** (Weder, Kast, Henríquez & Hesthaven); **POD-NN** (Hesthaven & Ubbiali) | 2024 / 2018 | Parametrik domainlerde (afin-parametrik dönüşüm) Helmholtz/Maxwell RB katsayıları | Referans domain'e çekme | — | Referans domain + düşük boyutlu parametre |
| **Geo-FNO** (Li et al.) | 2023 | Öğrenilmiş deformasyonla latent düzgün ızgarada FNO | Nokta bulutu / mesh / tasarım parametresi | — | Ortak referans domain fikri |
| **GINO** (Li, Kovachki et al.) | 2023 | SDF + GNO ↔ latent FNO | SDF, discretization-invariant | — | Quadratür-tutarlı GNO |
| **GNOT** (Hao et al.) | 2023 | Çok girdi fonksiyonlu lineer-attention transformer, geometric gating | Mesh noktaları + ek girdiler | — | Repo'nun omurgası |
| **Galerkin Transformer** (Cao) | 2021 | Softmax'sız attention ≈ Petrov–Galerkin projeksiyonu | — | — | Attention'ın ağırlıklı integral olarak yorumu (Öneri 3) |
| **Transolver / Transolver++** (Wu et al.; Luo et al.) | 2024 / 2025 | "Physics-attention": noktaları öğrenilmiş dilimlere topla | Mesh noktaları, milyon ölçek | — | Dilim-attention GNOT'a alternatif |
| **DIMON** (Yin et al.) | 2024 | Difeomorfik harita ile referans domain'e taşı, orada öğren | Referans domain + difeomorfizm | — | Yıldız-şekiller için polar harita doğal |
| **PCNO** (Zeng et al.) | 2025 | Nokta bulutu nöral operatörü, permütasyon invaryant, lineer karmaşıklık | Adaptif meshler, topoloji değişimi | — | Quadratür ağırlıklı integral katmanlar |
| **GAOT** (Wen et al.); **RIGNO** (Mousavi et al.) | 2025 | Çok ölçekli attentional GNO + ViT; bölgesel mesh GNN | Keyfi nokta bulutu | — | 28 benchmark, güçlü baseline'lar |
| **CGA-DL-ROM** (Brivio, Fresca & Manzoni) | 2024 | Geometri parametrizasyonu-farkında DL-ROM | Parametrik domain | — | Düşük boyutlu kodun endüktif önyargısı |
| **Forgetting hypothesis** (Xia & Aviles-Rivero) | 2026 | Derin operatörler katman katman geometriyi "unutuyor"; ara katmanlara geometri enjeksiyonu | — | — | GNOT'un blok-başı `global_context`'i doğru yönde; SpectralNO'da global kod yok |
| **SignNet/BasisNet** (Lim et al.) | 2023 | Özvektör *girdileri* için işaret/baz invaryant ağlar | Graf/mesh | O(d_i) invaryans | Çıktı tarafında gerekmez (bkz. §3) |
| **Sign-equivariant nets** (Lim, Robinson, Jegelka & Maron) | 2023 | İşaret-*eşdeğer* ağlar | — | İnvaryansın yetersiz kaldığı durumlar | Kavramsal |
| **Laplacian Canonization / MAP** (Ma, Wang & Wang) | 2023 | Özvektörleri kanonik yöne döndür | — | Özvektörlerin >%90'ı kanonikleşiyor; kalanı (izotrop) kanonikleşmiyor | PCA-özelliği uyarısı |
| **Frame averaging** (Puny et al.); **learned canonicalization** (Kaba et al.) | 2022 / 2023 | Çerçeve üzerinden ortalama / öğrenilmiş kanonikleştirme | — | Tam invaryans/eşdeğerlik | Öneri 6 |
| **E(2)-steerable CNN** (Weiler & Cesa); **G-FNO** (Helwig et al.) | 2019 / 2023 | Yapısal E(2)/dönme eşdeğer katmanlar | Izgara | — | Referans ızgara kullanılırsa |
| **Curse of isotropy / nested flags** (Szwagier & Pennec) | 2024 / 2025 | Yakın özdeğerli bileşenler yerine "principal subspace"; Grassmann → flag manifoldu | — | Boşluğa göre gruplama, iç içe projektörler | Doc 14 flag kaybının istatistik literatüründeki karşılığı |
| **Parametrik karmaşıklık / veri karmaşıklığı** (Lanthaler & Stuart; Kovachki, Lanthaler & Mhaskar) | 2023 / 2024 | Alt sınırlar: genel C^r/Lipschitz operatörler için üstel karmaşıklık | — | — | Yapı (holomorfi) kullanılmadan veri verimliliği yok |
| **Neural shape operator surrogates** (Harbrecht & Schwab) | 2026 | Referans domain'e difeomorf şekil aileleri; afin-parametrik kodlamayla holomorf parametrik PDE; ifade oranı sınırları | Afin-parametrik şekil kodu | — | Düşük boyutlu şekil kodu için teorik dayanak |
| **Parametrik holomorfi – özdeğer** (Bahn; Andreev & Schwab) | 2024 / 2012 | Temel özçiftin parametrik holomorfisi; sparse-tensor yaklaşımı | — | Basit özdeğer | Holomorf hedef seçimi |

### 2.2 Blok 1: Nöral özçözücüler ve özuzay-operatör öğrenimi

**Tek operatör çözücüleri** (SpIN, NeuralEF, NeuralSVD, Deep Ritz, PINN-eigen, STNet, Rowan et al.) üç gradyan-tabanlı mekanizmayla ilk K özfonksiyonu ayırır:

- (a) Ky Fan izi + sıralama için asimetrik/maskelenmiş gradyan (SpIN),
- (b) hiyerarşik ("EigenGame") amaçlar (NeuralEF),
- (c) iç içe (nested) düşük-rank amaçlar (NeuralSVD).

Üçü de özünde **flag** (iç içe alt uzay) optimizasyonudur. Dejenerelikte sıralamanın keyfi olduğunu ve bu durumda yalnız iç içe alt uzayların tanımlı kaldığını kendi analizlerinde kabul ederler. Bu, doc 14 §1'in Davis–Kahan argümanının ML tarafındaki karşılığıdır. Bu yöntemler tek bir operatör içindir. Onlardan alınacak ders kayıp tasarımı, özellikle **Rayleigh–Ritz enerjisini doğrudan minimize etmek**tir.

**Şekil ailesi üzerinde öğrenme:**

- **Shape Space Spectra** (Chang et al. 2025): Sürekli parametreli bir şekil ailesinde Dirichlet enerjisini nested kısıtlarla minimize eden neural field'lar kuruyor. Kesişimlerde modları özdeğere göre **dinamik olarak yeniden sıralıyor**. Yani sıralı-özdeğer konvansiyonunu (doc 14'ün "OT değil sıralama" önerisi) kullanıyor, ama kesişim noktasında tekil vektör hedefi yine süreksiz.
- **Operator Inference** (Li, Sun & Zhang 2025): Tam olarak "domain → Dirichlet λ'lar + özfonksiyonlar" problemini piksel domainlerle çözüyor. Kritik ön işlemeleri domain ölçekleme ve **ana-eksen hizalama**. Bu, repo'nun `scale` ve `cos/sin_principal` seçimleriyle birebir aynı ve aynı PCA-kararsızlığı riskini taşıyor (§3).
- **DEN** (Li, Sun & Zhang 2025): Bir sonraki adım. Özuzay haritasının parametreye göre Lipschitz olduğunu kanıtlıyor ve tekil fonksiyon yerine onu öğreniyor (geometri-adaptif POD bazı + FNO + bantlı çapraz-mod karıştırma).
- **NEO** (Yang, Du & Liu, SIGGRAPH 2026): Aynı fikri mühendislik olarak tamamlıyor. Redundant baz → $M$-ortonormalizasyon → gerçek matrisle RR. Nokta kütlelerini attention'a sokan "mass-aware attention" kullanıyor. Raporladıkları "span loss" 3.35·10⁻³ (ShapeNet), spektral enerji yakalama %99.7. 2k noktada eğitilip 512k+ noktaya sıfır-atış genelleme yapıyor.

**Sonuç:** 2026 SOTA'sı "alt uzay + RR". Repo'nun SpectralNO'su bu fikre en yakın model (Galerkin + eigh). Ancak (i) RR'yi öğrenilmiş quadratürle ve autograd ∇ψ ile yapıyor (doc 14 §3.1'in üç ihlali), (ii) kaybı hâlâ tekil modlar üzerinde kuruyor.

**PINN yöntemleri** tek geometri için tasarlanmış, bizim çok-şekilli rejim için doğrudan uygun değil. **Hibrit/sıcak başlangıç** tarafında NOWS (2025) ve NEO NN'i iteratif çözücünün başlangıcı olarak kullanıyor. DeepContour (Chen et al. 2025) NN ile kontur-integrali özçözücüsünün konturunu tasarlıyor (5.63×'e varan hızlanma).

### 2.3 Blok 2: Geometri-farkında nöral operatörler

**Kıyas düzeyi:** Transolver++ makalesindeki standart benchmark tablosunda Elasticity (nokta bulutu, değişken geometri) rel-L2 değerleri Geo-FNO **0.0229**, GNOT **0.0086**, Transolver **0.0064**. Airfoil'da Transolver **0.0053**. Yani mesh-noktası transformerları geometri-değişken *ileri* problemlerde %0.5–1 hatada. Ancak bu benchmark'ların hiçbiri özproblem değil; çıktı tekil ve süreklidir.

**Geometri kodlama seçenekleri:**

1. *Mesh/nokta tokenları* (GNOT, Transolver, GAOT, RIGNO, PCNO). Discretization-invariance için integral katmanların quadratür ağırlığıyla kurulması gerekir. PCNO ve GINO bunu açıkça yapıyor, NEO'nun mass-aware attention'ı da aynı fikir. Repo'nun GNOT'u bunu **yapmıyor** (Öneri 3).
2. *SDF* (GINO, SDF-tabanlı yüzey modelleri).
3. *Referans domain + difeomorfizm* (Geo-FNO'nun öğrenilmiş deformasyonu, DIMON, Galerkin POD-NN'in afin-parametrik dönüşümü, Harbrecht & Schwab 2026).
4. *Düşük boyutlu parametre* (CGA-DL-ROM, Shape Space Spectra).

Yıldız-şekilli domainlerde (3) ve (4) doğal olarak birleşir: $x=c+\rho\,r(\theta)(\cos\theta,\sin\theta)$ tek bir disk mesh'ini morfolar, r(θ)'nın Fourier katsayıları ise şekil kodudur. §5.2'deki oyuncak veri seti tam olarak böyle kuruldu.

**Geometriyi "unutma":** Xia & Aviles-Rivero (2026) spektral ve attention-tabanlı operatörlerin derinlik arttıkça geometri bilgisini kaybettiğini katman bazlı sondalamayla gösteriyor ve ara katmanlara geometri enjeksiyonu öneriyor. Repo'nun GNOT'u her blokta `AttentionPool(condition_emb)` bağlamını ekliyor, bu doğru yönde. SpectralNO'nun `basis_net`'i ise **tamamen noktasal**: ψ(x) yalnız x'in yerel özelliklerine bağlı, global bir şekil kodu yok (doc 14 §3.3'ün "g" kanalı bu eksikliği zaten işaret ediyordu).

### 2.4 Blok 3: Simetri ve iyi-tanımsızlık

- **İşaret/baz belirsizliği özvektör girdilerinde** (Laplacian positional encoding) SignNet/BasisNet (Lim et al. 2023), sign-equivariant ağlar (Lim et al. 2023) ve Laplacian Canonization (Ma et al. 2023) ile çözülüyor. Bizde özvektörler **çıktı**. Çıktı tarafında belirsizliği modelle değil **kayıpla/hedefle** çözmek gerekir: projektör, flag ya da RR. NEO ve DEN tam bunu yapıyor. Bu yüzden SignNet eklemek önermiyoruz. İstisna: ileride özvektörler başka bir ağa girdi olarak verilirse (örn. FEENet-tipi özbaz kodlaması) SignNet/BasisNet gerekir.
- **Kanonikleştirme riski:** Ma et al. MAP algoritmasının özvektörlerin >%90'ını kanonikleştirebildiğini, kalanın (simetrik/izotrop) kanonikleştirilemediğini raporluyor. Szwagier & Pennec yakın özdeğerli temel bileşenlerin "önemli dönme değişkenliği" taşıdığını gösteriyor. Repo'nun `principal_axis` = sınır noktalarının kovaryansının en büyük özvektörü tam olarak böyle bir bileşen: neredeyse izotrop şekillerde (ki dipol çifti tam bu şekillerde yakın-dejenere) eksen keyfidir. Bu, *girdide* şekle göre süreksizlik üretir. Çözümler: (i) kaldır + augmentasyon (şu an augment=True iken sıfırlanıyor, yani model zaten bunsuz eğitilebiliyor), (ii) 4 PCA çerçevesi (±e₁, ±e₂) üzerinde frame averaging (Puny et al. 2022), (iii) öğrenilmiş kanonikleştirme (Kaba et al. 2023).
- **Grassmann ve flag kayıpları:** Szwagier & Pennec (2025) Grassmann optimizasyonunu iç içe projektörlerle flag manifolduna kaldırmayı öneriyor. Bu, doc 14'ün $\sum_j\beta_j d_c^2(P_{\le j})$ kaybının istatistikteki tam karşılığı. NeuralSVD'nin "nesting"i de aynı şeyin NN versiyonu.
- **Önemli ayrım:** Flag kaybı *matrissiz* doğrudan regresyon için doğru araçtır. Çıkarımda operatör (mesh matrisleri) elimizdeyse, RR alt uzay içindeki sıralamayı ve yönü **kendisi** çözer. Bu durumda yalnız $P_{\le K}$ hedeflenmesi yeterli ve daha iyi (§5.2).

### 2.5 Blok 4: Teori

- **Genel alt sınırlar:** Yalnız $C^r$ veya Lipschitz düzenliliğiyle tanımlanan operatör sınıfları için sinir operatörlerinin parametre sayısı ve gereken örnek sayısı hataya göre üstel büyür (Lanthaler & Stuart 2023; Kovachki, Lanthaler & Mhaskar 2024; Kovachki, Lanthaler & Stuart 2024 derlemesi). Kovachki et al. ayrıca "parametrik verimlilik ⇒ veri verimliliği" olduğunu FNO için gösteriyor.
- **Holomorfi kurtarır:** Referans domain'e difeomorf şekil ailelerinde afin-parametrik şekil kodlaması PDE'yi holomorf parametrik bir aileye çeviriyor ve boyuttan bağımsız ifade oranları veriyor (Harbrecht & Schwab 2026; Schwab & Zech 2019). Özdeğer tarafında basit (izole) özdeğerler ve özvektörleri parametreye holomorf bağlıdır (Kato; Andreev & Schwab 2012; Bahn 2024, temel özçift için türev sınırları). Kümeler için simetrik fonksiyonlar ve küme projektörü analitiktir (doc 14 §1.1, Lamberti & Lanza de Cristoforis). DEN ise özuzayın *Lipschitz* sürekliliğini kanıtlıyor.
- **Sonuç (hedef seçimi):** Sıralı $\lambda_k$ kesişimlerde yalnız Lipschitz'tir, holomorf değildir. Teori bu yüzden (i) $\log\lambda_1$ (her zaman basit), (ii) ayrık kümelerin iz/ortalama değerleri ve projektörleri, (iii) $P_{\le K}$ (δ_K büyükken) hedeflerini "öğrenilebilir" sınıfa koyar. Tekil $u_2,u_3$ ise bu sınıfta değildir. Bu, Öneri 1'in teorik gerekçesidir.
- **Boyuttan bağımsız genelleme:** Lu & Lu (2022) temel durum spektral Barron uzayındaysa 2 katmanlı NN'in genelleme hatasının boyuttan bağımsız oranla azaldığını kanıtlıyor. Bu tek-operatör sonucudur, bizim operatör-öğrenimi rejimine doğrudan aktarılmaz.

---

## 3. Doc 14'ün eleştirisi

| Doc 14 önerisi | Literatür | Değerlendirme |
|---|---|---|
| **#1 Frekansı fizikten hesapla**, log-λ kaybı | Destekliyor. Zhao & Fogler ağın ölçek yasasını kendiliğinden öğrendiğini gösteriyor, ama tam kodlama her zaman daha iyi. Li–Sun–Zhang "domain scaling"i açık ön işleme olarak kullanıyor. | ✅ Aynen. Öneri 1 ile frekans başlığı tamamen ortadan kalkar. |
| **#2 max_nodes alt-örneklemesini kapat** | Destekliyor. GINO/PCNO/NEO quadratür-tutarlı integral katmanlarını discretization-invariance için şart koşuyor. Rastgele 1024 alt-örnek quadratürü bozar. | ✅ Aynen. Öneri 1'de P1 montajı tam mesh ister: eleman alt-kümesi değil, bucket batching. |
| **#3 L²(Ω) ağırlıkları, flag kaybı, ≥5 mod** | Ağırlıklar: NEO mass-aware tasarımıyla tam destek. Flag: Szwagier & Pennec ve NeuralSVD nesting ile destek. ≥5 mod: NEO/DEN'in redundant baz tercihi ve $\delta_K$ ihtiyacıyla destek. | ⚠️ Kısmen. Flag kaybı *doğrudan regresyon* için doğru, ama matrisler varken RR + $P_{\le K}$ kaybı flag'i belirgin biçimde geçiyor (§5.2). Ayrıca $L^2$ flag kaybı özdeğeri kontrol etmez (§5.1). Doc 14 bu ayrımı yapmıyor. |
| **#4 SpectralNO'yu H¹₀-uyumlu yap** (ψ = ω·N(RFF(x), g), eleman içi quadratür) | Literatürün baskın yolu *discretize-then-project*: NEO, DEN (POD), FEENet ve Galerkin POD-NN, ağın nodal/POD değerlerini bir FE fonksiyonu olarak yorumlayıp matrisleri **tam** kuruyor. | ⚠️ Öncelik tersine dönmeli. Nodal ψ değerlerini P1 fonksiyonu olarak alıp $\Phi^\top K_h\Phi$'yi eleman gradyanlarıyla kurmak uyumu (P1 ⊂ H¹₀), tam quadratürü ve Courant–Fischer üst sınırını **ücretsiz** verir. Autograd ∇ψ, `dir_bnd` lineerleştirmesi ve `bc_scale` sorunları da kalkar. Sürekli (mesh'siz) ψ ancak mesh'siz çıkarım gerekiyorsa değerlidir. |
| **#5 Hibrit çıkarım** (NN → blok ters iterasyon + RR) | Güçlü destek: NEO (RR refinement), NOWS (warm start), Dai et al. | ✅ ve **yükselt**: RR'yi yalnız çıkarımda değil **eğitim ileri geçişinde** kullan (Öneri 1). §5.1: ters iterasyon yüksek frekanslı hatayı 10⁵ kat siliyor. |
| **#6 Düşük boyutlu şekil girdisi + fiziksel taban** (r(θ) Fourier, PT2, landscape) | Teori destekliyor (Harbrecht & Schwab; holomorfi). DIMON, Geo-FNO ve CGA-DL-ROM parametre/referans-domain girdisini kullanıyor. | ⚠️ Destek *teorik*: özproblemde mesh-girdi vs. düşük-boyut-girdi karşılaştırması yapan bir benchmark bulamadım. Öneri: mesh girdisini *değiştirmek* yerine global şekil kodunu *eklemek* (Öneri 5). |
| **#7 Üretici düzeltmeleri** (smooth şekiller aslında köşeli) | ML literatürü nötr. Köşe tekilliği klasik FEM konusu. | ✅ Geçerli. Holomorfi argümanı açısından da önemli: köşe sayısı/açısı değişen poligon ailesi pürüzsüz bir şekil manifoldu değil. |
| **#8 MPS/Fredholm validasyonu** | Nötr | ✅ Geçerli, düşük öncelik. |

**Doc 14'ün eksikleri:**

1. **Enerji normu vs L²** (§5.1). Doc 14 §3.1 "özdeğer hatası = enerji normunda karesel alan hatası" diyor. Ama önerdiği alan kayıpları ($d_c^2$, flag, $1-\cos^2_M$) tamamen $L^2$'de. Bir kayıp ya enerji-farkında olmalı (Ritz değeri) ya da ters iterasyonla filtrelenmiş bir hedefe bakmalı.
2. **Redundant baz + eğitimde RR.** Doc 14 RR'yi SpectralNO'nun öğrenilmiş quadratürü içinde ya da çıkarım sonrası düşünüyor. Literatürün ara yolu (gerçek FE matrisleriyle eğitim-içi RR) daha basit ve daha güçlü.
3. **Mimari düzeyde kütle ağırlığı.** Doc 14 ağırlıkları kayıplara taşıyor, ama GNOT'un attention ve pooling'i de ağırlıksız bir Riemann toplamı (mesh-yoğunluğu bağımlı). NEO bunu mimaride çözüyor.
4. **Sign/basis literatürünün doğru konumu.** Doc 14 bunları kısaca anıyor. Asıl mesaj şu olmalı: çıktıda SignNet gerekmez, alt uzay yeterli. Girdide PCA-kanonikleştirmesi izotrop şekillerde kararsız.
5. **Değerlendirme protokolü.** Önerilen metrikler: $M$-ağırlıklı temel açılar ($P_1$, $P_{\le2}$, $P_{\le3}$), RR ve ters-iterasyon-sonrası λ hataları, ayrıca yakın-dejenere alt küme (δ<%5) için ayrı rapor. Repo'nun `field_rel_l2`'si ağırlıksız ve tekil mod bazlı.
6. **Teorik sınırlar.** Doc 14 örnek karmaşıklığından söz etmiyor. Parametrik/veri karmaşıklığı alt sınırları, düşük boyutlu kod ve holomorf hedef seçiminin neden "süs" değil gereklilik olduğunu açıklıyor.

---

## 4. Tasarım tartışması

### 4.1 Önerilen hedef ve kayıp (Öneri 1–2'nin matematiği)

Her geometri için $K_h, M_h$ (P1, mesh'in kendisi) ve ağ çıktısı $\Phi\in\mathbb R^{N\times m}$ ($m=2K$, sınır düğümlerinde sıfır: hard Dirichlet maskesi, uyum otomatik):

$$
G_M=\Phi^\top M_h\Phi=CC^\top,\quad Q=\Phi C^{-\top},\quad \hat K=Q^\top K_h Q=W\hat\Lambda W^\top,\quad \hat U=QW_{:,1:K}.
$$

- **Denetimli kayıp** ("capture"): $\mathcal L_{span}=K-\|Q^\top M_h U_K\|_F^2=\sum_{i\le K}\sin^2\theta_i(\mathrm{span}\,\Phi,P_{\le K})$. Bu, işaretten, küme içi dönmeden ve $\Phi$'nin baz seçiminden bağımsızdır. $m>K$ olduğunda $\mathrm{span}\,\Phi\supset P_{\le K}$ yeterli, eşitlik gerekmez.
- **Enerji terimi** (isteğe bağlı): $\sum_{k\le K}\hat\lambda_k/\lambda_k-1\ge0$ (Courant–Fischer: $\hat\lambda_k\ge\lambda_k^{h}$). Etiketsiz sürüm: $\sum_k\hat\lambda_k$ (Ky Fan).
- **Çıktılar:** $\hat\lambda_k$ ve $\hat u_k$ RR'den gelir. Sıralama eigh'ten gelir, OT'ye gerek yoktur. Frekans: $c\sqrt{\hat\lambda_k}/(2\pi s)$.
- **Çıkarım:** $W=K_h^{-1}M_h\hat U$ (1 seyrek LU), sonra ikinci bir RR. Mümkünse P2 matrisleriyle.
- **Maliyet:** P1 montajı eleman başına 3×m değer. N≈1.6k, m=6 için ihmal edilebilir. Mevcut autograd-∇ψ yolundan (M=16 VJP) daha ucuz.

**Neden işe yarıyor:** Ağ RR'nin çözebileceği şeyi (alt uzay içindeki yön ve sıra) öğrenmek zorunda kalmaz. Davis–Kahan'a göre $P_{\le K}$ yalnız $\delta_K$'ya bağlı olarak iyi koşulludur (bizim veride $\delta_3$ medyan 0.34). Yakın-dejenere dipol çiftinin yönünü gerçek operatör belirler. Bu, doc 14 §4.1'deki "W matrisini köşegenleştiren baz" sonucunun hesaplamalı karşılığıdır.

### 4.2 Özdeğerde enerji mi, L² mi?

$\hat X=\cos\theta\,U+\sin\theta\,Z$ ($Z\perp_M U$) için RR hatası yaklaşık $\sin^2\theta\,(\rho(Z)-\lambda_k)$'dir; burada $\rho(Z)$, hata yönünün Rayleigh bölümüdür. Pürüzsüz hata için $\rho\approx\lambda_{4..6}$, mesh-ölçekli gürültü için $\rho\sim h^{-2}$ olur. Tek bir ters iterasyon hata yönünü $\lambda_k/\rho(Z)$ oranında sönümler. Sonuç: RR+ters iterasyon zincirinde asıl önemli olan hata **düşük frekanslı** hatadır. Kaybı bu zincirin *çıktısı* üzerinde tanımlamak ("refine-aware loss", solver-in-the-loop) bunu otomatik olarak doğru ağırlıklandırır.

### 4.3 Düşük boyutlu parametrizasyon mesh-noktası girdisini yener mi?

- **Teori:** Evet, *hedef şekle holomorf bağlıysa ve kod afin-parametrikse*. Harbrecht & Schwab (2026) oranları kodun boyutuna değil katsayı dizisinin toplanabilirliğine bağlı. r(θ) katsayıları $|c_k|\sim k^{-2}$ ile azalıyor (doc 14 E4), yani bu koşul sağlanıyor. Poligon ailesi (köşe sayısı değişken) ise tek bir pürüzsüz manifold değil, holomorfi global olarak geçerli değil.
- **Ampirik:** Geometri-değişken benchmark'larda mesh-noktası operatörleri çok güçlü (Transolver Elasticity 0.64%). Ancak bunlar düşük boyutlu kodun *mevcut olmadığı* senaryolar. Özproblemde iki girdiyi doğrudan kıyaslayan bir çalışma bulamadım. Shape Space Spectra ve Operator Inference (piksel) iki uçtan birer örnek.
- **Pratik öneri:** İkisini birleştir. Mesh girdisi yerel özellikler (köşe tekilliği, sınır bandı) için, global kod ise holomorf ve düşük boyutlu parametre bağımlılığı için. GNOT'ta ek bir token, SpectralNO'da ψ'nin koşul vektörü.

---

## 5. Deneyler

Tüm deneyler CPU'da, saniyeler ile ~12 dakika arasında sürdü.

### 5.1 E15-1: Aynı L² hatası, çok farklı özdeğer hatası (repo meshleri)

**Kurulum.** Veri: `gen_400.pkl`'deki ilk 60 gerçek geometri (doc 14 ajanı: üretici varsayılanları, seed 7). Her mesh'te P1 Dirichlet $K_h,M_h$ ve ilk 6 özçift hesaplandı. $P_{\le3}$'ün her temel açısı $\sin^2\theta=\varepsilon$ olacak biçimde üç hata tipiyle bozuldu:

- (a) *pürüzsüz* (u₄–u₆ kombinasyonu),
- (b) *düğüm gürültüsü* (iç düğümlerde beyaz gürültü),
- (c) *duvar-bandı gürültüsü* (duvara <3 mm düğümler: GNOT tipi sınır hatası).

Script: `e_energy.py`.

Medyan göreli Ritz-değeri hatası (mod 1 / 2 / 3):

| Hata tipi | ε | RR (ham) | + 1 blok ters iterasyon |
|---|---|---|---|
| pürüzsüz | 1e-3 | 3.2e-3 / 1.0e-3 / 6.0e-4 | 1.8e-4 / 2.5e-4 / 2.2e-4 |
| düğüm gürültüsü | 1e-3 | 4.3e-1 / 2.0e-1 / 1.6e-1 | 1.4e-5 / 2.5e-5 / 2.9e-5 |
| duvar-bandı gürültüsü | 1e-3 | **9.3e-1 / 4.5e-1 / 3.6e-1** | **3.7e-6 / 7.6e-6 / 9.1e-6** |
| pürüzsüz | 1e-2 | 3.2e-2 / 1.0e-2 / 6.0e-3 | 1.8e-3 / 2.5e-3 / 2.3e-3 |
| duvar-bandı gürültüsü | 1e-2 | 9.2 / 4.5 / 3.6 | 3.7e-5 / 7.7e-5 / 9.2e-5 |

(Hatalar ε ile tam lineer ölçekleniyor. ε=1e-4 satırları aynı oranlarda.)

**Yorum.**

- Aynı $L^2$ alt uzay hatası, hatanın *frekans içeriğine* bağlı olarak üç mertebe farklı özdeğer hatası verir. $L^2$-tabanlı alan kaybı (repo'nun rel-L2'si ya da doc 14'ün flag kaybı) λ kalitesinin güvenilir bir göstergesi değildir.
- Ters iterasyon sıralamayı tersine çevirir. Yüksek frekanslı hata ~10⁵ kat sönümlenir, pürüzsüz hata yalnız ~λ_k/λ_{4..6} oranında azalır. Ağın öncelikle **düşük frekanslı** alt uzay doğruluğunu öğrenmesi gerekir. Sınır bandındaki gürültülü artıklar RR+ters iterasyon zincirinde neredeyse bedavaya temizlenir.

### 5.2 E15-2: Oyuncak operatör öğrenimi, kayıp varyantlarının karşılaştırması

**Kurulum (`toy_data.py`, `toy_train.py`).**

- *Şekiller:* $r(\theta)=1+\sum_{k=2}^{7}a_k\cos k\theta+b_k\sin k\theta$, $a_k,b_k\sim U(\pm0.1)$. 700 şekil, tek bir disk mesh'inin (545 düğüm, 1024 üçgen) morfolanmasıyla elde edildi (ortak bağlantılılık; DIMON/Geo-FNO tipi referans domain).
- *Etiketler:* P1 Dirichlet, 6 özçift.
- *Boşluk dağılımı:* $\delta_2$ medyan 0.20, <%5: %3.7, <%10: %15.6. Gerçek veriye benzer (doc 14: %4.8 / %11.7).
- *Model:* MLP(12 katsayı → 3×256 → N·m) ve sınırda sıfır maskesi. Adam, 3000 adım, batch 64, 500 eğitim / 200 test şekli.
- *Varyantlar:*
  - `direct_sign`: m=3, mod başına $1-\cos^2_M$ + log-λ başlığı (repo'nun tekil mod kaybı).
  - `direct_hard`: m=3, repo'nun sert küme kuralı (göreli boşluk <%2 → çiftte Grassmann).
  - `flag`: m=3, doc 14 flag kaybı (τ=0.03).
  - `span3`: m=3, yalnız $P_{\le3}$ Grassmann + RR.
  - `span6`: m=6 redundant, $P_{\le3}$ capture + RR (NEO/DEN tipi).
  - `span6_kf`: span6 + Ritz-değeri terimi.
  - `kyfan6`: m=6, **etiketsiz**, $\sum_{k\le3}\hat\lambda_k^{RR}$.
- *Metrikler:* Tüm iç çarpımlar $M_h$-ağırlıklı. Doğrudan varyantlarda alanlar ham çıktıdır, span/kyfan varyantlarında RR vektörleridir. "Yakın": test setinde $\delta_2<5\%$ olan 7 şekil. λ(1 inv. it.): tahmin edilen 3 alanla bir blok ters iterasyon + RR (doc 14 Öneri 5).

**Sonuçlar** (test medyanları; iki değer = seed 0 / seed 1, tek değer = yalnız seed 0):

| Varyant | $P_{\le3}$ $\sum\sin^2/3$ | mod-1 $\sin^2$ | mod-2 $\sin^2$ (δ₂>%10) | mod-2 $\sin^2$ (yakın) | λ başlık | λ RR | λ (1 inv. it.) |
|---|---|---|---|---|---|---|---|
| direct_sign | 4.0e-3 / 3.5e-3 | 1.3e-3 | 6.1e-3 | **0.20 / 0.29** | %0.96 / 0.84 / 0.98 | %11 / 7.3 / 7.9 | 6.2e-5 / 6.6e-4 / 1.1e-3 |
| direct_hard | 3.7e-3 | 1.2e-3 | 5.6e-3 | 0.32 | %0.81 / 0.89 / 0.92 | %11 / 6.9 / 7.5 | 5.7e-5 / 5.5e-4 / 1.1e-3 |
| flag (doc 14) | 2.4e-3 / 2.6e-3 | 9.1e-4 | —¹ | —¹ | %0.95 / 0.87 / 1.1 | %6.6 / 4.7 / 6.3 | 3.9e-5 / 4.0e-4 / 5.8e-4 |
| span3 | 6.5e-4 / 6.6e-4 | 3.1e-4 | 6.1e-4 | 5.7e-4 / 4.4e-4 | — | %3.6 / 1.7 / 1.9 | 1.5e-5 / 6.5e-5 / 1.6e-4 |
| **span6** | **5.5e-4 / 6.2e-4** | **1.7e-4** | **4.3e-4** | **4.8e-4 / 4.8e-4** | — | %1.7 / 1.0 / 1.2 | **9.4e-6 / 5.4e-5 / 1.8e-4** |
| span6_kf | 2.4e-3 | 9.0e-4 | 2.9e-3 | 3.6e-3 | — | %2.3 / 1.4 / 1.9 | 6.0e-5 / 4.4e-4 / 6.5e-4 |
| kyfan6 (etiketsiz) | 8.9e-3 | 1.9e-3 | 7.4e-3 | 9.4e-3 | — | %2.8 / 2.1 / 3.2 | 1.9e-4 / 1.4e-3 / 3.3e-3 |

¹ Flag kaybı kolonlarının iç içe span'lerini hedefler. $f_2$'nin içinde $u_1$ bileşeni olabilir, bu flag'in doğal serbestliğidir. Bu yüzden ham kolon bazlı mod-2 metriği flag için anlamlı değil (seed 1'de 0.46 / 0.92 çıktı). Flag için adil metrikler $P_{\le3}$ ve λ sütunlarıdır.

**Yorum.**

1. **Alt uzay + RR en iyisi** (NEO/DEN iddiası doğrulandı). `span6` doğrudan regresyona göre $P_{\le3}$ hatasını ~6–7×, iyi ayrık mod-2 hatasını ~14× düşürüyor. Yakın-dejenere mod-2'de fark **400–600×**: RR, çift içindeki yönü gerçek matristen alıyor, ağın öğrenmesine gerek kalmıyor. Redundansın katkısı (span3 → span6) orta düzeyde: mod-1'de ~2×, RR-λ'da ~2×. İki seed tutarlı.
2. **Repo'nun sert küme kuralı, işaret-invaryant per-mode kayba göre anlamlı bir kazanç sağlamıyor** (0.32 vs 0.20; 7 örnek, istatistiksel güç düşük). Beklenen de bu: yalnız δ<%2'yi yakalıyor, %2–5 bandı Davis–Kahan'a göre hâlâ kötü koşullu.
3. **Doc 14 flag kaybı** $P_{\le3}$'ü doğrudan regresyona göre ~1.5× iyileştiriyor ve λ(1 inv. it.)'i %35–50 düşürüyor. Ama span-RR'nin 4× gerisinde kalıyor. Matris varken flag'e gerek yok. Matrissiz çıkarım için hâlâ en iyi doğrudan kayıp.
4. **λ kaynağı:** L²-eğitilmiş span'den ham RR, λ başlığından (~%0.9) daha kötü (%1–11). Bu, E15-1'in öngördüğü mesh-ölçekli artık enerjisidir. Ters iterasyon sonrası ise span6, başlıktan **~50–1000×** daha iyi (mod 1: 9·10⁻⁶ vs ~1·10⁻²; mod 3: 1.8·10⁻⁴ vs ~1·10⁻²). Özdeğer bir regresyon başlığından değil, RR + 1 ters iterasyondan okunmalı.
5. **Ritz-değeri terimi** (`span6_kf`, ağırlık 1) bu bütçede *zarar verdi*. Olası neden: ölçek dengesizliği (Ritz terimi ~%2, capture ~5·10⁻⁴) ve kötü koşullanmış gradyan. Enerji-farkında terim ancak düşük ağırlıkla ya da ters iterasyonlu hedefle denenmeli. Bu negatif sonucu raporluyoruz.
6. **Etiketsiz Ky Fan** (`kyfan6`) hiç özvektör/özdeğer etiketi görmeden $P_{\le3}$ 8.9·10⁻³ ve RR-λ %2–3'e ulaştı. Denetimli span6'nın gerisinde, ama direct_sign'ın RR-λ'sından iyi. Etiketsiz ince ayar (yeni şekil aileleri) için umut verici. SpIN/NeuralSVD fikrinin operatör-öğrenimi rejimindeki karşılığı.

**Sınırlar.** Tek bir oyuncak aile: ortak mesh, düşük boyutlu girdi, MLP, P1 etiketler, köşesiz şekiller. Yakın-dejenere alt küme küçük (7). GNOT veya gerçek meshlerle eğitim yapılmadı. Sonuçlar *göreli* sıralama olarak okunmalı.

### 5.3 E15-3: P1 tabanı ve maliyet (repo meshleri)

`e_p1floor.py`, `gen_400`'ün ilk 80 geometrisi (medyan 1520 düğüm). Aynı mesh'te P1 özdeğerlerinin P2 etiketlerine göre farkı:

- sharp: medyan +0.25% / +0.48% / +0.67% (max +0.89%),
- smooth: +0.43% / +0.51% / +0.65%.

Hep pozitif (üst sınır). Montaj + `eigsh` süresi: P1 medyan 0.08 s, P2 0.19 s.

**Sonuç.**

- Öneri 1'in P1-RR'si %0.3–0.7'lik bir *yanlılık tabanına* sahiptir. Bu, bugünkü NN hatalarının çok altındadır, ama nihai λ için yeterli değildir. Nihai λ ya P2 matrisli ters iterasyonla ya da GNOT'un kenar-ortası sorgularıyla (gerçek P2 alanı, Öneri 7) alınmalıdır.
- Doc 14'ün önerdiği "kenar ortası = uç değerlerin ortalaması" interpolasyonu, P2 uzayında yine bir P1 fonksiyonu üretir. RR tabanını düşürmez; ancak sonrasındaki P2 ters iterasyonu için başlangıç olarak yeterlidir.

---

## 6. Uygulama taslağı (dosya bazında)

1. **`src/data/dataset_converter.py` / `dataset.py` / `gnot_collate_fn`:** `elements` (zaten var), eleman alanları ve P1 referans-gradyan operatörleri (ya da collate'te hesaplanır), `scale`, lumped `w`, ≥5 mod λ'sı (`lam_all`). `max_nodes: null` + bucket batching.
2. **Yeni `src/models/ritz_head.py`:** `assemble_p1(Phi, elems, G, area) → (Φᵀ K Φ, Φᵀ M Φ)`, `ritz_pairs(Phi, ...)` (Cholesky + eigh, float64, mevcut `_BroadenedEigh` yeniden kullanılabilir), `subspace_capture_loss`.
3. **`spectral_no.py`:** `_basis_gradient` (autograd), `_differentiable_inputs`, düğüm-quadratürü ve `_MonotoneFreqHead` yerine `ritz_head`. ψ nodal değerleri × sınır maskesi (hard Dirichlet). Global şekil kodu için ψ'ye pooled bağlam.
4. **`gnot.py`:** K yerine m=2K çıktı kolonu. `LinearAttention`/`AttentionPool`'da $w$ ağırlıkları. İsteğe bağlı olarak kenar-ortası sorgu noktaları.
5. **`lightning_module.py`:** `_compute_loss_*` → capture kaybı (+ düşük ağırlıklı Ritz terimi). Metrikler: $M$-temel açılar, RR-λ, 1-ters-iterasyon-λ, yakın-dejenere alt küme ayrı. `ot_match`, `detect_clusters`, `soft_procrustes_loss` yalnız geriye dönük uyumluluk için.
6. **`infer.py`:** `--refine {0,1,2}` (P2 matrisleriyle ters iterasyon + RR). Frekans = $c\sqrt{\lambda}/(2\pi s)$.

---

## 7. Referanslar

**Nöral özçözücüler ve özuzay öğrenimi**

- Pfau, D., Petersen, S., Agarwal, A., Barrett, D. G. T., Stachenfeld, K. L. (2019). *Spectral Inference Networks: Unifying Deep and Spectral Learning.* ICLR. [arXiv:1806.02215](https://arxiv.org/abs/1806.02215)
- Deng, Z., Shi, J., Zhu, J. (2022). *NeuralEF: Deconstructing Kernels by Deep Neural Networks.* ICML. [arXiv:2205.00165](https://arxiv.org/abs/2205.00165)
- Deng, Z., Shi, J., Zhang, H., Cui, P., Lu, C., Zhu, J. (2025). *Neural Eigenfunctions Are Structured Representation Learners.* IEEE TPAMI. [arXiv:2210.12637](https://arxiv.org/abs/2210.12637), [doi:10.1109/TPAMI.2025.3625728](https://doi.org/10.1109/tpami.2025.3625728)
- Ryu, J. J., Xu, X., Erol, H. S. M., Bu, Y., Zheng, L., Wornell, G. W. (2024). *Operator SVD with Neural Networks via Nested Low-Rank Approximation.* ICML. [arXiv:2402.03655](https://arxiv.org/abs/2402.03655)
- E, W., Yu, B. (2018). *The Deep Ritz Method.* Commun. Math. Stat. [arXiv:1710.00211](https://arxiv.org/abs/1710.00211)
- Han, J., Lu, J., Zhou, M. (2020). *Solving high-dimensional eigenvalue problems using deep neural networks: A diffusion Monte Carlo like approach.* J. Comput. Phys. 423. [arXiv:2002.02600](https://arxiv.org/abs/2002.02600)
- Lu, J., Lu, Y. (2022). *A Priori Generalization Error Analysis of Two-Layer Neural Networks for Solving High Dimensional Schrödinger Eigenvalue Problems.* Commun. AMS. [arXiv:2105.01228](https://arxiv.org/abs/2105.01228)
- Rowan, C., Evans, J., Maute, K., Doostan, A. (2025). *Solving engineering eigenvalue problems with neural networks using the Rayleigh quotient.* [arXiv:2506.04375](https://arxiv.org/abs/2506.04375)
- Fernández Bonder, J., Salort, A. M. (2025). *A PINNs approach for the computation of eigenvalues in elliptic problems.* [arXiv:2507.03126](https://arxiv.org/abs/2507.03126)
- Banderwaar, A. S., Gupta, A. (2025). *Fast PINN Eigensolvers via Biconvex Reformulation.* NeurIPS ML4PS. [arXiv:2511.00792](https://arxiv.org/abs/2511.00792)
- Wang, H., Jiang, Y., Wang, J., Li, X., Luo, J., Dong, H. (2025). *STNet: Spectral Transformation Network for Solving Operator Eigenvalue Problem.* NeurIPS. [arXiv:2510.23986](https://arxiv.org/abs/2510.23986)
- Dai, X., Fan, Y., Sheng, Z. (2024). *Subspace method based on neural networks for eigenvalue problems.* [arXiv:2410.13358](https://arxiv.org/abs/2410.13358)
- Chang, Y., Benchekroun, O., Chiaramonte, M. M., Chen, P. Y., Grinspun, E. (2025). *Shape Space Spectra.* ACM TOG. [arXiv:2408.10099](https://arxiv.org/abs/2408.10099), [doi:10.1145/3731148](https://dl.acm.org/doi/10.1145/3731148)
- Li, H., Sun, J., Zhang, Z. (2025). *Operator Inference for Elliptic Eigenvalue Problems.* [arXiv:2504.15733](https://arxiv.org/abs/2504.15733)
- Li, H., Sun, J., Zhang, Z. (2025). *Deep Eigenspace Network for Parametric Non-self-adjoint Eigenvalue Problems.* [arXiv:2512.20058](https://arxiv.org/abs/2512.20058)
- Yang, Z., Du, T., Liu, L. (2026). *Learning Laplacian Eigenspace with Mass-Aware Neural Operators on Point Clouds (NEO).* SIGGRAPH 2026. [arXiv:2605.24390](https://arxiv.org/abs/2605.24390), [doi:10.1145/3799902.3811185](https://doi.org/10.1145/3799902.3811185), [kod](https://github.com/Adversarr/NEO)
- Zhang, H., Ogren, A., Rudin, C., Guilleminot, J., Brinson, L. C. (2026). *Learning Metamaterial Eigenmodes with Wavelet-Encoded Fourier Neural Operators.* [arXiv:2609.08102](https://arxiv.org/abs/2609.08102)
- Zhao, Y., Fogler, M. M. (2022). *Can A Neural Network Hear the Shape of A Drum?* [arXiv:2203.08073](https://arxiv.org/abs/2203.08073)
- Kamkari, H., Nabizadeh, M. S., Solomon, J. (2026). *Learning Orthonormal Bases for Function Spaces.* [arXiv:2605.19959](https://arxiv.org/abs/2605.19959)
- Li, S., Salahshoor, H. (2026). *Finite Element Eigenfunction Network (FEENet): A Hybrid Framework for Solving PDEs on Complex Geometries.* [arXiv:2602.00870](https://arxiv.org/abs/2602.00870)
- Chen, Y., Liu, Z., Wang, H., Liu, L. (2025). *Learning-Guided Integration Contours Construction for Fast Large-Scale Generalized Eigensolvers (DeepContour).* [arXiv:2511.01927](https://arxiv.org/abs/2511.01927)
- *NOWS: Neural Operator Warm Starts for Accelerating Iterative Solvers* (2025). [arXiv:2511.02481](https://arxiv.org/abs/2511.02481) (yazarları doğrulanamadı; yalnız bağlam için).

**Geometri-farkında operatörler ve indirgenmiş modeller**

- Li, Z., Huang, D. Z., Liu, B., Anandkumar, A. (2023). *Fourier Neural Operator with Learned Deformations for PDEs on General Geometries (Geo-FNO).* JMLR. [arXiv:2207.05209](https://arxiv.org/abs/2207.05209)
- Li, Z., Kovachki, N., Choy, C., et al., Anandkumar, A. (2023). *Geometry-Informed Neural Operator for Large-Scale 3D PDEs (GINO).* NeurIPS. [arXiv:2309.00583](https://arxiv.org/abs/2309.00583)
- Hao, Z., et al. (2023). *GNOT: A General Neural Operator Transformer for Operator Learning.* ICML. [arXiv:2302.14376](https://arxiv.org/abs/2302.14376)
- Cao, S. (2021). *Choose a Transformer: Fourier or Galerkin.* NeurIPS. [arXiv:2105.14995](https://arxiv.org/abs/2105.14995)
- Wu, H., Luo, H., Wang, H., Wang, J., Long, M. (2024). *Transolver: A Fast Transformer Solver for PDEs on General Geometries.* ICML. [arXiv:2402.02366](https://arxiv.org/abs/2402.02366)
- Luo, H., et al. (2025). *Transolver++: An Accurate Neural Solver for PDEs on Million-Scale Geometries.* [arXiv:2502.02414](https://arxiv.org/abs/2502.02414)
- Yin, M., et al. (2024). *A scalable framework for learning the geometry-dependent solution operators of partial differential equations (DIMON).* Nature Comput. Sci. 4, 928–940. [arXiv:2402.07250](https://arxiv.org/abs/2402.07250), [doi:10.1038/s43588-024-00732-2](https://www.nature.com/articles/s43588-024-00732-2)
- Zeng, C., Zhang, Y., Zhou, J., Wang, Y., Wang, Z., Liu, Y., Wu, L., Huang, D. Z. (2025). *Point Cloud Neural Operator for Parametric PDEs on Complex and Variable Geometries.* CMAME 443. [arXiv:2501.14475](https://arxiv.org/abs/2501.14475)
- Wen, S., Kumbhat, A., Lingsch, L., Mousavi, S., Zhao, Y., Chandrashekar, P., Mishra, S. (2025). *Geometry Aware Operator Transformer as an Efficient and Accurate Neural Surrogate for PDEs on Arbitrary Domains (GAOT).* NeurIPS. [arXiv:2505.18781](https://arxiv.org/abs/2505.18781)
- Mousavi, S., Wen, S., Lingsch, L., Herde, M., Raonić, B., Mishra, S. (2025). *RIGNO: A Graph-based framework for robust and accurate operator learning for PDEs on arbitrary domains.* NeurIPS. [arXiv:2501.19205](https://arxiv.org/abs/2501.19205)
- Brivio, S., Fresca, S., Manzoni, A. (2024). *Handling geometrical variability in nonlinear reduced order modeling through Continuous Geometry-Aware DL-ROMs.* [arXiv:2411.05486](https://arxiv.org/abs/2411.05486)
- Xia, Y., Aviles-Rivero, A. I. (2026). *Do Neural Operators Forget Geometry? The Forgetting Hypothesis in Deep Operator Learning.* [arXiv:2605.05862](https://arxiv.org/abs/2605.05862)
- Lu, L., Jin, P., Karniadakis, G. E. (2021). *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.* Nat. Mach. Intell. 3. [arXiv:1910.03193](https://arxiv.org/abs/1910.03193)
- Weder, P., Kast, M., Henríquez, F., Hesthaven, J. S. (2024). *Galerkin Neural Network-POD for Acoustic and Electromagnetic Wave Propagation in Parametric Domains.* Adv. Comput. Math. [arXiv:2406.13567](https://arxiv.org/abs/2406.13567)
- Hesthaven, J. S., Ubbiali, S. (2018). *Non-intrusive reduced order modeling of nonlinear problems using neural networks.* J. Comput. Phys. 363. [doi:10.1016/j.jcp.2018.02.037](https://doi.org/10.1016/j.jcp.2018.02.037)
- Sukumar, N., Srivastava, A. (2022). *Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks.* CMAME 389. [arXiv:2104.08426](https://arxiv.org/abs/2104.08426)

**Simetri, işaret/baz, alt uzaylar**

- Lim, D., Robinson, J., Zhao, L., Smidt, T., Sra, S., Maron, H., Jegelka, S. (2023). *Sign and Basis Invariant Networks for Spectral Graph Representation Learning.* ICLR. [arXiv:2202.13013](https://arxiv.org/abs/2202.13013)
- Lim, D., Robinson, J., Jegelka, S., Maron, H. (2023). *Expressive Sign Equivariant Networks for Spectral Geometric Learning.* NeurIPS. [arXiv:2312.02339](https://arxiv.org/abs/2312.02339)
- Ma, G., Wang, Y., Wang, Y. (2023). *Laplacian Canonization: A Minimalist Approach to Sign and Basis Invariant Spectral Embedding.* NeurIPS. [arXiv:2310.18716](https://arxiv.org/abs/2310.18716)
- Puny, O., Atzmon, M., Ben-Hamu, H., Misra, I., Grover, A., Smith, E. J., Lipman, Y. (2022). *Frame Averaging for Invariant and Equivariant Network Design.* ICLR. [arXiv:2110.03336](https://arxiv.org/abs/2110.03336)
- Kaba, S.-O., Mondal, A. K., Zhang, Y., Bengio, Y., Ravanbakhsh, S. (2023). *Equivariance with Learned Canonicalization Functions.* ICML. [arXiv:2211.06489](https://arxiv.org/abs/2211.06489)
- Weiler, M., Cesa, G. (2019). *General E(2)-Equivariant Steerable CNNs.* NeurIPS. [arXiv:1911.08251](https://arxiv.org/abs/1911.08251)
- Helwig, J., Zhang, X., Fu, C., Kurtin, J., Wojtowytsch, S., Ji, S. (2023). *Group Equivariant Fourier Neural Operators for Partial Differential Equations.* ICML. [arXiv:2306.05697](https://arxiv.org/abs/2306.05697)
- Szwagier, T., Pennec, X. (2024). *The curse of isotropy: from principal components to principal subspaces.* [arXiv:2307.15348](https://arxiv.org/abs/2307.15348)
- Szwagier, T., Pennec, X. (2025). *Nested subspace learning with flags.* [arXiv:2502.06022](https://arxiv.org/abs/2502.06022)

**Teori**

- Lanthaler, S., Stuart, A. M. (2023). *The Parametric Complexity of Operator Learning.* [arXiv:2306.15924](https://arxiv.org/abs/2306.15924)
- Kovachki, N. B., Lanthaler, S., Mhaskar, H. (2024). *Data Complexity Estimates for Operator Learning.* [arXiv:2405.15992](https://arxiv.org/abs/2405.15992)
- Kovachki, N. B., Lanthaler, S., Stuart, A. M. (2024). *Operator Learning: Algorithms and Analysis.* [arXiv:2402.15715](https://arxiv.org/abs/2402.15715)
- Harbrecht, H., Schwab, C. (2026). *Neural Shape Operator Surrogates – Expression Rate Bounds.* [arXiv:2604.18012](https://arxiv.org/abs/2604.18012)
- Schwab, C., Zech, J. (2019). *Deep learning in high dimension: Neural network expression rates for generalized polynomial chaos expansions in UQ.* Anal. Appl. 17. [doi:10.1142/S0219530518500203](https://doi.org/10.1142/S0219530518500203)
- Bahn, B.-H. (2024/2025). *Parametric holomorphy of elliptic eigenvalue problems.* [arXiv:2408.01227](https://arxiv.org/abs/2408.01227)
- Andreev, R., Schwab, C. (2012). *Sparse Tensor Approximation of Parametric Eigenvalue Problems.* In: Numerical Analysis of Multiscale Problems, LNCSE 83, Springer. [doi:10.1007/978-3-642-22061-6_7](https://link.springer.com/chapter/10.1007/978-3-642-22061-6_7)
- Klasik sayısal analiz referansları (Babuška–Osborn, Davis–Kahan, Knyazev LOBPCG, Kato, Lamberti–Lanza de Cristoforis) için bkz. [[14_MATHEMATICAL_IMPROVEMENTS]] §9.

> **Kaynak notu:** arXiv sayfalarına doğrudan erişim bu oturumda ağ proxy'si tarafından engellendi. Künyeler ve özet bilgileri web aramasının döndürdüğü arXiv/yayıncı/GitHub sonuçlarından alındı; NEO'nun teknik ayrıntıları resmi GitHub deposundan okundu. Transolver benchmark sayıları (Elasticity/Airfoil) arama sonuçlarından doğrulandı.

---

## 🔗 Bağlantılar

- [[14_MATHEMATICAL_IMPROVEMENTS]] · [[04_MODEL_ARCHITECTURE]] · [[11_ORTHOGONALITY_ANALYSIS]] · [[13_RAYLEIGH_ANALYSIS]] · [[10_IMPROVEMENT_IDEAS]]

#literatür #sciml #nöral-operatör #özuzay #rayleigh-ritz #grassmann #flag #equivariance #operatör-öğrenimi-teorisi
