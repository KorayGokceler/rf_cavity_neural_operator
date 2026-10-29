# Uyarlamalı etiketler ancak gerçek inceltmeyle yakınsar

Maxwell özdeğer problemi (`curl curl E = k² E`, PEC duvar) için literatürde uyarlamalı inceltmenin kanıtlanmış tek bir tarifi var. Nédélec elemanlarıyla çözülür. Çekirdek (diverjans ve normal sıçrama) terimlerini de içeren artık (residual) tahmin edici, hedef penceredeki bütün modlar üzerinden toplanır. Dörfler (toplu, bulk) işaretleme ile en küçük eleman kümesi seçilir. İşaretli her eleman iç içe (nested) ikiye bölme (bisection) ile mutlaka inceltilir. Döngü, tahmin edici tolerans altına inince durur ([Boffi–Gastaldi 2019](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)). Üretim kodları bu tarifin yalnızca bir kısmını uygular. Palace ve Omega3P, bizim gibi Zienkiewicz–Zhu (ZZ) türü kurtarma (recovery) tahmin edicisi kullanır ve **tahmin edici eşiğinde** durur. HFSS ve CST ise ardışık geçişler arasındaki frekans farkına bakar; HFSS bunun yanında her geçişte tetlerin **%30'unu inceltip büyümeyi zorlar**. Bizim döngümüzün tahmin edici ailesi (Nicaise sınıfı ZZ ve duvar terimi) üretim pratiğiyle uyumlu. Asıl kusur durdurma kuralında ve inceltme garantisinde. Netgen yeniden ağ örme (remeshing) isteğini uygulamazsa ardışık fark ~1e-15 çıkıyor ve döngü, kendi hata tahmini 1,6e-3 iken `converged` diyor. Bir şey daha var. Depodaki ölçümlerden çıkan yakınsama hızlarıyla (pillbox'ta f ≈ N^-1,9, Q0 ≈ N^-1,6, Bpk ≈ N^-0,5) hesaplanınca, %5'lik bir DOF artışında ölçülen ardışık fark gerçek hatanın f için yaklaşık **1/10'u**, Q0 için **1/12'si**, Bpk için **1/40'ı** kadar kalıyor. Yani yalnızca "büyüme oldu mu" diye bakan bir kontrol yetmez; farkın büyüme oranına göre ölçeklenmesi gerekir. Önerilen plan altı aşamalı:

| Aşama | İçerik | Etiket değişir mi |
|---|---|---|
| 0 | Ölçüm ve işaretleme | Hayır |
| 1 | Büyümeye ölçeklenmiş fark ve tahmin edici sağlamasıyla durdurma (`--strict_stop` kural 1'in genellemesi) | Evet |
| 2 | İnceltme garantisi (kural 2'nin önüne daha ucuz bir sıkılaştırma basamağı) | Evet |
| 3 | Dörfler tabanlı boyut alanı | Evet |
| 4 | Tahmin edici ve küme düzeltmeleri | Evet |
| 5 | Yalnız hızlandırıcı mod için hedefe yönelik inceltme | Evet |

Aşama 1'den itibaren her değişiklik etiketleri değiştirdiği için sahibin onayına bağlı.

## Kanıtlanmış döngünün dört şartı: çekirdek terimli tahmin edici, Dörfler, zorunlu inceltme, ince başlangıç

### Artık, dengelenmiş ve kurtarma tahmin edicileri farklı şeyler garanti eder

**Artık tahmin edici, Maxwell özdeğer problemi için güvenilirliği (reliability) ve verimliliği (efficiency) kanıtlanmış tek ailedir.** Boffi, Gastaldi, Rodríguez ve Šebestová'nın (BGRS) yerel göstergesi dört terimden oluşur ([BGRS 2019](https://ci2ma.udec.cl/pdf/pre-publicaciones2/2017/pp17-15.pdf)):

`μ_K² = h_K²‖E_h − curl curl E_h/ω_h²‖²_K + h_K²‖div E_h‖²_K + ½Σ_F [h_F‖[[(curl E_h/ω_h²) × n]]‖²_F + h_F‖[[E_h·n]]‖²_F]`

Terimler sırasıyla eleman artığı, eleman diverjansı, curl'ün teğetsel sıçraması ve E'nin normal sıçramasıdır. Toplam yalnızca iç yüzler üzerinden alınır, yani PEC duvar terimi yoktur. Sonuç, yeterince küçük h için `‖E − E_h‖_curl ≤ Cμ` ve `|λ − λ_h| ≤ Cμ²` bağıntılarıdır. Yerel verimlilik karma (mixed) formda geçerlidir. Primal formda ise verimlilik global ω − ω_h büyüklüğünü içerdiği için tam yerel değildir ([Boffi–Gastaldi, arXiv 1804.02377](https://arxiv.org/pdf/1804.02377)).

Diverjans ve normal sıçrama terimleri süs değildir. Hatanın gradyan (çekirdek) bileşenini tam olarak bunlar ölçer ([BGRS 2017, arXiv 1602.00675](https://arxiv.org/pdf/1602.00675)). Fichera alanında en düşük mertebede normal sıçrama terimi tahminin **yaklaşık %80'ini** oluşturdu. Aynı testte μ²'nin gerçek özdeğer hatasına oranı (etkinlik indeksi, effectivity index) **5,2–6,3** aralığında kaldı. Bu, kalibre edilmemiş bir artık tahmininin hatayı yaklaşık beş kat abarttığı anlamına gelir ([BGRS 2019](https://ci2ma.udec.cl/pdf/pre-publicaciones2/2017/pp17-15.pdf)).

Kaynak problemindeki öncüller şunlar: Beck–Hiptmair–Hoppe–Wohlmuth'un girdap akımı tahmini ([M2AN 2000](https://opus.bibliothek.uni-augsburg.de/opus4/files/57039/57039.pdf)) ve onu genel Lipschitz alanlara taşıyan Schöberl 2008 ([Math. Comp.](https://www.ams.org/mcom/2008-77-262/S0025-5718-07-02030-3/)). Artık tahmin edicilerin sabitleri p'ye bağlıdır. Bu yüzden yüksek mertebede üst sınır gevşer ([Chaumont-Frelet, arXiv 2402.17309](https://ar5iv.labs.arxiv.org/html/2402.17309)).

**Dengelenmiş akı (equilibrated flux) tahmin edicileri sabitsiz ve p-sağlam (p-robust) üst sınır verir. Ama bunlar curl-curl kaynak problemi içindir, özdeğer problemi için değil.** Bu çizginin başlıca çalışmaları:

- Gedicke–Geevers–Perugia–Schöberl, keyfi dereceli Nédélec elemanları için güvenilirlik sabiti 1, verimlilik sabiti p'den bağımsız bir tahmin verir ([arXiv 2004.08323](https://arxiv.org/abs/2004.08323)).
- Chaumont-Frelet–Vohralík, köşe yamalarında (vertex patch) yerel minimizasyonlarla "garantili, tam hesaplanabilir, sabitsiz ve p-sağlam" Prager–Synge sınırları kurar ([SIAM J. Numer. Anal. 2023](https://doi.org/10.1137/21M141909X)).
- Zaman-harmonik Maxwell için asimptotik olarak sabitsiz bir sürüm de var ([Chaumont-Frelet, Numer. Math. 2025](https://arxiv.org/abs/2402.17309)).

Garantili özdeğer ve küme sınırları (Cancès–Dusson–Maday–Stamm–Vohralík) yalnızca ikinci mertebe eliptik operatörleri kapsar ([Math. Comp. 2020](https://www.ams.org/journals/mcom/2020-89-326/S0025-5718-2020-03549-2)). Maxwell özdeğerleri için garantili alt sınır (Gallistl–Olkhovskiy) yalnızca en düşük mertebede ve yarı-düzgün ağlarda çalışır. Yazarlar bunun üç boyutta pratik kullanımını "çok sınırlı" buluyor: birim küpte λ_h = 19,76 iken alt sınır 2,26e-9 çıkıyor ([arXiv 2110.02605](https://arxiv.org/pdf/2110.02605)). Kaydırılmış problemler çözülerek Lagrange elemanlarıyla iki taraflı sertifikalı kapsama da elde edilebiliyor ([Barrenechea–Boulton–Boussaïd 2014](https://strathprints.strath.ac.uk/50866/1/Barrenechea_etal_SIAMJSC_2014_Finite_element_eigenvalue_enclosures.pdf)). Kısacası p3, eğri, 3B ve uyarlamalı ağlarda Maxwell özdeğeri için bugün kullanılabilir bir garantili sınır yok.

**Kurtarma (ZZ) tahmin edicileri Maxwell için yalnızca kaynak probleminde analiz edildi.** Nicaise 2005, E ve curl E'yi sürekli polinomlara köşe yaması ortalamasıyla kurtarır. Ortaya çıkan iki terimli tahminin yüz sıçramalarına denk olduğunu gösterir ve eleman artığını içermez. Cai–Cao ise H(curl)/H(div) kurtarma uzaylarına bir eleman artığı ekleyerek katsayıdan bağımsız sabitler elde eder ([Cai–Cao, arXiv 1504.00898](https://arxiv.org/pdf/1504.00898)).

Bizim göstergemiz bu sınıfın bir üyesi. E_h Nédélec'te teğetsel olarak sürekli olduğu için `‖E − R E‖` normal sıçramayı, yani BGRS'nin çekirdek terimini yakalar. `‖curl E − R curl E‖` ise curl'ün teğetsel sıçramasını yakalar. Eksik kalan, p ≥ 1 için sıfır olmayan eleman artığı `h_K‖E_h − curl curl E_h/λ_h‖`. Bu benzetme kaynak problemi için kanıtlı, özdeğer problemine taşınması kanıtsızdır. Yine de bu aile üretimde sınanmıştır: Palace, Nicaise 2005'e dayanan bir ZZ tahmin edicisini (Nédélec ve Raviart–Thomas uzaylarına izdüşüm) kullanıyor ([Palace reference.md](https://github.com/awslabs/palace/blob/8bd206f7e10e5a2ec93a2483514894d14c8a5102/docs/src/reference.md)).

| Aile | Maxwell özdeğer için kanıt | p-sağlam | Garantili | Maliyet | Üretimde |
|---|---|---|---|---|---|
| Artık (BGRS) | güvenilir + verimli, h küçükken ([BGRS 2019](https://ci2ma.udec.cl/pdf/pre-publicaciones2/2017/pp17-15.pdf)) | hayır | hayır (etkinlik ≈ 5–6) | eleman + yüz integrali | yayımlanmış bir üretim kodu bulunamadı |
| Dengelenmiş akı | yalnız kaynak problemi ([Chaumont-Frelet–Vohralík](https://doi.org/10.1137/21M141909X)) | evet | evet (kaynak için) | mod başına, yama başına yerel problem | yok |
| Kurtarma (ZZ) | yalnız kaynak problemi ([Cai–Cao](https://arxiv.org/pdf/1504.00898)) | bilinmiyor | hayır | bir izdüşüm | Palace, Omega3P, MFEM, NGSolve öğreticileri |
| DWR (özdeğer) | 2B hp, Maxwell ([Harmon–Notaroš](https://atmos.colostate.edu/~notaros/Papers/TAP_Adjoints_Benchmark.pdf)) | — | hayır | zenginleştirilmiş (p+1) ağırlık | yok |

### Dörfler işaretleme hız için gerekli; yakınsama için yalnızca işaretli elemanın gerçekten inceltilmesi gerekli

İşaretleme literatürü iki farklı sonucu birbirinden ayırır.

**Hızsız yakınsama** için iç içe uzaylar, makul bir tahmin edici ve "işaretli her elemanın en az bir kez bölünmesi" yeterlidir. Bu durumda Dörfler, maksimum ve eşdağılım (equidistribution) işaretlemelerinin hepsi işler ([Morin–Siebert–Veeser 2008](https://amcaonline.org.ar/ojs/index.php/cmm/article/view/2534)). Özdeğer problemleri için bu sonuç, basit ve katlı özdeğerlerde, *herhangi bir* başlangıç ağından itibaren geçerlidir ([Garau–Morin–Zuppa 2009](https://arxiv.org/abs/0803.0365)).

**Daralma (contraction) ve hız optimalliği** ise yalnızca yeterince küçük θ'lı ve yaklaşık en küçük kümeli Dörfler işaretlemesi için kanıtlıdır. Kümeler (BGGG) için θ sınırı küme boyutuyla kötüleşen açık bir sabittir ([ar5iv 1504.06418](https://ar5iv.arxiv.org/html/1504.06418)). Bonito–Demlow, yüksek mertebede θ'nın kümeden bağımsız seçilebileceğini gösterir, ancak ince başlangıç ağı şartı yine durur ([arXiv 1508.06265](https://arxiv.org/abs/1508.06265)).

Pratikteki ayar Carstensen–Gedicke'den geliyor. θ = 1 (düzgün inceltme) alt-optimal hız verir ve hata θ = 0,4'e kadar azalır. **θ ≤ 0,2'de ise algoritma o kadar az eleman işaretler ki önceki seviyenin değerini zaman zaman bir sonraki seviye için kabul eder ve yakınsama grafiği merdivene döner.** Yazarların varsayılanı θ = 0,5 ([Carstensen–Gedicke 2012](https://ins.uni-bonn.de/media/public/publication-media/CG_2012.pdf?pk=1445)). Bu "merdiven", bizdeki sahte yakınsamanın hafif bir biçimidir. Bizim yeniden ağ örme yolumuz işaretleme yapmaz, her elemana bir hedef boyut verir. Bu, teorideki "eşdağılım" sınıfına girer: yakınsama için yeterlidir ama **yalnızca istenen inceltme gerçekten uygulanırsa**.

### Bisection iç içe uzay verir, yeniden ağ örme CAD'i korur; ikisinin de bedeli var

**Teori üç şey ister: iç içe uyumlu uzaylar, düzgün şekil düzenliliği ve sınırlı ek maliyetli bir kapanış (closure).** Kapanış, `#T_ℓ − #T₀ ≤ C Σ #M_k` bağıntısıdır. Stevenson bunu n boyuta genelleştirdi ([Math. Comp. 2008](https://www.ams.org/mcom/2008-77-261/S0025-5718-07-01959-X/)). Arnold–Mukherjee–Pouly tetrahedral bisection'ın sonlu sayıda benzerlik sınıfı ürettiğini, uyumluluk için gereken ek inceltmenin de sınırlı kaldığını gösterdi ([SIAM J. Sci. Comput. 2000](https://epubs.siam.org/doi/10.1137/S1064827597323373)). Red-green inceltme genelde iç içe değildir.

Metrik tabanlı yeniden ağ örme için bilinen tek yakınsama kanıtı Carson–Allmaras–Galbraith–Darmofal 2022'de. O da bir kaynak problemi için ve iç içelik yerine bir optimizasyon argümanına dayanıyor ([MIT DSpace](https://dspace.mit.edu/handle/1721.1/165659)). Özdeğer problemi ya da H(curl) için iç içe olmayan yeniden ağ örmenin yakınsama teorisi yok.

**Eğri sınırlarda iç içelik zaten bozulur.** Sınır kenarının orta noktası CAD'e taşındığında Ω_h değişir ve V_ℓ ⊄ V_{ℓ+1} olur. Dörfler–Rumpf, sınır noktalarını gerçek sınıra koyup tahmin ediciye bir sınır yaklaşım hatası ekler ([Math. Comp. 1998](https://www.ams.org/mcom/1998-67-224/S0025-5718-98-00993-4/)). Yüksek mertebeli parametrik geometride ise yalnızca koşullu daralma vardır: geometri, PDE'den en az onun kadar iyi çözülmelidir, ve hız iki hızdan zayıf olanıyla belirlenir ([Bonito vd., arXiv 1511.05019](https://arxiv.org/abs/1511.05019)). Eğri 3B H(curl) özdeğer problemi için böyle bir kanıt bulunamadı.

Kodlar iki farklı yol izliyor:

- **NGSolve:** `mesh.Refine()` Arnold–Mukherjee tipi işaretli-kenar bisection yapar ve yeni sınır noktalarını `PointBetween` ile CAD'e taşır ([netgen bisect.cpp](https://github.com/NGSolve/netgen/blob/4cd13652d00bd5ac889c19b34dc0ed630c6fb0f7/libsrc/meshing/bisect.cpp)). Ağ eğriyse eğri elemanları aynı mertebede kendiliğinden yeniden kurar ([nginterface_v2.cpp](https://github.com/NGSolve/netgen/blob/4cd13652d00bd5ac889c19b34dc0ed630c6fb0f7/libsrc/interface/nginterface_v2.cpp#L1114-L1135)).
- **MFEM ve Palace:** Çocuk elemanlar ebeveynin interpolasyonlu geometrisini miras alır, CAD'e geri izdüşüm yapılmaz ([mfem mesh.cpp](https://github.com/mfem/mfem/blob/30bf1c91e1909a600d66f1194b4a326a724e653d/mesh/mesh.cpp#L10208-L10221)). Bu çıkarıma göre Palace'ta geometri hatası giriş ağının çözünürlüğünde donar.

Bizim yeniden ağ örmemiz her seviyede CAD'den ağ örüp mertebe 3'e eğriliyor. Bonito vd.'nin "geometri en az alan kadar çözülmeli" şartı bu yolla kendiliğinden sağlanıyor. Bu, yeniden ağ örmenin savunulabilir yanı. Omega3P'nin RPI ağ değiştirme paketi de bir boyut alanını, eğri sınırları "doğru yaklaşımla" koruyarak gerçekliyor ([Ge vd., CEFC 2004](https://www.osti.gov/servlets/purl/827003)).

Bisection'ın bizdeki bedeli ölçülmüş durumda. Borulu pillbox'ta 8109 tetin 200'ü işaretlenince 29k tetlik bir ağ çıktı (işaretli tet başına 105 yeni tet) ve sandbox'ın belleği iki kez tükendi ([docs/30 §4](../docs/30_ADAPTIVE_LABELS.md)).

### Teori: optimal hız, çift mertebeli özdeğer, ama yalnız çokyüzlü alan ve basit özdeğer için

Boffi–Gastaldi'nin Teorem 3.2'si şunu söyler: başlangıç ağı ve θ yeterince küçükse, BGRS göstergesi, Dörfler ve "en küçük uygun inceltme" ile `δ(W, W_ℓ) ≤ C(#T_ℓ − #T₀)^{−s}` olur ve özdeğer hatası bunun karesiyle azalır. Kanıtın üç parçası ayrık güvenilirlik, yarı-diklik (quasi-orthogonality) ve daralmadır. Yazarlar ayrık güvenilirliği "en zor kısım" olarak niteliyor. Varsayımlar: alan çokyüzlü ve basit bağlantılı, özdeğer basit. Küme durumunun BGGG'deki argümanlarla geldiği söyleniyor ama 3B için ayrıca yazılmamış ([Boffi–Gastaldi 2019](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)).

Spektral doğruluğun ön şartı ayrık kompaktlıktır (discrete compactness): sahte modsuz özdeğer yakınsaması için hem yeterli hem gerekli ([Boffi–Costabel–Dauge–Demkowicz](https://perso.univ-rennes1.fr/monique.dauge/publis/BoCoDaDe05.pdf)).

Sayısal tablo net. Re-entrant kenarlı Fichera alanında düzgün ağlar `|λ − λ_h| ~ N^{-0,44…-0,45}` verirken, uyarlamalı ağlar optimal N^{-2/3}'ü yakalıyor (uydurulmuş eğimler −0,66 ile −0,79) ([BGRS 2017](https://arxiv.org/pdf/1602.00675), [BGRS 2019](https://ci2ma.udec.cl/pdf/pre-publicaciones2/2017/pp17-15.pdf)). Yani uyarlamalılığın kazancı tekilliğin olduğu yerde büyük. Bizim fillet'li, pürüzsüz kavitelerimizde bu kazanç daha küçük olmalı. Depodaki analitik pillbox ölçümü de bunu doğruluyor: düzgün inceltme orada "yaklaşık optimal" ([docs/30 §2](../docs/30_ADAPTIVE_LABELS.md)).

**Kümeler ve dejenere modlar.** Göstergeyi kümenin yalnızca bir alt kümesinden hesaplamak "yakınsamanın dejenerasyonuna" yol açar ([Boffi–Gastaldi, arXiv 1804.02377](https://arxiv.org/pdf/1804.02377)). Çare, işaretlemeyi küme toplamı `Σ_{j∈J} η²_{ℓ,j}` üzerinden yapmaktır ([BGGG](https://ar5iv.arxiv.org/html/1504.06418)). Yakınsama da dönmeye karşı değişmez büyüklüklerle ölçülür: küme özdeğerlerinin toplamı ve yoğunluk matrisi (spektral izdüşüm) ([Cancès vd.](https://www.ams.org/journals/mcom/2020-89-326/S0025-5718-2020-03549-2)). BGGG ayrıca küme dışındaki ayrık özdeğerlerin kümeden ayrık kalmasını şart koşar. Bu, ince başlangıç ağının pratikteki anlamıdır: mod sırası ve sayısı başlangıçta doğru olmalı.

### p ve hp: 3B Maxwell özdeğeri için kanıtlı hp-AFEM yok

Kanıtlı ve pratik sonuçlar şöyle dağılıyor:

- hp-FEM'in Maxwell özdeğerlerinde üstel yakınsaması yalnızca 2B çokgenlerde kanıtlı ([Costabel–Dauge–Schwab](https://costabel.perso.math.cnrs.fr/publis/CoDaSchwab04.pdf)).
- Kenar elemanlarında buna ek olarak ayrık kompaktlık hp ağlarda gösterilmeli; bu, asılı düğümlü dikdörtgenler için yapıldı ([BCDD](https://perso.univ-rennes1.fr/monique.dauge/publis/BoCoDaDe05.pdf)).
- Kanıtlı hp-AFEM algoritmaları 1B/2B kaynak problemleri içindir ve bir kabalaştırma adımı gerektirir ([Canuto–Nochetto–Stevenson–Verani](https://mox.polimi.it/reports-and-books/publication-results/?id=539)).
- Pratik hp özdeğer uyarlaması, h veya p kararını bir düzgünlük tahminiyle verir ve üstel hız raporlar ([Giani–Hall, DG](https://nottingham-repository.worktribe.com/output/1010717/an-a-posteriori-error-estimator-for-hp-adaptive-discontinuous-galerkin-methods-for-elliptic-eigenvalue-problems)). Harmon–Notaroš, 2B Maxwell'de hedefe yönelik hp ile tekil özfonksiyonlarda bile çok yüksek doğruluğa ulaşıyor ve analizin 3B'ye "doğrudan" geçtiğini söylüyor ([IEEE TAP 2022](https://atmos.colostate.edu/~notaros/Papers/TAP_Adjoints_Benchmark.pdf)).

Üretimde Omega3P 6. mertebeye kadar bazlar sunuyor, ama kendi belgesi "çoğu uygulama için 2. mertebe yeterli" diyor ([Ko vd., SRF2005](https://proceedings.jacow.org/SRF2005/papers/tha05.pdf)). CST'de çözücü mertebesi 1–3 arasında ([arXiv 2001.09686](https://arxiv.org/pdf/2001.09686)). Otomatik p-uyarlamalı bir üretim kodu bulunamadı.

## Frekans ikinci, duvar büyüklükleri birinci mertebede yakınsar: hedefe yönelik tahmin

### Özdeğer için DWR'nin dual çözümü özfonksiyonun kendisi

İkili ağırlıklı artık (dual-weighted residual, DWR) yönteminde öz-eşlenik (self-adjoint) basit bir özdeğer için dual problem ileri problemle aynıdır. L2-normlu özfonksiyonlarla geçerli özdeşlik şudur ([Harmon–Notaroš](https://atmos.colostate.edu/~notaros/Papers/TAP_Adjoints_Benchmark.pdf)):

`e_λ(1 − σ) = a(u_h, u − ψ_h) − λ_h m(u_h, u − ψ_h)`, burada `σ = ½‖u − u_h‖²`.

Eleman bazında kısmi integrasyon, `u − Π_h u` ile ağırlıklanmış bir eleman artığı (curl curl u_h − λ_h u_h) ve curl'ün teğetsel sıçramasını verir. Ağırlık için bir mertebe zenginleştirilmiş (p+1) çözüm kullanılır, çünkü u_h'nin kendisi Galerkin dikliğinden dolayı sıfır verir. İşaretli toplam λ_h için bir düzeltme de sağlar. Heuveline–Rannacher'in genel çerçevesi bu fikri doğrusal olmayan varyasyonel denklemlere gömüyor ([Gedicke tezi](https://edoc.hu-berlin.de/bitstreams/7235eacd-fc73-4929-a3d7-7cb5f2a57bc9/download), [Zschiedrich vd.](https://arxiv.org/pdf/physics/0702128)). Rannacher–Westenberger–Wollner da ayrıklaştırma hatasını Krylov çözücünün cebirsel hatasıyla dengeliyor ([J. Numer. Math. 2010](https://doi.org/10.1515/jnum.2010.015)).

Bizim hacim terimimiz, enerji özdeşliği `λ_h − λ = ‖curl e‖² − λ‖e‖²`'nin işaretsiz ve ucuz bir vekili. Ritz üst sınır değişmezi (θ_k ≥ λ_h,k ≥ λ_k) altında özdeğer hedeflerinin işareti zaten biliniyor: hepsi pozitif.

### Q0, R/Q ve tepe alanlar için dual problem kaydırılmış tek bir doğrusal çözüm, ama düzenlileştirme ister

Basit bir öz-çiftin J(u) fonksiyoneli için normalizasyonlu Lagrange fonksiyonundan şu dual problem çıkıyor:

`(K − λM) z = J′(u) − (J′(u)u) M u`, `z ⊥_M u`.

Bu, notlardaki bir türetmedir. Zschiedrich vd.'nin formuyla ve onların aktardığı Heuveline–Rannacher ile tutarlı, ama birincil kaynakta doğrulanmadı. Q0, G, R/Q ve tepe/Eacc oranları özvektörde 0. dereceden homojen olduğu için `J′(u)u = 0` olur. Sağ taraf kendiliğinden uyumlu hale gelir ve dual, `z ⊥_M u` koşuluyla `(K − λM) z = J′(u)` problemine indirgenir.

Bu sistemin özellikleri:

- Birinciden yüksek her mod için belirsiz (indefinite) bir sistemdir. CG çalışmaz; MINRES/GMRES ya da kenarlı (bordered) bir sistem gerekir.
- Her modun kayması farklı olduğu için K mod için K ayrı dual çözümü gerekir.
- Duvar kaybı fonksiyoneli H(curl) üzerinde sınırlı değildir. Zschiedrich vd. bunu sınır integralini sınır çevresinde bir hacim katmanına genişleterek düzenlileştirdi. Kayıp tabanlı uyarlama, sanal kısmı (kayıpları) enerji tabanlı uyarlamadan "özellikle düşük doğruluk taleplerinde" daha hızlı yakınsattı. Enerji tahmin edicisi ilk üç adımda sanal kısmı hiç iyileştirmedi, asimptotikte ise iki yöntem birbirine yaklaştı ([Zschiedrich vd., arXiv physics/0702128](https://arxiv.org/pdf/physics/0702128)).

Hızlandırıcı pratiği aynı yönde. SLAC/RPI, Omega3P'ye AMR'yi "duvar kaybı hesabındaki yakınsamayı iyileştirmek için" ekledi. Gösterge önce depolanan enerjinin gradyanıydı, sonra "E ve H alanları üzerinde dönüşümlü ZZ" oldu. Dar bir yüzey bölgesindeki yüksek duvar kaybının elle inceltmeyle kötü yakınsadığını not ettiler ([Ge vd., SLAC-PUB-10449](https://s3.cern.ch/inspire-prod-files-6/6f1ab226c1f92f358546c3b24013a673)). Bizim duvar terimimiz bu sezgisel geleneğin içinde: normalize edilmiş, kare alınmış duvar kurtarma hatası. Bir DWR ağırlığı değil, ama analitik pillbox'ta eşit DOF'ta Q0 ve Bpk hatasını **3× düşürdüğü ölçüldü** ([docs/30 §2](../docs/30_ADAPTIVE_LABELS.md)).

**Tepe alanlar (Epk, Bpk) için hiçbir a posteriori teori yok.** Özdeğer problemleri için tek titiz L∞ çalışması 2025 tarihli bir ön baskı ve skaler eliptik problemle sınırlı ([Li–Liang–Xu, arXiv 2511.06815](https://arxiv.org/pdf/2511.06815)). Pratik ampirik:

- TESLA çalışmalarında "Epk/Eacc ağa güçlü biçimde bağlı" bulundu ([Zu & Chen, PAC1993](https://proceedings.jacow.org/p93/PDF/PAC1993_1095.PDF)).
- ANSYS deneyiminde "daha sık ağ daha iyi sonuç garantilemiyor" ([Zaplatin, SRF2005](https://proceedings.jacow.org/SRF2005/papers/TUP42.pdf)).
- Kodlar arası uyum Bpk'da %1,3, Epk'da %8,2 ([Juntong & Jones](https://arxiv.org/pdf/0910.1305)).

Maksimum, ağla birlikte yer değiştiren türevlenemez bir fonksiyoneldir. Hedefe yönelik bir çözüm için iki aday var: ya düzgün bir vekil (duvarda |H_t|'nin büyük p'li L^p normu) ya da her döngüde yeniden bulunan maksimum noktası etrafında düzenlileştirilmiş bir nokta fonksiyoneli. İkisi de kaviteler için yayımlanmamış.

**Çok hedefli birleştirme.** Standart yol tek bir birleşik hedeftir: `J_c = Σ sign(J_i(ũ) − J_i(u_h)) J_i / |J_i(ũ)|`. Bu ağırlıklar her hedefi göreli hatada denetler ve işaretler hata iptalini önler ([Endtmayer, arXiv 2405.18567](https://arxiv.org/pdf/2405.18567)). 2026'daki bir analiz, işaret bilgisi olmadan dualler arasında döngüyle optimal hız kanıtlıyor, ama yalnızca doğrusal eliptik problem ve doğrusal hedefler için ([Becker vd., arXiv 2601.01965](https://arxiv.org/pdf/2601.01965)). Bizim 1/λ_k hacim ağırlığımız 1/|J_i| kuralının özdeğer hali. Toleransı farklı hedefler için doğal genelleme `w_i = 1/(|J_i|·tol_i)`.

### Depodaki ölçümler hızları veriyor ve Cauchy ölçütünün neyi gizlediğini gösteriyor

Analitik pillbox'ta bir düzgün inceltme adımı (992 → 7936 tet, 28k → 210k DOF, DOF oranı 7,5) her büyüklüğün gözlenen hızını veriyor ([docs/30 §2](../docs/30_ADAPTIVE_LABELS.md)). Aşağıda s, `hata ~ N^{-s}` üssüdür. Beklenen değerler standart a priori argümandan gelen bir çıkarımdır, kaynaklı bir teorem değil.

| büyüklük | 28k DOF | 210k DOF | gözlenen s | p = 3 için beklenen |
|---|---|---|---|---|
| f (6 modun en kötüsü) | 1,1e-5 | 2,2e-7 | **1,94** | 2p/3 = 2 |
| Q0 (TM010) | 1,9e-4 | 7,5e-6 | **1,60** | p/3 = 1 ile 2p/3 arası |
| Bpk/Eacc | 5,0e-4 | 1,8e-4 | **0,51** | ≤ p/3, örnekleme etkisiyle |

Bu hızlar mevcut durdurma kuralının zayıflığını sayıya döküyor. Asimptotik rejimde e_ℓ = C N_ℓ^{-s} ve g = N_ℓ/N_{ℓ−1} ise, ardışık fark `Δ = e_ℓ (g^s − 1)` olur. Buradan **ince seviyenin hatası `e_ℓ ≈ Δ/(g^s − 1)`** çıkar. Bu, Richardson tipi standart bir çıkarımdır.

**g = 1,05'te** (`--strict_stop`'un `min_growth` eşiği):

- f için e ≈ 10Δ, Q0 için ≈ 12Δ, Bpk için ≈ 40Δ.
- Dolayısıyla tol_q = 1e-3 altında kalan bir Δq, Q0 hatasının ~1,2e-2 olmasına izin verir. Bu, docs/30'daki %0,2 hedefinin **altı katı**.

**g = 2'de:**

- f için e ≈ Δ/3, Q0 için ≈ 0,5Δ, Bpk için ≈ 2,4Δ.

**g → 1'de** e_ℓ → ∞ olur. Yani bu formül sahte yakınsamayı ayrıca bir koşul eklemeye gerek kalmadan dışlıyor.

Bir yan bulgu: borulu pillbox'ta Bpk'nın en ince seviyeye göre farkı, inceltme ilerledikçe 2,0e-2'den 3,9e-2'ye *büyüdü*. Referansın kendisi de 400k DOF'ta Bpk için yakınsamamıştı ([docs/30 §4](../docs/30_ADAPTIVE_LABELS.md)). Tepe alanlar için monoton olmayan davranış literatürdeki gibi bizde de var.

## Üretim kodları teoriden basit, ama her biri bir büyüme ya da tahmin güvencesi taşıyor

| Kod | Gösterge | İşaretleme | İnceltme / geometri | Durdurma | Büyüme güvencesi |
|---|---|---|---|---|---|
| Palace | Nicaise ZZ (ND/RT izdüşümü), enerjiyle normalize, modlar üzerinden RMS ([errorindicator.cpp](https://github.com/awslabs/palace/blob/8bd206f7e10e5a2ec93a2483514894d14c8a5102/palace/fem/errorindicator.cpp#L10-L47)) | Dörfler `UpdateFraction` 0,7 | uyumsuz (`MaxNCLevels` 1) ya da uyumlu simpleks; ebeveyn geometrisi interpolasyonla | `err < Tol` (1e-2), `MaxIts`, `MaxSize` ([basesolver.cpp](https://github.com/awslabs/palace/blob/8bd206f7e10e5a2ec93a2483514894d14c8a5102/palace/drivers/basesolver.cpp#L215-L387)) | Dörfler her geçişte en az bir eleman işaretler; en az büyüme kontrolü yok |
| MFEM | ZZ / L2-ZZ (`CurlCurlIntegrator` ile H(curl)) | eşik: 0,5 × en büyük hata | varsayılan uyumlu | `total_err_goal` (varsayılan 0), eleman sınırı, işaretli eleman yoksa ([mesh_operators.cpp](https://github.com/mfem/mfem/blob/master/mesh/mesh_operators.cpp)) | yok; Maxwell özdeğer örneği ex13p yalnızca düzgün inceltme yapıyor ([ex13p.cpp](https://github.com/mfem/mfem/blob/30bf1c91e1909a600d66f1194b4a326a724e653d/examples/ex13p.cpp)) |
| NGSolve | öğreticide ZZ (Poisson) | 0,25 × en büyük hata | işaretli-kenar bisection, CAD izdüşümü, eğrilik yeniden kuruluyor | yalnızca `ndof` bütçesi ([adaptivity.ipynb](https://github.com/NGSolve/ngsolve/blob/d242b9b4bbad5617676171d8daf58e13d64fb1ad/docs/i-tutorials/unit-1.6-adaptivity/adaptivity.ipynb)) | yok; Maxwell EVP öğreticisi a priori `RefineHP` kullanıyor |
| deal.II | Kelly (yalnız Poisson) | `fixed_number` / `fixed_fraction` (Dörfler) ([GridRefinement](https://dealii.org/current/doxygen/deal.II/namespaceGridRefinement.html)) | asılı düğümler | kullanıcı döngüsü | yok |
| Omega3P | E, H ya da gradyanlarında ZZ | eşik + eleman sayısını en aza indiren boyut alanı | RPI ağ değiştirme, eğri sınırlar CAD'e uyuyor | toplam hata δ = Σ η_i < değer ([CEFC 2004](https://www.osti.gov/servlets/purl/827003)) | belgelenmemiş |
| CST | yayımlanmamış | yayımlanmamış | h-tet; 2024'te ilk geçişlerde "daha güçlü inceltme" seçeneği ([GoEngineer](https://www.goengineer.com/blog/simulia-cst-studio-suite-2024-whats-new)) | ardışık geçişlerde en büyük ΔF, en az ve en çok geçiş sayısı ([CAS 2023](https://indico.cern.ch/event/1294829/contributions/5442426/attachments/2698007/4683942/CAS2023_simulations_wCover.pdf)) | yayımlanmamış |
| HFSS | tetler "Maxwell denklemlerine uygunluğa göre derecelendiriliyor" | en kötü tetlerin %30'u ([Ansys GS 2020R2](https://innovationspace.ansys.com/courses/wp-content/uploads/2021/07/HFSS_GS_2020R2_EN_LE3_FEM_MeshSoln.pdf)) | tet inceltme | mod başına ΔF (%), `MinimumPasses`, `MinimumConvergedPasses`, `MaximumPasses` ([PyAEDT](https://github.com/ansys/pyaedt/blob/main/src/ansys/aedt/core/modules/setup_templates.py)) | **%30 zorunlu** |
| COMSOL | tüm özmodların ağırlıklı toplamı (ağırlık 1) | "Rough global minimum" ×1,7 ya da 0,5 kesir | en uzun kenar bisection'ı ya da "Rebuild mesh" | uyarlama sayısı (3B varsayılanı 1; hedefe yönelik "Auto" ile 10) ([COMSOL 6.3](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_solver.36.157.html)) | **×1,7** |
| Bizim | VectorH1(p) ortalamalı ZZ + duvar, modlar üzerinden toplam | yok (eşdağılım boyut alanı) | `RestrictH` birikimli netgen yeniden ağ örme, CAD'den, eğrilik 3 | Δf < 1e-6 ve Δq < 1e-3 (kümede değişmez büyüklükler) | yalnızca *tahmin edilen* eleman sayısı |

Tabloyu okumanın yolu şu. Ardışık farkla duran iki kod (HFSS, CST) ile tahmin edici eşiğiyle duranlar (Palace, Omega3P) farklı güvencelere dayanıyor. HFSS, farkı yalnızca her geçişte %30 inceltmeyi zorladıktan sonra anlamlı sayıyor. Ayrıca "en az yakınsamış geçiş" sayacı var, ama PyAEDT şablonundaki varsayılanı 1. Palace farka hiç bakmıyor, enerjiyle normalize tahmin `Tol` altına inince duruyor. Dörfler'in her geçişte en az bir eleman işaretlemesi büyümeyi sağlıyor, ama asgari bir büyüme oranını garanti etmiyor.

Bizim döngümüz CST tipi bir fark ölçütünü, büyümesi garanti edilmemiş bir yeniden ağ örmeyle birleştiriyor. Sahte yakınsama açısından bu en kırılgan birleşim. CST'nin 2024'te "ilk geçişlerde daha güçlü inceltme" seçeneğini eklemesi de, erken geçişlerin yeterince büyümediği sorununun satıcı tarafında da görüldüğünü düşündürüyor. CST'nin uyarlamalı ağ örmesinin geri kalanı için önceki rapora bakılabilir; orada yayımlanmış ölçütün yalnızca ΔF olduğu sonucuna varılmıştı.

Hiçbir kod, eleman sayısı neredeyse hiç değişmeyen bir geçişi reddetmiyor ya da uzatmıyor. Bu kontrol (`--strict_stop` kural 1) üretim pratiğinin ötesinde, ama literatürdeki "işaretli eleman inceltilmeli" şartının doğrudan karşılığı.

## Depodaki döngü: tahmin edici sağlam, durdurma ve inceltme garantisi kırık

Varsayılan döngü şu adımlardan oluşuyor ([adaptive.py:185-229](../src/data_gen/adaptive.py)):

1. Her seviyede sıfırdan bir HCurl p3 çözümü (shift-invert ARPACK, `tol=1e-10`, tam gradyan izdüşümü, K+1 mod).
2. Kümede değişmez QoI tablosu.
3. Hacim ve duvar ZZ göstergesi.
4. Durdurma testi: iki ardışık seviye arasında `df < tol_f` ve `dq < tol_q`.
5. Birikimli `RestrictH` kısıtlarıyla CAD'den yeni bir netgen ağı.

Literatürle ve üretim kodlarıyla karşılaştırınca doğru yapılanlar ve kopukluklar şöyle ayrışıyor.

**Doğru yapılanlar** literatürle örtüşüyor:

- Gösterge K mod üzerinden toplanıyor (BGGG küme toplamı).
- Durdurma, dönmeye karşı değişmez büyüklüklere bakıyor: küme ortalaması f, ortalama Q0 ve G, Σ R/Q, en büyük tepe değerleri ([adaptive.py:158-174](../src/data_gen/adaptive.py)). Bu, Cancès vd.'nin değişmezlerinin pratik karşılığı.
- 1/λ_k normalizasyonu Endtmayer'in 1/|J| kuralına karşılık geliyor.
- Duvar terimi SLAC'ın E+H ZZ pratiğine karşılık geliyor.
- Her seviyede CAD'den ağ örülüp eğrilik 3 uygulanıyor. Bu, Bonito vd.'nin "geometri en az alan kadar çözülmeli" şartını sağlıyor.
- Keskin kenar kalan geometrilerde tepe değerler durdurma ölçütünden çıkarılıyor (`peaks = sharp_edges == 0`). Bu, tekil kenarda tepe alanın ağla yakınsamamasıyla tutarlı.

**Kopukluklar** dört yerde toplanıyor.

*Birincisi: işaretleme yok ve inceltme garanti değil.* Boyut alanı `h·(ρ·mean(η²)/η_T²)^{1/(2p)}`, ρ = 0,25 ve [0,3; 1]·h kırpmasıyla ılımlı bir istek üretiyor. Ortalama hatalı bir eleman için istenen boyut yalnızca %21 küçük (0,25^{1/6} ≈ 0,79). Bütçe yalnızca tahmin edilen `ne_next = Σ(h/h_new)³` ile kontrol ediliyor. Netgen'in döndürdüğü ağ hiç istekle karşılaştırılmıyor ([adaptive.py:214-229](../src/data_gen/adaptive.py)). Sonuç ölçülmüş durumda:

- Kalibrasyon pillbox'ında tet sayıları 96 → 88 → 91 → 91 → 91 gitti. Döngü 2.864 DOF'ta `converged` dedi. O sırada kendi tahmini 1,6e-3'tü, yani tol_f'nin 1600 katı. Gerçek hatalar TE111 +1,8e-4, TM110 +1,2e-3 ([docs/31 §6 madde 14](../docs/31_SESSION_HANDOFF.md); `afc80c3` docs/30 §4a).
- `validate_field_labels.py --max_ndof 60000` çalışmasında son seviyeler 1324 → 1232 → 1257 tet gitti ve yine "yakınsadı".

Literatürün dilinde bu, "işaretli eleman bir sonraki ağda kalamaz" (`M_ℓ ∩ T_{ℓ+1} = ∅`) şartının çiğnenmesi ([Boffi–Gastaldi 2019](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)).

*İkincisi: durdurma yalnızca Cauchy tipi bir fark.* `est_rel_f` hesaplanıp kaydediliyor ama durdurmada kullanılmıyor ([adaptive.py:212](../src/data_gen/adaptive.py)). Bir önceki bölümün hesabına göre, küçük büyümede bu fark hatayı bir mertebe küçük gösteriyor.

*Üçüncüsü: tahmin edicinin kalibrasyonu belgelendiği gibi değil.* docs/30 "her seviyede 1,5× içinde" diyor. Ama kendi üçüncü çifti 2,1e-6 / 6,1e-7 = **3,4×**. Üstelik `est_rel_f` modlar üzerinden *ortalama* alırken, doğrulama gerçek hatanın *en kötü* moddaki değerine bakıyor. Bu iki sayı aynı şeyi ölçmüyor.

*Dördüncüsü: kümeler eksik kapanıyor.* Değişmezler yalnızca ilk K modda hesaplanıyor. K. mod dejenere bir çiftin bir üyesiyse Σ R/Q ve en büyük tepe değerleri dönmeye karşı değişmez olmaktan çıkıyor. Ayrıca kaba bir seviyede bir çift 1e-3'ten fazla ayrılırsa küme sayısı değişiyor, `df` atlanıyor ve o seviye duramıyor ([adaptive.py:198](../src/data_gen/adaptive.py)).

**Verimlilikle ilgili gözlemler.** Analitik pillbox'ta 210k DOF'luk düzgün inceltme, 157k DOF'luk uyarlamalı sonucu Epk dışında her sütunda geçiyor:

| | f | Q0 | R/Q | Bpk |
|---|---|---|---|---|
| düzgün, 210k DOF | 2,2e-7 | 7,5e-6 | 8,1e-7 | 1,8e-4 |
| uyarlamalı, 157k DOF | 6,1e-7 | 3,4e-5 | 1,2e-5 | 3,9e-4 |

Pürüzsüz geometride bu beklenen bir sonuç. Uyarlamalılığın kazancı borulu pillbox'ta ve keskin portlu eliptikte görülüyor: aynı doğruluk için 2–6× daha az eleman ([docs/30 §4](../docs/30_ADAPTIVE_LABELS.md)). Bir de doğrulama betiği yalnızca son seviyedeki hataya bakıyor. PASS hedefleri durdurma toleranslarından 10× (f) ve 2× (Q0, G, R/Q) gevşek, ve betik her durma nedenini kabul ediyor ([validate_field_labels.py:26-85](../scripts/validate_field_labels.py)). Yani bu betik durdurma kuralını sınamıyor.

**Maliyet.** Çözücü bir örneğin yaklaşık %90'ını alıyor. Borulu pillbox'ta 235k DOF'luk seviye 98 s, 430k DOF'luk seviye 277 s sürüyor; gösterge seviye başına 11–18 s ([docs/31](../docs/31_SESSION_HANDOFF.md)). Cebirsel tolerans her seviyede 1e-10'a sabit. Oysa Carstensen–Gedicke, özdeğer çözücüsünü ayrıklaştırma tahminine bağlı olarak durdurmayı öneriyor: `|λ_ℓ − λ̃_ℓ| + |||u_ℓ − ũ_ℓ|||² ≤ ω η²_ℓ`, deneylerinde ω = 10⁻³ ([CG 2012](https://ins.uni-bonn.de/media/public/publication-media/CG_2012.pdf?pk=1445)).

**Etiket doğruluğu ile model tabanı arasındaki uçurum.** Etiket doğruluğu (f'de 1e-6) ile modelin kendi frekans tabanı (1e-4…1e-2) arasında 2–4 mertebe var ([docs/30 §5](../docs/30_ADAPTIVE_LABELS.md)).

## Aşamalı plan: önce ölç, sonra durdurmayı ve inceltmeyi garanti et, en son hedefe yönelik ol

### Ortak doğrulama protokolü

Her aşama aynı dört referansla doğrulanır. Sonuç her seviye için bir tabloyla raporlanır (tet, DOF, tahmin, gerçek hata, Δ, durma nedeni, süre).

1. **Analitik pillbox.** `python scripts/validate_field_labels.py --max_ndof 300000` PASS vermeli. Ek olarak, durma nedeni `converged` iken son hatanın hedef içinde kaldığı ayrıca kontrol edilir.
2. **Kalibrasyon kutusu (`calib_box`).** 100 × 80 × 60 mm'lik düz duvarlı analitik PEC kutusu. Geometri hatası sıfır olduğu için tahmin edicinin etkinliğini alan hatasından ayrı ölçer.
3. **Kaba başlangıçlı kalibrasyon pillbox'ı (`calib_pillbox`, `--mode calibration`).** 96 tetlik sahte yakınsama vakasını yeniden üretir.
4. **Borulu pillbox (`pillbox_pipes`, docs/34 §8'deki test geometrisi 24).** Referans iki yoldan kurulur: düzgün inceltmeyle ve en ince ağda bir p4 çözümüyle. p4/p3 DOF oranı kabaca 1,7 olarak tahmin ediliyor. Bpk referansı 400k DOF'ta yakınsamadığı için bu iş TRUBA'da yapılmalı.

Komut: `python scripts/cst_calibration_set.py --out <dir> --cases calib_box calib_pillbox pillbox_pipes`. Birim testleri `tests/test_field_adaptive.py` içinde. Verimlilik iddiaları DOF'a ve duvar saati süresine karşı hata eğrileriyle ve aynı DOF'taki düzgün inceltmeyle kıyaslanarak verilir. Pillbox'ta düzgün inceltme zaten yaklaşık optimal olduğundan, uyarlamalılığın kazancı yalnızca borulu pillbox ve eliptik üzerinden iddia edilir.

| Aşama | Değişiklik | Etiketler değişir mi | Sahip onayı |
|---|---|---|---|
| 0 | Seviye tarihçesi, mod bazında tahmin, eski veride bayrak, doğrulamanın sıkılaştırılması | hayır (H5'e yalnızca öznitelik eklenir) | bilgilendirme |
| 1 | Büyümeye ölçeklenmiş fark + tahmin edici sağlaması ile `converged` | evet (sahte ya da erken yakınsayan örnekler) | **gerekli** |
| 2 | Yeniden ağ örmenin gerçekten incelttiğinin kontrolü, isteği sıkılaştırma, son çare bisection, hedef büyüme | evet | **gerekli** |
| 3 | Dörfler kümesine dayalı boyut alanı | evet | **gerekli** |
| 4 | Eleman artığı terimi, kümenin K sınırında tamamlanması, uyarlamalı küme eşiği | evet | **gerekli** |
| 5 | Tepe hedefli inceltme ve hızlandırıcı mod için Q0 DWR prototipi | evet (isteğe bağlı) | **gerekli** |
| 6 | Cebirsel toleransın seviyeye bağlanması | ara ağlar değişir | **gerekli** (düşük öncelik) |

### Aşama 0 — Ölç, kaydet, eski etiketleri işaretle

**Döngüde ne değişir.** Döngünün kararları değişmez, yalnızca ek tanılama yazılır:

- H5'e her seviyenin kaydı yazılır: `ne`, `ndof`, gerçekleşen büyüme `g`, mod bazında tahmin, `df`, `dq` ve büyüklük bazında Δ.
- Tahmin edici, ortalamanın yanında modların en kötüsünü de raporlar.
- `validate_field_labels.py`, durma nedeni ile son hatanın tutarlılığını kontrol eder.
- Docs/30'daki "1,5×" ifadesi ölçülen 1,1–3,4× aralığıyla düzeltilir.
- Mevcut veri kümeleri `afc80c3`'teki filtreyle işaretlenir: `n_tets_label < 1.05 * n_tets_start or est_rel_f > 100 * tol_f`.
- `adapt_stop` dağılımının istatistiği çıkarılır.

**Neden.** Tahmin edici tabanlı her durdurma (Palace, Omega3P, Carstensen–Gedicke) kalibre edilmiş bir etkinlik ister. Literatürdeki tek Maxwell değeri en düşük mertebe içindir (5–6, [BGRS 2019](https://ci2ma.udec.cl/pdf/pre-publicaciones2/2017/pp17-15.pdf)). Bizim p3 değerimiz yalnızca bir pillbox dizisinden geliyor. Ayrıca sahte yakınsamanın veri kümelerinde ne kadar yaygın olduğuna dair hiçbir istatistik yok. Aşama 1'in eşikleri bu ölçümlere dayanmalı.

**Beklenen etki.** Doğrulukta değişiklik yok, maliyet ihmal edilebilir. Kazanç iki sayı: kaç etiketin yeniden üretilmesi gerektiği ve κ ile s'nin hangi değerlerde seçileceği.

**Doğrulama.** Mevcut testler geçmeli. Dört referans vaka varsayılan kuralla çalıştırılıp bir "önce" tablosu oluşturulur.

**Etiketler.** Değişmez. H5 şemasına öznitelik eklendiği için sahip bilgilendirilir.

### Aşama 1 — `converged` için büyümeye ölçeklenmiş fark ve tahmin edici sağlaması (en yüksek öncelik)

**Döngüde ne değişir.** `converged` için dört koşul birlikte aranır:

- (a) Gerçekleşen DOF büyümesi `g = ndof_ℓ/ndof_{ℓ−1} ≥ g_min`. Öneri g_min = 1,3; `afc80c3`'teki 1,05 çok düşük.
- (b) Büyümeye ölçeklenmiş fark: `Δ_f/(g^{s_f} − 1) ≤ tol_f` ve `Δ_q/(g^{s_q} − 1) ≤ tol_q`. s_f = 2 (p3 için 2p/3; gözlenen 1,94), s_q = 1 (ihtiyatlı; Q0 için gözlenen 1,6). Tepe değerler dahilse onlar için s = 0,5.
- (c) Tahmin edici sağlaması: `est_f,max ≤ κ·tol_f`. κ Aşama 0'dan gelir; başlangıç değeri 10. Bu değer gözlenen 3,4× aşırı tahmini kabul eder, ama 1600×'lik sahte durumu reddeder.
- (d) İnceltmeyle frekansın tol_f'den fazla *artması* "monoton değil" diye işaretlenir. İç içe olmayan ağlarda Ritz monotonluğu garanti değil, ama büyük bir artış ön-asimptotik rejimin ya da geometri değişiminin işaretidir.

`--strict_stop` kural 1, bu kuralın g_min = 1,05 ile sınırlı en küçük halidir. Önerilen yol şu: `afc80c3` isteğe bağlı bayrak olarak ana dala alınır, sonra (a)–(c) aynı bayrağın arkasına eklenir. Varsayılan, sahibin onayından sonra değişir.

**Neden.** Teorik dayanaklar:

- Kanıtlı kurallar tahmin edicide durur ([Carstensen–Gedicke](https://ins.uni-bonn.de/media/public/publication-media/CG_2012.pdf?pk=1445)). Palace (`err < Tol`) ve Omega3P de böyle yapıyor ([basesolver.cpp](https://github.com/awslabs/palace/blob/8bd206f7e10e5a2ec93a2483514894d14c8a5102/palace/drivers/basesolver.cpp#L215-L387); [CEFC 2004](https://www.osti.gov/servlets/purl/827003)).
- Farkla duran HFSS, büyümeyi %30 inceltmeyle zorlar ([Ansys GS 2020R2](https://innovationspace.ansys.com/courses/wp-content/uploads/2021/07/HFSS_GS_2020R2_EN_LE3_FEM_MeshSoln.pdf)).
- Küçük θ'daki "merdiven" etkisi, büyüme olmadan farkın yanıltıcı olduğunun literatürdeki örneği.

(b), bu iki güvenceyi tek bir hata tahminine çeviriyor. g = 1'de tahmin sonsuza gider. g = 1,05'te f için yaklaşık 10, Q0 için yaklaşık 20 çarpanı devreye girer.

**Beklenen etki.** Doğruluk tarafında:

- Sahte yakınsama tanım gereği ortadan kalkar.
- Kalibrasyon pillbox'ında yalnızca kural 1+2 bile sonucu belirgin biçimde değiştiriyor. Varsayılan kural 2.864 DOF'ta durup TM110'u +1,2e-3 hatayla bırakıyor. `strict_stop` ise 96 → 613 → 1626 tet gidip 43k DOF'ta `budget` ile duruyor ve altı frekansın hepsi 7e-6 içinde (`afc80c3` docs/30 §4a).

Maliyet tarafında:

- Yalnızca sahte ya da erken yakınsayan örnekler ek seviye öder. Bir seviyenin maliyeti o boyuttaki çözümdür (borulu pillbox'ta 235k DOF'ta 98 s, 430k'da 277 s). Üst sınır `max_ndof` bütçesidir.
- Gerçek büyümeli (g ≈ 2) durumlarda f için kural bugünkünden *gevşek* olur (Δ_f ≤ 3·tol_f). Bu bazı örneklerde bir seviye kazandırabilir. Q için kural bugünküne denk kalır.

**Doğrulama.**

- `test_strict_stop_does_not_call_a_stalled_remesh_converged` korunur. Bilinen hızla küçülen sentetik bir hata dizisiyle (b)'nin birim testi eklenir.
- `calib_pillbox`, 2,9k DOF'ta `converged` diyemez.
- Analitik pillbox PASS vermeli ve durma nedeni hatayla tutarlı olmalı.
- `calib_box`'ta κ'nın her seviyede geçerli olduğu gösterilir.
- `pillbox_pipes`'ta son hatalar, seviye sayısı ve süre referansla ve varsayılanla karşılaştırılır.

**Etiketler.** Etkilenen örnekler daha ince bir son ağ ve farklı bir model ağı anlık görüntüsü (`keep_model_level`) alır. Sahibin onayı gerekir. Eski veri kümeleri için iki seçenek var: Aşama 0'ın bayrakladığı örnekleri yeniden üretmek ya da onları test ve doğrulama ayrımlarından çıkarmak.

### Aşama 2 — Yeniden ağ örmenin gerçekten inceltmesini garanti et

**Döngüde ne değişir.** Her yeniden ağ örmeden sonra döndürülen ağ istekle karşılaştırılır. İki ölçüye bakılır: gerçekleşen büyüme (`new.ne/mesh.ne`, tahmini `ne_next` ile) ve en büyük η'lı elemanların (Dörfler kümesi) bulunduğu bölgelerde yeni yerel boyutun istenenin 1,1 katı içinde kalan oranı. Yetersizse sırasıyla şu basamaklar denenir:

- (i) İşaretli kümedeki istenen boyutlar yarıya indirilir (ya da ρ küçültülür) ve en çok iki kez yeniden ağ örülür.
- (ii) Ancak bundan sonra `--strict_stop` kural 2 (Dörfler bisection) bütçe ve kapanış büyümesi korumasıyla çalışır.
- (iii) Yine büyüme olmazsa döngü yeni bir `stalled` nedeniyle durur, `converged` ile değil.

Ayrıca ρ, her geçişte tahmini büyüme g_target ≈ 1,7–2 olacak şekilde seçilir; bütçe sınırı korunur. Bu hem kural (b)'nin anlamlı bir g görmesini sağlar hem de seviye sayısını azaltır.

**Neden.** Tüm AFEM kanıtları işaretli elemanın inceltilmesini şart koşar ([Morin–Siebert–Veeser](https://amcaonline.org.ar/ojs/index.php/cmm/article/view/2534); [Boffi–Gastaldi](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)). Üretimde HFSS geçiş başına %30, COMSOL ×1,7 büyümeyi zorluyor ([COMSOL 6.3](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_solver.36.157.html)). CST 2024 ilk geçişler için daha güçlü inceltme ekledi.

Bizim isteğimiz ılımlı (ortalamada 0,79·h), ve netgen kaba ağlarda kendi `curvaturesafety`, `maxh` ve `grading` boyutlandırmasıyla bu isteği emebiliyor. Bu ikinci kısım bir çıkarım; depoda netgen'in `RestrictH`'yi ne sıklıkla uyguladığına dair bir ölçüm yok. Kural 2'yi ilk değil son çare yapmanın gerekçesi, bisection kapanışının ölçülmüş patlaması (işaretli tet başına 105 yeni tet).

**Beklenen etki.** Kural 1 belirtiyi yakalıyor; bu aşama nedeni gideriyor. Böylece boşa giden seviyeler (her biri tam bir çözüm) kalkar. Ek maliyet birkaç ek ağ üretimi. Çözücü örneğin %90'ını aldığına göre bunun küçük olması beklenir, ama ağ üretim süresi ölçülmüş değil; `/measure` ile ölçülmeli.

**Doğrulama.**

- `calib_pillbox` 96 tetten başlayınca her seviye g_min'i geçmeli.
- `validate_field_labels.py --max_ndof 60000`, 1324 → 1232 → 1257 dizisi yerine monoton büyüme göstermeli.
- Takılma (`stalls`) sayısı ve hangi basamağın onu çözdüğü kaydedilir.
- `pillbox_pipes`'ta DOF'a karşı hata eğrisi varsayılan ve düzgün inceltmeyle karşılaştırılır.

**Etiketler.** Ağ dizisi değişir, sahip onayı gerekir. Aşama 1 ile tek bir yeniden etiketleme turunda birleştirilebilir.

### Aşama 3 — Boyut alanını Dörfler kümesine bağla

**Döngüde ne değişir.** Saf eşdağılım yerine Dörfler kümesi M kullanılır (θ = 0,5; ayarlarda zaten var ama yeniden ağ örme yolunda kullanılmıyor). M'deki elemanlar `h·(ρ·mean/η)^{1/2p}` boyutunu [0,3; 0,7]·h aralığına kırpılmış olarak alır, yani en az %30 küçülür. Diğer elemanlar mevcut boyutlarında kısıtlanır.

**Neden.** Hız optimalliği yalnızca Dörfler için kanıtlı ([Stevenson 2008](https://www.ams.org/mcom/2008-77-261/S0025-5718-07-01959-X/); [BGGG](https://ar5iv.arxiv.org/html/1504.06418)). Carstensen–Gedicke en iyi sonuçları θ ≈ 0,4–0,5 aralığında aldı. Palace 0,7 kullanıyor. Eşdağılım yalnızca hızsız yakınsama garantisi veriyor.

**Beklenen etki.** Pürüzsüz olmayan geometrilerde (borulu pillbox, keskin portlu eliptik) aynı hatada daha az DOF beklenir. Pillbox'ta etki nötr ya da hafif olumsuz olabilir, çünkü orada düzgün inceltme yaklaşık optimal. Kazancın büyüklüğü bilinmiyor; ancak hata–DOF eğrileriyle ölçülebilir.

**Doğrulama.** `pillbox_pipes` ve eliptik vakalar, aynı DOF'ta düzgün inceltmeyle ve Aşama 2 sonrasındaki varsayılanla karşılaştırılır. Analitik pillbox PASS vermeli.

**Etiketler.** Değişir, sahip onayı gerekir. Sahip daha az yeniden etiketleme isterse Aşama 2 ile birleştirilebilir.

### Aşama 4 — Tahmin ediciyi BGRS'ye yaklaştır ve kümeleri tamamla

**Döngüde ne değişir.**

- (a) η_vol'a BGRS eleman artığı `h_T²‖E_h − curl curl E_h/λ_h‖²_T` eklenir, isteğe bağlı olarak `h_T²‖div E_h‖²_T` de. Eğri p3 elemanlarda curl curl'ün, yani haritanın ikinci türevlerinin NGSolve'da hesaplanabilirliği önce sınanmalı.
- (b) Mod bazında tahmin ve modların en kötüsü raporlanır.
- (c) K. mod, K+1. modla `cluster_rel` içindeyse (K+1. mod zaten çözülüyor) bu mod da göstergeye ve değişmezlere katılır.
- (d) Küme eşiği tahminle ilişkilendirilir: `rel = max(1e-3, c·est)`. Böylece kaba seviyede bir çiftin ayrılması `df`'yi atlatmaz.

**Neden.**

- BGRS'nin dört terimi kanıtlı bir güvenilirlik verir. Nicaise sınıfı ZZ'de eleman artığı yok ([BGRS 2019](https://ci2ma.udec.cl/pdf/pre-publicaciones2/2017/pp17-15.pdf); [Cai–Cao](https://arxiv.org/pdf/1504.00898)).
- Kümenin bir alt kümesiyle çalışmak yakınsamayı dejenere eder ([Boffi–Gastaldi](https://arxiv.org/pdf/1804.02377)).
- Değişmez ölçüler bütün kümenin üzerinden alınmalı ([Cancès vd.](https://www.ams.org/journals/mcom/2020-89-326/S0025-5718-2020-03549-2)).

**Beklenen etki.** Etkinlik indeksi seviyeler boyunca daha sabit kalır, böylece Aşama 1'deki κ sıkılaştırılabilir. Maliyet ek eleman integralleri; bugünkü gösterge seviye başına 11–18 s ve artışın ölçülmesi gerekiyor.

**Doğrulama.** `calib_box` ve analitik pillbox'ta est/gerçek oranı her seviyede dar bir bantta kalmalı (hedef: 1 ile 5 arası). Küme testi için hazır bir vaka var: pillbox'ta K = 2 seçilirse TM010'dan sonra gelen TE111 çifti tam K sınırında bölünür.

**Etiketler.** η değişeceği için ağlar değişir; sahip onayı gerekir.

### Aşama 5 — Hızlandırıcı mod için hedefe yönelik inceltme (araştırma, isteğe bağlı)

**Döngüde ne değişir.**

- (a) Tepe hedefli inceltme, docs/30 §7'deki açık kaldıraç. Her seviyede Epk ve Bpk'nın maksimum noktası yeniden bulunur. Çevresindeki duvar elemanlarına 0,5·h kısıtı verilir ya da duvar teriminde |H_t|'nin L^p vekili kullanılır.
- (b) Q0 ve R/Q için DWR prototipi, yalnızca hızlandırıcı mod için: `(K − λ_h M) z = J′(E_h)`, `z ⊥_M E_h`, gradyan izdüşümüyle. Duvar fonksiyoneli Zschiedrich tarzında bir hacim katmanıyla düzenlileştirilir. Ağırlık p+1 zenginleştirmeden ya da yerel kurtarmadan alınır.

**Neden.**

- Kayıp tabanlı uyarlama, ön-asimptotik rejimde enerji tabanlı uyarlamayı geçiyor ([Zschiedrich vd.](https://arxiv.org/pdf/physics/0702128)).
- SLAC duvar kaybı için AMR'ye geçti ([Ge vd.](https://s3.cern.ch/inspire-prod-files-6/6f1ab226c1f92f358546c3b24013a673)).
- Tepe alanlar en yavaş yakınsayan büyüklük; ölçülen s ≈ 0,5.

**Beklenen etki.** Belirli bir DOF'ta Bpk ve Q0 doğruluğunun artması beklenir. Tepe hedefli kısıtın maliyeti neredeyse sıfır. DWR, mod ve seviye başına belirsiz bir faktorizasyon ister; bugünkü faktorizasyon süreleri 235k–430k DOF'ta 20–106 s, bu da tek mod için seviye başına yaklaşık %20–40 ek maliyet demek.

**Doğrulama.** `pillbox_pipes`, TRUBA'daki ince referansla karşılaştırılır. Analitik pillbox'ta Bpk ≤ 1e-2 kalmalı ve iyileşmeli. Fillet'li eliptik vaka da denenir.

**Etiketler.** Değişir. Araştırma düzeyinde ve isteğe bağlı; sahip onayı gerekir.

### Aşama 6 — Cebirsel toleransı seviyeye bağla (düşük öncelik)

**Döngüde ne değişir.** Tahminin tol'un çok üstünde olduğu, yani son seviye olamayacağı belli seviyelerde ARPACK toleransı `max(1e-10, ω·est)` olur (ω = 1e-3). Saklanacak seviye her zaman 1e-10 ile çözülür.

**Neden.** Carstensen–Gedicke ve Rannacher–Westenberger–Wollner cebirsel hatayı ayrıklaştırma hatasına bağlıyor ([CG 2012](https://ins.uni-bonn.de/media/public/publication-media/CG_2012.pdf?pk=1445); [RWW 2010](https://doi.org/10.1515/jnum.2010.015)).

**Beklenen etki.** Ölçüm küçük bir kazanç gösteriyor: tol 1e-6'da ARPACK iterasyonları 74'ten 60'a iniyor, ARPACK süresinin yaklaşık %19'u ([docs/31 §3.8](../docs/31_SESSION_HANDOFF.md)). Bu yüzden en sonda.

**Doğrulama.** Son seviye etiketleri, sıkı tolerans çözümüyle 1e-10 düzeyinde aynı çıkmalı.

**Etiketler.** Ara ağlar az da olsa değişebilir; sahip onayı gerekir.

### Şimdilik önerilmeyenler

- **Tam NVB-AFEM'i varsayılan yapmak.** Teoriye en yakın yol bu, ama kapanış patlaması ölçülmüş durumda; bisection yalnızca son çare olarak kalmalı.
- **hp-uyarlamalılık.** 3B teori yok, geometriler fillet'li ve pürüzsüz, Omega3P bile "2. mertebe çoğu zaman yeterli" diyor.
- **Dengelenmiş garantili sınırlar ve Gallistl–Olkhovskiy alt sınırı.** Maxwell özdeğer problemi için analiz yok ya da yöntem yalnızca en düşük mertebede çalışıyor.
- **HFSS tarzı "iki ardışık yakınsamış geçiş".** Son aşamanın maliyetini ikiye katlar. Büyümeye ölçeklenmiş fark aynı korumayı tek seviyede sağlıyor. Yalnızca test kümeleri için düşünülebilir.

## Sonuç

Literatür, üretim kodları ve depodaki ölçümler bir araya gelince sorunun yeri değişiyor. Asıl mesele tahmin edicinin türü değil. ZZ, Palace ve Omega3P'de de kullanılan, savunulabilir bir seçim. Asıl mesele durdurma kuralı ile ağ büyümesi arasındaki bağın kopuk olması. Büyümeye ölçeklenmiş fark, `--strict_stop`'un ikili "büyüdü mü" kontrolünü nicel bir hata tahminine çeviriyor. Bunun için gereken hızlar zaten depodaki pillbox verisinde var (f için s ≈ 2, Q0 için 1,6, Bpk için 0,5). Aynı veri, tepe alanlar için 1e-3'lük bir fark toleransının gerçekte büyümeye bağlı olarak %0,24 (g = 2) ile %4 (g = 1,05) arasında bir hataya karşılık geldiğini de gösteriyor. Docs/30'daki %1'lik tepe hedefi, ancak geçiş başına büyüme yeterince güçlüyse tutturulabiliyor.

İkinci sonuç maliyetle ilgili. Güvenilir bir durdurma kuralı, eğitim kümelerinde gevşek toleransa geçmenin ön şartı. Bu tartışma docs/31 §5'te açık duruyor. Etiket doğruluğu modelin tabanından 2–4 mertebe daha sıkı. Durma kararı güvenilir hale gelince, eğitim etiketleri docs/30'daki hedeflerle (f 1e-5, Q0 %0,2) üretilebilir ve sıkı tolerans test kümelerine saklanabilir. Son olarak, eğri yüksek mertebeli H(curl) özdeğer problemi için Q0, R/Q ve tepe alanların ne a posteriori ne de yakınsama teorisi var. Bu yüzden `calib_box`, analitik pillbox ve `pillbox_pipes` referans kümesi bu proje için literatürün boş bıraktığı kanıt tabanının yerini tutuyor; TRUBA'da yakınsamış bir `pillbox_pipes` referansı kurmak, bu plandaki bütün aşamaların ortak önkoşulu.
