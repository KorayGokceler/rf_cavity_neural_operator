# 21 — 3D İnceleme: Fizik ve Veri Yarısı (H-alanı, Nédélec N0)

> **Kapsam:** `src/data_gen/dataset_generator_3d.py`, `src/data/dataset_converter_3d.py`, `convert_3d.py`, `src/models/hcurl.py` (Schur-projekteli Gram'lar, `KpSolve` CG + autograd, rank-revealing Ritz), `scripts/research_3d/n0lib.py`, `tests/test_data_gen_3d.py`, `tests/test_dataset_converter_3d.py` ve docs/18–20'deki iddialar.
> **İncelenen commit:** `50ce20e` (`claude/3d-maxwell`). Model, eğitim ve entegrasyon ikinci incelemecidedir.
> **Yöntem:** Her iddia bağımsız bir sayısal deneyle sınandı. Referanslar koddan bağımsız hesaplandı (kendi kuadratür montajı, yoğun `eigh(K, M)`, kendi analitik mod listeleri, kaba kuvvet uzaklık). Ortam: CPU (4 çekirdek), skfem 12.0.2, gmsh 4.15.2, SciPy 1.17.1.

## Özet

Fizik ve veri yarısında **kritik ya da majör hata bulunmadı**. Formülasyon, N0 montajı (yön ve işaret dahil), çekirdek, özçözücü, ölçekleme ve Rayleigh tutarlılığı makine hassasiyetinde doğrulandı. Dejenere kümelerde de hiçbir mod kaçırılmıyor ve sahte sıfır üretilmiyor. Bu yüzden kod değişikliği yapılmadı; bu branch'e yalnız bu not eklendi.

Bulguların hepsi minor veya nit düzeyindedir. En önemli ikisi `hcurl.py`'dedir:

- **(B1)** Bir örnekte **bütün** yönler atılırsa Ritz değerleri $\theta=\infty$ yerine $\theta=1$ olur. Bu değer fiziksel spektrumun **altındadır** ve belgelenen $\theta_k\ge\lambda_{h,k}$ garantisini bozar.
- **(B2)** Schur formu $M_{div}=M_V-B^\top Z$, CG hatasında **birinci derecedendir**. Açık Gram $(PV)^\top M(PV)$ ise ikinci derecedendir. Açık Gram'a geçmek, `drop_tol`'un koruduğu sayısal tabanı ~$10^{4}$ kat düşürür ve `kp_tol`'un gevşetilmesine izin verir: 30 iterasyonda $7.7\cdot10^{-6}$ doğruluk, bugün 70 iterasyonda $5.8\cdot10^{-9}$. PCG, CPU'da baskın maliyettir.

### Bulgu tablosu

| # | Önem | Yer | Başlık |
|---|---|---|---|
| B1 | minor | `src/models/hcurl.py:175` | Tüm yönler atılınca `big = 1e3·0 + 1 = 1` olur ve $\theta=1<\lambda_h$ çıkar; alanlar sıfır döner |
| B2 | minor (tasarım) | `src/models/hcurl.py:133`, `:144` | Schur $M_{div}$ CG hatasında birinci derecedendir; açık Gram ikinci derecedendir. `drop_tol=1e-6` gerekli, ama açık Gram'la gereksiz |
| B3 | minor | `src/models/hcurl.py:116–136` | `project_basis` float64'ü zorlamıyor: float32 girdide CG 2000 iterasyona gider ve $M_{div}$ göreli hatası 0.79 olur |
| B4 | minor | `src/models/hcurl.py:59–89` | PCG `maxiter`'da sessizce döner (uyarı ya da bayrak yok): `maxiter=20` ile $M_{div}$ hatası $2.5\cdot10^{-3}$ |
| B5 | minor | `src/data_gen/dataset_generator_3d.py:424` | $h$, $V^{1/3}$'e göre ölçekleniyor, bu yüzden uzun ya da ince geometrilerde kesit kaba kalıyor. Aynı `mesh_size` ile etiket hatası %0.2 ile %1.4 arasında değişiyor |
| B6 | minor | `dataset_generator_3d.py:204–244`, `:260–266` | `blob` ailesinde sliver tet'ler var: 22 blob'un 8'inde $q<0.1$, en kötüsü $q=1.3\cdot10^{-3}$; kenar uzunluğu $3\cdot10^{-4}$ m, oysa $h\approx7\cdot10^{-3}$ m |
| B7 | nit | `dataset_generator_3d.py:19–25`, `:312–314`; docs/19 §1 | Boşluk (void, $b_2>0$) reddi H-formülasyonu için gereksiz: sahte mod üretmiyor (ölçüldü) |
| B8 | nit | docs/19 §4 | "$O(h^2)$, hata pozitif" iddiası pillbox için doğru. Kutuda hata monoton değil, işaret değiştiriyor ve 0.05'te 8 modun 6'sında negatif |
| B9 | nit | `dataset_converter_3d.py:194–228` | "Tam" uzaklık, 16-en-yakın-ağırlık-merkezi sezgiseline dayanıyor: 29 870 iç düğümün 1'inde fazla tahmin var ($8.6\cdot10^{-5}$ göreli) |
| B10 | nit | `dataset_converter_3d.py:224–225` | Sınıra $10^{-12}$'den yakın **iç** düğüme `vn=0` atanıyor, dolayısıyla yön vektörü sıfır oluyor (pratikte oluşmaz) |
| B11 | nit | `dataset_converter_3d.py:247`; docs/19 §3 | `scale` $L^\infty$ normu (`max|x−c|` koordinat bazında), dönmeye göre değişmez değil; doküman bunu Öklid yarıçapı gibi okutuyor |
| B12 | nit | `dataset_generator_3d.py:206` | "a cut cannot split it" yalnız tek çıkıntı için doğru. Birden çok kesim delik ya da bölünme üretebilir, bunları topoloji ve hacim kontrolü yakalıyor (60 örnekte 0 red) |
| B13 | nit (perf.) | `dataset_generator_3d.py:364–365` | $\sigma=-10^{-2}\bar K_{ii}/\bar M_{ii}\propto h^{-2}$, bu yüzden $|\sigma|/\lambda_1$ ince mesh'te büyüyor (0.35'ten 2.2'ye) ve ARPACK çağrısı 50'den 70'e çıkıyor |
| B14 | nit | `dataset_generator_3d.py:91` | `OMP_NUM_THREADS` numpy/BLAS yüklendikten sonra (fork'tan sonra) ayarlanıyor; etkisiz |
| B15 | nit | `dataset_converter_3d.py:303`, docs/19 §3 | Operatörlerin PKL payı docstring'de "~%85", dokümanda "~%89" |

## Bulgular

### B1 (minor): tüm yönler atılınca $\theta=1$ oluyor (`hcurl.py:175`)

`big = 1e3 * max diag(Aw) + 1.0` yalnız tutulan yönlerin köşegeninden hesaplanıyor. Bir örnekte bütün $\mu\le$ `drop_tol` ise `W = 0`, `Aw = 0` ve `big = 1` olur. Sonuç olarak her Ritz değeri **1.0** çıkar (normalize birim) ve alanlar sıfırdır. Docstring bunun yerine "θ = ∞ (sorted last)" ve "no Ritz value pulled below λ_h" diyor.

**Deney:** 2 geometrili batch (Ne 4.2–5.1k). Taban $V=3\,G\phi+\varepsilon\,Y_{1..3}$; burada $Y$ gerçek özmodlardır, yani projekteli span tam olarak özmodlardır.

| $\varepsilon$ | $\mu$ (üç yön) | `hcurl_ritz` $\theta$ | $\lambda_{true}$ |
|---|---|---|---|
| $10^{-1}$ | $1.6$–$1.9\cdot10^{-6}$ | 5.876, 7.731, 7.735 (hata $\le1.2\cdot10^{-7}$) | 5.876, 7.731, 7.735 |
| $10^{-2}$ | $1.6$–$1.9\cdot10^{-8}$ | **1.0, 1.0, 1.0**; ‖modes‖ = 0 | 5.876, 7.731, 7.735 |

Gradyan sıfırdır (`big` detach edilmiş, `W` sütunları sıfır), dolayısıyla kayıp bu örnekten öğrenemez ve frekans metriği gerçeğin altında sahte bir tahmin görür. Pratikte olasılığı düşüktür: docs/20'de `grad_frac` ≈ 0.59.

**Öneri:** `big`'i tutulan yönlerden bağımsız bir ölçekle alın. Örneğin `diag(Dm(A_V))` yerine örnek başına sabit bir fiziksel üst ölçek ($\lambda$ üst sınırı ~ $\bar K_{ii}/\bar M_{ii}$, `batch`'ten) kullanılabilir. Ya da `keep.any(-1)` yanlış olan örnekler bir maske ile döndürülüp kayıp ve metrikten çıkarılabilir.

### B2 (minor/tasarım): Schur $M_{div}$ birinci derece, açık Gram ikinci derece

$Z=Z^*+e$ için $M_V-B^\top Z$ hatası $-B^\top e$'dir (birinci derece). $(PV)^\top M(PV)=M_V-2B^\top Z+Z^\top K_pZ$ ise $Z^*$'da durağandır, yani hata $O(e^2)$'dir. Near-gradient yönlerde ($\mu$ küçük) birinci derece hata $\mu$'ya bölünerek büyür. `drop_tol=1e-6` bu yüzden **gerekli**: $\mu=10^{-6}$'da $\theta$ hatası $10^{-7}$, $\mu\sim10^{-10}$'da ise Schur $\theta$'yı $\lambda_h$'nin %33 **altına** çekiyor. İlk beklentinin aksine `drop_tol` fazla temkinli değil.

**Deney** (aynı taban, `kp_tol=1e-8`, `drop_tol=1e-14`, $\theta/\lambda_{true}-1$):

| $\varepsilon$ ($\mu$) | Schur $M_{div}$ | açık $(PV)^\top M(PV)$ |
|---|---|---|
| $10^{-2}$ ($10^{-8}$) | $-1.5\cdot10^{-4}$ … $7.3\cdot10^{-4}$ | $\le5.3\cdot10^{-9}$ |
| $10^{-3}$ ($10^{-10}$) | $-0.33$, $-0.07$, $+2.45$ | $\le6.1\cdot10^{-7}$ |
| $10^{-4}$ ($10^{-12}$) | $-0.98$, $-0.88$, $+124$ | $\le9.1\cdot10^{-5}$ |

**Gevşek CG** (gerçekçi karışık taban, 8 kolon, $\max|\theta/\theta_{ref}-1|$):

| `kp_tol` | iterasyon | Schur | açık |
|---|---|---|---|
| $10^{-3}$ | 20 | $1.7\cdot10^{-2}$ | $1.6\cdot10^{-3}$ |
| $10^{-4}$ | 30 | $6.1\cdot10^{-4}$ | $7.7\cdot10^{-6}$ |
| $10^{-6}$ | 50 | $9.8\cdot10^{-7}$ | $4.9\cdot10^{-10}$ |
| $10^{-8}$ | 70 | $5.8\cdot10^{-9}$ | $4.1\cdot10^{-13}$ |

**Öneri:** `M_div = bgram(PV, spmm(M, PV))` kullanılmalı. Maliyeti bir `M` çarpımıdır; `PV` zaten `hcurl_ritz`'te hesaplanıyor. Bu hem çıktı alanlarıyla (modes = $PVc$) hem de `hcurl_grams`'ın kesin çapraz bloğuyla tutarlıdır. Ardından `kp_tol` $10^{-4}$'e çekilebilir: docs/20 §5'te PCG baskın maliyetti ve iterasyon sayısı ~2.3× azalır. `drop_tol` da $10^{-10}$ civarına inebilir. Değişiklik model ajanının alanı olduğundan uygulanmadı.

### B3 (minor): `project_basis` dtype'ı zorlamıyor

Docstring "float64" diyor, ama hesap `V.dtype` ile yapılıyor. float32 girdide: `iters=2000` (maxiter), $M_{div}$'in float64 referansına göre göreli hatası **0.79**. Tek çağıran (`eigenspace_operator_3d.py:86`) `.double()` geçtiği için bugün canlı bir hata değil. **Öneri:** `V = V.double()` ya da `assert V.dtype == torch.float64`.

### B4 (minor): PCG sessiz yakınsamama

`_pcg` `maxiter`'a ulaşınca uyarı vermeden döner. `maxiter=20` ile $M_{div}$ göreli hatası $2.5\cdot10^{-3}$ çıktı. `kp_cg_iters` metriği loglanıyor ama `== maxiter` durumu bayraklanmıyor. **Öneri:** Döngü sonunda son kalıntıyı döndürün ve `it == maxiter` ise `warnings.warn` verin (ya da `KpSolve.last_converged` tutun).

### B5 (minor): mesh boyutu $V^{1/3}$'e göre ölçekleniyor

`--mesh_size 0.12`, 6 mod, pillbox'ta $f$ göreli hataları:

| R, L [cm] | kesit | hata aralığı |
|---|---|---|
| 6, 2 | 2.7 katman | $2.0$–$2.8\cdot10^{-3}$ |
| 3, 10 | yarıçap boyunca ~4 eleman | TM010 **$1.4\cdot10^{-2}$**, TM011 $1.25\cdot10^{-2}$ |
| 6, 10 | | TM010 $8.4\cdot10^{-3}$ |
| 3, 2 | | $\le4.7\cdot10^{-3}$ |

Etiket hatası geometriyle ilişkili olarak ~7× değişiyor. Modelin öğreneceği "gürültü" ailelere ve boyut oranına bağlı. **Öneri:** $h$'yi dalga boyuna bağlayın, $h=c_h\cdot 2\pi/k_K$ ($k_K$ için Weyl veya ilk kaba çözüm tahmini). Ya da $\min(V^{1/3},\,2\,r_{in})$ kullanın.

### B6 (minor): `blob` ailesinde sliver tet'ler

60 rastgele örnek (`--mesh_size 0.1`, seed 123): pillbox ve axisym için $q_{min}\ge0.21$. Blob'da 22 örneğin 8'inde $q<0.1$ olan 1–6 tet var ($q$ = 3·r_in/R_circ). En kötü $q=1.3\cdot10^{-3}$. Bazı kenarlar $3\cdot10^{-4}$ m, yani `MeshSizeMin = 0.2h`'nin altında; bunlar OCC boolean'larının küçük kesişim eğrilerinden geliyor. Sliver sayısı az ve N0 bunlara görece dayanıklı, ama $K-\sigma M$'nin koşul sayısını ve yerel hatayı bozar. **Öneri:** `Mesh.OptimizeNetgen=1`, `occ.removeAllDuplicates()` ya da `Geometry.Tolerance`, ve bir $q_{min}<10^{-2}$ reddi (yeniden çekim döngüsü zaten var).

### B7 (nit): boşluk reddi gereksiz

Küresel boşluklu kutuda (`euler=2`, `n_shells=2`, reddediliyor) yoğun `eigh`: sıfır sayısı **tam $N_v-1$ = 476**, ilk fiziksel değerler ARPACK ile $10^{-13}$ içinde aynı. H-formülasyonunda çekirdek $\nabla P1\oplus\mathcal H^1$'dir ve boyutu $b_1$'dir; $b_2$ (boşluk) sahte mod eklemez. $\mathbf E$'deki Dirichlet alanlarının $\mathbf H$'de karşılığı $\mathbf H=0$'dır. docs/18 §1.4 bunu doğru söylüyor, docs/19 ve üretici ise "$b_1=b_2=0$" istiyor. Aileler zaten boşluk üretmediği için pratik etkisi yok.

### B8 (nit): yakınsama iddiaları

Doküman tablosu **birebir tekrar üretildi** (deterministik; örnek: 0.05'te kutu $2.21\cdot10^{-4}$, pillbox $1.20\cdot10^{-3}$).

- **Pillbox:** mod başına oranlar ($h\sim N_e^{-1/3}$) TM010 için 2.4 / 2.3 / 2.2 / 2.1. $O(h^2)$ ✔, hatalar hep pozitif ✔.
- **Kutu:** $\max|e|$ 0.07'den 0.05'e yalnız $3.0\cdot10^{-4}$'ten $2.2\cdot10^{-4}$'e iniyor. 0.05'te 8 modun 6'sı negatif, mod başına oranlar −3 ile 16 arasında. Kutu için "$O(h^2)$" gösterilmiş değil: yapısız mesh'te hata ön-asimptotik, işareti karışık ve süperyakınsak. docs/18'deki "yapısal mesh'te hepsi altta" gözlemiyle birlikte okunmalı.

### B9–B15 (nit)

- **B9:** 60 mesh'te (29 870 iç düğüm) kaba kuvvet karşılaştırması: yalnız 1 düğümde fark var ($8.6\cdot10^{-5}$ göreli). "Tam" yerine "neredeyse tam (k=16 aday)" denmeli.
- **B11:** $L^\infty$ ölçek, 90° dışı dönmelerde değişir. Augmentasyon ölçeği sabit tuttuğu için tutarlı, ama doküman netleştirilmeli.
- **B13:** Kalibrasyon pillbox'ı:

  | `mesh_size` | $\lvert\sigma\rvert/\lambda_1$ | ARPACK OPinv çağrısı |
  |---|---|---|
  | 0.2 | 0.35 | 50 |
  | 0.1 | 1.11 | 53 |
  | 0.07 | 2.21 | 70 |

  0.07'de $\sigma=-\lambda_1$ ile 53, $\sigma=-0.3\lambda_1$ ile 50 çağrı. Kazanç küçük (~%10–25). Bir $\lambda_1$ tahmini (ör. Faber–Krahn / Weyl) yeterli.
- **B14:** Worker'larda thread sınırı için `threadpoolctl` ya da ortam değişkenini `Pool` oluşturulmadan önce ayarlamak gerekir.

## Doğrulanmış (doğru) maddeler

**1. H-formülasyonu ve çekirdek**
- Tüm kenarlarda esas KS yok ✔.
- Yoğun `eigh(K, M)` ile sıfır özdeğer sayısı dört farklı mesh'te (gmsh küp, tensör küp, pillbox, kutu) ve boşluklu kutuda **tam olarak $N_v-1$**. İlk fiziksel değer ile en büyük "sıfır" arasındaki boşluk $2.5$–$6.3\cdot10^{12}$.
- Kernel = tüm P1 gradyanları (sınır düğümleri dahil) eksi sabitler ✔.
- Modlar $\max|G^\top MU|/\max|MU|=4$–$8\cdot10^{-15}$, bu zayıf $\mathbf n\cdot\mathbf H=0$ demektir ✔.
- Torus: $\lambda_1\approx10^{-12}$, reddediliyor ✔.
- $f=c\sqrt\lambda/2\pi$ ✔; $\lambda_{norm}=\lambda_{fiz}\,\mathrm{scale}^2$ ✔ (Rayleigh $2.9\cdot10^{-15}$).

**2. N0 montajı**
- Rastgele permütasyonlu, jitter'lı ve yerel sırası karıştırılmış mesh'te (**178/360 negatif hacimli tet**) kapalı form $M,K$:
  - kendi 4 noktalı Gauss montajımla $1.9\cdot10^{-16}$ / $2.2\cdot10^{-16}$;
  - skfem `ElementTetN0` ile (işaret eşlemesi dahil) $5.6\cdot10^{-16}$ / $3.2\cdot10^{-16}$.
- skfem kenarları düşük→yüksek ✔ (`orient` = `t[t1] > t[t2]`).
- $\|KG\|/\|K\|=2.4\cdot10^{-16}$ ✔.
- $K_p=G^\top MG$ ile skfem P1 Laplace arasındaki fark $4.6\cdot10^{-16}$, $K_p\mathbf 1=4\cdot10^{-16}$ ✔.
- $\|G x_c\|_M^2/|\Omega|=1\pm2\cdot10^{-16}$ ✔.
- Mevcut test de negatif yönü kapsıyor (`_box_mesh`: 180/360).

**3. Özçözücü sağlamlığı** (ARPACK ile yoğun `eigh` arasında en büyük fark $1.6$–$4.3\cdot10^{-13}$; hiç kaçan ya da tekrarlanan mod yok)
- **Küp** (gmsh, 12 mod): üçlü (110) kümesi, (111)×2 ve 6 katlı (210) kümesi.
- **Tensör küp:** tam dejenere çiftler, $k=1\ldots6$ bütün kesimlerde doğru.
- **Pillbox TM010/TE111 kesişimi** ($L=2.0301R$): 3633.08, 3633.62, 3687.70.
- $U^\top MU-I\le4.7\cdot10^{-15}$, küme içinde dahil.
- `freq_next`, yoğun çözümün $(K{+}1)$. değerine eşit ✔.

**4. Analitik doğruluk**
- Bağımsız TE/TM listeleri (kutu: $TE^z_{mnp}$ $p\ge1$, $TM^z_{mnp}$ $m,n\ge1$; pillbox: $j_{mn}, j'_{mn}$) üreticinin `box_spectrum` ve `pillbox_spectrum` çıktılarıyla birebir aynı ✔.
- Tablo için bkz. B8.
- Tepe RSS 0.05'te 858 MB, süre 31 s (kutu) / 38 s (pillbox): docs/19 ile tutarlı ✔.

**5. Topoloji sertifikası**
- Boşluk: `euler=2`, `n_shells=2`, reddediliyor.
- İki ayrık kutu: `n_components=2`, reddediliyor.
- Tepe noktasında birleşen iki tet (bowtie) ve kenarda birleşen iki tet: `euler=1` ama `euler_boundary=3`, reddediliyor.
- Analiz: manifold olmayan sıkıştırmalar $\chi(\partial\Omega)$'yı hep düşürür. $b_1=b_2\ge1$ olan bir yanıltma ancak tek kabuk ve $\chi=2$ ile mümkün olurdu, bu da `n_shells`/`euler_boundary` birlikte kullanıldığı için oluşamaz.

**6. Özellikler**
- Kapalı-nokta rutini 300 rastgele üçgende örneklemeyle tutarlı (en büyük fark $\le0$).
- Blob'larda yön vektörü kaba kuvvetle $\cos=1.0000$ ✔.
- Dış normal: $\sum\mathbf n\,dA\le2\cdot10^{-16}$, $\oint\mathbf x\cdot\mathbf n\,dA/3V=1.000000000000$ ✔.
- Hacim ağırlıklı merkez $\le5\cdot10^{-15}$ ✔.
- Torsiyon, kürede $(R^2-r^2)/6$ ile $7\cdot10^{-3}$ (P1, $h=0.12$) ✔.

**7. PKL, `Y` ve `--no_operators`**
- 6 geometri, 36 örnek: $|Y^\top MY-I|\le4\cdot10^{-9}$ (float32 $Y$); $G^\top MY$ göreli $\le1.6\cdot10^{-7}$.
- float32 `X`'ten yeniden kurulumda kenarlar birebir aynı; $\Delta K/K\le4.9\cdot10^{-7}$, $\Delta M/M\le3.6\cdot10^{-7}$, Rayleigh $\le5.4\cdot10^{-8}$. docs/19'daki "<1e-5" iddiası ✔.

**8. Üretici mühendisliği**
- İşçi sayısı 1 ve 3 ile H5'ler birebir aynı (nodes, freqs) ✔.
- `.partial` sonra `os.replace` ✔.
- Yeniden çekim döngüsü sınırlı ✔; zaman aşımında pool yeniden kuruluyor ✔. Çöken worker da zaman aşımıyla yakalanıyor.
- 60 örnekte 0 red, 0 hata, 37.5 s (4 işçi).
- **SuperLU pivotsuz:** $K-\sigma M$ simetrik, $\lambda_{min}=3.19>0$, yani SPD ✔. 10.6k DOF'ta aynı sıralamayla pivotlu 7.26 s, pivotsuz 0.58 s; dolum aynı (3.71M) ve geri hata $9\cdot10^{-15}$ ✔. SciPy varsayılanı (COLAMD) 2.3 s ve 8.7M dolum veriyor; docstring'deki "default" sözcüğü "aynı sıralama + pivot" olarak okunmalı. Sabitlenmiş $K_p$ geri hatası $10^{-13}$ ✔.

**9. `hcurl.py`**
- Karışık tabanda ve farklı boyutlu 2 örnekli batch'te (dolgulu):
  - $A_V$ ile $(PV)^\top K(PV)$ arasındaki fark $6\cdot10^{-16}$ ✔;
  - $M_{div}$ ile açık Gram arasındaki fark `kp_tol` $10^{-6}/10^{-8}/10^{-12}$ için $1.3\cdot10^{-6}/8.2\cdot10^{-9}/8.8\cdot10^{-14}$.
- Ortalamasız PCG doğru: Jacobi-PCG'de tutarlılık koşulu Öklid ortalamasıdır ($D^{-1/2}b\perp D^{1/2}\mathbf 1$). Saf gradyan kolonlarında $M_{div}/M_V$ oranı $2.6\cdot10^{-14}$.
- **Geri geçiş:** Jacobian $\Pi K_p^+\Pi$ simetrik, dolayısıyla $\bar B=\Pi K_p^+\Pi\bar Z$ doğru.
  - `KpSolve` yönlü FD ile autograd 10 hanede aynı (421.79875451).
  - `project_basis` + `projected_eigh` zinciri: broadening $10^{-14}$ ile FD ve autograd farkı $4\cdot10^{-8}$ göreli.
  - Varsayılan broadening $10^{-4}$'te sapma $5\cdot10^{-6}$ (beklenen yanlılık).
- `projected_eigh` cebiri ($W^\top D M_{div}D W=I$, ridge yalnız hangi yönün atılacağını etkiler) ✔.

## Açık riskler

1. **B1 ve B2 birlikte:** Eğitim ağı gradyana çok yakın yönler üretirse Schur tabanı (B2) ve `big` ölçeği (B1) etkileşir. Açık Gram ve ölçekten bağımsız bir `big` ile ikisi birden kapanır.
2. **Dejenere kesim:** Rastgele 60 örnekte axisym'lerin 5/17'si ve pillbox'ların 1/21'i $K=6$'da bir çifti bölüyor ($f_{next}/f_K-1<10^{-3}$). Veri doğru; model tarafının `freq_next`'i kullanması şart.
3. **Etiket doğruluğu geometriye bağlı** (B5): Hedef göreli hata $10^{-3}$ ise `mesh_size` tek başına bunu sağlamıyor.
4. **Tam dejenere kümelerde ARPACK:** Bütün testlerde sağlam çıktı, ama tek vektörlü Krylov'da tam katlı değerlerin bulunması yuvarlamaya dayanır. Çok yüksek simetrili (tam küp, yapısal mesh) üretimlerde $k+1$ yerine $k+2$ çözüp kontrol etmek ucuz bir sigortadır.
5. **Ölçek:** 0.05'te 0.86 GB/işçi ve ~35 s/örnek. Ne > 100k için bellek ~$N_e^{1.5}$ büyür; `sample_timeout=300` sessiz kayıplara yol açabilir (sayılıyor ama sebep loglanmıyor).

## Tekrar üretim

Deneyler kısa scriptlerle yapıldı (her biri saniyeler–dakikalar):

- **Montaj:** kendi kuadratürü ve skfem, negatif hacimli tet'ler.
- **Özçözücü:** yoğun `eigh` ve analitik çözüm; küp, tensör küp, pillbox kesişimi, kutu.
- **Yakınsama:** 0.2–0.05.
- **Topoloji:** boşluk, iki bileşen, bowtie.
- **Özellikler:** kaba kuvvet uzaklık, normal, torsiyon.
- **PKL paritesi.**
- **`hcurl`:** Schur ve açık Gram, FD, drop, float32, maxiter.
- **σ ve ARPACK çağrı sayısı.**
- **Aile taraması:** 60 örnek, kalite.
- **SuperLU.**

Tablo değerleri bu notta verilen parametrelerle yeniden elde edilir.

#3d #maxwell #nedelec #inceleme #hcurl #veri-hattı
