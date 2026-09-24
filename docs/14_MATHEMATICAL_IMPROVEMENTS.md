# 14 — Matematiksel İyileştirmeler (Spektral Teori Perspektifi)

> **Kapsam:** Araştırma notu — model/eğitim kodunda değişiklik yok, eğitim koşusu yok.
> **Soru:** Bu repo'nun çözdüğü problem — $\Omega \mapsto \{(\lambda_k, u_k)\}_{k=1}^{3}$, $-\Delta u = \lambda u$ ($\Omega$), $u=0$ ($\partial\Omega$), $f = c\sqrt{\lambda}/2\pi$ — matematiksel olarak nasıl *daha iyi* çözülebilir?
> **Kanıt:** Tüm sayılar bu not için yazılmış küçük CPU deneylerinden gelir (§7); repo'nun kendi üreticisi (`src/data_gen/dataset_generator.py`, varsayılan argümanlar, `--seed 7`, 6 mod) ile üretilmiş **400 geometri** (190 sharp, 210 smooth) kullanıldı.
> **Referans commit:** `6878849`.

---

## 🧭 Özet — Öncelik Sırasına Göre Öneriler

| # | Öneri | Beklenen etki | Efor | Kanıt (§7) |
|---|---|---|---|---|
| **1** | **Frekansı fizikten hesapla:** $f = c\sqrt{\lambda_{norm}}/(2\pi s)$ (s = converter'ın sakladığı `scale`); monoton MLP frekans başlığını kaldır, kaybı $\log\lambda$ üzerinde kur. | Çok yüksek: $\lambda_{norm}$ **mükemmel** tahmin edilse bile yalnız $\lambda_{norm}$'u gören bir başlığın indirgenemez frekans hatası medyan **%5.8** (MAE 0.39 GHz). Formül hatası $4\cdot10^{-16}$. | Düşük | E11 |
| **2** | **SpectralNO için rastgele `max_nodes=1024` alt-örneklemeyi kapat** (ya da deterministik kaba quadratür). | Çok yüksek: rastgele alt-örnekleme Ritz özdeğerlerinde **%4–5 göreli gürültü** ve **%3–7 aşağı yönlü bias** üretiyor; meshlerin %98'i >1024 düğüm. FEM etiketleri ise $10^{-4}$ doğrulukta. | Düşük | E8b, E9 |
| **3** | **Kayıpları $L^2(\Omega)$ iç çarpımına (lumped mass $w_i$) taşı; sert eşikli küme tespiti yerine boşluk-ağırlıklı "flag" (iç içe alt uzay) kaybı; üreticide ≥5 mod sakla** ($\lambda_4$ olmadan 3. mod iyi-tanımlı mı bilinemez). | Yüksek: düğüm toplamı sınır bandını ~5× fazla ağırlıklandırıyor; $\lambda_2\!\approx\!\lambda_3$ yakınlığında tekil vektörler alt uzaydan **~10×** daha hassas; geometrilerin %5'inde $(\lambda_4-\lambda_3)/\lambda_3<\%10$. | Düşük–Orta | E2, E5b, E9 |
| **4** | **SpectralNO'yu gerçekten $H^1_0$-uyumlu yap:** ψ yalnızca türevlenebilir uzamsal girdilere (x, gerçek segment-mesafesi, pürüzsüz ω) + x'ten bağımsız global geometri koduna bağlı olsun; donuk (türevlenmeyen) nodal feature'ları ψ'den çıkar; eleman-içi quadratür; `bc_scale` 0.02 → ≥0.1 veya $\omega$-çarpanı. | Yüksek: donuk açı feature'ı ile dipol Ritz değeri **8.5** (gerçek 14.68, −%42); düğüm quadratürü üst-sınır özelliğini bozuyor (λ₂ = 14.657 < 14.682); b=0.02 gate aynı doğruluk için ~2× baz gerektiriyor. Uyum sağlanınca $\hat\lambda_k \ge \lambda_k$ garantisi ve etiketsiz Ky Fan kaybı mümkün. | Orta | E8 |
| **5** | **Hibrit çıkarım:** ağın K alanını başlangıç alt uzayı olarak kullan; aynı mesh'in P2 matrisleriyle **bir blok ters iterasyon + Rayleigh–Ritz**. | Çok yüksek (doğruluk): %5 alan hatası → **3·10⁻⁵** özdeğer hatası (P1 matrisleri ile, ~5 ms); %20 → 8·10⁻⁴. Stratejik not: 20 ms'lik kaba P2 çözümü (H=4 mm) zaten $<10^{-3}$ hata veriyor — saf NN bu rejimde FEM'i doğrulukta yenemez. | Orta | E6, E10 |
| **6** | **Düşük boyutlu şekil girdisi + fiziksel taban çizgisi:** r(θ) Fourier katsayıları (K=16–32), 2. mertebe Rayleigh–Hadamard formülü ve/veya landscape normalizasyonu $\lambda_1\cdot\max w$; ağ yalnızca log-oran rezidüelini öğrensin. | Yüksek (veri verimliliği): $\lambda_1\cdot\max w$ tüm set üzerinde **CV %0.9** (1.436–1.491); K=16 harmonik λ'yı ≤9·10⁻⁴ korur; 2. mertebe PT + 18 invaryant özellik üzerinde ridge → 168 örnekle medyan %1.5. | Orta | E1, E3, E4 |
| **7** | **Üretici düzeltmeleri:** "smooth" şekiller aslında köşeli (100 noktalı örnekleme + kırpma → %98'inde >200° iç açı); spline/≥1000 nokta veya eğrilik sınırı. Sharp şekillerde köşe tekillik zenginleştirmesi (enrichment) modelin bazına. | Orta: etiket λ doğruluğu zaten iyi (medyan 5·10⁻⁵–1.3·10⁻⁴) ama alan gradyanları tekil; L-şeklinde 6 köşe fonksiyonu Ritz hatasını **1.4·10⁻² → 2·10⁻⁵** indiriyor. | Düşük–Orta | E6, E7b, E12 |
| 8 | Uzun vade: köşe-adaptif MPS / Fredholm determinantı ile referans/validasyon verisi; "öğrenilmiş MPS" yapısı. | Orta (validasyon), araştırma riski yüksek | Yüksek | §5 |

**Tek cümlelik ana mesaj:** Öğrenilmesi gereken nesne tekil özvektörler değil, **boşluklarla ayrılmış spektral projektörler ve boyutsuz özdeğerlerdir**; ölçek ve Rayleigh–Ritz yapısı *tam* olarak kodlanabilir, ve mesh zaten elimizdeyken en güçlü kullanım ağı FEM'in yerine değil, **FEM için başlangıç alt uzayı / kaba uzay** olarak koymaktır.

---

## 0. Notasyon

- $A = -\Delta_D$ (Dirichlet Laplace), $H^1_0(\Omega)$ üzerinde $a(u,v)=\int\nabla u\cdot\nabla v$, $m(u,v)=\int uv$. Spektrum $0<\lambda_1<\lambda_2\le\lambda_3\le\dots$ (çok katlılıkla).
- Ayrık: $K U = M U \Lambda$, $U^\top M U = I$. Düğüm ağırlıkları $w_i$ = lumped mass (komşu üçgen alanlarının 1/3'ü; converter'da kolon 5, max'a bölünmüş).
- $s$ = converter `scale` $=\max_i\|x_i-\bar x\|_\infty$; normalize domain $\Omega_n = (\Omega-\bar x)/s$, $\lambda_{norm}=\lambda s^2$.
- Göreli boşluk: $\delta_j = (\lambda_{j+1}-\lambda_j)/\lambda_j$.

---

## 1. İyi-Tanımlılık ve Hedefler

### 1.1 Hangi nesne süreklidir?

Şekil → spektral veri eşlemesinin düzenliliği klasik olarak bilinir:

- **Sıralı özdeğerler** $\Omega\mapsto\lambda_k(\Omega)$ her zaman süreklidir (Lipschitz; Weyl/min–max), ama kesişmelerde türevlenemez.
- **Basit bir özdeğer ve özvektörü** analitiktir (Kato 1966; Rellich 1969). Hadamard türevi: $\lambda'[V] = -\int_{\partial\Omega}(\partial_n u)^2\,V\!\cdot n\,ds$.
- **Çok katlı/küme özdeğerlerinde** tekil özvektörler süreksizdir; ama kümenin **simetrik fonksiyonları** (toplam, çarpım, ortalama) ve **küme projektörü** analitiktir (Lamberti & Lanza de Cristoforis 2004).
- Genel (simetrisiz) ailelerde dejenerelik **kodboyut-2**'dir (von Neumann & Wigner 1929). Parametre uzayında konik ("diabolik") noktalar oluşur (Berry & Wilkinson 1984, üçgen membranlar — bizim sharp poligonlarımızın tam analoğu). Böyle bir noktanın etrafındaki döngüde özvektör işaret değiştirir (Berry fazı π). Yani **işareti düzeltilmiş tekil özvektör için sürekli bir global seçim topolojik olarak imkânsızdır** (gerçek doğru demeti Möbius demetidir). Projektör $P=u u^\top M$ ise orada da pürüzsüzdür.

Sonuç: tekil $u_k$ ancak $\delta_{k-1}$ ve $\delta_k$ büyükken iyi-tanımlı bir hedeftir. Her zaman iyi-tanımlı hedefler şunlardır:

1. **Sıralı özdeğerler** (boyutsuz, log ölçekte).
2. **Riesz projektörü** $P_I = \frac{1}{2\pi i}\oint_\Gamma (z-A)^{-1}\,dz$. Burada $\Gamma$, $\{\lambda_k\}_{k\in I}$ kümesini spektrumun geri kalanından ayırır. Ayrık hâli $P_I = U_I U_I^\top M$, çekirdeği $p_I(x,y)=\sum_{k\in I}u_k(x)u_k(y)$ — işaretten ve küme içi dönmeden bağımsızdır.
3. **İç içe spektral alt uzaylar (flag)** $P_{\le j}=\sum_{k\le j}P_k$. $P_{\le j}$ yalnızca $\delta_j$'ye bağlı olarak iyi koşullanmıştır.

**Davis–Kahan sin Θ teoremi** (Davis & Kahan 1970): $\hat A = A+E$ ve $\hat P_{\le j}$ için
$$\|\sin\Theta(\hat P_{\le j}, P_{\le j})\| \;\le\; \frac{\|E\|}{\lambda_{j+1}-\lambda_j}\quad(\text{veya rezidüel formunda } \|R\|/\text{gap}).$$
Yani tekil bir $u_k$ hatası $\sim \|E\|/\min(\lambda_k-\lambda_{k-1},\lambda_{k+1}-\lambda_k)$ ile büyür. Kümenin projektörü ise yalnızca **kümenin dışına** olan boşluğa bağlıdır.

**Veride durum (E2, E5b):**

| Boşluk | medyan | <%1 | <%2 | <%5 | <%10 |
|---|---|---|---|---|---|
| $(\lambda_2-\lambda_1)/\lambda_1$ | 1.10 | 0 | 0 | 0 | 0 |
| $(\lambda_3-\lambda_2)/\lambda_2$ | 0.28 | 0.3% | 1.0% | 4.8% | 11.7% |
| $(\lambda_4-\lambda_3)/\lambda_3$ | 0.34 | 0 | 0.5% | 1.3% | 5.0% |
| $(\lambda_5-\lambda_4)/\lambda_4$ | 0.15 | 2.0% | 3.5% | 8.0% | 28% |

- Temel durum her zaman iyi ayrık (≥%10; Krein–Rutman ile basit). $P_1$ her zaman iyi-tanımlı.
- Tam dejenerelik (<%1) nadir (~%0.3). Ancak **orta boşluklar** (%2–10) %10 civarında görülüyor. Davis–Kahan'a göre sorun sadece "tam dejenere" durum değil, bu da.
- **Kritik:** $(\lambda_4-\lambda_3)/\lambda_3$ geometrilerin %5'inde <%10, ama veri yalnız 3 mod saklıyor. Bu durumda 3. mod ve hatta 3-boyutlu alt uzay $P_{\le 3}$ kötü koşulludur ve model bunu bilemez. E5b'deki son satırda ($\delta_3=0.059$) $10^{-3}$ sınır pertürbasyonu $u_3$'ü $6.9\cdot10^{-3}$ döndürdü, $\sin\Theta(P_{\le3})=6.7\cdot10^{-3}$ oldu. Aynı pertürbasyon iyi ayrık durumlarda ~$1\cdot10^{-3}$.
- E5b (aynı topolojide mesh morfolama, $10^{-3}$ radyal pertürbasyon): $\delta_2\approx$ %1–3 iken $\sin\angle(u_2)$ = 1–2.6·10⁻², çift alt uzayı $\sin\Theta(u_2,u_3)$ = 1–2·10⁻³. **Amplifikasyon ~10×** (δ<%2 için medyan 10.4; δ>0.2 için 0.4). Ölçülen katsayı $\sin\angle u_2\cdot\delta_2/\text{pert}\approx0.25$, Davis–Kahan ölçeklemesiyle uyumlu.

### 1.2 Doğru metrikler

- Alt uzay mesafesi: $M$-iç çarpımında temel açılar $\theta_i$ (Knyazev & Argentati 2002, *A-based scalar product*). $\hat Q, Q$ $M$-ortonormal ise
  $$d_c^2 = \tfrac12\|\hat P-P\|_F^2 = n - \|\hat Q^\top M Q\|_F^2 = \sum_i\sin^2\theta_i \quad(\text{chordal / projeksiyon F-mesafesi}).$$
  Geodezik mesafe $\sqrt{\sum\theta_i^2}$ yerine chordal mesafeyi kullanın: $\theta=0$'da gradyanı düzgündür (Edelman, Arias & Smith 1998; Ye & Lim 2016).
- Tek mod için bu $1-\cos^2_M\angle(\hat u,u)$ olur: işaretten ve genlikten bağımsızdır, **peak normalizasyonuna gerek kalmaz**.
- Özdeğer: $|\log\hat\lambda_k-\log\lambda_k|$ (boyutsuz, göreli). Küme için küme-ortalaması $\bar\lambda_I=\mathrm{tr}(AP_I)/|I|$ analitik, ayrışma $\lambda_{max}-\lambda_{min}$ ise Lipschitz'tir.
- A posteriori: tahmin çiftinin rezidüeli $\eta_k=\|K\hat u-\hat\lambda M\hat u\|_{M^{-1}}$ mesh üzerinde hesaplanabilir. Kato–Temple sınırı $\hat\rho-\eta^2/(\mu-\hat\rho)\le\lambda\le\hat\rho$ ($\mu$: sonraki özdeğerin alt tahmini) iki taraflı sertifika verir.

### 1.3 Mevcut kayıpların değerlendirmesi (`src/training/lightning_module.py`)

| Bileşen | Sorun | Öneri |
|---|---|---|
| `detect_clusters(ft_b, 0.05)` | Eşik **z-skorlu GHz** üzerinde: veri setinin std'sine (1.10 GHz) bağlı; mutlak 0.055 GHz ≈ %1 göreli frekans ≈ %2 göreli λ boşluğu. Sert eşik şekle göre süreksiz bir kayıp üretir: boşluğu %2.1 olan çift tekil-mod kaybı alıyor ama Davis–Kahan'a göre hâlâ ~5× hassas. Sadece K=3 içine bakıyor, $\lambda_4$ yok. | Boşluğu hedef λ'dan **göreli** hesapla; sert küme yerine sürekli **flag kaybı** (aşağıda); ≥5 mod sakla. |
| `grassmannian_loss`, `_masked_orthonormalize` | QR ve iç çarpımlar **ağırlıksız düğüm toplamı**. Graded mesh'te düğümlerin %50'si alanın %22'sinde (duvara 3 mm). max/min düğüm alanı ~63; mod-1 enerjisinin bu banttaki payı ağırlıksızda %2.4, $L^2$'de %0.5 → **~5× aşırı ağırlık** (E9). Metrik mesh'e bağımlı ve mesh-bağımsız bir operatör hedefi değil. | $E\leftarrow\sqrt{w}\odot E$ ile QR (≡ $M$-ortonormalizasyon). Rel-L2 metrikleri de $w$-ağırlıklı olsun. |
| Tekil mod: `min(‖e−t‖²,‖e+t‖²)/‖t‖²` | Peak-normalize hedefe göre genliğe duyarlı. Peak (max) normalizasyonu türevlenemez ve mesh'e bağlı (düğüm tepeye denk gelmezse $O(h^2)$ hata). | $1-\cos^2_M$ (yani n=1 chordal); genlik serbest. Görsel çıktı için peak normalizasyonu sadece raporlamada. |
| OT eşleştirme (GNOT) | Frekans + alan maliyeti. Yakın-dejenere çiftlerde alan maliyeti tanımsız; Hungarian argmin süreksiz. | Özdeğerler **sıralı** tahmin edilsin (sıralı özdeğerler süreklidir), eşleştirme sıraya göre; alanlar flag kaybıyla. |
| `freq` MSE (z-skor GHz) | Ölçekten bağımsız girdilerle tanımsız hedef (§2). | $\log\lambda_{norm}$ üzerinde MSE; $f$ sonradan tam formülle. |
| `_basis_conditioning_loss` | Makul (baz kollapsını önler). Ama Rayleigh–Ritz baz dönüşümünden bağımsız olduğu için fiziksel çıktıyı değiştirmez; sadece koşullamayı etkiler. | Kalabilir; Cholesky jitter'ı zaten var. |

**Önerilen alan kaybı (flag / boşluk-ağırlıklı projektör kaybı):**
$$\mathcal L_{field} = \sum_{j=1}^{K} \beta_j\; d_c^2\!\big(\hat P_{\le j}, P_{\le j}\big),\qquad \beta_j=\frac{\delta_j^2}{\delta_j^2+\tau^2},\ \ \tau\approx0.03,$$
$\hat P_{\le j}$ = sıralı ilk $j$ tahmin alanının $M$-ortonormal span'i.
- Tüm boşluklar büyükken ($\beta_j\to1$) bu, mod başına $\sin^2$ kayıplarının bir kombinasyonudur.
- Bir boşluk kapanırken ($\delta_j\to0$) ilgili iç içe alt uzay terimi yumuşakça söner; çiftin toplamını içeren $P_{\le j+1}$ terimi kalır.
- Davis–Kahan'a göre her terimin duyarlılığı $\beta_j\sin^2\Theta\lesssim\|R\|^2/(\delta_j^2+\tau^2)$ ile sınırlı: **kayıp şekle göre sürekli ve Lipschitz**.
- $j=K$ terimi için $\delta_K$ gerekir, yani $\lambda_{K+1}$ saklanmalı.

**Ek, işaretsiz noktasal hedef (GNOT için):** kümenin diyagonal çekirdeği $\rho_I(x)=p_I(x,x)=\sum_{k\in I}u_k(x)^2$ ($L^2$-normalize $u$). Diskte dipol çifti için $\rho_{\{2,3\}}=J_1(j_{11}r)^2/N^2$ **dönme simetriktir** — dejenerelik bu hedefte tamamen kaybolur. Direkt alan tahmini yapan GNOT için $\rho_1$, $\rho_{\le 2}$, $\rho_{\le 3}$ yardımcı çıktılar olarak iyi-tanımlı, işaretsiz ve sürekli hedeflerdir.

**Uygulama taslağı:**
- `src/data_gen/dataset_generator.py`: `--n_eigen_modes` varsayılanı 3 → 5. Converter yalnız ilk 3 alanı hedef yapar, $\lambda_4,\lambda_5$'i geometriye (`geometry_pool[g]['lam_all']`) yazar.
- `src/data/dataset.py`: batch'e `Y_lam` (normalize λ, hedef + ekstra) ve `W` (lumped mass, `node_area` kolonundan) ekle.
- `src/training/lightning_module.py`: `grassmannian_loss(E_hat, E_tgt, mask, w)` — `E*sqrt(w)` ile QR. Yeni `flag_loss(pred_sorted, tgt_sorted, lam_tgt_all, w, tau)`. `detect_clusters` sadece raporlama için, göreli λ boşluğuyla.

---

## 2. Ölçek ve Geometri İnvaryansı

### 2.1 Tam simetriler

$\Omega' = sQ\Omega + b$ ($Q\in O(2)$) için $\lambda_k(\Omega')=\lambda_k(\Omega)/s^2$ ve $u_k'(x) = u_k(Q^\top(x-b)/s)$ (L²-normalizasyonla ayrıca $1/s$ faktörü). E1: aynı mesh ölçek/dönme/öteleme sonrası $|\lambda s^2/\lambda_0-1|<10^{-14}$. gmsh'in bu şekiller için dönmeye eşdeğer mesh ürettiği de görüldü (yeniden meshlenmiş döndürülmüş kopyada $10^{-15}$).

### 2.2 Bulgu (a)'nın nicel hâli: frekans başlığı neden çuvallıyor

SpectralNO'nun Ritz değeri $\hat\lambda\approx\lambda_{norm}=\lambda s^2$. `_MonotoneFreqHead` $\log\hat\lambda\mapsto$ z-skorlu $f$ haritası öğreniyor, ama $f=c\sqrt{\lambda_{norm}}/(2\pi s)$ **s'ye de bağlı**. Veride $s$'nin CV'si %12 (0.030–0.059 m). E11: $\lambda_{norm}$ *tam* bilinse bile en iyi 1-D harita (7. derece polinom) medyan **%5.8**, 90. persentil %16 frekans hatası, **MAE 0.39 GHz** bırakıyor. Tam formül ise $4\cdot10^{-16}$.

**Öneri 1 (tam kodlama):**
$$\hat f_k = \frac{c}{2\pi s}\sqrt{\hat\lambda_{k}^{norm}}, \qquad \mathcal L_\lambda = \sum_k\big(\log\hat\lambda^{norm}_k-\log\lambda^{norm}_k\big)^2,\ \ \lambda^{norm}_k = (2\pi f_k/c)^2 s^2 .$$
Öğrenilebilir parametre yok; frekans sırası = özdeğer sırası otomatik. GNOT'ta frekans başlığı $\log\lambda^{norm}$ tahmin etsin.

### 2.3 Hangi boyutsuz normalizasyon?

Normalize koordinatlarda her boyutsuz hedef eşdeğerdir ($\lambda_{norm}A_{norm}=\lambda A$). Soru, hangi normalizasyonun **kalan varyansı** en aza indirdiği; bu, ağın öğrenmesi gereken rezidüeldir. E1/E2 (CV = std/ortalama):

| Büyüklük | CV (tüm) | Aralık | Not |
|---|---|---|---|
| $\lambda_1$ (fiziksel) | 0.145 | — | üretici boyutları dar |
| $\lambda_1 s^2$ ($L^\infty$ ölçek) | **0.289** | — | $s$ kötü boyut ölçüsü (sharp şekillerde sivri uçlar belirliyor); $R^2(\log f_1\sim\log s)=0.003$ |
| $\lambda_1 A/(\pi j_{01}^2)$ (Faber–Krahn, ≥1) | 0.172 | 1.04–1.94 | sharp 0.048, smooth 0.118 |
| $\lambda_1\rho^2$ (iç yarıçap ρ) | 0.119 | 2.59–4.54 | $\le j_{01}^2=5.78$ (monotonluk); konvekste $\ge\pi^2/4$ (Hersch) |
| $\lambda_1 T/A$ (Pólya, ≤1; $T=\int w$) | 0.061 | 0.567–0.712 | torsiyon fonksiyonu $-\Delta w=1$ |
| $\lambda_1\cdot\max w$ (landscape) | **0.009** | **1.436–1.491** | Arnold et al. 2019; 2D'de teorik ≈ $1+d/4=1.5$; medyan 1.454 |
| $\lambda_2/\lambda_1$ | — | max 2.48 | Ashbaugh–Benguria sınırı $j_{11}^2/j_{01}^2=2.539$ ✓ |

Yorum:
- $\lambda\cdot s^2$ normalizasyonu bu veri için doğal hedef değil. Model $\lambda_{norm}$'u öğrenmek zorunda (girdiler normalize), ama kaybı ve çıktı parametrizasyonunu **Faber–Krahn tabanına göre** kurmak hem varyansı düşürür hem pozitifliği garanti eder:
  $$\hat\lambda_1 = \frac{\pi j_{01}^2}{A_{norm}}\big(1+\mathrm{softplus}(z_1)\big),\quad \hat\lambda_2=\hat\lambda_1\Big(1+(\tfrac{j_{11}^2}{j_{01}^2}-1)\,\sigma(z_2)\Big),\quad \hat\lambda_3=\hat\lambda_2(1+\mathrm{softplus}(z_3)).$$
  Bu parametrizasyon Faber–Krahn, Ashbaugh–Benguria ve sıralamayı **yapısal** yapar. $A_{norm}=\sum_i w_i$ normalize alan (converter normalize etmeden önce saklanmalı).
- **Landscape fonksiyonu** son derece güçlü bir önseldir: $\log\lambda_1 = \log(1.454) - \log\max w + r$, $|r|\lesssim$ %2. $w$ bir **lineer** Poisson çözümüdür (tek seyrek Cholesky, özdeğer çözümünden çok daha ucuz ve sinir operatörleri için çok daha kolay bir hedef). $w$ ayrıca pürüzsüz, sınırda sıfır, köşelerde özfonksiyonlarla aynı tekillik davranışında bir çarpandır (§3.3).

### 2.4 Dönme/yansıma

- Mevcut augmentasyon (`dataset.py`) yalnız **train**'de principal-axis feature'larını (kolon 6–7) sıfırlıyor; val/test'te gerçek değerler geliyor → **train/val dağılım kayması**. Ya augmentasyonda bu feature'ları yeniden hesaplayın (dönme sonrası PCA değişmez: açılar dönme-invaryant, bu yüzden zaten sıfırlamaya gerek yok), ya da hiç kullanmayın.
  > ✅ **Güncelleme:** Bu kayma eğitim altyapısı düzeltmelerinde giderildi: `dataset.zero_gauge_features` augmentasyonlu koşularda tüm split'lere (ve checkpoint üzerinden `infer.py`'ye) uygulanıyor.
- Daha ilkeli seçenek **kanonikleştirme**: şekli ikinci moment tensörünün eksenlerine döndür. PCA işaret/eksen belirsizliği (neredeyse izotrop şekillerde kararsız) için 4 elemanlı çerçeve üzerinden **frame averaging** (Puny et al. 2022) veya öğrenilmiş kanonikleştirme (Kaba et al. 2023).
- Alan hedefleri skaler olduğundan dönmeyle birlikte taşınır; flag kaybı zaten $O(2)$-uyumludur.

---

## 3. Galerkin / İndirgenmiş-Baz Formülasyonları

### 3.1 SpectralNO min–max garantisi ne zaman geçerli?

Courant–Fischer: $V_M\subset H^1_0(\Omega)$, $\dim V_M=M$ ve formlar **tam** hesaplanıyorsa Ritz değerleri $\hat\lambda_k=\min_{S\subset V_M,\dim S=k}\max_{v\in S}a(v,v)/m(v,v)\ \ge\ \lambda_k$. Ayrıca $\hat\lambda_1-\lambda_1 \simeq \|u_1-\hat u_1\|_a^2 - \lambda_1\|u_1-\hat u_1\|^2$, yani özdeğer hatası **enerji normunda karesel alan hatasıdır** (Babuška & Osborn 1991; Boffi 2010). Garanti üç koşula bağlıdır. Mevcut kodda üçü de ihlal ediliyor:

1. **Uyum ($H^1_0$):**
   - Gate `tanh(d/2b)`'deki $d$ = *en yakın sınır düğümüne* uzaklık. Sınır kenarlarında $d>0$ (kenar ortasında ≈ $h_b/2\approx0.015$ normalize birim → gate ≈ 0.36), yani ψ gerçek sınırda sıfır değil.
   - ψ'ye giren `dir_bnd`, `dist_2nd/3rd`, `curvature`, `convexity` feature'ları sınır düğümlerinin **Voronoi hücrelerinde parçalı sabit/sıçramalıdır**. ψ hücre sınırlarında sıçrar, yani $\psi\notin H^1$. Nodal gradyan sıçramaların (sonsuz) enerjisini görmez.
2. **Donuk feature sızıntısı (bulgu c):** `cos/sin_principal` $x$'in bilinen fonksiyonları ama türevlenmiyor. E8c: $\psi_i=\omega\,r^{2i+1}c(x)$, $c$ = donuk açı feature'ı → dipol Ritz değeri **8.52** (n=6), tam gradyanla **14.684**, gerçek $j_{11}^2=14.682$. Açısal enerji $\int|\partial_\theta u|^2/r^2$ tamamen kayboluyor ve değer $j_{01}^2=5.78$'e doğru çekiliyor. Özdeğer-tabanlı herhangi bir kayıp ağı bu sızıntıyı **sömürmeye** iter.
3. **Quadratür ("variational crime", Strang 1972):**
   - Düğüm (vertex) kuralı $O(h^2)$. E8b: ω-bazında $\lambda_1$ +%0.5.
   - b=0.02 gate'li bazda **üst sınır bozuluyor**: düğüm quadratürüyle $\hat\lambda_2=14.657<14.682$; aynı fonksiyonlar tam quadratürle 14.707.
   - Banerjee & Osborn (1990): quadratür/lumping her iki yönde sapma yapabilir; lumped mass tipik olarak alt tarafa çeker (Armentano & Durán 2003).
   - **Rastgele 1024 düğüm alt-örnekleme** (`dataset.max_nodes`): 20 çekilişte $\hat\lambda$ göreli std **%4–5**, ortalama **%3–7 aşağıda** (Ritz minimumu gürültülü kuadratik formlar altında Jensen-tipi aşağı bias verir). Meshlerin %98'i >1024 düğüm (medyan 1694).

### 3.2 Gate genişliği

E8a (tam quadratür, birim disk, Ritz göreli hatası):

| Derece (M) | gate b=0.02 (λ₁, λ₂) | gate b=0.1 | $\omega=1-r^2$ |
|---|---|---|---|
| 4 (15) | 4.5e-3, 1.2e-1 | 0, 6.3e-2 | −4e-5, 1.3e-3 |
| 6 (28) | 2.6e-3, 2.0e-3 | 0, 7e-4 | taban (≈−4e-5, poligon/geometri tabanı) |
| 10 (66) | 1e-4, 4e-4 | taban | taban |

Normalize koordinatta $b=0.02$, sınır mesh adımından ($0.0012/0.04\approx0.03$) **küçük**: gate'in sınır katmanı düğümlerle çözünürlenmiyor ve ağ $d/\tanh(d/2b)$ gibi keskin bir düzeltmeyi öğrenmek zorunda. `bc_scale ≥ 0.1` veya pürüzsüz bir ω aynı doğruluğu yaklaşık yarı baz sayısıyla veriyor.

### 3.3 Tam uyumlu baz nasıl kurulur (Kantorovich–Rvachev)

$$\psi_m(x) = \omega(x)\,N_m\big(\mathrm{RFF}(x),\,g\big),$$
- **ω:** $\partial\Omega$'da sıfır, içeride pozitif, Lipschitz (tercihen pürüzsüz) bir fonksiyon. Kantorovich & Krylov 1958; Rvachev & Sheiko 1995 R-fonksiyonları; Sukumar & Srivastava 2022 poligonlar için yaklaşık mesafe alanları $\phi=(\sum_i\phi_i^{-p})^{-1/p}$; Lagaris et al. 1998. Seçenekler:
  (a) **poligon segmentlerine gerçek mesafe** $d_\partial(x)$ (düğüme değil; nokta–segment mesafesi, Lipschitz, $|\nabla d|=1$ a.e.);
  (b) Sukumar–Srivastava ADF (vertex dışında pürüzsüz);
  (c) **torsiyon/landscape fonksiyonu** $w$ ($-\Delta w=1$): pürüzsüz yüzeyde $\partial_n w\ne0$, konveks köşelerde özfonksiyonlar gibi hızla söner, re-entrant köşede $r^{\pi/\alpha}$ tipi davranır; ayrıca $\max w$ λ₁ tahmini verir.
- **g:** $x$'ten bağımsız global geometri kodu (sınır noktalarından set-encoder, r(θ) Fourier katsayıları, $A$, $\max w$…). Geometri bilgisi ψ'ye **yalnız bu kanaldan** girer; $\nabla_x g=0$ tam doğrudur.
- **Girdiden çıkarılacaklar:** `node_area` (mesh artefaktı; yalnız quadratür ağırlığı), `dist_2nd/3rd`, `curvature`, `convexity`, `dir_bnd` (hepsi süreksiz). `cos/sin_principal` ya analitik türevli olarak ($x$'in fonksiyonu) ya da hiç.
- **Quadratür:** ağı eleman içi noktalarda değerlendir: kenar-orta 3 nokta kuralı (kuadratiklerde tam) veya Dunavant 6 nokta. $x_q$ barisentrik, ağırlık $|T|\,\omega_q$. Maliyet ×3–6, ama kuralın hatası $\psi$ pürüzsüzlüğüne göre $O(h^4)$. `elements` zaten dataset'te var.
- **Sonuç:** $\hat\lambda_k\ge\lambda_k$ garantisi geri gelir. Bu durumda
  - $\mathcal L_\lambda=\sum_k(\hat\lambda_k-\lambda_k)/\lambda_k\ \ge0$ bir **enerji-normu alan kaybıdır**;
  - **Ky Fan** ilkesi $\sum_{k\le K}\lambda_k=\min_{\dim S=K}\mathrm{tr}(\ldots)$ ile etiketsiz (self-supervised) bir terim $\sum_k\hat\lambda_k$ eklenebilir. Minimumu tam olarak doğru alt uzayda (Deep Ritz: E & Yu 2018; Spectral Inference Networks: Pfau et al. 2019). Etiketsiz yeni şekillerde ince ayar mümkün.
  - **Uyarı:** uyum sağlanmadan bu terim sızıntıları sömürür (E8c).

### 3.4 İndirgenmiş-baz (RB) literatürüyle karşılaştırma

- **RB özdeğer yöntemleri:** Machiels, Maday, Oliveira, Patera & Rovas 2000 (çıktı sınırları); Horger, Wohlmuth & Dickopf 2017 (eşzamanlı çok-özdeğer RB, greedy); Fumagalli, Manzoni, Parolini & Verani 2016 (a posteriori sınırlar); Rozza, Huynh & Patera 2008 (geometrik parametrizasyon: referans domain + afin/nonafin harita). Kümeler ve kesişimler için kümeyi toplu yaklaştıran RB çalışmaları da var (ör. arXiv:2302.00898, "presence of clusters and intersections").
- SpectralNO **geometri-koşullu, öğrenilmiş bir RB**'dir: RB'de baz offline snapshot'lardan (POD/greedy) gelir ve **referans domain**'de yaşar. Burada ise her şekil için farklı mesh var.
- **Şekil-parametrik POD için gerekli adım: ortak referans domain.** Yıldız-şekiller için polar harita $x = c + \rho\,r(\theta)(\cos\theta,\sin\theta)$, $(\rho,\theta)\in[0,1]\times S^1$. Özfonksiyonları $(\rho,\theta)$ ızgarasına çekip (pull-back) POD alınırsa tüm veri seti için **tek bir sabit baz** çıkar. SpectralNO'nun ψ'si bu sabit bazın geometriye bağlı bir düzeltmesi olabilir: $\psi_m = \omega\cdot(\Phi_m\circ F^{-1} + \delta N_m)$.
- **Öğrenilmiş önkoşullayıcı / multigrid:** Greenfeld et al. 2019, Luz et al. 2020 (AMG prolongasyonlarını GNN ile öğrenme). Özdeğer için doğal analog: ağın çıktısını LOBPCG'ye (Knyazev 2001) **başlangıç bloğu** olarak vermek (§3.5).

### 3.5 Kaba FEM'e düzeltme (çok-sadakatli, iki-ızgara)

- Xu & Zhou 2001 iki-ızgara yöntemi: kaba $H$'de özçift, ince $h$'de bir lineer çözüm $K_h\tilde u = \lambda_H M_h I_H u_H$. Hata $O(H^{2p}+h^{p})$ tipi. E6'da iki-ızgara ince çözümün doğruluğunu aynen geri verdi (smooth 1.3e-4, sharp 4.2e-5). Ama bu problem boyutunda (~1300–2000 düğüm) ince `eigsh` zaten 76–115 ms, bu yüzden **zaman kazancı yok**. İki-ızgara ancak çok daha büyük meshlerde anlamlı.
- **Asıl stratejik bulgu (E6):** H = 4 mm tek-tip kaba P2 (286–438 düğüm) **17–24 ms**'de göreli λ hatası 2.7e-4 (sharp) / 7.9e-4 (smooth) veriyor. Tipik bir sinir operatörü $10^{-2}$ mertebesinde kalıyor. Yani **mesh verilmişse saf NN, kaba FEM'i doğrulukta yenemez**. NN'nin değeri (i) binlerce şekli GPU'da toplu değerlendirmek, (ii) türevlenebilir tasarım döngüsü, (iii) mesh'siz düşük boyutlu girdi (§4) ve (iv) FEM'i **ısıtmak**tır.
- **E10 (hibrit):** simüle NN çıktısı (%5 veya %20 pürüzsüz alan hatası + çift içinde rastgele dönme) → mesh'in kendi matrisleriyle bir blok ters iterasyon $W = K^{-1}M\hat U$ + Rayleigh–Ritz:

| NN alan hatası | Ritz (ham tahmin) λ hatası | 1 adım sonrası | süre |
|---|---|---|---|
| %5 | 3.8e-4 | **3.1e-5** | ~5 ms |
| %20 | 6.0e-3 | **8.3e-4** | ~4 ms |

  Not: bu deney P1 matrisleriyle yapıldı ve P1'in bu mesh'teki ayrıklaştırma tabanı P2 etiketlerine göre 6.7e-3. Etiket kalitesine ulaşmak için düzeltme adımı **P2** matrisleriyle yapılmalı (`solve_dirichlet_eigenmodes`'taki assemble; birkaç on ms).
- **Uygulama:** `infer.py`'a `--refine {0,1,2}` seçeneği. Mesh'ten `Basis(MeshTri, ElementTriP2())`, düğüm alanlarından P2 DOF'lara interpolasyon (kenar ortası = uç değerlerin ortalaması), `splu(K)` + blok ters iterasyon + `eigh(VᵀKV, VᵀMV)`. Eğitimde isteğe bağlı "solver-in-the-loop" (Um et al. 2020). Çok-sadakatli eğitim: hedef $\lambda_{fine}/\lambda_{coarse}$ (Meng & Karniadakis 2020).

---

## 4. Pertürbasyon Teorisi Önselleri

### 4.1 Hadamard ve dejenere Hadamard

Basit $\lambda$ için $\lambda'[V]=-\int_{\partial\Omega}(\partial_n u)^2 V_n\,ds$. $m$-katlı küme için türevler
$$W_{ab} = -\int_{\partial\Omega}\partial_n u_a\,\partial_n u_b\,V_n\,ds\qquad(a,b=1..m)$$
matrisinin özdeğerleridir. Sıfırıncı mertebe "doğru" baz $W$'yu köşegenleştiren bazdır (Rellich; Henrot & Pierre 2018). **Dipol çifti sorununun kesin çözümü budur:** şekil pertürbasyonu çiftin yönünü $W$'nun özvektörleri olarak belirler.

### 4.2 Disk etrafında ikinci mertebe (bu not için türetildi ve doğrulandı)

$\partial\Omega: r=R(1+h(\theta))$, $h=\sum_k h_k e^{ik\theta}$, $j=j_{m1}$, $\lambda_0=j^2/R^2$. Sınır koşulunu $r=1+h$'de açıp Green özdeşliğiyle:

- **Birinci mertebe:** $\lambda^{(1)} = -\oint h(\partial_r u_0)^2 d\theta/\|u_0\|^2$ (m=0 için $=-2\lambda_0h_0$).
- **İkinci mertebe (genel):**
  $$\lambda^{(2)} = \oint\Big[-h\,\partial_r u_1 + \tfrac{h^2}{2}\partial_r u_0\Big]\partial_r u_0\,d\theta + \frac{(\lambda^{(1)})^2}{2\lambda_0}.$$
  Burada $u_1$, $-h\partial_r u_0$ sınır verisinin rezonant olmayan harmonik uzantısıdır; mod $k$ için $\partial_r u_1|_{r=1}=g_k\,jJ_k'(j)/J_k(j)$. Rezonant ($|k|=m$) kısmın özel çözümünün $\partial_r$'si $r=1$'de sıfırdır, çünkü $J_m'(j)+jJ_m''(j)=0$.
  Kontrol: dilatasyonda $3\lambda_0h_0^2$ ✓.
- **Temel mod, $h_0=0$ ve $h=\sum_k\varepsilon_k\cos(k\theta+\varphi_k)$:**
  $$\boxed{\lambda_1 \approx \frac{j_{01}^2}{R^2}\Big[1+\tfrac12\sum_k \varepsilon_k^2\,c_k\Big],\quad c_k = 1+2j_{01}\frac{J_k'(j_{01})}{J_k(j_{01})}}$$
  $c_1=-1$ (öteleme + alan artışı tam sadeleşir: Faber–Krahn eşitliği). $c_2..c_8$ = 2.78, 5.44, 7.78, 10.00, 12.15, 14.26, 16.35 ($\approx 2k$). Lord Rayleigh'in (*Theory of Sound*) neredeyse-dairesel membran sonucunun modern hâlidir.
- **Dipol çifti (m=1), 2×2 etkin matris:**
  $$H = \lambda_0 I + W_1 + B + \frac{W_1^2}{2\lambda_0},\qquad W_1 \text{ Hadamard},\ B_{ab}=\oint[-h\,\partial_r u_1[u_b]+\tfrac{h^2}{2}\partial_r u_b]\,\partial_r u_a .$$
  Birinci mertebede, $h_0=0$: $\lambda_{2,3}\approx\frac{j_{11}^2}{R^2}(1\mp\varepsilon_2)$. **Göreli ayrışma ≈ $2\varepsilon_2$** ve yalnız **2. harmoniğe** bağlıdır. Özvektörler elips eksenlerine hizalanır (açı $-\varphi_2/2$); düşük mod uzun eksen boyuncadır.
- **Simetri:** $n\ge3$ katlı dönme simetrisinde ($C_n$) dipol çifti 2-boyutlu bir indirgenemez temsil olduğundan **tam** dejeneredir. E3'te $k=3$ ve $k=5$ harmonikleri çifti ayırmadı (FEM'de 4 hane eşit).

**E3 doğrulaması (birim disk + tek harmonik, 512-gon, P2):**

| k | ε | λ₁ FEM | λ₁ 1. mert. | λ₁ 2. mert. | λ₂/λ₃ FEM | λ₂/λ₃ 2. mert. |
|---|---|---|---|---|---|---|
| 2 | 0.05 | 5.8033 | 5.7832 | **5.8033** | 13.995 / 15.467 | 13.997 / 15.465 |
| 2 | 0.1 | 5.8640 | 5.7832 | **5.8637** | 13.393 / 16.364 | 13.410 / 16.346 |
| 2 | 0.2 | 6.1103 | 5.7832 | 6.1051 | 12.402 / 18.538 | 12.530 / 18.403 |
| 3 | 0.1 | 5.9416 | 5.7832 | 5.9403 | 14.777 (çift) | 14.778 (çift) |
| 5 | 0.2 | 6.9256 | 5.7832 | 6.9399 | 17.039 (çift) | 17.083 (çift) |

$\varepsilon\le0.1$'de λ₁ hatası ≤5e-5, dipol çifti ≤0.12%.

**Veri setindeki smooth şekiller (E3b):** Üretici $R=0.035$, $a_k\sim U(\pm0.008)$, $k=2..7$ kullanıyor. Pertürbasyon büyük: rms $h/R$ medyan 0.23, max 0.33. 210 şeklin 45'i $r=0.015$'te kırpılmış. Bu yüzden 2. mertebe PT tek başına yeterli değil, ama **mükemmel bir taban çizgisi**:
- Sıfırıncı mertebe (R'li disk): medyan hata %29 / %10 / %31.
- **2. mertebe: medyan %2.1 / %5.0 / %2.8** (λ₁/λ₂/λ₃).
- 2. mertebe + 18 dönme-invaryant özellik ($|c_k|^2$, triad'lar $\mathrm{Re}(c_ac_b\bar c_{a+b})$, $|c_k|^4$) üzerinde ridge (5-kat CV, ~168 eğitim örneği): **medyan %1.5 / %1.5 / %2.7**.
- Dipol ekseni (1. mertebe tahmin): δ>%10 olan 189 şekilde medyan 5.5° hata. δ=%2–5 aralığında (8 şekil) 38.7°: burada yön yüksek mertebe triad'larla belirleniyor — Davis–Kahan ile tutarlı.

### 4.3 r(θ) katsayıları üzerinde öğrenilmiş model

- Veri manifoldunun boyutu küçük: smooth = 12 parametre (6 genlik + 6 faz), sharp = $2n_v\le24$ parametre. Mesh-tabanlı model, ≤24 boyutlu bir manifold üzerindeki bir haritayı binlerce düğümlük girdiden öğrenmeye çalışıyor.
- **Evrensel yıldız-parametrizasyon (E4):** sharp poligonların $r(\theta)$'sı köşelerde kıvrık (kink) olduğundan $|c_k|\sim k^{-2}$ (ölçülen eğim −1.96). Truncation L² hatası K=16'da ~1e-2, K=32'de ~5e-3. Ama özdeğerler sınır detayına çok duyarsız: kesilmiş-Fourier domain'in λ'ları poligonunkine göre **K=8: 1.4e-3–1.2e-2, K=16: 2.6e-4–8.9e-4, K=32: 7e-5–4e-4, K=64: 1.5e-5–1.9e-4**. Yani 33–65 reel sayı, λ'yı etiket doğruluğunda korur.
- **Eşdeğerlik:** dönme $\alpha$ → $c_k\mapsto e^{-ik\alpha}c_k$. λ'lar invaryantların fonksiyonudur ($|c_k|$, triad'lar); dipol yönü $\approx-\arg(c_2)/2$ (1. mertebe) → **dönmeye eşdeğer yapı serbestçe kodlanabilir**.
- **Model taslağı** (yeni dosya `src/models/shape_spectral.py`): girdi $\{c_k\}_{k\le32}$ (alan ağırlık merkezine göre); çıktı $\log(\lambda_k/\lambda_k^{PT2})$. Alanlar için $(\rho,\theta)$ referans ızgarasında POD katsayıları + 2×2 etkin matrisin özvektörüyle çift yönü. `pert.py`'deki $H$ matrisi (§7) doğrudan taban olarak kullanılabilir.
- Sınırlama: yalnız yıldız-şekiller; re-entrant köşeli tekillikler alanlarda ek terim ister (§6).

---

## 5. Alternatif Çözücüler ve Temsiller

| Yöntem | Güçlü yanı | Bu repoda |
|---|---|---|
| **MPS** (Fox, Henrici & Moler 1967; Betcke & Trefethen 2005) | Her köşede $J_{k\pi/\alpha}(\sqrt\lambda r)\sin(k\pi\theta/\alpha)$ + iç noktalar; alt uzay açısı $\sigma(\lambda)$ minimumu (GSVD). Poligonlarda **üstel yakınsama** (1e-10+). Moler–Payne (1968) a posteriori sınırı: iç denklemi tam sağlayan fonksiyonun sınır değeri küçükse yakın bir özdeğer garantilidir. | Sharp (7–12 köşe) için ideal **referans/validasyon** üreticisi. 100-gon "smooth" set için kötü (100 köşe). |
| **BIE / Fredholm determinantı** (Zhao & Barnett 2015; Bornemann 2010) | Pürüzsüz sınırda spektral doğruluk; yalnız sınır ayrıklaştırma. | "Smooth" şekiller trigonometrik $r(\theta)$ olarak (100-gon değil) tanımlanırsa uygun. |
| **Konformal / polar harita + spektral yöntem** (Driscoll & Trefethen 2002; Trefethen 2000; Boyd 2001) | $-\Delta_w v=\lambda|f'(w)|^2v$ diskte: **tüm şekiller aynı ızgarada** (padding/mask yok), FNO için doğal. Schwarz–Christoffel haritasında $u\circ f$ köşe tekilliğini "emer" ($r^{\pi/\alpha}\circ f\sim$ lineer); tekillik ağırlık $|f'|^2$'ye taşınır. | Yıldız-şekiller için polar harita çok basit (§3.4). |

**Eğitim verisini ucuzlatır mı?** Mevcut FEM etiketleri zaten yeterince doğru (λ medyan 5e-5 sharp, 1.3e-4 smooth; max 3.6e-4) ve ucuz (örnek başına ~0.43 s, çoğu gmsh). Bugünkü NN hata seviyesinde (≥1e-3) veri doğruluğu darboğaz değil. Alternatif çözücülerin değeri (i) bağımsız validasyon, (ii) λ'nın 1e-6 altı gerektiği durumlar, (iii) yapısal önseldir.

**"Öğrenilmiş MPS":** ağ, köşe ve iç özel çözüm kümesinin *ağırlıklarını/açılarını/merkezlerini* üretir; λ, $\sigma(\lambda)$ minimumundan (veya GSVD'den) gelir. PDE iç bölgede **tam** sağlanır, yalnız Dirichlet koşulu yaklaşık → Moler–Payne tipi sertifika. Risk: $\sigma(\lambda)$'nın λ'ya göre türevlenebilir minimizasyonu ve kötü koşullama (Betcke–Trefethen'in iç nokta düzeltmesi gerekli). Önerim: önce MPS'yi validasyon aracı olarak ekleyin; yapısal model olarak sonra.

---

## 6. Keskin Köşeler

- **Tekillik:** iç açısı $\alpha>\pi$ olan köşede $u\sim r^{\pi/\alpha}\sin(\pi\theta/\alpha)$, $\nabla u\sim r^{\pi/\alpha-1}$ sınırsız (Grisvard 1985). Uniform mesh'te P2 özdeğer hatası $O(h^{2\pi/\alpha})$ (pürüzsüzde $O(h^4)$). Graded/geometrik mesh optimal oranı geri getirir (Babuška, Kellogg & Pitkäranta 1979).
- **Veride (E6):** sharp poligonların **%86**'sında re-entrant köşe var; $\alpha_{max}$ medyan 209°, max 267°; $\pi/\alpha_{max}$ medyan 0.86, min 0.675; beklenen P2 λ-oranı medyan 1.72.
- **Yakınsama (E6, α=257.8°, beklenen oran 1.40):** uniform mesh gözlenen oran **1.42 / 1.45 / 1.42** ✓. Köşeye graded mesh aynı h'de ~30–200× daha küçük hata (h=2 mm: 3.4e-4 → 5.3e-6; h=0.5 mm: 4.7e-5 → 2.4e-7).
- **Etiketlere etkisi:** üreticinin *sınır-bandı* rafinmanı (hmin=1.2 mm tüm duvarda) köşeleri de dolaylı olarak inceltiyor. Etiket λ hatası sharp için medyan 5.5e-5, max 2.2e-4 — **yeterli**. Asıl etkiler şurada:
  (i) alanların gradyanı (Sobolev kaybı, SpectralNO'nun $L$ matrisi) köşede tekil;
  (ii) **pürüzsüz baz** tekilliği temsil edemez ve Ritz hatası M ile yavaş (cebirsel) azalır.
- **"Smooth" şekiller de köşeli (E7b):** 100 noktalı örnekleme, yüksek harmoniklerin çukurlarında eğrilik yarıçapını ~1 mm altına düşürüyor. **%98**'inde >200°, **%67**'sinde >230° iç açılı tepe var (medyan max 238°). Kırpılanlarda medyan 249°. 100-gon etiketleri aynı $r(\theta)$'nın 1000-gon versiyonundan %0.1–0.6 farklı (E3b). Yani `random_smooth` etiketi yanıltıcı: veri **iki tip köşeli şekil** içeriyor.
- **Düzeltmeler:**
  1. Üretici: smooth şekilleri gmsh `BSpline`/`Spline` ile veya ≥1000 noktayla tanımlayın; genlikleri $\sum_k k^2|a_k| < R$ (eğrilik pozitifliği) veya min-eğrilik yarıçapı ≥ 3·hmin olacak şekilde sınırlayın; kırpma yerine yeniden örnekleyin.
  2. Sharp şekiller için köşe grading'i (`Distance` alanı `PointsList` ile; `common.py`'deki `corner_grading` gibi) — λ için şart değil, alan gradyanları için faydalı.
  3. **Tekil zenginleştirme** (Fix, Gulati & Wakoff 1973; Strang & Fix 1973; PUM/GFEM: Babuška & Melenk 1997): her re-entrant köşe için $\psi^{sing}_{c,j} = \chi(r_c)\,r_c^{\pi/\alpha_c}\sin(\pi\theta_c/\alpha_c)\cdot p_j(x)$ SpectralNO bazına eklenir. Öğrenilebilir parametresi yok, türevleri analitik. **E12 (L-şekli, λ₁=9.6397238):** pürüzsüz ω·polinom bazı M=15→91'de %1.4 → %0.54'te takılıyor; **+6 köşe fonksiyonuyla M=15'te 2.2e-5, M=45'te 2.3e-7**.
  4. Converter: köşe listesini (konum, $\alpha$, açıortay yönü) geometriye yazın; GNOT için noktasal özellik olarak $r_c^{\pi/\alpha_c}\sin(\pi\theta_c/\alpha_c)$ (pürüzsüz değil ama doğru tekil davranışta — GNOT türev kullanmıyor).

---

## 7. Sayısal Deneyler (Sonuçlar)

Scriptler: `scratchpad/math_agent/` (repo'ya dahil değil). Veri: `gen.py` → üreticinin `generate_sample_data`'sı, `--seed 7`, 6 mod, 400 örnek; örnek başına ortalama 0.43 s (mesh+P2 eigsh), ortalama 1635 düğüm.

| ID | Script | Sonuç |
|---|---|---|
| E1 | `e1_invariance.py` | Aynı mesh s∈{0.5,2,10} + dönme + öteleme: $|\lambda s^2/\lambda_0-1|\le9\cdot10^{-15}$. Normalizörler (150 geometri): CV λ₁ 0.145; λ₁A 0.172; λ₁ρ² 0.119; λ₁T/A 0.061; **λ₁·max w 0.009** (1.436–1.491). |
| E2 | `stats.py` | Boşluk tablosu §1.1. CV(λ·s²) > CV(λ) (0.29 vs 0.14); $R^2(\log f_1\sim\log s)=0.003$. Faber–Krahn oranı 1.03–2.11; λ₂/λ₁ max 2.48 < 2.539. Mevcut küme kuralı çift (2,3)'ü geometrilerin %1'inde işaretliyor (≈ göreli λ boşluğu <%2). |
| E3 | `pert.py`, `e3_hadamard.py` | 2. mertebe formül, ε≤0.1'de λ₁ ≤5e-5, dipol ≤1.2e-3 göreli hata; C₃/C₅ simetrisinde çift tam dejenere. |
| E3b | `e3b_dataset_pert.py` | Smooth set: 0. mertebe %29/%10/%31 → 2. mertebe %2.1/%5.0/%2.8 → +ridge %1.5/%1.5/%2.7 (medyan). 100-gon vs 1000-gon etiket farkı %0.1–0.6. |
| E4 | `e4_fourier.py` | Sharp r(θ): $|c_k|\sim k^{-1.96}$; λ hatası K=16 ≤9e-4, K=32 ≤4e-4, K=64 ≤1.9e-4. |
| E5b | `e5b_morph.py` | 1e-3 radyal pertürbasyon: δ₂<%2 iken tekil vektör/çift alt uzay amplifikasyonu medyan 10.4×; δ₂>0.2 iken 0.4×. δ₃=0.059 örneğinde $\sin\Theta(P_{\le3})=6.7$e-3 (~6×). |
| E6 | `e6_fem_accuracy.py` | Etiket λ hatası (graded referans): sharp medyan 5.5e-5 (max 2.2e-4), smooth 1.3e-4 (max 3.6e-4). Köşe yakınsama oranı 1.42 (teori 1.40); graded aynı h'de 30–200× iyi. Kaba P2 H=4 mm: 17–24 ms, 2.7e-4/7.9e-4. |
| E7b | `e7b_smooth_corners.py` | "Smooth" 100-gonların %98'inde >200°, %67'sinde >230° iç açı; %21 kırpılmış. |
| E8 | `e8_ritz.py` | §3.1–3.2 tabloları. Düğüm quadratürü üst-sınırı bozuyor (14.657<14.682); 1024 alt-örnekleme λ göreli std %4–5, bias −%3…−7; donuk açı feature'ı dipolü 8.52'ye çekiyor. Ek: converter'ın node-mean merkezlemesi graded mesh'te diski 0.0047 kaydırıyor ve yarıçapı 0.9954 yapıyor ($1-r^2$ tipi analitik ω'lar normalize koordinatta uyumsuz olur → ω mesh'ten hesaplanmalı). |
| E9 | `e9_weights.py` | Düğümlerin %50'si alanın %22'sinde; max/min düğüm alanı 63; mod-1 enerji payı ağırlıksız %2.4 vs $L^2$ %0.5. |
| E10 | `e10_refine.py` | §3.5 tablosu: %5 → 3.1e-5, %20 → 8.3e-4 (1 ters iterasyon, P1, ~5 ms); P1 tabanı 6.7e-3 → P2 ile yapılmalı. |
| E11 | `e11_freqhead.py` | Yalnız λ_norm gören frekans başlığı: medyan %5.8, 90. persentil %16, MAE 0.39 GHz; tam formül 4e-16. |
| E12 | `e12_enrich.py` | L-şekli: pürüzsüz baz %1.4→%0.54 (M=15→91); +6 köşe fonksiyonu: 2.2e-5 (M=15) → 2.3e-7 (M=45). |

---

## 8. Uygulama Yol Haritası (dosya bazında)

1. **Hemen (1–2 gün):**
   - `dataset.py`/`collate`: `scale` ve `W` (lumped mass) batch'e.
   - `spectral_no.py`: `freq = c·sqrt(λ)/(2π·scale)`; `_MonotoneFreqHead`'i kaldır.
   - `lightning_module.py`: log-λ kaybı; $w$-ağırlıklı QR/rel-L2.
   - `configs/spectral_no.yaml`: `max_nodes: null` (+ docs/10 #7 bucket batching).
2. **Kısa vade:**
   - Üretici `n_eigen_modes: 5`, converter `lam_all`.
   - Flag kaybı (§1.3), göreli boşluklu küme raporlama.
   - Augmentasyon/principal feature tutarlılığı (§2.4).
   - `infer.py --refine` (P2 blok ters iterasyon, §3.5).
3. **Orta vade:**
   - SpectralNO v2: $\psi=\omega\,N(\mathrm{RFF}(x),g)$; segment-mesafesi veya torsiyon ω.
   - Eleman quadratürü; süreksiz feature'ları ψ'den çıkar; köşe zenginleştirmesi.
   - Uyum sağlandıktan sonra Ky Fan terimi.
   - Faber–Krahn/Ashbaugh–Benguria çıktı parametrizasyonu.
4. **Araştırma:**
   - r(θ)-Fourier modeli + PT2 taban çizgisi; polar referans domain'de POD.
   - MPS validasyon seti (sharp); smooth şekiller için BIE.

---

## 9. Referanslar

- Arnold, D. N., David, G., Filoche, M., Jerison, D., Mayboroda, S. (2019). *Computing spectra without solving eigenvalue problems.* SIAM J. Sci. Comput. 41(1), B69–B92.
- Armentano, M. G., Durán, R. G. (2003). *Mass-lumping or not mass-lumping for eigenvalue problems.* Numer. Methods PDE 19(5), 653–664.
- Ashbaugh, M. S., Benguria, R. D. (1992). *A sharp bound for the ratio of the first two eigenvalues of Dirichlet Laplacians and extensions.* Ann. Math. 135, 601–628.
- Babuška, I., Kellogg, R. B., Pitkäranta, J. (1979). *Direct and inverse error estimates for finite elements with mesh refinements.* Numer. Math. 33.
- Babuška, I., Melenk, J. M. (1997). *The partition of unity method.* IJNME 40.
- Babuška, I., Osborn, J. (1991). *Eigenvalue problems.* Handbook of Numerical Analysis II.
- Banerjee, U., Osborn, J. (1990). *Estimation of the effect of numerical integration in finite element eigenvalue approximation.* Numer. Math. 56, 735–762.
- Berry, M. V., Wilkinson, M. (1984). *Diabolical points in the spectra of triangles.* Proc. R. Soc. Lond. A 392, 15–43.
- Betcke, T., Trefethen, L. N. (2005). *Reviving the method of particular solutions.* SIAM Review 47(3), 469–491.
- Boffi, D. (2010). *Finite element approximation of eigenvalue problems.* Acta Numerica 19.
- Bornemann, F. (2010). *On the numerical evaluation of Fredholm determinants.* Math. Comp. 79.
- Boyd, J. P. (2001). *Chebyshev and Fourier Spectral Methods.* Dover.
- Davis, C., Kahan, W. M. (1970). *The rotation of eigenvectors by a perturbation. III.* SIAM J. Numer. Anal. 7, 1–46.
- Driscoll, T. A., Trefethen, L. N. (2002). *Schwarz–Christoffel Mapping.* Cambridge.
- E, W., Yu, B. (2018). *The Deep Ritz method.* Commun. Math. Stat. 6.
- Edelman, A., Arias, T., Smith, S. (1998). *The geometry of algorithms with orthogonality constraints.* SIMAX 20.
- Fix, G., Gulati, S., Wakoff, G. I. (1973). *On the use of singular functions with finite element approximations.* J. Comput. Phys. 13.
- Fox, L., Henrici, P., Moler, C. (1967). *Approximations and bounds for eigenvalues of elliptic operators.* SIAM J. Numer. Anal. 4.
- Fumagalli, I., Manzoni, A., Parolini, N., Verani, M. (2016). *Reduced basis approximation and a posteriori error estimates for parametrized elliptic eigenvalue problems.* ESAIM: M2AN 50.
- Greenfeld, D., Galun, M., Kimmel, R., Yavneh, I., Basri, R. (2019). *Learning to optimize multigrid PDE solvers.* ICML.
- Grisvard, P. (1985). *Elliptic Problems in Nonsmooth Domains.* Pitman.
- Hadamard, J. (1908). *Mémoire sur le problème d'analyse relatif à l'équilibre des plaques élastiques encastrées.*
- Henrot, A. (2006). *Extremal Problems for Eigenvalues of Elliptic Operators.* Birkhäuser.
- Henrot, A., Pierre, M. (2018). *Shape Variation and Optimization.* EMS.
- Horger, T., Wohlmuth, B., Dickopf, T. (2017). *Simultaneous reduced basis approximation of parameterized elliptic eigenvalue problems.* ESAIM: M2AN 51(2), 443–465.
- Kaba, S.-O., Mondal, A. K., Zhang, Y., Bengio, Y., Ravanbakhsh, S. (2023). *Equivariance with learned canonicalization functions.* ICML.
- Kantorovich, L. V., Krylov, V. I. (1958). *Approximate Methods of Higher Analysis.*
- Kato, T. (1966/1995). *Perturbation Theory for Linear Operators.* Springer.
- Knyazev, A. V. (2001). *Toward the optimal preconditioned eigensolver: LOBPCG.* SISC 23.
- Knyazev, A. V., Argentati, M. E. (2002). *Principal angles between subspaces in an A-based scalar product.* SISC 23.
- Lagaris, I. E., Likas, A., Fotiadis, D. I. (1998). *Artificial neural networks for solving ordinary and partial differential equations.* IEEE TNN 9.
- Lamberti, P. D., Lanza de Cristoforis, M. (2004). *A real analyticity result for symmetric functions of the eigenvalues of a domain dependent Dirichlet problem for the Laplace operator.* J. Nonlinear Convex Anal. 5, 19–42.
- Luz, I., Galun, M., Maron, H., Basri, R., Yavneh, I. (2020). *Learning algebraic multigrid using graph neural networks.* ICML.
- Machiels, L., Maday, Y., Oliveira, I. B., Patera, A. T., Rovas, D. V. (2000). *Output bounds for reduced-basis approximations of symmetric positive definite eigenvalue problems.* C. R. Acad. Sci. Paris 331.
- Meng, X., Karniadakis, G. E. (2020). *A composite neural network that learns from multi-fidelity data.* J. Comput. Phys. 401.
- Moler, C. B., Payne, L. E. (1968). *Bounds for eigenvalues and eigenvectors of symmetric operators.* SIAM J. Numer. Anal. 5.
- von Neumann, J., Wigner, E. (1929). *Über das Verhalten von Eigenwerten bei adiabatischen Prozessen.* Phys. Z. 30.
- Pfau, D., Petersen, S., Agarwal, A., Barrett, D., Stachenfeld, K. (2019). *Spectral Inference Networks.* ICLR.
- Pólya, G., Szegő, G. (1951). *Isoperimetric Inequalities in Mathematical Physics.* Princeton.
- Puny, O., Atzmon, M., Ben-Hamu, H., Misra, I., Grover, A., Smith, E., Lipman, Y. (2022). *Frame averaging for invariant and equivariant network design.* ICLR.
- Rayleigh, J. W. S. (1877/1894). *The Theory of Sound.*
- Rellich, F. (1969). *Perturbation Theory of Eigenvalue Problems.* Gordon & Breach.
- Rozza, G., Huynh, D. B. P., Patera, A. T. (2008). *Reduced basis approximation and a posteriori error estimation for affinely parametrized elliptic coercive PDEs.* Arch. Comput. Methods Eng. 15.
- Rvachev, V. L., Sheiko, T. I. (1995). *R-functions in boundary value problems in mechanics.* Appl. Mech. Rev. 48.
- Strang, G. (1972). *Variational crimes in the finite element method.* In: The Mathematical Foundations of the FEM.
- Strang, G., Fix, G. (1973). *An Analysis of the Finite Element Method.* Prentice-Hall.
- Sukumar, N., Srivastava, A. (2022). *Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks.* CMAME 389, 114333.
- Trefethen, L. N. (2000). *Spectral Methods in MATLAB.* SIAM.
- Um, K., Brand, R., Fei, Y., Holl, P., Thuerey, N. (2020). *Solver-in-the-loop: learning from differentiable physics to interact with iterative PDE-solvers.* NeurIPS.
- Xu, J., Zhou, A. (2001). *A two-grid discretization scheme for eigenvalue problems.* Math. Comp. 70(233), 17–25.
- Ye, K., Lim, L.-H. (2016). *Schubert varieties and distances between subspaces of different dimensions.* SIMAX 37.
- Zhao, L., Barnett, A. (2015). *Robust and efficient solution of the drum problem via Nyström approximation of the Fredholm determinant.* SIAM J. Numer. Anal. 53.

---

## 🔗 Bağlantılar

- [[09_PHYSICS_BACKGROUND]] · [[11_ORTHOGONALITY_ANALYSIS]] · [[13_RAYLEIGH_ANALYSIS]] · [[10_IMPROVEMENT_IDEAS]] · [[04_MODEL_ARCHITECTURE]]

#matematik #spektral-teori #davis-kahan #hadamard #rayleigh-ritz #reduced-basis #köşe-tekilliği
