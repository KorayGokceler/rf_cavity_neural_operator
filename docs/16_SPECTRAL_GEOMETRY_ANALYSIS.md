# 16 — Spektral Geometri Analizi: Şekil Pertürbasyonu, İzoperimetrik İnvaryantlar, Simetri, Köşeler ve Konformal Formülasyon

> **Kapsam:** Araştırma notu. Model, eğitim veya üretici kodunda değişiklik yok, eğitim koşusu yok.
> **Soru:** Dirichlet problemi $-\Delta u=\lambda u$ ($\Omega$), $u|_{\partial\Omega}=0$ için ilk $K=3$ özçifti öğrenen bu repo, spektral geometrinin hangi **kesin sonuçlarını** hedef, önsel, çıktı kısıtı veya özellik olarak kullanabilir? [[14_MATHEMATICAL_IMPROVEMENTS]]'deki iddialar doğru mu?
> **Kanıt:** Bu not için yazılmış küçük CPU deneyleri (§9). Veri: repo üreticisi `src/data_gen/dataset_generator.py` (`generate_sample_data`, varsayılan mesh argümanları, `--seed 11`, **10 mod**, `eigen_sigma=0`) ile **200 geometri** (99 sharp, 101 smooth), ortalama 0.58 s/örnek (mesh + P2 eigsh + torsiyon + konformal yarıçap). docs/14 `--seed 7` kullanmıştı; burada bağımsız bir örneklem var.
> **Referans commit:** `fd2d27c`. Scriptler: `scratchpad/math_geom/` (repo'ya dahil değil).

---

## 🧭 Özet — Öncelik Sırasına Göre Öneriler

| # | Öneri | Kanıt (§9) | Etkilenen dosyalar |
|---|---|---|---|
| **1** | **$\lambda_1$'i torsiyon (landscape) fonksiyonuna göre parametrize et:** $\log\lambda_1 = \log\frac{j_{01}^2}{4} - \log\max w + r_1$, $-\Delta w=1$, $w|_{\partial\Omega}=0$. Ağ yalnız küçük rezidüel $r_1$'i öğrensin. $w$ tek bir seyrek lineer çözümdür (özdeğer çözümünden ucuz); aynı zamanda sınırda sıfır, pürüzsüz, köşe davranışı doğru bir **düğüm özelliği**dir. | $\lambda_1\max w$: **CV %0.94** (sharp %0.35, smooth %0.91), aralık 1.422–1.495. Sabit taban $j_{01}^2/4$ ile $\log$-rezidüel std **0.0093**; 10 ucuz invaryant üzerinde ridge (5-kat CV) ile **0.0037** (medyan |hata| %0.12). Ham $\log\lambda_1$ std'si 0.143. | `dataset_converter.py` ($w$, $\max w$, $T=\int w$ hesapla/sakla; $w$'yi feature kolonu yap), `dataset.py` (batch'e ekle), `models/gnot.py` ve `models/spectral_no.py` (λ₁ başlığı: taban + rezidüel), `lightning_module.py` (kayıp $r_1$ üzerinde) |
| **2** | **Dipol çifti için doğru öğrenme nesnesi:** tekil $(\lambda_2,u_2),(\lambda_3,u_3)$ yerine çiftin **ortalaması** $\bar\lambda=(\lambda_2+\lambda_3)/2$ ve **spin-2 ayrışma vektörü** $s=(S_{11}-S_{22},\,2S_{12})$ (dönmede $2\alpha$ döner). $\lambda_{2,3}=\bar\lambda\mp|s|/2$, düşük modun ekseni $\tfrac12\arg s$. Hedefler hedef alanlardan hesaplanır: dipol momentleri $d_i=\int u_i(x-\bar x)$, $G=\sum_{i=2,3}d_id_i^\top$, $S=\sum_{i=2,3}\lambda_i d_id_i^\top$ — **çift içindeki dönmeye göre invaryant ve kesişmede bile analitik**. Önsel: $|s|/\bar\lambda\approx0.95\,e_{tor}$ (torsiyon-ağırlıklı atalet anizotropisi). | Konik kesişme (E4): $\lambda_2,\lambda_3$ kesişmede **kıvrık** (ikinci fark ±3800), $S_{11}-S_{22}$ **lineer** (ikinci fark −0.95); döngü sonunda $u_2\to-u_2$ (monodromi, örtüşme −1.000). Veri (E3b): ayrışma–$e_{tor}$ korelasyonu **0.967 (sharp)**, 0.73 (smooth); eksen hatası medyan **2.8°**. | `dataset_converter.py` ($d_i$, $G$, $S$ hedefleri), `lightning_module.py` (çift kaybı; `detect_clusters` yerine), model başlıkları |
| **3** | **Sert çıktı kısıtları (yalnız sıkı ve kanıtlı olanlar):** $\lambda_1\le\lambda_2\le\lambda_3$ (softplus artışları); **Ashbaugh–Benguria** $\lambda_2/\lambda_1\le j_{11}^2/j_{01}^2=2.539$ (sigmoid). Gevşek olanları (Faber–Krahn, Pólya, Pólya–Szegő, Kohler-Jobin, PPW/Yang) yalnız **tanılama/ihlal sayacı** olarak kullan. $\lambda_3/\lambda_1$ için kanıtlı sıkı sınır yok; sayısal üst değer ≈3.2 (Levitin–Yagudin) — yumuşak ceza en fazla. | $\lambda_2/\lambda_1$ veride 1.555–**2.491** (sınırın %98'i). $\lambda_3/\lambda_1\le3.016$. Yang sınırının λ₃'te kullanılan payı en çok %73, PPW-HP %70 → kısıt olarak işe yaramaz. | `models/*` (çıktı parametrizasyonu), `infer.py` (ihlal raporu) |
| **4** | **Üretici:** `--n_eigen_modes` ≥ 6 (λ₄… Weyl ve boşluk için), torsiyon/landscape istatistiklerini, köşe listesini (konum, iç açı) ve smooth şekillerin **gerçek** $r(\theta)$ katsayılarını H5'e yaz. Ayrı bir **simetrik test seti** ($C_n$, $n\ge3$: dipol tam dejenere; $D_1$/$C_2$: ayrık) ve disk yakını şekiller ekle; bunlar dejenerelik davranışının birim testleridir. | E2/E4: $C_3,C_4,C_5,C_7$ şekillerinde çift FEM'de 4+ hane eşit; $(3,4)$ harmonikli şekilde çift **3. mertebede** ayrılıyor (docs/14 bunu kaçırıyor). $(\lambda_4-\lambda_3)/\lambda_3<\%10$: %3. | `dataset_generator.py`, `dataset_converter.py`, `validate_data.py` |
| **5** | **Daha çok mod öğrenilecekse** hedefleri 3-terimli Weyl tahminine göre normalize et: $N_W(\lambda)=\frac{A\lambda}{4\pi}-\frac{L\sqrt\lambda}{4\pi}+\sum_i\frac{\pi^2-\alpha_i^2}{24\pi\alpha_i}$, $N_W(\lambda_k^W)=k-\tfrac12$. K=3 için **gereksiz** (landscape λ₁'de 10× daha iyi). | $\lambda_k/\lambda_k^W$ CV: k=1: %4.7, k=3: %7.7, k=7–10: %3.0–3.6 (karşılaştırma $\lambda_kA$: %18 → %5.6). Tek başına tahmin hatası k=1'de medyan %9.4, k=10'da %2.3. | `dataset_converter.py` (A, L, köşe açıları), model başlıkları |
| **6** | **PT2 (disk etrafında 2. mertebe) tabanını kullanma** (docs/14 öneri #6'nın bu kısmı): üretici genlikleri pertürbasyon serisinin yakınsama bölgesinin çok dışında. PT2 yalnız çift yönü/simetri seçim kuralları için teorik rehber. | Smooth set: $\max|h_\theta|$ medyan **2.37** (≫1), $\max|h|$ medyan 0.51. PT2 medyan hata λ₁ **%3.0**, λ₂ %5.8, λ₃ %4.6 — landscape tabanı (λ₁) %0.9. | — |
| **7** | **Köşe tekillikleri:** re-entrant köşelerde $r^{\pi/\alpha}\sin(\pi\theta/\alpha)$ özellikleri/zenginleştirmesi (docs/14 #7 ile aynı yönde; burada üs sayısal olarak doğrulandı). | E5b. | `dataset_converter.py`, `models/spectral_no.py` |
| 8 | **Uzun vade — konformal disk formülasyonu:** $-\Delta_w v=\lambda|f'(w)|^2v$ birim diskte; tüm şekiller **aynı disk mesh'ini ve aynı katılık matrisini** paylaşır, şekil yalnız skaler ağırlık $\rho=|f'|^2$ (veya sınırdaki $\log|f'|$) ile girer. Konformal yarıçap $r_c$ ile $\lambda_1\le j_{01}^2/r_c^2$ (Pólya–Szegő). | E6: aynı disk mesh'inde konformal çözüm, görüntü domain FEM'i ile **2·10⁻⁶** uyumlu (6 şekil). Veride $\lambda_1r_c^2/j_{01}^2\in[0.844,0.995]$, CV %3.3. | yeni model/dataset yolu (araştırma) |

**Tek cümlelik ana mesaj:** Bu veri seti için en güçlü tek önsel docs/14'ün de bulduğu $\lambda_1\max w$ sabitliğidir — ama doğru taban değeri $1.5$ değil **$j_{01}^2/4=1.4458$**'dir; dipol çifti için ise tekil özvektörler değil, **analitik** olan çift-ortalaması + spin-2 ayrışma tensörü öğrenilmelidir.

---

## 0. Notasyon

- $\Omega\subset\mathbb R^2$ sınırlı, basit bağlantılı; $A=|\Omega|$, $L=|\partial\Omega|$, $\rho$ = iç yarıçap, $r_c$ = (maksimal) konformal yarıçap.
- $0<\lambda_1<\lambda_2\le\lambda_3\le\dots$, $\|u_k\|_{L^2}=1$. $j_{mn}$: $J_m$'nin $n$. sıfırı ($j_{01}=2.4048$, $j_{11}=3.8317$).
- Torsiyon fonksiyonu $w$: $-\Delta w=1$, $w|_{\partial\Omega}=0$; $M=\max w$, $T=\int_\Omega w$ (torsiyonel rijitlik).
- Disk etrafında pertürbasyon: $\partial\Omega=\{r=R(1+h(\theta))\}$, $h=\sum_k\varepsilon_k\cos(k\theta+\varphi_k)$; $d_n(j)=j\,J_n'(j)/J_n(j)$.
- Üretici: smooth şekiller $r(\theta)=0.035+\sum_{k=2}^{7}a_k\cos(k\theta+\varphi_k)$, $a_k\sim U(\pm0.008)$, $r\ge0.015$ kırpma, 100 noktalı poligon; sharp şekiller 7–12 köşeli yıldız poligonlar (`randint(7,13)`).

---

## 1. Şekil Pertürbasyonu: Türetmeler

### 1.1 Birinci ve ikinci mertebe Hadamard formülü (genel)

$\partial\Omega$'yu normal boyunca $x\mapsto x+tV(x)\,n(x)$ ile hareket ettirelim ($V$: normal hız). $u_t=u+tu'+\tfrac{t^2}{2}u''+\dots$ ve $\lambda_t=\lambda+t\lambda'+\tfrac{t^2}{2}\lambda''$ açılımını sınır koşulu $u_t(x+tVn)=0$'a koyunca:

$$u'=-V\,\partial_nu,\qquad u''=-V^2\partial_{nn}u-2V\partial_nu' \quad(\partial\Omega\text{ üzerinde}).$$

$u=0$ ve $\Delta u=-\lambda u=0$ olduğundan sınırda $\partial_{nn}u=-H\,\partial_nu$ ($H$ = eğrilik, dışbükeyde pozitif). Denklemler $-\Delta u'=\lambda u'+\lambda'u$, $-\Delta u''=\lambda u''+2\lambda'u'+\lambda''u$. Green özdeşliğini $u$ ile kullanıp normalizasyonu $\int uu'=0$ seçersek:

$$\boxed{\lambda'=-\oint_{\partial\Omega}(\partial_nu)^2V\,ds,\qquad \lambda''=2\!\int_\Omega\!\big(|\nabla u'|^2-\lambda u'^2\big)+\oint_{\partial\Omega}H\,(\partial_nu)^2V^2\,ds.}$$

(Birinci terim $=2\oint u'\partial_nu'$; $u'$, sınır verisi $-V\partial_nu$ olan ve $u$'ya dik rezonanssız Helmholtz çözümüdür.) Seçilen normalizasyon sonucu değiştirmez: $u'\to u'+cu$ dönüşümünde iki terimin değişimi birbirini götürür. Bu, Henrot & Pierre (2018) ve Grinfeld (2010)'daki ikinci varyasyon formülünün salt normal deformasyon hâlidir (teğet ve ivme bileşenleri Hadamard yapı teoremi gereği yalnız ek bir $\lambda'[Z]$ terimi verir). Tarihçe: Hadamard (1908) birinci mertebe; Rayleigh (*Theory of Sound*) neredeyse-dairesel zar; Joseph (1967) parametre/domain bağımlılığının sistematik açılımı; Grinfeld (2010, 2013) hareketli yüzeyler kalkülüsüyle çok katlı ve ikinci mertebe varyasyonlar.

### 1.2 Disk: temel mod ($m=0$)

$R=1$, $V=h$, $H=1$, $u_0=J_0(jr)/(\sqrt\pi J_1(j))$, $j=j_{01}$. $h=\varepsilon\cos k\theta$ için $\partial_nu_0=-j/\sqrt\pi$, $u'=\frac{\varepsilon j}{\sqrt\pi}\frac{J_k(jr)}{J_k(j)}\cos k\theta$, $\partial_nu'=\frac{\varepsilon j}{\sqrt\pi}d_k\cos k\theta$. Böylece $\lambda'=0$ ($k\ge1$) ve $\lambda''=j^2\varepsilon^2(1+2d_k)$:

$$\boxed{\lambda_1\approx\frac{j_{01}^2}{R^2}\Big[1+\tfrac12\sum_k\varepsilon_k^2c_k\Big],\qquad c_k=1+2j_{01}\frac{J_k'(j_{01})}{J_k(j_{01})}}$$

— **docs/14'teki formül doğrudur** (bağımsız türetme + FEM, §3). İkinci mertebede harmonikler arasında çapraz terim yoktur (Dirichlet-to-Neumann operatörü Fourier'de köşegen). Sayılar: $c_1=-1$, $c_2..c_8=2.783,5.435,7.783,10.001,12.152,14.262,16.347$.

Ek gözlemler (docs/14'te yok):
- **Asimptotik:** $x J_k'(x)/J_k(x)=k-\frac{x^2}{2(k+1)}-\frac{x^4}{8(k+1)^2(k+2)}+\dots$, dolayısıyla
  $c_k = 2k+1-\frac{j_{01}^2}{k+1}-\frac{j_{01}^4}{4(k+1)^2(k+2)}+O(k^{-3})$ (k=4: 7.788 vs 7.783). docs/14'teki "$c_k\approx2k$" kaba bir yuvarlamadır; doğru büyüme $2k+1-j^2/(k+1)$.
- **Faber–Krahn normalizasyonu:** $A=\pi R^2(1+\tfrac12\sum\varepsilon_k^2)$ olduğundan
  $$\frac{\lambda_1A}{\pi j_{01}^2}-1\approx\sum_k\varepsilon_k^2\,(1+d_k)\ \ge0,$$
  $k=1$ katsayısı tam sıfırdır (öteleme) ve $k\ge2$ için pozitiftir: Faber–Krahn eşitsizliğinin ikinci mertebe (lineer kararlılık) hâli. Bu, Brasco–De Philippis–Velichkov (2015) kantitatif Faber–Krahn'ının disk etrafındaki "katsayılarıdır".
- **Hata mertebesi:** Tek harmonikte $\varepsilon\to-\varepsilon$ dönüşümü şekli $\pi/k$ kadar döndürür, dolayısıyla $\lambda(\varepsilon)$ **çifttir** ve PT2 hatası $O(\varepsilon^4)$'tür. Birden çok harmonik olduğunda ise $a+b=c$ üçlüleri (triad) **üçüncü mertebe** terim üretir: $\lambda^{(3)}\propto\varepsilon_a\varepsilon_b\varepsilon_c\cos(\varphi_a+\varphi_b-\varphi_c)$. Üretici $k=2..7$ kullandığı için (2+3=5, 2+4=6, 3+4=7, 2+5=7) PT2'nin baskın hatası üçüncü mertebedir (E2: $h=\varepsilon(\cos2\theta+\cos3\theta+\cos5\theta)$ için tek kısım $\approx-65\,\varepsilon^3$, ε=0.1'de λ'nın %1.1'i; PT2'nin çift kısım hatası yalnız 1e-3).

### 1.3 Dejenere durum: dipol çifti (Rellich)

$m\ge1$ için $\lambda_0=j_{m1}^2$ iki katlıdır; taban $u_a=J_m(jr)e_a(\theta)$, $e_c=\cos m\theta$, $e_s=\sin m\theta$, $\|u_a\|^2=N=\pi J_m'(j)^2/2$. Aynı açılım, çözülebilirlik koşulunun $2\times2$ matris formunu verir. Ayrıntılar:
- Birinci mertebe: $W_{ab}=-\frac1N\oint h\,\partial_ru_a\partial_ru_b=-\frac{2j^2}{\pi}\oint h\,e_ae_b\,d\theta$.
- $u_1$'in rezonant ($|n|=m$) kısmı özel çözüm $\lambda^{(1)}\,rJ_m'(jr)/(2j)\,e(\theta)$ ile karşılanır. $J_m'(j)+jJ_m''(j)=0$ olduğundan bu çözümün $r=1$'deki radyal türevi sıfırdır (docs/14'teki ifade doğru).
- $\int_0^1r^2J_mJ_m'\,dr=-J_m'(j)^2/(2j)$ integraliyle, ikinci mertebede etkin matris
  $$H=\lambda_0I+W+R+\frac{W^2}{2\lambda_0},\qquad R_{ab}=\frac{2j^2}{\pi}\Big[\oint D_\perp(he_a)\,(he_b)\,d\theta+\tfrac12\oint h^2e_ae_b\,d\theta\Big]$$
  elde edilir. Burada $D_\perp$, Fourier modu $n$'yi $d_n(j)$ ile çarpan ve $|n|=m$ modlarını silen simetrik operatördür. $R$ simetriktir, $H$'nin özdeğerleri $\lambda_{2,3}$'ü $O(\varepsilon^2)$ doğrulukla verir. Bu, docs/14 §4.2'deki $H=\lambda_0I+W_1+B+W_1^2/(2\lambda_0)$ ile **aynıdır** (bağımsız olarak doğrulandı, $B=R$).
- **Birinci mertebe ayrışma:** $h=\varepsilon_2\cos(2\theta+\varphi_2)$ ($\varphi_2=0$) için $(\cos\theta,\sin\theta)$ tabanında $W=-j_{11}^2\varepsilon_2\,\mathrm{diag}(1,-1)$ (izsiz), özdeğerler $\mp j_{11}^2\varepsilon_2$ ⇒ $(\lambda_3-\lambda_2)/\bar\lambda=2|\varepsilon_2|+O(\varepsilon^2)$ ✓. Düşük mod uzun eksen boyunca salınır (açı $-\varphi_2/2$) ✓.
- **Yüksek mertebe seçim kuralı (docs/14'te eksik):** $p$. mertebede ayrışma, toplamı $\pm2$ olan $p$ harmoniğin çarpımlarından gelir (açısal momentum korunumu: $e^{\pm i\theta}$'yı $e^{\mp i\theta}$'ya bağlamak için net $\pm2$ gerekir). Yani
  - 1. mertebe: yalnız $\varepsilon_2$;
  - 2. mertebe: $\varepsilon_k\varepsilon_{k+2}$ çiftleri, yani (2,4), (3,5), (4,6), (5,7) (üreticide hepsi var) ve $\varepsilon_1\varepsilon_1$, $\varepsilon_1\varepsilon_3$;
  - 3. mertebe: ör. (3,4): $4-3=1$, $1+1=2$ yoluyla $\varepsilon_3^2\varepsilon_4$-tipi terimler.
  E2 bunu doğruluyor: $\varepsilon(\cos3\theta+\cos5\theta)$ ve $\varepsilon(\cos4\theta+\cos6\theta)$ **2. mertebede** ayrılır (PT2 FEM'e %1–3 yakın); $\varepsilon(\cos3\theta+\cos4\theta)$ PT2'de ayrışmaz ama FEM'de ayrışır: ε=0.05'te 0.0103, ε=0.1'de 0.081 (oran 7.9 ≈ 2³, **üçüncü mertebe**). docs/14'teki "ayrışma yalnız 2. harmoniğe bağlıdır" cümlesi yalnız birinci mertebede doğrudur.

### 1.4 Seri ne zaman yakınsar?

- **Analitiklik (Rellich 1937–42; Kato 1966, Bölüm VII):** Şekli referans diske $x\mapsto(1+\varepsilon h(\theta))x$ ile çekersek elde edilen operatör ailesi $\varepsilon$'da holomorftur (tip (B)). Basit bir özdeğer ve projektörü $|\varepsilon|<\varepsilon_c$ diskinde analitiktir. $\varepsilon_c$ iki şeyle sınırlıdır: (i) karmaşık $\varepsilon$ düzleminde en yakın **dallanma (exceptional) noktası**, yani komşu bir özdeğerle karmaşık çarpışma (boşluk küçüldükçe küçülür); (ii) çekme dönüşümünün eliptikliği/tersinirliğini kaybetmesi. Burada ölçek $\varepsilon\max|h|$ değil, **eğim** $\varepsilon\max|h_\theta|\sim k\varepsilon$'dur (Jacobian $h_\theta$ içerir).
- **Rellich teoremi (1 parametre):** Dejenere bir özdeğerde bile 1-parametreli analitik bir yolda özdeğer dalları ve özvektörler analitik seçilebilir. **Çok parametreli** ailelerde bu yanlıştır (§5.3, konik nokta). Orada yalnızca kümenin simetrik fonksiyonları ve toplam projektör analitiktir (Lamberti & Lanza de Cristoforis 2004).
- **Sayısal gözlem (E2):** PT2 hatası/ε⁴: k=2'de ≈ −0.59, k=3'te ≈ −2.06 (sabit). k=7'de ise +36 → +13; hata ε değil **kε** ile ölçekleniyor: k=7, ε=0.2'de hata %2.1. Veri setinde smooth şekillerin $\max|h_\theta|$ medyanı 2.37 (max 4.36), $\max|h|$ medyanı 0.51. Yani üretici genlikleri pratik yakınsama bölgesinin **dışında**. PT2'nin veride %3–6 hata vermesinin nedeni budur (§3).

---

## 2. İzoperimetrik Eşitsizlikler ve Boyutsuz İnvaryantlar

### 2.1 Kullanılan kesin sonuçlar

| Eşitsizlik | Form (2D) | Eşitlik/uç durum | Kaynak |
|---|---|---|---|
| Faber–Krahn | $\lambda_1A\ge\pi j_{01}^2$ | disk | Faber 1923, Krahn 1925; kantitatif: Brasco–De Philippis–Velichkov 2015 |
| Krahn–Szegő | $\lambda_2A\ge2\pi j_{01}^2$ | iki eş disk | Pólya–Szegő 1951 |
| Payne–Pólya–Weinberger / **Ashbaugh–Benguria** | $\lambda_2/\lambda_1\le3$ (PPW 1956); $\lambda_2/\lambda_1\le j_{11}^2/j_{01}^2=2.539$ (AB 1992) | disk | PPW 1956; Ashbaugh–Benguria 1992 |
| Hile–Protter / PPW-tipi | $\lambda_{k+1}\le(1+\tfrac4d)\frac1k\sum_{i\le k}\lambda_i$ | — | Hile–Protter 1980 |
| Yang | $\sum_{i\le k}(\lambda_{k+1}-\lambda_i)^2\le\tfrac4d\sum_{i\le k}(\lambda_{k+1}-\lambda_i)\lambda_i$ | — | Yang 1991 |
| Pólya (torsiyon) | $\lambda_1T/A\le1$ | ince şeritlerde yaklaşılır | Pólya 1948; van den Berg–Ferone–Nitsch–Trombetti 2016 |
| Saint-Venant | $T\le A^2/(8\pi)$ | disk | Pólya–Szegő 1951 |
| Kohler-Jobin | $\lambda_1\sqrt T\ge j_{01}^2\sqrt{\pi/8}$ | disk | Kohler-Jobin 1978 |
| Landscape / torsiyon maksimumu | $\lambda_1M\ge1$ (Filoche–Mayboroda tipi); üstten $\lambda_1M\le c_d$ | — | Filoche–Mayboroda 2012; van den Berg–Carroll 2009; Vogt 2019; Henrot–Lucardesi–Philippin 2018 |
| Pólya–Szegő (konformal yarıçap) | $\lambda_1\le j_{01}^2/r_c^2$ | disk | Pólya–Szegő 1951 |
| İç yarıçap | $\lambda_1\rho^2\le j_{01}^2$ (monotonluk); dışbükeyde $\ge\pi^2/4$ (Hersch 1960); basit bağlantılıda $\ge1/4$ (Makai 1965), $\ge0.6197$ (Bañuelos–Carroll 1994) | — | — |
| Pólya (çevre, dışbükey) | $\lambda_1\le\frac{\pi^2}{4}\big(\frac{L}{A}\big)^2$ | — | Pólya 1960 |
| Pólya sanısı / Li–Yau | $\lambda_k\ge4\pi k/A$ (sanı; karo döşeyen domainler ve **disk** için kanıtlı); $\frac1k\sum_{i\le k}\lambda_i\ge2\pi k/A$ (Berezin–Li–Yau) | — | Pólya 1961; Li–Yau 1983; Filonov–Levitin–Polterovich–Sher 2023 |

### 2.2 Veride sıkılık (E1, 200 geometri)

| Boyutsuz büyüklük | teorik | ortalama | **CV** | min–max | sharp CV | smooth CV |
|---|---|---|---|---|---|---|
| $\lambda_1M$ | $\ge1$ | 1.459 | **0.0094** | 1.422–1.495 | **0.0035** | 0.0091 |
| $\lambda_1M/(j_{01}^2/4)$ | disk=1 | 1.009 | 0.0094 | 0.984–1.034 | | |
| $\lambda_1r_c^2/j_{01}^2$ | $\le1$ | 0.955 | 0.033 | 0.844–0.995 | 0.017 | 0.036 |
| $\lambda_1T/A$ | $\le1$ | 0.661 | 0.061 | 0.565–0.714 | 0.017 | 0.046 |
| $\lambda_1\sqrt T/(j_{01}^2\sqrt{\pi/8})$ | $\ge1$ | 1.091 | 0.062 | 1.008–1.274 | 0.020 | 0.052 |
| $\lambda_1\rho^2/j_{01}^2$ | $\le1$ | 0.682 | 0.130 | 0.492–0.862 | | |
| $\lambda_1A/(\pi j_{01}^2)$ | $\ge1$ | 1.320 | 0.182 | 1.030–1.931 | 0.051 | 0.131 |
| $T/(A^2/8\pi)$ | $\le1$ | 0.720 | 0.217 | 0.428–0.958 | | |
| $\lambda_1(A/L)^2\cdot4/\pi^2$ | $\le1$ (dışbükey) | 0.488 | 0.084 | 0.387–0.561 | | |
| $\lambda_2/\lambda_1$ | $\le2.539$ | 2.103 | 0.102 | 1.555–**2.491** | | |
| $\lambda_3/\lambda_1$ | (≈3.2 sayısal) | 2.630 | 0.073 | 1.996–3.016 | | |
| $\lambda_2A/(2\pi j_{01}^2)$ | $\ge1$ | 1.370 | 0.124 | 1.129–1.836 | | |
| $\lambda_3/\mathrm{HP}$ | $\le1$ | 0.567 | 0.097 | 0.459–0.700 | | |
| $\lambda_3/\mathrm{Yang}$ | $\le1$ | 0.611 | 0.084 | 0.484–0.732 | | |
| $\lambda_kA/(4\pi k)$, k=1…10 | Pólya sanısı $\ge1$ | 1.91 → 1.39 | | min 1.24 | | |

Ek: dışbükey şekiller (15, hepsi sharp): $\lambda_1M=1.4467\pm0.2\%$ (disk değeri 1.4458). Dışbükey olmayanlar diskin **iki yanına** da yayılıyor (0.984–1.034 × disk).

**Yorum — hangileri hedef/önsel, hangileri kısıt?**
1. **Tek başına tahminci:** $\lambda_1\approx\frac{j_{01}^2}{4M}$, log-hata std **0.93%**, en kötü sapma %2.6. $\log\lambda_1\sim\log M$ regresyonunun eğimi **−1.007**; boyut analiziyle beklenen −1 neredeyse tam. Ardından konformal yarıçap (%2.8), Pólya $T/A$ (%4.8) ve iç yarıçap (%7.7) geliyor. Alan (Faber–Krahn) en kötüsü (%13). Yani **"alan normalizasyonu" (docs/14'ün $\lambda_1A$ parametrizasyonu) λ₁ için zayıf bir tabandır**; landscape 14× daha iyi.
2. **Neden bu kadar sabit? Kesin özdeşlik:** $\int_\Omega u_1=\int u_1(-\Delta w)=\int(-\Delta u_1)w=\lambda_1\int wu_1$, dolayısıyla
   $$\lambda_1=\frac{\int_\Omega u_1}{\int_\Omega w\,u_1}\quad\Longrightarrow\quad\lambda_1M=\Big\langle\frac{w}{M}\Big\rangle_{u_1}^{-1}\ \ge1,$$
   burada $\langle\cdot\rangle_{u_1}$, $u_1\ge0$ ağırlıklı ortalamadır. Sabitlik, normalize torsiyonun temel mod altındaki ortalamasının şekle zayıf bağlı olması demektir (disk: $\langle1-r^2\rangle_{J_0}=4/j_{01}^2=0.692$). Bu aynı zamanda $w$'nin $u_1$'e ne kadar benzediğinin ölçüsüdür. Özdeşlik eğitimde de kullanılabilir: tahmin edilen $\hat u_1$ ile $\hat\lambda_1=\int\hat u_1/\int w\hat u_1$ (hata birinci mertebe, ama çarpanı küçük).
3. **Kısıt olarak:** yalnız **Ashbaugh–Benguria** hem kanıtlı hem sıkı (veri max'ı sınırın %98.1'i). Faber–Krahn (min 1.03), Pólya $T/A$ (max 0.71), Pólya–Szegő (max 0.995) sağlanıyor ama çıktı katmanına gömülmeleri ancak ek girdi ($A$, $T$, $r_c$) gerektirir ve etkin sınıra nadiren yaklaşılır. PPW-HP/Yang (%70–73) ve Pólya sanısı (λ₁'de 1.49×) **boştur**.
4. **λ₂, λ₃ için** tek bir invaryant yeterli değil: $\lambda_2M$ CV %10, $\lambda_3M$ %6.7, $(\lambda_2+\lambda_3)/(2\lambda_1)$ %6.3. Landscape tabanı + 10 ucuz invaryant (T/(AM), L²/A, r_c²/M, A/M, $e_{tor}$, $e_{tor}^2$, alan anizotropisi, Weyl köşe sabiti, max iç açı kosinüsü, tip) ile ridge CV rezidüel std: $\log\lambda_2$ **0.021**, $\log\lambda_3$ 0.040, $\log\bar\lambda$ 0.017, göreli ayrışma 0.056. Bireysel λ₂, λ₃'ün zor kısmı **ayrışmadır** (§5), ortalama değil.

---

## 3. docs/14 İddialarının Doğrulanması / Düzeltilmesi

| # | docs/14 iddiası | Sonuç | Sayılar (bu not) |
|---|---|---|---|
| a | $\lambda_1\approx\frac{j_{01}^2}{R^2}[1+\frac12\sum\varepsilon_k^2c_k]$, $c_k=1+2j_{01}J_k'/J_k$ | ✅ **Doğru** (bağımsız türetme §1.2; $c_2..c_8$ değerleri aynı). Kapalı form ile sayısal PT2 kodu $10^{-15}$ uyumlu. | FEM (1600-gon, P2, disk tabanı 2.6e-6 düzeltildi), $h=\varepsilon\cos k\theta$: ε=0.05'te göreli hata k=2: −3.7e-6, k=3: −1.3e-5, k=5: +2.8e-5, k=7: +2.1e-4; ε=0.1'de −5.8e-5 / −2.1e-4 / +3.6e-4 / +2.7e-3. |
| a′ | "$c_k\approx2k$" | ⚠️ Kaba. $c_k=2k+1-\frac{j_{01}^2}{k+1}-\frac{j_{01}^4}{4(k+1)^2(k+2)}+\dots$ | k=8: 16.347 (asimptotik 16.347) |
| a″ | (örtük) PT2 hatası küçük/üst mertebe | ⚠️ **Eksik:** tek harmonikte hata $O(\varepsilon^4)$ (çiftlik), ama üreticinin çok-harmonikli şekillerinde **$O(\varepsilon^3)$ triad** terimi baskın; ölçek parametresi $k\varepsilon$. | triad (2,3,5): tek kısım $-65\varepsilon^3$; ε=0.1'de %1.1 (PT2 çift-kısım hatası 1e-3). Veri smooth set PT2 medyan hata λ₁ %3.0, λ₂ %5.8, λ₃ %4.6 (docs/14: %2.1/%5.0/%2.8, farklı örneklem). |
| b | Dipol çifti ≈ $2\varepsilon_2$ göreli ayrışır | ✅ **Birinci mertebede doğru**; ❌ veri setindeki şekiller için yanıltıcı. | Tek harmonik: ε₂=0.025/0.05/0.1/0.2 → FEM 0.0500/0.1000/0.1996/0.3967. Veri (smooth, 101): FEM ayrışması ile $2|\varepsilon_2|$ korelasyonu **0.75**, medyan mutlak hata 0.060 (ortalama ayrışma 0.24, yani ~%25). PT2 2×2: korelasyon 0.88, hata 0.041. **Torsiyon-ağırlıklı anizotropi** $0.95\,e_{tor}$: sharp 0.967, smooth 0.73. |
| b′ | "ayrışma yalnız 2. harmoniğe bağlıdır" | ❌ Yalnız 1. mertebe. 2. mertebede $(k,k+2)$ çiftleri, 3. mertebede ör. (3,4). | (3,5), ε=0.1: FEM 0.772, PT2 0.823; (4,6), ε=0.1: 1.181 / 1.219; (3,4): FEM 0.0103 (ε=.05) → 0.081 (ε=.1), PT2 = 0 (3. mertebe). |
| c | 3- veya 5-katlı simetride çift tam dejenere | ✅ **Doğru ve genelleşir:** $C_n$, $n\ge3$ ise $m=1$ çifti her zaman dejenere; açısal indisi $m$ olan çift ancak $2m\equiv0\pmod n$ ise ayrılabilir (§5.1). | FEM: $C_3$ (ε=0.2) λ₂=λ₃=15.0479; $C_4$ 16.2205 (çift); $C_5$ 17.0395; $C_7$ 18.1847 (hepsi ≥4 hane eşit). Konformal: $f=w+0.12i\,w^4$ ($C_3$) λ₂=λ₃=14.10581; $f=w+0.2w^2$ (simetrisiz) ayrık. **Ama:** üreticinin şekillerinde simetri yok, bu yüzden veri setinde tam dejenerelik yok (ayrışma <%2: %0.5). |
| d | $\lambda_1\cdot\max w$ neredeyse sabit | ✅ **Doğru** (bağımsız örneklem). | CV %0.94 (docs/14 %0.9), 1.422–1.495 (docs/14 1.436–1.491), medyan 1.456 (docs/14 1.454). |
| d′ | "2D'de teorik ≈ $1+d/4=1.5$" | ❌ **Yanlış atıf.** $1+d/4$, Arnold–David–Filoche–Jerison–Mayboroda (2019)'un **potansiyelli** Schrödinger operatörlerinde yerelleşmiş durumlar için buldukları ampirik katsayıdır; kesin bir değer değildir. $V=0$'da doğal referans diskin kesin değeri $j_{01}^2/4=1.4458$ (kare: $2\pi^2\cdot0.07367=1.454$). Kanıtlı olanlar: $\lambda_1M\ge1$ ve boyuta bağlı üst sınırlar (van den Berg–Carroll 2009; Vogt 2019: $\lambda_1M\le1+d/8+c\sqrt d$ tipi, $c\approx0.6$). **1.5 taban alınırsa sistematik %2.8–3.7 bias oluşur.** | veri ortalaması 1.459 = 1.009 × $j_{01}^2/4$; dışbükeyler 1.4467 ± %0.2. |
| e | $\lambda_2/\lambda_1\le2.539$ (AB) veride sağlanıyor, max 2.48 | ✅ | max 2.491, min 1.555. |
| f | $\lambda_1T/A$ Pólya ≤1, aralık 0.567–0.712 | ✅ | 0.565–0.714. |
| g | $\lambda_1\rho^2\le j_{01}^2$; dışbükeyde $\ge\pi^2/4$ (Hersch) | ✅ | $\lambda_1\rho^2\in[2.85,4.98]$, dışbükeyler ≥3.57 > 2.47. |
| h | Etkin matris $H=\lambda_0I+W_1+B+W_1^2/(2\lambda_0)$ | ✅ (bağımsız türetme §1.3, $B$ simetrik) | Tek harmonik k=2, ε=0.1: PT2 13.410/16.346 vs FEM 13.393/16.364 (docs/14 tablosuyla aynı). |
| i | Öneri 6: "2. mertebe PT ... mükemmel bir taban çizgisi" | ⚠️ **Landscape tabanı tarafından domine ediliyor.** PT2 yalnız smooth şekiller için tanımlı ve orada λ₁'de %3; landscape her iki tipte de %0.9, sharp'ta %0.35. | E1, E3b |
| j | "smooth" şekiller aslında köşeli | ✅ Bağımsız destek: 100-gonların Weyl köşe sabiti $\sum(\pi^2-\alpha^2)/(24\pi\alpha)$ ortalama **0.203**; pürüzsüz basit bağlantılı sınır için değer 1/6=0.167. Fark $\approx\sum\beta_i^2/(24\pi^2)$ (β = dönme açısı) büyük dönüşlerden geliyor. | E1 (Weyl) |

---

## 4. Weyl Asimptotiği: Sınır ve Köşe Düzeltmeleri

**Formül.** Isı izi $Z(t)=\sum_ke^{-\lambda_kt}\sim\frac{A}{4\pi t}-\frac{L}{8\sqrt{\pi t}}+c_0$. Poligon için $c_0=\sum_i\frac{\pi^2-\alpha_i^2}{24\pi\alpha_i}$ (Kac 1966; van den Berg & Srisatkunarajah 1988/1990), pürüzsüz basit bağlantılı sınır için $c_0=1/6$ (McKean–Singer 1967). "Pürüzsüz köşeli" sınırlar için bu katsayının süreksiz olduğu bilinir (Nursultanov–Rowlett–Sher; "heat trace anomaly"). Laplace ters dönüşümüyle düzgünleştirilmiş sayma fonksiyonu
$$N(\lambda)\approx\frac{A\lambda}{4\pi}-\frac{L\sqrt\lambda}{4\pi}+c_0 .$$
Nokta-nokta iki terimli Weyl yasası Ivrii (1980) tarafından periyodik bilardo yörüngelerinin ölçüsü sıfır koşuluyla kanıtlandı. Sabit terim ise yalnız ortalama anlamdadır.

**Deney (E1-Weyl, 200 şekil, 10 mod):** $\lambda_k^W$, $N_W(\lambda)=k-\frac12$'den bulundu.

| mod k | 1 | 2 | 3 | 4 | 5 | 7 | 10 |
|---|---|---|---|---|---|---|---|
| 1-terim medyan \|hata\| | 0.72 | 0.61 | 0.48 | 0.47 | 0.41 | 0.37 | 0.31 |
| 2-terim | 0.32 | 0.042 | 0.11 | 0.038 | 0.050 | 0.029 | 0.028 |
| 3-terim (köşeli) | 0.094 | 0.074 | 0.069 | 0.049 | 0.032 | 0.026 | 0.023 |
| CV($\lambda_k/\lambda_k^W$) | 0.047 | 0.059 | 0.077 | 0.052 | 0.045 | 0.036 | 0.033 |
| CV($\lambda_kA$) | 0.182 | 0.124 | 0.151 | 0.095 | 0.101 | 0.069 | 0.056 |

**Sonuç:**
- Düşük modlarda Weyl bir **tahminci değildir**: λ₁'de %9 hata, landscape'in 10 katı. Nedeni, $N$'nin basamak fonksiyonu olması ve düşük modlarda salınımlı (periyodik yörünge) kalanın $c_0$ ile aynı mertebede olmasıdır. Dipol çiftinde "k−½" kuralı da tutarsızdır.
- Ama **normalizör** olarak işe yarar: $\lambda_kA$'ya göre CV'yi 2–4× düşürür. K ≥ 6 mod öğrenilirse, $\log(\lambda_k/\lambda_k^W)$ iyi bir boyutsuz hedeftir; köşe terimi sharp poligonlarda gerçekten katkı veriyor (0.20–0.36).
- Pólya sanısı ($\lambda_kA\ge4\pi k$) tüm k≤10 için veride geniş payla sağlanıyor (oran ≥1.24). Kısıt olarak boş.

---

## 5. Simetri ve Dejenerelik

### 5.1 Grup-teorik sınıflandırma

$\Omega$'nın simetri grubu $G\subset O(2)$ sonlu ise ($C_n$ dönmeler veya $D_n$ = dönmeler + yansımalar) her özuzay $G$'nin reel bir temsilidir. $-\Delta$ ile $G$ komüte ettiği için özuzaylar (jenerik olarak) **indirgenemez** temsillerdir:
- $C_n$'nin reel indirgenemez temsilleri: trivial; ($n$ çiftse) işaret temsili; ve $\rho_m(\text{rot}_{2\pi/n})=R(2\pi m/n)$ 2-boyutlu temsiller, $1\le m<n/2$. Reel katsayılı operatörde $e^{\pm im\theta}$ karmaşık eşlenik çifti tek bir 2-boyutlu reel temsilde birleşir.
- $D_n$: 2 veya 4 adet 1-boyutlu temsil ve $E_m$ ($1\le m<n/2$) 2-boyutlu temsiller.

**Kural:** Disk modu $J_m(j_{mk}r)(\cos m\theta,\sin m\theta)$ simetri $C_n$/$D_n$'ye indirgenince, $m\bmod n\notin\{0,n/2\}$ ise (yani $2m\not\equiv0\pmod n$) **2-boyutlu indirgenemez temsilde kalır ⇒ tam dejenere**. Aksi hâlde iki 1-boyutlu temsile ayrılır. Ayrıca $m'\equiv\pm m\pmod n$ olan disk modları birbirine karışır.
- Dipol ($m=1$): $n\ge3$ ise **daima** dejenere. $C_2$ (elips, $\cos2\theta$) veya $D_1$ (tek yansıma) ile ayrılır.
- Kuadrupol ($m=2$): $C_3$'te dejenere ($m\equiv-1\pmod3$), $C_4$'te ayrılabilir ($2m\equiv0\pmod4$; iki ayrı 1-boyutlu temsil).
- $f(w)=w+aw^n$ konformal görüntüsü $C_{n-1}$ simetriktir: E6'da $w+0.12i\,w^4$ ($C_3$) ve $w+0.1w^7$ ($C_6$) için λ₂=λ₃ tam (5 hane), $w+0.15w^3$ ($C_2$) için ayrık (12.26 / 16.30).
- **Kodboyut:** Simetri yokken çakışma kodboyut-2'dir (von Neumann & Wigner 1929; genel operatör aileleri için Teytel 1999). Simetri korunan bir ailede **farklı** temsillere ait seviyeler kodboyut-1'de (tek parametreyle) çaprazlanabilir. Aynı 2-boyutlu temsile ait çift ise kodboyut-0'da (her yerde) dejeneredir.

### 5.2 Jenerik basitlik

Uhlenbeck (1976) ve Albert (1975): jenerik (Baire anlamında) pürüzsüz domain için tüm Dirichlet özdeğerleri basittir. Hillairet & Judge (2009): **≥4 köşeli basit bağlantılı poligonların hemen hemen hepsi** basit spektruma sahiptir. Bu, sharp aile (7–12 köşe) için doğrudan geçerlidir. Sonuç: üreticinin iki ailesinde de tam dejenerelik ölçü-sıfırdır. Sorun **yakın** dejenereliktir. Veride göreli dipol ayrışması <%2: %0.5, <%5: %7.0, <%10: %20 ($(\lambda_3-\lambda_2)/\lambda_2$ ile docs/14: <%2 %1.0, <%5 %4.8). Küçük boşlukta CDF ~ δ² davranışı (kodboyut-2 ⇒ Wigner tipi seviye itmesi, $p(\delta)\propto\delta$) ile uyumlu: 0.02→0.05 aralığında ×14, 0.05→0.10 aralığında ×2.9 (en küçük kutuda yalnız 1 örnek; istatistik gürültülü).

### 5.3 Konik kesişme ve öğrenme için sonuçları

**Deney E4:** $h=a\cos2\theta+b\sin2\theta+0.1\cos3\theta$. $a=b=0$'da $C_3$ simetrisi ⇒ çift tam dejenere. PT'ye göre $(\lambda_2,\lambda_3)\approx\bar\lambda\mp j_{11}^2\sqrt{a^2+b^2}$, yani $(a,b)$ düzleminde bir **koni** (diabolik nokta; Berry & Wilkinson 1984).
- ρ₀=0.03 çemberi (12 nokta): ayrışma 0.8526–0.8541 (PT 0.8809, fark $a\,c^2$-tipi 3. mertebe), yani koni izotrop.
- $u_2$'nin dipol ekseni tam olarak $\varphi/2$ dönüyor: 0°, 15°, …, 180°. Sürekli izlenen $u_2$ döngü sonunda **$-u_2$** oluyor (örtüşme −1.0000000). Bu, Berry fazı π'dir: özvektör demeti Möbius demetidir ve **küresel sürekli işaret seçimi yoktur**.
- $b=0$ kesiti: λ₂, λ₃ tepe noktasında $|a|$ gibi kıvrık (ikinci farklar −3757, +3825); $\lambda_2+\lambda_3$ pürüzsüz (67.8, eğrilik); $S_{11}-S_{22}$ ($S=\sum\lambda_id_id_i^\top$) **lineer**: −0.403, −0.302, …, +0.402 (ikinci fark −0.95).

**Öğrenme için sonuçlar:**
1. Şekil → sıralı $(\lambda_2,\lambda_3)$ haritası Lipschitz'tir ama kodboyut-2 bir küme üzerinde **konik tekillik** taşır. Pürüzsüz bir ağ tepeyi yuvarlar; hata ~ (ağın çözünürlük ölçeği) × eğim.
2. Şekil → $u_2$ haritası kesişme etrafında **süreksizdir** (monodromi). $\pm$ işaret-değişmez kayıp bile yetmez, çünkü $u_2$ ile $u_3$ tepe etrafında yer değiştirir. docs/14'ün "projektör/flag" önerisi bunun genel çözümüdür.
3. **Doğru nesne (somut, düşük boyutlu):** Riesz projektörü $P_{23}$ analitiktir (Kato; Lamberti–Lanza de Cristoforis). Onun iki küçük özeti hedef alanlardan hesaplanabilir:
   $$D=[d_2\ d_3],\ d_i=\int_\Omega u_i(x-\bar x)\,dx;\qquad G=DD^\top,\quad S=D\Lambda D^\top,\ \Lambda=\mathrm{diag}(\lambda_2,\lambda_3).$$
   $D\to DQ$ ($Q\in O(2)$, çift içi dönme veya işaret) altında $G$ ve $S$ **değişmez**. $P_{23}$ analitik olduğu için kesişmede bile analitiktirler. $\lambda_{2,3}$ genelleştirilmiş özdeğer problemi $Sv=\mu Gv$'den tam olarak geri gelir ($G^{-1}S=D^{-\top}\Lambda D^\top\sim\Lambda$). Özvektörlerin eksenleri de buradan çıkar. Pratik parametrizasyon: $\bar\lambda=\tfrac12\mathrm{tr}(G^{-1}S)$ ve spin-2 vektör $s$ (traceless kısım). $s$ dönme-eşdeğerdir ($\Omega\to R_\alpha\Omega$ ⇒ $s\to R_{2\alpha}s$). Yani ağ çıktısının eşdeğerliği de kodlanabilir.
4. **Önsel:** birinci mertebede $s\propto$ şeklin spin-2 momenti. Veri (E3b): lineer tahminci $|s|/\bar\lambda\approx0.95\,e_{tor}$, $e_{tor}=|(J_{11}-J_{22},2J_{12})|/\mathrm{tr}J$, $J=\int w\,(x-\bar x)(x-\bar x)^\top/T$. Korelasyon sharp 0.967 (rezidüel 0.027), smooth 0.73, hepsi 0.82. Alan ataleti ($J=\int(x-\bar x)(x-\bar x)^\top$) daha zayıf (0.91 / 0.51). Düşük modun ekseni ile $J_{tor}$'un uzun ekseni arasındaki açı: ayrışma>0.1'de medyan **2.8°** (alan ataleti 6.1°), ayrışma<0.05'te 13.5°.
5. Kayıp tasarımı: $\mathcal L_{pair}=|\log\hat{\bar\lambda}-\log\bar\lambda|^2+\|\hat s-s\|^2/\bar\lambda^2$ + (alanlar için) projektör kaybı. `detect_clusters` sert eşiği gereksiz hâle gelir, çünkü kayıp boşluk kapansa da sürekli kalır.

---

## 6. Köşeler

### 6.1 Tekillikler (Kondrat'ev 1967; Grisvard 1985)

Açısı $\alpha$ olan bir köşede, köşe merkezli polar koordinatlarda
$$u(r,\theta)=\sum_{n\ge1}c_n\,J_{n\pi/\alpha}(\sqrt\lambda\,r)\sin\frac{n\pi\theta}{\alpha}=c_1r^{\pi/\alpha}\sin\frac{\pi\theta}{\alpha}+O(r^{\min(2\pi/\alpha,\ \pi/\alpha+2)}).$$
$\alpha>\pi$ (re-entrant) ise $\nabla u\sim r^{\pi/\alpha-1}\to\infty$ ve $u\notin H^2$. P2 özdeğer hatası uniform mesh'te $O(h^{2\pi/\alpha})$ olur. Kondrat'ev ağırlıklı Sobolev uzayları ve Mellin dönüşümüyle bu açılımı genel eliptik problemler için kurdu; Grisvard (1985/2011) poligonlar için standart referanstır.

**Deney E5b** (veri setindeki en büyük iç açılı sharp poligon, köşeye $2\cdot10^{-5}$ graded mesh):
- Poligon: 11 köşe, en büyük iç açı $\alpha=271.86°$ ⇒ $\pi/\alpha=0.6621$.
- $u$'nun açıortay boyunca $\log|u|$–$\log r$ eğimi: **mod 1: 0.6625** ($r\in[10^{-4},6\cdot10^{-4}]$), 0.6626 ($r\in[10^{-3},5\cdot10^{-2}]$); **mod 2: 0.6630** (yakın), 0.7009 (uzak: bir sonraki terimler devreye giriyor).
- Yani teori %0.1 içinde doğrulandı. Veri setinde sharp poligonların %85'i (99'dan 84'ü) re-entrant köşeli (docs/14: %86). Bu şekillerde $\nabla u_k$ sınırsızdır ve alanların $H^1$ normu köşe çevresinde yavaş yakınsar.

### 6.2 Poligonlar için özdeğer asimptotiği

- **Düzgün N-gon (alan π):** Grinfeld & Strang (2004, 2012) hareketli yüzeyler kalkülüsüyle
  $$\frac{\lambda_1(P_N)}{j_{01}^2}=1+\frac{4\zeta(3)}{N^3}+\frac{(12-2j_{01}^2)\zeta(5)}{N^5}+O(N^{-6})$$
  (sonraki katsayılar: Jones 2017 altıncı mertebeye kadar; Berghaus–Georgiev–Monien–Radchenko 2024 çoklu zeta değerleriyle $n\le14$).
  **E5a** (graded P2, $h$-değişimi ~2e-8):
  | N | 5 | 6 | 7 | 8 | 10 | 12 | 16 |
  |---|---|---|---|---|---|---|---|
  | $\lambda_1$ (FEM) | 6.022138 | 5.917418 | 5.866449 | 5.838491 | 5.811260 | 5.799370 | 5.789992 |
  | $(\lambda_1/j_{01}^2-1)N^3$ | 5.165 | 5.014 | 4.938 | 4.896 | 4.855 | 4.836 | 4.820 |
  | $O(N^{-5})$ serisinin göreli hatası | 2.6e-3 | 8.7e-4 | 3.5e-4 | 1.6e-4 | 4.2e-5 | 1.4e-5 | 2.5e-6 |
  | hata × $N^6$ | 40.8 | 40.7 | 40.9 | 41.1 | 41.6 | 41.8 | 41.5 |

  $(\lambda_1/j_{01}^2-1)N^3\to4\zeta(3)=4.808$ ✓. Kalan hata tam olarak $\approx41.5/N^6$ ölçekleniyor, yani seri yapısı ve normalizasyon (alan π) doğrulandı. Bir sonraki katsayı $C_6\approx41.5$ (göreli) olarak ölçüldü.
- **Pratik anlam:** 7–12 köşeli *düzgüne yakın* poligonlarda köşe sayısının λ₁'e etkisi $4\zeta(3)/N^3$ = %1.4 (N=7) … %0.28 (N=12). Bu, sharp ailede "köşe sayısı" özelliğinin λ₁'e doğrudan katkısının küçük olduğunu gösterir. Asıl varyans yarıçap düzensizliğinden (düşük harmonikler) gelir. $\lambda_1M$'nin sharp ailede %0.35 CV ile en sabit olması da bununla tutarlıdır.
- Weyl köşe terimi (§4) λ'yı ortalama olarak $\sim c_0$ kadar kaydırır. Sharp setinde $c_0\in[0.20,0.36]$.

### 6.3 Öneri

docs/14 §6'daki köşe zenginleştirmesi ve üreticinin köşe listesini saklaması önerisi geçerlidir. Bu notun ekledikleri:
1. **Özellik:** her re-entrant köşe için $\phi_c(x)=\chi(r_c)\,r_c^{\pi/\alpha_c}\sin(\pi\theta_c/\alpha_c)$; ayrıca **$w$ (torsiyon)** zaten doğru köşe üssüne sahiptir (aynı Kondrat'ev açılımı, kaynak terimi pürüzsüz), dolayısıyla $w$ düğüm özelliği köşe bilgisini bedava taşır.
2. **Metrik:** köşe yakınında gradyan-tabanlı (Sobolev, $H^1$) alan kayıplarını ağırlıklandırmayın. Tekil gradyan kaybı domine eder.

---

## 7. Konformal Formülasyon

### 7.1 Türetme

$\Omega$ basit bağlantılı, $f:\mathbb D\to\Omega$ Riemann haritası, $v=u\circ f$. $\Delta_w(u\circ f)=|f'|^2(\Delta_zu)\circ f$ ve Dirichlet enerjisi konformal-değişmez olduğundan
$$-\Delta_wv=\lambda\,|f'(w)|^2\,v\ \ (\mathbb D),\qquad v|_{\partial\mathbb D}=0,\qquad \lambda=\min_v\frac{\int_{\mathbb D}|\nabla v|^2}{\int_{\mathbb D}|f'|^2v^2}.$$
Şekil yalnız **ağırlık** $\rho=|f'|^2=e^{2\,\mathrm{Re}\log f'}$ ile girer. $\log f'$ diskte analitik olduğundan $\rho$, sınırdaki tek bir periyodik reel fonksiyon $\log|f'(e^{i\phi})|$ ile (Poisson/Hilbert dönüşümü) tamamen belirlenir. Bu, 1-boyutlu, dönme-eşdeğer bir şekil kodlamasıdır (dönme = $\phi$'de kayma).

**Konformal nakil (transplantation) sınırları:**
- $v=u_0^{\mathbb D}$ deneme fonksiyonu ve $|f'|^2$'nin alt-harmonikliği ($\frac1{2\pi}\oint|f'(re^{i\phi})|^2d\phi\ge|f'(0)|^2$) ile Pólya–Szegő: $\lambda_1\le j_{01}^2/r_c^2$, $r_c=|f'(0)|$ (merkez serbest seçilir; en iyisi maksimal konformal yarıçap).
- Daha keskini: $f(w)=\sum_na_nw^n$ ($a_1=1$) için açısal ortalama $\sum n^2|a_n|^2r^{2n-2}$ olduğundan
  $$\lambda_1\le\frac{j_{01}^2}{\sum_nn^2|a_n|^2\mu_n},\qquad\mu_n=\frac{\int_0^1r^{2n-1}J_0(j_{01}r)^2dr}{\int_0^1rJ_0(j_{01}r)^2dr}.$$
  Bu, **Taylor katsayılarından kapalı-form** bir üst sınırdır ("konformal momentler").

### 7.2 Sayısal/öğrenme avantajları ve deney

**E6:** Tek bir sabit disk mesh'i (60 385 P2 DOF; katılık matrisi bir kez kurulur), şekle göre yalnız ağırlıklı kütle matrisi $M_\rho$ değişir.

| $f(w)$ | konformal λ₁,λ₂,λ₃ | görüntüde FEM | göreli fark λ₁ | nakil sınırı / λ₁ |
|---|---|---|---|---|
| $w+0.2w^2$ | 5.36818, 13.18430, 14.06233 | 5.36817, 13.18428, 14.06230 | 1.9e-6 | 1.041 |
| $w+0.15w^3$ ($C_2$) | 5.64205, 12.26036, 16.30478 | 5.64204, 12.26033, 16.30475 | 1.8e-6 | 1.009 |
| $w+0.12i\,w^4$ ($C_3$) | 5.71837, 14.10581, 14.10581 | 5.71836, 14.10579, 14.10579 | 1.8e-6 | 1.003 |
| $w+0.15w^2+0.08w^3+0.03w^5$ | 5.46023, 13.14850, 14.54533 | 5.46022, 13.14847, 14.54530 | 1.9e-6 | 1.034 |
| $w+0.1w^7$ ($C_6$) | 5.75859, 14.51092, 14.51092 | 5.75858, 14.51090, 14.51090 | 1.6e-6 | 1.0005 |
| $w+0.3w^3$ | 5.25370, 9.87913, 16.11220 | 5.25370, 9.87911, 16.11219 | 5.8e-7 | 1.034 |

- **Doğrulama:** formülasyon kesin; farklar mesh tabanında (~2e-6).
- **Öğrenme avantajları:** (i) tüm şekiller **aynı ızgarada**: padding/mask yok, FNO/spektral yöntem doğal (Fourier–Chebyshev diskte); (ii) çıktı $v$'ler sabit domain'de yaşar, dolayısıyla POD/indirgenmiş baz sabit bir uzayda; (iii) girdi 1-boyutlu sınır fonksiyonu $\log|f'|$ veya katsayılar $a_n$; (iv) nakil sınırı analitik bir taban verir (burada λ₁'e %0.05–4 yakın).
- **Poligonlar (Schwarz–Christoffel):** $f'(w)=C\prod_k(1-w/w_k)^{\alpha_k/\pi-1}$ (Driscoll & Trefethen 2002). Köşede $z-z_c\sim(w-w_k)^{\alpha/\pi}$ ⇒ $r^{\pi/\alpha}\sim|w-w_k|$. Yani $u\circ f$'nin **baskın tekil terimi $w$'de lineer, pürüzsüz** hâle gelir. Tekillik ağırlığa geçer: $\rho\sim|w-w_k|^{2(\alpha/\pi-1)}$ (re-entrant'ta sıfıra gider, dışbükeyde integre edilebilir tekillik). Kalan terimler $|w|^{1+2\alpha/\pi}$ mertebesinde, $\alpha\ge\pi/2$ için $C^2$. Cureton & Kuttler (1999) düzgün poligonları tam olarak böyle, diskte ağırlıklı Rayleigh–Ritz ile çözdü.
- **Maliyet/risk:** SC parametre problemi (prevertex'ler) çok köşede "crowding" ile kötü koşullu; sayısal konformal harita için modern alternatif AAA/"lightning" rasyonel temsil (Gopal & Trefethen 2019). Yıldız-şekillerde Theodorsen iterasyonu $|h_\theta/(1+h)|<1$ koşulu ister; üreticinin smooth şekilleri bunu ihlal ediyor ($\max|h_\theta|$ medyan 2.37). Bu yüzden konformal yol **uzun vadeli** bir araştırma seçeneğidir. Kısa vadede $r_c$'yi (tek harmonik çözüm; E1'de mesh üzerinde $G=-\log|z-z_0|$ sınır verisiyle hesaplandı) bir **skaler özellik** olarak eklemek ucuzdur (λ₁ tek başına %2.8).

---

## 8. Uygulama Taslağı (dosya bazında)

1. `src/data/dataset_converter.py`
   - Her geometri için P2 (veya P1) torsiyon çözümü $w$: `solve(*condense(K, b, D=D))`, $K$ zaten özdeğer çözümündekiyle aynı. Normalize koordinatta $w_{norm}=w/s^2$.
   - Skaler özellikler: $\log(A/M)$, $\log(T/(AM))$, $\log(L^2/A)$, $\log(r_c^2/M)$, $e_{tor}$ (ve spin-2 vektörü $(J_{11}-J_{22},2J_{12})/\mathrm{tr}J$), Weyl köşe sabiti $c_0$, max iç açı.
   - Düğüm özelliği: $w/M$ (sınırda 0, merkezde 1; mevcut süreksiz `dist_*`/`curvature` özelliklerinin pürüzsüz alternatifi).
   - Hedefler: $\log(\lambda_1M)-\log(j_{01}^2/4)$; çift için $G,S$ (veya $\bar\lambda$, $s$).
2. `src/data/dataset.py`: yeni skaler/hedef alanlarını batch'e ekle; augmentasyonda spin-2 vektörünü $2\alpha$ ile döndür (dönme augmentasyonu varsa).
3. `src/models/gnot.py`, `src/models/spectral_no.py`: özdeğer başlığı
   $\hat\lambda_1=\frac{j_{01}^2}{4M}e^{r_1}$, $\hat\lambda_2=\hat\lambda_1\big(1+(\frac{j_{11}^2}{j_{01}^2}-1)\sigma(z_2)\big)$, $\hat\lambda_3=\hat\lambda_2+\mathrm{softplus}(z_3)\hat\lambda_1$; veya çift için $(\hat{\bar\lambda},\hat s)$ → $\hat\lambda_{2,3}=\hat{\bar\lambda}\mp|\hat s|/2$ (AB kısıtıyla birlikte: $\hat\lambda_2$'yi sigmoid ile sınırlayıp $\hat\lambda_3=\hat\lambda_2+|\hat s|$).
4. `src/training/lightning_module.py`: $r_1$ üzerinde MSE; çift kaybı (§5.3); `detect_clusters` yalnız raporlama.
5. `src/data_gen/dataset_generator.py`: `--n_eigen_modes` varsayılanı ≥6; köşe listesi ve smooth katsayıları `geom_params`'a; `--mode symmetric` test seti ($r=R(1+\varepsilon\cos n\theta)$, $n=2..7$; düzgün $N$-gonlar).
6. `validate_data.py`: simetrik sette dipol dejenereliği ($|\lambda_3-\lambda_2|/\lambda_2<10^{-4}$), AB ve Faber–Krahn ihlal sayacı, $\lambda_1M\in[1,1.6]$ akıl sağlığı kontrolü.

---

## 9. Deneyler (Sonuçlar)

Scriptler `scratchpad/math_geom/` altında (repo'ya dahil değil). FEM: scikit-fem P2, gmsh, `eigsh` shift-invert σ=0.

| ID | Script | İçerik | Ana sonuç |
|---|---|---|---|
| E1 | `gen.py`, `analyze1.py` | Repo üreticisiyle 200 geometri (seed 11, 10 mod) + torsiyon, iç yarıçap, maksimal konformal yarıçap ($G$ harmonik, $G=-\log|z-z_0|$ sınırda, $r_c=e^{-G(z_0)}$, 60 aday merkez), atalet tensörleri, dipol momentleri | §2.2 tablosu. $\lambda_1M$ CV %0.94; eğim −1.007. |
| E1-W | `weyl.py` | 1/2/3-terimli Weyl | §4 tablosu. |
| E2 | `pt.py`, `e2_pert.py` | Disk + tek harmonik / triad / harmonik çiftleri, 1600-gon, P2 (disk tabanı 2.6e-6, bölünerek düzeltildi) | PT2 hatası $O(\varepsilon^4)$ tek harmonikte, triad'da $-65\varepsilon^3$ tek kısım; seçim kuralları doğrulandı (§1.3). |
| E3b | `e3_dataset.py` | Veri setinde ayrışma ~ $2\varepsilon_2$, PT2, atalet anizotropisi | $2\varepsilon_2$: kor. 0.75; PT2: 0.88; $e_{tor}$: 0.967 (sharp). |
| E4 | `e4_cone.py` | $C_3$ tepe etrafında konik kesişme, monodromi, $G,S$ pürüzsüzlüğü | Berry fazı π; $S_{11}-S_{22}$ lineer. |
| E5 | `e5_corners.py` | Düzgün N-gon serisi; köşe üssü | §6. |
| E6 | `e6_conformal.py` | Diskte ağırlıklı problem vs görüntü FEM; nakil sınırı | 2e-6 uyum; sınır/λ₁ 1.0005–1.041. |
| E7 | `e7_priors.py` | 10 ucuz invaryant ile ridge (5-kat CV) | log λ₁ rezidüel 0.0037; λ₂ 0.021; λ₃ 0.040. |

Ek istatistikler: $(\lambda_4-\lambda_3)/\lambda_3$ medyan 0.37, <%5: %1.5, <%10: %3.0 (docs/14: %1.3 / %5.0).

---

## 10. Referanslar

*Bağlantılar web araştırmasında bulunan sayfalardır.*

**Şekil pertürbasyonu**
- Hadamard, J. (1908). *Mémoire sur le problème d'analyse relatif à l'équilibre des plaques élastiques encastrées.* Mém. Sav. Étrang. 33.
- Rayleigh, J. W. S. (1877/1894). *The Theory of Sound*, Cilt 1. — https://books.google.com/books/about/The_Theory_of_Sound.html?id=zTYIAAAAIAAJ
- Joseph, D. D. (1967). *Parameter and domain dependence of eigenvalues of elliptic partial differential equations.* Arch. Rational Mech. Anal. 24, 325–351. — (Joseph 1967'ye atıf yapan domain-bağımlılığı çalışması) https://arxiv.org/pdf/1203.2093
- Henrot, A., Pierre, M. (2018). *Shape Variation and Optimization: A Geometrical Analysis.* EMS Tracts in Mathematics 28. — https://www.abebooks.com/9783037191781/Shape-Variation-Optimization-Geometrical-Analysis-3037191783/plp
- Grinfeld, P. (2010). *Hadamard's formula inside and out.* J. Optim. Theory Appl. 146(3), 654–690. — https://link.springer.com/article/10.1007/s10957-010-9681-6
- Grinfeld, P. (2013). *Introduction to Tensor Analysis and the Calculus of Moving Surfaces.* Springer. — https://link.springer.com/book/10.1007/978-1-4614-7867-6
- Rellich, F. (1937). *Störungstheorie der Spektralzerlegung. I.* Math. Ann. 113. — https://link.springer.com/article/10.1007/BF01571652 ; (seri I–V, 1937–1942) https://eudml.org/doc/159886
- Kato, T. (1966/1995). *Perturbation Theory for Linear Operators.* Springer.
- Lamberti, P. D., Lanza de Cristoforis, M. (2004). *A real analyticity result for symmetric functions of the eigenvalues of a domain dependent Dirichlet problem for the Laplace operator.* J. Nonlinear Convex Anal. 5, 19–42. — (ilgili Neumann versiyonu) https://link.springer.com/article/10.1007/s00009-007-0128-8

**İzoperimetrik eşitsizlikler ve invaryantlar**
- Payne, L. E., Pólya, G., Weinberger, H. F. (1956). *On the ratio of consecutive eigenvalues.* J. Math. Phys. 35, 289–298. — (tartışma) https://link.springer.com/article/10.1007/BF02829638
- Ashbaugh, M. S., Benguria, R. D. (1992). *A sharp bound for the ratio of the first two eigenvalues of Dirichlet Laplacians and extensions.* Ann. Math. 135(3), 601–628. — (ikinci ispat) https://link.springer.com/article/10.1007/BF02099533
- Hile, G. N., Protter, M. H. (1980). *Inequalities for eigenvalues of the Laplacian.* Indiana Univ. Math. J. 29(4), 523–538. — http://www.iumj.indiana.edu/docs/29040/29040.asp
- Yang, H. C. (1991). *An estimate of the difference between consecutive eigenvalues.* ICTP preprint IC/91/60. — (özet) https://link.springer.com/article/10.1007/BF02829638
- Levitin, M., Yagudin, R. (2003). *Range of the first three eigenvalues of the planar Dirichlet Laplacian.* LMS J. Comput. Math. 6, 1–17. — https://arxiv.org/pdf/math/0203231
- He, Y., Tang, Q., Zhang, H. (2026). *Counterexamples to a higher-index Dirichlet eigenvalue-ratio conjecture.* arXiv:2609.26198. — https://arxiv.org/html/2609.26198
- Brasco, L., De Philippis, G., Velichkov, B. (2015). *Faber–Krahn inequalities in sharp quantitative form.* Duke Math. J. 164(9), 1777–1831. — https://arxiv.org/abs/1306.0392
- Pólya, G., Szegő, G. (1951). *Isoperimetric Inequalities in Mathematical Physics.* Princeton.
- Hersch, J. (1960). *Sur la fréquence fondamentale d'une membrane vibrante: évaluations par défaut et principe de maximum.* ZAMP 11, 387–413. — https://link.springer.com/article/10.1007/BF01604498
- Makai, E. (1965). *A lower estimation of the principal frequencies of simply connected membranes.* Acta Math. Hungar. — https://link.springer.com/article/10.1007/BF01904840
- Bañuelos, R., Carroll, T. (1994). *Brownian motion and the fundamental frequency of a drum.* Duke Math. J. 75(3), 575–602. — https://projecteuclid.org/euclid.dmj/1077287810
- van den Berg, M., Ferone, V., Nitsch, C., Trombetti, C. (2016). *On Pólya's inequality for torsional rigidity and first Dirichlet eigenvalue.* Integral Equations Operator Theory 86, 579–600. — https://arxiv.org/abs/1602.04618
- van den Berg, M., Carroll, T. (2009). *Hardy inequality and $L^p$ estimates for the torsion function.* Bull. LMS 41, 980–986.
- van den Berg, M. (2012). *Estimates for the torsion function and Sobolev constants.* Potential Anal. 36, 607–616. — https://link.springer.com/article/10.1007/s11118-011-9246-9
- van den Berg, M. (2017). *Spectral bounds for the torsion function.* Integral Equations Operator Theory 88, 387–400. — https://arxiv.org/abs/1701.02172
- Vogt, H. (2019). *$L_\infty$-estimates for the torsion function and $L_\infty$-growth of semigroups satisfying Gaussian bounds.* Potential Anal. 51, 37–47. — https://www.researchgate.net/publication/310122525
- Henrot, A., Lucardesi, I., Philippin, G. (2018). *On two functionals involving the maximum of the torsion function.* ESAIM: COCV 24(4), 1585–1604. — https://www.esaim-cocv.org/articles/cocv/abs/2018/04/cocv170100/cocv170100.html
- Kohler-Jobin, M.-T. (1978). *Une méthode de comparaison isopérimétrique de fonctionnelles de domaines de la physique mathématique.* ZAMP 29. — (genelleme ve tarih) https://arxiv.org/pdf/2105.09353
- Filoche, M., Mayboroda, S. (2012). *Universal mechanism for Anderson and weak localization.* PNAS 109(37), 14761–14766. — https://www.pnas.org/doi/abs/10.1073/pnas.1120432109
- Arnold, D. N., David, G., Filoche, M., Jerison, D., Mayboroda, S. (2019). *Computing spectra without solving eigenvalue problems.* SIAM J. Sci. Comput. 41(1), B69–B92. — https://doi.org/10.1137/17m1156721
- Filonov, N., Levitin, M., Polterovich, I., Sher, D. A. (2023). *Pólya's conjecture for Euclidean balls.* Invent. Math. 234, 129–169. — https://arxiv.org/abs/2203.07696

**Weyl asimptotiği**
- Kac, M. (1966). *Can one hear the shape of a drum?* Amer. Math. Monthly 73(4).
- McKean, H. P., Singer, I. M. (1967). *Curvature and the eigenvalues of the Laplacian.* J. Diff. Geom. 1.
- van den Berg, M., Srisatkunarajah, S. (1988/1990). *Heat equation for a region in $\mathbb R^2$ with a polygonal boundary.* J. London Math. Soc. 37; *Heat flow and Brownian motion for a region in $\mathbb R^2$ with a polygonal boundary.* PTRF 86, 41–52. — https://people.maths.bris.ac.uk/~mamvdb/pdfs/Publication23.pdf
- Ivrii, V. (1980). *Second term of the spectral asymptotic expansion of the Laplace–Beltrami operator on manifolds with boundary.* Funct. Anal. Appl. 14. — ve Ivrii, V. (2016). *100 years of Weyl's law.* Bull. Math. Sci. 6 — https://link.springer.com/article/10.1007/s13373-016-0089-y
- Nursultanov, M., Rowlett, J., Sher, D. A. (2019/2020). *How to hear the corners of a drum.* — https://arxiv.org/abs/2012.03366

**Simetri ve dejenerelik**
- von Neumann, J., Wigner, E. (1929). *Über das Verhalten von Eigenwerten bei adiabatischen Prozessen.* Phys. Z. 30.
- Albert, J. H. (1975). *Genericity of simple eigenvalues for elliptic PDE's.* Proc. AMS 48, 413–418.
- Uhlenbeck, K. (1976). *Generic properties of eigenfunctions.* Amer. J. Math. 98, 1059–1078. — https://www.semanticscholar.org/paper/Generic-Properties-of-Eigenfunctions-Uhlenbeck/4798c75a04c5b45d119b4f73c92b0e6b81619034
- Teytel, M. (1999). *How rare are multiple eigenvalues?* Comm. Pure Appl. Math. 52, 917–934. — https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-0312(199908)52:8%3C917::AID-CPA1%3E3.0.CO;2-S
- Hillairet, L., Judge, C. (2009). *Generic spectral simplicity of polygons.* Proc. AMS 137(6), 2139–2145. — https://arxiv.org/pdf/math/0703616
- Berry, M. V., Wilkinson, M. (1984). *Diabolical points in the spectra of triangles.* Proc. R. Soc. Lond. A 392, 15–43. — https://royalsocietypublishing.org/doi/10.1098/rspa.1984.0022
- Cureton, L. M., Kuttler, J. R. (1999). *Eigenvalues of the Laplacian on regular polygons and polygons resulting from their dissection.* J. Sound Vib. 220, 83–98. — https://ui.adsabs.harvard.edu/abs/1999JSV...220...83C/abstract

**Köşeler ve poligonlar**
- Kondrat'ev, V. A. (1967). *Boundary value problems for elliptic equations in domains with conical or angular points.* Trudy Moskov. Mat. Obšč. 16, 209–292 (Trans. Moscow Math. Soc. 16, 227–313). — https://m.mathnet.ru/eng/mmo186
- Grisvard, P. (1985/2011). *Elliptic Problems in Nonsmooth Domains.* Pitman; SIAM Classics 69. — https://books.google.com/books/about/Elliptic_Problems_in_Nonsmooth_Domains.html?id=LdMk297ExhkC
- Grinfeld, P., Strang, G. (2012). *Laplace eigenvalues on regular polygons: a series in 1/N.* J. Math. Anal. Appl. — https://www.sciencedirect.com/science/article/pii/S0022247X1100583X ; (2004) *The Laplacian eigenvalues of a polygon.* Comput. Math. Appl. — https://www.sciencedirect.com/science/article/pii/S0898122104003505
- Jones, R. S. (2017). *The fundamental Laplacian eigenvalue of the regular polygon with Dirichlet boundary conditions.* arXiv:1712.06082 — https://arxiv.org/abs/1712.06082
- Berghaus, D., Georgiev, B., Monien, H., Radchenko, D. (2024). *On Dirichlet eigenvalues of regular polygons.* J. Math. Anal. Appl. — https://arxiv.org/pdf/2103.01057
- Jones, R. S. (2017). *Computing ultra-precise eigenvalues of the Laplacian within polygons.* Adv. Comput. Math. — https://link.springer.com/article/10.1007/s10444-017-9527-y

**Konformal yöntemler**
- Driscoll, T. A., Trefethen, L. N. (2002). *Schwarz–Christoffel Mapping.* Cambridge Monographs on Applied and Computational Mathematics 8. — https://www.cambridge.org/core_title/gb/202581
- Gopal, A., Trefethen, L. N. (2019). *Representation of conformal maps by rational functions.* Numer. Math. 142, 359–382. — https://link.springer.com/article/10.1007/s00211-019-01023-z
- (Pólya–Szegő konformal yarıçap tamamlayıcıları) *On symmetric membranes and conformal radius: some complements to Pólya's and Szegö's inequalities.* Arch. Rational Mech. Anal. — https://link.springer.com/article/10.1007/BF00282359

---

## 🔗 Bağlantılar

- [[14_MATHEMATICAL_IMPROVEMENTS]] · [[13_RAYLEIGH_ANALYSIS]] · [[11_ORTHOGONALITY_ANALYSIS]] · [[09_PHYSICS_BACKGROUND]] · [[01_DATA_GENERATION]]

#matematik #spektral-geometri #hadamard #rellich #faber-krahn #ashbaugh-benguria #torsiyon #landscape #weyl #konik-kesişme #köşe-tekilliği #konformal
