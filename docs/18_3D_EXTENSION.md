# 18 — 3B'ye Genişletme: Fizik, Veri Üretimi, Model Tasarımı ve Yol Haritası

> **Kapsam:** Araştırma ve fizibilite notu. Model ve eğitim kodunda değişiklik yok, eğitim koşusu yok. Yalnız küçük CPU deneyleri yapıldı (§2.4, §3.4–3.6).
> **Soru:** Bugünkü skaler 2D TM problemi ($-\Delta E_z=k^2E_z$, $E_z|_{\partial\Omega}=0$, $f=ck/2\pi$) için yazılmış EigenspaceOperator hattı, gerçek 3D RF kavitelerine ($\nabla\times\nabla\times\mathbf E=k^2\mathbf E$, $\mathbf n\times\mathbf E=0$) nasıl taşınır? Hangi ara adımlar (2.5D eksenel simetri, 3D skaler) değerlidir?
> **Kanıt:** scikit-fem 12.0.2 (`ElementTetN0`, `ElementTriN1`, `ElementTriP2`), gmsh 4.15.2 (OCC), SciPy 1.17.1; karşılaştırma için izole bir venv'de NGSolve 6.2.2607 ve pypardiso (MKL PARDISO). 4 çekirdekli CPU. Scriptler: [`scripts/research_3d/`](../scripts/research_3d/) (§6).
> **Referans commit:** `37fda03` (`claude/neo-eigenspace`). İncelenen dosyalar: `src/data_gen/dataset_generator.py`, `src/data/dataset_converter.py`, `src/models/eigenspace_operator.py`, `src/models/spectral_no.py` (`_p1_galerkin`, `_ritz`), `src/training/lightning_module.py` (`eigenspace_grams`, `span_residual`, `ritz_compliance`).

---

## 🧭 Özet

**Ana bulgular:**

1. **Nereden başlanmalı: eksenel simetrik (2.5D) meridyen modeli.** $m=0$ monopol (hızlandırıcı) modları için SUPERFISH tipi skaler $H_\varphi$ formülasyonu, bugünkü kodun neredeyse aynısıdır: üçgen mesh, P1/P2 Lagrange, düğüm ağırlıkları, Ritz. Tek fark, ağırlıklı ($r$, $1/r$) Gram montajı ve Dirichlet kapısının duvardan **eksene** taşınmasıdır. **Sahte (spurious) mod yoktur.** Deneyde TESLA orta hücresi (Aune vd. 2000 geometrisi) için π-modu **1.30098 GHz**, 0-modu **1.27668 GHz**, hücreler arası kuplaj **$k_{cc}=1.886\%$** çıktı (tasarım değeri 1.87 %). 2 mm mesh'te (5.7k P2 DOF) bu çözüm **~0.1 s** sürüyor. Pillbox'ta P2 hatası $h=R/20$ için $\le 6.5\cdot10^{-6}$. Kodun tahminen **~%80'i** aynen taşınır.
2. **Tam 3D Maxwell'in kalbi çekirdek (kernel) problemidir.** $H_0(\mathrm{curl})$'da $\ker(\nabla\times)=\nabla H^1_0$ sonsuz boyutludur. Ayrık Nédélec uzayında bu, **iç düğüm sayısı kadar $\lambda=0$ modu** demektir: $32^3$ kutuda 8 fiziksel moda karşı **13 950** sıfır. Naif shift-invert yalnız sıfırları döndürdü (8/8 sahte). Bunun **Rayleigh–Ritz başlığı için sonucu ağırdır**: span'e karışan her gradyan bileşeni Ritz değerini gerçeğin **altına** çeker. Gradyan kütle payı %8 iken $\lambda_1$ %12.8 düştü; saf bir gradyan kolonu $\theta=0$ verdi ve kendinden denetimli compliance kaybını $10^{14}$'e fırlattı (§3.4). Çare hem ucuz hem tamdır: kütle Gram'ı yerine **Schur tümleyeni** $M_{div}=V^\top MV-B^\top K_p^{-1}B$ ($B=G^\top MV$, $K_p$ = P1 Laplace) kullanılır. Bu, Ritz değerlerini $10^{-14}$ doğrulukla geri getiriyor ve $m=16$ kolon için ~5–100 ms tutuyor.
3. **Sınır koşulu için en iyi yol $\mathbf H$-alanı formülasyonudur, $\mathbf E$ kapısı değil.** PEC'de yalnız teğetsel $\mathbf E$ sıfırdır, normal bileşen serbest (ve genelde maksimum) kalır. Bu yüzden bugünkü skaler torsiyon kapısını vektöre uygulamak yanlıştır. Deneyde (§3.5) kapı + Hodge projeksiyonu TM010'da $h$'ye bağlı %7–20'lik bir taban bıraktı. Sınır DOF'larını sıfırlamak $1/h$ sınır tabakası enerjisi üretti. $\mathbf H$ formülasyonunda ise PEC koşulları **doğal**dır (hiç esas BC yok): tüm kenarlar kullanılır ve yalnız $\nabla P1$ projekte edilir. Bu yol **mesh'ten bağımsız** hata verdi: TM010 $+3.3\cdot10^{-3}$, $h=0.18/0.13/0.10$'da aynı. Bu, 2D TM'nin (Dirichlet $E_z$) TE (Neumann $H_z$) ikizinin 3D karşılığıdır. SUPERFISH'in $rH_\varphi$ seçimiyle de aynı mantığı taşır.
4. **Veri üretimi 3D'de mümkün ama 10–100× pahalıdır.** Lowest-order Nédélec (N0) ile pillbox (R=L=1) sonuçları: $h=0.1$'de 15k DOF, TM010 hatası $-1.8\cdot10^{-3}$, ilk 10 modda en fazla $3.8\cdot10^{-3}$, toplam ~1–8 s; $h=0.05$'te 125k DOF, $-3.5\cdot10^{-4}$, ~80 s, 2.4 GB. **İkinci/üçüncü derece eğri Nédélec** (NGSolve) aynı doğruluğu 5–30× daha az DOF ile veriyor: $p=3$, 3.4k serbest DOF'ta TM010 $-9\cdot10^{-6}$ ve 10 modda en fazla $8\cdot10^{-4}$; 12k DOF'ta 10 modda en fazla $9.5\cdot10^{-5}$. Örnek başına depolama ~1–3 MB (2D'de ~50 KB). 5k örneklik bir set ~5–15 GB tutar ve 4 çekirdekte ~7–20 saatte üretilir.
5. **Literatürde doğrudan rakip yok.** Özuzay/Ritz tabanlı, geometri → 3D Maxwell özmodları haritalayan bir nöral operatör bulamadık. En yakınları şunlar: NEO (yüzeylerde Laplace–Beltrami özuzayı), FieldTNN (tek geometride 3D Maxwell özdeğeri, div-cezası), SRF ters tasarımı (Yaker vd. 2026: skaler gözlemlenenlere DNN, COMSOL verisi, ~2 dk/örnek, ~%5 doğruluk) ve Whitney-form/FEEC tabanlı yapı-koruyan öğrenme (Trask grubu). Açık alan tam da buradaki kombinasyon: **geometri-koşullu, Whitney-uyumlu, Hodge-projekteli Ritz özuzayı**.

### Öncelik sıralı öneri

| Sıra | Faz | Neden | Efor | Başarı ölçütü |
|---|---|---|---|---|
| **1** | **Faz 0a — eksenel simetrik $m=0$ ($H_\varphi$, P1/P2)** | En yüksek değer/maliyet. Gerçek SRF hücreleri (TESLA, eliptik) eksenel simetrik; etiketler ~0.1 s; kod ~%80 aynı; sahte mod yok | 1–2 hafta | Eliptik hücre ailesinde π-modu frekansı < %0.1, $k_{cc}$ hatası < 0.05 puan, $E_{pk}/E_{acc}$, $B_{pk}/E_{acc}$, $R/Q$ < %2 |
| **2** | **Faz 0b — eksenel simetrik $m\ge1$ (N1×P1, 2D'de H(curl))** | Dipol/HOM bandı (wakefield, HOM sönümleme). Nédélec + çekirdek projeksiyonu makinesini **ucuz 2D'de** test etme imkânı | 2–3 hafta | Pillbox'ta $m=1$ ilk 6 mod < %0.5; eliptik hücrede ilk dipol bandı < %0.5; çekirdek sızıntısı (Ritz < λ_h) = 0 |
| 3 | Faz 1 — 3D skaler Dirichlet (P1/P2 tet) | 3D altyapısı: tet mesh, $N=10^4$–$10^5$ nokta ölçeklemesi, 3D özellikler, bellek. Fizik değeri düşük (akustik/Dirichlet) | 2–3 hafta | 3D yıldız-şekillerde $\lambda_{1..6}$ < %1 (Ritz, P1), eğitim $N\approx2$k → çıkarım $N\approx50$k sıfır-atış |
| 4 | **Faz 2 — tam 3D Maxwell, $\mathbf H$ formülasyonu, Whitney N0 kenar DOF'ları + Schur/Hodge projeksiyonu** | Portlu/kuplörlü, simetrisi bozuk kaviteler; 3D'ye özgü hibrit modlar | 6–10 hafta | Pillbox/kutu/eliptik hücre 3D'de ilk 10 mod < %1 (Ritz), dejenere çiftlerde alt uzay hatası < %2, Ritz ≥ λ_h her örnekte |
| 5 | Faz 3 (kapsam dışı) — açık portlar (dalga kılavuzu BC, kompleks özdeğer, $Q_{ext}$), kayıplı duvarlar ($Q_0$), çok hücreli yapılar | Gerçek tasarım döngüsü | aylar | — |

**Tek cümlelik ana mesaj:** Önce meridyen düzleminde $m=0$ ile başlayın. Bu, bugünkü kodun bir hafta uzağında olan, gerçek SRF kaviteleri için anlamlı ve etiketleri SUPERFISH kalitesinde ucuz üretilebilen tek yoldur. Sonra $m\ge1$ ile Nédélec/çekirdek makinesini 2D'de oturtun. Tam 3D'de ise $\mathbf E$ yerine **$\mathbf H$ alanını, kenar DOF'ları ve Schur-projekteli Gram'larla** öğrenin.

---

## 1. Fizik: skalerden vektöre

### 1.1 Bugünkü problem ve 3D'de değişenler

Bugün çözülen, $z$ yönünde düzgün sonsuz bir kavitenin TM$_{mn0}$ modlarıdır:
$-\Delta E_z=k^2E_z$, $E_z|_{\partial\Omega}=0$. Doğru Hilbert uzayı $H^1_0(\Omega)$'dır. Lagrange elemanlar uyumludur ve Ritz değerleri gerçek üst sınırdır (Courant–Fischer). EigenspaceOperator'ın bütün tasarımı bu üç özelliğe dayanır: torsiyon kapısı ($\psi=w\cdot N$, $\psi|_{\partial\Omega}=0$), P1 Gram'ları ve Ritz'in üst-sınır garantisi.

Sonlu uzunluklu, PEC duvarlı bir 3D kavitede zaman-harmonik alanlar
$$\nabla\times\nabla\times\mathbf E=k^2\mathbf E\ \ (\Omega),\qquad \mathbf n\times\mathbf E=0\ \ (\partial\Omega),\qquad \nabla\cdot\mathbf E=0$$
problemini sağlar ($k=\omega/c$, $\mathbf H=\nabla\times\mathbf E/(-i\omega\mu_0)$). Değişenler şunlar:

| | 2D TM (bugün) | 3D Maxwell |
|---|---|---|
| Bilinmeyen | skaler $E_z$ | vektör $\mathbf E$ (3 bileşen) |
| Uzay | $H^1_0$ | $H_0(\mathrm{curl})=\{\mathbf u\in L^2:\nabla\times\mathbf u\in L^2,\ \mathbf n\times\mathbf u=0\}$ |
| Operatör | pozitif tanımlı | pozitif **yarı**-tanımlı; $\ker=\nabla H^1_0\oplus\mathcal H$ **sonsuz boyutlu** |
| Sınır koşulu | $E_z=0$ (tüm alan) | yalnız **teğetsel** $\mathbf E$ sıfır; $E_n$ serbest (yüzey yükü) |
| Ek kısıt | yok | $\nabla\cdot\mathbf E=0$ (zayıf: $(\mathbf E,\nabla q)=0$, $\forall q\in H^1_0$) |
| Doğru FE | Lagrange P1/P2 | Nédélec (Whitney 1-form) kenar elemanları |
| Ritz üst sınırı | var | yalnız ayrık diverjanssız alt uzayda; ayrık özdeğerler gerçeğin **altında** da olabilir (§2.4) |
| Weyl sayımı | $N(k)\approx\frac{|\Omega|}{4\pi}k^2$ | $N(k)\approx\frac{|\Omega|}{3\pi^2}k^3$ (skaler 3D'nin 2 katı) |

**Zayıf form:** $\mathbf E\in H_0(\mathrm{curl})$ bulunur, öyle ki her $\mathbf v\in H_0(\mathrm{curl})$ için
$$(\nabla\times\mathbf E,\nabla\times\mathbf v)=k^2(\mathbf E,\mathbf v).$$
Her $\mathbf E=\nabla\phi$ ($\phi\in H^1_0$) bu denklemi $k=0$ ile sağlar. Bu **gradyan çekirdeği**, $k^2>0$ olan her fiziksel modun $M$-ortogonal tümleyenidir. Fiziksel modlar bu yüzden kendiliğinden diverjanssızdır: $(\mathbf E,\nabla q)=k^{-2}(\nabla\times\mathbf E,\nabla\times\nabla q)=0$. Ek olarak $\mathcal H$ harmonik alanları vardır: $\nabla\times=0$, $\nabla\cdot=0$, $\mathbf n\times=0$. Boyutları (sınır bileşeni sayısı − 1)'dir, yani yalnız iç iletken duvarlarla temas etmeden "yüzüyorsa" ortaya çıkarlar.

**Ayrık de Rham dizisi:** $P1\xrightarrow{\ \nabla\ }N0\xrightarrow{\ \nabla\times\ }RT0\xrightarrow{\ \nabla\cdot\ }P0$. Bu dizi **tamdır**: $\nabla P1\subset N0$ tam olarak içerilir ve $\ker(\nabla\times|_{N0})=\nabla P1$ (basit bağlantılı $\Omega$). Ayrık gradyan matrisi $G$ (kenar × düğüm, girdileri $\pm1$) topolojiktir ve kesindir. Kütle matrisiyle $G^\top M_{N0}G=K_{P1}$ özdeşliği geçerlidir (P1 Laplace rijitliği). Bu iki gerçek, §3.4'teki Schur-projeksiyonunun temelidir.

### 1.2 Neden nodal/Lagrange elemanlar yanlış, neden Nédélec/Whitney?

- **Çekirdek yakalanamaz.** $[P1]^3$ nodal uzayında $\nabla\times$'in çekirdeği, ayrık gradyanlar uzayıyla örtüşmez ($\nabla P2$ süreksizdir, $[P1]^3$'te yoktur). Gradyan-benzeri alanlar ayrık operatörde küçük ama **sıfır olmayan** özdeğerler alır ve fiziksel spektrumun içine yayılır. Deneyde (E8, kutu $1\times0.8\times0.6$) saf curl–curl nodal P1 ile her mesh'te $\lambda_8$'in altında **15–16 sahte değer** çıktı. $n=12$'de en düşük sıfır-olmayan $k^2=1.1$ iken gerçek $\lambda_1=25.3$'tür. Sahte değerler mesh inceldikçe kaybolmaz, yeniden dizilir.
- **Düzenlileştirilmiş nodal** ($(\nabla\times,\nabla\times)+s(\nabla\cdot,\nabla\cdot)$) konveks domainde çalışır (E8'de $s=1$ ile değerler doğru yakınsıyor). Ama iki sorunu vardır: (i) $s\,\mu_j$'de gradyan modları üretir ($\mu_j$: skaler Dirichlet özdeğerleri; kutuda $s\mu_1=52.7$, TE/TM$_{111}$ ile çakışıp sahte bir üçlü oluşturdu). (ii) Re-entrant kenar/köşeli domainlerde **yanlış limite yakınsar**, çünkü $H_0(\mathrm{curl})\cap H(\mathrm{div})$ içinde $H^1$ alanlar yoğun değildir (Costabel & Dauge 2002). İris ve port birleşimleri tam da böyle kenarlardır.
- **Nédélec (Whitney) kenar elemanları** doğal serbestlik dereceleriyle çalışır: $u_e=\int_e\mathbf u\cdot\mathbf t\,ds$. Teğetsel süreklilik, dolayısıyla $H(\mathrm{curl})$-uyumluluk yapısaldır; normal bileşen elemanlar arasında sıçrayabilir (yüzey yükü). $\mathbf n\times\mathbf E=0$ yalnızca **sınır kenarlarının DOF'larını sıfırlamak** demektir, normal bileşene dokunmaz. Tam dizi sayesinde ayrık çekirdek tam olarak $\nabla P1$'dir. Özdeğerler ya tam 0'dır ya da doğru yakınsar (discrete compactness; Kikuchi, Boffi). Yakınsama lowest-order'da $O(h^2)$ (E1: $4\times$ incelmede $\sim11\times$), derece $p$'de $O(h^{2p})$'dir.

### 1.3 Mod aileleri ve dejenereler

- **Pillbox** (yarıçap $R$, boy $L$): TM$_{mnp}$: $k^2=(j_{mn}/R)^2+(p\pi/L)^2$, $p\ge0$. TE$_{mnp}$: $k^2=(j'_{mn}/R)^2+(p\pi/L)^2$, $p\ge1$. $m\ge1$ her mod $\cos/\sin$ ile **iki kat** dejeneredir. Hızlandırıcı modu TM$_{010}$ ($k=2.405/R$) yalnız $L<2.03R$ ise en düşüktür; aksi hâlde TE$_{111}$ ondan aşağı iner. $j'_{01}=j_{11}$ olduğu için TE$_{01p}$ ile TM$_{11p}$ **kazara dejeneredir** (R=L=1'de $k=4.955$'te üçlü: TE011 + 2×TM111). 2D'deki dipol çiftinin karşılığı 3D'de $\pm m$ çiftleridir. Ama artık farklı ailelerin (TE/TM) kesişmeleri de olağandır.
- **Dikdörtgen kutu:** $k^2=(m\pi/a)^2+(n\pi/b)^2+(p\pi/d)^2$. Üç indeks de $\ge1$ ise TE ve TM aynı frekansta (iki kat), tam bir indeks 0 ise tek mod, iki indeks 0 ise mod yok. Küpte ilk değer **üç kat** (TE/TM$_{110}$ permütasyonları), ikinci **iki kat**, sonra yine **üç kat** dejeneredir.
- **Hibrit modlar:** Simetri bozulunca (portlar, kuplörler, hücre eksen kaçıklığı) TE/TM ayrımı kaybolur ve modlar hibritleşir (HEM). Eksenel simetri korunursa her azimutal indeks $m$ ayrık bir problemdir. $m=0$'da TM$_0$ ($E_r,E_z,H_\varphi$) ile TE$_0$ ($E_\varphi,H_r,H_z$) tamamen ayrışır. $m\ge1$'de iki aile tek bir vektör probleminde birleşir.
- **Öğrenme için sonuç:** 3D'de dejenere ve yakın-dejenere kümeler 2D'den **daha sık ve daha büyüktür** (küp: 3'lü; pillbox: 2'li ve kazara 3'lü). Bugünkü alt uzay (span) kaybı ve Ritz başlığı bu yüzden daha da gereklidir. Tekil özvektör regresyonu 3D'de tamamen umutsuzdur.

### 1.4 $\mathbf E$ mi, $\mathbf H$ mi? (tasarımın kilit gözlemi)

$\mathbf H$ alanı aynı spektrumu verir: $\nabla\times\nabla\times\mathbf H=k^2\mathbf H$. PEC'de ise $\mathbf n\cdot\mathbf H=0$ ve $\mathbf n\times(\nabla\times\mathbf H)=0$ sağlanır; bu **iki koşul da doğaldır**. Zayıf form $\mathbf H\in H(\mathrm{curl})$ ile **esas sınır koşulu olmadan** yazılır: $(\nabla\times\mathbf H,\nabla\times\mathbf v)=k^2(\mathbf H,\mathbf v)$, her $\mathbf v\in H(\mathrm{curl})$ için. $\mathbf n\cdot\mathbf H=0$, tüm $q\in H^1$'lerin gradyanlarına $M$-ortogonallikten kendiliğinden çıkar. Çekirdek $\nabla H^1$'dir (sabitler hariç). Sahte harmonik alanlar yalnız domainde "kulp" varsa ortaya çıkar (birinci Betti sayısı; örn. iç iletkeni iki uç plakaya bağlı koaksiyel/QWR kavite).

Bu, 2D'deki TM (Dirichlet $E_z$) / TE (Neumann $H_z$) ikiliğinin 3D karşılığıdır. SUPERFISH de aynı nedenle $rH_\varphi$ için çözer: PEC duvarlar Neumann (doğal) koşula döner. Ağ açısından sonucu kritiktir: **$\mathbf H$ formülasyonunda ağın çıktısına hiçbir sınır kapısı gerekmez** (§3.5).

### 1.5 Ara seçenekler: maliyet / değer

| Seçenek | Problem | Ayrıklaştırma | Sahte mod | Örnek başına etiket (bu not) | Fiziksel değer | Mevcut koda uzaklık |
|---|---|---|---|---|---|---|
| **(a) 2.5D, $m=0$** | TM$_0$ monopol, $H_\varphi(r,z)$ skaler | P1/P2 Lagrange, meridyen üçgen mesh | **yok** | 0.05–0.25 s (P2, 2k–7k DOF) | **yüksek**: hızlandırıcı modu, $R/Q$, $E_{pk}$, $B_{pk}$, $k_{cc}$; eliptik SRF hücreleri | çok küçük (~%80 aynen) |
| (a') 2.5D, $m\ge1$ | dipol/HOM, $(E_r,E_z)\in N1$, $rE_\varphi\in P1$ | karma N1×P1 | var (2D, küçük; $\nabla P1$ projeksiyonu) | 0.08–0.3 s (2k–7k DOF) | yüksek: HOM, wakefield, BBU | orta: çift-alan başlık + projeksiyon |
| (b) 3D skaler | $-\Delta u=\lambda u$ tet üzerinde (akustik / Dirichlet) | P1/P2 tet | yok | 0.2 s (P1, 1.7k) … 0.5–7 s (P2, 7.5k–17k) | düşük (EM değil) | küçük (yalnız tet + 3D özellikler) |
| **(c) 3D Maxwell** | $\nabla\times\nabla\times$, PEC | Nédélec N0 (veya $p=2,3$) | **evet** (#iç düğüm kadar) | 1–8 s (15k), ~15 s (45k), ~80 s (125k DOF); $p=3$: 0.6 s (3.4k) | en yüksek: simetrisiz geometri, portlar, kuplörler | büyük: vektör başlık, kenar Gram'ları, projeksiyon |

---

## 2. Veri üretimi

### 2.1 Geometri aileleri

| Aile | Parametrizasyon | Mesh | Analitik referans | Not |
|---|---|---|---|---|
| Pillbox ($R$, $L$) | 2 parametre (+ köşe yuvarlatma $r_c$) | 2.5D: dikdörtgen meridyen; 3D: `occ.addCylinder` | TM/TE$_{mnp}$ (§1.3) | Kalibrasyon; $L\gtrless2.03R$ mod sırası değişir |
| Kutu ($a,b,d$) | 3 parametre | `MeshTet.init_tensor` veya OCC `addBox` | $k^2=\sum(n_i\pi/a_i)^2$ | 3'lü dejenereler (küp) |
| **Eliptik hücre** (TESLA tipi) | yarım hücre: $R_{iris},R_{eq},L,A,B,a,b$ (iki elips + ortak teğet; Aune vd. 2000) | 2.5D: `addSpline`+çizgi; 3D: meridyen profilin `occ.revolve` ile döndürülmesi | yok (SUPERFISH/SLANS düzeyinde P2 referansı) | Uç hücre + beam pipe, 1–3 hücre; spline pertürbasyonu (Yaker vd. 2026 da spline öneriyor) |
| Rastgele yıldız-cisimler | $r(\theta,\varphi)=r_0+\sum_{\ell,m}a_{\ell m}Y_\ell^m$ ($\ell\le4$–6) | OCC BSpline yüzey veya nokta bulutundan STL → gmsh | yok | 3D skaler faz 1 ve 3D Maxwell çeşitliliği için |
| Delikler / portlar | beam pipe (silindir birleşimi), kuplör portu (yan silindir, OCC `fuse`), iç iletken (QWR: `cut`) | OCC boolean | yok | Kapalı port = PEC kapak veya magnetik duvar (açık port Faz 3'te) |

Tüm bunlar gmsh OCC ile mümkündür: `addCylinder`, `addBox`, `addSphere`, `revolve`, `fuse`, `cut`, `addBSplineSurface`. Boyut alanı (Distance/Threshold) bugünkü gibi kullanılır, yalnız `SurfacesList` ile. Bugünkü üreticinin gmsh düğüm-etiketi → satır eşlemesi ve kullanılmayan düğüm atma mantığı 3D'ye aynen taşınır (tip 4 = 4 düğümlü tet; bkz. `scripts/research_3d/n0lib.py::gmsh_mesh`).

### 2.2 Çözücü seçenekleri

| Araç | Eleman | Çekirdek yönetimi | Artı | Eksi |
|---|---|---|---|---|
| **scikit-fem** (kurulu) | `ElementTetN0` (= `ElementTetN1`, en düşük derece, 6 DOF/tet); 2D'de `ElementTriN1/N2/N3` | elle: $G$ matrisi (`mesh.edges`, yön $e_0\to e_1$ = +), SciPy ile projeksiyon | saf Python, mevcut koda en yakın; bu nottaki tüm skfem deneyleri | 3D'de **yalnız lowest-order** Nédélec; eğri eleman yok; SuperLU 3D'de yavaş |
| **NGSolve** (pip wheel var) | HCurl her derece, eğri elemanlar, `CreateGradient()` | $G$ hazır; PINVIT/LOBPCG + gradyan projeksiyonu (resmî tutorial) | $p=2,3$ + eğri geometri ile DOF başına çok daha doğru (§2.4); OCC entegre | `nograds=True` + P1 projeksiyonu kombinasyonu bu notta **yanlış** sonuç verdi (§2.4); tam uzay kullanılmalı |
| **FEniCSx/dolfinx** | `N1curl` her derece | SLEPc shift-invert + filtre veya karma form | PETSc/SLEPc/MUMPS, paralel, eğri eleman | conda kurulumu; forum'da 3D Maxwell özdeğer örnekleri var |
| **MFEM / Palace** (AWS) | Nédélec yüksek derece, eğri, AMR | kendi özçözücüsü (SLEPc/ARPACK + AMS/divergence-free projeksiyon) | endüstriyel, GPU, port/kayıp desteği | C++, girdi dosyası tabanlı |
| **ACE3P / Omega3P** (SLAC) | 6. dereceye kadar hiyerarşik Nédélec, kuadratik tet | paralel özçözücü | hızlandırıcı standardı | kurum lisansı |
| SUPERFISH / SLANS / URMEL / CLANS2 | 2.5D ($m=0$; URMEL/CLANS2 $m\ge1$) | — | endüstri referansı, doğrulama | kapalı/eski; toplu üretim için elverişsiz |

**Çekirdekten kurtulma yöntemleri** (E1'de ölçüldü; kutu $1\times0.8\times0.6$, $n=12$, 5 087 DOF, çekirdek boyutu 594):

| Yöntem | Formül | Sonuç (ilk 8 mod) | Süre | Yorum |
|---|---|---|---|---|
| Naif shift-invert, $\sigma=0.3\lambda_1$ | $(K-\sigma M)^{-1}M$ | **8/8 sıfır** ($6\cdot10^{-13}$) | 7.4 s | sıfırlar σ'ya en yakın olanlar |
| Naif, $\sigma=-1$ | aynı | 8/8 sıfır | 1.6 s | $K-\sigma M$ SPD ama sıfırlar yine en yakın |
| Filtre, $\sigma=0.6\lambda_8$ | shift-invert + $|\lambda|<\epsilon$ atla | 8/8 doğru (2 sıfır atıldı) | 39 s | $\sigma>\lambda_K/2$ bilinmeli (önsel tahmin) |
| Filtre, $\sigma=0.9\lambda_1$ | aynı | **yalnız 3/8** fiziksel | 125 s | $\sigma<\lambda_K/2$ ⇒ başarısız |
| grad–div ceza, $s=0.2$ | $K+s\,(MG)D^{-1}(MG)^\top$ | sahte 10.4, 16.2, 19.3 | 1.1 s | $s\mu_1<\lambda_K$ ⇒ gradyan modları içeride |
| ceza, $s=1$ | aynı | bir sahte (52.07) araya girdi | 0.9 s | $s$ seçimi hassas |
| ceza, $s=5$ | aynı | 8/8 doğru | 0.95 s | ama nnz **13×** (2-halka dolum) |
| **Projeksiyon (önerilen)** | $\sigma<0$, $\mathrm{OP}=P(K-\sigma M)^{-1}$, $P=I-GK_p^{-1}G^\top M$ | **8/8 doğru**, $\|G^\top M u\|/\|Mu\|\sim10^{-15}$ | **0.34 s** | Arbenz & Geus (1999) tipi; $K_p=G^\top MG$ = P1 Laplace (küçük, SPD) |
| Karma form (Kikuchi) | $\begin{bmatrix}K&MG\\G^\top M&0\end{bmatrix}$, $B=\mathrm{diag}(M,0)$ | projeksiyonla matematiksel olarak eşdeğer | — | ölçülmedi; SLEPc'de doğrudan kurulabilir |
| Tree–cotree gauge | ağaç kenar DOF'larını sil | kaynak problemleri için doğru, **özdeğer için yanlış**: kalan alt uzay değişmez değildir | — | önerilmez |

Uygulama notu: ARPACK'in mode-3'ü `OPinv`'i $M y$'ye (dual vektör) uygular. $(K-\sigma M)^{-1}$ gradyanları gradyanlara taşıdığından $P(K-\sigma M)^{-1}My=(K-\sigma M)^{-1}MPy$ olur, yani projeksiyonu **çözümden sonra** uygulamak yeterlidir. Çözümden önce $P$ uygulamak (dual vektöre primal projeksiyon) ilk denemede hatalı sonuç verdi; `n0lib.solve_projected` doğru sırayı kullanıyor.

### 2.3 Eksenel simetrik çözücüler (Faz 0'ın etiketleri)

**$m=0$ (TM$_0$) — $H_\varphi$ formülasyonu** (meridyen $(z,r)$, $x_0=z$, $x_1=r$):
$$\int_\Omega\Big[\partial_zH\,\partial_zv+\tfrac1r\partial_r(rH)\,\tfrac1r\partial_r(rv)\Big]\,r\,dA=k^2\int_\Omega Hv\,r\,dA .$$
Burada $\tfrac1r\partial_r(rH)=\partial_rH+H/r$. BC'ler şöyledir: eksende $H=0$ (esas). Magnetik simetri düzleminde de $H=0$ (π-modunun iris düzlemi). PEC duvar ve elektrik simetri düzlemi **doğal**dır. Alan büyüklükleri $E_z=\frac{1}{i\omega\varepsilon_0}\frac1r\partial_r(rH_\varphi)$ ve $E_r=-\frac{1}{i\omega\varepsilon_0}\partial_zH_\varphi$ ile elde edilir. $\nabla\times(H_\varphi\hat\varphi)=0\Rightarrow rH_\varphi=$ sabit $\Rightarrow H=0$ olduğundan **çekirdek yoktur**: Lagrange elemanlar tamamen doğrudur. Aynı operatör, $E_\varphi$ için PEC'de Dirichlet ile TE$_0$ ailesini verir.

**$m\ge1$ — karma formülasyon:** Alan $\mathbf E=(E_r\cos m\varphi,\ E_\varphi\sin m\varphi,\ E_z\cos m\varphi)$ biçimindedir. $\mathbf E_{mer}=(E_z,E_r)\in N1$ ve $u=rE_\varphi\in P1$ ile:
$$\int_\Omega\Big[r\,(\mathrm{rot}\,\mathbf E_{mer})^2+\tfrac1r\,|m\,\mathbf E_{mer}+\nabla u|^2\Big]dA=k^2\int_\Omega\Big[r|\mathbf E_{mer}|^2+\tfrac{u^2}{r}\Big]dA .$$
PEC duvarda ve eksende tüm sınır DOF'ları esastır: duvarda teğetsel $\mathbf E_{mer}$ ve $u$ sıfır, eksende $E_z=0$ ve $u=0$. Çekirdek $\{(\nabla p,\,-mp):p\in P1_0\}$'dır, yani 3D gradyanlar $\nabla(p\cos m\varphi)$. Bu çekirdek aynı projeksiyonla atılır (`exp_axisym.solve_m`). $1/r$ ağırlıkları, kuadratür noktaları eksenin üzerinde olmadığı için sorun çıkarmaz.

### 2.4 Fizibilite deneyleri: sonuçlar

**E1 — PEC kutu, N0, projeksiyonlu shift-invert** (`exp_box.py`; göreli hata $\lambda_h/\lambda-1$, ilk 8 mod):

| Geometri | $n$ | tet | N0 DOF | çekirdek boyutu | max\|hata\| | $\lambda_1$ hatası | montaj / faktör / özçözüm |
|---|---|---|---|---|---|---|---|
| küp $1^3$ | 8 | 3 072 | 3 032 | 343 | $2.5\cdot10^{-2}$ | $-1.1\cdot10^{-2}$ | 0.05 / 0.18 / 0.15 s |
| küp | 16 | 24 576 | 26 416 | 3 375 | $6.5\cdot10^{-3}$ | $-2.7\cdot10^{-3}$ | 0.40 / 0.32 / 10.4 s |
| küp | 24 | 82 944 | 91 656 | 12 167 | $2.9\cdot10^{-3}$ | $-1.2\cdot10^{-3}$ | 2.1 / 1.2 / 31 s |
| küp | 32 | 196 608 | 220 256 | 29 791 | $1.6\cdot10^{-3}$ | $-6.9\cdot10^{-4}$ | 5.4 / 5.5 / 64 s |
| kutu $1\times0.8\times0.6$ | 8 | 1 440 | 1 345 | 140 | $1.5\cdot10^{-2}$ | $-7.4\cdot10^{-4}$ | 0.02 / 0.10 / 0.05 s |
| kutu | 16 | 12 480 | 13 105 | 1 620 | $4.3\cdot10^{-3}$ | $-4.3\cdot10^{-4}$ | 0.18 / 0.17 / 4.2 s |
| kutu | 32 | 94 848 | 104 931 | 13 950 | $1.3\cdot10^{-3}$ | $-2.0\cdot10^{-4}$ | 1.9 / 2.2 / 20 s |

Yakınsama $O(h^2)$'dir. Bu yapısal meshlerde **bütün ayrık özdeğerler gerçeğin altında** çıktı, yani N0 ile Courant–Fischer üst sınırı yoktur. Özçözüm süreleri, SciPy ARPACK ile pypardiso'nun birlikte kullanımında çözüm başına ~50 ms'lik bir ek yük içeriyor (tek başına 4 ms ölçüldü). Üretim ortamında NGSolve/SLEPc ile bu kısım 5–10× kısalır (aşağıya bakın).

**E2 — PEC pillbox $R=L=1$ (gmsh OCC, düz tet, N0)** (`exp_pillbox3d.py`, `exp_timing.py`; ilk 10 mod: TM010, 2×TE111, 2×TM110, TM011, 2×TE211, TE011, TM111):

| $h$ | tet | düğüm | N0 DOF | TM010 hatası | max\|hata\| (10 mod) | mesh / montaj / faktör / özçözüm | tepe RSS |
|---|---|---|---|---|---|---|---|
| 0.18 | 2 778 | 712 | 2 537 | $-5.9\cdot10^{-3}$ | $2.0\cdot10^{-2}$ | 0.06 / 0.06 / 0.12 / 0.13 s | — |
| 0.13 | 7 045 | 1 616 | 6 846 | $-3.4\cdot10^{-3}$ | $8.9\cdot10^{-3}$ | 0.12 / 0.15 / 0.15 / 0.35 s | — |
| 0.10 | 14 944 | 3 196 | 15 121 | $-1.8\cdot10^{-3}$ | $3.8\cdot10^{-3}$ | 0.26 / 0.39 / 0.30 / 7.5 s | 239 MB |
| 0.07 | 42 820 | 8 439 | 45 144 | $-7.3\cdot10^{-4}$ | $1.6\cdot10^{-3}$ | 0.75 / 1.2 / 2.3 / 11 s | 632 MB |
| 0.05 | 115 136 | 21 432 | 124 701 | $-3.5\cdot10^{-4}$ | $7.1\cdot10^{-4}$ | 2.1 / 4.1 / 50 / 26 s | 2.4 GB |

Fiziksel ölçek: $R=88.3$ mm alınırsa TM010 = 1.300 GHz olur. $h=0.1R$'de frekans hatası $\approx-0.9\cdot10^{-3}$, yani **−1.1 MHz**'dir. Dejenere çiftler ayrık düzeyde $10^{-3}$ ile ayrışır (mesh simetrisizliği), örneğin TE111 3.6391 / 3.6394.

**E2' — NGSolve, eğri elemanlı yüksek derece Nédélec** (`exp_ngsolve.py`; tam HCurl uzayı, $\nabla H^1_{p+1}$ projeksiyonu, aynı SciPy çözücü):

| derece | maxh | serbest DOF | TM010 hatası | max\|hata\| (10 mod) | süre (çözüm) |
|---|---|---|---|---|---|
| 1 (`nograds`) | 0.10 | 11 477 | $-1.7\cdot10^{-3}$ | $2.3\cdot10^{-3}$ | 1.0 s (NGSolve PINVIT) / 5.2 s (SciPy) |
| 1 (`nograds`) | 0.07 | 49 182 | $-3.4\cdot10^{-4}$ | $1.1\cdot10^{-3}$ | 6.3 s (NGSolve PINVIT) |
| 2 | 0.30 | 4 329 | $+3.1\cdot10^{-4}$ | $4.7\cdot10^{-3}$ | 0.95 s |
| 2 | 0.20 | 11 964 | $+8.2\cdot10^{-5}$ | $1.4\cdot10^{-3}$ | 18 s* |
| **3** | **0.40** | **3 436** | $\mathbf{-9.2\cdot10^{-6}}$ | $\mathbf{8.2\cdot10^{-4}}$ | 0.6 s |
| 3 | 0.30 | 11 980 | $-5.6\cdot10^{-6}$ | $9.5\cdot10^{-5}$ | 11 s |
| 3 | 0.20 | 32 780 | $-1.1\cdot10^{-6}$ | $1.3\cdot10^{-5}$ | 122 s*, 3.5 GB |

\* Tam uzayda gradyan uzayının $K_p$ matrisi ($H^1_{p+1}$) SuperLU ile yavaş çözülüyor; $p=2$, maxh=0.14 koşusu 900 s'lik zaman aşımını geçti. NGSolve'un kendi seyrek Cholesky'si veya AMS ile bu süre düşer.

**Tuzak:** `HCurl(..., nograds=True)` yüksek dereceli gradyan fonksiyonlarını tabandan çıkarıyor. Bu, P1-gradyan projeksiyonuyla birlikte kullanıldığında $p=2,3$'te ~%2'lik **yanlış** sonuç verdi. Hata derece arttıkça azalmıyor. Tam uzay + tam gradyan uzayı kullanılmalıdır.

**Sonuç:** Aynı doğruluk için ($\sim10^{-3}$, 10 mod) $p=3$ eğri eleman ~3.4k DOF isterken N0 ~45k DOF ister (**~13×**). TM010'da fark 100×'ü aşar. Etiket kalitesi için veri üreticisinde $p=2$–3 eğri eleman tercih edilmelidir. Ağın Ritz'i ise ucuz N0 Gram'larıyla yapılabilir; kalan fark hibrit rafine adımıyla kapatılır, 2D'deki P1-model / P2-etiket ayrımının aynısı.

**E3 — eksenel simetrik** (`exp_axisym.py`; pillbox $R=L=1$ ve TESLA orta hücresi):

| Durum | Eleman | $h$ | DOF | Hata / sonuç | Süre |
|---|---|---|---|---|---|
| pillbox, $m=0$ (ilk 5 TM$_{0np}$) | P1 | 0.025 | 1 890 | $1.1\cdot10^{-4}$ … $1.4\cdot10^{-3}$ | 60 ms |
| pillbox, $m=0$ | P2 | 0.10 | 512 | TM010 $2.6\cdot10^{-7}$, max $1.1\cdot10^{-4}$ | 42 ms |
| pillbox, $m=0$ | P2 | 0.05 | 1 932 | max $6.5\cdot10^{-6}$ | 100 ms |
| pillbox, $m=0$ | P2 | 0.025 | 7 480 | max $4.2\cdot10^{-7}$ ($O(h^4)$) | 245 ms |
| pillbox, $m=1$ (TE111, TM110, TM111, TE121, TE112, TM120) | N1×P1 | 0.05 | 1 813 (çekirdek 434) | $1.2\cdot10^{-3}$ … $5.5\cdot10^{-3}$ | 80 ms |
| pillbox, $m=1$ | N1×P1 | 0.025 | 7 241 (çekirdek 1 771) | $3.9\cdot10^{-4}$ … $1.4\cdot10^{-3}$ | 290 ms |
| **TESLA orta yarım-hücre**, $m=0$ | P2 | 2 mm | 5 748 | $f_\pi=1.30125$, $f_0=1.27684$ GHz, $k_{cc}=1.893\%$ | 96 + 92 ms |
| TESLA | P2 | 1 mm | 22 201 | $f_\pi=1.30103$, $f_0=1.27671$, $k_{cc}=1.887\%$ | 0.9 + 0.6 s |
| TESLA | P2 | 0.5 mm | 87 169 | $f_\pi=\mathbf{1.30098}$, $f_0=\mathbf{1.27668}$ GHz, $k_{cc}=\mathbf{1.886\%}$ | 5.0 + 3.9 s |
| TESLA **tam 3D** (döndürülmüş profil, iris düzlemleri PEC ⇒ 0-modu) | N0 | 12 mm | 7 278 | TM010 1.2773 GHz (2.5D'ye göre +0.05 %); ilk dipol çifti 1.8233 / 1.8237, sonraki 1.8914 / 1.8921 GHz | ~1 s |

TESLA geometrisi: $R_{iris}=35$, $R_{eq}=103.3$, $L=57.7$, $A=B=42$, $a=12$, $b=19$ mm (Aune vd. 2000). Duvar açısı düşeyden 13.3°. π-modu için yarım hücrede iris düzlemi **magnetik** duvar ($H_\varphi=0$), ekvator düzlemi **elektrik** duvar (doğal) alınır. 0-modunda ikisi de elektrik duvardır. $k_{cc}=2(f_\pi-f_0)/(f_\pi+f_0)$ TESLA'nın yayımlanmış değeri olan 1.87 %'ye 0.02 puan yakındır. π-modu 1.3 GHz hedefinin %0.08 içindedir; kalan fark için gerçek hücre boyutları ve ayar payları ayrıca kontrol edilmelidir. 2.5D (0.1 s) ile 3D (1 s, kaba) aynı TM010'u veriyor.

**E6 — 3D skaler Dirichlet (Faz 1 maliyeti)** (aynı pillbox tet mesh'leri, $\lambda_1=(j_{01}/R)^2+(\pi/L)^2$): $h=0.1$'de P1 1 685 DOF ile $+2.1\cdot10^{-2}$ (0.2 s), P2 16 806 DOF ile $+5.3\cdot10^{-4}$. $h=0.13$'te P2 7 553 DOF, $+9.6\cdot10^{-4}$, 0.55 s. Skaler 3D, Maxwell 3D'den ~3–10× ucuzdur ve 2D'deki P1-taban deneyiminin (+%0.2–0.7) 3D karşılığı P1'de ~%2'dir. Yani 3D'de P1 Ritz tabanı **daha kötüdür** ve P2 sorgu noktaları (kenar ortaları) daha önemli hâle gelir.

### 2.5 Örnek başına maliyet ve depolama tahmini

| Rejim | Tipik boyut | Etiket süresi (4 çekirdek) | Depolama/örnek (float32, gzip öncesi) | 5 000 örnek |
|---|---|---|---|---|
| Bugünkü 2D TM | ~1.5k düğüm, P2 | ~0.15 s | ~50 KB | ~0.25 GB, ~3 dk |
| 2.5D $m=0$ + $m=1$ | 2–6k düğüm, P2 + N1×P1 | 0.1 + 0.3 s | ~150–300 KB | ~1.5 GB, **~30 dk** |
| 3D skaler | 2–4k düğüm (P1) / 15k (P2) | 0.2–2 s | ~0.3–1 MB | 2–5 GB, 0.5–3 saat |
| 3D Maxwell N0, $h\approx R/10$ | 15k kenar, 3.2k düğüm, 15k tet | 2–8 s | düğüm 38 KB + tet 240 KB + 8 mod × 15k kenar 480 KB + özellikler 250 KB ≈ **1 MB** | ~5 GB, **7–11 saat** |
| 3D Maxwell N0, $h\approx R/14$ | 45k kenar | ~15 s | ~3 MB | ~15 GB, ~20 saat |
| 3D Maxwell $p=3$ eğri (etiket), TESLA hücresi + port | $10^5$–$10^6$ DOF | 1–10 dk | etiketler kaba N0 mesh'e örneklenirse ~1–3 MB | 1k örnek ≈ 1–2 gün, 32 çekirdek |

Weyl sayımıyla ($N\approx|\Omega|k^3/3\pi^2$) ilk 10 modun dalga boyu pillbox'ta $\sim1.3R$ olur. N0'da %0.1 düzeyi için dalga boyu başına ~15–20 kenar gerekir, bu da $h\approx0.07R$ ve ~45k DOF demektir.

---

## 3. Model tasarımı

### 3.1 Mevcut koddan ne taşınır?

| Bileşen | Bugün (2D, `eigenspace_operator.py`) | 2.5D $m=0$ | 3D Maxwell |
|---|---|---|---|
| Kodlayıcı (RFF + özellikler → MLP) | $d=2$ | aynen ($x=(z,r)$) | `grid_dim=3`; 3D özellikler |
| Kütle-farkında lineer attention | düğüm alanı $w_n$ | $w_n=2\pi r_n\cdot$alan | $w_n$ = düğüm hacmi (Voronoi-benzeri $\tfrac14\sum$ tet) |
| Başlık | $D\to m$ skaler, torsiyon kapısı | $D\to m$ skaler, **eksen kapısı** $\psi=\tilde r\cdot N$ (+ magnetik duvar) | $D\to 3m$ düğüm vektörü → **Whitney kenar DOF'u** (§3.3), **kapı yok** ($\mathbf H$) |
| Gram montajı | `_p1_galerkin` (kapalı form) | yeni `_p1_galerkin_axisym` (r-ağırlıklı kuadratür) | yeni `_n0_galerkin` veya dataloader'dan seyrek $K,M$ |
| Çekirdek | yok | yok | **Schur-projekteli kütle** $M_{div}$ (§3.4) |
| Ritz (`_ritz`, `_BroadenedEigh`) | aynen | aynen | aynen ($M\to M_{div}$) |
| Span / compliance kayıpları | `eigenspace_grams`, `span_residual`, `ritz_compliance` | aynen (yeni Gram ile) | aynen, $G_M\to G_{M,div}$, $G_A$ = curl–curl |
| Frekans | $f=c\sqrt\lambda/(2\pi s)$ | aynen | aynen |

### 3.2 Ölçek: $N=10^4$–$10^6$ nokta

- **Bugünkü mass-aware lineer attention** $O(ND^2)$'dir ve çözünürlükten bağımsızdır: quadrature ağırlıklı integral operatörüdür, bu yüzden NEO 2k noktada eğitip 512k'de sıfır-atış çalışıyor. $N=10^5$, $D=128$, 4 katman için ileri geçiş ~10 GFLOP eder, aktivasyonlar örnek başına ~1 GB tutar ve 24 GB GPU'da batch 4–8 sığar. **Öneri:** Başlangıçta aynen kullanılmalı. Eğitim kaba mesh'te ($N\approx3$–10k düğüm), Ritz/değerlendirme ince mesh'te yapılmalı. Ritz Gram montajı $O(T\,m^2)$'dir ve incede de ucuzdur.
- **Transolver / Transolver++ dilim attention'ı** ($M\approx32$–64 öğrenilmiş dilim, lineer) milyon noktaya ölçeklenen, en olgun seçenektir (Transolver++ tek GPU'da $10^6$ nokta). Mass-aware sürümü doğaldır: dilim ağırlıkları $\sum_n w_n s_{nk}$ ile hesaplanır. Düşük modlar global ve pürüzsüz olduğu için dilim tokenleri iyi bir kaba temsil sunar.
- **GINO / GAOT (latent ızgara + GNO):** SDF girdisi, yerel GNO ile ızgaraya kodlama ve düzgün ızgarada FNO/ViT. Düşük özmodlar pürüzsüz olduğundan $32^3$–$64^3$ latent ızgara yeterlidir. Portlar/kenar tekillikleri ise yerel GNO ister. GAOT3D milyon-noktalı 3D CFD'de SOTA. Kavite gibi **dışbükeye yakın tek parça** domainlerde latent ızgara çok uygundur.
- **Öneri sırası:** (1) Mevcut lineer attention, düğümlerde; (2) sığmazsa Transolver++-tipi dilimleme; (3) port/kuplör gibi çok ölçekli geometride GAOT/GINO. 2.5D'de ölçek sorunu hiç yoktur (meridyen mesh 2–6k düğüm).

### 3.3 Baz $H(\mathrm{curl})$'da nasıl temsil edilir?

Seçenekler:

1. **Kenar DOF'larını doğrudan tahmin etmek** (Whitney 1-form katsayıları): Bu, kenar-orta noktalarında yöne duyarlı bir çıktı ister ($u_e=\boldsymbol\psi(x_e)\cdot\mathbf t_e$). Kenar sayısı ~6× düğüm sayısıdır (pillbox'ta 19.6k kenar / 3.2k düğüm) ve işaret kenar yönüne bağlıdır. Ağ sorgu tabanlıysa kenar ortalarında değerlendirmek mümkündür. Bu, 2D'deki "P2 sorgu noktaları" önerisinin (docs/15 #7) 3D karşılığıdır.
2. **Düğüm vektörü tahmin edip kenara eşlemek:** $u_e=\tfrac12(\boldsymbol\psi(a)+\boldsymbol\psi(b))\cdot(b-a)$. Bu, seyrek, sabit, lineer bir haritadır ($R\in\mathbb R^{E\times3N}$). Attention düğümlerde kalır, $H(\mathrm{curl})$-uyumluluk ise yapısaldır. **Maliyeti (E4):** analitik modları düğümlerde örnekleyip eşleyen "ideal ağ"da Ritz hatası $n=12/24/32$ için $2.1\cdot10^{-2}/5.5\cdot10^{-3}/3.1\cdot10^{-3}$ çıktı. Kanonik kenar-integrali interpolantında bu değerler $2.3\cdot10^{-3}/6.3\cdot10^{-4}/3.5\cdot10^{-4}$'tür. Her ikisi de $O(h^2)$, ama trapez eşlemenin sabiti **~9×** büyüktür. Bu, 2D'deki P1 tabanının 3D karşılığıdır. Simpson ($\tfrac16[\psi(a)+4\psi(m)+\psi(b)]$, orta noktada bir sorgu) bu farkın çoğunu kapatır.
3. **Potansiyel tahmin etmek:** $\mathbf E=\nabla\times\mathbf A$ ile $\nabla\cdot\mathbf E=0$ otomatik sağlanır (Richter-Powell vd. 2022 tipi). Ama ayrık düzeyde $\nabla\times N0\subset RT0$'dır, yani sonuç kenar uzayında değil yüz uzayındadır. Ritz'i $RT0$'da yapmak için $\mathbf H$-dual problem ve karma form gerekir, üstelik $\mathbf A$'nın kendi gauge'u vardır. **Önerilmez**; Schur projeksiyonu aynı işi kesin ve ucuz yapar.

**Seçim:** (2) düğüm vektörü → Whitney eşleme ile başlanmalı, yakınsama tabanı darboğaz olursa kenar-orta sorgusu (Simpson) eklenmelidir.

**Vektör çıktı ve dönme:** Çıktı artık bir vektör alanıdır ve $\boldsymbol\psi(Qx)=Q\boldsymbol\psi(x)$ gibi dönüşmelidir. Seçenekler şunlar: (i) PCA/atalet çerçevesinde kanonikleştirme. Pillbox ve eliptik hücreler eksenel simetrik olduğundan tam da "izotropi laneti" durumudur (docs/15 #6); eksen doğrultusu kararlıdır ama azimut kararsızdır. (ii) Frame averaging. (iii) e3nn tipi eşdeğer katmanlar (Geiger & Smidt 2022). Başlangıç için $SO(3)$ dönme augmentasyonu yeterlidir. Ritz span'e bağlı olduğundan dönme altında **span**'in eşdeğer olması yeterlidir.

### 3.4 Ritz değerlerini çekirdekten uzak tutmak: Schur-projekteli Gram

$V\in\mathbb R^{E\times m}$ (iç kenar DOF'ları) ve $P=I-GK_p^{-1}G^\top M$ ($M$-ortogonal projeksiyon, ayrık diverjanssız alanlara) için $KG=0$ olduğundan:
$$(PV)^\top K(PV)=V^\top KV,\qquad (PV)^\top M(PV)=V^\top MV-B^\top K_p^{-1}B=:M_{div},\quad B=G^\top MV .$$
Yani rijitlik Gram'ı değişmez, yalnız kütle Gram'ına bir **Schur tümleyeni** eklenir. Bu işlem, $\mathrm{span}(V)\oplus\nabla P1$ üzerinde Ritz yapıp sıfırları atmakla birebir aynıdır. Hedef modlar $T$ gradyanlara $M$-ortogonal olduğundan $V^\top MT=(PV)^\top MT$ olur ve span kaybının çapraz blokları da değişmez.

**E4 ölçümleri** (`exp_ritz_ml.py`; kutu, $n=24$: 41 571 DOF, 5 382 iç düğüm; $T$ = ayrık ilk 8 mod):

| Deney | Düz Ritz (mevcut kod aynen) | Schur-projekteli |
|---|---|---|
| $V=T_{1..6}+\varepsilon\nabla\phi$, gradyan kütle payı %1 | $\theta/\lambda_h-1$: −1.0, −1.1, −1.0, −2.0 … % | $\max|\cdot|=1.4\cdot10^{-14}$ |
| aynı, pay %8 | **−12.8, −11.8, −5.7, −9.7** … % | $1.5\cdot10^{-14}$ |
| aynı, pay %50 | **−76, −27, −12, −18** … % | $2.8\cdot10^{-14}$ |
| span kaybı $r_M$ (kütle normu), pay %8 / %50 | 0.060 / 0.174 | $\sim10^{-16}$ |
| + 1 saf gradyan kolon | $\theta_1=1.6\cdot10^{-13}$ (**çöküş**); compliance $\mathrm{tr}(G_A^{-1}G_M)=6\cdot10^{12}$ (temiz span: 0.146) | kolon yok edilir (normalize özdeğer $<10^{-14}$) → rank-revealing atma / ridge |
| 16 rastgele pürüzsüz alan (eğitilmemiş başlık vekili) | medyan gradyan payı %6; $\theta_1$ düz 171.7 < projekteli 179.3 | — |
| maliyet ($K_p$ faktörü + $m=16$ çözüm, SuperLU, CPU) | — | $n=24$: 160 + 35 ms; $n=32$ (14k iç düğüm): 177 + 105 ms |

Sonuçlar:

- **Mevcut `_ritz` + `ritz_compliance` 3D'ye aynen taşınırsa yanlıştır.** Kendinden denetimli compliance kaybı ağı gradyan üretmeye iter, çünkü sınırsız bir ödül vardır. Gradyan payı küçük olsa bile Ritz frekansları sistematik olarak düşük çıkar.
- **Denetimli span kaybı ($M$-normu) projeksiyonsuz da "gradyan-karşıtı"dır.** Gradyan içeriği hedefleri açıklamaya katkı vermez, $G_{VV}$'yi büyütür ve $r_M$'yi artırır (%50 payda 0.174). Ceza gibi davranır ama kesin değildir. **Enerji normu** ($K$, curl–curl) gradyanları hiç görmez.
- **Öneri:** Eğitimde ve çıkarımda $G_M\to M_{div}$. `eigenspace_grams` fonksiyonuna $B=G^\top M[V|T]$ ve $K_p^{-1}B$ eklenir. $K_p$ örnek başına sabittir (P1 Laplace; $\mathbf E$'de Dirichlet, $\mathbf H$'de Neumann + bir düğüm sabitleme). Faktörü dataloader'da (CPU, CHOLMOD/PARDISO) bir kez alınır. `torch.autograd.Function` içinde ileri geçiş $Z=K_p^{-1}B$, geri geçiş $\bar B=K_p^{-1}\bar Z$ olur ($K_p$ simetrik). GPU'da cuDSS tabanlı seyrek Cholesky de bir seçenektir. Tam-sıfır kütleli yönler (saf gradyanlar) için mevcut `_jacobi_cholesky`'nin ölü-kolon mantığı ($d\to0$) yeterlidir: gradyan kolonunda $G_A$ ve $M_{div}$ birlikte sıfırlanır.
- **Upper bound:** Projekteli Ritz değerleri **ayrık** $\lambda_h$'nin üst sınırıdır (ayrık diverjanssız alt uzayda min–max). Kesin $\lambda$'nın üst sınırı değildir; N0'da $\lambda_h<\lambda$ çıktı (E1/E2). Sertifika istenirse Faz 2'de ayrı bir iş olarak ele alınmalıdır.
- Alternatifler: (i) $\|G^\top MV\|^2$ cezası ucuzdur ama kesin değildir ve Ritz'i kurtarmaz. (ii) Tree–cotree: §2.2'deki nedenle özdeğerde yanlıştır.

### 3.5 Sınır koşulu: kapı mı, $\mathbf H$ mı? (E5)

Ağın başlığını taklit eden sabit bir "sözlük" kullanıldı: derece $\le p$ monomiyal × $\mathbf e_i$ düğüm vektörleri ($m=30/60/105$), düğüm→kenar eşleme ve projekteli Ritz (`exp_gate.py`). Pillbox, $h=0.18/0.13/0.10$. Aşağıdaki tablo $h=0.10$'da (15k DOF) $\theta/\lambda_h-1$ değerlerini veriyor:

| Varyant | $p=3$: TM010 / TE111 / 4.–6. mod | $p=4$: TM010 / TE111 / 4.–6. mod | $h$'ye bağımlılık |
|---|---|---|---|
| A: skaler torsiyon kapısı $\psi=w\,\mathbf N$ + Hodge | +7.6e-2 / +2.6e-2 / +7.6–8.2e-2 | +6.5e-2 / +2.4e-2 / +7.6–8.2e-2 | taban $h$ ile değişiyor (h=0.18'de +2.0e-1); $p$ ile **iyileşmiyor** |
| B: kapı, projeksiyonsuz | **−74 %** / −63 % | **−74 %** / −83 % | Ritz < gerçek (çöküş) |
| C: kapısız, sınır DOF'ları sıfır | +2.9e-2 / +1.8e-1 / +1.0–3.1e-1 | +1.0e-3 / +5.6e-3 / +3.1–7.2e-2 | $p=2$'de **h küçüldükçe kötüleşiyor** (+1.9 → +3.4): $1/h$ sınır tabakası |
| D: teğetsel kapı $w\mathbf N+(1-w)(\mathbf n\cdot\mathbf N)\mathbf n$ | +3.1 / +1.1 | +1.7 / +0.7 | $\mathbf n=-\nabla w/|\nabla w|$ kenar/medial eksende süreksiz ⇒ en kötü |
| **H: $\mathbf H$ formülasyonu, kapısız, tüm kenarlar, Neumann-$\nabla P1$ projeksiyonu** | **+3.3e-3 / +3.0e-2** / +0.36–0.61 | **+3.3e-3 / +3.0e-2 / +1.1–1.4e-2** | **mesh'ten bağımsız** (üç $h$'de aynı) |

Yorum:

- **A neden başarısız?** Kapı + projeksiyon teorik olarak yoğundur, $\mathbf E=P(w\mathbf N)$. Ama bunun için $(\mathbf E+\nabla q)/w$'nin sınırlı olması gerekir. Silindirin kenar çemberinde ($r=R$, $z=0$) iki duvar kesişir ve $q$'nun iki normal türevi aynı anda istenen değerleri alamaz. $q=0$ iken $\partial_zq=-E_z$ ve $\partial_rq=0$ çelişir. Bu yüzden gereken $\mathbf N$ kenarda tekildir ve yaklaşım yavaşlar. Gerçek kavitelerde iris–beam pipe birleşimi ve portlar tam böyle kenarlardır.
- **C** tam teğetsel BC'yi doğru uygular ama ağın çıktısı duvarda teğetsel olarak sıfıra gitmezse $O(1/h)$ enerji üretir. Sonuç çözünürlüğe bağımlıdır, bu da NEO'nun "kaba mesh'te eğit, incede çalıştır" özelliğini bozar.
- **H** hem uyumlu hem çözünürlükten bağımsızdır. Hatası tamamen sözlüğün ifade gücünden gelir: $J_1(j_{01}r)\hat\varphi$ düşük dereceli polinomla yavaş yaklaşır, ağ çok daha zengindir. Bu, bugünkü torsiyon kapısının 2D'de sağladığı iki özelliğin (uyumluluk + çözünürlük bağımsızlığı) 3D'deki tek temiz karşılığıdır.
- **Bedel:** $\mathbf E$ artık türetilmiş bir büyüklüktür, $\mathbf E=\nabla\times\mathbf H/(i\omega\varepsilon_0)\in RT0$ (tet başına sabit). $E_{acc}$, $R/Q$, $E_{pk}$ bir derece düşük doğrulukla hesaplanır; gerekirse E-formülasyonunda 1–2 ters iterasyonla (`src/fem_refine.py`'nin 3D karşılığı) rafine edilir. $B_{pk}$ (duvardaki teğetsel $\mathbf H$) ise doğrudan gelir. Ayrık özdeğerler bu testte gerçeğin **üstünde** çıktı ($\mathbf H$-N0: TM010 +0.38 %, $\mathbf E$-N0: −0.18 %). Bu garanti değildir ama pratik bir iki-yönlü kontrol sağlar.

### 3.6 Span / compliance kayıpları curl–curl ile

- $G_A=V^\top K_{curl}V$ (curl–curl), $G_M=M_{div}$ (Schur) ile `span_residual` ve `ritz_compliance` **değişmeden** çalışır (E4, E5 H satırı).
- **Enerji normunun fiziksel anlamı ($\mathbf H$ formülasyonunda):** $\|\nabla\times(\mathbf H-\Pi_V\mathbf H)\|^2=\omega^2\varepsilon_0^2\,\|\mathbf E-\Pi\mathbf E\|^2$. Yani $\mathbf H$'nin enerji-normu span kaybı, $\mathbf E$ alanının $L^2$ span kaybıdır. `span_norm: both`, iki alanı aynı anda denetler.
- Hedefler: Üretici, $\mathbf H$-N0 özvektörlerini (tüm kenarlar) doğrudan saklar. $\mathbf E$-hedefleri gerekirse $\nabla\times\mathbf H$'den türetilir.
- Ky Fan / compliance: $\mathrm{tr}(G_A^{-1}M_{div})$'in supremumu, pencilin diverjanssız kısıtlamasının en düşük $m$ özuzayında alınır. Projeksiyonsuz $M$ ile supremum $+\infty$'dur (E4: $6\cdot10^{12}$).

### 3.7 Eksenel simetrik yol (Faz 0): kodun ne kadarı taşınır?

- **Veri:** Meridyen üçgen mesh (bugünkü gmsh 2D akışı). Üretici, `solve_dirichlet_eigenmodes` yerine $m=0$ için `solve_m0` (P2, $r$-ağırlıklı formlar; eksen ve magnetik düzlem için Dirichlet, PEC için doğal koşul) ve $m=1$ için `solve_m` (N1×P1 + projeksiyon) çağırır. Geometri: eliptik hücre parametreleri + beam pipe + spline pertürbasyonu.
- **Model ($m=0$):** Bugünkü EigenspaceOperator'dan farklar şunlar: (i) `dirichlet_factor` duvar yerine eksende sıfır olur. $H_\varphi\propto r$ olduğundan $\psi=(r/r_{max})\cdot N(x)$ doğrudur; magnetik simetri düzlemi varsa ek bir çarpan gerekir. (ii) `_p1_galerkin` yerine $r$-ağırlıklı P1 Gram:
  $$L_T=\int_T\big[r\,\nabla\psi\cdot\nabla\phi+\psi\,\partial_r\phi+\phi\,\partial_r\psi+\psi\phi/r\big],\qquad M_T=\int_T r\,\psi\phi .$$
  İlk üç terim kapalı formdadır (P1 gradyanı sabit, $r$ lineer). $\psi\phi/r$ ve $M_T$ 6-noktalı Gauss ile hesaplanır; eksen düğümlerinde $\psi=0$ olduğundan tekillik yoktur. (iii) Kütle ağırlıkları $w_n\propto r_n\cdot$alan. Geri kalan her şey aynen kalır: attention, başlık, `_ritz`, span/compliance kayıpları, frekans formülü, OT'siz sıralama.
- **$m=1$:** Başlık iki alan verir: $\mathbf E_{mer}$ (2 bileşen, düğüm → N1 kenar eşlemesi) ve $u$ (skaler). Çekirdek projeksiyonu 2D'dedir ve ucuzdur (7k DOF'ta 0.3 s'lik tam çözümün içinde). Bu, Faz 2'nin tüm Nédélec makinesinin küçük bir provasıdır. $\mathbf H$-formülasyonlu $m\ge1$ varyantı da duvar kapısını kaldırır; yalnız eksen koşulları kalır.
- **Özellikler:** $r$, $z$, duvara uzaklık, duvar yönü, eğrilik (profil eğrisinin), eksene uzaklık = $r$; torsiyon (meridyen düzleminde Dirichlet) ve operatöre uyarlanmış landscape $\mathcal L w=1$ (eksende Dirichlet, duvarda Neumann). Global şekil kodu: hücre parametreleri ($A,B,a,b,R_{iris},R_{eq},L$) bir token olarak verilir (docs/15 #5).
- **Tahmini taşınma oranı:** model ~%85, kayıplar ~%100, converter ~%60 (yeni özellikler), üretici ~%50 (yeni formlar/geometri).

### 3.8 3D özellikler (torsiyon/uzaklık özelliklerinin karşılığı)

| 2D (bugün) | 3D karşılığı | Maliyet |
|---|---|---|
| `dist_bnd`, `dir_bnd` | SDF ve en yakın yüzey normali (KD-ağaçlı sınır üçgenlerine uzaklık) | $O(N\log N_b)$ |
| `curvature`, `convexity` | en yakın duvar noktasındaki ortalama ve Gauss eğrilikleri $H$, $K$ (ayrık açı açığı / kotanjant formülü) | ucuz |
| `torsion` $w/\max w$ | 3D torsiyon $-\Delta w=1$, $w|_{\partial\Omega}=0$ (bir P1 tet çözümü, pillbox $h=0.1$'de <0.2 s); $\max w$ saklanır. Dirichlet skaler için $\lambda_1\max w$ topta $\pi^2/6=1.645$ (2D diskte $j_{01}^2/4=1.446$) | 1 seyrek çözüm |
| — | $\mathbf H$ formülasyonu için "magnetik landscape": $\nabla w$ ve Neumann-tipi landscape; eksenel yapılarda eksene uzaklık $\rho$ ve azimut $(\cos\varphi,\sin\varphi)$ | ucuz |
| `cos/sin_principal` | atalet tensörünün asal eksenleri (eksenel simetride dejenere; frame averaging) | ucuz |
| `node_area` | düğüm hacmi (tet hacimlerinin 1/4'ü) = attention ağırlığı | ucuz |
| global kod | eliptik hücre parametreleri; yıldız cisimde $a_{\ell m}$ katsayıları | — |

Deney E5'te torsiyon gradyanından türetilen normal (D varyantı) kenar/medial eksende süreksizdi. Normal özelliği SDF'nin gradyanından ya da en yakın sınır üçgeninin normalinden alınmalıdır.

---

## 4. Literatür

| Çalışma | Yıl | Ne yapar | Geometri / boyut | Buradaki önemi |
|---|---|---|---|---|
| **Yaker, Markovic, Reineri, Kurkcuoglu, Zorzetti** — *Neural-Network Inverse Design of SRF Cavities and Transmons for Bosonic Quantum Computation* | 2026 | DNN ile SRF kavite (ve transmon) ters tasarımı; COMSOL verisi (2 000 üç-hücreli + 2 000 silindirik geometri, ~2 dk/örnek); gözlemlenenler için ~%5, transmon için ~%2 doğruluk; eliptik yerine **spline** parametrizasyonu | eksenel simetrik profil, skaler çıktılar | Bizim Faz 0'ın skaler-çıktılı karşılığı; alan/özuzay öğrenmiyor. Spline ailesi önerisini doğruluyor |
| **Kranjčević, Adelmann, Arbenz, Citterio, Stingelin** — *Multi-objective shape optimization of RF cavities using an evolutionary algorithm* | 2019 | Eksenel simetrik FEM + paralel evrimsel algoritma ile Pareto cephesi | 2.5D | SRF tasarım döngüsünün maliyet bağlamı |
| **Kranjčević, Zadeh, Adelmann, Arbenz, van Rienen** — *Constrained multiobjective shape optimization of SRF cavities considering robustness against geometric perturbations* | 2019 | Temel mod + ilk dipol bandı, geometrik pertürbasyona dayanıklılık | 2.5D, $m=0$ ve $m=1$ | Faz 0b'nin ($m=1$) neden gerekli olduğunu gösteriyor |
| **Wang, Tang, Wu, Feng** — *Multiobjective Bayesian optimization for the shape design of rf cavity in particle accelerators* | 2026 | Çok amaçlı Bayes optimizasyonu, kavite şekli | 2.5D | Yüzey-modeli/BO karşılaştırma tabanı |
| **Ziegler, Hahn, Isensee, Nguyen, Schöps** — *Gradient-based eigenvalue optimization for electromagnetic cavities with built-in mode matching* | 2024 | Maxwell özdeğerinin şekil türeviyle optimizasyon, mod takibi | kavite, FEM | Diferansiyellenebilir tasarım; bizim Ritz başlığının otomatik türevi aynı işi amortize eder |
| **Jiang, Wang, Wang, Xie** — *FieldTNN-based machine learning method for Maxwell eigenvalue problems* | 2024/25 | Tensör NN ile 2D/3D Maxwell özdeğerleri; diverjans kısıtı kayıpta, sahte çiftler otomatik elenir; düzgün kavitelerde <%1 | tek geometri (kare/küp/L) | 3D Maxwell nöral özçözücü; çekirdek sorununu **ceza** ile çözüyor (biz: kesin projeksiyon) |
| **Yang, Du, Liu** — *NEO: Learning Laplacian Eigenspace with Mass-Aware Neural Operators on Point Clouds* | 2026 | Nokta bulutu → LBO düşük-frekans özuzayı; redundant baz + RR; 2k'da eğit, 512k'da çalış, ARPACK'e göre 88× | 3D **yüzeyler** (skaler) | Bugünkü modelin şablonu; 3D'de ölçeklenebilirlik kanıtı |
| **Chang vd.** — *Shape Space Spectra* | 2025 | Sürekli şekil ailesi üzerinde özfonksiyon neural field'ları, iç içe varyasyonel ilke, kesişmede dinamik yeniden sıralama | 2D/3D şekil aileleri (skaler) | Parametrik şekil ailesi (eliptik hücre) için doğrudan analog |
| **Li, Kovachki, Choy, … , Anandkumar** — *Geometry-Informed Neural Operator (GINO)* | 2023 | SDF + GNO ↔ latent FNO; ayrıklaştırmadan bağımsız | 3D araç yüzeyleri/hacim | Latent ızgara seçeneği (§3.2) |
| **Wu, Luo, Wang, Wang, Long** — *Transolver* | 2024 | Physics-attention: noktaları öğrenilmiş dilimlere topla, lineer | genel geometri | Dilim attention |
| **Luo vd.** — *Transolver++* | 2025 | Milyon noktalı geometrilerde paralel Transolver | 3D endüstriyel ($10^6$ nokta) | $N=10^6$ ölçeği |
| **Wen, Kumbhat, Lingsch, Mousavi, Zhao, Chandrashekar, Mishra** — *GAOT* | 2025 | Çok ölçekli attentional GNO + ViT, geometri gömmesi; GAOT3D | 2D/3D keyfi domain | Port/kuplör gibi çok ölçekli 3D için |
| **Trask, Huang, Hu** — *Enforcing exact physics in SciML: a data-driven exterior calculus on graphs* | 2022 | Kombinatoryal Hodge teorisiyle tam dizili, yapı-koruyan öğrenilmiş modeller; H(div)/H(curl) örnekleri | graf | $G$/tam dizi fikrinin ML karşılığı |
| **Actor vd. (Trask grubu)** — *Data-driven Whitney forms for structure-preserving control volume analysis* | 2023/24 | Whitney formlarıyla öğrenilen kontrol hacimleri, veri-güdümlü FEEC | mesh | Whitney-uyumlu öğrenilmiş taban |
| **Kinch, Shaffer, Armstrong, Meehan, Hewson, Trask** — *Structure-Preserving Digital Twins via Conditional Neural Whitney Forms* | 2025 | Latent değişkene koşullu, FEEC içinde öğrenilen indirgenmiş FE tabanı; attention ile koşullama | mesh | **Koşullu Whitney tabanı = bizim geometri-koşullu kenar başlığı** |
| **Shaffer, Koohy, Kinch, Hsieh, Trask** — *Structure-Preserving Learning Improves Geometry Generalization in Neural PDEs* | 2026 | Yapı koruma → geometri genellemesi (Geo-NeW) | genel geometri | Yapı korumanın geometri genellemesine katkısı |
| **Shaffer, Kinch, Hsieh, Trask** — *A meshfree exterior calculus … from point clouds* | 2026 | Nokta bulutunda dış hesap | nokta bulutu | Mesh'siz 3D'de Hodge yapısı |
| **Richter-Powell, Lipman, Chen** — *Neural Conservation Laws: A Divergence-Free Perspective* | 2022 | Diferansiyel formlarla yapısal diverjanssız NN | sürekli | Potansiyel seçeneği (§3.3, önerilmedi) |
| **Geiger, Smidt** — *e3nn: Euclidean Neural Networks* | 2022 | E(3)-eşdeğer katmanlar | nokta/graf | Vektör çıktının dönme eşdeğerliği |
| **Christensen vd.** — *Predictive and generative ML models for photonic crystals* (Nanophotonics) | 2020 | Fotonik kristal bant yapısı tahmini/üretimi | periyodik 2D | Fotonik bant = periyodik Maxwell özproblemi |
| *Predicting band structures for 2D photonic crystals via deep learning* | 2024 | CNN ile bant yapısı, çözücüye göre mertebeler hızlı | 2D | aynı |
| *Symmetry-aware framework for bidirectional modeling of 3D photonic crystals and band structures* (Materials & Design) | 2026 | Simetri-farkında attention ile 3D PhC bant tahmini + difüzyonla ters tasarım | **3D** periyodik | Tek 3D Maxwell-bant ML örneği; skaler çıktılar (yazar listesi doğrulanamadı) |
| *Meshless optical mode solving using scalable deep deconvolutional neural network* (Sci. Rep.) | 2023 | Parametre → mod görüntüsü, dalga kılavuzu mod çözücü | 2D kesit | Kılavuz mod çözücü vekili |
| *A Physics-Informed Neural Network-Based Waveguide Eigenanalysis* (IEEE) | 2024 | PINN ile kapalı kılavuz modları | 2D kesit | PINN tabanı |
| *High precision, full-vector optical mode solving in waveguides via fourth-order derivative PINNs* | 2025 | Tam-vektör kılavuz modları (teğetsel E ve H) | 2D kesit | Vektör mod çözücü ML |
| **Arbenz, Geus** — *A comparison of solvers for large eigenvalue problems occurring in the design of resonant cavities* | 1999 | Rezonans kavite FEM özçözücüleri (subspace, block Lanczos, IRL, Jacobi–Davidson), gradyan projeksiyonu | 3D kavite | §2.2 projeksiyon yöntemi |
| **Boffi** — *Finite element approximation of eigenvalue problems* (Acta Numerica) | 2010 | Sahte modlar, discrete compactness, kenar elemanları | teori | §1.2 |
| **Costabel, Dauge** — *Weighted regularization of Maxwell equations in polyhedral domains* | 2002 | Düzenlileştirilmiş nodal FE'nin re-entrant kenarlarda yanlış limiti ve ağırlıklı düzeltme | 2D/3D | §1.2 |
| **Nédélec** — *Mixed finite elements in ℝ³* | 1980 | H(curl)/H(div)-uyumlu elemanlar | teori | §1.2 |
| **Bossavit** — *Whitney forms: a class of finite elements for three-dimensional computations in electromagnetism* | 1988 | Whitney formlarının EM'de kullanımı | teori | kenar DOF'larının geometrik anlamı |
| **Aune vd.** — *Superconducting TESLA cavities* (PRST-AB 3, 092001) | 2000 | TESLA 9-hücre geometrisi, $k_{cc}$, alan oranları | 2.5D | E3 doğrulama geometrisi |
| **Halbach, Holsinger** — *SUPERFISH* | 1976 | Eksenel simetrik kavite çözücüsü ($rH_\varphi$) | 2.5D | Faz 0 referansı |
| **Omega3P / ACE3P** (SLAC) | — | 6. dereceye kadar hiyerarşik Nédélec, kuadratik tet, paralel | 3D | Endüstriyel referans |
| **Palace** (AWS) | 2023– | MFEM tabanlı 3D FE EM; özmod, eğri eleman, AMR | 3D | Açık kaynak üretim çözücüsü |

**Boşluk:** Geometri → (3D Maxwell veya 2.5D $m\ge0$) **özuzay** haritası öğrenen, Whitney-uyumlu ve Hodge-projekteli Ritz kullanan bir nöral operatör bulamadık. Yakın komşular üç koldan geliyor: NEO (skaler, yüzey), FieldTNN (tek geometri, ceza) ve koşullu neural Whitney formları (özdeğer değil, dinamik).

---

## 5. Yol haritası

### Faz 0a — Eksenel simetrik, $m=0$ monopol (önerilen başlangıç)

- **Yeni dosyalar:** `src/data_gen/axisym_generator.py` (eliptik hücre + beam pipe + spline ailesi; `solve_m0` P2; π/0 BC seçenekleri; $E_{acc}$, $R/Q$, $E_{pk}/E_{acc}$, $B_{pk}/E_{acc}$ türetilmiş etiketleri), `src/models/axisym_galerkin.py` (`_p1_galerkin_axisym`, `axis_factor`).
- **Değişenler:** `src/data/dataset_converter.py` (meridyen özellikleri: $r$, eksen/duvar uzaklıkları, landscape; `node_area`→$r\cdot$alan; şekil kodu), `src/models/eigenspace_operator.py` (`geometry: axisym` anahtarı → Gram ve kapı seçimi), `src/training/lightning_module.py` (`eigenspace_grams` Gram fonksiyonunu parametre olarak alır), `configs/axisym_m0.yaml`, `validate_data.py` (pillbox TM$_{0np}$ analitik), `src/fem_refine.py` (ağırlıklı P2 rafine).
- **Veri/hesap:** 5–10k geometri × (π ve 0 BC) × 6 mod; ~0.2 s/örnek ⇒ 4 çekirdekte <1 saat, ~1.5 GB. Eğitim, bugünkü 2D ile aynı ölçektedir (tek GPU, saatler).
- **Zorluklar:** Eksen civarında $1/r$ ağırlıklı kuadratür; mod aileleri (TM$_0$ passband ile TE-benzeri modlar karışmaz, $m=0$ TM'de yalnız TM$_0$ vardır); beam pipe kesim frekansının üstündeki modlar kapalı-uç BC ile yapaydır (kapatma düzleminde elektrik/magnetik duvar seçimi etiketlenmelidir).
- **Başarı ölçütleri:** Test setinde π-modu $|f-f_{ref}|/f<10^{-3}$ (medyan), $k_{cc}$ mutlak hatası <0.05 puan, $R/Q$ ve alan oranları <%2. Ritz her örnekte $\ge$ P2 etiketi (aynı mesh'te P1 ⊂ P2 ve formülasyon uyumlu olduğundan bu bir üst sınırdır). Kaba mesh'te eğitip 4× ince mesh'te sıfır-atış hatasının artmaması.

### Faz 0b — Eksenel simetrik, $m\ge1$ (dipol/HOM)

- **Yeni:** `solve_m` (N1×P1, projeksiyon; `exp_axisym.py`'den), iki-alan başlık (`mer_edge_head`: düğüm vektörü → N1 kenar eşlemesi + skaler $u$), 2D `G` matrisi ve Schur-projekteli Gram (`src/models/hcurl.py`: `edge_map`, `schur_mass`, `KpSolve` autograd).
- **Veri/hesap:** 5k geometri × 6–10 dipol modu; ~0.3 s/örnek.
- **Zorluklar:** Çekirdek projeksiyonunun eğitim döngüsüne girmesi (burada 2D'de, $K_p$ 1–5k boyutlu, ucuz), kenar yönü işaretleri, $\pm$ azimutal çiftler (m≥1 her mod zaten $\cos/\sin$ çifti; meridyende tek).
- **Başarı:** Pillbox $m=1$ ilk 6 mod <%0.5; eliptik hücre ilk dipol passband'ı <%0.5; eğitimde projekteli ve projeksiyonsuz Ritz farkı (gradyan sızıntısı) raporlanıp sıfıra yakın.

### Faz 1 — 3D skaler (altyapı)

- **Yeni:** `src/data_gen/generator3d.py` (gmsh OCC: yıldız cisim $Y_\ell^m$, kutu, silindir, top; tet çıkarımı `n0lib.gmsh_mesh` gibi), `_p1_tet_galerkin` (kapalı form: $\nabla\lambda_i$ tet başına sabit; kütle $|T|(1+\delta_{ij})/20$), 3D özellikler (§3.8).
- **Değişen:** `dataset_converter.py` (3D özellik çıkarımı; `_torsion_function` için `MeshTet`), `dataset.py` (tet'ler, değişken $N$ padding, bellek), `eigenspace_operator.py` (`grid_dim=3`, tet Gram).
- **Veri/hesap:** 5k örnek × 3–10k düğüm; P2 etiket ~1 s ⇒ ~1.5 saat, 2–5 GB. Eğitim tek GPU, batch 8, $N\le10^4$.
- **Zorluklar:** P1-taban tabanı 3D'de ~%2 (E6) ⇒ P2 sorgusu/rafine gerekir; bellek; attention'ın $N$ ile ölçeklenmesi.
- **Başarı:** $\lambda_{1..6}$ Ritz hatası <%1 (P2 sorgulu); $N=2$k'dan $N=50$k'ya sıfır-atış ölçekleme.

### Faz 2 — 3D Maxwell ($\mathbf H$ formülasyonu, Whitney N0)

- **Yeni:** `src/data_gen/maxwell3d.py` (N0 veya NGSolve $p=2$–3 etiketleri; **H-formülasyonu**, tüm kenarlar, Neumann-$\nabla P1$ projeksiyonu; E-formülasyonu isteğe bağlı iki-yönlü kontrol; hedef = kenar DOF'ları, etiket mesh'i ≠ model mesh'i ise kaba N0 mesh'e $L^2$ izdüşüm), `src/models/hcurl.py` (3D `G`, `edge_map`, `_n0_galerkin`: Whitney $\mathbf w_{ij}=\lambda_i\nabla\lambda_j-\lambda_j\nabla\lambda_i$, $\nabla\times\mathbf w_{ij}=2\nabla\lambda_i\times\nabla\lambda_j$, kütle $\int\lambda_a\lambda_b=|T|(1+\delta_{ab})/20$; ya da dataloader'dan seyrek $K,M$), `KpSolve` (CPU CHOLMOD/PARDISO veya GPU cuDSS), `src/fem_refine3d.py` (E veya H formülasyonunda 1–2 LOBPCG adımı).
- **Değişen:** `eigenspace_operator.py` (vektör başlık $D\to3m$, kapı yok), `lightning_module.py` (`eigenspace_grams` → $G_{M,div}$; enerji normu = curl–curl), `dataset.py` (kenar listesi + yön işaretleri, $G$, $K_p$ faktörü), `configs/maxwell3d.yaml`.
- **Veri/hesap:** 5k örnek, N0 $h\approx R/10$–$R/14$ (15–45k DOF): 7–20 saat/4 çekirdek, 5–15 GB; ya da NGSolve $p=3$ ~3–10k DOF ile ~1 s/örnek. Eğitim: $N\approx3$–10k düğüm, batch 4–8, tek 24–40 GB GPU; Schur çözümleri CPU'da ~5–100 ms/örnek.
- **Zorluklar (risk sırasıyla):** (1) Gradyan sızıntısı ve compliance kaybının istismarı (Schur ile çözülüyor ama eğitim döngüsünde doğru uygulanmalı). (2) Etiketlerin dejenere/kazara-dejenere kümeleri; span kaybı bunu çözüyor ama $K$ seçimi kümeleri bölmemeli (≥ bir sonraki boşluk). (3) Düğüm→kenar trapez tabanı (~%0.3–2), Simpson/kenar-orta sorgusu ile. (4) Vektör çıktının dönme eşdeğerliği. (5) Kulplu domainler (koaksiyel QWR): $\mathbf H$-formülasyonunda harmonik alanlar çekirdeğe eklenmeli (kohomoloji tabanı). (6) Bellek ($N=10^5$).
- **Başarı:** Pillbox, kutu ve 3D eliptik hücrede ilk 10 mod <%1 (Ritz), rafine sonrası <$10^{-3}$. Dejenere kümelerde alt uzay hatası <%2. Her örnekte projekteli Ritz $\ge\lambda_h$. Eksenel simetrik geometrilerde Faz 0 modeliyle tutarlılık.

### Neden bu sıra?

Faz 0a en düşük riskli ve en yüksek fiziksel değerli adımdır. Etiketleri zaten doğrulandı (TESLA $k_{cc}$ 1.886 %), kod farkı birkaç yüz satırdır ve sonuç doğrudan bir SRF tasarım aracıdır (π-modu, $R/Q$, alan oranları). Faz 0b, Faz 2'nin tüm yeni matematiğini (kenar DOF'ları, $G$, Schur Gram, projeksiyonlu Ritz, kayıpların curl–curl sürümü) 10–100× daha ucuz bir ortamda test etmeyi sağlar. Faz 1 yalnız altyapı değeri taşır ve Faz 2 ile paralel yürütülebilir; ölçekleme sorunları Faz 2'ye kalırsa atlanabilir. Faz 2 en pahalı ve en riskli adımdır. Ama E4/E5, gerekli iki yapısal kararın ($\mathbf H$ formülasyonu, Schur-projekteli Gram) doğru olduğunu bugünden gösteriyor.

---

## 6. Deney scriptleri (`scripts/research_3d/`)

| Script | İçerik | Süre (4 çekirdek) |
|---|---|---|
| `n0lib.py` | gmsh 3D mesh çıkarımı, skfem N0 montajı, $G$ (yön: `mesh.edges[0]→[1]` = +, doğrulandı), projeksiyonlu / naif / filtreli / cezalı çözücüler, analitik kutu ve pillbox spektrumları | — |
| `exp_box.py` | E1: kutu/küp yakınsama + çekirdek yöntemleri karşılaştırması | ~6 dk |
| `exp_pillbox3d.py` | E2 + E6: pillbox N0, 3D skaler P1/P2 | ~1 dk |
| `exp_timing.py <h> [pillbox\|tesla]` | E2 zamanlama/bellek; 3D TESLA hücresi (profil döndürme) | 5 s – 2 dk |
| `exp_ngsolve.py <maxh> <p> [<curve>] [full\|nograds]` | E2': NGSolve yüksek derece (ayrı venv: `pip install ngsolve`) | 1 s – 15 dk |
| `exp_axisym.py` | E3: $m=0$ (P1/P2) ve $m=1$ (N1×P1) pillbox; TESLA π/0 modu, $k_{cc}$ | ~30 s |
| `exp_ritz_ml.py` | E4: gradyan kirlenmesi, Schur-projekteli Ritz, düğüm→kenar tabanı | ~2 dk |
| `exp_gate.py` | E5: sınır koşulu varyantları (kapı / sıfır-sınır / teğetsel kapı / $\mathbf H$) | ~3 dk |
| `exp_nodal.py` | E8: nodal P1 vektör elemanlarında sahte modlar | ~1 dk |

Çalıştırma: `cd scripts/research_3d && python exp_axisym.py`. PARDISO isteğe bağlıdır (`pip install pypardiso`); yoksa SuperLU kullanılır ve 3D'de yavaştır. NGSolve betiği `ngsolve` paketini gerektirir.

---

## 7. Referanslar

- Yaker, J., Markovic, J., Reineri, A., Kurkcuoglu, D. M., Zorzetti, S. (2026). *Neural-Network Inverse Design of SRF Cavities and Transmons for Bosonic Quantum Computation.* [arXiv:2607.02289](https://arxiv.org/abs/2607.02289)
- Kranjčević, M., Adelmann, A., Arbenz, P., Citterio, A., Stingelin, L. (2019). *Multi-objective shape optimization of radio frequency cavities using an evolutionary algorithm.* NIM A. [ScienceDirect](https://sciencedirect.com/science/article/pii/S0168900218318801), [arXiv:1810.02990](https://arxiv.org/pdf/1810.02990)
- Kranjčević, M., Zadeh, S. G., Adelmann, A., Arbenz, P., van Rienen, U. (2019). *Constrained multiobjective shape optimization of superconducting rf cavities considering robustness against geometric perturbations.* PRAB 22, 122001. [APS](https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.22.122001), [arXiv:1905.13693](https://arxiv.org/pdf/1905.13693)
- Wang, Y., Tang, Y., Wu, C.-F., Feng, G. (2026). *Multiobjective Bayesian optimization for the shape design of rf cavity in particle accelerators.* PRAB 29, 034601. [APS](https://journals.aps.org/prab/abstract/10.1103/mrr8-z48f)
- Ziegler, A., Hahn, R., Isensee, V., Nguyen, A. D., Schöps, S. (2024). *Gradient-based eigenvalue optimization for electromagnetic cavities with built-in mode matching.* IET Sci. Meas. Technol. [Wiley](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/smt2.12190), [arXiv:2310.15751](https://arxiv.org/abs/2310.15751)
- Jiang, J., Wang, Y., Wang, Y., Xie, H. (2024/2025). *FieldTNN-based machine learning method for Maxwell eigenvalue problems.* J. Comput. Phys. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0021999125008873), [arXiv (html)](https://arxiv.org/html/2411.15828v1)
- Yang, Du, Liu (2026). *Learning Laplacian Eigenspace with Mass-Aware Neural Operators on Point Clouds (NEO).* SIGGRAPH. [arXiv:2605.24390](https://arxiv.org/abs/2605.24390), [GitHub](https://github.com/Adversarr/NEO)
- Chang, Y., et al. (2025). *Shape Space Spectra.* ACM TOG. [arXiv:2408.10099](https://arxiv.org/abs/2408.10099)
- Li, Z., Kovachki, N., Choy, C., Li, B., Kossaifi, J., Otta, S., Nabian, M. A., Stadler, M., Hundt, C., Azizzadenesheli, K., Anandkumar, A. (2023). *Geometry-Informed Neural Operator for Large-Scale 3D PDEs.* NeurIPS. [Proceedings](https://proceedings.neurips.cc//paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)
- Wu, H., Luo, H., Wang, H., Wang, J., Long, M. (2024). *Transolver: A Fast Transformer Solver for PDEs on General Geometries.* ICML. [PMLR](https://proceedings.mlr.press/v235/wu24r.html)
- Luo, H., et al. (2025). *Transolver++: An Accurate Neural Solver for PDEs on Million-Scale Geometries.* ICML. [PMLR](https://proceedings.mlr.press/v267/luo25o.html), [arXiv:2502.02414](https://arxiv.org/abs/2502.02414)
- Wen, S., Kumbhat, A., Lingsch, L., Mousavi, S., Zhao, Y., Chandrashekar, P., Mishra, S. (2025). *Geometry Aware Operator Transformer as an Efficient and Accurate Neural Surrogate for PDEs on Arbitrary Domains (GAOT).* NeurIPS. [arXiv:2505.18781](https://arxiv.org/abs/2505.18781), [GitHub](https://github.com/camlab-ethz/GAOT)
- Trask, N., Huang, A., Hu, X. (2022). *Enforcing exact physics in scientific machine learning: a data-driven exterior calculus on graphs.* J. Comput. Phys. 456. [arXiv:2012.11799](https://arxiv.org/abs/2012.11799)
- Actor, J., et al. (2023/24). *Data-driven Whitney forms for structure-preserving control volume analysis.* J. Comput. Phys. [OSTI](https://www.osti.gov/pages/biblio/2311272-data-driven-whitney-forms-structure-preserving-control-volume-analysis) (yazar listesi bu çalışmada doğrulanmadı)
- Kinch, B., Shaffer, B., Armstrong, E., Meehan, M., Hewson, J., Trask, N. (2025). *Structure-Preserving Digital Twins via Conditional Neural Whitney Forms.* [arXiv:2508.06981](https://arxiv.org/pdf/2508.06981)
- Shaffer, B. D., Koohy, S., Kinch, B., Hsieh, M. A., Trask, N. (2026). *Structure-Preserving Learning Improves Geometry Generalization in Neural PDEs.* [arXiv:2602.02788](https://arxiv.org/abs/2602.02788)
- Shaffer, B. D., Kinch, B., Hsieh, M. A., Trask, N. (2026). *A meshfree exterior calculus for generalizable and data-efficient learning of physics from point clouds.* [arXiv:2605.08436](https://arxiv.org/abs/2605.08436)
- Richter-Powell, J., Lipman, Y., Chen, R. T. Q. (2022). *Neural Conservation Laws: A Divergence-Free Perspective.* NeurIPS. [arXiv:2210.01741](https://arxiv.org/abs/2210.01741)
- Geiger, M., Smidt, T. (2022). *e3nn: Euclidean Neural Networks.* [arXiv:2207.09453](https://arxiv.org/abs/2207.09453)
- Christensen, T., et al. (2020). *Predictive and generative machine learning models for photonic crystals.* Nanophotonics. [Wiley](https://onlinelibrary.wiley.com/doi/10.1515/nanoph-2020-0197)
- *Predicting band structures for 2D Photonic Crystals via Deep Learning* (2024). [arXiv:2411.06063](https://arxiv.org/abs/2411.06063)
- *Symmetry-aware framework for bidirectional modeling of 3D photonic crystals and band structures* (2026). Materials & Design. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0264127526015388) (yazarlar doğrulanamadı)
- *Meshless optical mode solving using scalable deep deconvolutional neural network* (2023). Sci. Rep. [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9852487/)
- *A Physics-Informed Neural Network-Based Waveguide Eigenanalysis* (2024). IEEE. [IEEE Xplore](https://ieeexplore.ieee.org/document/10659891/)
- *High precision, full-vector optical mode solving in waveguides via fourth-order derivative physics-informed neural networks* (2025). [PubMed](https://pubmed.ncbi.nlm.nih.gov/40984242/)
- Arbenz, P., Geus, R. (1999). *A comparison of solvers for large eigenvalue problems occurring in the design of resonant cavities.* Numer. Linear Algebra Appl. 6, 3–16. [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1099-1506(199901/02)6:1%3C3::AID-NLA142%3E3.0.CO;2-I)
- Boffi, D. (2010). *Finite element approximation of eigenvalue problems.* Acta Numerica 19, 1–120. [Cambridge](https://www.cambridge.org/core/journals/acta-numerica/article/abs/finite-element-approximation-of-eigenvalue-problems/4BD87CC520C7E11CF402981AA58D77E2)
- Costabel, M., Dauge, M. (2002). *Weighted regularization of Maxwell equations in polyhedral domains.* Numer. Math. 93, 239–277. [Springer](https://link.springer.com/article/10.1007/s002110100388)
- Nédélec, J.-C. (1980). *Mixed finite elements in ℝ³.* Numer. Math. 35, 315–341. [Springer](https://link.springer.com/article/10.1007/BF01396415)
- Bossavit, A. (1988). *Whitney forms: a class of finite elements for three-dimensional computations in electromagnetism.* IEE Proc. A. [ResearchGate](https://www.researchgate.net/publication/3359692_Whitney_forms_A_class_of_finite_elements_for_three-dimensional_computations_in_electromagnetism)
- Arnold, D. N., Falk, R. S., Winther, R. (2006). *Finite element exterior calculus, homological techniques, and applications.* Acta Numerica 15, 1–155. (bağlantı bu çalışmada doğrulanmadı)
- Aune, B., et al. (2000). *Superconducting TESLA cavities.* Phys. Rev. ST Accel. Beams 3, 092001. [APS](http://link.aps.org/doi/10.1103/PhysRevSTAB.3.092001)
- Halbach, K., Holsinger, R. F. (1976). *SUPERFISH — a computer program for evaluation of RF cavities with cylindrical symmetry.* Part. Accel. 7. [OSTI/ETDEWEB](https://www.osti.gov/etdeweb/biblio/7118635)
- SLAC ACE3P / Omega3P. [Confluence](https://confluence.slac.stanford.edu/spaces/AdvComp/pages/59146985/Omega3P), [ICAP2015](https://proceedings.jacow.org/ICAP2015/papers/fraji3.pdf)
- AWS Palace. [GitHub](https://github.com/awslabs/palace)
- FEniCSx: *Maxwell eigenvalue problem in 3D using Nédélec elements* (forum). [FEniCS Discourse](https://fenicsproject.discourse.group/t/maxwell-eigenvalue-problem-in-3d-using-nedelec-elements/14741)

---

## 🔗 Bağlantılar

- Veri üretimi (2D): [[01_DATA_GENERATION]] · Fizik: [[09_PHYSICS_BACKGROUND]]
- Özuzay/Ritz tasarımı: [[14_MATHEMATICAL_IMPROVEMENTS]], [[15_SCIML_LITERATURE_REVIEW]]
- Torsiyon/landscape önseli: [[16_SPECTRAL_GEOMETRY_ANALYSIS]] · Etiket doğruluğu/hibrit: [[17_NUMERICAL_ANALYSIS_REVIEW]]

#3d #maxwell #nedelec #whitney #hcurl #eksenel-simetri #srf #tesla #hodge-projeksiyonu #yol-haritası
