# 17 — Sayısal Analiz İncelemesi (Numerical Analysis Review)

> **Kapsam:** Araştırma notu. Kaynak kodda değişiklik yok, eğitim koşusu yok.
> **Soru:** Etiketler ($\lambda_k$, $u_k$) ne kadar doğru? SpectralNO'nun Rayleigh–Ritz yapısı ne zaman gerçek bir üst sınır verir? Sertifikalı $[\lambda_{low},\lambda_{up}]$ aralıkları ucuza üretilebilir mi? Ağ + çözücü hibriti ne kazandırır? `docs/14_MATHEMATICAL_IMPROVEMENTS.md`'deki sayısal iddialar doğru mu?
> **Kanıt:** Bu not için yazılmış küçük CPU deneyleri (scikit-fem 12.0.2, gmsh 4.15.2, SciPy 1.17.1, PyTorch 2.14). Scriptler repo dışında: `scratchpad/math_numerics/` (§10). Örnekler repo'nun kendi üreticisiyle üretildi (`generate_sample_data`, varsayılan argümanlar, `--seed 7`). Süreler 4 çekirdekli bir makinede **tek süreç, tek BLAS thread** ile ölçüldü.
> **Referans commit:** `fd2d27c`.

---

## 🧭 Özet — Öncelik Sırasına Göre Öneriler

| # | Öneri | Beklenen etki (ölçülen) | Efor | Etkilenen dosyalar |
|---|---|---|---|---|
| **1** | **SpectralNO'da uyumlu (conforming) FE assembly:** ağın düğüm değerlerini $\Psi\in\mathbb R^{N\times M}$ P1 interpolantı olarak yorumla, $L=\Psi^\top K_{P1}\Psi$, $M=\Psi^\top M_{P1}\Psi$ (consistent mass). Autograd ile $\nabla\psi$ hesaplamayı kaldır. SpectralNO için rastgele `max_nodes=1024` alt-örneklemeyi kapat. | Çok yüksek. 72 (konfigürasyon, mod) çiftinde üst-sınır ihlali: mevcut kod **%17** (en kötü −%2.5), P1-interp **%0**. Alt-örnekleme gürültüsü (medyan **%4.8**) tamamen kalkar. CPU'da ileri+geri geçiş **4.5 s → 84 ms (54×)**. Bedeli: P1 tabanı, mükemmel bazda bile $+0.20\ldots+0.73\%$. P2 interpolasyonu (ağ kenar ortalarında da değerlendirilir, ×3.85 nokta) bu tabanı etiket seviyesine indirir. | Orta (1–2 gün) | `src/models/spectral_no.py`, `src/data/dataset.py` (batch'e `elements`), `configs/spectral_no.yaml` (`max_nodes: null`) |
| **2** | **Veri setine sertifikalı özdeğer aralıkları ekle:** $\lambda_{up}$ = mevcut P2 değeri (Courant–Fischer), $\lambda_{low}$ = **Lehmann–Goerisch** (P2 deneme fonksiyonları + RT akısı; $\rho$ = Crouzeix–Raviart alt sınırı). $n_{eig}\ge K+2$ mod çöz. | Yüksek. Mevcut mesh'te **+~0.3 s/örnek**. $\lambda_{1..3}$ için aralık genişliği medyan **≈2–6·10⁻⁴** (aralık 7·10⁻⁶–1.3·10⁻³). 8/8 şekilde 24/24 sınır geçerli. Yalnız CR sınırı mevcut mesh'te %0.4–2 genişlik verir, bu yüzden tek başına zayıf. | Orta | `src/data_gen/dataset_generator.py` (yeni `lam_low`/`lam_up` alanları), `src/data/dataset_converter.py`, `validate_data.py` |
| **3** | **Hibrit çıkarım `infer.py --refine {0,1,2}`:** ağ alanları → P2 lift → aynı mesh'in P2 matrisleriyle **LOBPCG, ön-koşullayıcı = K'nın tam LU'su**, 1–2 iterasyon. | Çok yüksek (doğruluk). %5 "pürüzsüz" (düşük-mod) alan hatası: 1 iterasyon **≈4·10⁻⁵**, 2 iterasyon **≈2·10⁻⁶** (etiket hatasının altında). Uçtan uca 48–86 ms/şekil (assembly + LU dahil). Aynı mesh'te tam `eigsh`'tan yalnız **1.4–1.7×** hızlı. Asıl kazanç doğruluk ve sertifika. | Orta | `infer.py`, yeni `src/solvers/refine.py` |
| **4** | **Etiket kalitesi:** sharp şekillerde re-entrant köşe grading'i + P3, 2× kaba mesh (`gen2x+corner_P3`). Smooth şekillerde grading yerine geometriyi düzelt (spline / eğrilik sınırı; docs/14 #7). | Orta. Sharp: max-mod hata medyanı **4.7·10⁻⁵ → 5.3·10⁻⁶ (9×)**, üstelik **%26 daha hızlı** (116 → 86 ms). Smooth şekillerin 100-gon köşeleri etiket hatasına hâkim: max iç açı ile $\log$ hata korelasyonu **0.92**. | Düşük | `src/data_gen/dataset_generator.py` |
| **5** | **Modelleri sertifikalı aralığa göre değerlendir/eğit:** aralık-normalize hata $e_k=(\hat\lambda_k-\lambda_{up})/(\lambda_{up}-\lambda_{low})$. Aralık dışı için hinge kaybı. #1 yapılınca $\hat\lambda\ge\lambda\ge\lambda_{low}$ otomatik olur ve $\log(\hat\lambda/\lambda_{up})$ bir enerji-normu hatasıdır. | Orta–yüksek (ölçümün güvenilirliği) | Düşük | `src/training/lightning_module.py`, `validate_data.py` |
| **6** | **MPS referans/validasyon seti (sharp poligonlar):** Betcke–Trefethen particular solutions. | Orta. Şekil başına 3 mod, **~0.3–1 s**, **~10⁻⁸** doğruluk (P4-graded FEM referansından 30× daha doğru, 10× daha hızlı). Etiketlerin ve sınırların bağımsız kontrolü. | Düşük–orta | yeni `scripts/mps_reference.py`, `validate_data.py` |
| 7 | **Özdeğer ≥ 5–6 mod sakla.** LG için $\rho\le\lambda_{K+1}$ gerekir. $(\lambda_4-\lambda_3)/\lambda_3<\%5$ şekillerin %3'ünde. | Orta (#2'nin ön koşulu) | Düşük | `configs/*.yaml` (`n_eigen_modes`) |
| 8 | Veri üretimini hızlandırma (LOBPCG, iki-ızgara, FEAST, RB): **öncelik düşük.** Örnek başına 150 ms: %39 gmsh, %19 assembly, %42 `eigsh`. Çözücü tarafında en fazla ~%25–40 kazanç var; 5000 örnek 4 çekirdekte ≈3.4 dk sürüyor. | Düşük | Yüksek | — |

**Tek cümlelik ana mesaj:** Etiketler zaten $10^{-4}$ düzeyinde doğru ve her zaman üst sınır. Sistemin zayıf halkası SpectralNO'nun **uyumsuz, düğüm-quadratürlü** assembly'si. Bu assembly'yi mesh'in kendi FE matrisleriyle değiştirmek hem doğru hem ucuz; aynı altyapı **sertifikalı aralıklar** ve **1–2 LOBPCG adımlı hibrit çıkarım** için de kullanılabilir.

---

## 1. docs/14 Sayısal İddialarının Doğrulaması

| # | docs/14 iddiası | docs/14 değeri | Bu çalışmada ölçülen | Yöntem (§) | Hüküm |
|---|---|---|---|---|---|
| 1 | Rastgele 1024-düğüm alt-örnekleme Ritz özdeğerinde gürültü | göreli std **%4–5** | medyan **%4.8** (sharp %3.6, smooth %5.6; aralık %2.4–7.2). İdeal baz (16 FEM özfonksiyonu): %1.1–7.7 | 4 şekil × 6 baz × 50 çekiliş (§6.2) | ✅ doğrulandı |
| 1b | … ve aşağı yönlü bias | **−%3…−7** | ortalama **−%0.25 (M=10), −%0.61 (M=21), −%1.28 (M=36)**. %74'ü negatif. Uç değerler −%5.0…+%3.5 | aynı | ⚠️ Yön doğru, büyüklük abartılı ve baza bağlı. Bias M ile büyüyor |
| 2 | Düğüm (vertex) quadratürü RR üst sınırını bozar | disk, $\hat\lambda_2=14.657<14.682$ (−%0.17) | Uyumlu ψ + düğüm quadratürü: vakaların **%3**'ünde ihlal (en kötü −%0.61). **Repo'daki gerçek kurulum** (en yakın *sınır düğümüne* uzaklıkla gate, $\nabla d=-\text{dir}$, düğüm quadratürü): **%17** ihlal, en kötü **−%2.47**. Tam quadratür / P1 / P2 interp: **%0** | §6.1 | ✅ doğrulandı. Asıl sorun daha büyük: gate uyumsuz, çünkü ψ sınır *kenarlarında* sıfır değil |
| 3 | %5 hatalı ağ tahmininden 1 blok ters iterasyon + RR | **3·10⁻⁵**, **~5 ms** (P1 matrisleri) | Hata spektruma bağlı. **Kaba/beyaz gürültü** %5: 2.0–3.0·10⁻⁵ ✅. **Pürüzsüz (düşük-mod, spektral-bias tipi)** %5: **4.6–5.4·10⁻⁴** (15× kötü). Adımın kendisi 2.0–3.4 ms, ama assembly (21–49 ms) + LU (16–26 ms) ile uçtan uca **42–69 ms** | §5, P2 matrisleri, 4 şekil × 5 deneme | ⚠️ kısmen. Süre iddiası assembly ve faktorizasyonu dışarıda bırakıyor. Pürüzsüz hatada LOBPCG(LU) 1 iterasyon gerekir (≈4·10⁻⁵) |
| 4 | Kaba P2 (H = 4 mm) | **<10⁻³ hata, ~20 ms** | medyan (max mod 1–3) **3.0·10⁻⁴ (sharp) / 8.0·10⁻⁴ (smooth)** ✅. Ama şekillerin **%11 / %32**'sinde bir mod >10⁻³ (max 1.6·10⁻³). Süre: gmsh hariç 16–22 ms, dahil **31 ms (sharp) / 60 ms (smooth)** | 40 şekil, P4 referans (§2) | ⚠️ medyan için doğru, "<10⁻³" her şekil için doğru değil. P3 @4 mm: max 5.1·10⁻⁴, 49/90 ms |
| 5 | Re-entrant poligonda P2 yakınsama oranı | **≈1.42** (α=257.8°, teori 1.40) | Aynı şekil (s=17): $\lambda_1$ **1.42/1.46/1.42**, $\lambda_3$ 1.46/1.46/1.42. **Ama $\lambda_2$ 1.65/1.64**: 217.8°'lik ikinci köşe belirliyor, teori 1.65. L-şekli $\lambda_1$: 1.32–1.34 (teori 4/3). P3 de aynı oranda (1.33) | §3 | ✅ doğrulandı. Oran **moda bağlı**; yüksek derece grading olmadan işe yaramıyor |
| 6 | L-şekli, 6 köşe tekil fonksiyonu Ritz hatasını düşürür | **1.4·10⁻² → 2·10⁻⁵** | pürüzsüz ω·polinom, M=15: **1.42·10⁻²** ✅. +6 tekil fonksiyon (M=21): **3.5·10⁻⁶**. M=12 (derece 2 + 6): 7.2·10⁻⁵. Tek bir tekil fonksiyon bile (M=16): 8.4·10⁻⁵ | kesin graded kompozit Gauss quadratürü (§6.4) | ✅ doğrulandı (büyüklük mertebesi) |
| 7 | (ek) Etiket λ doğruluğu | medyan 5·10⁻⁵ (sharp), 1.3·10⁻⁴ (smooth) | **4.7·10⁻⁵ / 1.5·10⁻⁴** (max mod 1–3), max 4.2·10⁻⁴. Tümü pozitif (üst sınır) | §2 | ✅ |
| 8 | (ek) İki-ızgara zaman kazandırmaz | — | Doğru. İnce mesh'te bir LU (16–26 ms) + birkaç çözüm gerekiyor, `eigsh` (51–79 ms) da aynı LU'yu kullanıyor. Kazanç ≤~2× | §5, §8 | ✅ |
| 9 | (ek) Örnek başına üretim süresi | 0.43 s | **150 ms** medyan (k=3, tek thread). Bu çalışmada k=6 ile de ~0.25–0.37 s | §8 | ⚠️ ölçüm koşuluna bağlı. Önemsiz |

**Yeniden üretilemeyen/düzeltilen iddialar:**
- (1b) −%3…−7 bias üretilemedi. Poligon-gate bazlarında bias ortalaması −%0.3…−%1.3.
- (3) "3·10⁻⁵ / 5 ms" yalnız kaba (yüksek frekanslı) hata için ve yalnız çözüm adımı sayılırsa geçerli. Ağların tipik hatası düşük-frekanslı olduğundan (spektral bias; Zhang vd. 2024) gerçekçi değer 1 ters iterasyonda ~5·10⁻⁴.
- (4) "<10⁻³" medyan için doğru. Smooth ailede şekillerin üçte biri bu sınırı aşıyor.

---

## 2. Etiket Doğruluğu Çalışması

### 2.1 Yöntem

- **Örnekler:** `--seed 7`, s = 0…39 (18 sharp, 22 smooth), üretici kod yolu aynen (`generate_sample_data`, 6 mod).
- **Referans:** aynı poligon, P4, mesh $h_{min}=0.6$ mm, $h_{max}=2.5$ mm (üreticinin yarısı). Her re-entrant köşede ($\alpha>1.02\pi$) cebirsel grading uygulandı: $h(r)=\max(2\cdot10^{-5},\,h_{min}(r/4\text{ mm})^{0.4})$.
- **Referans doğruluğu iki bağımsız yolla kontrol edildi:**
  1. Rafinman dizisi: P4 f=0.5 → f=0.25 farkı ≤ 3·10⁻⁷ (sharp s=17) ve ≤ 1.1·10⁻⁶ (smooth s=44).
  2. MPS (§7): sharp poligonlarda MPS ile P4 referans arasındaki fark 1–3·10⁻⁷. Bu, P4'ün kendi üst-sınır hatası. Referans hatası etiket hatasından ≥100× küçük.
- **Alan hataları:** üretici P2 alanı referans mesh'in quadratür noktalarında değerlendirildi (hızlı trifinder + barisentrik P2 değerlendirme).
  - $L^2$ açısı: $\sin\angle_{L^2}(u_h,u)$.
  - Sup hatası.
  - Eğitim hedefi (converter'ın vertex değerleri / max|·|) üzerindeki max ve rms hata.

### 2.2 Özdeğer hataları (göreli, $|\lambda_h/\lambda-1|$; medyan / p90 / max)

| Mod | sharp (n=18) | smooth (n=22) |
|---|---|---|
| $\lambda_1$ | 3.6e-5 / 1.4e-4 / 1.8e-4 | **1.5e-4** / 3.4e-4 / **4.2e-4** |
| $\lambda_2$ | 1.7e-5 / 5.4e-5 / 6.9e-5 | 7.4e-5 / 1.5e-4 / 1.9e-4 |
| $\lambda_3$ | 4.0e-5 / 1.3e-4 / 2.2e-4 | 1.0e-4 / 1.5e-4 / 2.7e-4 |
| $\lambda_4$ | 2.9e-5 / 5.3e-5 / 7.0e-5 | 5.8e-5 / 1.9e-4 / 2.3e-4 |
| $\lambda_5$ | 4.0e-5 / 1.0e-4 / 1.5e-4 | 7.6e-5 / 1.7e-4 / 3.8e-4 |
| $\lambda_6$ | 6.3e-5 / 9.8e-5 / 1.9e-4 | 7.1e-5 / 1.2e-4 / 1.4e-4 |

- **240 etiketin 240'ı pozitif hatalı**, yani uyumlu P2 Ritz değerleri gerçek üst sınır ($\lambda_h\ge\lambda$). Bu, §4'teki sertifikalı aralığın üst ucu olarak doğrudan kullanılabilir.
- **"Smooth" aile daha az doğru.** 100-gon örneklemesi çukurlarda re-entrant köşeler üretiyor: smooth ailede max iç açı medyanı 238°, %100'ünde re-entrant köşe var. Sharp ailede bu değerler 223° ve %89. Hata max iç açıyla belirleniyor ($\mathrm{corr}(\log e_{\lambda_1},\alpha_{max})=0.92$):

| $\alpha_{max}$ | n | $\lambda_1$ hata medyanı | max-mod hata medyanı |
|---|---|---|---|
| <180° | 2 | 2.6e-6 | 2.1e-5 |
| 180–220° | 7 | 8.2e-6 | 2.1e-5 |
| 220–240° | 17 | 7.1e-5 | 9.0e-5 |
| 240–270° | 14 | 1.6e-4 | 1.8e-4 |

### 2.3 Alan (eigenfunction) hataları (mod 1 / 2 / 3; medyan, parantezde max)

| Ölçü | sharp | smooth |
|---|---|---|
| $\sin\angle_{L^2}$ | 9.2e-5 (2.9e-4) / 1.7e-4 (1.1e-3) / 3.2e-4 (1.1e-3) | 1.9e-4 (4.8e-4) / 2.0e-4 (9.8e-4) / 3.6e-4 (1.7e-3) |
| **Eğitim hedefi**, vertex max hata (peak'e göre) | 2.3e-4 (1.1e-3) / 1.5e-4 (1.0e-3) / 3.5e-4 (1.6e-3) | 5.9e-4 (1.3e-3) / 4.3e-4 (1.1e-3) / 8.1e-4 (1.8e-3) |
| Eğitim hedefi, vertex rms hata | 2.7e-5 / 1.8e-5 / 3.8e-5 | 7.4e-5 / 5.8e-5 / 1.1e-4 |
| Sup hata (tüm noktalar) | 4.2e-3 / 3.1e-3 / 5.1e-3 | 5.8e-3 / 6.4e-3 / 7.7e-3 |

**Yorum:**
- Vertex değerleri süper-yakınsak. Eğitim hedeflerinin hatası ≤ 1.8·10⁻³ (peak'e göre), rms ≤ 10⁻⁴. **Etiket kalitesi bugünkü ağ hatalarının (~10⁻²) 1–2 mertebe altında.** Eğitim için darboğaz değil.
- Sup hatası (%0.3–0.8) eleman *içinde*, re-entrant köşelerin yanında. Buradaki gradyan tekilliği $\nabla u\sim r^{\pi/\alpha-1}$ (Grisvard).
- Enerji normu: basit özdeğerde $\lambda_h-\lambda=\|u-u_h\|_a^2-\lambda\|u-u_h\|^2$ olduğundan $|u-u_h|_1/|u|_1\approx\sqrt{e_\lambda}\approx$ **%0.6–1.2**. Etiket *gradyanları* (Sobolev kaybı, köşe akısı) yalnız ~%1 doğru. Gradyan kullanan bir kayıp eklenecekse bu önemli.

### 2.4 Daha iyi/ucuz etiket seçenekleri (40 şekil doğruluk; 16 şekil temiz zamanlama; mod 1–3 max hata)

| Seçenek | sharp: medyan / max hata | sharp süre | smooth: medyan / max | smooth süre |
|---|---|---|---|---|
| **Mevcut üretici (P2)** | 4.7e-5 / 2.2e-4 | 116 ms | 1.5e-4 / 4.2e-4 | 206 ms |
| aynı mesh, P3 | 1.2e-5 / 6.8e-5 | 252 ms | 4.7e-5 / 1.4e-4 | 406 ms |
| aynı mesh + köşe grading, P2 | 1.0e-5 / 2.4e-5 | 158 ms | 1.7e-5 / 4.5e-5 | 489 ms |
| **2× kaba + köşe grading, P3** | **5.3e-6 / 3.2e-5** | **86 ms** | 1.9e-5 / 5.1e-5 | 329 ms |
| 2× kaba, P2 | 3.1e-4 / 8.1e-4 | 37 ms | 4.9e-4 / 1.1e-3 | 85 ms |
| uniform 4 mm, P2 | 3.0e-4 / 1.3e-3 | 31 ms | 8.0e-4 / 1.6e-3 | 60 ms |
| uniform 4 mm, P3 | 8.2e-5 / 4.2e-4 | 49 ms | 2.1e-4 / 5.1e-4 | 90 ms |
| uniform 4 mm + köşe, P3 | 1.1e-5 / 7.7e-5 | 64 ms | 4.9e-5 / 1.8e-4 | 202 ms |

**Öneri 4'ün gerekçesi:**
- Sharp ailede `2×kaba + köşe grading + P3` hem 9× daha doğru hem daha hızlı.
- Smooth ailede grading pahalı: gmsh'in `Distance(PointsList)` alanı onlarca köşeyle 3×'e kadar yavaşlıyor. Asıl çözüm, "smooth" geometrinin gerçekten pürüzsüz olması (spline ya da eğrilik sınırı). Bu hem etiketleri hem aile tanımını düzeltir.
- skfem P3'te de vertex DOF'ları ilk $N$ sırada (kontrol edildi), bu yüzden converter dilimlemesi değişmeden çalışır.

### 2.5 Boşluk istatistiği (40 şekil; göreli $(\lambda_{k+1}-\lambda_k)/\lambda_k$)

| Boşluk | medyan | <%1 | <%5 | <%10 |
|---|---|---|---|---|
| 1→2 | 1.07 | 0 | 0 | 0 |
| 2→3 | 0.30 | 0 | 0.05 | 0.07 |
| 3→4 | 0.33 | 0 | 0.03 | 0.07 |
| 4→5 | 0.19 | 0.03 | 0.07 | 0.20 |
| 5→6 | 0.17 | 0 | 0.07 | 0.28 |

docs/14 §1.1 ile uyumlu. $\lambda_4$'ün saklanması hem boşluk-ağırlıklı kayıplar hem LG sınırı için gerekli.

---

## 3. Yakınsama Çalışması (re-entrant köşeler)

Teori: iç açısı $\alpha>\pi$ olan köşede $u\sim r^{\pi/\alpha}\sin(\pi\theta/\alpha)$. Uniform mesh'te
$$\lambda_h-\lambda \simeq C\,h^{2\min(p,\;\pi/\alpha)}$$
olur (Babuška–Osborn 1991; Grisvard 1985). Graded mesh'te optimal oran $h^{2p}$ geri gelir, yani $N^{-p}$ (Babuška, Kellogg & Pitkäranta 1979). hp-FEM'de üstel yakınsama elde edilir (Schwab 1998).

**L-şekli** (kesin $\lambda_{1,2,3}=9.6397238440,\,15.1972519265,\,2\pi^2$; Trefethen & Betcke 2006). Tabloda uniform $h$ ile gözlenen oran var:

| Eleman | $\lambda_1$ | $\lambda_2$ | $\lambda_3$ (pürüzsüz) | $\lambda_5$ |
|---|---|---|---|---|
| P1 | 1.65 → 1.45 (→4/3) | 1.9–2.0 | 2.0 | 1.9 → 1.67 |
| P2 | **1.34 → 1.32** | 3.4 → 2.8 (→8/3) | **3.9–4.05** | 1.67 → 1.32 |
| P3 | **1.33** (P2 ile aynı!) | 2.6–2.7 | 5.8–6.4 | 1.32 |
| P2 + köşeye graded mesh | **3.7–3.9 (≈$N^{-2}$)** | 3.9–4.0 | 3.9–4.0 | 3.9–4.0 |

**Veri setinden sharp s=17** (docs/14'ün şekli; köşeler 257.8°, 217.8°, 196.2° → $2\pi/\alpha$ = 1.40, 1.65, 1.83):
- Uniform P2 oranları: $\lambda_1$ 1.42 / 1.46 / 1.42; $\lambda_2$ 2.0 / 1.65 / 1.64; $\lambda_3$ 1.46 / 1.46 / 1.42.
- Üretici mesh ailesi (hmin, hmax birlikte ölçeklenerek): $\lambda_1$ oranı $N^{-0.71\ldots-0.84}$.

**Çıkarımlar:**
- docs/14'ün 1.42'si doğru, ama oran **moda bağlı**. Her mod, özfonksiyonunda katsayısı sıfır olmayan en tekil köşeyle belirleniyor.
- **Derece artırmak tek başına işe yaramıyor** (L-şeklinde P3 = P2 oranı). Etkili araç köşe grading'i.
- Rafinman N ile yavaş: 10× DOF ile ~5× doğruluk.
- Adaptif FEM (Dai, Xu & Zhou 2008) ve residual tahmin edici (Durán, Padra & Rodríguez 2003) bu grading'i otomatik bulur:
  $$\eta^2=\sum_T h_T^2\|\lambda_hu_h+\Delta u_h\|_T^2+\sum_e h_e\|[\partial_nu_h]\|_e^2$$
  Bu veri setinde köşe listesi zaten bilindiğinden **a priori grading** daha basit ve yeterli.

---

## 4. Sertifikalı Özdeğer Sınırları

### 4.1 Araçlar

- **Üst sınır (bedava):** $V_h\subset H^1_0$ uyumlu olduğundan Courant–Fischer $\lambda_{k,h}\ge\lambda_k$ verir. Mevcut P2 etiketleri (§2.2'de %100 pozitif). Kayan nokta ve `eigsh` toleransı dışında kesin.
- **Crouzeix–Raviart (CR) alt sınırı** (Carstensen & Gedicke 2014; Liu 2015). $H$ = maksimum eleman çapı, $\kappa=0.1893$:
  $$\lambda_k\;\ge\;\frac{\lambda_{k,CR}}{1+\kappa^2H^2\lambda_{k,CR}}$$
  Genişlik $\approx\kappa^2H^2\lambda$ ile sınırlı. Graded mesh'in ince bölgesi bu terime yardım etmez, çünkü $H$ global.
- **Lehmann–Goerisch (LG)** (Behnke & Goerisch 1994; Liu 2015; Vejchodský 2018).
  - Girdiler:
    - Deneme fonksiyonları $\tilde u_1..\tilde u_n\in H^1_0$ (burada: üreticinin P2 özvektörleri).
    - Kaydırma $\gamma>0$ ile $X=L^2(\Omega)^2\times L^2(\Omega)$, $Tv=(\nabla v,\sqrt\gamma v)$.
    - Her $q_i\in H(\mathrm{div})$ için kabul edilebilir $w_i=\big(q_i,(\tilde u_i+\mathrm{div}\,q_i)/\sqrt\gamma\big)$.
  - $q_i$, $\|q\|^2+\gamma^{-1}\|\tilde u_i+\mathrm{div}\,q\|^2$'yi RT (skfem `ElementTriRT2`, div∈P1) üzerinde minimize eder. Bu bir SPD sistemidir.
  - Matrisler:
    $$A_0=a(\tilde u_i,\tilde u_j)+\gamma(\tilde u_i,\tilde u_j),\quad A_1=(\tilde u_i,\tilde u_j),\quad A_2=\langle w_i,w_j\rangle$$
  - $\rho\le\lambda_{n+1}+\gamma$ ($\lambda_{n+1}$'in CR alt sınırı $+\gamma$) ve $A=A_0-\rho A_1$, $B=A_0-2\rho A_1+\rho^2A_2\succ0$ için $Ax=\mu Bx$'in negatif özdeğerleri $\mu_1\le\mu_2\le\dots$ şu sınırı verir:
    $$\lambda_{n+1-i}\;\ge\;\rho-\frac{\rho}{1-\mu_i}-\gamma$$
  - Birkaç $\gamma$ denenir, en iyisi alınır; her biri geçerli.
- **Kato–Temple** (Kato 1949) aynı yapıdadır ama $\|\Delta\tilde u+\tilde\lambda\tilde u\|_{L^2}$ ister. $C^0$ FEM için bu hesaplanamaz ($\Delta u_h$ dağılımsal). Pratik karşılığı, akı rekonstrüksiyonlu LG. MPS'de (§7) ise PDE içeride tam sağlandığından Moler–Payne/Kato-tipi sınır doğrudan uygulanır.
- İlgili modern çerçeveler:
  - Cancès, Dusson, Maday, Stamm & Vohralík (2017, 2018, 2020): garantili ve robust sınırlar, çok katlılık/kümeler.
  - Liu & Vejchodský 2022: özvektör sınırları.
  - Carstensen & Puttkammer 2024: adaptif GLB.

### 4.2 Ölçümler (P4 referansa göre göreli; hepsi geçerli)

**CR alt sınırı** ($\lambda_1/\lambda_2/\lambda_3$ için $\lambda_{low}/\lambda-1$):

| Mesh | $H$ | DOF | süre | sharp (s=2) | sharp re-ent. (s=17) | smooth (s=13) |
|---|---|---|---|---|---|---|
| üretici mesh | 3.6–5.2 mm | 3–6k | 30–60 ms | −5.2e-3 / −1.1e-2 / −1.8e-2 | −5.1e-3 / −9.9e-3 / −1.4e-2 | −7.8e-3 / −1.0e-2 / −1.2e-2 |
| 2× uniform rafine | 0.9–1.3 mm | 54–95k | 0.6–1.5 s | −3.3e-4 / −7.1e-4 / −1.1e-3 | −4.0e-4 / −6.4e-4 / −9.6e-4 | −6.7e-4 / −7.2e-4 / −8.6e-4 |

- **Ham CR değeri alt sınır değil.** Konveks sharp şekillerde $\lambda_{1,CR}>\lambda_1$ (+4.4e-4, +6.2e-4). Düzeltme terimi şart.

**LG alt sınırı**, üreticinin kendi mesh'i ve P2 özvektörleri, $n=5$ deneme fonksiyonu, $\rho$ = CR($\lambda_6$). Aralık genişliği $(\lambda_{up}-\lambda_{low})/\lambda$:

| Şekil | $\lambda_1$ | $\lambda_2$ | $\lambda_3$ | ek süre (CR + akı) |
|---|---|---|---|---|
| s=2 sharp (konveks) | 1.1e-5 | 4.0e-5 | 1.7e-4 | 35 + 190 ms |
| s=9 sharp (konveks) | 7.2e-6 | 3.6e-5 | 9.4e-5 | 49 + 250 ms |
| s=7 sharp | 1.9e-4 | 2.7e-4 | 4.1e-4 | 43 + 235 ms |
| s=16 sharp | 3.6e-4 | 1.8e-4 | 1.2e-3 | 37 + 185 ms |
| s=17 sharp (258°) | 4.1e-4 | 1.4e-4 | 9.5e-4 | 37 + 198 ms |
| s=0 smooth | 4.0e-4 | 2.5e-4 | 6.5e-4 | 70 + 369 ms |
| s=3 smooth | 1.2e-3 | 3.3e-4 | 5.0e-4 | 65 + 332 ms |
| s=13 smooth | 1.1e-3 | 8.2e-4 | 1.3e-3 | 59 + 371 ms |
| **medyan** | **≈3.8e-4** | **≈2.2e-4** | **≈5.8e-4** | **≈0.3 s** |

- 1 uniform rafinmanla (1.5–2.9 s/şekil) genişlikler $\lambda_1$: 1.4e-6…4.1e-4, $\lambda_3$: 6.7e-6…4.3e-4.
- $n=3$ deneme fonksiyonuyla $\lambda_3$ genişliği 7e-3'e kadar çıkıyor, çünkü $\rho$, $\lambda_4$'e yakın ve kaba. Bu yüzden **$n\ge K+2$ mod çözülmeli**.
- **Kural:** LG alt-sınır hatası ≈ 1.7–4 × üst-sınır hatası. Sertifikalı genişlik ≈ 3–5 × etiket hatası.

### 4.3 Veri seti ve ağ için anlamı

1. **Evet, veri seti sertifikalı aralık taşıyabilir.** Mevcut mesh'te örnek başına +~0.3 s (üretimi ~2.5× yavaşlatır; 5000 örnek 4 çekirdekte ≈ +6 dk). Önerilen H5 alanları:
   - `lam_up` (P2, mevcut)
   - `lam_low` (LG)
   - `lam_low_cr` (yedek)
   - `bound_method`

   Kayan nokta: kesin (interval arithmetic) sertifika gerekiyorsa Liu–Oishi'nin INTLAB yolu izlenir. Burada sınırlar "yuvarlama ve çözücü toleransı mertebesinde" garantilidir. Bu toleranslar (~1e-12) genişliklerin çok altında.
2. **Değerlendirme:**
   - Aralık-normalize hata $e_k=(\hat\lambda_k-\lambda_{up})/(\lambda_{up}-\lambda_{low})$. $e_k\in[-1,0]$ ise tahmin, verinin *kanıtlanabilir* doğruluğundan ayırt edilemez.
   - Model doğruluğu ~10⁻² iken bu yalnız hibrit çıkarım (§5) için anlamlı.
   - Asıl kazanç: "etiket mi yanlış, model mi?" sorusunu kesin cevaplamak ve çıkarımda sertifika verebilmek.
3. **Eğitim:**
   - Aralık dışında hinge kaybı:
     $$\mathcal L=\sum_k\big[(\log\lambda_{low,k}-\log\hat\lambda_k)_+^2+(\log\hat\lambda_k-\log\lambda_{up,k})_+^2\big]$$
     Aralık içindeki değerler aynı cezayı alır, dar aralıklarda MSE'ye dönüşür.
   - SpectralNO uyumlu hâle getirilince (#1) $\hat\lambda_k\ge\lambda_k\ge\lambda_{low,k}$ yapısal olur. $(\hat\lambda_k-\lambda_{up,k})/\lambda_{up,k}$, etiket hatası payıyla birlikte bir **enerji-normu alan hatası**dır.
4. **Çıkarımda sertifika:** hibrit çıkarım (§5) sonucu zaten uyumlu bir Ritz değeri (üst sınır). Aynı LG adımı eklenirse (~0.3 s) *ağ + 1 LOBPCG* çıktısı sertifikalı bir aralıkla döner.

---

## 5. Hibrit Çözücü: Doğruluk / Zaman Takası

**Kurulum:**
- 4 şekil (s=2, 17 sharp; 0, 13 smooth).
- Ağ çıktısı simüle edildi: gerçek vertex alanları + göreli $L^2$ hatası $\varepsilon$ + mod karışımı (küçük rastgele dönme).
- Hata tipleri:
  - **pürüzsüz**: modlar 4–33'ün $1/j$ ağırlıklı karışımı. Sinir ağlarının spektral biasına karşılık gelen, "zor" durum.
  - **kaba**: düğüm başına beyaz gürültü.
- Vertex → P2 lift: kenar DOF'u = iki ucun ortalaması.
- Tüm matrisler üreticinin kendi mesh'i ve P2.
- Hata, **aynı mesh'in tam P2 çözümüne** göre, max mod 1–3, 4 şeklin medyanı.
- Bu mesh'lerin etiket hatası (P2 vs gerçek) 2.5e-5…3.8e-4.

| Yöntem | %5 pürüzsüz | %20 pürüzsüz | %5 kaba | %20 kaba | ek süre (mesh verilmiş) |
|---|---|---|---|---|---|
| Saf ağ, sadece RR (P2 matrisleri, çözüm yok) | 1.0e-2 | 9.3e-2 | 2.2e-1 | 3.5 | assembly 21–49 ms |
| (%1 pürüzsüz hatada bile RR) | 7.3e-3 | | | | ← P1 lift tabanı |
| + 1 blok ters iterasyon (docs/14) | 4.8e-4 | 7.5e-3 | 2.2e-5 | 3.2e-4 | + LU 16–26 ms + 2.0–3.4 ms |
| + 2 ters iterasyon | 2.0e-4 | 2.9e-3 | 1.2e-6 | 1.4e-5 | + ~4–6 ms |
| + 3 ters iterasyon | 8.9e-5 | 1.3e-3 | 2.7e-7 | 3.1e-6 | + ~5–9 ms |
| + blok-Krylov RR, span{W₁,W₂,W₃} | ≈1.6e-6 | ≈3.5e-5 | ≈5e-8 | ≈1e-6 | ≈ ters iterasyon ×3 |
| **+ LOBPCG (ön-koşul = LU), 1 iter.** | **≈3.8e-5** | ≈4.3e-4 | 7.5e-7 | 1.3e-5 | + LU + 7.3–11 ms |
| **+ LOBPCG (LU), 2 iter.** | **≈2e-6** | ≈3.7e-5 | 5e-8 | 1.3e-6 | + LU + ~10–15 ms |
| LOBPCG (ILU, 5 iter.) | 2.8e-6 | 7e-5 | 5.8e-6 | 5e-4 | +37–66 ms (ILU dahil) |
| Tam `eigsh` (σ=0), aynı mesh | 0 (tanım) | | | | 51–79 ms |

**Uçtan uca karşılaştırma** (mesh verilmiş, assembly dahil, tek thread):

| Hat | Doğruluk (gerçeğe göre) | Süre/şekil |
|---|---|---|
| Saf ağ (GPU'da toplu) | ~10⁻² (docs/14; bu repoda eğitilmiş model ölçülmedi). SpectralNO+P1 tabanı ≥2e-3 | ~ms |
| Ağ + RR (P2 matrisleri) | ≥7e-3 (vertex→P2 lift tabanı) | 21–49 ms |
| **Ağ + 1 LOBPCG(LU)** | ≈ etiket hatası (+≤5e-5 çözücü hatası) | **48–86 ms** |
| Ağ + 1 ters iterasyon | 2e-5…5e-4 (hata tipine göre) + etiket hatası | 42–69 ms |
| Tam P2 çözümü, aynı mesh (asm + eigsh) | etiket hatası | 74–119 ms |
| Kaba FEM, uniform 4 mm P2 (gmsh dahil) | medyan 3e-4 / 8e-4; %11 / %32'si >1e-3 | 31 / 60 ms |
| Kaba FEM, uniform 4 mm P3 (gmsh dahil) | medyan 8e-5 / 2e-4; max 5.1e-4 | 49 / 90 ms |
| Üretici (gmsh + P2 + eigsh) | 4.7e-5 / 1.5e-4 | 116 / 206 ms |

**Sonuçlar:**
1. docs/14'ün stratejik mesajı doğru: mesh verilmişse saf ağ, kaba FEM'i doğrulukta yenemez. Ağın değeri toplu/diferansiyellenebilir değerlendirme ve **ısıtma (warm start)**.
2. Isıtmanın CPU'daki zaman kazancı küçük (**1.4–1.7×**). Assembly ve LU, `eigsh` ile ortak. Asıl kazanç doğruluk: 1 LOBPCG adımı ~10⁻² → ~4·10⁻⁵. Çıktı ayrıca bir üst sınırdır.
3. Ağın **düşük-frekanslı** hatası (spektral bias) ters iterasyonun en yavaş söndürdüğü bileşen: $\lambda_3/\lambda_4\approx0.75$ oranıyla. Bu yüzden:
   - bir **blok genişletme** (LOBPCG'nin önceki yönü, blok-Krylov) veya
   - ağın **K+2 alan** tahmin etmesi

   birkaç kat kazandırır. Fanaskov vd. (ICLR 2026) da hedeften *büyük* bir alt uzay tahmin etmenin LOBPCG ısıtmasını iyileştirdiğini raporluyor. Rastgele ek vektörler işe yaramıyor (ölçüldü: `inv1+os3` ≈ `inv1`).
4. **Vertex-only tahmin P1 tabanı getirir.** Ağ yalnız vertex değeri ürettiği için lift edilmiş alanın Rayleigh oranı mükemmel alanlarda bile ~%0.7 yüksek. En az bir çözücü adımı bu tabanı kaldırır.
5. Literatürle uyum:
   - HINTS: ağ + relaksasyon, spektral biası tamamlayıcı kullanır (Zhang vd. 2024).
   - NOWS: nöral ısıtma ile Krylov'da %90'a kadar süre azalması (Eshaghi vd. 2025/26).
   - Öğrenilmiş multigrid/prolongasyon (Greenfeld vd. 2019; Luz vd. 2020) ve GNN ön-koşullayıcılar (Chen 2025).
   - Bu problem boyutunda (≤10⁴ DOF) seyrek LU zaten ucuz ve öğrenilmiş ön-koşullayıcının getirisi sınırlı. 3D / ≥10⁶ DOF'ta (LU'nun ölçeklenmediği yerde) öncelik kazanır.

---

## 6. SpectralNO İçinde Quadratür ve Ayrıklaştırma

### 6.1 Neden mevcut Ritz değerleri üst sınır değil?

Mevcut assembly:
$$M_{mn}=\sum_iw_i\psi_m(x_i)\psi_n(x_i),\qquad L_{mn}=\sum_iw_i\nabla\psi_m(x_i)\cdot\nabla\psi_n(x_i)$$
Burada $w_i$ lumped düğüm alanı, $\nabla\psi$ autograd ile ve $\nabla d=-\text{dir}$ lineerizasyonu ile hesaplanıyor.

Courant–Fischer garantisi için gerekenler:
- (i) $\psi_m\in H^1_0(\Omega)$
- (ii) $a,m$ formlarının **tam** hesaplanması

İki koşul da bozuk:
- **Gate uyumsuz.** $d$ = en yakın *sınır düğümüne* uzaklık. Sınır kenarı üzerinde $d>0$, yani ψ sınırda sıfır değil. Nodal quadratür bunu göremez.
- **Donuk feature'lar.** Voronoi-hücre sabitleri ψ'de sıçramalar yaratır ($\psi\notin H^1$). Nodal gradyan sıçrama enerjisini görmez.
- **Quadratür kuralı.** Vertex kuralı $O(h^2)$ ve işaretsiz (Banerjee & Osborn 1990).

**Ölçüm** (4 gerçek şekil × polinom derecesi {3, 5, 7} × `bc_scale` {0.02, 0.1} = 24 baz, her birinde $\lambda_{1..3}$; ψ = gate(d)·monom, converter'ın normalize koordinatları; referans P2 etiketi):

| Assembly | $\hat\lambda<\lambda$ oranı | en kötü | not |
|---|---|---|---|
| **Mevcut:** düğüm-uzaklık gate + nodal quadratür | **%17** | **−%2.47** | M=21–36, b=0.02'de sistematik |
| + rastgele 1024-alt-örnekleme (50 çekiliş ort.) | %26 | −%2.9 | |
| Gerçek segment-uzaklığı gate (uyumlu) + nodal quadratür | %3 | −%0.61 | yalnız quadratür suçu |
| Segment-uzaklığı + tam quadratür (derece 10) | 0 | +%0.27 (min) | Courant–Fischer |
| **Düğüm değerlerinin P1 interpolantı + tam P1 matrisleri** | **0** | +%0.32 (min) | garanti: $\hat\lambda\ge\lambda_{P1}\ge\lambda$ |
| Kenar ortalarında da değerlendirip P2 interpolant + P2 matrisleri | 0 | +%0.35 (min) | garanti |

### 6.2 Rastgele alt-örnekleme

`dataset.max_nodes=1024`, meshlerin çoğunda 1225–2180 düğümden ~yarısını seçer. Ağırlıklar $w_i$ tam mesh'ten gelir ve 63× değişir. Sonuç, Ritz matrislerinin gürültülü bir oran tahmin edicisi:
- göreli std: medyan %4.8 (sharp %3.6, smooth %5.6)
- mükemmel bazda (FEM özfonksiyonları): %1.1–7.7
- bias: M ile büyüyen negatif, ortalama −%0.3 → −%1.3; Ritz minimumu için Jensen-tipi etki

FE assembly tüm mesh'i gerektirdiği için alt-örnekleme SpectralNO'da kapatılmalı. Bellek gerekiyorsa rastgele düğüm alt kümesi yerine **daha kaba bir mesh** (ör. 2× kaba, ~600–2000 düğüm) üretilmeli.

### 6.3 Minimal değişiklik ve maliyeti

Değişiklik:
- Ağ aynı kalır. Düğüm değerleri $\Psi$ (gate sınır düğümlerinde zaten tam 0, çünkü `dist_bnd=0`) P1 interpolantı olarak yorumlanır.
- Eleman başına barisentrik gradyanlar $G_T\in\mathbb R^{3\times2}$ ve alan $|T|$ ile:
  $$\nabla\psi_h|_T=\sum_{a=1}^3\Psi_{T_a}\,G_{T,a},\qquad L=\sum_T|T|\,(\nabla\psi_h|_T)(\nabla\psi_h|_T)^\top,$$
  $$M=\sum_T\frac{|T|}{12}\Big[\big(\textstyle\sum_a\Psi_{T_a}\big)\big(\sum_a\Psi_{T_a}\big)^\top+\sum_a\Psi_{T_a}\Psi_{T_a}^\top\Big]\quad(\text{consistent P1 mass})$$
- Batched torch ile gather + 3 `einsum`. Autograd ile $\nabla\psi$ hesaplama, `_differentiable_inputs` ve `_basis_gradient` gereksizleşir. Donuk feature'lar artık "sızıntı" üretmez, çünkü gradyan interpolanttan geliyor ve sıçramaları da görüyor.
- `elements` batch'e eklenir (`[B, E_max, 3]` int + maske; $E\approx2N$).

**Maliyet** (CPU, 4 thread, B=8, N≤2072, E≤3808, M=16, ileri + geri):
- mevcut: **4517 ms**
- P1 assembly: **84 ms (54×)**

Mevcut kodun maliyeti büyük ölçüde vmapped `autograd.grad(create_graph=True)`'den geliyor. P2 varyantı ağı ×3.85 noktada değerlendirir (P2 DOF / vertex oranı ölçüldü). Tahmini ~0.3 s, yine mevcuttan ~15× ucuz.

**P1 mi P2 mi?**

| | P1 interp | P2 interp |
|---|---|---|
| Üst sınır | ✅ ($\ge\lambda_{P1}$) | ✅ ($\ge\lambda_{P2}$) |
| Taban (mükemmel ψ düğüm değerleri) | +%0.20…+%0.73 | etiket seviyesi (~1e-4) |
| Ağ değerlendirme | N | ~3.85 N (+ kenar ortası feature'ları) |
| Gerekli veri | `elements` | `elements` + kenar ortası feature'ları (KD-tree sorgusu) |

Öneri: önce P1. Taban bugünkü model hatasının (~%1) altında. Hedef $\lambda$ olarak P1 tabanı biliniyorsa ($\lambda_{P1}$ üretici tarafında ucuz) kayıp tutarlı hâle getirilebilir. Doğruluk %0.2'nin altına inince P2'ye geçilir.

**Mass lumping takası** (P1 sertliği + mükemmel baz, 4 şekil):

| Kütle | $\hat\lambda/\lambda-1$ aralığı |
|---|---|
| consistent | **+2.4e-3 … +7.3e-3** (her zaman üst sınır) |
| lumped ($\Psi^\top\mathrm{diag}(w)\Psi$ = mevcut kütle matrisi) | −1.0e-2 … +2.8e-3 (işaret belirsiz) |

Lumping P1 sertlik fazlasını kısmen iptal eder ve bazen daha küçük mutlak hata verir. Ama işaret garantisini yok eder. Armentano & Durán (2003) da lumped değerlerin pürüzsüz özfonksiyonlarda altta kaldığını raporlar. Sertifika ve Ky Fan / etiketsiz kayıp için **consistent** kütle kullanılmalı.

### 6.4 Köşe zenginleştirmesi (docs/14 #7 doğrulaması)

L-şekli, $\omega=(1-x^2)(1-y^2)(r-x-y)$ (uyumlu, Lipschitz). Tekil fonksiyonlar $(1-x^2)(1-y^2)\,r^{2k/3}\sin(2k\varphi/3)$. Köşeye geometrik bölünmüş 45 seviyeli 14×14 Gauss quadratürü kullanıldı, yani kesin.

| Baz | M | $\lambda_1$ hata | $\lambda_2$ | $\lambda_3$ |
|---|---|---|---|---|
| ω·polinom, derece 4 | 15 | 1.42e-2 | 4.4e-2 | 1.7e-2 |
| ω·polinom, derece 12 | 91 | 5.4e-3 | 6.1e-3 | 7.4e-4 |
| derece 4 + 1 tekil | 16 | 8.4e-5 | 4.4e-2 | 1.2e-2 |
| derece 4 + 3 tekil | 18 | 4.8e-5 | 4.1e-4 | 6.6e-4 |
| derece 4 + 6 tekil | 21 | **3.5e-6** | 2.2e-4 | 1.3e-4 |
| derece 8 + 6 tekil | 51 | 1.7e-7 | 3.7e-6 | 2.2e-6 |

Pürüzsüz baz cebirsel ve yavaş yakınsıyor. Tek bir $r^{2/3}$ terimi $\lambda_1$'i 170× iyileştiriyor, $\lambda_2$ için $r^{4/3}$ terimi gerekiyor. SpectralNO'ya eklenmesi #1'den **sonra** anlamlı, çünkü tekil fonksiyonların gradyanı ancak doğru quadratürle doğru entegre edilir. P1/P2 interpolantı köşede $h^{2\pi/\alpha}$ tabanı getirir, bu durumda enrichment P1 fonksiyonlarına eklenir (GFEM; Fix, Gulati & Wakoff 1973).

---

## 7. Alternatif Yüksek Doğruluklu Yöntemler (validasyon)

| Yöntem | Bu veri seti için | Ölçülen / beklenen |
|---|---|---|
| **MPS** (Fox–Henrici–Moler; Betcke & Trefethen 2005). Her köşede $J_{k\pi/\alpha_j}(\sqrt\lambda r)\sin(k\pi\theta/\alpha_j)$; sınır + iç noktalar; QR sonrası $\sigma(\lambda)=\sigma_{min}(Q_B)$ minimizasyonu | **Sharp (7–12 köşe) için en ucuz referans.** | L-şekli: N=14/köşe (84 fonk.) → **5.4e-9**, 0.28 s. Veri seti: s=2, 9 (konveks) P4 referansla **1e-8** uyum, 3 mod 0.3–0.7 s. s=16, 17 (re-entrant) fark 3e-7 = P4'ün kendi hatası, σ≈1e-9, 1–3 s. Moler–Payne sınırıyla sertifika da verir. Jones (2017) aynı yaklaşımla yüzlerce basamağa çıkıyor. |
| **BIE / Fredholm determinantı** (Bornemann 2010; Zhao & Barnett 2015) | Yalnız sınır ayrıklaştırması. Pürüzsüz sınırda spektral doğruluk. Köşelerde graded panel gerekir. | "Smooth" aile **analitik $r(\theta)$ olarak** tanımlanırsa ideal. Bugünkü 100-gon'da 100 köşe hem MPS'i hem BIE'yi pahalılaştırıyor. |
| **Schwarz–Christoffel / konformal harita** (SC Toolbox; Beceanu vd. 2026) | $-\Delta_wv=\lambda|f'(w)|^2v$ diskte. Köşe tekilliği $|f'|^2$ ağırlığına taşınır. | Referans etiket için MPS'ten pahalı ve karmaşık (SC parametre problemi + ağırlıklı spektral çözüm). Değeri daha çok **ortak referans domain** (FNO/POD için) olarak. |
| **P4 + graded FEM** (bu notun referansı) | Genel, basit | ~3e-7, şekil başına 5–10 s |

**Öneri:**
- Sharp şekiller ve kalibrasyon için MPS tabanlı küçük bir `scripts/mps_reference.py` (≈60 satır, SciPy `jv`). 1000 şekillik bir validasyon seti ~10 dakikada ~1e-8 doğrulukla üretilir.
- Smooth aile için önce geometri düzeltilmeli (docs/14 #7). Sonra BIE veya P4-graded FEM kullanılır.

---

## 8. Veri Üretimini Hızlandırma

**Ölçülen dağılım** (60 örnek, tek thread, k=3):

| | sharp (medyan) | smooth (medyan) | tümü (medyan / ort.) |
|---|---|---|---|
| toplam | 123 ms | 203 ms | 150 / 165 ms |
| gmsh `generate` | 46 | 85 | 58 (%39) |
| P2 assembly + condense (skfem) | 23 | 33 | 28 (%19) |
| `eigsh` (σ=500, shift-invert) | 51 | 72 | 64 (%42) |
| bunun içinde `splu(K)` | 16 | 26 | 24 |

5000 örnek / 4 worker ≈ **3.4 dk**. Hızlandırma seçenekleri:

| Yöntem | Beklenen kazanç (bu boyutta) | Not |
|---|---|---|
| LOBPCG (Knyazev 2001), LU veya AMG ön-koşullu | `eigsh` 64 → ~35 ms: toplamda ~%20 | pyamg kurulu değil. ILU'lu LOBPCG ölçüldü, `eigsh`'tan hızlı değil |
| İki-ızgara (Xu & Zhou 2001) | ~%20 | ince LU yine gerekli |
| Şekil ailesi boyunca ısıtma / alt-uzay geri dönüşümü | ~%20 | farklı mesh'ler arası aktarım (interpolasyon) gerekir |
| FEAST / Sakurai–Sugiura (kontur integrali) | yok (tek çekirdek); paralelde ölçeklenir | aralıktaki tüm modlar, bağımsız çözümler |
| RB (Machiels vd. 2000; Horger vd. 2017; Fumagalli vd. 2016) | online ~ms | ortak referans domain + geometrik harita (EIM) gerekir. ~24 parametreli poligon ailesinde offline maliyet ve sertifika zor. Kümeler/kesişmeler RB'yi zorlar |
| Sabit topolojili mesh morfolama (yıldız-şekiller) | gmsh (%39) + sembolik faktorizasyon tekrar kullanımı | köşe grading'i ile çelişir |

**Sonuç:** Veri üretimi darboğaz değil. Hesabı **hıza değil doğruluğa ve sertifikaya** harcamak daha değerli: #2 (LG, +0.3 s) ve #4 (sharp'ta daha hızlı ve 9× doğru).

---

## 9. Literatür Özeti (konu başlıklarına göre)

1. **Garantili sınırlar:**
   - CR tabanlı alt sınırlar Carstensen & Gedicke (2014) ve Liu (2015) ile açık sabitlidir ($\kappa=0.1893$).
   - Liu & Oishi (2013) re-entrant köşeli keyfi poligonlarda ilk doğrulanmış sınırlar.
   - LG yöntemi (Behnke & Goerisch 1994) akı rekonstrüksiyonu ile (Vejchodský 2018a) en keskin alt sınırları verir. Vejchodský (2018b) üç yöntemi karşılaştırır.
   - Cancès vd. (2017; 2018; 2020) ve Liu & Vejchodský (2022) özvektör ve küme sınırları.
   - Kato (1949) Temple eşitsizliğinin genellemesi.
2. **Mesh/veri kalitesi:**
   - Babuška, Kellogg & Pitkäranta (1979): graded mesh.
   - Schwab (1998): hp üstel yakınsama.
   - Durán, Padra & Rodríguez (2003): residual tahmin edici.
   - Dai, Xu & Zhou (2008): adaptif FEM optimal karmaşıklık.
   - Gallistl (2015): kümeler için optimal AFEM.
3. **Çözücüler:**
   - LOBPCG (Knyazev 2001), iki-ızgara (Xu & Zhou 2001), FEAST (Polizzi 2009), Sakurai–Sugiura (2003).
   - RB: Machiels vd. 2000; Horger, Wohlmuth & Dickopf 2017; Fumagalli, Manzoni, Parolini & Verani 2016.
4. **ML + çözücü:**
   - Öğrenilmiş multigrid (Greenfeld vd. 2019; Luz vd. 2020).
   - GNN ön-koşullayıcılar (Chen 2025).
   - HINTS (Zhang vd. 2024).
   - NOWS (Eshaghi vd. 2025/26).
   - Alt-uzay regresyonu ile LOBPCG ısıtma (Fanaskov vd. 2026; 2–3× hızlı yakınsama, ~2 mertebe düşük hata).
   - Solver-in-the-loop eğitim (Um vd. 2020).
5. **Referans çözücüler:**
   - MPS (Betcke & Trefethen 2005; Trefethen & Betcke 2006; Jones 2017; MPSpack).
   - Fredholm determinantı (Bornemann 2010; Zhao & Barnett 2015).
   - Konformal harita (SC Toolbox; Beceanu vd. 2026).
6. **Quadratür:**
   - Banerjee & Osborn (1990): yeterli dereceli quadratür a priori oranları korur ama işaret garantisi vermez.
   - Armentano & Durán (2003): mass lumping.

---

## 10. Deney Scriptleri (repo dışında, `scratchpad/math_numerics/`)

| Script | İçerik |
|---|---|
| `common.py`, `p2eval.py` | üretici poligonlarını yeniden kurma, mesh (üretici alanı + köşe grading), P2/P4 çözüm, hızlı nokta değerlendirme |
| `t1_refcheck.py` | referans (P3/P4 graded) yakınsama kontrolü |
| `t2_labels.py`, `t2_table.py` | 40 şekil etiket çalışması + alternatifler (§2) |
| `t3_conv.py`, `t3_rates.py` | L-şekli ve s=17 yakınsama oranları (§3) |
| `t4_quad.py`, `t4b_ideal.py`, `t12_lump.py` | SpectralNO assembly/quadratür/alt-örnekleme, lumping (§6) |
| `t5_hybrid.py`, `t5_table.py`, `t5b_krylov.py` | hibrit çözücü (§5) |
| `t6_bounds.py`, `t7_lg.py` | CR ve Lehmann–Goerisch sınırları (§4) |
| `t8_lshape.py` | köşe zenginleştirmesi (§6.4) |
| `t9_timing.py`, `t13_alt_timing.py` | üretici zaman dağılımı, alternatiflerin temiz zamanlaması (§2.4, §8) |
| `t10_cost.py` | SpectralNO mevcut vs P1-assembly maliyeti (§6.3) |
| `t11_mps.py` | method of particular solutions (§7) |

---

## 11. Referanslar

- Armentano, M. G., Durán, R. G. (2003). *Mass-lumping or not mass-lumping for eigenvalue problems.* Numer. Methods PDE 19(5), 653–664. [semanticscholar](https://www.semanticscholar.org/paper/Mass%E2%80%90lumping-or-not-mass%E2%80%90lumping-for-eigenvalue-Armentano-Dur%C3%A1n/d9c6a25c4a754927de99b43b5714ea7499f23e81)
- Babuška, I., Kellogg, R. B., Pitkäranta, J. (1979). *Direct and inverse error estimates for finite elements with mesh refinements.* Numer. Math. 33, 447–471. [link](https://link.springer.com/article/10.1007/BF01399326)
- Banerjee, U., Osborn, J. E. (1990). *Estimation of the effect of numerical integration in finite element eigenvalue approximation.* Numer. Math. 56, 735–762. [link](https://link.springer.com/article/10.1007/BF01405286)
- Barnett, A. H., Betcke, T. *MPSpack: particular solutions and integral equations toolbox.* [GitHub](https://github.com/ahbarnett/mpspack)
- Beceanu, M., Hong, J., Kwon, H.-K., Lim, M. (2026). *The eigenvalue problem for the Laplacian via conformal mapping and the Gohberg–Sigal theory.* Potential Analysis 64(3). [arXiv:2112.11026](https://arxiv.org/abs/2112.11026)
- Behnke, H., Goerisch, F. (1994). *Inclusions for eigenvalues of selfadjoint problems.* In: Topics in Validated Computations, Elsevier, 277–322. (Künye ve modern kullanımı: Vejchodský 2018a/b bağlantıları.)
- Betcke, T., Trefethen, L. N. (2005). *Reviving the method of particular solutions.* SIAM Review 47, 469–491. [PDF](https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_112.pdf)
- Bornemann, F. (2010). *On the numerical evaluation of Fredholm determinants.* Math. Comp. 79, 871–915. [arXiv:0804.2543](https://arxiv.org/abs/0804.2543)
- Cancès, E., Dusson, G., Maday, Y., Stamm, B., Vohralík, M. (2017). *Guaranteed and robust a posteriori bounds for Laplace eigenvalues and eigenvectors: conforming approximations.* SIAM J. Numer. Anal. 55, 2228–2254. [MR3702871](https://mathscinet.ams.org/mathscinet/relay-station?mr=3702871). Devamı (2020, Math. Comp. 89, 2563–2611): *…multiplicities and clusters.* [arXiv:2008.04140](https://arxiv.org/pdf/2008.04140)
- Carstensen, C., Gedicke, J. (2014). *Guaranteed lower bounds for eigenvalues.* Math. Comp. 83, 2605–2629. [PDF](https://ins.uni-bonn.de/media/public/publication-media/CG_2014.pdf?pk=1442)
- Carstensen, C., Puttkammer, S. (2024). *Adaptive guaranteed lower eigenvalue bounds with optimal convergence rates.* Numer. Math. 156, 1–38. [link](https://www.researchgate.net/publication/358975290_Adaptive_guaranteed_lower_eigenvalue_bounds_with_optimal_convergence_rates)
- Chen, J. (2025). *Graph neural preconditioners for iterative solutions of sparse linear systems.* ICLR 2025. [arXiv:2406.00809](https://arxiv.org/abs/2406.00809)
- Dai, X., Xu, J., Zhou, A. (2008). *Convergence and optimal complexity of adaptive finite element eigenvalue computations.* Numer. Math. 110, 313–355. [link](https://link.springer.com/article/10.1007/s00211-008-0169-3)
- Driscoll, T. A. *Schwarz–Christoffel Toolbox* (bkz. Driscoll & Trefethen 2002, *Schwarz–Christoffel Mapping*, Cambridge). [MATLAB Central](https://www.mathworks.com/matlabcentral/fileexchange/1316-schwarz-christoffel-toolbox)
- Durán, R. G., Padra, C., Rodríguez, R. (2003). *A posteriori error estimates for the finite element approximation of eigenvalue problems.* Math. Models Methods Appl. Sci. 13(8), 1219–1229. [link](https://www.worldscientific.com/doi/10.1142/S0218202503002878)
- Eshaghi, M. S., Anitescu, C., Valizadeh, N., Wang, Y., Zhuang, X., Rabczuk, T. (2025). *NOWS: Neural Operator Warm Starts for accelerating iterative solvers.* arXiv:2511.02481 (CMAME 2026). [arXiv](https://arxiv.org/abs/2511.02481)
- Fanaskov, V., Trifonov, V., Rudikov, A., Muravleva, E., Oseledets, I. (2026). *Deep learning for subspace regression.* ICLR 2026. [arXiv:2509.23249](https://arxiv.org/pdf/2509.23249)
- Fumagalli, I., Manzoni, A., Parolini, N., Verani, M. (2016). *Reduced basis approximation and a posteriori error estimates for parametrized elliptic eigenvalue problems.* ESAIM: M2AN 50, 1857–1885. [numdam](http://www.numdam.org/articles/10.1051/m2an/2016009/)
- Gallistl, D. (2015). *An optimal adaptive FEM for eigenvalue clusters.* Numer. Math. [link](https://link.springer.com/article/10.1007/s00211-014-0671-8)
- Greenfeld, D., Galun, M., Kimmel, R., Yavneh, I., Basri, R. (2019). *Learning to optimize multigrid PDE solvers.* ICML 2019. [PMLR](http://proceedings.mlr.press/v97/greenfeld19a.html)
- Horger, T., Wohlmuth, B., Dickopf, T. (2017). *Simultaneous reduced basis approximation of parameterized elliptic eigenvalue problems.* ESAIM: M2AN 51(2), 443–465. [link](https://www.esaim-m2an.org/articles/m2an/abs/2017/02/m2an150113/m2an150113.html)
- Jones, R. S. (2017). *Computing ultra-precise eigenvalues of the Laplacian within polygons.* Adv. Comput. Math. 43, 1325–1354. [link](https://link.springer.com/article/10.1007/s10444-017-9527-y)
- Kato, T. (1949). *On the upper and lower bounds of eigenvalues.* J. Phys. Soc. Japan 4, 334–339. (Künye kaynağı: Tosio Kato's work on non-relativistic QM, [arXiv:1711.00528](https://arxiv.org/pdf/1711.00528))
- Knyazev, A. V. (2001). *Toward the optimal preconditioned eigensolver: locally optimal block preconditioned conjugate gradient method.* SIAM J. Sci. Comput. 23(2), 517–541. [semanticscholar](https://www.semanticscholar.org/paper/Toward-the-Optimal-Preconditioned-Eigensolver:-Knyazev/b80a3e1a89714724f5b3aba96a586c1e7d01c3d7)
- Liu, X. (2015). *A framework of verified eigenvalue bounds for self-adjoint differential operators.* Appl. Math. Comput. 267, 341–355. [link](https://www.sciencedirect.com/science/article/abs/pii/S0096300315003628)
- Liu, X., Oishi, S. (2013). *Verified eigenvalue evaluation for the Laplacian over polygonal domains of arbitrary shape.* SIAM J. Numer. Anal. 51(3), 1634–1654. [arXiv:1204.4119](https://arxiv.org/abs/1204.4119)
- Liu, X., Vejchodský, T. (2022). *Fully computable a posteriori error bounds for eigenfunctions.* Numer. Math. 152, 183–221. [link](https://link.springer.com/article/10.1007/s00211-022-01304-0)
- Luz, I., Galun, M., Maron, H., Basri, R., Yavneh, I. (2020). *Learning algebraic multigrid using graph neural networks.* ICML 2020, PMLR 119, 6489–6499. [PMLR](https://proceedings.mlr.press/v119/luz20a.html)
- Machiels, L., Maday, Y., Oliveira, I. B., Patera, A. T., Rovas, D. V. (2000). *Output bounds for reduced-basis approximations of symmetric positive definite eigenvalue problems.* C. R. Acad. Sci. Paris Sér. I 331(2), 153–158. [link](https://www.sciencedirect.com/science/article/abs/pii/S0764444200002706)
- Polizzi, E. (2009). *Density-matrix-based algorithm for solving eigenvalue problems* (FEAST). Phys. Rev. B 79, 115112. [arXiv:0901.2665](https://arxiv.org/abs/0901.2665v1)
- Sakurai, T., Sugiura, H. (2003). *A projection method for generalized eigenvalue problems using numerical integration.* J. Comput. Appl. Math. 159, 119–128. [ResearchGate](https://www.researchgate.net/publication/222698866_A_projection_method_for_generalized_eigenvalue_problems)
- Schwab, C. (1998). *p- and hp-Finite Element Methods: Theory and Applications in Solid and Fluid Mechanics.* Oxford/Clarendon. [Google Books](https://books.google.com/books/about/P_and_hp_finite_element_methods.html?id=PwNt8GLWUqsC&hl=en)
- Trefethen, L. N., Betcke, T. (2006). *Computed eigenmodes of planar regions.* Contemp. Math. 412, 297–314. [PDF](https://people.maths.ox.ac.uk/trefethen/publication/PDF/2006_116.pdf)
- Um, K., Brand, R., Fei, Y., Holl, P., Thuerey, N. (2020). *Solver-in-the-Loop: learning from differentiable physics to interact with iterative PDE-solvers.* NeurIPS 33. [link](https://proceedings.neurips.cc/paper/2020/hash/43e4e6a6f341e00671e123714de019a8-Abstract.html)
- Vejchodský, T. (2018a). *Flux reconstructions in the Lehmann–Goerisch method for lower bounds on eigenvalues.* J. Comput. Appl. Math. [arXiv:1801.02561](https://arxiv.org/abs/1801.02561)
- Vejchodský, T. (2018b). *Three methods for two-sided bounds of eigenvalues — a comparison.* Numer. Methods PDE. [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/num.22251)
- Xu, J., Zhou, A. (2001). *A two-grid discretization scheme for eigenvalue problems.* Math. Comp. 70(233), 17–25. [AMS](https://www.ams.org/journals/mcom/2001-70-233/S0025-5718-99-01180-1/viewer/)
- Zhang, E., Kahana, A., Kopaničáková, A., Turkel, E., Ranade, R., Pathak, J., Karniadakis, G. E. (2024). *Blending neural operators and relaxation methods in PDE numerical solvers* (HINTS). Nature Machine Intelligence. [arXiv:2208.13273](https://arxiv.org/abs/2208.13273)
- Zhao, L., Barnett, A. (2015). *Robust and efficient solution of the drum problem via Nyström approximation of the Fredholm determinant.* SIAM J. Numer. Anal. 53(4), 1984–2007. [arXiv:1406.5252](https://arxiv.org/abs/1406.5252)

Klasik (bağlantısız) arka plan: Babuška & Osborn (1991) *Eigenvalue problems*, Handbook of Numer. Anal. II; Grisvard (1985) *Elliptic Problems in Nonsmooth Domains*; Fix, Gulati & Wakoff (1973) J. Comput. Phys. 13.

---

## 🔗 Bağlantılar
- Önceki matematik notu: [[14_MATHEMATICAL_IMPROVEMENTS]]
- Veri üretimi: [[01_DATA_GENERATION]]
- Rayleigh–Ritz / SpectralNO: [[13_RAYLEIGH_ANALYSIS]], [[04_MODEL_ARCHITECTURE]]

#sayisal-analiz #fem #ozdeger-sinirlari #spectral-no #hibrit-cozucu
