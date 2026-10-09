# 27 — Etiket kalitesi: N0 etiketleri ve yüksek mertebe (NGSolve) etiketleri

**Soru.** Mevcut etiketler (mesh 0.10'da Whitney N0) CST'ye ne kadar yakın — frekansta **ve alanda**?
NGSolve p2/p3 eğri eleman etiketleri ne kazandırır, ne kadara mal olur?

**Araçlar.** `src/data_gen/highorder.py` (eğri netgen ağı + tam HCurl p uzayı + kendi ayrık gradyanıyla
projeksiyonlu shift-invert, static condensation, doğrudan yüksek mertebe QoI) ve
`scripts/label_benchmark.py` (aynı geometri: üreticinin gmsh OCC katısı BREP olarak yazılır).

```bash
python scripts/label_benchmark.py --families elliptical reentrant --ids 0 --k 6 \
    --ref 3:3:2:2:0.3 3:3:3:2:0.5 --cand 2:3:3:2:0.5 --n0_mesh_sizes 0.05 --json out.json --md out.md
```

## 1. Metrikler

Hepsi referansa göre, ilk K = 6 mod; dejenere kümeler (referans frekansları %0.5 içinde) birlikte uydurulur:

- **Δf/f** — K mod üzerinde en büyük bağıl frekans hatası.
- **E L2 / H L2** — ‖E_ref − Σc_j E_j‖ / ‖E_ref‖. Integral etiket ağının tet kareleme noktalarında alınır;
  c en iyi küme katsayılarıdır (işaret ve genlik serbest). H aynı c ile curl E'dir. Tablodaki değer modlar
  üzerinden medyandır.
- **QoI** — mod 0'ın Q0, R/Q, Epk/Eacc ve Bpk/Eacc değerleri (bakır, β = 1, docs/24).
  - Yüksek mertebe yöntemlerde QoI doğrudan alandan hesaplanır: duvar değerleri komşu hacim elemanından
    alınır (`BoundaryFromVolumeCF`).
  - N0 yöntemlerinde etiketlerdeki operatörler kullanılır (`src/qoi`).
- **best N0** — referansın etiket ağının N0 uzayına L2 projeksiyonu. Bu ağda **herhangi bir** N0 alanının
  ulaşabileceği en iyi E doğruluğudur: modelin çıktı temsilinin sınırı. Projeksiyon E'yi optimize eder,
  curl E'yi optimize etmez; bu yüzden H ve QoI değerleri anlamlı değildir.
- **p…→N0** — yüksek mertebe çözümün etiket ağındaki N0 kenar DOF'larına interpolasyonu (u_e = ∫_e E·t).
  Yüksek mertebe etiketlerle eğitilen bir modelin tam olarak göreceği hedef budur.

Yöntem adları: `p{p}c{eğri}h{maxh/h}g{grading}`, referans `p:eğri:maxh/h:curvaturesafety:grading`.

## 2. Sonuçlar (aile başına 1 örnek, deformasyonsuz)

| aile | ref | yöntem | max\|Δf/f\| | E L2 | H L2 | Q0 | R/Q | Epk/Eacc | Bpk/Eacc | süre [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| pillbox_pipes | 4:4:2:2:0.3 | N0@0.1 | 2.4e-03 | 13% | 12% | -4% | -0.77% | -24% | +0.078% | 2 |
| pillbox_pipes | 4:4:2:2:0.3 | best N0 | 0 | 12% | — | — | — | — | — | — |
| pillbox_pipes | 4:4:2:2:0.3 | N0@0.07 | 2.4e-03 | 11% | 11% | -3.5% | +0.3% | +2.8% | -0.21% | 3 |
| pillbox_pipes | 4:4:2:2:0.3 | **p3c3h3g0.5** | **5.4e-06** | **0.037%** | **0.19%** | **-0.0014%** | **+0.00016%** | **+0.29%** | **+0.033%** | 56 |
| pillbox_pipes | 4:4:2:2:0.3 | p3c3h3g0.5→N0 | 5.4e-06 | 13% | 13% | -7% | -1.2% | -21% | +2.3% | 56 |
| pillbox_pipes | 4:4:2:2:0.3 | p2c3h3g0.5 | 2.4e-04 | 0.22% | 1.7% | -0.39% | +0.0042% | +6% | +6.1% | 19 |
| composite | 3:3:2:2:0.3 | N0@0.1 | 4.4e-03 | 19% | 17% | -4.5% | +1.3% | -49% | -25% | 1 |
| composite | 3:3:2:2:0.3 | best N0 | 0 | 15% | — | — | — | — | — | — |
| composite | 3:3:2:2:0.3 | N0@0.05 | 1.6e-03 | 8.9% | 8.7% | -2.5% | +0.45% | -40% | -23% | 15 |
| composite | 3:3:2:2:0.3 | p2c3h3g0.5 | 1.1e-03 | 1.5% | 4.8% | -2.1% | +0.033% | -21% | +23% | 8 |
| composite | 3:3:2:2:0.3 | p2c3h3g0.5→N0 | 1.1e-03 | 17% | 17% | -4.2% | -0.28% | -47% | -24% | 8 |
| elliptical | 3:3:2:2:0.3 | N0@0.1 | 3.5e-03 | 18% | 18% | -4.6% | +0.98% | -13% | +0.78% | 1 |
| elliptical | 3:3:2:2:0.3 | best N0 | 0 | 16% | — | — | — | — | — | — |
| elliptical | 3:3:2:2:0.3 | N0@0.05 | 7.8e-04 | 8.9% | 8.6% | -2.7% | +0.98% | +3% | -0.42% | 37 |
| elliptical | 3:3:2:2:0.3 | p2c3h3g0.5 | 5.5e-05 | 0.15% | 0.97% | -0.24% | -0.0068% | +0.39% | +3.4% | 145 |
| elliptical | 3:3:2:2:0.3 | p2c3h3g0.5→N0 | 5.5e-05 | 18% | 18% | -9.1% | -2.4% | -10% | +4.4% | 145 |
| spoke ¹ | 3:3:3:2:0.5 | N0@0.1 | 1.3e-02 | 20% | 15% | +33% | -3.3% | -23% | -28% | 2 |
| spoke ¹ | 3:3:3:2:0.5 | best N0 | 0 | 17% | — | — | — | — | — | — |
| spoke ¹ | 3:3:3:2:0.5 | N0@0.05 | 7.7e-03 | 14% | 11% | +26% | -1.5% | -20% | -20% | 13 |
| spoke ¹ | 3:3:3:2:0.5 | p2c3h3g0.5 | 1.2e-03 | 1.4% | 4.4% | +1.9% | -0.16% | +1.1% | +9.5% | 10 |
| hwr ¹ | 3:3:3:2:0.5 | N0@0.1 | 4.8e-03 | 23% | 19% | +34% | +0.27% | -36% | -39% | 1 |
| hwr ¹ | 3:3:3:2:0.5 | best N0 | 0 | 21% | — | — | — | — | — | — |
| hwr ¹ | 3:3:3:2:0.5 | N0@0.05 | 1.6e-03 | 12% | 9.7% | +17% | -0.86% | -25% | -22% | 14 |
| hwr ¹ | 3:3:3:2:0.5 | p2c3h3g0.5 | 6.8e-04 | 1% | 3% | +1.9% | +0.023% | -0.079% | +13% | 12 |
| reentrant ¹ | 3:3:3:2:0.5 | N0@0.1 | 2.4e-02 | 21% | 14% | +1.4% | -0.18% | -7.2% | -5.9% | 2 |
| reentrant ¹ | 3:3:3:2:0.5 | best N0 | 0 | 19% | — | — | — | — | — | — |
| reentrant ¹ | 3:3:3:2:0.5 | N0@0.05 | 2.0e-02 | 16% | 10% | +1.1% | -1.1% | -10% | -4.1% | 10 |
| reentrant ¹ | 3:3:3:2:0.5 | p2c3h3g0.5 | 3.0e-04 | 0.58% | 1.8% | +0.1% | +0.0064% | +6.8% | +4.6% | 22 |

¹ Bu ailelerde referans ve p2 adayı aynı ağı kullanır (daha ince referans 800k DOF sınırını aştı). Buradaki p2
hatası yalnız p-hatasıdır.

Süreler 4 iş parçacığı içindir. N0 süresi = mesh + çözüm; yüksek mertebe süresi = netgen ağı + çözüm.
N0@0.07 pillbox_pipes'ta kenar sayısını yalnız 1.4× artırdı (küçük özellik sınırı `h_cap`).

## 3. Bulgular

1. **Mevcut etiketler (N0 @ 0.10).**
   - Frekans: %0.25–2.4 (reentrant'ın kapasitif aralığı en kötüsü).
   - Alan: E %13–23, H %12–19.
   - Q0: %−5…+34; en kötüsü iç iletkenli spoke ve hwr.
   - Epk/Eacc: %−7…−49. R/Q: ≤ %3.
   - Frekansta "CST'ye yakın" sayılabilir, ancak alanda ve yüzey büyüklüklerinde değil.
2. **Daha çok mesh bir çözüm değil.**
   - N0 @ 0.05, 7–30 kat maliyetle alan hatasını ancak yarıya indirir (%9–16).
   - Spoke ve hwr'de Q0 hatası hâlâ %17–26.
3. **Yüksek mertebe eğri elemanlar.**
   - **p3** (pillbox_pipes, referans p4): frekans 5e-6, E %0.04, H %0.2, QoI ≤ %0.3. CST sınıfı doğruluk.
   - **p2**: frekans ≤ 1.2e-3 (çoğu ≤ 3e-4), E %0.15–1.5, H %1–5, Q0 ≤ %2.1, R/Q ≤ %0.16.
     Epk/Bpk %0.1–23 hata verir (tepe değerleri p2'de zayıf); yüzey tepeleri için p3 gerekir.
   - Maliyet: örnek başına p2 için 8–145 s, p3 için yaklaşık 2–3 katı; N0 ise 1–2 s.
4. **Alan doğruluğunun darboğazı modelin çıktı temsili.**
   - "best N0" satırına göre etiket ağındaki **en iyi** N0 alanının E hatası %12–21.
   - Yüksek mertebe etiketin N0'a interpolasyonu (p…→N0) mevcut N0 etiketi kadar hatalı: E %13–23,
     Q0 %4–39, Epk %7–47.
   - Yani model N0 kenar DOF'u ürettiği sürece **etiket ne kadar iyi olursa olsun** alan ve yüzey
     büyüklükleri CST'ye yaklaşamaz.
   - p3 etiketi doğrudan yalnız frekans etiketini iyileştirir, çünkü frekans etiketi bir skalerdir. Ancak modelin
     frekansı N0 Ritz değeri olduğu için %0.25–2.4'lük ayrıklaştırma tabanı kalır; öğrenilmiş bir frekans
     düzeltmesi gerekir.
5. **Eğri ağ üretimi sağlam değil** (üretim etiketleri için asıl mühendislik işi):
   - Netgen'in varsayılan eğrilik inceltmesi (`curvaturesafety` 2) küçük pahları çözer, ama ağ boyutu
     aileden aileye 1.3k–266k tet arasında oynar. Reentrant'ın burun pahı p3'te 7M DOF'a çıkar.
   - Eğrilik inceltmesi azaltılırsa kaba tetler küçük pahların üstünde katlanır (det J ≤ 0). Frekans neredeyse
     değişmez ama yüzey alanları ve duvar kaybı çöp olur: Jacobian oranı 0.05'te Q0 10 kat yanlış çıkar.
     `netgen_mesh` her ağı denetler ve gerekirse inceltir.
   - ridged_box ve dtl'nin CAD'inde sıfır uzunlukta kenarlar ve 0.14 h² büyüklüğünde yüzler var. Bu
     ailelerde geçerli eğri ağ elde edilemedi; ölçülmediler.
   - Gerekenler: üreticide geometri temizliği (en küçük pah yarıçapı, şerit yüzlerin kaldırılması,
     `occ.healShapes`), ya da gmsh'in yüksek mertebe ağ üretimi (`HighOrderOptimize`).

## 4. Pratik notlar (highorder.py)

- gmsh, döndürülen profil yüzeyini BREP'e serbest yüz olarak da yazar. Bu yüz netgen'de "boundary mesh is
  overlapping" hatası verir. Çözüm: yalnız katıyı (`shape.solids[0]`) ağlamak.
- HCurl için sınır değerleri sınır elemanından değil komşu hacim elemanından alınmalı
  (`BoundaryFromVolumeCF`). Düz bir sınır değerlendirmesi teğetsel izi verir: PEC duvarda E ≈ 0 olur ve
  n·curl E = 0 çıkar.
- Eğri elemanlarda duvara 1e-8 m mesafedeki noktaları aramak ters eşlemede başarısız olabilir ve gürültülü
  tepeler üretir. Bu yüzden nokta konumlamadan kaçınılır.
- Static condensation (`condense=True` + harmonic extension) p ≥ 3'te çarpanlara ayırmayı bir mertebe
  hızlandırır.
- p4 için bellek ≈ 1 GB / 70k DOF. Kıyaslama varsayılanı 800k DOF sınırıdır.
