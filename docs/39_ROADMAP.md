# 39 — Uzun vadeli plan: model, veri, ölçek ve veri yolu (2026-11-12)

Bu belge projenin bugünkü durumunu, kritik riskleri ve aşamalı planı toplar. Sayıların kaynağı: docs/37 (etiketler),
docs/38 (izdüşümsüz eğitim, pilot, GPU ölçümleri), docs/31 (oturum notları). "Tahmin" yazan her sayı ölçülmemiştir.
Küçük CPU denemeleri bilerek dışarıda bırakıldı: güçlü hesaplama kaynağı yoktu, kanıt değerleri düşük.

## 1. Özet

- **Hat uçtan uca çalışıyor:** geometri → eğri ağ → uyarlamalı p2 etiket → H5 → eğitim → izdüşüm + Ritz tahmin →
  arayüz. 13 aile × 1024 geometri üretildi (TAG `F_p2p2_ms0.10_k10_mf0.05_v3`, ~13 300 geometri, ~115 GB).
- **İlk pilot model zayıf, ama nedeni anlaşılır:** küçük model (0,85 M), 686 geometri, 50 epoch. Eğitim = doğrulama
  kaybı (ezber yok, az öğrenme); test: kütle normunda uzay hatası %8, **enerji normunda %25**, frekans hatası %7,5,
  bazın %15'i gradyan; OOD ailelerde belirgin kötü.
- **Tek cümlelik teşhis:** Ritz frekans hatası, enerji normundaki uzay hatasının **karesiyle** gider
  ((θ−λ)/λ ≈ ε_A²; 0,25² ≈ %6 — ölçülen %7,5 ile uyumlu). Yani bütün plan tek bir metriği düşürmeye bağlanabilir:
  **enerji normunda uzay hatası ε_A** (`span_rel_l2_energy`).

| hedef frekans hatası | gereken ε_A |
|---|---|
| %7,5 (bugün) | 0,25 |
| %1 | ~0,10 |
| %0,1 (etiket tabanı 1e-3) | ~0,03 |

- **Üç ana iş kolu:** (A) ε_A'yı düşürmek: veri × model × kayıp; (B) veri yolu: GPU'nun veri beklemesini bitirmek —
  kalıcı çözüm operatörleri GPU'da matrissiz kurmak; (C) genelleme ve değer önerisi: OOD, tahmin ağı kayması,
  FE'ye göre gerçek hız kazancı.

## 2. Kritik ve kırılgan noktalar

Önem sırasıyla; her biri için "nasıl anlarız" ve "ne yaparız".

### 2.1 Değer önerisi: FE'ye göre gerçek hız kazancı (stratejik)

- Bugün tahmin süresi CPU'da geometri başına ~10–30 s: ağ üretimi + operatör kurulumu baskın, modelin kendisi < 1 s
  (GPU). Aynı ağda p2 FE çözümü 4 çekirdekte ~45–150 s. Yani bugünkü kazanç **~3–5 kat**, "saniyeler" değil.
- Kazancın büyümesi için: (i) operatörlerin GPU'da matrissiz kurulması (§4.3) → tahmin < 1–2 s + ağ üretimi;
  (ii) ağ üretiminin hızlanması / parametrik ailelerde ağ dönüştürme (morphing); (iii) toplu (batch) tarama.
- **Yedek değer önerisi (güçlü):** modelin alt uzayı FE özdeğer çözücüsüne **başlangıç** olarak verilir (LOBPCG sıcak
  başlatma, docs/31 §3.4'te altyapı var). Sonuç FE kadar kesin, süre daha kısa. Model "yaklaşık çözücü" olmazsa bile
  "hızlandırıcı" olur; makale ve araç için güvenli zemin. Ön koşullandırıcı (rot-rot için AMS/BDDC) işin zor kısmı.

### 2.2 Enerji normu: modelin ince yapıyı öğrenememesi

- Kütle normu %8 iken enerji normu %25: model alanın şeklini biliyor, **rot E**'yi (duvar yakını, kenarlar, yüksek
  modlar) bilmiyor. Frekansı belirleyen bu.
- Olası nedenler: (a) az öğrenme (kesin: kayıp hâlâ düşüyordu); (b) gövdenin spektral yanlılığı: doğrusal dikkat
  düşük ranklı küresel bilgi taşır, yerel ince yapı için zayıftır; RFF uzunluk ölçeği sabit 0,1; (c) kaybın kütle ve
  enerjiyi eşit tartması.
- Nasıl anlarız: veri ve model ölçekleme eğrilerinde ε_A'nın eğimi (§3). Eğim düzleşirse mimari sınır.
- Ne yaparız: §3.3 (kayıp), §3.4 (mimari kaldıraçlar).

### 2.3 İzdüşümsüz eğitimin bedeli

- Test bazının %15'i gradyan (kapasite kaybı); frekans ve selfsup kaybı yok; `grad_frac_lb` göstergesi bunu
  yakalamadı (0,0015 gösterdi).
- Ucuz çözüm adayı: **diverjans cezası** — `‖GᵀMV‖²` (köşegen Kp⁻¹ ağırlıklı). Yalnız seyrek çarpım ister, CG yok.
  docs/31 §3.5'te reddedilen şey bunun *Ritz'in yerine* kullanılmasıydı; burada yalnız eğitim kaybında gradyanı
  bastıran terim, tahminde izdüşüm aynen duruyor. G'nin eğitim verisinde yeniden kurulması gerekir (~0,5 s/örnek;
  matrissiz yolda bedava).
- Diğer seçenekler: sonda kısa izdüşümlü ince ayar (frekans kaybıyla); N adımda bir izdüşümlü adım.

### 2.4 Tahmin ağı ≠ eğitim ağı (ölçülmedi, sinsi)

- Etiketler uyarlamalı inceltilmiş ağda; tahmin **başlangıç ağında** (seçenek a). Zor örneklerde eğitim ağı 2–6 kat
  daha ince. Model başlangıç ağlarını eğitimde yalnız kolay örneklerde gördü.
- Ayrıca "nokta hacmi" girdi özelliği ağ sıklığının parmak izi: kısayol riski (docs/38 §12).
- Nasıl anlarız: test örneklerini başlangıç ağında yeniden kurup `eval_by_family` ile ölçmek (etiket: aynı geometrinin
  ince ağ frekansları).
- Ne yaparız: nokta hacmi özelliğini çıkarmak; eğitimde aynı geometrinin iki ağını da göstermek (başlangıç ağında tek
  p2 çözümü ucuz: üreticiye "başlangıç ağı etiketi de sakla" seçeneği — üretici davranışı değişir, sahip onayı);
  uzun vadede seçenek (b): tahminde uyarlamalı tur.

### 2.5 Genelleme (OOD)

- Girdi dağılımı OOD ailelerde eğitim aralığının içinde; sorun öğrenilen fonksiyonun yeni şekil sınıflarına
  taşınmaması (aile başına 85 örnekle beklenen). Aile sayısı (8) örnek sayısından daha kısıtlayıcı olabilir.
- Ne yaparız: §5.

### 2.6 Hesaplama kaynağı ve veri güvenliği

- TRUBA GPU: `normal` QOS ile akya-cuda haftalarca kuyruk. Proje hesabı (QOS) olmadan ölçekli eğitim TRUBA'da
  yapılamaz; Colab G4 ücretli ve oturum sınırlı. **Proje başvurusu planın ön koşullarından.**
- Scratch 30 günde siliniyor: 115 GB'lık v3 verisi yedeklenmeli (Drive 2 TB planı ya da kurum deposu).

### 2.7 Daha küçük ama gerçek riskler

- Keskin kenar yedeği (composite'ın tamamı, spoke/hwr'nin yarısı): tepe alan etiketleri NaN; CAD fillet ve katlanma
  sorunları açık (docs/37 §3.3).
- `qoi_ok` düşük aileler (reentrant %65, dtl %81): Q0 / R/Q değerlendirmesi bu örneklerde yapılamıyor.
- Tepe alanlar (Epk, Bpk) noktasal büyüklük: ε_A küçülmeden güvenilir olmaz; beklenti buna göre kurulmalı.
- Yön sabit (x, y, z mutlak), normalizasyon L∞: uzun-ince şekiller sıkışıyor (elliptical_long).
- Etiket doğruluğu 1e-3: model %0,1'e yaklaşırsa `tol_f_p2` sıkılaştırılmalı (maliyet ~2–3 kat).

## 3. Model ve eğitim planı

### 3.1 Ölçekleme deneyi (ilk büyük iş)

Amaç: ε_A'nın veri (N) ve modele (P) göre eğrisini çıkarmak; "daha çok veri mi, daha büyük model mi" sorusunu
sayıyla kapatmak. Sabit: aynı test kümesi (geometri kimliğiyle), aynı adım bütçesi tanımı, ≥ 2 tohum uçlarda.

| | N = 1k | N = 4k | N = 8k (eldeki eğitim ailelerinin tamamı) |
|---|---|---|---|
| small (0,85 M) | ✓ | ✓ | ✓ |
| base (2,8 M) | ✓ | ✓ | ✓ |
| large (6,5 M) | | ✓ | ✓ |

- Ölçülen: ε_A, ε_M, frekans hatası, alan hatası, `grad_frac` — eğitim aileleri testi ve OOD ayrı (`eval_by_family`).
- Okuma: hata ∝ N^(−α). α ≳ 0,3 ise veri üretmeye devam (10k/aile hedefi anlamlı); α ≈ 0 ve model büyüyünce düşüyorsa
  kapasite; ikisi de düzse mimari/kayıp (§3.3–3.4).
- Bütçe (tahmin; G4'te ideal veri yoluyla ~0,3 s/örnek): N = 8k için epoch ~40 dk; 30 epoch ~20 saat ≈ 17 $.
  Tablonun tamamı ~150–200 GPU-saat. Veri yolu düzelmeden (§4) bu 2 katı.

### 3.2 Eğitim reçetesi (pilottan dersler)

- Adım bütçesi epoch'la değil **toplam adımla** tanımlanmalı (pilot: 34k adım, kayıp düşerken LR sıfırlandı). İlk
  hedef: ≥ 150k adım, ısınma + kosinüs, EMA ağırlıkları.
- `base` varsayılan; GPU veri beklediği sürece büyük model neredeyse bedava.
- Etkin batch 4–8 (biriktirme ya da boyuta göre gruplanmış gerçek batch, §4.2).
- Büyük GPU'da `grad_checkpoint` kapalı (bellek ~2,6 kat, adım ~%20–30 hızlı; ölçülecek).
- Gövdede bf16/fp16 otomatik karışık hassasiyet; Gram'lar FP64 kalır (ölçülecek: ε_A'ya etkisi).

### 3.3 Kayıp tasarımı (ucuz denemeler, her biri tek ayar)

1. Enerji ağırlığı: `span_norm: both` (0,5/0,5) yerine enerjiye ağırlık (ör. 0,25/0,75) — frekansı belirleyen o.
   Dikkat: yalnız enerji + izdüşümsüz yasak (gradyana baskı kalmaz).
2. Diverjans cezası (§2.3).
3. Mod ağırlıkları: düşük modlar tasarımda daha önemli; ağırlıklı ortalama.
4. Son aşama: izdüşümlü + frekans kayıplı kısa ince ayar (toplam adımların ~%5–10'u).
5. Karşılaştırma kolu: baştan sona izdüşümlü eğitim (aynı GPU-saat) — izdüşümsüz yolun gerçekten kazandırıp
   kazandırmadığını bir kez ölçmek gerekir.

### 3.4 Mimari kaldıraçlar (ölçekleme eğrisi düzleşirse)

- **Yerel katmanlar:** dikkat küresel/düşük ranklı; ince yapı için eleman komşuluğunda mesaj geçirme ya da yerel
  evrişim katmanı (ağ grafı zaten elde). En olası kazanç burada.
- **Çok ölçekli konum kodlaması:** RFF uzunluk ölçekleri {0,3; 0,1; 0,03}, öğrenilebilir.
- **Token azaltma:** gövde tet başına 14 nokta yerine 4–5 noktada çalışır, alan başlığı 14 noktaya çıkarır → gövde
  maliyeti ~3 kat düşer, aynı bütçeye daha geniş model sığar.
- **Baz boyutu:** m = 24 → 32–48 (K = 10 için pay); Ritz maliyeti önemsiz, Gram maliyeti m ile doğrusal.
- **Düzeltme adımı:** model uzayına 1–2 blok alt uzay yinelemesi (FE operatörüyle) — doğruluğu garantili artırır,
  §2.1'deki hibrit çözücüye köprü.
- **Global tanımlayıcılar:** hacim, en-boy, b1, duvar bileşeni sayısı gövdeye ek girdi (bugün yok).
- **Yön artırımı:** eksen permütasyonu / yansıma (operatörler de dönmeli: matrissiz yolda kolay, bugün zor).

## 4. Veri yolu: GPU'ya veri yetiştirme

### 4.1 Ölçülen durum

- G4'te yalın GPU adımı 0,26 s (23,7k tet) / 0,57 s (51k); gözlenen ~0,5 s/örnek; GPU kullanımı 8 işçiyle ~%40,
  16 işçiyle ~%45–50. Ana süreç tek çekirdekte ~%82; `/dev/shm` 16 işçide 64 GB.
- Örnek başına CPU işi: önbellekten okuma + açma ~1,5–2,5 s, `field_item` + collate ~0,5–2 s; önbelleksiz +6–12 s.
- Kök neden: her örnek için **yüzlerce MB'lık global seyrek matrisler** (K, M, C; 23k tette ~50 M sıfır olmayan)
  CPU'da kuruluyor / açılıyor, süreçler arası taşınıyor, GPU'ya kopyalanıyor. Sorun işçi sayısı değil, taşınan şeyin
  kendisi.

### 4.2 Hemen (kod hazır, ölçüm bekliyor — docs/38 §11)

1. `scripts/profile_loader.py` ve `training.profile_steps` ile gerçek dökümü almak (yükleyici mi, ana süreç mi).
2. `dataset.cache_compress: fast` (açma 3,1 → ~1,5 s; dosya 1,5 kat), `training.diag_metrics_every_n_steps: 10`,
   ilerleme çubuğu seyrek, `training.gpu_prefetch: true`, `prefetch_factor: 1`, doğrulama işçilerini ayırmak.
3. Boyuta göre gruplama + gerçek batch (küçük örnekler birlikte): adım başına sabit yük, DDP'de bekleme azalır.
4. Beklenti (tahmin): GPU kullanımı %40 → %70–80. Bu yol ~8k örnekte yeter; 10k/aile ölçeğinde önbellek ~1,2 TB+
   olur ve tıkanır.

### 4.3 Kalıcı çözüm: operatörleri GPU'da matrissiz kurmak

Fikir: global matrisleri hiç saklamamak ve taşımamak. HCurl p2 için her şey eleman-yereldir:

- Referans eleman baz fonksiyonları ve rot'ları 14 kuadratür noktasında **sabit** tablolar (20 × 14 × 3).
- Geometri başına gereken: eleman → DOF haritası (+ yön işaretleri), kuadratür noktalarında Jacobian (eğri eleman:
  nokta başına 3 × 3) ve ağırlıklar. Tet başına ~0,6 kB (float32); global matrislerde ~26 kB: **~40 kat küçük**.
  23k tetlik örnek ~15 MB; 8k örnek ~140 GB (bugünkü sıkıştırılmış önbellek ~1,2 TB olurdu). İç elemanların çoğu düz
  (tek Jacobian yeter), yalnız duvar elemanları eğri: bunu kullanınca ~5 kat daha küçülür (~30 GB, RAM'e sığar).
- GPU'da: DOF'ları elemanlara topla → Piola dönüşümü (kovaryant: J⁻ᵀ; rot: J/det J) → kuadratür noktalarında alan
  ve rot → ağırlıklı çarpım → geri dağıt. `M·V`, `K·V`, C ve P'nin karşılığı, G (gradyan haritası) dahil.
- Kazanç: CPU kurulumu, önbellek, açma, collate, `/dev/shm`, host→GPU kopyası **hepsi kalkar**; yükleyici yalnız
  küçük dizileri okur. İzdüşüm için G bedavaya gelir (diverjans cezası, izdüşümlü ince ayar). Yön artırımı ve
  tahmin hızı (§2.1) da buradan çıkar. FP64 seyrek çarpım yerine yoğun küçük einsum: yeni GPU'larda daha iyi ölçeklenir.
- Zor kısım ve risk: NGSolve'un yüksek mertebe HCurl bazının **yön/işaret kuralları** ve eğri eleman geometrisi.
  Güvenli yol: bazı yeniden türetmek yerine NGSolve'dan eleman başına DOF numaraları ve referans→fiziksel dönüşümü
  dışa aktarmak; kabul ölçütü: `M·V`, `K·V`, C·ψ sonuçları NGSolve matrisleriyle ≤ 1e-12 bağıl fark (her ailede,
  deforme ve onarılmış ağlar dahil), Ritz frekansları aynı. Tahmini iş: 2–3 hafta + inceleme.
- Ara adım (1 hafta, daha az risk): yalnız **aktarımı** küçültmek — K ve M'yi saklamak yerine B (DOF → kuadratür
  değerleri) ve B_rot'u NGSolve'dan bir kez alıp `M·V = Bᵀ(w ⊙ BV)` olarak uygulamak. Boyut kazancı az, ama doğrulama
  altyapısını ve GPU tarafını hazırlar.

### 4.4 Ölçek büyüyünce

- Çok GPU (DDP): gradyan iletişimi önemsiz (model küçük); boyuta göre gruplama şart. Proje QOS'u gerekir.
- Veri biçimi: aile başına tek H5 yerine parçalı (shard) + bellek eşlemeli diziler; matrissiz yolda örnek ~15–25 MB.

## 5. Veri planı

1. **Eldeki 13 × 1024** ile ölçekleme deneyi (§3.1). Önce bu; yeni üretim eğrinin eğimine göre.
2. **Çeşitlilik, miktardan önce:** `freeform` ailesi (hiç üretilmedi); hücre sayısı aralığını genişletmek
   (elliptical 1–5 → 1–9); bileşik/rastgele birleşim ailesi; OOD kümesini yeniden tanımlamak (yakın OOD:
   pillbox_port, box; uzak OOD: junction, coax_qw). Yeni aile = üretici değişikliği → sahip onayı, `/new-family`.
3. **Aile-dışarıda (leave-one-family-out) testleri:** OOD davranışını eğitim ailelerinin içinden ölçmek.
4. **Miktar:** eğim destekliyorsa aile başına 4k → 10k. 130k geometri ≈ 20 bin çekirdek-saat (550 çekirdek-s/geometri),
   ~1,1 TB. Scratch kotası ve yedekleme planı önceden.
5. **Etiket kalitesi:** composite `max_rounds` (%14) için bir tur daha; keskin kenar yedeğini azaltacak CAD işleri;
   `qoi_ok` düşük ailelerde başlangıç ağı kalitesi.
6. **Çok sadakatli veri (sonra):** az sayıda p3 / sıkı toleranslı örnek, ağırlıklı (docs/34 altyapısı var).

## 6. Aşamalar ve karar noktaları

| aşama | süre (tahmin) | iş | çıkış ölçütü |
|---|---|---|---|
| **0. Teşhis** | 1 hafta | veri yedeği; pilot checkpoint ile `eval_by_family` (ID test + OOD + nokta hacmi kapalı); başlangıç ağı kayması testi; yükleyici / adım profili (7b) | hangi hatanın nereden geldiği sayılarla; veri yolu ayarları seçildi |
| **1. İlk gerçek model** | 2–3 hafta | 8 aile × ~8k, `base`, ≥ 150k adım; kayıp denemeleri (enerji ağırlığı, diverjans cezası, son ince ayar); izdüşümlü karşılaştırma kolu | ID test: ε_A ≤ 0,12, frekans hatası ≤ %2 |
| **2. Veri yolu** | 3–4 hafta (1 ile paralel) | matrissiz GPU operatörleri (önce B-yolu, sonra Jacobian-yolu); token azaltma; karışık hassasiyet | GPU kullanımı ≥ %85; örnek başına ≤ 1,2 × yalın GPU adımı; tahmin ≤ 2 s (ağ hariç) |
| **3. Ölçekleme** | 4–6 hafta | §3.1 tablosu; eğime göre veri üretimi (çeşitlilik + miktar); mimari kaldıraçlar gerekiyorsa | ID frekans hatası ≤ %0,5 (ε_A ≤ 0,07); aile-dışarıda hata ≤ 3 × ID |
| **4. Güvenilir tahmin** | 4 hafta | tahminde uyarlamalı tur (seçenek b); hibrit: model alt uzayı + LOBPCG; QoI değerlendirmesi; CST karşılaştırması | kullanıcıya hata göstergesi; FE'ye göre ≥ 10 kat (yaklaşık) ya da ≥ 3 kat (kesin, hibrit) |
| **5. Yayın ve araç** | sürekli | kıyas modelleri (kaba FE eşit sürede, parametre-GP, GINO/Transolver doğrudan regresyon); makale; arayüz | — |

**Durdur / yön değiştir ölçütleri**

- Aşama 1 sonunda ε_A > 0,18 ve kayıp eğrisi düzse: veri/modeli büyütmeden önce mimariye (§3.4 yerel katmanlar) geç.
- Aşama 3'te veri eğimi α < 0,15 ve model eğimi de düzse: saf tahmin yerine hibrit çözücüyü (§2.1) ana ürün yap.
- Başlangıç ağı testinde hata ince ağdakinin > 2 katıysa: iki-ağlı eğitim ve nokta hacmi özelliğinin çıkarılması
  Aşama 1'e alınır.

## 7. Kaynak ihtiyacı (tahmin)

| kalem | miktar |
|---|---|
| Aşama 1 eğitimleri (5–6 koşu) | ~150 GPU-saat (G4/A100 sınıfı) |
| Ölçekleme tablosu | ~150–200 GPU-saat |
| Aşama 3 büyük koşular (N ≥ 30k) | 4–8 GPU × günler; proje QOS'u şart |
| Veri üretimi 10k/aile | ~20 bin çekirdek-saat, ~1,1 TB |
| Depolama | kalıcı ≥ 2 TB (scratch dışı) |

Öncelik: TRUBA proje hesabı / GPU QOS başvurusu (danışman üzerinden). O gelene kadar Colab G4 (≈ 0,9 $/saat) ile
Aşama 0–1 yürütülebilir.

## 8. Hemen yapılacaklar (sıra)

1. v3 verisini yedekle; `status` + özet tablosu.
2. PR #11'i birleştir; pilot checkpoint ile `eval_by_family` (ID test, OOD, `--ablate point_volume`).
3. Not defteri 7b: yükleyici ve adım profili → §4.2 ayarlarını seç.
4. Aşama 1 koşusu: 8 aile × 1024, `base`, toplam adım bütçesiyle; yanında enerji ağırlığı denemesi.
5. Matrissiz operatör işini başlat (B-yolu; kabul testi NGSolve ile 1e-12).
6. TRUBA proje QOS başvurusu.
