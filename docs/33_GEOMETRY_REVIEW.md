# 33 — Geometri aileleri: literatürle karşılaştırma ve öneriler

> Tarih 2026-10-22, dal `claude/field-model`.
>
> **Kaynak.** Dört araştırma ajanının raporları ve bu oturumdaki kod okuması ile ölçümler. Ajanların alanları:
> - eliptik SRF kaviteler;
> - düşük-β TEM kaviteler ve DTL;
> - normal iletken ve özel kaviteler;
> - makine öğrenmesi ve hesaplamalı tasarım.
>
> Ham raporlar, kaynak listeleri (URL'leriyle) ve betikler `docs/research/2026-10_geometry/` altında.
>
> **Erişim sınırı.** Ortamın ağ politikası arXiv, JACoW, OSTI, CERN CDS, APS ve INSPIRE'ı engelledi; hiçbir
> makalenin tam metni okunamadı.
> - Literatürden gelen her sayı bir arama motoru alıntısıdır; raporlarda öyle işaretli.
> - Bir değeri hedef ya da doğrulama referansı yapmadan önce tam metinden kontrol edin.
> - Bu sınırdan etkilenmeyenler: kod okuması, bizim dağılım ölçümlerimiz ve 3. ajanın eksenel simetrik
>   çözücüyle yaptığı hesaplar. O çözücü analitik pillbox'ta frekansı 3·10⁻¹¹, R/Q'yu 3·10⁻⁴ doğrulukla
>   veriyor.
>
> Bu belge bir öneri listesidir. Üretici ve etiket değişiklikleri sahibin onayıyla yapılır (AGENTS.md).

## 1. Özet

1. **Etiketlerde üç tanım sorunu var.** Üçü de büyük TRUBA üretiminden önce ele alınmalı (§2).
   - ridged_box'ın ışın ekseni kavitenin dışında kalıyor; R/Q, T ve Epk/Eacc anlamsız.
   - Hızlandırma uzunluğu boruları da kapsıyor. Epk/Eacc ve Bpk/Eacc ortanca 1,2–2,4 kat büyük çıkıyor (en
     kötü %5'te 4 katı aşıyor) ve boru uzunluğuna bağlı.
   - Bütün büyüklükler β = 1'de; düşük-β ailelerde R/Q ve T tasarım değerleriyle karşılaştırılamıyor.
2. **Üreticide bir hata var** (doğrulandı, §3.1). Eliptik yarım hücre çözücüsünün toleransı, geçerli rastgele
   çekilişlerin %7,4'ünü ve yayınlanmış iki uç hücreyi reddediyor.
3. **Aileler gerçek tasarımları kısmen kapsıyor.**
   - Eliptikte elips oranları TESLA'ya ortalı: tam tanımlı 9 gerçek orta hücreden 8'i aralıkların dışında.
     Re-entrant duvarlar hiç üretilmiyor; düşük kayıplı LL, ICHIRO ve NLSF hücreleri tam red sınırında.
   - reentrant ağırlıkla dar aralıklı klistron tipi hücreler üretiyor; depolama halkası hücreleri aralığın
     dışında.
   - spoke ve dtl yüksek ve düşük β uçlarını kaçırıyor.
4. **Çok hücreli OOD testi hızlandırma modunu sormuyor** (§3.9). Model 6 mod öğreniyor; 7–9 hücreli
   kavitelerde çalışma modu (π) 7.–9. mod. Yapılmış linac kavitelerinin çoğu 6, 7 ya da 9 hücreli.
5. **ridged_box gerçek bir kaviteye karşılık gelmiyor.** RF-dipol crab kavitesinin dejenere bir hâli ve en
   pahalı ailemiz (38–89k tet). Yerine parametrik bir RF-dipol ailesi önerilir.
6. **En yaygın düşük-β kavite olan QWR eğitimde yok.** Yalnız keskin kenarlı bir OOD sondası var.
7. **Eklenmesi ucuz ve değerli aileler:** QWR, DTL birim hücresi, 1,6 hücreli RF tabancası, RF-dipol. Sonra
   büyük açıklıklı eliptik halka hücreleri, DQW, disk yüklü duran dalga kesiti ve basık hücreler gelir.
8. **Sentetik aileler literatürde yok.** composite'in çoğu gerçek bir hücre ve gerçekçi eklentilere
   (kuplör, tuner, dalga kılavuzu) dönüştürülmeli. Eliptikteki içi boş saplar da kuplör değil.
9. **Deformasyon imalat toleransının ~40 katı** ve TRUBA ile arayüzde örneklerin yarısına uygulanıyor. Bir
   çeşitlilik aracı; imalat hatası için ayrı bir ön ayar gerekir.
10. **Doğruluk.** Nihai tasarım 10⁻⁵–10⁻⁴ frekans doğruluğu ister; modelin p2 tabanı %0,05–0,9. Model
    çözücünün yerini alamaz, hibrit mod bu yüzden önemli. Tarama için hedef, büyüklüklerde ≤ %2.
11. **Literatürdeki yerimiz.** Ağ girdisi alan, 3B alan tahmin eden ve birden çok aile kapsayan bir RF kavite
    ML çalışması bulunamadı. Bulunan çalışmalar tek aile, düşük boyutlu parametre ve ≤ 4000 geometriyle
    çalışıyor. Kıyas için basit bir parametrik temel model önerilir.

## 2. Etiketlerdeki sorunlar (TRUBA üretiminden önce)

**Eğitime şu an etkisi yok:** field3d'de QoI kaybı kapalı (`qoi_weight = 0`).

**Etkiledikleri:** H5'te saklanan büyüklükler, test ölçütleri, web arayüzü ve ileride açılacak bir QoI kaybı.
500 000 çekirdek-saatlik üretim bu tanımlarla yapılırsa, düzeltmeler sonradan ancak saklanan p2 alanlarından
yapılabilir. Bunun doğruluğu proj_err kadardır (%0,2–0,5).

### 2.1 ridged_box: ışın ekseni kavitenin dışında (hata, kodda doğrulandı)

- **Hangi eksen kullanılıyor.** `qoi.operators.beam_axis` yalnız hwr için eksen tanımlıyor. Diğer bütün
  aileler (0, 0, 0)'dan geçen z doğrusunu kullanıyor (`field_labels._beam_axis` → `highorder.mode_qoi`).
- **ridged_box'ta bu ne demek.** `build_ridged_box` kutuyu [0, a] × [0, b] × [0, d] aralığına yerleştiriyor ve
  bütün kenarları pahlıyor. z doğrusu x = y = 0 kenarına denk geliyor; pahlanmış kenar kavitenin dışında.
- **Sonuç:**
  - Eksen kirişi ≈ 0 ve V ≈ 0; R/Q ≈ 0.
  - T ve Epk/Eacc tanımsız (0/0).
  - Borular x yönünde, (y = b/2, z = z_c) doğrusu boyunca uzanıyor.
- **Düzeltme.** Boru varsa eksen boru ekseni olmalı. Boru yoksa ışın büyüklükleri NaN yazılmalı. Aile RF-dipol
  ile değiştirilirse (§3.4) transvers R/Q da gerekir.
- **OOD ailelerde aynı tür sorun:**
  - coax_qw'da eksen koaksiyel eksen boyunca; gerçek bir QWR'da ışın enine geçer.
  - box'ta z doğrusu kutunun keskin bir kenarı.

### 2.2 Hızlandırma uzunluğu boruları kapsıyor

- **Bugünkü tanım.** `mode_qoi`'de E_acc = V / L ve L, eksenin Ω içindeki kirişi; borular dahil (docs/24 §1.3).
- **Literatürdeki tanımlar:**
  - eliptik: N·β_g·λ/2, yani hücre uzunluklarının toplamı;
  - tek hücre ya da nose-cone: hücre ya da aralık uzunluğu;
  - iki aralıklı düşük-β yapılar: βλ.
- **Şişme katsayısı.** L_kiriş / L_standart, üreticinin kendi örnekleyicileriyle. pillbox_pipes ve reentrant için
  200 000, eliptikte her durum için 5000 çekiliş (`docs/research/2026-10_geometry/scripts/chord_inflation.py`):

| aile | ortanca | %5–95 |
|---|---|---|
| pillbox_pipes | 2,27 | 1,51–4,31 |
| reentrant | 2,35 | 1,62–3,83 |
| elliptical, 1 hücre | 2,19 | 1,78–3,24 |
| elliptical, 2 hücre | 1,60 | 1,40–2,10 |
| elliptical, 5 hücre | 1,24 | 1,16–1,45 |
| elliptical, eğitim karışımı (1–5 hücre) | 1,60 | 1,22–2,74 |

- **Örnek geometride** (pillbox_pipes, docs/31 §3.6): saklanan Epk/Eacc 6,59, hücre uzunluğuyla 1,96 olur.
  Bpk/Eacc 7,76'dan 2,30'a iner.
- **Etkilenenler:** Epk/Eacc, Bpk/Eacc, E_acc.
- **Etkilenmeyenler:** V, R/Q, R_sh, T, G, Q0. V'nin bütün eksen boyunca integrali standarttır.
- **Düzeltme kolay.** E_pk ve V değişmediği için Epk/Eacc_std = Epk/Eacc × L_std / L_kiriş tam olarak
  doğrudur. L_std aile parametrelerinden (`geom_params`) gelir, kiriş de saklanan mesh'ten.

### 2.3 β = 1

- **Tasarım pratiği.** Bütün büyüklükler β = 1'de. Düşük-β yapılarda tasarımcılar R/Q, T ve E_acc'yi β_opt'ta
  verir: geçiş zamanı katsayısının en büyük olduğu β. Çoğu zaman yanında bir geometrik β_g de verilir.
- **Etkilenen aileler:** hwr, spoke, dtl ve eliptiğin %25'lik düşük-β hücreleri (β 0,5–0,95). dtl'de hücre
  uzunluğundan β ≈ 0,2–0,35 çıkıyor (tahmin).
- **Etkinin büyüklüğü.** β_opt ≈ 0,1 olan iki aralıklı bir yapıda β = 1'deki R/Q, tasarım değerinin yaklaşık
  %2–3'ü (ince aralık kestirimi, 2. ajan).
  - Bu örnekler QoI ölçütlerindeki `rq_floor` (%1) süzgecine takılabilir; kontrol edilmeli.
- **Tek aralıklı hücrelerde** (reentrant, pillbox) β = 1 elektron ve pozitron için doğru. Proton ve klistron
  hücreleri ise β < 1'de değerlendirilir.

### 2.4 Tasarımcıların karşılaştırdığı ama saklamadığımız büyüklükler

- **Çok hücreli eliptik:** hücreler arası bağlaşım k_cc, alan düzlüğü, hücre başına R/Q, G·R/Q.
- **Dipol modlar ve crab kaviteler:** transvers R/Q (Panofsky–Wenzel), tanımı belirtilerek.
- **DTL:** ZT², yani birim uzunluk başına etkin şönt empedansı.
- **Düşük-β:** β_opt ve β_g.

### 2.5 Öneri: tanımdan bağımsız ham değerleri saklamak

- **Eksendeki alan.** Her mod için eksen boyunca karmaşık E_z(z) örnekleri. `mode_qoi` bunları zaten 2001
  noktada hesaplıyor; TRUBA'nın K = 10'u için complex64 ile örnek başına ~0,16 MB.
- **Ham büyüklükler:** U, P_c, V, E_pk, B_pk; aile parametrelerinden L_std ve β_g.
- **Ne kazanılır.** Her β, her L tanımı ve her eksen seçimi yeniden çözüm yapmadan sonradan hesaplanabilir.
- **ridged_box ekseni** de düzeltilir (§2.1).
- **Mevcut H5'ler:**
  - §2.2'deki yeniden ölçekleme tam doğru.
  - β'ya bağlı büyüklükler p2 alanlarından, proj_err doğruluğunda (%0,2–0,5) hesaplanabilir.
- **Maliyet:** ihmal edilebilir; yeni çözüm gerekmiyor.

## 3. Aile aile değerlendirme

| aile | gerçek karşılığı | kapsadığı | en önemli eksik | öneri | maliyet |
|---|---|---|---|---|---|
| elliptical | TESLA/ILC/XFEL, LCLS-II, SNS, ESS, PIP-II, CEPC, FCC-ee | Ri/Req ve L/Req gerçekçi; TESLA; β 0,5–1; 1–5 hücre | elips oranları dar (9 gerçek orta hücreden 8'i dışarıda); re-entrant duvar reddediliyor; çözücü toleransı geçerli çekilişlerin %7,4'ünü atıyor; 6–9 hücre yalnız OOD'de | çözücüyü düzelt, aralıkları genişlet, hafif re-entrant duvara izin ver, çapaları düzelt | ucuz |
| pillbox_pipes | tabanca hücreleri, TM110 deflektörler, disk yüklü hücreler | yayınlanan pillbox tipi hücreler aralıkta | kalın tam yuvarlak iris, çok hücre, katot duvarı | küçük ayar (rp/R 0,45'e kadar); ayrı tabanca ve disk aileleri (§4) | ucuz |
| reentrant | depolama halkası, klistron/IOT, proton hücreleri | dar aralıklı klistron tipi (%71) | depolama halkası bölgesi dışarıda | Lc/R ≤ 1,6, g/Lc ≤ 0,75 | ucuz |
| ridged_box | yok (dejenere RF-dipol) | — | gerçek tasarım değil; en pahalı aile; eksen hatası | RF-dipol ailesiyle değiştir, eskisi OOD | orta |
| composite | yok | ucuz çeşitlilik | literatürde karşılığı yok | çoğunu gerçek hücre + gerçekçi eklentiye çevir | ucuz |
| hwr | FRIB β 0,29 / 0,53, IFMIF, PIP-II | orta ve yüksek β oranları | ince düşük-β HWR, ışın deliğinde racetrack iç iletken | aralığı uzat; racetrack kesit | orta–pahalı |
| spoke | SSR1, SSR2, ESS çift spoke, ANL üçlü spoke | β ≈ 0,2–0,3 tek spoke | şekilli uç duvarlar, uzun tanklar, konik spoke, küçük delik | uç duvar, uzunluk, konik spoke | ucuz–orta |
| dtl | Linac4, J-PARC, ESS, SNS | drift tüpü / tank oranı | düşük-β hücreler, uçta yarım drift tüpü, yüz açısı | dtl_cell birim hücre ailesi; uçlara yarım tüp | ucuz |
| coax_qw (OOD) | QWR | — | gerçekçi QWR eğitimde yok; eksen yanlış yönde | eğitime gerçekçi `qwr` ailesi; coax_qw OOD kalsın | orta |
| box, junction, pillbox_port (OOD) | analitik ve topoloji testleri | — | pillbox_port'un portları sönümleyici bölgesine hiç girmiyor | OOD kalsın | ucuz |

### 3.1 elliptical

Kaynak: 1. ajanın raporu (§3–5) ve bu oturumda bizim kodumuzla yapılan kontroller.

**Oranlar.**
- **Ri/Req ve L/Req, β = 1 için gerçekçi.**
  - Iris ve ekvator yarıçapı bilinen bütün β = 1 orta hücreler 0,25–0,42 içinde. Yalnız C3795 (0,421)
    sınırda.
  - 1,3 GHz hücreleri yalnız 0,30–0,35 bandını kullanıyor. Üst kısım (0,38–0,42) halka, çarpıştırıcı ve 3,9 GHz
    hücrelerine ait.
- **Elips oranları dar ve TESLA'ya ortalı.** Tam tanımlı 9 β = 1 orta hücreden yalnız TESLA altı aralığın
  hepsinin içinde. Başlıca kaçanlar:
  - B/A < 0,85: TESLA sonrası düşük kayıplı ailenin tamamı (LL 0,729, ICHIRO 0,684, NLSF 0,665, NLSF-RE 0,720,
    Cornell ERL 0,797);
  - LL ve ICHIRO'da a/L 0,132 < 0,15 ve A/L 0,867 > 0,85;
  - FCC-ee 800 MHz'te b/a 0,765 < 1 ve a/L 0,323 > 0,30;
  - FCC-ee 400 MHz hücresi bambaşka bir köşede: A/L 0,393, B/A 1,79, a/L 0,568.
- **Örneklem küçük.** 1,3 GHz şekillerine yatkın ve 9 hücreden 6'sı yalnız açık kaynak bir tasarım kodunda
  var. Sonuç "TESLA sonrası tasarımların çoğu" diye okunmalı, bir nüfus istatistiği diye değil.
- **β < 1 dalı (çekilişlerin %25'i).** L ∝ β fiziği doğru. Ama SNS, ESS ve PIP-II hücre boyutları
  bulunamadı, aralıklar doğrulanmadı.

**Duvar açısı** (θ: duvar doğrusunun ışın eksenine açısı; 90° dikey, > 90° re-entrant).
- **Bizim dağılımımız** (kabul edilen orta hücreler, 3000 çekiliş, çapalar dahil): %1'lik dilim 61,5°, %10
  68,7°, ortanca 78,9°, %90 87,1°, %99 89,7°. TESLA çapası 76,7°.
- **Gerçek β = 1 hücreler:** re-entrant olmayanlar 67,6–89,9°; re-entrant'lar dahil 101,1°'ye kadar
  (NLSF-RE). Cornell'in RE hücreleri de re-entrant.
- **Sınırda kalanlar.** LL, ICHIRO ve NLSF 89,9°'de, `ilc_ll` çapamız tam 90,0°'de (L = A + a); yani hepsi red
  sınırının üstünde.
  - Çapanın ±%10 komşuluğundaki çekilişlerin %33'ü re-entrant diye reddediliyor. Komşuluk tek taraflı
    örnekleniyor.
- **RE şekli Pareto cephesinin düşük Bpk köşesinde.** ILC TDR tablosunda Bpk/Eacc 3,54 mT/(MV/m), G·R/Q
  41 208 Ω² (4. ajan). Shemelin de re-entrant hücreleri yararlı buluyor.

**Reddedilen çekilişler.** Rastgele β = 1 yarım hücrelerin yarısı `_valid_half` tarafından reddediliyor.
Yerlerine yeni çekiliş alındığı için örnek kaybı yok, ama dağılım daralıyor. Bizim kodumuzla 4000 çekilişin
dökümü (bu oturumda):

| sonuç | pay |
|---|---|
| kabul | %50,0 |
| çakışma kuralı (Riris + b ≥ Req − B) | %26,7 |
| çözücü toleransı (geçerli duvar var) | %7,4 |
| re-entrant ya da teğet yok | %15,9 |

- **Çözücü toleransı hatası** (1. ajan buldu, bu oturumda doğrulandı).
  - `half_cell_wall` teğeti `least_squares`'in varsayılan toleranslarıyla (10⁻⁸) çözüyor. Kalan artık
    1–3·10⁻⁹, kabul eşiği 10⁻⁹.
  - Aynı başlangıç ve aynı geçerlilik testleri `xtol = ftol = gtol = 1e-15` ile bu hücreleri kabul ediyor.
  - Reddedilenlerin duvar açıları sıradan (%5/%50/%95: 68,6° / 78,8° / 88,2°). Kayıp çoğunlukla boşa
    çekiliş, hafif bir yanlılıkla.
  - Yayınlanmış iki uç hücre de bu yüzden reddediliyor: Cornell ERL sağ uç hücresi (θ 84,9°) ve FCC-ee 800
    MHz C3795 uç hücresi (θ 67,6°).
- **Çakışma kuralı gereğinden katı.**
  - FCC-ee 400 MHz C3794 orta hücresini çakışma kuralı reddediyor. Kural kaldırılsa da tek başlangıç noktası
    ([π/2, 0]) teğeti bulamıyor.
  - Başka başlangıç noktaları bizim testlerimizden geçen tekdüze bir duvar buluyor (θ 72,8°).
- **Etkisi** (1. ajan). Üç değişiklik birlikte bugünkü aralıklarda kabulü %50,1'den %75,9'a çıkarıyor:
  - tolerans düzeltmesi;
  - çakışma kuralı yerine "duvar boyunca r tekdüze" testi;
  - ağ korumasıyla 105°'ye kadar re-entrant duvar.

  Kabul edilenlerin %5,4'ü re-entrant.
- **Düzeltmenin maliyeti.** Düzeltme ucuz, ama hangi çekilişlerin kabul edildiğini değiştiriyor. Aynı tohum
  başka geometri üretir; sahibin onayı gerekli.

**`ilc_ll` çapası iki tasarımı karıştırıyor** (mm; yayınlanan değerler Juntong ve Jones, arXiv:0910.1305
Tablo 1):

| hücre | Ri | Req | L | A | B | a | b |
|---|---|---|---|---|---|---|---|
| LL (ILC low-loss) | 30 | 98,58 | 57,7 | 50,052 | 36,50 | 7,6 | 10,0 |
| ICHIRO (KEK) | 30 | 98,14 | 57,7 | 50,052 | 34,222 | 7,6 | 9,945 |
| bizim `ilc_ll` | 30 | 98,6 | 57,7 | 50,1 | 34,2 | 7,6 | 10,0 |

- Req ve b LL'den, B ICHIRO'dan.
- Hiçbir etiket yanlış değil: çapalar ±%10 bozuluyor ve etiketler FE çözümünden geliyor. Ama çapa LL tasarımı
  değil. Koddaki "doğrulanmadı" notu bu değerlerle değiştirilebilir.

**Uç hücreler.**
- **Uç irisi.** Orta irisin ×1,0–1,35'i olan aralığımız gerçek uçları kapsıyor: TESLA ×1,114, Cornell ERL
  ×1,057 ve ×1,114, C3795 ×1,111, 3,9 GHz ×1,333.
- **İris elipsi.** ±%15'lik pertürbasyon iris elipsi için dar. Gerçek uç hücreler a'yı ×0,57–0,83, b'yi
  ×0,52–0,71 küçültüyor. Örneğin TESLA'nın Aune tablosundaki uç kupası 1'de a 12 → 10 ve b 19 → 13,5 mm.
- **Özdeş uçlar.** Kavitelerimizin %40'ında uç yarım hücreler orta hücreyle aynı. Bu meşru (TESLA'nın uç
  kupası 2 A = B = 42 mm'yi koruyor), ama tamamen özdeş bir zincir özel bir durum. CEPC'nin ilk tasarımında
  özdeş yarım hücreler tuzaklanmış modlar üretmişti.

**Borular ve kapaklar.**
- **TM010 alanı kapakta.** TESLA boyutunda (rp 39 mm) iris değerinin %7,5'i (1,2 rp) ile %0,45'i (2,5 rp)
  arasında. TE111 bandında %23 ile %4,9 arasında.
- **Kesim üstündeki modlar.** Büyük açıklıklı tasarımlarda saklanan modların bir kısmı borunun TE11
  kesiminin üstünde kalıyor (FCC-ee 400 MHz borusunda 1,46 f0). Kapalı kapakla bunlar kavite modu değil, boru
  rezonansı.

**Kuplör portları.**
- **Bugünkü saplar kuplör değil.** İçi boş saplar TESLA boyutunda TE11 kesimini 2,9–6,9 f0'a koyuyor; saklanan
  her mod onlarda sönümlü. Anteni sökülmüş bir port gibi: eksenel simetriyi bozan küçük bir pertürbasyon.
- **Bedelleri var.** Port birleşimindeki pah eğri mesh'i katlıyor; bu örnekler keskin kenara düşüyor ve NaN
  tepe değeri saklıyor (docs/30).
- **Kapalı kapak altına anten konmamalı.** Kapaklı bir koaks hattı λ/4 rezonatörüdür. Bizim sap
  uzunluklarımızla 0,77–1,48 GHz verir; 1,3 GHz bandının üstüne gerçekte olmayan modlar ekler.
- **Gerçek yerleşim.** Bulunan bütün SRF eliptik tasarımlarda kuplörler ışın borularında. Hücre gövdesinde
  radyal port NC HOM sönümlü kavitelere özgü; pillbox_port bu yüzden geçerli bir simetri testi, ama SRF
  eliptiği temsil etmiyor.
- **Qext gerekirse** Balleyguier yöntemi: kuplör hattının iki farklı sonlandırmasıyla iki kapalı özdeğer
  çözümü. Simetrik problem ve Ritz üst sınırları korunur.

**Öneriler** (hepsi üretici çıktısını değiştirir; sahibin onayı gerekli):
1. **Çapalar.** `ilc_ll`'i yayınlanan LL orta hücresiyle değiştirmek, ICHIRO'yu üçüncü çapa olarak eklemek.
   Maliyet yok.
2. **Hafif re-entrant duvarlara izin vermek (θ ≤ 100–105°).**
   - Tekdüze-z testi yerine üç test: duvar boyunca r tekdüze, kendini kesmiyor, iris yayından sonraki en küçük
     z ≥ 0,05 Req (komşu hücreler arasındaki metal ağın yarısı).
   - Gerçek L − A değerleri: LL 7,65 mm, NLSF-RE 8,7 mm, TESLA 15,7 mm.
   - Profil tek bir düzgün spline olarak kalıyor; yeni kenar ya da pah yok.
3. **Yarım hücre çözücüsü.** Toleransları 10⁻¹⁵ yapmak (ya da çok başlangıçlı çözüm) ve çakışma kuralını r
   tekdüzeliği testiyle değiştirmek.
4. **Elips aralıklarını genişletmek:** B/A 0,6–1,3, a/L 0,12–0,35, A/L 0,6–0,9, b/a 0,65–1,9.
   - Yeni kurallarla kabul %65,3. Kabul edilenlerin %13,9'u re-entrant; θ %1/%50/%99: 60,9° / 78,8° /
     102,2°. Bugünkü kurallarla aynı aralıklarda kabul %41,8 (1. ajan).
   - Fillet tabanıyla tutarlı kalmak için iris ucu yarıçapına koruma: a²/b ≥ 0,05 Req. Gerçek hücrelerde
     0,059 (LL, ICHIRO) ile 0,074 (TESLA, ERL) Req arası. Bugünkü aralıklar 0,036 Req'e, genişletilmişler
     0,028 Req'e izin veriyor; küçük yarıçaplar eğri mesh'i 30k sınırına iter.
5. **Uç hücreler.**
   - TESLA çapasında yayınlanan, birbirinden farklı uç kupaları 1 ve 2.
   - Rastgele uçlarda a ve b ×0,5–1,05, A ×0,9–1,2, B ×0,85–1,4. L ±%10 kalsın, uç irisi ×1,0–1,35 kalsın.
6. **Büyük açıklıklı halka ve çarpıştırıcı alt ailesi.**
   - Parametreler: 1–2 hücre (800 MHz'te 5), Ri/Req 0,38–0,47, B/A 1,8'e kadar, a/L 0,6'ya kadar, θ 65–90°,
     uzun borular.
   - KEKB, CESR-B, CEPC, FCC-ee ve LHC tipi kaviteleri kapsar. 3. öneriye bağlı; kesim üstündeki modlar
     yüzünden 7. öneriyle birlikte.
7. **Boru kesim frekanslarını saklamak.** Her boru ve port için TE11 ve TM01 kesimi; üstündeki modlar
   işaretlenir. Yalnız üst veri, maliyet yok.
8. **Saplara karar vermek.** Ya eliptikten çıkarmak (port fiziği OOD ailelerde kalır) ya da açıkça
   "simetri bozucu pertürbasyon" olarak belgelemek. Anten eklenmemeli.
9. **β < 1 aralıklarını doğrulamak.** SNS, ESS ve PIP-II hücre geometrisiyle karşılaştırmak; yalnız
   literatür işi.

### 3.2 pillbox_pipes

- **Ölçülen dağılım.** 3. ajan bizim profil kodumuzla 400 çekilişi eksenel simetrik çözdü:
  - R/λ 0,384–0,410;
  - R/Q (linac, β = 1) 91–205 Ω, ortanca 149;
  - G 139–323 Ω;
  - T 0,39–0,93.
- **Kapsadığı.** Yayınlanan pillbox tipi hücreler aralığın içinde: BNL/SLAC tabancası ve SLAC 2π/3 hücresi.
- **Eksikler:**
  - Kalın, tam yuvarlak iris: tabancada ρ/R 0,23, bizde ≤ 0,14 R.
  - Çok hücreli zincirler.
  - Katot duvarı, yani borusuz yarım hücre.
  - Racetrack ve basık hücreler.
  - Depolama halkası tek hücrelerinin büyük açıklığı: rp/R 0,45'e kadar (bugün ≤ 0,35; 1. ajanın tahmini).
- **Boru ucunun etkisi.**
  - Monopol temel mod az etkileniyor: kapakta kalan TM01 genliği 1,5 rp boruda 0,03, 4 rp boruda 10⁻⁴.
  - Dipoller etkileniyor: rp/R 0,35 ve 1,5 rp boruda TM110 benzeri bir modun alanının %15'i PEC kapağa
    ulaşıyor (4 rp'de 6·10⁻³).
  - Öneri: rp/R > 0,25 iken boru en az 2,5 rp olsun. Birkaç örnekte manyetik kapakla kontrol yapılmalı.

### 3.3 reentrant

- **Ölçülen dağılım** (400 çekiliş, eksenel simetrik):
  - R/Q (linac, β = 1) 74–321 Ω, ortanca 161;
  - G 74–200 Ω, ortanca 127;
  - g/λ ortanca 0,074. Çekilişlerin %71'i ≤ 0,1, yani klistron, IOT ve buncher rejiminde.
- **Depolama halkası kaviteleri** (PEP-II, EU/ALBA/ESRF, ALS, KEK PF):
  - R/Q ≈ 223–300 Ω (linac; linac/devre dönüşümü 3. ajanın çıkarımı), G ≈ 167–258 Ω.
  - Çekilişlerimizin %13'ü R/Q ≥ 226 Ω'a, yalnız %4,3'ü hem R/Q ≥ 226 Ω hem G ≥ 175 Ω'a ulaşıyor.
  - Tarama bu bölgeyi Lc/R 1,2–1,4 ve g/Lc 0,4–0,75 aralığına koyuyor. Bizim üst sınırlarımız (Lc/R 1,0,
    g/Lc 0,6) tam bu bölgeyi kesiyor.
- **Öneriler:**
  - Lc/R 0,4–1,6 ve g/Lc 0,15–0,75.
  - İsteğe bağlı, çekilişlerin %30'u kadar bir "depolama halkası" alt modu: Lc/R 1,0–1,6, g/Lc 0,4–0,7,
    rp/R 0,13–0,2, koni 10–30°.
  - Kapasite yüklü (MAX IV tipi) bir alt mod pahalı; düşük öncelik.
- **Maliyet.** Daha geniş aralıklar tet sayısını düşürmeli. Bugün 4 çekilişin 2'si 30k'nin üstünde; bunu
  küçük aralıklar yapıyor.

### 3.4 ridged_box → RF-dipol

- **Neden değiştirilmeli:**
  - Gerçek bir kaviteye karşılık gelmiyor. Tam boy ince çıkıntıları ve geniş aralığıyla, uç boşluğu sıfır
    olan dejenere bir RF-dipol crab kavitesi.
  - En pahalı ailemiz: 38–89k eğri tet. Çekilişlerin çoğu yeniden çekiliyor, aile büyük pahlara kayıyor.
- **Önerilen parametrik RF-dipol ailesi** (HL-LHC, EIC ve ILC tasarımları). Uzunluklar tank genişliği D'ye
  oranla verildi; bir kısmı tahmin.

| parametre | aralık | dayanak |
|---|---|---|
| tank köşe yuvarlatması ρ_t/D | 0,1–0,5 (0,5 = silindir) | LHC yuvarlak kare, PoP silindir |
| tank uzunluğu Lt/D | 1,2–2,0 | PoP 1,60 |
| kutup aralığı g_p/D | 0,2–0,4 | LHC 0,30 |
| kutup uzunluğu / Lt | 0,5–0,9 (sıfırdan büyük uç boşluğu) | tahmin |
| kutup yan açısı | 10–30° | ILC 22,5° |
| kutup ucu genişliği / D | 0,15–0,4 | tahmin |
| ışın boruları | tank ekseninde, rp = (0,4–0,5)·g_p | açıklık ≈ kutup aralığı |

- **Model için anlamı.** Temel mod TM010 benzeri değil, TE11 benzeri saptırıcı bir mod. Model yeni bir mod
  sıralaması görür.
- **Gereken büyüklük:** transvers R/Q.
- **Eski ridged_box** bir OOD varyantı olarak kalabilir.

### 3.5 composite

- **Literatürde karşılığı yok.** Kavite şekil optimizasyonu ve ML çalışmaları gerçekçi profil parametreleri
  kullanıyor.
- **Öneri:** çoğu gerçek bir hücre gövdesi ve 1–3 gerçekçi eklenti olsun. Eklentiler:
  - kuplör portları;
  - kavite içine 0–0,3 R giren silindirik plancır tuner;
  - pick-up portları;
  - dalga kılavuzu uçları.

  %10–20'si saf CSG olarak kalsın; fotonikteki sentetik hücre veri kümeleri gibi bir düzenleyici görevi görür.

### 3.6 hwr

- **Kapsadığı.**
  - ri/Ro aralığı TEM optimumunu (≈ 1/e) içeriyor.
  - IFMIF'in delik oranı (rb/Ro 0,22) aralıkta.
  - Ölçülerimiz orta ve yüksek β HWR'lara karşılık geliyor (FRIB β 0,29–0,53).
  - Işın orta uzunlukta enine geçiyor; bütün gerçek HWR'larda olduğu gibi.
- **Eksikler:**
  - Kaynaklardaki en kalabalık HWR sınıfı olan ince düşük-β HWR'lar (β ≈ 0,1'de Ro/L ≈ 0,1: IFMIF, PIP-II,
    China-ADS).
  - Işın deliğinde racetrack ya da halka biçimli iç iletken.
  - Orta düzlemde kuplör portu.
- **Öneriler** (orta öncelik):
  - Ro/L'yi 0,1'e indirmek pahalı: ince halka tet sayısını artırır.
  - Racetrack kesit CAD'de ucuz, tet sayısında orta maliyetli.

### 3.7 spoke

- **Kapsadığı.** β ≈ 0,2–0,3 tek spoke kaviteler (SSR1 sınıfı). Paralel ya da 90° döndürülmüş ikinci spoke;
  döndürülmüş çift CH'ye benziyor.
- **Eksikler:**
  - **Şekilli uç duvarlar:** re-entrant, dışbükey ya da balon (Spoke040, SSR2, TRIUMF). Bizde düz plaka, bu
    yüzden spoke modu daha az yüklü: bizde D/λ ≈ 0,50, gerçekte 0,53–0,69.
  - **Yüksek β için uzunluk.** β ≥ 0,4 tek spoke'ta Lt/Rt ≈ 1,9; çift spoke'ta 3–4; üçlü spoke'ta 3,7.
    Bizim çift spoke aralığımız 2,2'de bitiyor.
  - **Konik spoke.** Gerçek spoke'larda taban kesiti merkezdekinden farklı; bizimki tek biçimli bir eliptik
    silindir.
  - **Delik.** Gerçek rb/Rt 0,06–0,09, bizimki 0,07–0,23.
  - **Üçlü spoke** yok.
- **Öneriler:**
  - Uç duvar şekli; reentrant'ın burun profili yeniden kullanılabilir.
  - Lt/Rt tek spoke'ta 0,9–2,0, çift spoke'ta 1,5–3,2.
  - Konik spoke (OCC ThruSections).
  - Delik alt sınırı 0,06.
- **Maliyet.** Ailede 4 katlık tet payı var (5–8k).

### 3.8 dtl

- **Kapsadığı.** Rd/Rt, J-PARC (0,25) ve tek hücre örneğiyle (0,27) uyumlu. Linac4 (0,17) sınırın biraz altında.
- **Eksikler:**
  - **Düşük-β hücreler** (Lc/Rt 0,26–0,5): gerçek hücrelerin çoğu bunlar. ESS'nin ilk tankında 61 hücre var.
  - **Uçta yarım drift tüpleri.** Onlar olmadan uç hücreler pillbox gibi davranıyor ve 0-mod düz değil.
  - **Delik ve burun yarıçapları fillet tabanının altında.** Gerçek delik ≈ 0,04 Rt; burun ve köşe
    yarıçapları Rt'nin %0,4–4'ü. Taban ise %5. Bu yüzden Epk muhtemelen düşük çıkıyor.
  - Yüz açısı, hücre uzunluğu artışı (ESS'de hücre başına ~%1,5) ve post kuplörler.
- **Öneriler:**
  - **dtl_cell birim hücre ailesi.** Tek hücre, iki yarım drift tüpü, drift tüpü orta düzlemlerinde PEC uç
    düzlemleri. 0-mod için bu tam doğru, çünkü o düzlemlerde teğetsel E sıfır. Superfish'in tasarım nesnesi
    de budur.
  - Mevcut tanka uçlarda yarım tüp.
  - Rd/Rt alt sınırı 0,15.
- **Bırakılacaklar.** 4 hücreden uzun tanklar sığmaz: 4 hücrelik çekilişlerin ikisi şimdiden 30k'yi aşıyor.

### 3.9 OOD aileler

- **box:** analitik spektrum ve keskin kenar testi olarak kalsın.
- **junction:** topoloji testi olarak kalsın.
- **pillbox_port:**
  - Portların TE11 kesim frekansı ≥ 2,55 f0, gerçek sönümleyici kılavuzlarda fc/f0 1,23–1,34. Bu yüzden
    sönümlü kavite rejimini hiç test etmiyor.
  - Geçerli bir simetri bozma testi, ama SRF eliptik kaviteleri temsil etmiyor: onlarda kuplörler ışın
    borularında (§3.1).
  - İsteğe bağlı olarak geniş portlu bir OOD varyantı eklenebilir.
- **elliptical_long** (6–9 hücre): hücre sayısı genellemesi için kalsın, ama bugünkü hâliyle hızlandırma
  modunu test etmiyor.
  - **Gerçek hücre sayıları.**
    - 9: TESLA, XFEL, ILC, LCLS-II, 3,9 GHz. 7: C100, Cornell ERL. 6: SNS, ESS orta β.
    - 5: ESS yüksek β, PIP-II, FCC-ee 800 MHz, LHeC, CEBAF OC. 2: CEPC. 1: KEKB, CESR-B, FCC-ee 400 MHz.
    - Yapılmış linac kavitelerinin çoğu (6, 7, 9 hücre) yalnız OOD kümesinde.
  - **Mod sayısı.**
    - Model 6 mod öğreniyor (`configs/field_3d.yaml`, `num_field_modes: 6`).
    - Saklanan mod sayısı üreticide varsayılan 6, TRUBA'da 10.
    - N hücreli bir kavitenin TM010 geçiş bandı en alttaki N mod, çalışma modu (π) N'inci mod. N ≥ 7'de bu mod
      modelin tahmin ettiği 6 modun dışında kalıyor.
  - **Bant sıklaşıyor.** π modunun komşusundan ayrımı TESLA benzeri hücrelerle 5 hücrede 2,32 MHz (1,8·10⁻³),
    9 hücrede 0,73 MHz (5,6·10⁻⁴); bağlı osilatör kestirimi.
  - **Öneri:** çok hücreli örneklerde K ≥ N + 2 mod saklamak ve öğrenmek, ya da en azından geçiş bandı
    indeksini etiketlemek. 9 hücreli mesh'ler bugünkü çözünürlükte 30k teti aşar.
- **coax_qw:** keskin kenarlı QWR sondası olarak kalsın. Gerçekçi QWR eğitime eklenmeli (§4).

## 4. Eklenebilecek yeni aileler

| aile | neden | ana parametreler | tahmini tet | öncelik |
|---|---|---|---|---|
| `qwr` | en yaygın düşük-β SRF kavite: FRIB (~100 adet β 0,085), ATLAS, ISAC-II, SPIRAL2, HIE-ISOLDE; asimetrik alan ve yönlendirme | H/D 1,5–3,5 (başlangıç), ri/Ro 0,25–0,45, uçta drift tüpü, enine ışın deliği rb/Ro 0,1–0,15 | 9–27k (hwr gibi) | yüksek |
| `rfd` | ridged_box'ın yerine; HL-LHC, EIC crab | §3.4 | 10–30k (tahmin) | yüksek |
| `dtl_cell` | DTL'nin tasarım nesnesi; 0-mod için PEC uçları tam | tek hücre, iki yarım drift tüpü, isteğe bağlı yarım gövde; Lc/Rt 0,3–1,3 | düşük | yüksek |
| `gun` (1,6 hücre) | standart NC foto-enjektör; 0/π çifti Δf/f 1,2–5·10⁻³, yakın dejenere modlar için keskin test | L_yarım/L_tam 0,6–0,75; L_tam/R 0,7–0,9; a/R 0,25–0,35; iris kalınlığı 0,3–0,5 R | 8–20k | yüksek |
| `dqw` | ikinci HL-LHC crab tipi; hwr ile coax_qw arasında | gövde ~1,23:1:1, iki direk, kapasitif aralık, isteğe bağlı şapka | hwr gibi | orta |
| disk yüklü duran dalga kesiti | en yaygın NC hızlandırıcı yapı; yoğun geçiş bandı | 2–5 periyot, a/b 0,22–0,33, t/b 0,08–0,17, d/b 0,6–1,3 | periyot başına 4–6k | orta |
| basık / racetrack hücre | KEKB crab, LCLS tabancası, TM110 deflektörler; dipol kutuplaşma ayrımı | enine ölçek 0,8–1 ya da racetrack kayması 0–0,2 R | düşük | orta |
| üçlü spoke / CH kesiti | çok aralıklı H-modu | 3 spoke ya da 3–4 çapraz çubuk | orta–yüksek | orta |
| HOM sönümlü hücre (dalga kılavuzlu) | PEP-II, EU, ARES | fc/f0 1,2–1,4; kılavuz 0,8–1 λ uzun olmalı | yüksek; HOM etiketleri PEC kapak artefaktı | yalnız OOD |

**Bırakılacaklar:**
- yapı olarak sığmayanlar: choke-mode hücre, side-coupled / ACS zincirleri, 4-rod crab, RFQ, IH ve tam boy
  DTL / CH tankları;
- kapalı PEC modelde anlamı olmayanlar: ferrit ve SiC soğurucular, anten ve hook kuplörler.

Sebepleri: fillet tabanının çok altındaki ayrıntılar, 30k tet bütçesi ve kapalı PEC modelin dışında kalan
fizik.

## 5. Deformasyon ve imalat hataları

- **TRUBA koşularında açık.** Üreticinin kendi varsayılanı `--deform_prob 0` (kapalı). TRUBA
  (`cluster/truba/config.sh`) ve arayüz ise DEFORM_PROB 0,5 ve DEFORM_MAX 0,5 kullanıyor: veri üretiminde
  geometrilerin yarısı aşağıdaki büyüklükte bozuluyor.
- **Haritanın büyüklüğü.** Yer değiştirme sınırlayıcı kutunun köşegeniyle ölçekleniyor (`deform_map`).
  TESLA boyutunda tek hücre ve 70 mm borularla (köşegen 388 mm), 400 çekilişte ortanca en büyük yer
  değiştirme:

| deform_max | ortanca en büyük yer değiştirme | %5–95 | Req'e oranı |
|---|---|---|---|
| 0,5 (TRUBA ve arayüz varsayılanı) | 8,42 mm | 3,88–15,5 mm | %8,2 |
| 0,05 | 0,84 mm | 0,39–1,55 mm | %0,82 |
| 0,01 | 0,17 mm | 0,08–0,31 mm | %0,16 |

- **Gerçek imalat hataları:**
  - hücre şekli toleransı ±0,2 mm (European XFEL), derin çekmede 0,4 mm;
  - eksantriklik 0,2 mm;
  - ekvator kaynak çekmesi ~0,9 mm (kesilerek düzeltilir);
  - kimyasal aşındırma 0,1–0,2 mm.
- **Sonuç.** Varsayılan değer bir tasarım çeşitliliği aracı: imalat toleransının ~40 katı.
- **İmalat hatası ön ayarı önerisi:**
  - deform_max 0,01–0,02;
  - hücre başına yarıçap ve uzunluk hataları, rijit hücre kaymaları;
  - aynı mesh topolojisinde nominal ve bozulmuş çiftler.
  - Veriye dayalı alternatif: ölçülmüş kavitelerden çıkarılmış Karhunen–Loève modları (kodu açık).
- **Not.** İmalat düzeyindeki ~10⁻³'lük frekans kaymaları modelin tabanında ya da altında; model bunları tek
  başına çözemez.

## 6. Gereken doğruluk ve modelin rolü

- **Nihai tasarım ve imalat** 10⁻⁵–10⁻⁴ frekans doğruluğu ister:
  - European XFEL π modu 1299,7 ± 0,1 MHz;
  - Omega3P'nin hedefi "GHz'te onlarca kHz";
  - FRIB QWR'da 80,5 MHz'te ~6 kHz.
- **Modelin p2 tabanı** model mesh'inde %0,05–0,9 (frekans).
  - Tanım `√(rq_model / λ) − 1`; docs/30 §5'teki değerler bu oturumda bu tanımla yeniden üretildi.
  - Bu, nihai tasarımın istediğinden 5–90 kat kaba. Model çözücünün yerini alamaz. Doğru yol hibrit mod:
    modelin uzayı LOBPCG'ye başlangıç olur (docs/31 §5).
- **Tarama ve Pareto araması.**
  - Frekans hatası yeniden ölçeklemeyle ya da ekvator ayarıyla giderilir. Önemli olan G, R/Q, Epk/Eacc ve
    Bpk/Eacc.
  - Yayınlanan kod-kod farkları %1–8; tasarımlar arası farklar %10–35.
  - Hedef: büyüklüklerde ≤ %2.
- **Çok hücreli alan düzlüğü** hücre frekanslarında 10⁻⁵ ister; model için ulaşılmaz.
- **Etiket çözücümüzün doğruluğu** (pillbox: f 1,4·10⁻⁶, Q0 3,6·10⁻⁵) yayınlanan kod-kod farklarının çok
  altında.

## 7. ML literatüründeki yerimiz

- **Bulunan RF kavite ML çalışmaları** hep düşük boyutlu bir CAD parametre vektörüyle çalışıyor:
  - eliptik yarım hücrenin 7 parametresi, spline ya da NURBS kontrol noktaları, PCA modları;
  - tek aile ve skaler çıktı;
  - en büyük veri kümesi ≤ 4000 geometri.
- **Bulunamayanlar:** mesh girdisi alan, 3B alan tahmin eden ya da birden çok aileyi kapsayan bir RF
  çalışması. Ortak bir veri kümesi ya da benchmark da yok.
- **Analoglar.** Üç boyutlu nöral operatörler (GINO) ~500–600 geometride %7–8 alan hatasına ulaşıyor.
  Fotonikte çok doğruluklu ve önceden eğitilmiş yaklaşımlar yüksek doğruluklu etiket ihtiyacını 10–100 kat
  azaltıyor.
- **Büyük üretimden önce:**
  - aile parametrelerinden büyüklüklere giden basit bir temel model (MLP ya da GP);
  - küçük bir öğrenme eğrisi (300 / 1000 / 3000 örnek).

## 8. Doğrulama seti önerisi

- **TESLA orta hücre ve 9 hücre** (Aune 2000). Uç hücreleri ve boruları dahil tam tanımlı tek tasarım.
  - Hedefler: G 270 Ω, Epk/Eacc 2,0, Bpk/Eacc 4,26 mT/(MV/m), k_cc %1,87.
  - R/Q 518 Ω devre tanımıyla; bizim linac tanımımızla 1036 Ω.
  - Borular ≥ 3 rp olsun: kapakta TM010 alanı iris değerinin ~1,5·10⁻³'ü kalır. Kıyas kapak yerine bağlı
    olmaz.
- **ILC şekilleri.**
  - LL ve ICHIRO orta hücreleri tam (1. ajan): k_cc %1,52, Epk/Eacc 2,36, Bpk/Eacc 3,61 mT/(MV/m), hücre
    başına R/Q 133,7 Ω, G 284 Ω. Uç grupları bulunamadı; tek hücre ya da periyodik π modu olarak kullanılabilir.
  - RE ve LSF'nin boyutları bulunamadı. RE için ayrıca re-entrant duvar desteği gerekli.
- **SLAC 2π/3 hücresi.** Üç periyot, elektrik duvarlı; 3. ajanın yeniden kurulumu 2855,2 MHz verdi
  (yayınlanan 2856).
- **BNL/SLAC/UCLA 1,6 hücreli tabanca.** Yayınlanan tablodan kuruldu; iris kalınlığı alıntıda yok, tam yuvarlak
  19 mm varsayıldı.
  - Hesaplanan modlar 2859,1 ve 2879,0 MHz; yayınlanan 0 modu 2852,5, π modu 2856,0 MHz.
  - Ayrım 19,9 MHz, yayınlanan 3,5 MHz. Varsayılan iris bağlaşımı çok güçlü yapıyor; gerçek çizim
    gerekli.
- **Dauge'nin Fichera köşesi** (λ₁ = 3,21987401386): tekil kenar testi.
- **Düşük-β:** ANL TSR630, SSR1 / SSR2, IFMIF HWR, ISAC-II / SPIRAL2 QWR. Hepsi için tam metin gerekli.
- **Maliyet:** 20–30 geometri, ≤ 15 çekirdek-saat.

## 9. Önerilen sıra ve bütçe

| adım | iş | bütçe |
|---|---|---|
| A | Etiket düzeltmeleri (§2): ridged_box ekseni, ham değerlerin saklanması, L_std; boru kesim frekansları | yok denecek kadar az |
| B | Doğrulama seti (§8) ve TRUBA referans koşuları | < 1 000 çekirdek-saat |
| C | Üretici düzeltmeleri ve ucuz aralık değişiklikleri. Eliptik: yarım hücre çözücüsü ve çakışma kuralı, çapalar, elips aralıkları, hafif re-entrant duvar, uç hücreler. Ayrıca reentrant aralıkları, spoke uzunluğu ve uç duvarı, dtl'ye uçta yarım tüp. Her biri 4 Sobol çekilişiyle tet sayısı ölçülerek | küçük |
| D | Yeni aileler, sırayla: qwr, dtl_cell, rfd (ridged_box yerine), gun; ardından büyük açıklıklı eliptik alt ailesi, dqw, disk kesiti, basık hücre | küçük |
| E | Pilot eğitim ve öğrenme eğrisi; parametrik temel model | eğitim ağırlıklı |
| F | Ana üretim: 10–12 aile × 2–4k geometri | ~5–50k çekirdek-saat (örnek başına 0,15–1 çekirdek-saat; B ölçecek) |

## 10. Sahibin kararı gereken sorular

1. §2'deki etiket düzeltmeleri ve ham değerlerin saklanması TRUBA üretiminden önce yapılsın mı?
2. Yarım hücre çözücüsünün toleransı ve çakışma kuralı düzeltilsin mi (§3.1)? Aynı tohum farklı geometri
   üretir; mevcut veri kümeleri eski kodla kalır.
3. Eliptikte çapalar düzeltilsin, elips aralıkları genişletilsin ve 100–105°'ye kadar re-entrant duvara izin
   verilsin mi?
4. Eliptikteki içi boş saplar kalsın mı (simetri bozucu olarak belgelenerek), yoksa kaldırılsın mı?
5. Çok hücreli örneklerde saklanan ve öğrenilen mod sayısı K ≥ N + 2 olsun mu (§3.9)?
6. Veri üretimindeki deformasyon (TRUBA ve arayüzde DEFORM_PROB 0,5, DEFORM_MAX 0,5) çeşitlilik ve imalat hatası ön
   ayarlarına ayrılsın mı (§5)?
7. ridged_box RF-dipol ailesiyle değiştirilsin mi? Eski hâli OOD'de kalsın mı?
8. Yeni ailelerin sırası (qwr, dtl_cell, rfd, gun) uygun mu?
9. composite'in çoğu gerçek hücre ve gerçekçi eklentilere dönüştürülsün mü?
10. Tam metin erişimi için literatür alan adları ortamın ağ ayarlarında açılsın mı?

## 11. Doğrulanamayanlar

- **Tam metinler** okunamadı; bütün literatür sayıları alıntı düzeyinde.
- **R/Q tanımları.**
  - EU ve ESRF kavitelerinin R/Q'su için "devre tanımı" bir çıkarım.
  - TESLA'nın 518 Ω'u devre tanımı. Hücre başına 113,8 Ω'luk linac değeriyle tutarlı: 9 × 113,8 = 1024 Ω,
    linac 9 hücre değeri 1036 Ω (1. ajan).
  - ESS ve CEPC'nin R/Q tanımı alıntılarda yok.
- **Tahmine dayanan aralıklar:** RF-dipol ve DQW parametrelerinin bir kısmı, QWR yükseklikleri, DTL'de g/βλ.
- **Bulunamayan boyutlar:**
  - RE, LSF, SNS, ESS ve LB650 hücrelerinin elipsleri.
  - NLSF, Cornell ERL ve FCC-ee hücreleri yalnız açık kaynak bir tasarım kodunda (CavityDesignHub) var;
    kullanmadan önce yayınla karşılaştırılmalı.
- **Küçük yarıçaplar.** Gerçek kavitelerdeki küçük yarıçapların (burun, drift tüpü, spoke tabanı) %5'lik
  fillet tabanına göre dağılımı ölçülmedi.
- **Mesh maliyetleri.** Yeni ailelerin tet sayıları benzetme yoluyla tahmin edildi; 4 Sobol çekilişiyle
  ölçülmeli.

## 12. Uygulama durumu (2026-10-22)

Sahip düzeltmelere onay verdi. Yeni kod yalnız gereken yerlere yazıldı; ayar ile zaten yapılabilenler belgelendi.
Üretici artık aynı tohumdan başka geometri üretiyor. Bu yüzden veri kümesi etiketleri `_v1`'den `_v2`'ye
geçti (TRUBA, arayüz, `src/service`); eski ve yeni parçalar aynı klasörde karışmaz.

**Yapılanlar:**

| konu | değişiklik | yer |
|---|---|---|
| ridged_box ve box ekseni (§2.1) | ridged_box: x yönünde, çıkıntı aralığının ortasından (boruların ekseni); box: kutu merkezinden z | `qoi.operators.beam_axis(…, params)` |
| hızlandırma uzunluğu (§2.2) | boruları dışarıda bırakan iç uzunluk: eliptikte hücreler, pillbox / reentrant'ta hücre, spoke / dtl'de tank, hwr'de dış çap, ridged_box ve box'ta kutu boyu; ışını olmayan ailelerde kiriş | `qoi.operators.acceleration_length`; alan etiketleri, N0 etiketleri ve N0 eğitimindeki model QoI'si aynı tanımı kullanıyor |
| β (§2.3) | her mod için eksen boyunca E_z (U = 1 J) saklanıyor: T, V ve R/Q her β'da yeniden çözümsüz hesaplanır | `mode_qoi` → H5 `axis_t`, `axis_ez` |
| çözücü toleransı (§3.1) | `xtol = ftol = gtol = 1e-15` ve beş başlangıç noktası; doğru teğet türü (iki merkez doğrunun iki yanında) denetleniyor | `half_cell_wall` |
| çakışma kuralı (§3.1) | kaldırıldı; yerine "r duvar boyunca artar" | `half_cell_wall`, `_valid_half` |
| re-entrant duvar (§3.1) | 105°'ye kadar, iris düzlemine ≥ 0,05 Req uzaklıkla | `MAX_WALL_ANGLE`, `WEB_MIN` |
| düz teğet parçası | spline'ın düz parçada sapmaması için elips adımıyla örnekleniyor (genişletilmiş aralıklarda bu parça uzuyor; test yakaladı) | `half_cell_wall` |
| çapalar (§3.1) | `ilc_ll` yayınlanan LL değerleri; ICHIRO eklendi. `anchor` = 1 + sıralı ad indeksi (ichiro 1, ilc_ll 2, tesla 3) | `ANCHORS` |
| elips aralıkları (§3.1) | A/L 0,6–0,9, B/A 0,6–1,3, a/L 0,12–0,35, b/a 0,65–1,9 | `_draw_half` |
| uç hücreler (§3.1) | L ×0,9–1,1, A ×0,9–1,2, B ×0,85–1,4, a ve b ×0,5–1,05, iris ×1,0–1,35 | `END_CELL_RANGES` |
| saplar (§3.1) | kod değişmedi; simetri bozucu pertürbasyon olarak belgelendi | `cavity_shapes` belge dizesi, docs/19 |
| reentrant (§3.3) | Lc/R 0,4–1,6, g/Lc 0,15–0,75 | `build_reentrant` |

**Ölçülenler** (bu oturumda):
- Yayınlanmış 14 yarım hücrenin hepsi kabul ediliyor (TESLA orta ve iki uç, LL, ICHIRO, NLSF, NLSF-RE, Cornell
  ERL orta ve iki uç, FCC-ee C3794, C3795 orta ve uç, FCCUROS5). Önce dördü reddediliyordu.
- Rastgele β = 1 yarım hücrelerin kabulü yeni aralıklarla %65,1 (1. ajanın tahmini %65,3; eski kod ve
  aralıklarla %50).
- `draw_elliptical` çekiliş başına 57 ms; meshleme ve etiketin yanında ihmal edilebilir.
- Testler: tam takım 271 test geçti; sonraki düzenlemelerden ve inceleme düzeltmelerinden etkilenen dosyalar
  yeniden çalıştırıldı (146 ve 124 test geçti);
  `ruff check .` temiz; `npm run build` temiz.

**Fizik incelemesi** (bağımsız alt ajan): ridged_box ekseni, etiketlerle model QoI'sinin aynı uzunluğu
kullanması, `axis_ez` ölçeklemesi, teğet türü testi (400 hücrede 12×12 başlangıçla kaba kuvvet: yanlış teğet 0)
ve kapalı profil (3000 çekilişte 254 re-entrant duvar, en büyük açı 104,6°, en ince ağ 0,0501 Req, kendini
kesme yok) doğru bulundu. Bulgular ve yapılanlar:
- arayüz QoI'si `geom_params`'ı kullanmıyordu → veri kümesi okuyucuları ve örnek uç noktası artık geçiriyor;
  çözüm yanıtında `L_acc_mm` var;
- düz teğet parçası gereğinden sık örnekleniyordu (parça başına 396 noktaya kadar) → en çok 2n nokta;
- H5 tek başına V(β)'yı kuramıyordu → `axis_dir`, `axis_point`, `axis_dt`, `L_acc` grup özniteliği;
- `calib_box` da box gibi ele alınıyor;
- deformasyonlu örneklerde eksen ve L_acc nominal parametrelerden geliyor (docs/24'te sınırlama olarak yazıldı;
  düzeltmesi deformasyon haritasını N0 H5'e de taşımayı gerektirir, sahibin kararı).

**Uçtan uca kontrol** (bu oturumda, `--max_ndof 150000`):
- pillbox_pipes: L_acc 72 mm, eksen kirişi 132 mm. Temel mod Epk/Eacc 3,47 (eski tanımla 1,83 kat büyük
  olurdu).
- elliptical, 3 hücre: L_acc 279 mm, kiriş 385 mm.
- `axis_ez`'den hesaplanan R/Q, etiketteki R/Q ile her modda aynı.
- ridged_box (N0 yolu, 6 geometri): yeni eksen kiriş 52–99 mm, eski eksen 0 mm. En alttaki saptırıcı modlarda
  R/Q ≈ 0, üst modlarda 165 Ω'a kadar.
- Analitik pillbox doğrulaması PASS: f 1,4·10⁻⁶, Q0 3,6·10⁻⁵, R/Q 5·10⁻⁶.

**Kod gerektirmeyenler:**
- **Deformasyon (§5).** İmalat düzeyi koşu bugün de mümkün: `DEFORM_MAX=0.01`–`0.02`. TRUBA varsayılanı
  (0,5 / 0,5) çeşitlilik için olduğu gibi kaldı; parametreler her örnekte saklanıyor.
- **Çok hücrede mod sayısı (§3.9).** Eğitimde N ≤ 5, π modu öğrenilen 6 modun içinde. 7–9 hücreli OOD testi
  için TRUBA zaten 10 mod saklıyor; değerlendirmede `model.num_field_modes=10` ile eğitilmiş bir model gerekir.
- **Boru kesim frekansları.** Boru yarıçapı `geom_params`'ta; kesim frekansı sonradan hesaplanır.
- **Boru uzunluğu kuralı (§3.2).** Etiketler kapalı PEC problemi için doğru; yalnız gerçek tasarımlarla
  kıyasta önemli. Doğrulama geometrilerinde borular ≥ 3 rp alınır (§8).

**Ayrı iş olarak kalanlar** (yeni kod ister; sahibin sıra kararıyla):
- yeni aileler: qwr, rfd (ridged_box yerine), dtl_cell, 1,6 hücreli tabanca, büyük açıklıklı eliptik alt
  ailesi;
- spoke uç duvarları ve konik spoke, dtl'de uçta yarım drift tüpü;
- composite'in gerçek hücre ve eklentilere dönüştürülmesi;
- doğrulama seti (§8).
