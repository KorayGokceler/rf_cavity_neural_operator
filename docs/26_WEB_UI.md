# 26 · Web arayüzü: RF Cavity Neural Solver

> **Partly historical:** the parts about the removed N0 model (`eigenspace3d`, `EigenspaceOperator3D`, PKL / `convert_3d.py`, `eval_3d.py` / `eval_qoi.py`, the torch QoI loss) no longer apply; removed in the `claude/field-model-ui` PR. The current model is `field3d` (docs/28, docs/29).
> Since the same PR the service's **Predict runs field3d** (§1b); generation, training and the TRUBA page
> are field-label only.

> Durum: **v0.1**. Hedef: dışarıya açık ürün/demo. Teknoloji: FastAPI +
> React + vtk.js. v1 kapsamı: dosya yükle + tahmin, parametrik kurucu, tarama / hedef frekans,
> güven + FE doğrulama.

## 1. Mimari

```
 Tarayıcı: React + TypeScript + vtk.js (web/)
   │  REST (JSON; büyük diziler base64 float32 / uint32)
   ▼
 FastAPI (src/service/api.py) ─ giriş kontrolü: uzantı, boyut, mesh boyu, tet sayısı
   ├─ geometri: src/service/geometry.py, ayrı süreçte (spawn), süre + bellek sınırı
   │            STEP/IGES/BREP → gmsh tet mesh · .msh/.vtu → meshio · aile + kimlik → üretici
   └─ model:    src/service/model.py; model bir kez yüklenir, kilitle tek forward
                tahmin → Ritz alanları + f → Q0, G, R/Q, R_sh, T, Epk/Eacc, Bpk/Eacc + rezidü η
                görüntü alanları: U = 1 J'de E [V/m], H [A/m]; E(t) = E·cos φ, H(t) = H·sin φ
```

- CLI (`scripts/predict_geometry.py`), TRUBA (`cluster/truba/predict.sh`) ve web aynı servis
  katmanını kullanır.

## 1b. Tahmin: field3d modeli (eğri mesh, HCurl p2)

`ModelService.predict` (`src/service/model.py`), `scripts/eval_fidelity.py`'nin etiketsiz yolu:

1. **Eğri mesh** (`geometry.mesh_src`: netgen `.vol` + katının BREP'i [m] + eğrilik mertebesi + deformasyon,
   alan-etiketi H5'iyle aynı saklama). Kaynaklar:
   - **CAD yükleme** (STEP/IGES/BREP): katı ayrı süreçte metreye ölçeklenir, `field_labels.start_mesh` ile
     eğri mesh (maxh = 3 h, curvaturesafety 1.5 → 2). Başarısızsa gmsh mesh'i yalnız gösterim için kalır.
   - **Aile örneği:** üretici alan-etiketi ayarlarıyla (kenar yuvarlatma, OCC onarımı, Sobol, seed 0) — alan
     datasetlerindeki geometrinin aynısı. Eğri mesh başarısız olursa üretici gibi geometri yeniden çekilir.
   - **Alan dataseti öğesi:** saklanan eğri mesh; tahmin, saklanan FE çözümüyle aynı gösterim mesh'inde olur
     (model–FE karşılaştırması çalışır).
   - **Tahmin edilemeyenler:** mesh dosyaları (.msh/.vtu…), `freeform` ailesi (CAD katısı yok). Geometri yanıtında `can_predict: false` ve `predict_blocker` (kısa İngilizce neden);
     `POST /api/predictions` aynı nedenle 422 döner.
   - **Bütçe:** eğri mesh `RFCAV_MAX_PREDICT_TETS` (30000, etiketlerin eleman bütçesi) tet'i aşarsa 422.
2. `field_labels.model_normalisation` → `highorder.field_operators(order=2)` → `field_item` / `field_collate`
   → model. Eğri mesh ve operatörler geometri kimliği başına önbellekte (`RFCAV_FIELD_CACHE`, son 2 geometri).
3. **Frekanslar:** modelin Ritz değerleri (`out['freq']`, `freq_stats = (0, 1)` ile GHz). Aynı mesh'in ayrık
   p2 özdeğerlerinin üst sınırıdır (test: kutu, CAD pillbox).
4. **Kavite değerleri** (Q0, G, R/Q, R_sh, T, Epk/Eacc, Bpk/Eacc): tahmin edilen p2 alanlarından
   `highorder.mode_qoi` ile, etiketlerle aynı kod, eksen ve hızlanma boyu (`qoi_source: p2_model`).
   Hızlandırıcı mod bu değerlerle yeniden seçilir.
5. **η:** alan operatörlerinin K, M'siyle ‖K u − λ M u‖_{D⁻¹} / λ.
6. **Gösterim:** p2 alanı doğrudan çizilir, Whitney N0 izdüşümü yok (§1d). Çözüm tembeldir: `predict` yalnız DOF'ları,
   frekansları, değerleri ve η'yı tutar; alanlar ilk `/fields` isteğinde hesaplanır.

Süreler (`time_s`): `mesh`, `operators` (önbellekten 0), `model`, `qoi`, `display`, `total`. Yanıtta ayrıca
`qoi_source`, `n_dof` (p2 DOF), `n_tets` (eğri tet). Ölçüm: docs/31.

Eğitilmemiş demo modu: checkpoint yoksa (ya da `--checkpoint_optional` ile yüklenemezse) rastgele
başlatılmış küçük bir field3d modeli; uyarı "untrained demo model". `load_model` yalnız `model_type: field3d`
checkpoint'lerini kabul eder; `serve_web.py --checkpoint auto` başka türdekileri atlar.
- **v0.1'de** sonuçlar bellekte tutulur: LRU, en fazla 64 kayıt, 1 saat, diske yazılmaz.
- **v0.2'de** iş kuyruğu (Redis + RQ) gelir: FE doğrulama, tarama.

### 1c. Yükleme birimi, sınırlar ve uyarılar

- **Birim:** STEP / IGES dosyaları dosyanın bildirdiği birimde okunur (OCC içe aktarırken çevirir;
  `Geometry.OCCTargetUnit = M` kullanılır); `unit` alanı yalnız BREP (birimsiz; üretici BREP'leri metre) ve
  ağ dosyaları için geçerlidir, arayüzde STEP / IGES için seçici kapalıdır. Yanıt `source.unit_used`
  ('file' ya da seçilen birim) bildirir.
- **CAD yüklemeleri:** ışın ekseni dosya orijininden geçen z ekseni, `L_acc` = borular dahil eksen kirişi
  (her yükleme tahmininde uyarı); çıkıntılı (re-entrant) keskin kenarlar sayılır (`source.n_sharp_edges`),
  varsa Epk/Eacc ve Bpk/Eacc raporlanmaz.
- **Karşılaştırma (`compare_to`):** modelin kavite sayıları tahminin kendi p2 değerleridir (izole modlar);
  FE'nin dejenere kümesindeki modlar için '—'.
- **Sınırlar:** `RFCAV_MAX_PREDICT_TETS` (varsayılan 30000 eğri tet) üstünde tahmin reddedilir; eğitim
  mesh'leri ≤ 10000 tet olduğundan 10000 üstünde "doğruluğu doğrulanmadı" uyarısı verilir. Tahmin hataları
  HTTP 422 döner. Başlangıç mesh'inin tet sayısını katı eğrilik belirler, `mesh_size` değil (ölçüm, 15000
  sınırıyla: reentrant 25–27k, hwr 18–19k, elliptical 7,5k – 29,6k, pillbox_pipes 8,9k, spoke 4–15k, box < 0,3k).
- **Rastgele örnek numaraları** [2^24, 2^25): `families.tsv` bloklarının hepsinin ötesinde.

### 1d. Gösterim hattı: p2 alanı doğrudan (`src/viz/hcurl_display.py`)

Eskiden p2 alanı doğrusal mesh'e Whitney N0 olarak (`Set(dual)`) izdüşürülüp köşelerde ortalanıyordu; resmin
hatası sayılardan çok büyüktü (§ aşağıdaki tablo). Şimdi eğri mesh'in p2 alanı kendisi çizilir. Model tahmini ve
FE etiketi (dataset öğesi) aynı yoldan geçer; ikisi de `HCurl(p2, dirichlet='wall')` DOF vektörüdür.

| öğe | nasıl |
|---|---|
| **düğümler** | her tet'in 10 P2 noktası (4 köşe + 6 kenar ortası), eğri eleman haritasıyla (`MapToAllElements`) metre cinsinden; ortak düğümler ortalanır. `SetDeformation` dahildir: deforme örnekler artık deforme çizilir (açık iş 20'nin gösterim kısmı) |
| **hücreler** | tet başına 8 doğrusal alt-tet (4 köşe tet'i + iç sekizyüzlü en kısa köşegenle 4 tet'e bölünür). Konform, hepsi pozitif hacimli; yarım eleman boyunda doğrusal bir mesh |
| **alanlar** | düğümlerde E ve H = −curl E/(ωμ0) (p2'nin tam değerlendirmesi, düğümlerde ortalama), U = ½ε0∫\|E\|² = 1 J. Tarayıcı alt-tet içinde doğrusal ara değer alır |
| **duvar** | her sınır yüzünde üçgen kafes, seviye 4 (9k yüze kadar; üstünde 3, 16k yüzden sonra 2; en çok ~150k üçgen), eğri yüzeyden (doğrusal yüzey değil); kaynaklanmış, dışa bakan. Değerler komşu hacim elemanından `BoundaryFromVolumeCF` ile (Epk / Bpk'nin yolu, tek taraflı); \|E\| ve \|H\| float16. Geometri yanıtının `surface`'ı da bu kafestir; `check` hacmi ve bbox'ı eğri katıdan (`Integrate(1)` ve kafes) gelir |
| **eksen** | E_z profili 401 noktada p2'den, kavite dışı `null`; merkez = alt-tet hacim merkezi. FE ve fark için de aynı yol |
| **kesit, hacim** | `/plane` ve `/volume` sunucuda tam p2 değerlendirmesi (`eval_points`; NGSolve nokta arama), arayüz bunları kullanmaz ama test ve dış kullanım için doğruluk ölçütüdür. Deforme örnekte nokta araması deformasyonu görmez: `hcurl_display.locate` ön görüntüyü Newton ile bulur |

**Enerji ölçeği** ölçüldü (varsayım değil): alan operatörlerinin M'si fiziksel M / `scale`'dir
(`\|M·scale − M_fiziksel\|` = 1e-16 bağıl), yani U = ½ε0 · scale · uᵀMu (N0 yolundaki `scale³` değil). Aynı kuralla
(order 8) `Integrate` ile makine hassasiyetinde, order 12 ile 5e-8 içinde eşit (`tests/test_hcurl_display.py`).

**`/fields` şeması** (bir çözüm için tek istek):

| anahtar | içerik |
|---|---|
| `n_modes`, `n_nodes`, `n_cells`, `n_tets`, `n_surface`, `f_GHz` | boyutlar (`n_cells` = 8·`n_tets`, `n_surface` geometrinin `surface.n_points`'ı) |
| `nodes_b64` | P2 düğümleri [n_nodes, 3] float32, mm |
| `tets_b64` | alt-tet'ler [n_cells, 4] uint32 |
| `E_b64`, `H_b64` | [K, n_nodes, 3] float16, her mod kendi `E_max` / `H_max`'ına bölünmüş |
| `surf_E_b64`, `surf_H_b64` | [K, n_surface] float16, duvar noktalarında \|E\|, \|H\| (aynı bölenler) |
| `E_max`, `H_max` | mod başına, düğümler **ve** duvar üzerinden en büyük değer |
| `axis` | `z_mm`, `Ez[K][P]` (kavite dışı `null`) |

Fark çözümü (model − FE): düğüm alanları çıkarılır, duvar değeri vektör farkının büyüklüğüdür (sunucuda).
Karşılaştırma `align`'ı p2 DOF uzayında gerçek kütle matrisiyle yapar (`rel_l2` artık iki p2 alanın gerçek M
normundaki hatasıdır, eski N0 ölçüsünden biraz farklı çıkar).

**Ölçümler** (`scripts/measure_display.py`; boş makine, 4 çekirdek, eğitilmemiş model, tek koşu, eliptik id 3):

| | 10,7k tet önce | 10,7k sonra | 27,5k tet önce | 27,5k sonra |
|---|---|---|---|---|
| `predict` gösterim aşaması | 1,66 s | 0,26 s | 4,51 s | 0,76 s |
| ilk `/fields` isteği (sunucu) | 0,02 s | 0,67 s | 0,02 s | 1,63 s |
| `/fields` ham / gzip | 581 kB / 333 kB | 4,59 MB / 2,80 MB | 1,35 MB / 819 kB | 11,1 MB / 6,7 MB |
| geometri yanıtı (yüzey dahil) | 73 kB | 1,16 MB | 144 kB | 2,28 MB |
| tepe RSS (geometri + tahmin + `/fields`) | 2111 MB | 1994 MB | 3453 MB | 3404 MB |
| kesit kaydırıcı, JS (Chromium, orta / en çok) | 1,2–1,7 / 7 ms | 7–9 / 16 ms | 2,6–4,8 / 13 ms | 19–21 / 30 ms |
| 3B ızgara (`rasterize`, Chromium) | 20 ms | 49 ms | 39 ms | 76 ms |

Hedefler: gösterim aşaması ≤ 0,3 s (10k) tutuyor; `/fields` ≤ 6 MB ham (10k) tutuyor, bu yüzden şema basit
kaldı (mesh geometri başına bir kez gönderilmiyor). 30k tet'te 11 MB ham / 6,7 MB gzip: sıradaki azaltma
adımları (`GET /api/geometries/{gid}/display_mesh` ile mesh'i bir kez göndermek, alt-tet'leri `tet10` olarak
gönderip istemcide açmak, H'yi anahtarda istemek) yapılmadı. JS kesit süresi 50 ms altında kaldığı için alt-tet
ön süzgeci eklenmedi.

**Doğruluk** (pillbox TM010, R = 40 mm, L = 50 mm, curve 3, p2; gösterilen alanın tam alandan noktasal hatası,
tepe E0 / H0'ın kesri; 5000 rastgele iç nokta, tarayıcının yaptığı barisentrik ara değerle):

| tet | eski E maks / rms | yeni E maks / rms | eski H maks / rms | yeni H maks / rms |
|---|---|---|---|---|
| 124 | 2,4e-1 / 1,2e-1 | 8,1e-2 / 2,0e-2 | 2,5e-1 / 1,3e-1 | 9,8e-2 / 3,2e-2 |
| 581 | 1,4e-1 / 5,9e-2 | 3,0e-2 / 7,8e-3 | 1,3e-1 / 5,3e-2 | 4,5e-2 / 9,2e-3 |
| 1773 | 9,6e-2 / 3,3e-2 | 1,6e-2 / 3,9e-3 | 6,7e-2 / 2,9e-2 | 2,0e-2 / 4,5e-3 |
| 6678 | 5,5e-2 / 1,6e-2 | 5,5e-3 / 1,6e-3 | 3,6e-2 / 1,4e-2 | 9,1e-3 / 1,9e-3 |
| 12170 | 4,0e-2 / 1,1e-2 | 4,4e-3 / 9,6e-4 | 2,6e-2 / 9,1e-3 | 4,8e-3 / 1,1e-3 |
| 21537 | 3,9e-2 / 9,1e-3 | 2,5e-3 / 6,9e-4 | 2,0e-2 / 7,4e-3 | 3,9e-3 / 8,8e-4 |

**Dürüst sınır:** gösterilen alan, P2 düğümlerinin alt-tet'lerinde P1'dir. İnce mesh'lerde hatası tepenin ~1e-3'ü
(21k tet'te E: 2,5e-3 maks, 7e-4 rms) kalır; 6,7k tet'te p2 çözümün kendi noktasal hatası E için 6e-4 maks / 1,2e-4 rms,
yani gösterim bunun ~10 katı. N0 yoluna göre ~13–15 kat iyidir ama sıfır değildir. Tam görmek için `/plane`, `/volume` (tam p2) kullanılır. Düğümlerde ortalama,
elemanlar arası normal E sıçramasını (HCurl yalnız teğetsel sürekliliği korur) yumuşatır; duvarın kendisi ortalanmamış
sınır yolundan çizilir. Çizilen P1 alanın enerjisi tam 1 J değildir: ölçek tam (kütle matrisi) ama P1 ara değer tepeyi hafif
keser (6,7k tet'te −%0,5, 1,8k'da −%1,3; hedef 1e-3 tutmadı, tam enerji `field_energy` ile alınır). Duvar
kafesinin radyal sapması seviye-4'te 9,6e-5 R (doğrusal yüzeyde en çok %2 R).

## 2. API (v0.1)

| uç nokta | ne yapar |
|---|---|
| `GET /api/info` | model (alan, parametre sayısı, cihaz, eğitilmemiş mi, `model_type`), aileler, sınırlar (`max_predict_tets` dahil), kurallar |
| `POST /api/geometries` (multipart: `file`, `unit`, `mesh_size`, `n_cells`) | mesh + kontroller (tek parça / manifold, b1, hacim, bbox) + görüntü yüzeyi + `can_predict`, `predict_blocker` |
| `POST /api/geometries/sample` `{family, id?, mesh_size}` | üretici geometrisi (id boşsa eğitimde kullanılmamış rastgele bir kimlik) + `params`, `can_predict`, `predict_blocker` |
| `POST /api/predictions` `{geometry_id}` | mod tablosu (f, kavite değerleri, η, dejenere, hızlandırıcı mod), `qoi_source`, `n_dof`, `n_tets`, süreler, uyarılar; tahmin edilemezse 422 |
| `GET /api/predictions/{id}/plane?mode&axis&pos&res` | kesit düzlemi, tam p2 değerlendirmesi: noktalar, üçgenler, E, H |
| `GET /api/predictions/{id}/volume?mode&n` | kutu üzerinde ~n noktalık düzenli ızgara (x en hızlı), tam p2 değerlendirmesi: E, H, kavite içi maskesi |
| `GET /api/predictions/{id}/fields` | bütün modlar P2 düğümlerinde + alt-tet mesh'i + duvar değerleri, tek yanıtta (§1d; çalışma alanı bunu kullanır). `/modes/{k}` kaldırıldı |
| `GET /api/predictions/{id}/export` | JSON dosyası |

**Çalışma alanı alanı tarayıcıda hesaplar** (`web/src/fieldlab.ts`). Bir çözüm için tek istek atılır:
`/fields` (§1d). İçinde bütün modların P2 düğümlerindeki alanları (float16, her mod kendi E_max / H_max'ına bölünmüş),
düğümler, alt-tet'ler ve duvar noktalarındaki \|E\|, \|H\| var. Yanıt gzip'li (tünel: Colab, telefon). Sonrası tarayıcıda:
- **kesit:** alt-tet'ler düzlemle kesilir (gösterim mesh'inin gerçek kesiti), kenar üstündeki değer kenar boyunca doğrusal;
- **3B ızgara:** `locate_grid`'in aynısı (her tet kutusundaki ızgara noktalarını barisentrik testle bulur);
- **mod, E/H, faz:** yalnız hangi diziden okunacağı değişir.

Hız için:
- Bir geometrinin FE, model ve fark çözümleri aynı mesh'i paylaşır. Kesitler ve ızgara mesh başına bir kez
  hesaplanır. Tahminden sonra öteki kaynaklar arkadan indirilir; kaynak değiştirmek de beklemez.
- Faz yalnız renk eşlemesinin aralığını değiştirir (genlik × |cos φ| / |sin φ|). Hacim dokusu yeniden
  yüklenmez; her adımda yalnız oklar yeniden kurulur.
- Bir değişiklik birkaç efekti tetikler; 3D görünüm yine kare başına bir kez çizer.

Ölçüm (Playwright, tünel benzetimi 150 ms gecikme, 2 MB/s; yerleşimin GPU'su yok, çizim yazılımla):

| işlem | önce | sonra |
|---|---|---|
| mod değiştir (kesit) | 0.47–0.49 s, 708 kB | 46–76 ms, 0 kB |
| E ↔ H | 0.27–0.30 s | 51–55 ms |
| kesit konumu | 0.39 s, 438 kB | 37 ms, 0 kB |
| 3B hacme geç | 0.93 s, 2.9 MB | 0.16 s, 0 kB |
| mod değiştir (hacim + oklar) | 2.0 s, 3.1 MB | 0.44 s, 0 kB |
| 10 adım faz kaydırma (hacim) | 6.2 s | 0.65 s |
| örneği aç (ilk indirme) | 763 kB | 529 kB (bütün modlar) |

Önceden her mod değişimi ve kesit konumu yeni bir istekti. Yanıt gecikince bir sonraki değişiklik onu iptal
ediyordu; görüntü bazen hiç değişmiyordu. Artık ilk indirmeden sonra hiçbir etkileşim ağa gitmez.

**Kesit ve hacim örneklemesi** (`nedelec.locate_grid`): her tet, sınır kutusundaki ızgara noktalarını
barisentrik testle bulur. Maliyet ızgaraya doğrusal. Eski `locate`, kavite dışındaki her nokta için 16
→ 256 aday tet arıyordu:
- merkezden uzak bir kesit saniyeler sürüyordu, kaydırıcı ilerledikçe önceki yanıt iptal edildiği için
  görüntü ilk kesitte kalıyordu. 11k tetlik bir örnekte kesit artık 20–60 ms sürüyor;
- 88k noktalık hacim 23 s'den 0.4 s'ye indi.

## 3. Arayüz

Görünüm yönü: **CAD / mühendislik aracı** (ParaView, CST, Onshape gibi yoğun, sakin paneller ve araç çubukları). Koyu tema
birinci sınıf, açık tema duruyor. Tasarım dili `web/src/styles.css` içindeki CSS belirteçlerinde (boşluk ölçeği 2–24 px,
yarıçaplar 3 / 5 / 8 px, yazı ölçeği 11–18 px, açık ve koyu renk kümeleri); simgeler satır içi SVG (`Icon.tsx`), UI kitaplığı yok.
Değerler `tabular-nums` ile hizalanır; odak halkaları, üzerine gelme durumları, bölümlü denetimler ve simge düğmeleri ortaktır.

### 3.1 Kabuk

- **Gezinme çubuğu (sol):** Çalışma alanı · Dataset · Üretim · Eğitim · TRUBA. Simge + etiket, "Daralt" ile yalnız simgeye
  iner (tercih saklanır). 860 px altında alt gezinme çubuğuna dönüşür.
- **Üst çubuk:** başlık, sayfa adı, model / cihaz çipi (eğitilmemişse "eğitilmemiş" etiketi), uyarı çipi, kısayol yardımı,
  TR/EN, tema. **Uyarılar** (eğitilmemiş model, OOD aile, ışın ekseni kavitede değil …) tam genişlikte şerit değil, çipin açtığı
  listede; böylece yükseklik yenmez. Hatalar ve bitişler sağ altta **bildirim** (toast) olarak çıkar.
- **Bildirimler:** sağ altta, kendiliğinden kapanır (hata 9 s, öteki 4.5 s), `aria-live` bölgesi. Konular: geometri hazır /
  kurulamadı, tahmin tamam / başarısız, üretim ve eğitim işi bitti / başarısız / durdu. İş bildirimleri sayfaların mevcut
  yoklamasına bağlıdır (ilk yoklama yalnız durumu kaydeder; "çalışıyordu → bitti" geçişi bildirim üretir). Üretim ve Eğitim
  sayfaları bir kez açıldıktan sonra arka planda canlı kalır (form durumu ve yoklama sürer).

### 3.2 Çalışma alanı

- **Tuval:** 3B görünüm kullanılabilir yüksekliği doldurur. Yalnız tuval boyutu değişince (panel daraltma, bölünmüş görünüm)
  vtk.js yeniden boyutlanır (`ResizeObserver`); gizli sayfada çizim yapılmaz.
- **Sol "Geometri" paneli (daraltılabilir):** kaynak (örnek kavite: aile + kimlik + mesh boyu · dosya yükle: CAD/mesh, birim,
  hücre sayısı, mesh boyu), **son açılanlar**, kontroller (kapalı, tek parça, b1, tet / düğüm, hacim, OOD), "Tahmin et",
  **Kapat**, geometri bilgisi. Daralınca 34 px'lik bir sekme kalır.
  - **Tahmin edilemeyen geometri** (`can_predict: false`: mesh dosyası, `freeform`, bütçeyi aşan eğri mesh,
    başarısız kontrol): "Tahmin et" pasif; `predict_blocker` hem düğmenin ipucunda hem de düğmenin üstünde görünür bir
    uyarı kutusunda ("Tahmin edilemez: …"). Tuval ve denetçi "nedeni Geometri panelinde" der. Yükleme sekmesi, mesh
    dosyalarının yalnız görüntülendiğini ve tahmin bütçesini (`limits.max_predict_tets` eğri tet) yazar.
  - **Tahmin sürerken** (CPU'da birkaç saniyeden ~30 s'ye): düğmede dönen halka + "Tahmin ediliyor… N s" (kendi sayacı; panel
    saniyede bir yeniden kurulmaz), altında adımlar notu (eğri mesh, p2 operatörleri, model, kavite değerleri); Oluştur / Yükle /
    son açılanlar ve Tahmin et pasif; tuvalde ve durum çubuğunda da "Tahmin ediliyor…". Bitince bildirim toplam süreyi ve
    adımları yazar: `Tahmin tamam · 4.11 s (mesh 0.12 · operatörler 1.99 · model 0.91 · değerler 0.31 · gösterim 0.70)`
    (toplam sunucunun `time_s.total`'ı; önceden parçalar toplamla birlikte toplanıyordu, süre iki katı görünüyordu).
- **Sağ denetçi (daraltılabilir):** mod tablosu (★ hızlandırıcı, ≈ dejenere çift, η rozeti; karşılaştırmada f FE / f model / Δf /
  rel-L2), seçili modun kavite değerleri, FE / model / Δ% tablosu, süreler (mesh, operatörler, model, değerler, gösterim, toplam)
  + modelin mesh'i (p2 DOF, eğri tet), JSON indir. Mod tablosu başlığında CSV düğmesi. Tablonun altında değerlerin kaynağı:
  model için "frekanslar modelin Ritz değerleri (bu mesh'in p2 özdeğerlerinin üst sınırları); kavite değerleri tahmin edilen
  yüksek mertebe p2 alanlarından" (`qoi_source: p2_model`), alan datasetinin FE'si için "saklanan p3 etiketleri" (`p3_labels`).
- **Yüzen araç çubuğu (tuvalin sol üstü, gruplu, simge düğmeleri ve ipuçları):**

  | grup | denetim |
  |---|---|
  | gösterim | alan büyüklüğü ya da modelin girdi özelliği |
  | kaynak | FE · Model · Fark (yalnız birden çoksa) |
  | alan | E · H |
  | görünüm | kesit · 3B hacim, oklar |
  | eksen | x · y · z (kesit normali; hacimde bakış ekseni) |
  | yüzey | saydam · duvar alanı · mesh · gizli |
  | 2. satır | kesit konumu (mm), faz 0–180°, ▶ oynat |
  | araçlar | renk ölçeği, nokta okuyucu, iki görünüm (+ kamera bağı), dışa aktar |

  Girdi özelliği görünümünde kaynak / E-H / kesit-hacim yerine özellik türü (ve ham kanal) seçilir.
- **Görünüm şeridi (sağ üst):** Ev, ±X, ±Y, ±Z, İzometrik (§3.7). Yalnız kamerayı değiştirir; geometriye sığdırılır, araç
  çubuğunun kapladığı üst bölge boş bırakılır (pencere merkezi kayar). Sol alt köşedeki x / y / z üçlüsü kamerayı izler.
- **Renk çubuğu (tuvalin altı):** başlık = nicelik · mod · frekans · kaynak, ölçek türü (doğrusal / log10 · n dekad), rozetler
  (kilitli / elle), nokta okuyucu satırı, renk rampası ve uç değerleri.
- **Eksen grafiği:** alt dokta, daraltılabilir; genişliğe uyar. Işın ekseni boyunca E_z(z); tek seri, artı imleç ve ipucu.
- **Durum çubuğu:** geometri, tet, düğüm, p2 DOF + eğri tet (tahminden sonra), mod ve frekans, süreler (toplam + adımlar), kaynak,
  model (`field3d` · parametre · cihaz). Üst çubuktaki model çipi de model türünü yazar.
- **Renk her zaman alan büyüklüğüdür** (bileşen seçeneği yok; yön okların işi). Kesitte ~300, hacimde ~700 ok, boy ∝ |alan|.
  **Kesit:** yarı saydam kavite + renkli düzlem. **3B hacim:** alan büyüklüğü bütün hacimde, zayıf alan saydam, yalnız arka
  duvarlar çizilir. **Renkler:** büyüklük için tek tonlu mavi rampa, işaretli büyüklükler için mavi ↔ gri ↔ kırmızı; rampalar
  açık ve koyu temada ayrı; hacim rampanın %35'inden başlar.
- **Telefon (≤ 860 px):** tuval ekranı doldurur; alt gezinmenin üstünde üç sekme (Geometri · Gösterim · Modlar) alttan çıkan
  **sayfalar** açar (araç çubuğu "Gösterim" sayfasında, eksen grafiği "Modlar" sayfasında). İki görünüm üst üste dizilir.
  Hiçbir sayfada yatay kaydırma yok (390 px'te `scrollWidth == innerWidth` denetlendi). 3B görünüm dokunmatik: tek parmak
  döndür, iki parmak yakınlaştır. Dikey tuvalde kamera, sınırlar yatayda kırpılmasın diye geri çekilir.

### 3.3 Klavye kısayolları

Bir kutuya yazarken (metin / sayı / seçim) çalışmaz; kaydırıcıda ok tuşları kaydırıcıya aittir; düğme odaktayken Boşluk / Enter
düğmeyi çalıştırır. `?` ile kısayol penceresi açılır.

| tuş | işlev |
|---|---|
| `1` … `7` | kamera: +X, −X, +Y, −Y, +Z, −Z, izometrik |
| `H` | ev görünümü |
| `←` `→` | önceki / sonraki mod |
| `E` `M` | elektrik / manyetik alan |
| `S` | kaynak: FE → Model → Fark |
| `V` | kesit ↔ 3B hacim |
| `A` | oklar |
| `X` `Y` `Z` | kesit (bakış) ekseni |
| `[` `]` | kesiti kaydır |
| `Boşluk` | fazı oynat / durdur |
| `P` | nokta okuyucu |
| `D` | iki görünüm |
| `L` | renk ölçeğini kilitle |
| `G` | logaritmik ölçek |
| `Esc` | önce açık pencere / menü; yoksa açık geometriyi kapat (onay yok: yalnız arayüz durumu) |
| `?` | kısayol penceresi |

### 3.4 Dosya sürükle-bırak, son açılanlar

- Bir CAD / mesh dosyası çalışma alanının herhangi bir yerine bırakılınca yükleme formuyla **aynı yoldan** gider (formdaki
  birim, hücre sayısı, mesh boyu). Üstte "bırakın" kaplaması çıkar; uzantı sunucunun `limits` listesinde değilse hata bildirimi.
- **Son açılanlar:** son 8 geometri (dataset öğeleri: aile + kimlik + dataset; örnekler: aile + kimlik + mesh boyu), tek
  tıkla yeniden açılır. `localStorage`'da (`rfcav.recent`), tüm erişimler `try/catch` içinde; depolama yoksa sayfa yine çalışır.
  Yüklenen dosyalar listelenmez (dosya saklanmıyor). Açılamayan kayıt (başka dataset kökü, yeniden başlatma) listeden düşer.

### 3.5 Dışa aktarma

- **PNG:** araç çubuğunda *Dışa aktar → PNG*. vtk.js `captureNextImage` ile tuval cihaz piksel oranında alınır, 2B tuvale
  başlık (nicelik · mod · frekans · kaynak), **renk çubuğu** (aynı duraklar; log ölçekte dekad işaretleri), birim + ölçek türü,
  varsa nokta okuyucu satırı ve altta bir **gösterge satırı** (geometri, E/H ve faz, kesit konumu, U = 1 J) eklenir. İki
  görünümde iki panel yan yana. Dosya adı: `rfcav_<geometri>_<E0>_<kaynak>.png`.
- **CSV:** *Dışa aktar → CSV* ya da mod tablosu başlığındaki düğme. UTF-8, başlık satırı, `.` ondalık, eksik değer boş hücre.
  Sütunlar: `mode, f_GHz, degenerate, accelerating, eta, Q0, G_ohm, R_over_Q_ohm, R_sh_MOhm, T_transit, Epk_Eacc,
  Bpk_Eacc_mT_per_MVm`; karşılaştırma varsa ek olarak `f_FE_GHz, f_model_GHz, f_rel_err, rel_L2` ve her büyüklük için
  `<ad>_FE, <ad>_model, <ad>_delta_pct`. Satırlar gösterilen kaynağınkidir.
- **JSON:** seçili mod panelindeki "JSON indir" (sunucu uç noktası, §2).

### 3.6 Nokta okuyucu, renk ölçeği, iki görünüm

**Nokta okuyucu (`P`):** kesit görünümünde tuvale tıklanınca (sürükleme değil) kamera ışını kesit düzlemiyle kesilir
(`Viewer3D.rayAt`: vtk kamerasının iğne deliği modeli, pencere merkezi dahil); noktayı içeren kesit üçgeninde |E| ya da |H|
**barisentrik** enterpole edilir ve geçerli **faz** uygulanır (|cos φ| / |sin φ|). Kırmızı küre noktayı işaretler; renk çubuğunda
`|E| = … V/m · (x, y, z) mm` okunur (kavite dışı: "kavite dışında"). Nokta düzlem içi koordinatlarıyla saklanır: mod, kaynak,
E/H, faz ve kesit konumu değişince değer yeniden okunur, eksen değişince silinir. FE, Model ve Fark kaynaklarında çalışır
(Fark için |ΔE|); ağ isteği yok, tarayıcıdaki kesit verisi kullanılır. Doğrulama: işaretçi tıklama noktasına ≤ 1.5 px
uyuyor (ev, izometrik, +X, z ekseni).

**Renk ölçeği (araç çubuğunda palet menüsü):**
- **Kilit:** açıldığı andaki en büyük değeri (E ve H için ayrı) bütün modlara ve kaynaklara uygular; iki görünümde ikisi de
  aynı ölçeği kullanır, böylece doğrudan karşılaştırılır. Geometri değişince sıfırlanır.
- **Elle en büyük:** fiziksel birimde (V/m ya da A/m) yazılır, kilidi geçer; boş = otomatik.
- **Logaritmik:** en büyük değerin altında 2 ya da 3 dekad. Skaler log10(|alan|) olarak yüklenir (sıfır / negatif aralığın bir
  dekad altına kıstırılır), faz çarpanı yalnız renk aralığını log10 g kadar kaydırır: faz adımında veri yeniden yüklenmez.
  Renk çubuğu "log10 · 3 dekad" yazar.

**İki görünüm (`D`):** tuval iki panele bölünür (telefonda üst üste). A paneli ana denetimleri izler; B paneli kendi
(kaynak, mod) çiftini panel çubuğundan seçer (aynı mod için FE ↔ Model, ya da başka bir mod). İkisi de zaten indirilmiş
`/fields` verisini kullanır (ilk bölünmede başka kaynak varsa o, yoksa sonraki mod). Kameralar varsayılan olarak **bağlı**
(bir paneli döndürmek ötekini döndürür); bağ simgesi çözer. Ev / ön ayar düğmeleri iki panele de uygulanır; nokta okuyucu iki
panelde de değer gösterir; PNG iki paneli birlikte verir. Girdi özelliği görünümünde kapalıdır.

### 3.7 Görünüm şeridi ve uyarılar

- **Şerit:** Ev (açılıştaki görünüm), ±X / ±Y / ±Z (o yönden bak; X ve Y görünümlerinde z dikey, Z görünümlerinde y yukarı) ve
  İzometrik (+X, −Y, +Z yönünden). Hepsi geometrinin sınırlarına sığdırılır; mod, kesit konumu, renk ölçeği, kaynak aynı kalır.
- **Kapat:** açık geometriyi kapatır (sol panel ya da `Esc`); görüntüleyici, alan verisi, tahmin / karşılaştırma, ölçek ve
  nokta okuyucu temizlenir. Yalnız arayüz tarafı; sunucudaki kayıt `RFCAV_TTL_S` ile kendiliğinden silinir.
- **Uyarılar:** eğitilmemiş model (üst çubuk çipi + listede), OOD aile, ışın ekseni kavitede değil.
- **Dil:** TR/EN (tercih saklanır; metinler `web/src/i18n.ts`).

### 3.8 Diğer sayfalar

Dataset, Üretim, Eğitim ve TRUBA aynı panel / form bileşenlerini kullanır: bölümlü formlar, **"Gelişmiş" grupları kapalı
başlar**, sayı alanlarında satır içi doğrulama (aralık / tamsayı ihlali alanın altında, başlatma düğmesi geçersizken pasif).
**Üretim ve TRUBA sayfaları yalnız alan etiketi üretir** (etiket türü seçimi kaldırıldı, salt okunur bir satır yazıyor; PKL
dönüşümü / lean seçenekleri yok). **Eğitim sayfası** field3d'yi (`configs/field_3d.yaml`) yalnız alan-etiketi datasetleriyle
eğitir: listede yalnız `dataset_kinds` türündeki datasetler var; QoI kayıp ağırlığı ve "operatörleri RAM'de tut" kaldırıldı,
yükleyici işçi sayısı (`num_workers`) eklendi; varsayılanlar `GET /api/train/config` → `defaults`'tan (batch 2, sabır 40, işçi 2).

**Ölçüm (field3d arayüz uyarlaması, Playwright, yazılım GL, 1440×900; aynı yöntem iki derlemede, dönüşümlü 2 + 2 çalıştırma;
örnek / tahmin: tıklama → düğme hazır / mod tablosu, 3 kimliğin medyanı; mod ve E↔H: tuş → iki kare → 1 px GL okuması):**

| işlem | önce | sonra |
|---|---|---|
| kutu örneği (204 tet) | 1.69 / 1.36 s | 1.44 / 1.35 s |
| kutu tahmini → mod tablosu | 0.44 / 0.43 s | 0.28 / 0.44 s |
| eliptik örnek (4.9k tet) | 3.51 / 3.85 s | 3.78 / 3.53 s |
| eliptik tahmin → mod tablosu | 6.39 / 5.38 s | 5.44 / 5.37 s |
| mod değiştir (kesit) | 53 / 54 ms | 48 / 52 ms |
| E ↔ H | 48 / 44 ms | 50 / 49 ms |

Tahmin süresi sunucudadır (eliptik: operatörler ~1.3–2 s, model ~0.7–0.9 s); etkileşimlerde ağ isteği 0.

**Arayüz ölçümü (yeniden tasarım, yazılım GL, 48 852 tetlik eliptik örnek, aynı yöntem: tıklama / girdi → iki çizim karesi → 1 pikselik GL
okuması; 5 çalıştırmanın medyanı, ağ isteği 0):**

| işlem | önce (eski yerleşim) | sonra |
|---|---|---|
| mod değiştir (kesit) | 136 ms | 128 ms |
| E ↔ H | 114 ms | 145 ms |
| kesit konumu | 105 ms | 117 ms |
| 3B hacme geç | 540 ms | 499 ms |
| mod değiştir (hacim + oklar) | 205 ms | 227 ms |
| faz adımı (hacim + oklar) | 187 ms | 204 ms |

Eski ve yeni ölçümler dönüşümlü alındı. JS payı eşit (E↔H: 11 ms ↔ 11 ms; faz adımı ≈ 14 ms); fark yazılım GL'de: yeni tuval 740×602, eskisi 712×556 (+%12 piksel) ve her çizim karesi için bileşik katman sayısı arttı. Gerçek GPU'da ölçülmedi. Yeni yerleşimde mod değiştirmek ilk denemede 2.5× yavaştı: yan panellerdeki DOM değişimi bütün sayfayı yeniden boyuyordu.
Tuvalin çevresindeki panellere ayrı bileşik katman (`will-change: transform`) verilince fark kapandı; yan paneller ayrıca
yalnız kendi girdileri değişince yeniden kuruluyor (`useMemo`).

## 3b. Geliştirme aracı: dataset, özellikler, FE etiketleri, model–FE karşılaştırması

Arayüz geliştirme için de kullanılır. Geometriden model girdisine, FE çözümüne ve modelin hatasına kadar her şey görülebilir.

- **Dataset sayfası:** sunucuya `RFCAV_DATA` ile verilen datasetler listelenir (`src/service/datasets.py`).
  - **H5 parçaları** klasör başına bir dataset (alan etiketi, tür `field`). TRUBA düzeninde `h5/<aile>/` her aile için ayrı bir dataset olur. Alan etiketli olmayan H5 dosyaları / klasörleri sessizce atlanır (eski N0 etiketli H5 görüntüleyicisi kaldırıldı).
  - **Genel bakış:** aile başına geometri sayısı (tıklayınca filtreler), en düşük mod frekansı f₀ histogramı, model p2 DOF histogramı.
  - **Tablo:** id, aile, §3c'deki alan-etiketi sütunları, b1, f₀…f₅. Sıralanabilir, sayfalı, aileye göre filtrelenir. "Aç" geometriyi çalışma alanına gönderir.
- **Geometri açılınca:**
  - mesh ve kontroller;
  - **FE (etiket) çözümü:** modlar, alanlar, kavite değerleri. alan datasetlerinde değerler saklanan p3 etiketleridir;
  - geometri bilgisi: dataset, dosya, id, kenar (gösterim mesh'i), ölçek, torsiyon maksimumu, (K+1). mod frekansı, üretici parametreleri.
- **"Tahmin et ve FE ile karşılaştır":**
  - **Kaynak** seçimi: FE / Model / **Fark**. Fark = hizalanmış model − FE: işaret ve dejenere çiftler hizalanır, renk ölçeği FE alanına göre.
  - Mod mod f_FE, f_model, Δf, alan rel-L2 ve η.
  - Seçili mod için kavite değerleri yan yana: FE, model, Δ%.
- **Gösterim → Girdi özelliği:** her özellik kendi doğasına uygun biçimde gösterilir (`src/service/features.py`), skaler harita tek tip değil:

  | grup | kanallar | gösterim |
  |---|---|---|
  | Konum çerçevesi | x, y, z | bir alan değil, koordinat sistemi: kesit üzerinde normalize çerçevenin 0.25 adımlı ızgarası, merkezden geçen eksenler, merkez işareti; ±1 = `scale` mm |
  | Duvar vektörü | dist_to_boundary + dir_bnd_x/y/z | tek vektör özellik (en yakın duvar noktasına vektör): kesitte oklar, uçları duvara değer; arkada uzaklık haritası + eş-uzaklık çizgileri. Okların ayrıldığı çizgi orta eksen, yönün süreksiz olduğu yer |
  | Torsiyon | torsion | pürüzsüz skaler: renk + 0.1 adımlı eş-değer çizgileri |
  | Düğüm hacmi | node_volume | mesh bilgisi: kesiti kesen tet'ler kenarlarıyla ("crinkle cut"), düğüm hacmine göre renkli |
  | Ham kanal | herhangi biri | tek kanal skaler olarak (hata ayıklama) |

  Uç nokta: `GET /api/geometries/{id}/feature_view?kind=position|wall|torsion|mesh|raw&channel&axis&pos&res` → düzlem / hücreler / çizgi katmanları (rol: grid, axis, marker, iso, arrow).
- **Yüzey → mesh (tel kafes):** yüzey üçgenleri, yani mesh çözünürlüğü.
- **Hızlandırmayan modlar:** ışın ekseninde alan yoksa (R/Q < 1e-6 Ω) R/Q, R_sh, T, Epk/Eacc ve Bpk/Eacc "—" gösterilir; eksen grafiği de bunu söyler.

**Uç noktalar:**
- `GET /api/datasets`
- `GET /api/datasets/{d}/items?family&offset&limit&sort=id|family|n_edges|n_nodes|betti1|f0&desc`
- `GET /api/datasets/{d}/stats`
- `POST /api/datasets/{d}/items/{id}/open`
- `GET /api/geometries/{id}/features`, `…/feature_plane?name&axis&pos&res`, `…/feature_surface?name`
- `POST /api/predictions {geometry_id, compare_to}`: ek olarak `comparison.rows` ve `comparison.diff_id` döner.

**Bellek:** H5 listesi yalnız öznitelikleri okur ve hızlıdır; bir geometri açılınca yalnız o grup okunur.

## 3c. Arayüzden dataset üretimi (Google Drive'a)

**"Üretim" sayfası:**
- **Ayarlar:** aileler (eğitim / OOD), aile başına geometri, mesh boyu, saklanan mod sayısı, örnekleme (Sobol), deformasyon, seed, parça boyu, işçi × thread, TAG ve alan-etiketi ayarları (aşağıda). Yalnız alan etiketi üretilir (`labels: field`; N0 etiket türü üreticiden de kaldırıldı, `--labels` gizli ve etkisiz).
- **İşler:** her aile ayrı bir iş olarak kuyruğa girer, tek bir worker sırayla çalıştırır (`src/service/jobs.py`). İlerleme çubuğu, durum ve log gösterilir; iş iptal edilebilir.
- **Kayıt:** `RFCAV_GEN_ROOT` klasörüne, TRUBA ile aynı düzende ve aynı kimlik bloklarıyla (`cluster/truba/families.tsv`) yazılır. Bu yüzden arayüzde, Colab'da ve TRUBA'da üretilen datasetler birbiriyle karışabilir:
  ```
  <GEN_ROOT>/<TAG>/h5/<aile>/<aile>_s00000.h5 …
  <GEN_ROOT>/<TAG>/step/<aile>/<aile>_<kimlik>.step
  ```
- **STEP dosyaları:** deforme olmayan her geometrinin CAD katısı, mm biriminde, parçasıyla birlikte `step/<aile>/`
  klasörüne (Drive'da da) yazılır. CST ya da başka bir CAD aracında aynı geometriyi çözmek için. Dosya adındaki
  kimlik H5'teki `sample_<kimlik>` ile aynı. Deforme örneklerin etiketleri bükülmüş mesh'e ait olduğundan onlara
  STEP yazılmaz; CST karşılaştırması için deformasyon olasılığını 0 seçin. Katı, kavitenin vakum hacmidir
  (borular PEC kapaklı); CST'de arka planı PEC (ya da kayıplı bakır, σ = 5,8·10⁷ S/m) seçip katıyı vakum olarak
  içe aktarın.
- **Google Drive:** `Colab_WebUI.ipynb` (tek hücre) ya da `Colab_Maxwell3D.ipynb` hücre 15, `MyDrive/rf_cavity_3d/ui_datasets` klasörünü kullanır. Üretim **Colab VM'sinin CPU'larında** çalışır (çekirdek başına bir işçi; GPU gerekmez).
  - Parçalar önce yerel diske yazılır, tamamlanınca Drive'a taşınır (Drive yavaş bir ağ diski).
  - Bağlantı koparsa biten parçalar Drive'da kalır. Aynı işi yeniden başlatınca bitmiş parçalar atlanır, kaldığı yerden devam eder.
- **Dataset sayfası:** iş bitince otomatik yenilenir. Üretilenler `generated/<TAG>/…` adıyla görünür.
- **Sınırlar:** aile başına ≤ `RFCAV_GEN_MAX` (20000) geometri. Üretim, `RFCAV_GEN_ROOT` verilmezse kapalıdır.

**Alan etiketleri (tek etiket türü: yüksek mertebe alan; docs/29–30):** TRUBA'daki `LABELS=field` ile aynı üretici
ve aynı ayarlar:
- uyarlamalı inceltme (aç / kapat), frekans ve Q0 / R/Q / tepe toleransı, etiket DOF bütçesi, model mesh
  bütçesi;
- etiket / model mertebesi, fillet tabanı, eleman bütçesi, maxh çarpanı, örnek zaman aşımı;
- keskin iç kenarları yuvarlatma, OCC onarımı.

`freeform` seçilemez (CAD katısı yok). Çıktı yalnız H5'tir (eğitim `field3d`). TAG,
TRUBA'nınkiyle aynı kuralla kurulur: `F_p3p2_ms0.10_k10_mf0.05_v2`, tek çözümde `_1s`, yuvarlatmasız
`_sharp` eki.

**İşçi × thread planı** (`jobs.field_plan`): sunucu çekirdek sayısını (CPU affinity, cgroup kotası) ve
belleği (`MemTotal`, cgroup sınırı) okur. Bir işçinin tepe belleği ≈ 1 GB + 12 GB · (DOF bütçesi / 900k).
Belleğin %85'ine sığan sayıda 4 thread'li işçi açılır; artan çekirdekler işçilere ek thread olarak
dağıtılır. Örnekler:

| makine | DOF bütçesi | plan |
|---|---|---|
| 64 çekirdek, 251 GB | 900k | 16 işçi × 4 thread |
| 64 çekirdek, 100 GB | 900k | 6 × 10 |
| 2 çekirdek, 12.7 GB (ücretsiz Colab) | 600k (varsayılan) | 1 × 2 |

Varsayılan DOF bütçesi, en az bir işçinin belleğe sığdığı en büyük standart değerdir (900k / 600k /
300k / 150k). İşçi ve thread elle verilebilir; çekirdek ya da bellek aşılırsa sayfa uyarır.

**Dataset sayfası, alan datasetleri** (tür `field`): tabloda model ve etiket mesh'i (tet), model DOF,
uyarlamalı seviye sayısı, durma nedeni (yakınsadı / bütçe / en çok seviye), tahmini δf/f, yuvarlatılan
ve keskin kalan kenarlar görünür. "Aç", geometriyi model mesh'i üzerinde çalışma alanına getirir:
- **Alanlar:** p2 etiket alanları (`u_model`) olduğu gibi çizilir (§1d); N0 izdüşümü yok. Fiziksel kütle matrisi ve eğri duvar kafesi `open` sırasında hesaplanır.
- **Değerler:** Q0, R/Q ve tepe değerleri saklanan p3 etiketleridir (`qoi_source: p3_labels`). Eksende
  alanı olmayan modlarda gerilime bağlı büyüklükler "—" olarak gösterilir.

**Uç noktalar:**
- `GET /api/jobs/config` (makine: `cpu_count`, `mem_gb`; alan varsayılanları ve planı)
- `GET /api/jobs/plan?max_ndof&threads&workers`
- `POST /api/jobs/generate`
- `GET /api/jobs`, `GET /api/jobs/{id}`
- `POST /api/jobs/{id}/cancel`
- `POST /api/datasets/rescan`

## 3d. Arayüzden model eğitimi

**"Eğitim" sayfası** (`src/service/training.py`, `RFCAV_RUNS_ROOT` / `--runs_root`; Colab: `<Drive>/rf_cavity_3d/training_logs`):
- **Veri:** Dataset sayfasındaki **alan-etiketi** datasetlerinden (tür `field`) bir ya da birkaçı seçilir; H5
  dosyaları liste olarak `train.py --config configs/field_3d.yaml --override dataset.data_path=[…]`'e verilir
  (FieldDataset eğri mesh'leri ve operatörleri geometri başına yeniden kurar). Eğitim yalnız alan datasetlerini görür (N0 etiketli H5 artık listelenmez).
  - Bölme 80/10/10 eğitim/doğrulama/test (geometri bazında, seed 42; en az ~10 geometri).
- **Model boyu:** `tiny` (0.03 M, boru hattı denemesi) · `small` 0.85 M · `base` 2.8 M · `large` 6.5 M · `xl` 21.7 M. `cluster/truba/config.sh` ile aynı ön ayarlar (testle eşitliği denetlenir).
- **Ayarlar:** epoch, batch (2), öğrenme hızı, tahmin edilen mod sayısı, erken durdurma sabrı (40), yükleyici işçi sayısı (`num_workers`, 2; operatörleri işçiler kurar). Varsayılanlar `GET /api/train/config` → `defaults`.
- **Çalışma:** aynı anda bir eğitim (tek GPU). `train.py` alt süreç olarak çalışır; checkpoint'ler (`best-…ckpt`, `last.ckpt`) her doğrulamada doğrudan `<runs_root>/<ad>/` altına (Drive) yazılır.
- **İlerleme:** `train.py`, her çalıştırma klasörüne `progress.json` yazar (`src/training/progress.py`): durum, epoch, batch, epoch başına metrikler, en iyi checkpoint, test sonuçları. Sayfa bundan çizer:
  - alan hatası (göreli L2, M-normu),
  - frekans hatası,
  - kayıp,
  - öğrenme hızı (eğitim ↔ doğrulama).

  Notebook ya da TRUBA eğitimleri de aynı klasördeyse listede görünür.
- **Durdur / Devam et:**
  - Durdurulan ya da Colab kopmasıyla kesilen eğitim `last.ckpt`'den devam eder; hedef epoch artırılabilir.
  - Ayarlar `train_params.json`'dan okunur.
- **Bu modeli kullan:** sunucudaki modeli o çalıştırmanın en iyi checkpoint'iyle değiştirir (`POST /api/model/load`). Yalnız `runs_root` altındaki çalıştırmalar yüklenebilir: checkpoint bir pickle dosyasıdır, rastgele yol kabul edilmez.

**Uç noktalar:**
- `GET /api/train/config`
- `GET /api/train/runs`, `GET /api/train/runs/{ad}`
- `POST /api/train/start`
- `POST /api/train/runs/{ad}/resume`
- `POST /api/train/stop`
- `POST /api/model/load`

## 3e. TRUBA sekmesi: tek satır komut üretici

**"TRUBA" sayfası** (`src/service/truba.py`, `web/src/components/TrubaPage.tsx`): form, TRUBA login düğümüne yapıştırılacak **tek satıra** dönüşür (docs/29 §5):

`[ -d ~/rf_cavity_neural_operator ] || git clone <repo> …; cd … && git checkout <dal> && git pull && bash cluster/truba/run.sh <işlem> KEY=VALUE …`

- **Bölümler:**
  - İşlem: veri + eğitim, yalnız veri, yalnız eğitim, durum, kurulum.
  - Etiket türü: yalnız yüksek mertebe alan (field3d); komut her zaman `LABELS=field` yazar (`config.sh` varsayılanı da `field`; başka değer hata verir).
  - Aileler: eğitim ve OOD grupları.
  - Veri ayarları: geometri sayısı, mesh boyu, mod, p3/p2, fillet tabanı, thread, eleman bütçesi;
    uyarlamalı inceltme, toleranslar, model mesh ve DOF bütçesi, kenar yuvarlatma, OCC onarımı (docs/30).
  - SLURM: kuyruk, çekirdek, süre, hesap.
  - Eğitim: EXP, model boyu, epoch, batch, öğrenme hızı, GPU, önbellek, ek `train.py` ayarları.
  - Depo: adres, dal, klasör.
- **Çıktı:**
  - Komut ve **Kopyala** düğmesi.
  - TAG ve çıktı klasörü.
  - Kaba tahmin: geometri, çekirdek-saat, disk.
  - Uyarılar: model mertebesi 1, ridged_box maliyeti, operatör önbelleği yok.
- **Güvenlik:**
  - Komut sunucuda üretilir ve doğrulanır: sayılar aralıkla; adlar, süreler, dal, adres ve yollar desenle (yolda yalnız `$USER` / `$HOME`).
  - Değerler kabuk için tırnaklanır. Yalnız `config.sh` varsayılanından farklı olanlar yazılır.
  - Komut parola ya da token içermez; özel depo için TRUBA'da SSH anahtarı ve `git@github.com:…` adresi kullanılır.
  - Arayüz kümeye hiçbir şey göndermez.
- **Uç noktalar:** `GET /api/truba/config`, `POST /api/truba/command` (geçersiz değer → 422).

## 4. Çalıştırma

```bash
# geliştirme
pip install -r requirements.txt -r requirements-web.txt
RFCAV_CHECKPOINT=runs/large RFCAV_DATA=/arf/scratch/$USER/rfcav3d/E_ms0.1_k10_v2 \
    uvicorn src.service.api:app --port 8000          # checkpoint yoksa eğitilmemiş demo modeli; RFCAV_DATA: dataset kökleri (':' ile)
cd web && npm install && npm run dev                                     # http://localhost:5173 (/api → :8000)

# tek süreç (derlenmiş arayüz FastAPI'den servis edilir)
cd web && npm run build && cd .. && uvicorn src.service.api:app --port 8000   # http://localhost:8000

# konteyner
mkdir -p models && cp -r <runs/EXP> models/large
docker compose up --build                                                 # http://localhost:8000
```

**Ortam değişkenleri:**
- `RFCAV_CHECKPOINT`: `.ckpt` dosyası ya da eğitim klasörü.
- `RFCAV_DEVICE`: `cpu` | `cuda`.
- Sınırlar: `RFCAV_MAX_UPLOAD_MB` (25), `RFCAV_MESH_TIMEOUT` (120 s), `RFCAV_MESH_MEM_MB` (4096), `RFCAV_MAX_TETS` (400k).
- Bellek deposu: `RFCAV_MAX_ITEMS` (64), `RFCAV_TTL_S` (3600).

**Ölçüm** (4 çekirdek CPU, küçük model, ~15k tet eliptik kavite):
- örnek geometri ~2 s,
- tahmin toplam ~1.7 s: operatörler 0.7 s, model 0.4 s, QoI 0.2 s.

## 4b. Telefondan kullanım

`scripts/serve_web.py` arayüzü telefondan açılacak şekilde başlatır:
1. Gerekirse arayüzü derler.
2. API'yi bir **erişim anahtarıyla** başlatır (`RFCAV_TOKEN`).
3. Bağlantıyı ve telefon kamerasıyla okutulacak **QR kodu** yazar.

| nerede | komut | telefondaki bağlantı |
|---|---|---|
| **Colab** (en kolayı) | `Colab_WebUI.ipynb`: sıfırdan tek hücre (§4c) | `https://….trycloudflare.com/?token=…` (her yerden) |
| kendi bilgisayarın, aynı Wi-Fi | `python scripts/serve_web.py --checkpoint runs/large --data <dataset kökü>` | `http://<bilgisayarın IP'si>:8000/?token=…` |
| kendi bilgisayarın, her yerden | aynı komut + `--tunnel` | `https://….trycloudflare.com/?token=…` |

- **Tünel:** `--tunnel` bir Cloudflare quick tunnel açar. Hesap gerekmez; `cloudflared` yoksa script indirir. Bağlantı, script çalıştığı sürece geçerlidir.
- **Erişim anahtarı:**
  - Bağlantı ilk açıldığında anahtar HttpOnly bir çereze yazılır.
  - Anahtarsız istekler 401 alır. Anahtar her başlatmada yeniden üretilir.
  - Bağlantıyı bir parola gibi paylaş.
  - `--no_token` yalnız güvenilir bir ağda kullanılmalı.
- **Telefon arayüzü:** 860 px altında alt gezinme çubuğu; tuval ekranı doldurur, Geometri / Gösterim / Modlar alttan çıkan sayfalarda (§3.2). 3D görünüm dokunmatik: tek parmakla döndür, iki parmakla yakınlaştır.
- **TRUBA:** hesaplama düğümleri internete kapalı, dolayısıyla telefondan doğrudan erişilemez. Ya Colab'ı kullan, ya da checkpoint'i ve datasetleri kendi bilgisayarına kopyalayıp orada çalıştır.

## 4c. Colab: sıfırdan tek hücre

`Colab_WebUI.ipynb` boş bir Colab defterinde tek hücreyle her şeyi kurar (hücre kodu yeni bir deftere aynen yapıştırılabilir):

1. Drive'ı bağlar, repoyu `/content/rf_cavity_neural_operator`'a indirir ya da `BRANCH`'e günceller.
2. `scripts/colab_webui.py`:
   - gmsh sistem kütüphaneleri (apt),
   - `requirements.txt` + `requirements-web.txt` (pip),
   - Node.js 22,
   - arayüz derlemesi (`web/dist`; `web/` değişince yeniden derlenir, `web/dist/.sources`),
   - Drive klasörleri.

   Her adım yapılmışsa atlanır: aynı VM'de yeniden çalıştırmak saniyeler sürer.
3. Checkpoint: `CHECKPOINT = "auto"` → `<DRIVE_DIR>/training_logs` altında en yeni `.ckpt`'yi içeren eğitim (o klasörde en iyi `val_rel_l2`, yoksa `last.ckpt`). Bulunamaz ya da yüklenemezse eğitilmemiş demo modeli + arayüzde uyarı (`--checkpoint_optional`, `RFCAV_CHECKPOINT_OPTIONAL=1`).
4. Sunucu: `--data <DATA_DIRS>` (varsayılan hücrede `[GEN_DIR]`; `[DRIVE_DIR]` verilirse Drive'daki bütün H5'ler) + `--gen_root <GEN_DIR>` + `--runs_root <DRIVE_DIR>/training_logs` (eğitim) + tünel + QR. `GEN_DIR`, `DRIVE_DIR`'in içindeyse oradaki datasetler yalnız bir kez, `generated/…` adıyla listelenir.

Colab notları:
- **Bağlantı açılmıyorsa:** `--tunnel` bağlantısı artık tünel Cloudflare'e kaydolup dışarıdan yanıt verene kadar basılmaz (HTTP/2, UDP kapalı ağlarda da çalışır). Erken açılan yeni `trycloudflare.com` adı DNS hatası verir ve telefon / operatör DNS'i bunu birkaç dakika önbellekte tutar: bir dakika bekleyip yeniden deneyin ya da Wi-Fi ile mobil veri arasında geçin. Bağlantı yalnız hücre çalıştığı sürece yaşar ve `?token=…` kısmı olmadan "access token required" döner; bazı kurum ağları `trycloudflare.com`'u engeller.
- **Açılış yavaşsa:** Dataset sayfası açılışta `DATA_DIRS`'i 5 seviye derinliğe kadar tarar ve her H5 klasörünün ilk dosyasını açar; Drive bir ağ diski olduğundan çok parçalı büyük bir klasör dakikalar sürer. Yalnız gereken klasörü verin (`DATA_DIRS = [GEN_DIR]`). `CHECKPOINT = "auto"` da `training_logs`'u özyinelemeli tarar ve checkpoint'i Drive'dan yükler; model gerekmiyorsa (ör. CST kıyası) `CHECKPOINT = ""`. Yeni bir VM'de ilk kurulum (apt, pip, Node.js, `npm ci`, derleme) birkaç dakika sürer; aynı VM'de tekrar çalıştırmak saniyeler.
- Sekme açık kalmalı. Boşta kalan oturumu Colab kapatır; kapanırsa hücreyi yeniden çalıştırıp aynı işi başlat (bitmiş parçalar atlanır, yarım parça baştan üretilir).
- Üretim süresi çekirdek sayısıyla ölçeklenir: arayüzdeki tahmin, sunucunun gördüğü çekirdek sayısını kullanır.
- Alan etiketlerinde işçi × thread, VM'nin çekirdeğinden ve belleğinden otomatik planlanır (§3c). Örnek:
  64 çekirdek ve ≥ 250 GB RAM'de 16 × 4. Bellek azsa daha az işçi, işçi başına daha çok thread.
- Eğitim için GPU çalışma zamanı seçin. Üretim CPU'da, eğitim GPU'da aynı anda çalışabilir.

## 5. Güvenlik notları (dışarıya açık)

- Yüklenen dosya güvenilmez veri sayılır:
  - diske rastgele bir adla yazılır, istemcinin dosya adı kullanılmaz;
  - uzantı beyaz listesi, boyut sınırı;
  - OCC/gmsh/meshio ayrı bir süreçte çalışır: süre sınırı, adres alanı sınırı (`RLIMIT_AS`), çökerse yalnız o süreç ölür.
- Mesh boyu ≥ 0.05 ve tet sayısı sınırı var: hesap ve bellek patlamasına karşı.
- Dışarı açılan sunucuda (`serve_web.py`) paylaşılan erişim anahtarı (`RFCAV_TOKEN`) zorunlu; anahtar HttpOnly çerezde tutulur.
- **v0.4'te gelecek:** giriş (davetli e-posta / OAuth), kullanıcı başına kota ve hız sınırı, HTTPS'i ters vekil (Caddy/nginx) sağlayacak.
- **Gizlilik:** tasarımlar diske yazılmaz, 1 saatte bellekten düşer.
- **Lisans:** gmsh GPL'dir. Servis olarak çalıştırmak dağıtım sayılmaz, ama masaüstü dağıtımda GPL geçerli olur. vtk.js BSD, OCC LGPL.

## 6. Yol haritası

| sürüm | içerik |
|---|---|
| **v0.1b** ✓ | geliştirme aracı: Dataset sayfası (genel bakış, tablo), FE etiketleri, FE / Model / Fark, girdi özellikleri, mesh görünümü |
| **v0.1** ✓ | servis katmanı, API, React + vtk.js görünüm, faz animasyonu, mod tablosu, kavite değerleri, η, eksen grafiği, dışa aktarma, Dockerfile |
| v0.2 | parametrik kurucu (aile başına parametre şeması + önizleme), OOD skoru, iş kuyruğu + "FE ile doğrula" |
| v0.3 | tarama (parametreye göre f, R/Q eğrileri) ve hedef frekansa ayarlama, PDF rapor |
| v0.4 | giriş, kota, hız sınırı, sunucuya dağıtım, kullanım ve gizlilik sayfası |

## 7. Bilinen sınırlar (v0.1)

- **Bellek deposu:** tek süreç için tasarlandı. Çok worker'lı uvicorn ya da yeniden başlatma sonuçları kaybeder (v0.2: Redis).
- **Gösterim P1:** p2 alanı P2 düğümlerinin alt-tet'lerinde doğrusal çiziliyor (§1d); ince mesh'lerde tepenin ~1e-3'ü kadar hata taşır
  (p2 çözümün kendisinden ~10–15 kat fazla). Tam değerler `/plane`, `/volume`'dadır. Büyük örneklerde `/fields` büyük (27k tet: 11 MB ham).
- **Kesit kenarları:** düzlem düzenli bir ızgarayla örnekleniyor; duvar kenarı ızgara çözünürlüğünde basamaklı görünür.
- **Docker imajı:** derleme bu geliştirme ortamında yalnız kısmen doğrulandı. Frontend aşaması (`npm ci`) geçti. Debian paketleri ve download.pytorch.org ağ politikası nedeniyle indirilemedi. Gerçek bir makinede `docker compose up --build` ile denenmeli.
