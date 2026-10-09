# 26 · Web arayüzü: RF Cavity Neural Solver

> Durum: **v0.1** (branch `claude/web-ui`). Hedef: dışarıya açık ürün/demo. Teknoloji: FastAPI +
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
- **v0.1'de** sonuçlar bellekte tutulur: LRU, en fazla 64 kayıt, 1 saat, diske yazılmaz.
- **v0.2'de** iş kuyruğu (Redis + RQ) gelir: FE doğrulama, tarama.

## 2. API (v0.1)

| uç nokta | ne yapar |
|---|---|
| `GET /api/info` | model (alan, parametre sayısı, cihaz, eğitilmemiş mi), aileler, sınırlar, kurallar |
| `POST /api/geometries` (multipart: `file`, `unit`, `mesh_size`, `n_cells`) | mesh + kontroller (tek parça / manifold, b1, hacim, bbox) + görüntü yüzeyi |
| `POST /api/geometries/sample` `{family, id?, mesh_size}` | üretici geometrisi (id boşsa eğitimde kullanılmamış rastgele bir kimlik) |
| `POST /api/predictions` `{geometry_id}` | mod tablosu (f, kavite değerleri, η, dejenere, hızlandırıcı mod), süreler, uyarılar |
| `GET /api/predictions/{id}/modes/{k}` | yüzey E/H (faz için iki alan) + eksen boyunca E_z |
| `GET /api/predictions/{id}/plane?mode&axis&pos&res` | kesit düzlemi: noktalar, üçgenler, E, H |
| `GET /api/predictions/{id}/export` | JSON dosyası |

**Faz kaydırıcısı tarayıcıda çalışır.** E ve H bir kez gönderilir, φ istemcide uygulanır; sunucuya gidip gelme yoktur.

## 3. Arayüz (v0.1)

- **Sol panel:** "Örnek kavite" (aile + kimlik + mesh boyu) ya da "Dosya yükle" (CAD/mesh, birim, hücre sayısı, mesh boyu).
  - Kontrol rozetleri: ✓ kapalı, tek parça; kulp sayısı b1; tet sayısı; hacim; OOD.
  - "Tahmin et" tuşu.
- **Orta:**
  - **3D görünüm:** yarı saydam kavite ve renkli kesit düzlemi. Kamera kesitin normalinden, hafif eğik bakar.
  - **Kontroller:** E/H, bileşen (|·|, x, y, z), kesit ekseni ve konumu (mm), faz 0–180° kaydırıcısı ve ▶ oynat, yüzey (saydam / duvar alanı / gizli).
  - **Eksen grafiği:** eksen boyunca E_z(z); tek seri, artı imleç ve ipucu.
- **Sağ:**
  - Mod tablosu: ★ hızlandırıcı mod, ≈ dejenere çift, η rozeti (✓ iyi < %2, ! orta < %10, ✕ zayıf).
  - Seçili modun bütün kavite değerleri, süreler, JSON indir.
- **Renkler:** büyüklük için tek tonlu sıralı mavi rampa, işaretli bileşen için mavi ↔ gri ↔ kırmızı. Rampalar açık ve koyu temada ayrı seçildi.
- **Dil:** TR/EN.
- **Uyarılar:** eğitilmemiş model, OOD aile, ışın ekseni kavitede değil.

## 3b. Geliştirme aracı: dataset, özellikler, FE etiketleri, model–FE karşılaştırması

Arayüz geliştirme için de kullanılır. Geometriden model girdisine, FE çözümüne ve modelin hatasına kadar her şey görülebilir.

- **Dataset sayfası:** sunucuya `RFCAV_DATA` ile verilen datasetler listelenir (`src/service/datasets.py`).
  - Her **PKL** dosyası bir dataset; **H5 parçaları** klasör başına bir dataset. TRUBA düzeninde `h5/<aile>/` her aile için ayrı bir dataset olur.
  - **Genel bakış:** aile başına geometri sayısı (tıklayınca filtreler), en düşük mod frekansı f₀ histogramı, N0 DOF sayısı histogramı.
  - **Tablo:** id, aile, düğüm, N0, b1, f₀…f₅. Sıralanabilir, sayfalı, aileye göre filtrelenir. "Aç" geometriyi çalışma alanına gönderir.
- **Geometri açılınca:**
  - mesh ve kontroller;
  - **FE (etiket) çözümü:** modlar, alanlar, kavite değerleri. Değerler aynı operatörlerle yeniden hesaplanır; PKL'de saklı etiket varsa değerin ipucunda gösterilir;
  - geometri bilgisi: dataset, dosya, id, N0 DOF, ölçek, torsiyon maksimumu, (K+1). mod frekansı, üretici parametreleri.
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

**Bellek:** PKL'ler sunucu belleğine bir kez yüklenir. `RFCAV_MAX_PKL_GB` (8) sınırının üstündeki PKL'ler için H5 parçalarını göster; H5 listesi yalnız öznitelikleri okur ve hızlıdır.

## 4. Çalıştırma

```bash
# geliştirme
pip install -r requirements.txt -r requirements-web.txt
RFCAV_CHECKPOINT=runs/large RFCAV_DATA=/arf/scratch/$USER/rfcav3d/E_ms0.1_k10_v1 \
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
| **Colab** (en kolayı) | `Colab_Maxwell3D.ipynb` hücre 15 | `https://….trycloudflare.com/?token=…` (her yerden) |
| kendi bilgisayarın, aynı Wi-Fi | `python scripts/serve_web.py --checkpoint runs/large --data <dataset kökü>` | `http://<bilgisayarın IP'si>:8000/?token=…` |
| kendi bilgisayarın, her yerden | aynı komut + `--tunnel` | `https://….trycloudflare.com/?token=…` |

- **Tünel:** `--tunnel` bir Cloudflare quick tunnel açar. Hesap gerekmez; `cloudflared` yoksa script indirir. Bağlantı, script çalıştığı sürece geçerlidir.
- **Erişim anahtarı:**
  - Bağlantı ilk açıldığında anahtar HttpOnly bir çereze yazılır.
  - Anahtarsız istekler 401 alır. Anahtar her başlatmada yeniden üretilir.
  - Bağlantıyı bir parola gibi paylaş.
  - `--no_token` yalnız güvenilir bir ağda kullanılmalı.
- **Telefon arayüzü:** 760 px altında tek sütun. Sıra: geometri → 3D görünüm → kontroller → modlar → ayrıntılar → geometri bilgisi. 3D görünüm dokunmatik: tek parmakla döndür, iki parmakla yakınlaştır.
- **TRUBA:** hesaplama düğümleri internete kapalı, dolayısıyla telefondan doğrudan erişilemez. Ya Colab'ı kullan, ya da checkpoint'i ve datasetleri kendi bilgisayarına kopyalayıp orada çalıştır.

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
- **Kesit kenarları:** düzlem düzenli bir ızgarayla örnekleniyor; duvar kenarı ızgara çözünürlüğünde basamaklı görünür.
- **Docker imajı:** derleme bu geliştirme ortamında yalnız kısmen doğrulandı. Frontend aşaması (`npm ci`) geçti. Debian paketleri ve download.pytorch.org ağ politikası nedeniyle indirilemedi. Gerçek bir makinede `docker compose up --build` ile denenmeli.
