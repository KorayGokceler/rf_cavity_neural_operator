# 19 — 3D Veri Hattı: Maxwell Kavite Modları (H-alanı, Nédélec N0)

> **Kapsam:** docs/18 Faz 2'nin **veri** yarısı: üretici (`src/data_gen/dataset_generator_3d.py`), dönüştürücü (`src/data/dataset_converter_3d.py`, `convert_3d.py`) ve model tarafının dayandığı PKL sözleşmesi. Model/eğitim kodu bu notun kapsamında değil.
> **Ortam:** scikit-fem 12.0.2, gmsh 4.15.2, SciPy 1.17.1 (SuperLU). 4 çekirdekli CPU. Yeni bağımlılık yok.

## 1. Fizik ve çözücü

- **Formülasyon (docs/18 §1.4):** $\mathbf H\in H(\mathrm{curl})$, $(\nabla\times\mathbf H,\nabla\times\mathbf v)=k^2(\mathbf H,\mathbf v)$. **Kenar DOF'larında esas sınır koşulu yok.** PEC'de $\mathbf n\cdot\mathbf H=0$ ve $\mathbf n\times\nabla\times\mathbf H=0$ ($\Leftrightarrow\mathbf n\times\mathbf E=0$) doğaldır. Frekans $f=c\,k/2\pi$.
- **Ayrıklaştırma:** skfem `ElementTetN0`, tüm kenarlar üzerinde. $K$ curl–curl, $M$ kütle matrisidir.
- **Çekirdek:** Tüm P1 fonksiyonlarının gradyanları, $G$ (kenar × tüm düğümler). Sabitin gradyanı sıfırdır, bu yüzden 0. düğüm sabitlenir ($G$'nin 0. sütunu atılır) ve $K_p=G^\top MG$ SPD olur. Çözücü araştırma tarifinin aynısıdır (`n0lib.solve_projected`): $\sigma<0$ ile shift-invert ve her çözümden sonra $P=I-GK_p^{-1}G^\top M$ uygulanır.
- **Hız:** $K-\sigma M$ ve $K_p$ SPD olduğundan SuperLU pivotlamasız çalıştırılır (`diag_pivot_thresh=0`, `SymmetricMode`). 10k DOF'ta faktör süresi 7.2 s'den 0.35 s'ye iner ve dolum aynı kalır.
- **Denetimler (her örnekte):**
  - $\lambda_{min}>10^{-6}\,\bar K_{ii}/\bar M_{ii}$;
  - $\max|G^\top MU|/\max|MU|<10^{-6}$ (ölçülen değer $\sim10^{-14}$);
  - $U^\top MU=I$.

  $K+1$ mod çözülür. $(K{+}1)$. modun frekansı `freq_next` olarak saklanır; $f_{next}\approx f_K$ ise $K$ kesimi dejenere bir kümeyi bölüyor demektir.
- **Topoloji kısıtı: kulp yok.** Birinci Betti sayısı $b_1>0$ olan bir domainde (delik, torus, iki plakaya değen iç iletken) $\mathbf H$-çekirdeğine $b_1$ tane harmonik alan eklenir. Bu alanlar gradyan projeksiyonuyla silinmez ve $\lambda=0$ sahte modu olarak görünür. Test: torusta $\lambda_1=1.8\cdot10^{-12}$ çıkar ve üretici bu örneği reddeder. Her mesh çözümden önce topolojik top olarak sertifikalanır: bağlı olmalı, $V-E+F-T=1$ ve tek sınır kabuğu için $\chi(\partial\Omega)=2$ olmalıdır ($\Rightarrow b_1=b_2=0$). Reddedilen rastgele geometri yeniden çekilir (`--max_geom_tries`).

## 2. CLI

```bash
# kalibrasyon (çift id: PEC kutu 10×8×6 cm, tek id: pillbox R=4, L=5 cm)
python -m src.data_gen.dataset_generator_3d --mode calibration --n_total 2 --n_eigen_modes 8 --h5_filename calib3d.h5
# rastgele aileler
python -m src.data_gen.dataset_generator_3d --n_total 1000 --n_eigen_modes 6 --mesh_size 0.10 \
    --families pillbox axisym_cell blob --seed 0 --n_workers 4 --sample_timeout 300 --h5_filename rf3d.h5
python convert_3d.py --h5_filepath rf3d.h5 --output_path data/rf3d.pkl [--modes 0 1 2] [--no_operators]
```

Parametreler:

- `--mesh_size`: tet boyutu, $V^{1/3}$'e göre göreli. `--mesh_size_abs` verilirse metre cinsinden mutlak boyut kullanılır.
- Tohum: örnek $s$ için `default_rng([seed, s])`. Sonuç işçi sayısından bağımsızdır.
- H5 dosyası önce `.partial` uzantısıyla yazılır, iş bitince atomik olarak yeniden adlandırılır.

**Aileler** (gmsh OCC, metre):

| Aile | Parametreler |
|---|---|
| `pillbox` | $R\in[3,6]$ cm, $L\in[2,10]$ cm |
| `axisym_cell` | Eliptik hücreye benzer meridyen spline'ı: $r(z)=r_{uç}+(r_{eq}-r_{uç})\sin^\alpha(\pi s^\gamma)+$ pencereli 2 harmonik. Profil $z$ ekseni etrafında `revolve` ile döndürülür. Uç plakalar düz ve PEC'dir. $L\in[4,10]$, $r_{eq}\in[3.5,6]$ cm, $r_{uç}/r_{eq}\in[0.3,0.7]$ |
| `blob` | Silindir ya da kutu tabana 1–3 elipsoit veya kutu çıkıntısı eklenir (`fuse`) ya da oyulur (`cut`). Çıkıntılar rastgele döndürülür, merkezleri yüzeydedir, boyutları $\le0.4\,d_{min}$'dir. Sonuç tek hacim olmalı ve topoloji sertifikasından geçmelidir |

**H5 (örnek başına `sample_XXXX`):**

- `nodes` [Nv,3]
- `tets` [Nt,4]
- `edges` [Ne,2]: skfem DOF sırası, (kuyruk, baş), her zaman kuyruk < baş
- `h_edges` [Ne,K]: $\int_{kuyruk\to baş}\mathbf H\cdot d\mathbf l$, fiziksel birimde $M$-ortonormal, işaret keyfi
- `freqs` [K] GHz
- attrs: `shape_type`, `geom_params` (JSON) ve her parametre ayrı attr olarak, `freq_next`, `div_residual`, `mesh_h`, `volume`, `t_mesh`, `t_solve`. Kalibrasyonda ek olarak `freqs_analytic`.

Dosyanın `metadata` attr'ı (JSON) formülasyonu ve DOF/topoloji konvansiyonlarını içerir.

## 3. PKL sözleşmesi (model ajanı buna dayanır)

```
geometry_pool[g_id]:
  X            float32 [Nv,3]   (x − center)/scale, center = hacim ağırlıklı ağırlık merkezi, scale = max|x − center|
  Input_funcs  float32 [Nv,9]   FEATURE_NAMES_3D
  edges        int64 [Ne,2]     low→high; satırlar sözlük sıralı = np.unique(sıralı tet kenarları)  ← DOF sırası
  tets         int64 [Nt,4]
  M, K         CSR (indptr int64, indices int64, data float64) [Ne×Ne], NORMALİZE mesh
  G            CSR [Ne×Nv]  G[e,high]=+1, G[e,low]=−1
  Kp           CSR [Nv×Nv]  GᵀMG (P1 Neumann; sabitlerde tekil → tüketici sabitler/ridge)
  scale float, center float64[3], shape_type str, n_nodes, n_edges,
  torsion_max, volume (normalize), freq_next (GHz)                     ← ek anahtarlar
samples[i]: geom_id, Y float32 [Ne] (normalize mesh'te ‖Y‖_M = 1, işaret keyfi),
            Theta float32 [3] = [mode_idx, f_GHz, sample_id]
metadata: freq_stats{mean,std,mode_stats}, n_modes, mode_indices, field='H', element='N0',
          feature_names, n_samples, n_geometries, operators_stored, rayleigh_max_rel_err, ...
```

Konvansiyonlar:

- **Baz fonksiyonu:** $\mathbf w_{ab}=\lambda_a\nabla\lambda_b-\lambda_b\nabla\lambda_a$ ($a<b$) ve $\nabla\times\mathbf w_{ab}=2\nabla\lambda_a\times\nabla\lambda_b$. DOF, $a\to b$ yönündeki çizgi integralidir.
- **Montaj:** `assemble_n0` kapalı formdur (numpy). skfem `ElementTetN0` ile $10^{-12}$ düzeyinde aynıdır (test edildi).
- **Ölçek:** $\lambda_{norm}=\lambda_{fiz}\,\mathrm{scale}^2$, buradan $f=c\sqrt{\lambda_{norm}}/(2\pi\,\mathrm{scale})$.
- **Mod indeksi:** `Theta[0]`, `mode_indices` içindeki sıradır. Varsayılan "tüm modlar" olduğundan ham mod indeksine eşittir.
- **Özellikler** (`FEATURE_NAMES_3D`):
  - `x, y, z`;
  - `dist_to_boundary`: $\partial\Omega$'nın en yakın noktasına tam nokta–üçgen uzaklığı (normalize birim);
  - `dir_bnd_x/y/z`: o noktaya birim yön; sınır düğümlerinde dış normal;
  - `node_volume`: $\sum|T|/4$, maksimuma bölünmüş;
  - `torsion`: $w/\max w$ (P1).

  Medial eksendeki eşit uzaklıklı noktalarda yön doğası gereği süreksizdir.
- **Operatörler, float32 `X`'ten yeniden kurulursa:** `geometry_operators(X, tets)` ile. $K$'deki göreli fark $<10^{-5}$ olur ve Rayleigh $10^{-5}$ içinde kalır.
- **`--no_operators`:** M/K/G/Kp'yi PKL'den çıkarır. Bunlar dosya boyutunun ~%89'udur. Yükleyici operatörleri `geometry_operators` ile yeniden kurar (50k kenarda ~1.5 s).

## 4. Doğrulama

**Kalibrasyon**, ilk 8 mod, $\max|f/f_{analitik}-1|$ ($h=$ `mesh_size`$\cdot V^{1/3}$):

| `mesh_size` | kutu Ne | kutu hata | pillbox Ne | pillbox hata | pillbox TM010 |
|---|---|---|---|---|---|
| 0.20 | 1 950 | $9.6\cdot10^{-3}$ | 1 345 | $2.1\cdot10^{-2}$ | $+1.7\cdot10^{-2}$ |
| 0.15 | 2 554 | $8.1\cdot10^{-3}$ | 2 392 | $1.2\cdot10^{-2}$ | $+1.1\cdot10^{-2}$ |
| 0.12 | 4 616 | $2.8\cdot10^{-3}$ | 4 357 | $7.5\cdot10^{-3}$ | $+6.7\cdot10^{-3}$ |
| 0.10 | 7 285 | $2.0\cdot10^{-3}$ | 6 859 | $4.8\cdot10^{-3}$ | $+4.8\cdot10^{-3}$ |
| 0.07 | 19 595 | $3.0\cdot10^{-4}$ | 18 276 | $2.4\cdot10^{-3}$ | $+2.4\cdot10^{-3}$ |
| 0.05 | 50 074 | $2.2\cdot10^{-4}$ | 47 231 | $1.2\cdot10^{-3}$ | $+1.2\cdot10^{-3}$ |

- **Yakınsama:** $O(h^2)$. Pillbox'ta hata H-N0 TM010 tarafından belirlenir ($H_\varphi\propto J_1$, eğri duvar düz yüzeylerle yaklaşıklanıyor). Hata pozitiftir, yani özdeğerler gerçeğin üstünde çıkar (docs/18 §3.5 ile tutarlı).
- **Testlerdeki eşik:** `mesh_size 0.12`'de kutu için $5\cdot10^{-3}$, pillbox için $10^{-2}$.

**Diğer değişmezler** (`tests/test_data_gen_3d.py`, `tests/test_dataset_converter_3d.py`, 19 test, ~7 s):

- Gradyan ve sıfır mod:
  - $\max|G^\top MU|/\max|MU|\sim10^{-14}$;
  - $\lambda_{min}\gg0$;
  - $\|KG\|/\|K\|<10^{-11}$;
  - $K_p\mathbf 1=0$.
- Rayleigh ve normalizasyon:
  - saklanan $K,M$ ile Rayleigh($Y$) frekansı tekrar üretir (smoke'ta en büyük göreli hata $5\cdot10^{-15}$, float32 $Y$ ile $<10^{-5}$);
  - $\|Y\|_M=1$;
  - $G^\top MY\approx0$.
- Ölçek değişmezliği: ölçekli ve ötelenmiş girdi aynı `X`, özellikler, $M$ ve $K$'yi verir. Fiziksel matrisler için $M_{fiz}=\mathrm{scale}\,M$ ve $K_{fiz}=K/\mathrm{scale}$ sağlanır.
- Kenar yönü:
  - $e_x$ sabit alanının DOF'u $x_{high}-x_{low}=G\,x$'tir ve $\|e_x\|^2_M=|\Omega|$ olur;
  - ters yönlü H5 kenarları işaretle doğru eşlenir.
- Kutuda uzaklık özelliği analitik değerle birebir aynıdır.
- Tohum tekrar üretilebilir. Üç ailenin hepsi topolojik top verir.

## 5. Smoke verisi ve maliyet

`smoke3d.h5` / `smoke3d.pkl`:

- 30 geometri: 9 pillbox, 9 axisym_cell, 12 blob.
- Üretim parametreleri: `--mesh_size 0.10 --n_eigen_modes 6 --seed 0`.
- Toplam 180 örnek.

| Aile | Nv | Ne | Nt | mesh + çözüm (tek işçi) |
|---|---|---|---|---|
| pillbox | 1 169–1 267 | 6 789–7 218 | 4.9–5.2k | 0.24 + 0.81 s |
| axisym_cell | 1 157–1 211 | 6 788–7 163 | 4.9–5.3k | 0.23 + 0.86 s |
| blob | 1 284–1 659 | 7 353–9 528 | 5.3–6.8k | 0.48 + 1.00 s (en fazla 2.1 s) |

- **Süre ve boyut:** 4 işçiyle 30 örnek 12 s sürdü. H5 13.3 MB (0.44 MB/örnek). PKL 140 MB (4.7 MB/geometri, operatörlerle), `--no_operators` ile 15.9 MB (0.53 MB/geometri). Dönüştürme 0.2 s/geometri.
- **Frekans:** 1.61–6.19 GHz aralığında; ortalama 3.71, std 0.83.
- **Küme bölünmesi:** 30 geometrinin 5'inde $f_{next}/f_6-1<10^{-3}$ (4 axisym_cell, 1 pillbox). Yani eksenel simetride $K=6$ çoğu zaman bir dejenere çifti bölüyor. Model tarafında `freq_next` kullanılmalı veya $K$ kümeye göre seçilmelidir.

**Üretim çözünürlüğüne ölçekleme** (4 çekirdek, 4 işçi, işçi başına bir SuperLU; 1000 örnek; H5 gzip; PKL float32 `Y`):

| `mesh_size` | Ne | örnek başına (tek işçi) | tepe RSS/işçi | 1000 örnek süresi | H5 | PKL (tam / `--no_operators`) | doğruluk (kutu / pillbox) |
|---|---|---|---|---|---|---|---|
| 0.10 | ~7–9k | ~1.2 s | 0.15 GB | ~6 dk | 0.45 GB | 4.7 / 0.5 GB | $2\cdot10^{-3}$ / $5\cdot10^{-3}$ |
| 0.07 | ~19k | ~4.5 s | 0.3 GB | ~20 dk | ~1.2 GB | ~12 / ~1.4 GB | $3\cdot10^{-4}$ / $2.4\cdot10^{-3}$ |
| 0.05 | ~50k | ~31 s | 0.85 GB | ~2.2 saat | ~3 GB | ~30 / ~3.5 GB | $2\cdot10^{-4}$ / $1.2\cdot10^{-3}$ |

Öneriler:

- **Hedef:** Eğitim verisi için `mesh_size 0.07` (~%0.1–0.25) iyi bir denge noktasıdır.
- **İnce mesh:** 0.05 ve altında `--no_operators` kullanılmalıdır.
- **Maliyet dağılımı:** Çözüm süresinin çoğu ARPACK iterasyonlarından gelir (~60 çözüm × LU geri yerine koyma). pypardiso/CHOLMOD veya NGSolve $p=2$–3 eğri elemanlar bu süreyi 5–10× kısaltır (docs/18 §2.4). Bu ortamda kurulmadılar.
- **Bellek ölçeklemesi:** Bellek ve faktör süresi $\sim N_e^{1.5}$ ile büyür.
- **Ne > 100k:** Örnek başına ~2–3 dk ve ~3 GB RAM gerekir. İşçi sayısı RAM'e göre düşürülmelidir.

## 🔗 Bağlantılar

- [18_3D_EXTENSION.md](18_3D_EXTENSION.md): fizik, çözücü seçenekleri, Schur-projekteli Gram (§3.4)
- [01_DATA_GENERATION.md](01_DATA_GENERATION.md), [02_FEATURE_ENGINEERING.md](02_FEATURE_ENGINEERING.md): 2D karşılıkları

#3d #maxwell #nedelec #hcurl #veri-hattı #sözleşme
