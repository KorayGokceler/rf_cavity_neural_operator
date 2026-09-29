# 19 — 3D Veri Hattı: Maxwell Kavite Modları (E-alanı varsayılan, H isteğe bağlı; Nédélec N0)

> **Güncelleme (`claude/3d-e-formulation`):** Varsayılan formülasyon artık **E** (`--field E`); kulplu
> kaviteler (spoke, yarım dalga koaksiyel, DTL gövde + sap) ve yüzen iç iletkenler desteklenir. Ayrıntılar,
> ölçümler ve yeni aileler: **§6 E formülasyonu**. §1–§5 H formülasyonunu anlatır (`--field H` ile aynen
> çalışmaya devam eder).

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
    --families elliptical reentrant pillbox_pipes freeform --seed 0 --n_workers 4 --sample_timeout 300 --h5_filename rf3d.h5
python convert_3d.py --h5_filepath rf3d.h5 --output_path data/rf3d.pkl [--modes 0 1 2] [--no_operators]
```

Parametreler:

- `--mesh_size`: tet boyutu, $V^{1/3}$'e göre göreli. `--mesh_size_abs` verilirse metre cinsinden mutlak boyut kullanılır.
- Tohum: örnek $s$ için `default_rng([seed, s])`. Sonuç işçi sayısından bağımsızdır.
- H5 dosyası önce `.partial` uzantısıyla yazılır, iş bitince atomik olarak yeniden adlandırılır.

**Geometri çeşitliliği — g3** (`claude/3d-geometry-diversity`; notebook `GEOM_VER = "g3"`):

| Ne | Nasıl | Bayrak / yer |
|---|---|---|
| **Düzgün rastgele deformasyon** (tüm aileler) | Mesh'ten sonra $x \to x + \delta(x)$, $\delta = \sum_m a_m \sin(\omega_m\cdot x + \varphi_m)$ (3–8 dalga, dalga boyu 0.25–1.5 × köşegen). $\|\nabla\delta\|_2 \le \sum|a_m||\omega_m| = L < 1$ ölçeklenir ⇒ $\det(I+\nabla\delta) \ge (1-L)^3 > 0$: birebir, hiçbir tet ters dönmez, topoloji aynı. Ayrı RNG akışı ⇒ aynı id farklı mesh boyutunda aynı $\delta$ (çözünürlük çalışmasıyla uyumlu). Yan etki: eksenel simetrik şekillerin dejenere çiftlerini böler | `--deform_prob 0.5 --deform_max 0.5` |
| **elliptical v2** | Gerçek tasarım ankrajları (TESLA orta hücre; ILC low-loss yaklaşık) ±%10 + boyut ölçekleme (%30); düşük-β hücreler ($L \propto \beta$, β∈[0.5,0.95], %25); farklı uç yarım-hücreler (%60; uç iris = tüp yarıçapı ×[1, 1.35]); 1–5 hücre (0.35/0.25/0.2/0.1/0.1); tüplerde 1–3 radyal kuplör portu (FPC/HOM, %35). Mesh boyutu hücre başına hacimden | `cavity_shapes.draw_elliptical` |
| **reentrant v2** | Koni açısı −25…35°: negatif → mantar burun | `build_reentrant` |
| **ridged_box** (yeni) | Dikdörtgen kavite, 1–2 tam boy sırt (sırtlı dalga kılavuzu rezonatörü), %40 ışın tüpü, tüm kenarlar yuvarlatılmış (keskin kenarlı düz kutu OOD'da kalır) | `build_ridged_box` |
| **composite** (yeni) | CSG ağacı: silindir/elipsoit gövde + 1–4 ekli silindir (saplama) / elipsoit (lob), derinlik ≤ 2, + 0–2 elipsoit girinti; ağaç yapısı delik oluşturmaz, oluşursa topoloji sertifikası reddeder | `build_composite` |
| **Sobol örnekleme** | Her örneğin ilk 64 rastgele çekimi (aile + şekil parametreleri) id başına bir karıştırılmış Sobol noktasından: aileler ve parametre aralıkları düzgün kapsanır (64 örnekte aile sayıları ±1); aile yeniden çekimlerde sabit | `--sampling sobol` |
| **Aktif örnekleme** | Adaylar yalnız mesh'lenir, model tahmin eder, $\eta_k = \|K u_k - \lambda_k M u_k\|_{D^{-1}}/\lambda_k$ (etiketsiz artık) ile sıralanır; en kötüler `--ids_file` ile etiketlenir | `scripts/active_sampling.py`, notebook hücre 13 |
| **Kapsama raporu** | Aile başına sayı, boyut / en-boy / frekans / kenar sayısı aralıkları, deformasyon / ankraj / düşük-β / port / dejenere payları | `scripts/dataset_stats_3d.py`, hücre 5b |

**Sınır (yalnız `--field H`):** spoke kaviteler ve iki ucuna değen iç iletkenli koaksiyel (yarım dalga) rezonatörler kulplu topolojidir ($b_1 \ge 1$); H formülasyonunda harmonik (λ = 0) alanlar oluşturur, topoloji sertifikasıyla reddedilir. **E formülasyonu (varsayılan) bu sınırı kaldırır — §6.**

**Aileler — g2 (varsayılan, `src/data_gen/cavity_shapes.py`, metre).** Eksenel simetrik
kaviteler $(z, r)$ meridyen profilinden $z$ ekseni etrafında döndürülür. Işın tüpleri düz PEC
kapaklarla kapatılır (kapalı özmod çözümü, CST'deki gibi). Tet boyutu, en küçük detaya göre
sınırlanır: $h=\max(\min(h_V, h_{detay}), 0.7\,h_V)$.

| Aile | Parametreler |
|---|---|
| `elliptical` | TESLA tipi eliptik hücre, 1–3 hücre (olasılık 0.5/0.3/0.2) + ışın tüpleri. Yarım hücre: iris elipsi $(a,b)$ + ekvator elipsi $(A,B)$ + ortak teğet doğru. $R_{eq}\in[4,11]$ cm, $R_{iris}/R_{eq}\in[0.25,0.42]$, $L/R_{eq}\in[0.45,0.65]$, $A/L\in[0.6,0.85]$, $B/A\in[0.85,1.25]$, $a/L\in[0.15,0.3]$, $b/a\in[1,1.9]$; içe dönük (re-entrant) duvar reddedilir. **Doğrulama:** TESLA orta hücre boyutları, tüplü tek hücre → yakınsamış TM010 = 1.288 GHz (Richardson; gerçek yapının periyodik π-modu 1.300), mesh 0.10'da +%1.3; 3 hücre: 3 TM010 geçiş bandı modu 1.294–1.317 GHz (mesh 0.07; genişlik ≈ %1.8, TESLA hücreler arası kuplajı ≈ %1.9 ile aynı mertebe) |
| `reentrant` | Burun konili (klystron/IOT tipi): $R\in[3,7]$ cm, $L_c/R\in[0.4,1]$, tüp $r_p/R\in[0.1,0.25]$, burun aralığı $g/L_c\in[0.15,0.6]$, uç kalınlığı $[0.08,0.2]R$, koni açısı $0$–$35°$; burun ucu, kökü ve dış köşeler yuvarlatılır (büyük dış yarıçap → toroidal duvar) |
| `pillbox_pipes` | Işın tüplü pillbox: $R\in[3,6]$ cm, $L_c/R\in[0.4,1.6]$, $r_p/R\in[0.12,0.35]$, yuvarlatılmış iris kenarı ve dış köşeler |
| `freeform` | Süperelipsoit (yarı eksen oranı $[0.45,1]$, üsler $[0.35,1.6]$: kutumsu → elipsoit → sivri) × düzgün rastgele modülasyon × 0–4 çıkıntı/girinti, ardından büküm, burulma, daralma. Hepsi birebir dönüşüm olduğundan yüzey kendini kesmez ve cisim delik içermez. Kapalı üçgen yüzey gmsh'e verilir, `classifySurfaces` + `createGeometry` ile $h$ boyutunda yeniden örülür |

**Dağılım dışı (OOD) test aileleri** (`cavity_shapes.OOD_FAMILIES`; varsayılan eğitim ailelerinde
yoktur, `--families box coax_qw pillbox_port elliptical_long junction` ile üretilir; notebook hücre 11):

| Aile | Neden OOD | Parametreler |
|---|---|---|
| `box` | Düz yüzler, keskin kenar/köşe (eğitimde köşeler yuvarlatılmış) | $a,b,d\in[3,10]$ cm; analitik spektrumla %0.16 uyum (mesh 0.10) |
| `coax_qw` | Çeyrek dalga koaksiyel: bir uç kapaktan çıkan iç iletken + kapasitif boşluk; dar halka bölge; $f_1\approx0.4$–$1$ GHz (eğitim frekanslarının altında) | $R_o\in[2,5]$ cm, $L/R_o\in[1.5,4]$, $r_i/R_o\in[0.2,0.45]$, boşluk $/L\in[0.08,0.3]$ |
| `pillbox_port` | Gövdede 1–2 radyal yan port (g3 eğitiminde portlar yalnız eliptik hücrelerin ışın tüplerinde) | $R\in[3,6]$ cm, port yarıçapı $[0.12,0.3]\min(R,L_c)$, uzunluk $[0.4,1.2]R$ |
| `elliptical_long` | 6–9 hücre (g3 eğitiminde 1–5); hücre şekli aralıkları aynı, port yok | `elliptical` ile aynı |
| `junction` | L/T/X kollu kutular (%30 dikey kol): yıldız biçimli değil, keskin iç köşeler | kol genişliği $[2,4]$ cm, kalınlık $[0.5,1.2]w$, kol uzunluğu $[1,2.5]w$ |

Değerlendirme: `python scripts/eval_3d.py --checkpoint <dir> --data_path ood.pkl --split all --csv ood.csv`
(şekil tipi kırılımı). `physics_freq` ile tahmin edilen GHz, frekans normalizasyonundan bağımsızdır; bu
nedenle OOD PKL kendi `freq_stats`'ını kullanabilir.

**Aileler — v1** (`--families pillbox axisym_cell blob`, gmsh OCC, metre):

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

**Diğer değişmezler** (`tests/test_data_gen_3d.py`, `tests/test_dataset_converter_3d.py`; E + H ile birlikte 54 test, ~40 s):

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

## 6. E formülasyonu

### 6.1 Neden

Her iki formülasyon da aynı $k^2$ spektrumunu verir; fark, curl operatörünün çekirdeğinin **topolojiye** nasıl bağlı olduğundadır:

| | H: $\mathbf H\in H(\mathrm{curl})$, doğal SK | E: $\mathbf E\in H_0(\mathrm{curl})$, $\mathbf n\times\mathbf E=0$ esas SK |
|---|---|---|
| curl çekirdeği | $\nabla H^1$ ⊕ **$b_1$ harmonik Neumann alanı** (her kulp için bir, $\mathbf n\cdot\mathbf h=0$) | $\nabla H^1_0$ ⊕ **$b_2$ harmonik Dirichlet alanı** (her yüzen iletken için bir, $\mathbf n\times\mathbf h=0$) |
| harmonik alanın ayrık karşılığı | P1 potansiyellerin gradyanı **değildir** → projeksiyon silemez → λ ≈ 0 sahte mod | $\nabla\varphi$, $\varphi$ = iletken üzerinde 1, dış duvarda 0 → **bir P1 gradyanıdır** → sınır bileşeni başına bir potansiyelle tam silinir |
| kabul edilen geometri | yalnız topolojik top ($b_1=b_2=0$) | her bağlı, manifold PEC kavite |

Yani H'nin sorunu (kulplar) kohomoloji tabanı gerektirir; E'nin sorunu (yüzen iletkenler) ise sadece **ek bir potansiyel sütunudur**. Kulplu RF yapıları (spoke, HWR, DTL sapları) E'de hiçbir ek işlem gerektirmez.

### 6.2 Ayrıklaştırma ve çözücü

- **DOF:** N0 tüm kenarlarda monte edilir (aynı $K$, $M$). Duvar kenarları (bir sınır yüzünün kenarı) atılır (esas SK). H5'e ise alan **tüm kenarlara genişletilerek** yazılır; duvar satırları tam 0'dır. Böylece kenar başına bir DOF düzeni korunur.
- **Potansiyel matrisi** (`dataset_converter_3d.e_potential_matrix`, üretici ve dönüştürücü aynı kodu kullanır): önce iç düğümler (artan indeks sırasıyla), sonra ilki hariç her sınır bileşeni için bir sütun ($=G_{tam}\cdot\mathbb 1_{bileşen}$). Sınır bileşenleri, sınır yüzlerinin (kenar ya da düğüm paylaşan) bağlı bileşenleridir. Sıralama düğüm sayısına göre azalandır (eşitlikte en küçük düğüm indeksi); bileşen 0 dış duvardır ve $\varphi=0$ referansıdır. $\varphi$ her duvar bileşeninde sabit olduğundan duvar satırları kendiliğinden sıfırdır. $K_p=G^\top MG$ **sabitleme olmadan SPD**'dir.
- **Çözücü** `solve_e_modes`: H ile aynı tarif (σ < 0 shift-invert + $P=I-GK_p^{-1}G^\top M$), serbest kenarlar üzerinde. Denetimler de aynıdır: sıfır/gradyan modu yok, $\max|G_E^\top MU|/\max|MU|<10^{-6}$ (E çekirdek potansiyelleriyle; ölçülen $\sim10^{-15}$) ve $U^\top MU=I$. `component_potentials=False` yalnız doğrulama içindir.
- **Topoloji sertifikası (E):** bağlı (`n_components == 1`) ve manifold mesh. Hiçbir yüz 2'den fazla tet'e ait olamaz, her sınır kenarı tam 2 sınır yüzünde olmalı ve $\chi(\partial\Omega)=2\chi(\Omega)$ sağlanmalıdır (3-manifold özdeşliği; sıkışmış düğüm/kenarda bozulur). $b_2=$ `n_bnd_components` − 1, $b_1=1+b_2-\chi$ hesaplanır ve saklanır (bilgi amaçlı). H sertifikası değişmedi.

### 6.3 Doğrulama (ölçülen)

**Kalibrasyon**, ilk 8 mod, $\max|f/f_{analitik}-1|$. E aşağıdan, H yukarıdan yakınsar; ikisi de $O(h^2)$, iki formülasyon gerçek değeri **sıkıştırır** (E ≤ tam ≤ H):

| `mesh_size` | kutu E | kutu H | pillbox E | pillbox H | TM010 E | TM010 H |
|---|---|---|---|---|---|---|
| 0.20 | $1.9\cdot10^{-2}$ | $9.6\cdot10^{-3}$ | $1.7\cdot10^{-2}$ | $2.1\cdot10^{-2}$ | $-0.75\%$ | $+1.72\%$ |
| 0.15 | $1.1\cdot10^{-2}$ | $8.1\cdot10^{-3}$ | $1.2\cdot10^{-2}$ | $1.2\cdot10^{-2}$ | $-0.49\%$ | $+1.08\%$ |
| 0.12 | $8.0\cdot10^{-3}$ | $2.8\cdot10^{-3}$ | $7.3\cdot10^{-3}$ | $7.5\cdot10^{-3}$ | $-0.31\%$ | $+0.67\%$ |
| 0.10 | $5.1\cdot10^{-3}$ | $2.0\cdot10^{-3}$ | $5.3\cdot10^{-3}$ | $4.8\cdot10^{-3}$ | $-0.24\%$ | $+0.48\%$ |
| 0.07 | $2.0\cdot10^{-3}$ | $3.0\cdot10^{-4}$ | $2.2\cdot10^{-3}$ | $2.4\cdot10^{-3}$ | $-0.10\%$ | $+0.24\%$ |
| 0.05 | $9.6\cdot10^{-4}$ | $2.2\cdot10^{-4}$ | $1.0\cdot10^{-3}$ | $1.2\cdot10^{-3}$ | $-0.05\%$ | $+0.12\%$ |

Test eşikleri (`mesh_size 0.12`): E kutu ve pillbox için $10^{-2}$; H için eskisi gibi $5\cdot10^{-3}$ / $10^{-2}$. **TESLA hücresi** (Richardson ref. 1.288 GHz): E-N0 TM010 mesh 0.10'da $+0.15\%$, 0.07'de $+0.03\%$ (H: $+1.34\%$ / $+0.64\%$). E'nin hızlanan mod için belirgin şekilde daha doğru olmasının nedeni, $E_z$'nin düzgün olması ve H'deki gibi çokyüzlü duvarda $H_\varphi$ hatası taşımamasıdır. E çözümü, duvar DOF'ları atıldığı için H'den biraz daha hızlıdır (kutu 0.10: 0.45 s / 0.61 s).

**Topoloji testleri:**

| Durum | Topoloji | E | H |
|---|---|---|---|
| Yarım dalga koaksiyel $R_o=50$, $r_i=15$, $L=150$ mm, $h=8$ mm (Ne 14.9k) | $b_1=1$ | $f=0.9931, 1.8001, 1.8008, 1.9846$ GHz; TEM $c/2L$ $-0.62\%$, $c/L$ $-0.70\%$ (h = 6 mm: $-0.36\%$ / $-0.41\%$); TE11 benzeri çift 1.80 GHz | sahte $\lambda=3\cdot10^{-12}$ → reddedilir |
| Torus $R=50$, $r=20$ mm | $b_1=1$ | sıfır mod yok; $f_1$ = 4.537 / 4.433 / 4.386 GHz ($h$ = 12 / 8 / 5 mm) | sahte $\lambda=1.8\cdot10^{-12}$; sıfır olmayan $f_1$ = 4.371 / 4.351 / 4.352 → aynı limit |
| Kutu 100×80×60 mm içinde yüzen küre $r=15$ mm | $b_2=1$, 2 kabuk | bileşen potansiyeli **olmadan**: $\lambda=0$ sahte mod; **ile**: temiz, $f_1$ = 2.028 / 2.049 GHz ($h$ = 8 / 5 mm) | ($b_1=0$ → H geçerli) $f_1$ = 2.123 / 2.090 GHz, E ile aynı limit |
| Kutu içinde yüzen silindir $r=10$, $l=20$ mm | $b_2=1$ | olmadan: $\lambda=2\cdot10^{-12}$; ile: $f_1$ = 2.087 GHz | — |

### 6.4 Yeni aileler (kulplu, `cavity_shapes.HANDLE_FAMILIES`, varsayılan `FAMILIES` içinde)

| Aile | Geometri | $b_1$ | $f_1$ (ölçülen, mesh 0.10) |
|---|---|---|---|
| `hwr` | Yarım dalga koaksiyel rezonatör: $R_o\in[3,6]$ cm, $L\in[8,20]$ cm, $r_i/R_o\in[0.2,0.5]$; iç iletken iki uç plakaya değer. %50 konik iç iletken (orta yarıçap $r_m$), iç iletken–plaka birleşiminde ve dış köşelerde yuvarlatma. %50 enine ışın portu: dış duvardan ve iç iletkenden geçen $x$ yönlü delik ($r_b = [0.45,0.7]\min(r_i,r_m)$), dışarıda PEC kapaklı tüpler | 1 (port ile 2) | 0.77–1.58 GHz ≈ $c/2L$ |
| `spoke` | Silindirik tank ($R_t\in[5,10]$ cm, eksen $z$) ve tank çapını $x$ boyunca kesen 1–2 spoke (dairesel veya eliptik kesit, $a_z/R_t\in[0.15,0.28]$). Spoke'lar iki uçta tank duvarına değer; ikinci spoke paralel ya da $z$ etrafında 90° döndürülmüş olabilir. %60 olasılıkla $z$ yönlü ışın deliği (tanktan ve her spoke'tan geçer, spoke kesiti yarış pisti gibi genişletilir), aksi halde uç plakalarda tüp saplamaları. OCC boolean'ı sağlam: hacim denetimi (`_check_volume`) sessiz bozuk kesimleri yakalar ve geometri yeniden çekilir. Yuvarlatma yok | spoke başına 1 (delikli spoke başına 2) | 0.77–1.49 GHz |
| `dtl` | Alvarez benzeri tank ($R_t\in[5,10]$ cm, $n+1$ hücre, $L_c/R_t\in[0.5,0.9]$). Hücre sınırlarında 1–3 sürüklenme tüpü: delikli halka, $R_d/R_t\in[0.2,0.3]$, $r_b/R_d\in[0.35,0.5]$, yuvarlatılmış burunlar. Her tüp tank duvarına $+y$ yönlü bir sapla bağlanır. Tank uçlarında ışın tüpleri | tüp başına 1 | 0.40–0.77 GHz (sap modları: tüp başına bir düşük mod), ardından TM010 benzeri hızlandırıcı mod |

Kenar sayısı mesh 0.10'da: `hwr` 7–9k (portlu ~21k), `spoke` 11–26k, `dtl` 21–27k. Bu aileler küçük detaylar içerdiği için `_h_cap` taban kuralına ($h\ge0.7\,h_V$) takılır. `--field H` ile bu aileler varsayılandan çıkar (`H_FAMILIES`); açıkça istenirlerse örnek hata verir.

### 6.5 CLI, H5 ve PKL

```bash
python -m src.data_gen.dataset_generator_3d --n_total 1000 --mesh_size 0.10 --sampling sobol --deform_prob 0.5 \
    --h5_filename rf3d_e.h5            # --field E varsayılan; tüm FAMILIES (hwr, spoke, dtl dahil)
python -m src.data_gen.dataset_generator_3d --field H ...   # eski H formülasyonu (yalnız topolojik toplar)
python convert_3d.py --h5_filepath rf3d_e.h5 --output_path data/rf3d_e.pkl   # alan H5'ten okunur
```

- **H5:** E için veri kümesi `e_edges` [Ne,K]: tüm kenarlarda $\int\mathbf E\cdot d\mathbf l$, duvar satırları 0, fiziksel birimde $M$-ortonormal. H için `h_edges`. Ek attr'lar: `field`, `n_bnd_components`, `betti1`. Dosya `metadata`'sında `field` ve `dataset` bulunur.
- **PKL (`metadata['field'] = 'E'`):**
  - `G`: E potansiyel matrisi [Ne×Nv]. İlk `n_pot` sütun potansiyellerdir, kalanlar sıfırdır.
  - `Kp` $=G^\top MG$: `n_pot` bloğunda SPD, dışında sıfır.
  - Ek anahtarlar: `bnd_edge` bool [Ne], `n_pot`, `n_bnd_components`, `betti1`, `field`.
  - `Y`: E DOF'ları; duvar satırları tam 0, normalize mesh'te $\|Y\|_M=1$.
  - Dönüştürücü her örnekte duvar satırlarının sıfır olduğunu, $G^\top MY\approx0$ olduğunu ve Rayleigh frekansını denetler.
  - H PKL'leri değişmedi (`field='H'` veya anahtar yok).
- **Yalın PKL:** `geometry_operators(X, tets, field=metadata['field'])` saklanan tüm anahtarları yeniden kurar. Varsayılan `field='H'` eski çağrılarla bayt düzeyinde uyumludur; E PKL'lerde `field` mutlaka verilmelidir.

## 🔗 Bağlantılar

- [18_3D_EXTENSION.md](18_3D_EXTENSION.md): fizik, çözücü seçenekleri, Schur-projekteli Gram (§3.4)
- [01_DATA_GENERATION.md](01_DATA_GENERATION.md), [02_FEATURE_ENGINEERING.md](02_FEATURE_ENGINEERING.md): 2D karşılıkları

#3d #maxwell #nedelec #hcurl #veri-hattı #sözleşme


## Mesh çözünürlüğü ve gerçek simülasyonlar

Yakınsama (N0, düz kenarlı tet; hata $O(h^2)$, frekans yukarıdan yakınsar):

| mesh | tet | kenar DOF | pillbox $f_1$ hatası (analitik) | TESLA hücresi $f_1$ hatası (Richardson ref.) | süre (tek çekirdek) |
|---|---|---|---|---|---|
| 0.14 | 2k | 3k | +0.93 % | +2.4 % | <1 s |
| **0.10** (eğitim) | 5k | 7k | +0.48 % | +1.3 % | 0.5 s |
| 0.07 | 14k | 19k | +0.24 % | +0.63 % | 2–4 s |
| 0.05 | 37k | 48k | +0.12 % | +0.33 % | 13–20 s |
| 0.035 | 107k | 133k | +0.06 % | +0.16 % | 2–6 dk |

Üretim simülasyonları (CST/HFSS/ACE3P, SRF hücre tasarımı) tipik olarak 2. dereceden eğri kenarlı
elemanlarla $10^5$–$10^6$ tet kullanır; hedef frekans doğruluğu $10^{-4}$–$10^{-5}$. Buradaki veri
(mesh 0.10) bundan 2–3 mertebe kabadır: tasarım taraması, mod sınıflandırması, eğilimler ve alan
desenleri için yeterli; nihai frekans ayarı, yüzey tepe alanları ($E_{pk}$, $H_{pk}$) ve hassas HOM
empedansı için değil. Çözünürlük genellemesi: `scripts/resolution_study.py` (aynı geometriler, farklı
mesh; model ve düz FE hatası en ince mesh'e göre), notebook hücre 12.
