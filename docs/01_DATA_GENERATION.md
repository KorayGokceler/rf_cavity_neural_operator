# 01 — Veri Üretimi (Data Generation)

> **Dosya:** `src/data_gen/dataset_generator.py`  
> **Bağımlılıklar:** `gmsh`, `scikit-fem (skfem)`, `multiprocessing`  
> **Çıktı:** `.h5` dosyası (gzip sıkıştırmalı)

---

## 🎯 Ne Yapıyor?

Rastgele 2D RF kavite geometrileri oluşturuyor, bunların üçgen meshlerini üretiyor ve sonlu elemanlar yöntemi (FEM) ile Helmholtz denkleminin ilk 3 eigenmode'unu çözüyor.

Tüm geometri ve mesh parametreleri `configs/default.yaml`'dan okunur ve `run_pipeline.py` → CLI argümanları yoluyla `dataset_generator.py`'ye iletilir.

---

## 🔬 Fizik Arka Planı

### Helmholtz Denklemi
RF kavitelerde elektrik alan dağılımı, **Helmholtz eigenvalue problemi** ile belirlenir:

$$\nabla^2 E + k^2 E = 0$$

Burada:
- $E$ = Elektrik alan (eigenfunction / mod şekli)
- $k$ = Dalga sayısı (eigenvalue)
- Sınır koşulu: $E = 0$ (mükemmel iletken duvar → **Dirichlet BC**)

### Eigenvalue → Frekans Dönüşümü
Çözücü `k²` değerlerini (eigenvalue) döndürür, bunlar frekansa şöyle çevrilir:

$$f = \frac{c \cdot \sqrt{k^2}}{2\pi} \quad \text{[Hz]}$$

Kodda bu satır bunu yapar:
```python
freqs = (299792458 * np.sqrt(np.abs(vals.real))) / (2 * np.pi) / 1e9  # GHz
```

### FEM Çözücü Detayları
- **Element Tipi:** `ElementTriP2` — Her üçgenin 6 düğümü vardır (köşeler + kenar ortaları). Bu, lineer elemanlardan (~3x) çok daha hassas alan profilleri üretir.
- **Stiffness Matrisi (K):** $K_{ij} = \int \nabla \phi_i \cdot \nabla \phi_j \, dA$ → Laplacian operatörünün ayrık karşılığı
- **Mass Matrisi (M):** $M_{ij} = \int \phi_i \cdot \phi_j \, dA$ → Eigenvalue problemindeki ağırlık matrisi
- **Condensation:** Dirichlet BC'yi zorlamak için sınır satır/sütunlarını matrislerden çıkarır
- **Shift-Invert:** `sigma` parametresi ile düşük frekans modlarına odaklanır (varsayılan: 500.0). Shift-invert *sigma'ya en yakın* k eigenvalue'yu döndürür; eğer `max|λ_i − σ| < σ` ise (σ spektrumun içinde → alt modlar atlanmış olabilir) çözüm otomatik olarak `σ = 0` ile tekrarlanır, bu her zaman en düşük k modu verir.
- **Eigen solver:** `solve_dirichlet_eigenmodes()` simetrik `eigsh` (`utils.solver_eigen_scipy_sym`) kullanır. skfem'in varsayılanı simetrik olmayan `eigs`'tir: kompleks çıktı verir ve sıralama garantisi yoktur (dejenere çiftlerde `f1 > f2` görüldü). Şimdi modlar **artan frekansa göre sıralı** ve **M-ortonormal** (`vecs.T @ M @ vecs = I`, dejenere çiftler dahil).
- **Sınır koşulu:** Sadece Dirichlet (`E_z = 0`, PEC) → **TM modları**. TE modları ($H_z$, Neumann $\partial_n H_z = 0$) bu pipeline'da **yok**.

> **Not:** P2 elemanlar çözümü `n_nodes * ~2.5` boyutunda üretir (H5'e tamamı yazılır). skfem `ElementTriP2`'de ilk `n_nodes` DOF vertex node'lardır (kodda assert ediliyor); converter sadece bu kısmı alır → `Y` uzunluğu `X` ile aynı. Edge node'lar atılır.

### Analitik Doğrulama (calibration modu)
```bash
python src/data_gen/dataset_generator.py --mode calibration --n_total 6 --n_plot 0 --h5_filename calib.h5
python validate_data.py --h5_filename calib.h5
```
Kare / daire / halka için ilk 8 mod analitik değerlerle karşılaştırıldı: ortalama bağıl hata **%0.006**, maksimum **%0.015** (varsayılan mesh). Halka referansı $J_n(ka)Y_n(kb) - J_n(kb)Y_n(ka) = 0$ köklerinden hesaplanır.

---

## 🔷 Geometri Üretimi

### Sharp (Köşeli) Geometriler
```python
n_pts = np.random.randint(ARGS.sharp_n_pts_range[0], ARGS.sharp_n_pts_range[1])  # ör. 7-12 köşeli
delta = 2π / n_pts                       # Eşit açı aralığı
angles[i] = i * delta + U(-δ/3, δ/3)    # Açıya rastgele pertürbasyon
r = np.random.uniform(ARGS.sharp_r_range[0], ARGS.sharp_r_range[1], n_pts)
```
Sonuç: Düzensiz yıldız/poligon geometrileri.

### Smooth (Yumuşak) Geometriler
```python
t = linspace(0, 2π, 100)
h_min, h_max = ARGS.smooth_harmonics   # ör. [2, 8] → range(2, 8) = k ∈ {2..7}
r(t) = ARGS.smooth_base_r + Σ_{k=h_min}^{h_max - 1} a_k·cos(k·t + φ_k)
```
Burada `a_k ~ U(-ARGS.smooth_perturb, ARGS.smooth_perturb)` ve `φ_k ~ U(0, 2π)`.

Bu, Fourier serisinin **rastgele katsayılarla** parametrelenmiş versiyonudur. Pürüzsüz ama çeşitli kavite şekilleri üretir.

### Calibration Geometrileri
- **Kare:** `side = 0.08` → Analitik frekanslar: $f_{mn} = \frac{c}{2}\sqrt{(m/a)^2 + (n/a)^2}$
- **Daire:** `radius = 0.04` → Bessel sıfırları: $f_{mn} = \frac{c \cdot j_{mn}}{2\pi R}$
- **Halka (Annulus):** İç yarıçap `0.02`, dış yarıçap `0.045` → $J_n(ka)Y_n(kb) - J_n(kb)Y_n(ka) = 0$
- Geometri parametreleri her grubun attr'larına yazılır (`side`, `radius`, `r_inner`, `r_outer`); `validate_data.py` bunları okur.

---

## 🕸️ Adaptif Mesh Stratejisi

```python
gmsh.model.mesh.field.setNumber(2, "SizeMin", ARGS.mesh_size_min)  # Sınırda minimum
gmsh.model.mesh.field.setNumber(2, "SizeMax", ARGS.mesh_size_max)  # Merkezde maksimum
gmsh.model.mesh.field.setNumber(2, "DistMin", ARGS.mesh_dist_min)  # Geçiş başlangıcı
gmsh.model.mesh.field.setNumber(2, "DistMax", ARGS.mesh_dist_max)  # Geçiş bitişi
```

**Mantık:** Sınırda alan gradyanı çok keskin olduğu için mesh ~4x daha yoğun. Merkezde alan düzgün olduğu için seyrek mesh yeterli.

```
Sınır ←──── DistMin ──── DistMax ────→ Merkez
  │            │            │            │
  SizeMin      geçiş        SizeMax      SizeMax
  (0.0012)     bölgesi      (0.005)      (0.005)
```

Tüm bu değerler `configs/default.yaml` → `run_pipeline.py` aracılığıyla CLI'a iletilir, hiçbiri hardcoded değildir.

---

## 📦 Çıktı Formatı (.h5)

Her `sample_XXXX` grubu:
```
sample_0001/
├── nodes      [N, 2]    float64   → x, y koordinatları
├── elements   [M, 3]    int32     → üçgen bağlantıları (0-indexed)
├── freqs      [3]       float64   → 3 rezonans frekansı (GHz)
├── vecs       [N', 3]   float64   → 3 mod şekli vektörü (P2 DOF'ları; ilk N satır = vertex'ler), M-ortonormal
└── attrs:
    ├── shape_type → 'random_sharp' | 'random_smooth' | 'square' | 'circle' | 'annulus'
    ├── n_nodes, fem_element='P2'
    └── (calibration) side | radius | r_inner, r_outer
```
Dosya attr'ı `metadata` (JSON): generator argümanları, `bc`, birimler (`m`, `GHz`). `freqs` artan sıradadır.

### Robustluk / Tekrarlanabilirlik
- **Atomik yazım:** Çıktı önce `<h5_filename>.partial`'a yazılır, her chunk'ta flush edilir, sonunda `os.replace` ile adlandırılır → yarıda kalan çalışma son dosyayı bozmaz. Hiç geçerli örnek yoksa hata ile çıkar.
- **Seed:** Örnek `s_id` için `np.random.seed(s_id*13 + seed*1000003)`; `--seed 0` (varsayılan) eski datasetlerin **sharp** geometrilerini birebir üretir (mesh aynı, frekans farkı ~1e-15). **Delikler:** `--hole_prob p` (config `data_gen.hole_prob`, varsayılan config'lerde 0.3) ile geometrilerin p oranına iç teğet çember içinde, duvar kalınlığı ≥ 0.15·r_iç olan 1..`--max_holes` eliptik delik açılır (OCC boolean cut) → `shape_type` sonuna `_hole{n}` eklenir (ör. `random_sharp_hole1`). `hole_prob 0` (CLI varsayılanı) hiç rastgele sayı çekmez; eski datasetler değişmez. **Smooth** geometriler artık 100 noktalı çokgen yerine periyodik C2 spline ile çiziliyor ve `clip` yerine pertürbasyon ölçekleniyor (çokgen köşeleri >200° iç açılı köşe tekillikleri üretiyordu, bkz. docs/16–17); bu yüzden eski smooth geometriler birebir üretilmez.
- **Worker'lar:** `--n_workers` (varsayılan `cpu_count()`); `ARGS` pool initializer ile aktarılır → `spawn` start method'u (macOS/Windows varsayılanı) da çalışır (önceden her örnek `ARGS=None` ile sessizce başarısız oluyordu).
- **Takılma koruması:** `--sample_timeout` (s, varsayılan 300) — takılan örnek atlanır, pool yenilenir.
- **Mesh çıkarımı:** gmsh node tag'leri açıkça satır indekslerine map edilir; sadece 3-node üçgen kabul edilir; kullanılmayan node'lar atılır.

### Smoke Test
```bash
python src/data_gen/dataset_generator.py --mode random --n_total 10 --n_plot 0 --n_workers 4 --h5_filename smoke.h5 --plot_dir smoke_plots
python convert.py --h5_filepath smoke.h5 --output_path smoke.pkl --modes 0 1 2
```
(~3 s toplam, CPU.)

> `elements` raw H5 dosyasında saklanır. Feature conversion (`convert.py`) aşamasından sonra eğitim pipeline'ına iletilmez — RAM tasarrufu için dataset init'te atılır.

---

## ⚠️ Bilinen Kısıtlamalar ve Geliştirme Fikirleri

### Kısıtlamalar
1. **2D Sınırlaması:** Gerçek RF kaviteleri 3D'dir. Bu model sadece TM modlarını (2D kesit) modelliyor.
2. **Sabit Malzeme:** Tüm kaviteler PEC (mükemmel elektrik iletken) kabul ediliyor. Kayıplı malzeme (Q-factor) modellenmemiş.
3. **P2 → P1 Truncation:** Eigenvector'lerin sadece vertex node kısmı alınıyor. Kenar ortası bilgisi atılıyor.

### Geliştirme Önerileri
1. **Solution-Based Adaptive Refinement:** Mevcut mesh sadece geometriye bağlı. Bir ilk çözüm yapılıp, hatanın yüksek olduğu bölgelerde mesh sıklaştırılabilir.
2. **Parametre Aralığını Genişletme:** Daha büyük ve daha küçük kaviteler eklenebilir.
3. **3D Kavitelere Geçiş:** `gmsh` 3D mesh destekliyor. `tetrahedra` elemanlarla 3D Helmholtz çözülebilir.
4. **Frekans Aralığı Kontrolü:** ~~`eigen_sigma` adaptif~~ → σ spektrumun içindeyse artık otomatik `σ=0` fallback var.

---

## 🔗 Bağlantılar
- Sonraki adım: [[02_FEATURE_ENGINEERING]]
- Fizik detayları: [[09_PHYSICS_BACKGROUND]]
- Validasyon: [[07_VALIDATION_TOOLS]]

#veri-uretimi #fem #helmholtz #mesh #gmsh
