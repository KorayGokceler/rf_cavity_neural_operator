# 02 — Feature Engineering (Öznitelik Mühendisliği)

> **Dosyalar:** `src/data/dataset_converter.py`, `convert.py`  
> **Girdi:** `.h5` dosyası (ham mesh + çözüm)  
> **Çıktı:** `.pkl` veya `.h5` (GNOT-ready format)

---

## 🎯 Ne Yapıyor?

Ham mesh verilerini (node koordinatları + üçgen bağlantıları) alıp, modelin geometriyi "anlayabilmesi" için zengin bir öznitelik vektörüne dönüştürüyor. Her node için **12 boyutlu** bir feature vektörü üretilir (commit 41a830d'den beri; eski 8-feature PKL'ler için `dataset.feature_indices: [0..7]` veya yeniden dönüştürme).

Bu dosya, projenin **en kritik** parçalarından biridir. Doğru feature seçimi, modelin başarısını doğrudan belirler.

---

## 📐 Feature Vektörü (val_dim = 12)

Her node $i$ için şu öznitelikler hesaplanır:

| İndeks | Feature | Açıklama | Fiziksel Motivasyon |
|--------|---------|----------|---------------------|
| 0 | `x_norm` | Normalize x koordinatı | Uzamsal konum |
| 1 | `y_norm` | Normalize y koordinatı | Uzamsal konum |
| 2 | `dist_to_boundary` | En yakın sınıra mesafe | Dirichlet BC: $E \to 0$ sınırda |
| 3 | `dir_bnd_x` | Sınıra yön vektörü (x) | Gradyan yönü bilgisi |
| 4 | `dir_bnd_y` | Sınıra yön vektörü (y) | Gradyan yönü bilgisi |
| 5 | `node_area` | Yerel mesh yoğunluğu | Çözücünün hassasiyetini yansıtır |
| 6 | `cos_principal` | Ana eksene göre açının cos'u | Kavite yönelimi/asimetrisi |
| 7 | `sin_principal` | Ana eksene göre açının sin'i | Kavite yönelimi/asimetrisi |
| 8 | `dist_2nd_boundary` | 2. en yakın sınır node'una mesafe | Çok-ölçekli sınır bilgisi (≈ dist + mesh adımı) |
| 9 | `dist_3rd_boundary` | 3. en yakın sınır node'una mesafe | 〃 |
| 10 | `curvature` | En yakın sınır noktasındaki işaretli eğrilik (max-abs ile normalize) | **+ konveks duvar, − konkav/girintili duvar** |
| 11 | `convexity` | `dist_to_boundary * curvature` (clip ±1) | < 0: konkav "cep" bölgesi |
| 12 | `torsion` | $w/\max w$, $-\Delta w = 1$ (Ω), $w = 0$ (∂Ω); tek P1 çözüm | Düzgün "landscape"; $\lambda_1 \approx j_{01}^2/(4\max w)$ (~%1, docs/16). `torsion_max` geometri başına saklanır |

---

## 🔍 Her Feature'ın Detaylı Matematiksel Tanımı

### 1. Isotropic Normalization (x_norm, y_norm)
```python
center = nodes.mean(axis=0)
nodes_centered = nodes - center
global_scale = max(|nodes_centered|)
nodes_norm = nodes_centered / global_scale
```

**Neden "isotropic"?** Her eksen aynı `global_scale` ile ölçeklenir. Bu, kavitelerin en-boy oranını (aspect ratio) korur. Eğer her ekseni ayrı normalize etsek, kare bir kavite ile dikdörtgen bir kavite aynı görünürdü — model geometri bilgisini kaybederdi.

### 2. Distance to Boundary
```python
boundary_indices = _find_boundary_nodes(elements)  # Sadece 1 kez paylaşılan kenarlar
tree = cKDTree(boundary_nodes)
dist, nearest_idx = tree.query(nodes_norm)
```

**Fizik:** Dirichlet sınır koşulunda $E(x_{\text{boundary}}) = 0$. Bu feature, modele "duvara ne kadar yakınsın" bilgisini verir. Alan genliği genel olarak sınırdan uzaklaştıkça artar.

**Boundary Bulma Algoritması:** Bir kenar (edge) sadece **tek bir üçgene ait** ise sınır kenarıdır. İç kenarlar her zaman 2 üçgen tarafından paylaşılır.

### 3. Direction to Boundary (dir_bnd_x, dir_bnd_y)
```python
dir_vec = nearest_boundary_point - current_node
dir_to_boundary = dir_vec / |dir_vec|  # Birim vektör
```

**Fizik:** Alan, sınıra dik yönde en hızlı azalır (Neumann-tipi davranış). Bu 2D vektör, modele düdğümden sınıra doğru "işaret eder."

### 4. Node Area (Yerel Mesh Yoğunluğu)
```python
for her üçgen [v0, v1, v2]:
    alan = 0.5 * |cross(v1 - v0, v2 - v0)|
    areas[v0] += alan / 3
    areas[v1] += alan / 3
    areas[v2] += alan / 3
areas = areas / max(areas)  # [0, 1] normalize
```

**Fizik:** FEM çözücüsü sınıra yakın yerlerde daha sık mesh kullandığı için, `node_area` küçük = "burada çözücü daha hassas" anlamına gelir. Model bu bilgiyi fiziksel öncelik olarak kullanabilir.

### 4b. Boundary Curvature (curvature, convexity)
Sınır, mesh topolojisinden **kapalı döngüler** olarak çıkarılır (tek üçgene ait yönlü kenarlar; üçgenler önce CCW'ye çevrilir). Her döngü **alan solda kalacak** şekilde yönlüdür: dış duvar CCW, delikler (ör. halka iç duvarı) CW. Eğrilik:
$$\kappa_i = \frac{\hat t_{in} \times \hat t_{out}}{\tfrac12(|e_{in}| + |e_{out}|)}$$
Bu sayede işaret **kanoniktir**: node numaralamasına, üçgen yönüne ve aynalamaya bağlı değil; daire her yerde +, halka iç duvarı −, L-şeklinin girintili köşesi −.

> **Düzeltme:** Önceki greedy nearest-neighbour sıralaması yönü rastgele seçiyordu → eğrilik işareti geometriden geometriye (ve aynalamada) rastgele dönüyordu; halkada iki döngü birbirine karışıyordu. 12-feature PKL'ler yeniden dönüştürülmeli.

### 5. Principal Axis Angle (cos/sin)
```python
# PCA on boundary nodes
cov = np.cov(boundary_centered.T)  # 2x2 kovaryans matrisi
eigenvalues, eigenvectors = np.linalg.eigh(cov)
principal_axis = eigenvectors[:, -1]  # En büyük eigenvalue'nun eigenvektörü

# Her node için açı
cos_angle = dot(node_vec, principal_axis) / |node_vec|
sin_angle = cross(node_vec, principal_axis) / |node_vec|
```

**Fizik:** Çok önemli bir feature! Kavitelerde **dipol modları** (Mode 1, Mode 2) belirli bir eksende polarize olur. Bu eksen, kavite geometrisinin ana ekseniyle ilişkilidir. PCA ile bulunan bu eksen, modele "bu kavite hangi yöne uzanıyor" bilgisini verir. `cos` ve `sin` birlikte tam açı bilgisi sağlar ($\theta$ yerine $\cos\theta, \sin\theta$ kullanmak sürekliliği korur).

---

## 🔄 Mod Normalizasyonu (işaret sabitlenmez)

Eigenvalue problemlerinde mod şekilleri bir **global işaret belirsizliğine** (ve dejenere çiftlerde bir 2D döndürme belirsizliğine) sahiptir:

$$\text{Eğer } E(x) \text{ çözümse, } -E(x) \text{ de çözümdür.}$$

Converter **sadece genliği** normalize eder: `Y = Y / max(|Y|)` → [-1, 1]. İşaret/altuzay belirsizliği bilinçli olarak sabitlenmez (commit 88d2219), gauge-invariant (sign-agnostic / Grassmannian) loss tarafından ele alınır. (Eski "peak-sign" normalizasyonu kaldırıldı.)

H5'teki ham `vecs` M-ortonormaldir; max-abs normalizasyonu bu ölçeği değiştirir (ortogonallik korunur, normlar değil).

---

## 🧮 Theta (Koşul) Vektörü

Her sample için model şu bilgiyi alır:
```python
theta = [mode_index, frequency, sample_id]  # float32
```

- `mode_index`: **`--modes` listesi içindeki slot** (0..len-1), ham FEM mod indeksi değil (ör. `--modes 1 2` → 0, 1). Ham indeks ayrıca `sample['mode_idx']`'te (H5 çıktısında `attrs['mode_idx']`). Converter modları önce frekansa göre sıralar.
- `frequency`: Rezonans frekansı (GHz) → Frekans branch'ine hedef olarak verilir
- `sample_id`: Geometri kimliği → Kullanılmıyor, debug amaçlı

---

## 📦 Çıktı Yapısı (.pkl)

```python
{
    'geometry_pool': {
        geom_id: {
            'X':           np.array([N, 2]),   # Normalize koordinatlar
            'Input_funcs': np.array([N, 12]),  # 12 feature
            'elements':    np.array([M, 3]),   # Üçgen bağlantıları
            'scale':       float,              # 1 normalize birim = scale [m]
            'center':      np.array([2]),      # çıkarılan merkez [m]
            'shape_type':  str,
        }
    },
    'samples': [
        {
            'geom_id': int,
            'Y':       np.array([N, 1]),       # Peak-sign normalized alan
            'Theta':   np.array([3]),           # [slot, freq_GHz, id]
            'mode_idx': int,                    # ham FEM mod indeksi
        },
        ...
    ],
    'metadata': {
        'mode_indices': [0, 1, 2],
        'freq_stats': {'mean': float, 'std': float, ...}
    }
}
```

**Fiziksel ölçek:** Tüm feature'lar ölçekten bağımsızdır (koordinatlar `scale`'e bölünür, `node_area` max'a normalize). Frekans ise boyutla ters orantılıdır: normalize domain'de hesaplanan Laplacian eigenvalue'su $\lambda_{norm}$ için $f = c\sqrt{\lambda_{norm}}/(2\pi\,\text{scale})$. Mutlak boyut bilgisi sadece `scale` anahtarındadır (modele şu an verilmiyor).

**Önemli:** Geometri bilgisi (`geometry_pool`) ve fiziksel çözümler (`samples`) ayrıdır. Bir geometrinin 3 modu = 3 ayrı sample, ama aynı geometri ID'sini paylaşır. Bu, bellek tasarrufu sağlar.

---

## ⚠️ Bilinen Kısıtlamalar ve Geliştirme Önerileri

### Kısıtlamalar
1. **Sabit Feature Sayısı:** 12 feature, tüm mesh topolojilerinde aynı. Bazı bilgiler (komşuluk yapısı, lokal eğrilik detayı) kaybolabilir.
2. **Global PCA:** Tüm sınır noktalarına tek bir PCA uygulanıyor. Asimetrik kavitelerde bu eksen yanıltıcı olabilir.

### Geliştirme Önerileri
1. **Graph-Based Features:** Her node'un komşu sayısı ve bağlantı kalitesi (aspect ratio) feature olarak eklenebilir.
2. **Spectral Features:** Mesh Laplacian'ının eigendecomposition'ı → kavite şeklinin "frekans domain" temsili.
3. ~~**Curvature Features**~~ → eklendi (feature 10-11).
4. **Multi-Scale Distance:** Sınıra mesafe yerine, birden fazla mesafe ölçeğinde temsil (ör. Gaussian RBF kernel ile ağırlıklandırma).
5. **Signed Distance Function (SDF):** `dist_to_boundary` şu an unsigned. İç/dış ayrımı için signed distance kullanılabilir — meshing farklı olduğunda yararlı.

---

## 🔗 Bağlantılar

- Önceki adım: [[01_DATA_GENERATION]]
- Sonraki adım: [[03_DATASET_LOADER]]
- Modelde nasıl kullanılıyor: [[04_MODEL_ARCHITECTURE]]
- Ablation çalışması: [[07_VALIDATION_TOOLS]]

#feature-engineering #normalizasyon #pca #boundary
