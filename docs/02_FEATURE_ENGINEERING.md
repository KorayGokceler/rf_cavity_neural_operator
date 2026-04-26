# 02 — Feature Engineering (Öznitelik Mühendisliği)

> **Dosyalar:** `src/data/dataset_converter.py`, `convert.py`  
> **Girdi:** `.h5` dosyası (ham mesh + çözüm)  
> **Çıktı:** `.pkl` veya `.h5` (GNOT-ready format)

---

## 🎯 Ne Yapıyor?

Ham mesh verilerini (node koordinatları + üçgen bağlantıları) alıp, modelin geometriyi "anlayabilmesi" için zengin bir öznitelik vektörüne dönüştürüyor. Her node için **8 boyutlu** bir feature vektörü üretilir.

Bu dosya, projenin **en kritik** parçalarından biridir. Doğru feature seçimi, modelin başarısını doğrudan belirler.

---

## 📐 Feature Vektörü (val_dim = 8)

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

## 🔄 Peak-Sign Normalization

Eigenvalue problemlerinde mod şekilleri bir **global işaret belirsizliğine** sahiptir:

$$\text{Eğer } E(x) \text{ çözümse, } -E(x) \text{ de çözümdür.}$$

Bu, aynı geometriden farklı çalıştırmalarda ters işaretli alanlar alınabileceği anlamına gelir. Eğer model bazen `+` bazen `-` hedef görürse öğrenemez.

**Çözüm:**
```python
max_idx = argmax(|Y|)       # En yüksek genlikli node
if Y[max_idx] < 0:
    Y = Y * -1.0             # Tüm alanı ters çevir
Y = Y / max(|Y|)            # [-1, 1] normalizasyonu
```

Bu sayede her zaman "en büyük tepe yukarı bakar" — tutarlı bir hedef.

---

## 🧮 Theta (Koşul) Vektörü

Her sample için model şu bilgiyi alır:
```python
theta = [mode_index, frequency, sample_id]  # float32
```

- `mode_index` (0, 1, 2): Hangi mod tahmin ediliyor? → FiLM katmanlarına gider
- `frequency`: Rezonans frekansı (GHz) → Frekans branch'ine hedef olarak verilir
- `sample_id`: Geometri kimliği → Kullanılmıyor, debug amaçlı

---

## 📦 Çıktı Yapısı (.pkl)

```python
{
    'geometry_pool': {
        geom_id: {
            'X':           np.array([N, 2]),   # Normalize koordinatlar
            'Input_funcs': np.array([N, 8]),   # 8 feature
            'elements':    np.array([M, 3]),   # Üçgen bağlantıları
        }
    },
    'samples': [
        {
            'geom_id': int,
            'Y':       np.array([N, 1]),       # Peak-sign normalized alan
            'Theta':   np.array([3]),           # [mode, freq, id]
        },
        ...
    ],
    'metadata': {
        'mode_indices': [0, 1, 2],
        'freq_stats': {'mean': float, 'std': float, ...}
    }
}
```

**Önemli:** Geometri bilgisi (`geometry_pool`) ve fiziksel çözümler (`samples`) ayrıdır. Bir geometrinin 3 modu = 3 ayrı sample, ama aynı geometri ID'sini paylaşır. Bu, bellek tasarrufu sağlar.

---

## ⚠️ Bilinen Kısıtlamalar ve Geliştirme Önerileri

### Kısıtlamalar
1. **Sabit Feature Sayısı:** 8 feature, tüm mesh topolojilerinde aynı. Bazı bilgiler (komşuluk yapısı, lokal eğrilik detayı) kaybolabilir.
2. **Global PCA:** Tüm sınır noktalarına tek bir PCA uygulanıyor. Asimetrik kavitelerde bu eksen yanıltıcı olabilir.

### Geliştirme Önerileri
1. **Graph-Based Features:** Her node'un komşu sayısı ve bağlantı kalitesi (aspect ratio) feature olarak eklenebilir.
2. **Spectral Features:** Mesh Laplacian'ının eigendecomposition'ı → kavite şeklinin "frekans domain" temsili.
3. **Curvature Features:** Sınır noktalarında lokal eğrilik hesaplanıp tüm node'lara interpolasyonla yayılabilir.
4. **Multi-Scale Distance:** Sınıra mesafe yerine, birden fazla mesafe ölçeğinde temsil (ör. Gaussian RBF kernel ile ağırlıklandırma).
5. **Signed Distance Function (SDF):** `dist_to_boundary` şu an unsigned. İç/dış ayrımı için signed distance kullanılabilir — meshing farklı olduğunda yararlı.

---

## 🔗 Bağlantılar

- Önceki adım: [[01_DATA_GENERATION]]
- Sonraki adım: [[03_DATASET_LOADER]]
- Modelde nasıl kullanılıyor: [[04_MODEL_ARCHITECTURE]]
- Ablation çalışması: [[07_VALIDATION_TOOLS]]

#feature-engineering #normalizasyon #pca #boundary
