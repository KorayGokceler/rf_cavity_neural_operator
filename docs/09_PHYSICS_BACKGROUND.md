# 09 — Fizik Arka Planı

> Bu not, projenin dayandığı fiziği ve matematiği detaylı açıklar.

---

## 🌊 RF Kaviteler Nedir?

RF (Radyo Frekansı) kaviteleri, elektromanyetik dalgaları belirli rezonans frekanslarında hapseden metal yapılardır. Parçacık hızlandırıcılarda, parçacıklara enerji vermek için kullanılırlar.

**Gerçek dünya uygulaması:** CERN LHC, DESY PETRA, GSI/FAIR gibi hızlandırıcılarda binlerce RF kavite bulunur.

---

## ⚡ Maxwell Denklemlerinden Helmholtz'a

### Başlangıç: Maxwell Denklemleri
Boşlukta (vakumda), zamanla harmonik ($e^{j\omega t}$) alanlar için:

$$\nabla \times \vec{E} = -j\omega\mu_0 \vec{H}$$
$$\nabla \times \vec{H} = j\omega\epsilon_0 \vec{E}$$

### Helmholtz Denklemi
İkinci denklemden $\vec{H}$'yi eliminasyonla:

$$\nabla^2 \vec{E} + k^2 \vec{E} = 0$$

burada $k = \omega/c = 2\pi f / c$ dalga sayısıdır ve $c = 299\,792\,458$ m/s ışık hızıdır.

### 2D İndirgeme (TM Modları)
2D kesitte, elektrik alan sadece z-yönünde ($E_z$) bir skaler olur. TM (Transverse Magnetic) modlar için:

$$\nabla^2 E_z(x,y) + k^2 E_z(x,y) = 0$$

Bu, bir **eigenvalue problemidir:**
- **Eigenvalue:** $k^2$ → Rezonans frekansını verir
- **Eigenfunction:** $E_z(x,y)$ → Alan dağılımını (mod şeklini) verir

### Sınır Koşulu
Mükemmel elektrik iletken (PEC) duvarlarda:

$$E_z\big|_{\partial\Omega} = 0 \qquad \text{(Dirichlet BC)}$$

---

## 🔺 Finite Element Method (FEM)

### Zayıf (Weak) Formülasyon
Helmholtz denkleminin her iki tarafını test fonksiyonu $\phi_i$ ile çarpıp alan üzerinde integre ederiz:

$$\int_\Omega \nabla \phi_i \cdot \nabla E_z \, dA = k^2 \int_\Omega \phi_i \cdot E_z \, dA$$

Bu, matris formunda:

$$K \vec{u} = k^2 M \vec{u}$$

burada:
- $K_{ij} = \int_\Omega \nabla \phi_i \cdot \nabla \phi_j \, dA$ → **Stiffness matrisi** (Laplacian)
- $M_{ij} = \int_\Omega \phi_i \cdot \phi_j \, dA$ → **Mass matrisi**
- $\vec{u}$ → Düğüm değerleri (çözüm vektörü)

### P2 Elemanlar
Bu projede `ElementTriP2` kullanılıyor — her üçgenin 6 düğümü var:
- 3 köşe + 3 kenar orta noktası
- Quadratic (2. derece) interpolasyon
- Doğruluk: $O(h^3)$ (h = mesh boyutu)

P1'e göre (lineer, 3 düğüm) çok daha hassas — özellikle kavite modlarının tepe bölgelerinde.

### Shift-Invert Methodu
`sigma=500.0` parametresi, eigenvalue solver'a "500 civarında eigenvalue ara" der:

$$(K - \sigma M)^{-1} M \vec{u} = \frac{1}{k^2 - \sigma} \vec{u}$$

Bu, düşük frekanslı modları hızlı ve güvenilir şekilde bulmayı sağlar. Yoksa solver en yüksek frekansları bulma eğiliminde olur.

---

## 🎵 Rezonans Modları (Eigenmodes)

### Mode 0: Monopol (TM₀₁₀-benzeri)
- **Yapı:** Merkezde tek bir tepe, sınıra doğru düşüş
- **Simetri:** Radyal simetrik (dairesel kavitelerde)
- **Fiziksel anlam:** En düşük rezonans frekansı, parçacık hızlandırmada kullanılan temel mod

### Mode 1 & 2: Dipol (TM₁₁₀-benzeri)
- **Yapı:** Bir eksende pozitif, dik eksende negatif (düğüm çizgisi ortadan geçer)
- **Simetri:** Asimetrik, belirli bir eksende polarize
- **Degenerate çift:** Dikdörtgen/dairesel kavitelerde Mode 1 ve Mode 2 aynı frekansa sahip olabilir ama farklı yönlerde polarize olur
- **Fiziksel anlam:** İstenmeyen "parasitik" modlar — kavite tasarımında bastırılmaya çalışılır

### Mode Sıralaması
Modlar frekansa göre sıralanır: $f_0 < f_1 \leq f_2$

Mode 1 ve Mode 2 degenerate olabilir ($f_1 = f_2$) — bu, kavite simetrik olduğunda gerçekleşir. Asimetrik kavitelerde degeneracy bozulur.

---

## 🔢 Frekans Hesaplama Formülleri

### Dikdörtgen Kavite ($a \times b$)
$$f_{mn} = \frac{c}{2} \sqrt{\left(\frac{m}{a}\right)^2 + \left(\frac{n}{b}\right)^2}$$

### Dairesel Kavite (yarıçap $R$)
$$f_{mn} = \frac{c \cdot j_{mn}}{2\pi R}$$

Bessel sıfırları $j_{mn}$:
| | n=0 | n=1 | n=2 |
|---|---|---|---|
| m=0 | 2.4048 | 5.5201 | 8.6537 |
| m=1 | 3.8317 | 7.0156 | 10.1735 |

### Coaxial (Halka) Kavite
Analitik çözüm daha karmaşık — Bessel fonksiyonlarının kombinasyonu gerekir.

---

## 🧠 Neural Operator Teorisi

### Operator Learning
Klasik NN: $f: \mathbb{R}^n \to \mathbb{R}^m$ (vektör → vektör)

Neural Operator: $\mathcal{G}_\theta: \mathcal{A} \to \mathcal{U}$ (fonksiyon → fonksiyon)

Burada:
- $\mathcal{A}$: Giriş fonksiyon uzayı (kavite geometrisi)
- $\mathcal{U}$: Çıkış fonksiyon uzayı (alan dağılımı)

### GNOT'un Katkısı
GNOT, orijinal FNO (Fourier Neural Operator) ile Transformer mimarisini birleştirir:
- **Mesh-independent:** Farklı boyutlardaki meshlerde çalışabilir (padding + masking)
- **Attention-based:** Uzun mesafeli bağımlılıkları yakalar (FNO'nun global spectral convolution'ına alternatif)
- **Multi-input:** Hem koordinat hem de geometri özellikleri aynı anda işlenebilir

---

## 🏭 Hızlandırıcı Fiziğinde Uygulama

### Kavite Tasarım Döngüsü (Mevcut)
```
Tasarım → CST/COMSOL (FEM) → Sonuç Analizi → Tasarımı Güncelle → ...
                 ↑
          Saatler/Günler sürüyor
```

### Hedeflenen Döngü (Bu Proje İle)
```
Tasarım → GNOT (Neural Operator) → Anlık Sonuç → Tasarımı Güncelle → ...
                    ↑
             Milisaniyeler
```

Bu, **inverse design** (ters tasarım) için kritiktir: "İstediğim rezonans frekansını üreten kavite şekli ne olmalı?"

---

## 📚 Referans Kaynaklar

1. **GNOT Paper:** "General Neural Operator Transformer" — `examples/2302.14376v3.pdf`
2. **FNO:** Li et al., "Fourier Neural Operator for Parametric PDEs" (2020)
3. **RF Kavite Fiziği:** Wangler, "RF Linear Accelerators" (2008)
4. **FEM Temelleri:** Zienkiewicz & Taylor, "The Finite Element Method"
5. **scikit-fem:** https://scikit-fem.readthedocs.io/

---

## 🔗 Bağlantılar

- Veri üretimi (FEM çözücü): [[01_DATA_GENERATION]]
- Model mimarisi: [[04_MODEL_ARCHITECTURE]]
- Geliştirme fikirleri: [[10_IMPROVEMENT_IDEAS]]

#fizik #helmholtz #maxwell #rf-kavite #fem #eigenvalue
