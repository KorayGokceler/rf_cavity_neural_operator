# 🧬 GNOT: Graph Neural Operator for RF Cavities
## Archi-Lecture: Derinlemesine Mimari ve Teknik Dokümantasyon

Bu doküman, `src/models/gnot.py` içerisinde yer alan GNOT modelinin her bir bileşenini, matematiksel mantığını ve konfigürasyon seçeneklerini detaylandırır.

---

## 1. Temel Bileşenler (Building Blocks)

### 1.1 LinearAttention (Doğrusal Dikkat)
Geleneksel Transformer modellerindeki Multi-Head Attention, bellek kullanımı açısından $O(N^2)$ (nokta sayısının karesi) maliyetine sahiptir. GNOT, bunu $O(N)$ seviyesine indirmek için **Linear Attention** kullanır.

- **Stabilite:** `F.elu(q) + 1.0` (ELU+1 kernel) kullanılarak negatif değerler elenir ve dikkat skorları stabilize edilir.
- **Verimlilik:** Önce Key ve Value çarpılır (`torch.einsum('bhnd,bhne->bhde', k, v)`), sonra Query ile işleme sokulur. Bu, bellek limitlerini aşmadan on binlerce mesh noktası üzerinde etkileşim kurmayı sağlar.

### 1.2 AttentionPool (Özetleme Mekanizması)
Mesh üzerindeki binlerce noktadan tek bir global değere (Frekans gibi) geçmek için kullanılır.

- **Mantık:** Öğrenilebilir bir "Query Tohumu" (`nn.Parameter`), tüm geometrideki özellikleri tarar ve en önemli özellikleri "yoğunlaştırarak" tek bir vektöre indirger.
- **Kullanım:** Frekans tahmini (`freq_decoder`) öncesinde son adım olarak çalışır.

### 1.3 GeometricGatingFFN (Mixture of Experts - MoE)
GNOT'un en ayırt edici özelliğidir. Sabit bir Feed-Forward katmanı yerine, mesh noktasının **geometrideki yerine göre** farklı uzmanları çalıştırır.

- **Experts (Uzmanlar):** Genellikle 4 adet tam bağlantılı (MLP) ağdır.
- **Gating (Kapılama):** `gating_net`, noktanın koordinatlarına (`coords`) bakarak o nokta için hangi uzmanın (veya uzmanların ağırlıklı kombinasyonunun) en iyi sonucu vereceğine karar verir.
- **Avantaj:** Köşelerdeki metal yüzeylerdeki fiziği öğrenen bir uzman ile boşluğun ortasındaki vakum alanını öğrenen uzman birbirinden farklılaşabilir.

---

## 2. Sistem Mimarisi (GNOTModel)

Model, görev paylaşımı ve verimlilik için **Shared-to-Specific (Ortak-to-Özel)** bir dallanma yapısı kullanır.

### 2.1 Encoders (Giriş Kodlayıcılar)
- **Query Encoder:** Mesh koordinatlarını (`X, Y`) yüksek boyutlu (`embed_dim`) vektörlere çevirir.
- **Input Func Encoder:** Geometrik özellikleri (SDF, Normaller vb.) kodlar.
- **Theta Encoder:** Seçilen modun indeksini (1., 2. mod vb.) kodlar.

### 2.2 Shared Blocks (Ortak İşlem Birimi)
Katmanların bir kısmı (`n_layers // 3`) tüm görevler (Field ve Frequency) için ortak çalışır. Burada geometrinin genel fiziksel yapısı öğrenilir.

### 2.3 Branching (Dallanma)
Daha sonra model iki kola ayrılır:
1.  **Field Branch:** Her nokta için yerel alan şiddetini tahmin etmek üzere ek GNOT blokları çalıştırır.
2.  **Freq Branch:** Tüm mesh'i özetleyip tek bir frekans değeri üretmek üzere ek GNOT blokları ve `AttentionPool` çalıştırır.

---

## 3. Ölçeklendirme Reçeteleri (Architectural Recipes)

Modeli ihtiyacınıza göre şu konfigürasyonlarla başlatabilirsiniz:

| Parametre | **Min (Tiny)** | **Standard (Mevcut)** | **Large (HRP)** | **Ultra (Deep)** |
| :--- | :---: | :---: | :---: | :---: |
| `embed_dim` | 64 | 128 | 256 | 512 |
| `n_layers` | 3 | 6 | 9 | 12 |
| `num_experts`| 2 | 4 | 8 | 16 |
| `n_heads` | 2 | 4 | 8 | 16 |
| **GPU RAM** | ~2GB | ~6GB | ~12GB | ~24GB+ |

### Nasıl Modifiye Edilir?
`train.py` çalıştırırken şu argümanları değiştirerek bu ölçeklere geçebilirsiniz:
`--hidden_dim 256 --n_layers 9`

---

## 4. Performans ve Bellek İpuçları

- **Gradient Checkpointing:** Bellek yetmiyorsa `--use_checkpoint` bayrağıyla aktif edilir. Bu, ileri hesaplamadaki 'activation'ları saklamaz, geri pas sırasında tekrar hesaplar.
- **Embed Dim Darboğazı:** Modeli küçültmek istiyorsanız katman sayısından önce `embed_dim` değerini (örn. 128'den 96'ya) düşürün. Bu, doğruluğu daha az etkilerken belleği daha çok rahatlatır.
- **num_experts:** Çok karmaşık geometrilerde uzman sayısını artırmak (örn. 8), modelin ince detayları öğrenme kapasitesini artıracaktır.

---
*Doküman Sonudur*
