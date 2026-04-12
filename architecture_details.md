# GNOT Model Mimari Detayları (Ayrıntılı Teknik Döküm)

Bu döküman, mevcut `src/models/gnot.py` içindeki modelin tüm bileşenlerini ve veri akışını (128-dim, 2-layer konfigürasyonuyla) açıklar.

## 1. Giriş Kodlama (Input Encoding)
Veri modele girdiğinde iki farklı koldan MLP (Multi-Layer Perceptron) süzgecine girer:

- **Query Encoder**: Sadece sorgu koordinatlarını (`X` ve opsiyonel `RFF`) alır. Şekli: `[B, N, 2+64] -> [B, N, 128]`. Bu, "Tahmin yapmak istediğimiz her noktanın uzaydaki kimliğini" temsil eder.
- **Input Function Encoder**: Mesh üzerindeki kaynak veriyi (`Input_funcs` [B, N, 6]), koordinatlarla (`X`) ve Fourier özellikleriyle (`RFF`) birleştirir. Şekli: `[B, N, 6+2+64] -> [B, N, 128]`. Bu, "Giriş fonksiyonunun geometrik bağlamıyla birlikte temel temsilidir".

## 2. Küresel Bilgi Havuzu (Global Context - AttentionPool)
Model, kavitenin "Büyük Resmini" tek bir vektöre sığdırır:
- **AttentionPool**: Öğrenilmiş bir `Parameter` (query) kullanarak tüm mesh verisi (`input_func_encoder` çıktısı) üzerinden bir dikkat (attention) toplar.
- Sonuç: `[B, 128]` boyutunda tek bir vektördür. Bu vektör, "Bu nasıl bir kavite? (Asimetrik mi, ne kadar uzun?)" sorusunun özetidir.

## 3. GNOTBlock (Transformer & MoE Mekanizması)
Bu, modelin kalbidir. Her blok şu 4 adımdan oluşur:

### A. Cross-Attention
- Sorgu noktaları, mesh üzerindeki kaynak verilere bakar. "Benim bulunduğum yerdeki alan, mesh'teki hangi değerlerden etkileniyor?" sorusuna yanıt arar. `LinearAttention` (verimli bellek kullanımı) kullanılır.

### B. Global Injection (Kritik Nokta)
- `AttentionPool`'dan gelen `[B, 128]` boyutundaki küresel özet, her bir noktaya **Toplama (Addition)** yoluyla eklenir. 
- *Analiz:* Bu işlem, tüm noktaları aynı yöne ("bias") çeker.

### C. Self-Attention
- Sorgu noktaları kendi aralarında konuşur. Bu, alanın sürekliliğini ve mesh üzerindeki fiziksel bağımlılıkları korumak içindir.

### D. Geometric Gating FFN (Spatial MoE)
- **Gating Net**: Bir kerede bir noktanın sadece koordinatlarına (`x, y`) bakarak o noktanın hangi "uzmana" (expert) gitmesi gerektiğine karar verir.
- **Experts (4 Adet MLP)**: Model içinde 4 adet bağımsız küçük ağ vardır. Örneğin, Uzman 1 "lob merkezleri" konusunda, Uzman 2 ise "sıfır geçişleri" konusunda uzmanlaşabilir.
- Her nokta, kendi koordinatına göre en yüksek skor alan **2 uzmanı** aktif eder ve onların sonuçlarını ağırlıklı olarak birleştirir.

## 4. Fiziksel Dallanma (Mode-Specific Branches)
Model, paylaşımlı bloklardan (`Shared Blocks`) sonra batchetteki örnekleri modlarına göre (TM01, Dipol vb.) ayırır:
- **Mode-Specific Physics Branch**: Her modun (şu an izole testte sadece Mode 1) kendine ait `GNOTBlock` katmanları vardır.
- **FiLM Layer**: Mod numarasına (Mode ID) göre özellikleri `gamma` ve `beta` ile çarpar/toplar. (Daha yüksek fiziksel kontrol sağlar).

## 5. Çıkış (Output Heads)
- **Field Head**: Her mod için ayrı bir lineer kafa [B, N, 1] çıktısı üretir. 
- **Freq Decoder**: Modelin ortasındaki global bilgiyi alıp tek bir skaler [B, 1] frekans değeri üretir.

---
**Özet:** Model, geometrik şekli AttentionPool ile bir kere anlayıp tüm noktalara aşılamaya çalışıyor. Takıldığımız %30-40 hata payı, bu aşılama yönteminin (Toplama) dipolün asimetrisini (bir taraf artı bir taraf eksi) "ters çevirmeye" yetmemesinden kaynaklanıyor olabilir.
