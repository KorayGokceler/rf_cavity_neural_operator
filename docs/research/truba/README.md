# TRUBA Kullanıcı Dokümanları — yerel kopya

`pages/` altında `https://docs.truba.gov.tr` sitesinin bu projeyi ilgilendiren 32 sayfasının Markdown çevirisi
var (2026-10-30'da alındı; her dosyanın ilk satırı kaynak URL'sini ve sayfanın "son değişiklik" tarihini verir).
Özet ve bizim ayarlarla karşılaştırma: [docs/36](../../36_TRUBA_REFERENCE.md). Burası ham kaynaktır;
bir sayıya güvenmeden önce ilgili sayfaya bakın (`grep -rn "orfoz" pages/`).

- **Kapsam:** hesap ve kullanım kuralları (`0-truba__*`), ARF / ARF-ACC / ARF-AI kuyruk ve depolama sayfaları,
  küme tablosu, GPU, SLURM betik ve komutlar, interaktif iş, konteyner (Apptainer), Python kılavuzu, SSS,
  duyurular, ücretli kullanım girişi. Alınmayanlar: 119 eğitim / seminer sayfası (CFD, MATLAB, R, PyTorch
  öğreticileri…), EuroHPC ve MareNostrum 5 sayfaları, yazılım kurulum notları, bağlantı sayfaları (içlerinde iç ağ
  IP adresleri var; repoya konmaz).
- **Eski içerik uyarısı:** eğitim bölümündeki PyTorch kurulum sayfası (2025-07) kullanıcı ev dizinine Anaconda
  kurmayı anlatıyor; yeni Python kılavuzu (2025-12) ve depolama sayfası (2026-04) bunu yasaklıyor. Çelişkide
  **tarihi yeni olan** sayfa geçerlidir.
- **Sayfalar nasıl alındı:** site yalnızca yaprak sertifikayı gönderiyor, ara sertifika (`TUBITAK Kamu SM SSL
  Sertifika Hizmet Saglayicisi - Surum 2`) eksik, bu yüzden sandbox'ta TLS doğrulaması başarısız oluyor.
  Sahibin açık onayıyla yalnızca bu sunucunun sertifikası (SHA-256
  `0F:5A:06:29:C0:E9:40:09:B6:8F:98:B1:CD:9C:B0:21:CB:9B:2E:F1:EE:90:DF:6D:79:3F:CF:96:0D:06:AE:4F`, geçerlilik
  2026-02-03 … 2027-02-03) proxy CA paketine eklenerek `curl --cacert` ile kullanıldı; sayfa listesi sitenin
  `searchindex.js` dosyasından geldi (241 sayfa). Doğrulama hiçbir yerde kapatılmadı. Yeniden almak için sahibin
  yeniden onayı gerekir; sertifika yenilenirse parmak izi değişir.
