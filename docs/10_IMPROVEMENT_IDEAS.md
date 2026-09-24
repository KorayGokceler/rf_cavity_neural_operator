# 10 — Geliştirme Fikirleri & Yol Haritası

> Tüm modüllerden toplanan iyileştirme önerileri, öncelik ve zorluk seviyeleriyle birlikte.

---

## 🟢 Düşük Zorluk — Hemen Uygulanabilir

### 1. `visualize_features.py` Güncellemesi ✅
- **Durum:** Tamamlandı — feature haritası güncel 8-feature çıkarıcıya göre yazıldı.
- **Çözüm:** Feature isimleri ve indeksleri güncellenmeli.
- **Etki:** Debug kalitesi artar.

### 2. Sayısal Inference Raporu ✅
- **Durum:** Tamamlandı — `scripts/infer_val_all.py` tüm split için CSV/JSON/NPZ üretir; `scripts/analyze_val_errors.py` grafikleri çizer.
- **Çözüm:** CSV/JSON ile Rel L2, frekans hatası, mode-bazlı istatistikler kaydet.
- **Etki:** Deneyler arası karşılaştırma kolaylaşır.

### 3. Hata Haritası (Error Map) Paneli ✅
- **Durum:** Tamamlandı — `infer.py` her mod için GT | Pred | Error sütunları çizer.
- **Çözüm:** `infer.py`'a 3. sütun olarak hata haritası ekle.
- **Etki:** Modelin nerede hata yaptığı hemen görülür.

### 4. `Notlarim.txt` Tamamlanması
- **Durum:** `dataset_converter.py` kısmı eksik bırakılmış.
- **Çözüm:** Bu dokümanlardan bilgi çekilip tamamlanabilir.

---

## 🟡 Orta Zorluk — Önemli İyileştirmeler

### 5. Data Augmentation (Geometri Dönüşümleri) 🟨 kısmen
- **Durum:** `training.augment: true` ile rotasyon + yansıma augmentasyonu mevcut (`configs/spectral_no.yaml`).
- **Fikir:** Mevcut geometrilere rotasyon, aynalama, küçük pertürbasyon uygulayarak dataset'i büyüt.
- **Dikkat:** PCA-based features (cos/sin_principal) rotasyonla bozulur → feature'lar yeniden hesaplanmalı.
- **Etki:** Generalizasyon ↑, özellikle az veri durumunda.

### 6. Sobolev Loss (Gradyan Tabanlı Kayıp)
- **Fikir:** Sadece $E$ değil, $\nabla E$'yi de hedefle. $\mathcal{L} = \|E_{pred} - E_{true}\|^2 + \alpha\|\nabla E_{pred} - \nabla E_{true}\|^2$
- **Zorluk:** Gradyanın mesh üzerinde hesaplanması gerekli (FEM basis fonksiyonları veya otomatik diferansiasyon).
- **Etki:** Fiziksel olarak daha pürüzsüz tahminler.

### 7. Dynamic Batching (BucketBatchSampler)
- **Fikir:** Benzer mesh boyutlarını aynı batch'e koyarak padding israfını azalt.
- **Etki:** Eğitim hızı %20-40 artabilir.

### 8. Curriculum Learning
- **Fikir:** Eğitime basit geometrilerle (daire, kare) başla, sonra karmaşık şekillere geç.
- **Uygulama:** Epoch bazlı dataset filtreleme.
- **Etki:** Daha hızlı yakınsama.

### 9. Mixed Precision Training (AMP)
- **Fikir:** Float16 ile eğitim → bellek ve hız kazanımı.
- **Uygulama:** PyTorch Lightning'te `precision=16` flag'i.
- **Etki:** ~2x hızlanma, VRAM %40 tasarruf.

---

## 🔴 Yüksek Zorluk — Araştırma Seviyesi

### 10. Graph Neural Network Entegrasyonu
- **Fikir:** Mesh topolojisini (element bağlantıları) doğrudan kullan. Şu an `elements` tensörü modelde hiç kullanılmıyor!
- **Uygulama:** GNN (Message Passing) katmanları → her node komşularıyla bilgi paylaşır.
- **Etki:** Lokal mesh yapısı doğrudan öğrenilir.

### 11. Helmholtz Residual Loss (Fizik-Bilinçli Regularization)
- **Fikir:** Tahmin edilen alanın $\nabla^2 E + k^2 E = 0$ denklemini ne kadar sağladığını ek loss terimi olarak ekle.
- **Zorluk:** $\nabla^2$ hesaplamak için mesh üzerinde ikinci türev gerekli.
- **Etki:** Model fizik denklemlerini daha iyi öğrenir.

### 12. Orthogonality Constraint 🟨 kısmen
- **Durum:** SpectralNO'da M-ortogonallik yapısal (`eigh`); GNOT için `slot_ortho_weight` ve Grassmann/subspace kaybı mevcut.
- **Fikir:** Modlar birbirine dik olmalı: $\int E_m \cdot E_n \, dA = 0$ (m ≠ n).
- **Uygulama:** Aynı geometrinin 3 modu için ortogonalite loss'u.
- **Zorluk:** Batch içinde aynı geometrinin tüm modlarını eşlemek gerekir.
- **Etki:** Mode karışımını önler.

### 13. 3D Kavitelere Genişleme
- **Fikir:** 2D kesit yerine tam 3D kavite geometrileri.
- **Zorluk:** 3D mesh = tetrahedra, node sayısı 10-100x artar. VRAM ve hesaplama maliyeti çok yükselir.
- **Uygulama:** Point cloud tabanlı yaklaşım (PointNet++) veya voxel tabanlı yaklaşım.

### 14. Inverse Design
- **Fikir:** "İstediğim frekans $f^*$ olan kavite şekli ne olmalı?" sorusunu yanıtla.
- **Uygulama:**
  - **Yöntem A:** Gradient-based optimization — GNOT frozen, geometri parametreleri optimize edilir.
  - **Yöntem B:** Conditional generative model (cVAE, Diffusion) — frekans verilip geometri üretilir.
- **Etki:** Otomatik kavite tasarımı — hızlandırıcı mühendisliğinde devrim.

### 15. Uncertainty Quantification
- **Fikir:** Her tahmin için güven aralığı.
- **Uygulama:** MC-Dropout, Deep Ensemble, veya Bayesian NN.
- **Etki:** "Bu tahminine ne kadar güveniyorsun?" sorusunu yanıtla.

---

## 🧹 Repo Sağlığı (2026-09-24 denetimi)

- ✅ `requirements.txt` alt sürüm sınırları; eksik `tensorboard` (TensorBoardLogger) ve `pandas` eklendi; `requirements-dev.txt` (pytest, ruff).
- ✅ `pyproject.toml` (pytest `testpaths`/marker'lar, ruff kuralları) ve `.github/workflows/ci.yml`.
- ⬜ `infer.py`: iki `main` ve iki `if __name__ == "__main__"` bloğu var → script inference'ı iki kez çalıştırıyor; `--deg_threshold` verilirse ikinci parser hata veriyor.
- ⬜ `scripts/debug_modes.py` eski API'ye göre yazılmış (çalışmıyor); `run_ablation.py` ve `scripts/check_mode_data.py` argparse'sız (`--help` bile iş başlatıyor).
- ⬜ `run_pipeline.py` alt süreçleri `"python"` ile başlatıyor → `sys.executable` kullanılmalı (venv/Colab uyumsuzluğu).
- ⬜ `configs/spectral_no.yaml`: `data_convert.output_path` (`..._v12.pkl`) ile `dataset.data_path` (`gnot_dataset_5k.pkl`) farklı; olmayan `scripts/convert_dataset.py`'ye atıf var.
- ⬜ Checkpoint adı `best-{epoch:02d}-{val/field_rel_l2:.4f}` → `/` yüzünden alt klasör oluşuyor.
- ⬜ Büyük/tekrarlı dosyalar: `examples/2302.14376v3 (2).pdf` (4.3 MB, arXiv'de mevcut), `examples/rf_cavity.ipynb` (2 MB, çıktılar dahil) ve kopyası; `GNOT_Colab_Runner.ipynb` `.gitignore`'da olduğu halde takipte.

---

## 📊 Öncelik Matrisi

| ID | İyileştirme | Zorluk | Etki | Öncelik |
|----|-------------|--------|------|---------|
| 1 | visualize_features güncelle | 🟢 | Düşük | ✅ Tamam |
| 2 | Sayısal inference raporu | 🟢 | Orta | ✅ Tamam |
| 3 | Hata haritası | 🟢 | Orta | ✅ Tamam |
| 9 | Mixed Precision | 🟡 | Yüksek | Kısa vadeli |
| 7 | Dynamic Batching | 🟡 | Orta | Kısa vadeli |
| 5 | Data Augmentation | 🟡 | Yüksek | Orta vadeli |
| 6 | Sobolev Loss | 🟡 | Yüksek | Orta vadeli |
| 10 | GNN Entegrasyonu | 🔴 | Çok Yüksek | Uzun vadeli |
| 14 | Inverse Design | 🔴 | Çok Yüksek | Uzun vadeli |
| 13 | 3D Genişleme | 🔴 | Çok Yüksek | Uzun vadeli |

---

## 🔗 Bağlantılar

- Dashboard: [[00_DASHBOARD]]
- Fizik: [[09_PHYSICS_BACKGROUND]]
- Model: [[04_MODEL_ARCHITECTURE]]
- Eğitim: [[05_TRAINING_SYSTEM]]

#gelistirme #roadmap #fikirler #todo
