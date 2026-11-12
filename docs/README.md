# Belgeler

Güncel çalışma belgeleri (kod bunlara göre yazılır):

| belge | konu |
|---|---|
| [24_CAVITY_QOI](24_CAVITY_QOI.md) | kavite büyüklükleri (Q0, G, R/Q, R_sh, T, Epk/Eacc, Bpk/Eacc): sözleşme (`src/qoi/figures.py`, `highorder.mode_qoi`), fizik; §2–§5 kaldırılan N0 uygulaması (tarihsel) |
| [26_WEB_UI](26_WEB_UI.md) | web arayüzü: API, dataset gezgini, üretim, eğitim, Colab / telefon |
| [27_LABEL_BENCHMARK](27_LABEL_BENCHMARK.md) | tarihsel (N0 etiket türü ve onu ölçen betikler kaldırıldı; tablolar kayıt olarak durur): etiket kalitesi: N0 ve NGSolve p2/p3 eğri eleman etiketleri, frekans + alan + QoI |
| [28_FIELD_MODEL](28_FIELD_MODEL.md) | model (`field3d`, EigenspaceOperatorField): yüksek mertebe H(curl) uzayında öğrenilen alan, gövde, kernel projeksiyonu, Ritz |
| [29_FIELD_LABELS](29_FIELD_LABELS.md) | alan etiketleri: eğri mesh + p3 etiket → p2 model uzayı, maliyet, TRUBA tek satır komut ve arayüz sayfası |
| [30_ADAPTIVE_LABELS](30_ADAPTIVE_LABELS.md) | adaptif etiketler: hata göstergesi (hacim + duvar), yeniden mesh'leme, keskin kenar yuvarlatma, etiket / model mesh ayrımı, ölçümler |
| [31_SESSION_HANDOFF](31_SESSION_HANDOFF.md) | oturum devri: güncel durum, ölçümler (eğitim adımı, veri hazırlama, çözücü, CG), kararlar, açık işler; her oturum sonunda güncellenir |
| [32_AGENT_WORKFLOW](32_AGENT_WORKFLOW.md) | ajanlarla çalışma: kurulum (Claude Code ve diğer araçlar), roller ve modeller, worktree ile paralel iş, görev yaşam döngüsü, mesaj şablonları |
| [23_ROADMAP](23_ROADMAP.md) | yol haritası |
| [../cluster/truba/README.md](../cluster/truba/README.md) | TRUBA (SLURM) betikleri |

Araştırma ve inceleme kayıtları (tarihsel; kararların gerekçesi burada, kodla birebir güncel tutulmaz):

| belge | konu |
|---|---|
| [19_3D_DATA_PIPELINE](19_3D_DATA_PIPELINE.md) | kaldırılan N0 modelinin veri hattı: geometri aileleri, N0 E-alanı üreticisi, H5 / PKL sözleşmesi, doğruluk ölçümleri (§6) |
| [20_3D_MODEL](20_3D_MODEL.md) | kaldırılan N0 modeli EigenspaceOperator3D (`eigenspace3d`): gövde, kenar başlığı, kernel projeksiyonu, Ritz, kayıplar |
| [25_PIPELINE](25_PIPELINE.md) | N0 modelinin uçtan uca akışı: üretim → PKL birleştirme → eğitim → değerlendirme; §7 (yeni kavitenin tahmini) field3d için güncel |
| [18_3D_EXTENSION](18_3D_EXTENSION.md) | 3B'ye geçiş araştırması: çözücü seçenekleri, N0 / NGSolve yüksek mertebe deneyleri (E2'), eksenel simetrik çözüm |
| [21_3D_REVIEW_PHYSICS](21_3D_REVIEW_PHYSICS.md) | fizik / sayısal analiz incelemesi ve bulguları |
| [22_3D_REVIEW_MODEL](22_3D_REVIEW_MODEL.md) | model / eğitim kodu incelemesi ve bulguları |
| [33_GEOMETRY_REVIEW](33_GEOMETRY_REVIEW.md) | geometri ailelerinin literatürle karşılaştırılması (2026-10): etiket tanımı sorunları, aile aile değerlendirme, yeni aileler, deformasyon, doğrulama seti, sahibin kararları |
| [34_MULTI_FIDELITY](34_MULTI_FIDELITY.md) | çoklu doğruluk: düşük doğruluklu (p2, tek çözüm) etiketle eğitim, yüksek doğruluklu (p3) veride ve daha ince mesh / p3 uzayında çıkarım; maliyet ve etiket farkı ölçümleri, `Fidelity` / `fidelity_weights` / `train_extra` / `--init_from`, `scripts/eval_fidelity.py` |
| [35_PAPER_CHECKS](35_PAPER_CHECKS.md) | preprint için hakem kontrolleri (İngilizce): varsayım (A) rank denetimi, Ritz üst sınırı stres testi (inexact CG), parametrik regresyon ve kaba FE taban çizgileri, makale şekilleri; betikler `scripts/paper/` |
| [36_TRUBA_REFERENCE](36_TRUBA_REFERENCE.md) | TRUBA platform bilgisi (docs.truba.gov.tr tam metninden): kullanım kuralları, kuyruklar, GPU erişimi, kotalar, depolama (scratch 30 gün), conda yasağı ve Apptainer, `config.sh` ile karşılaştırma; ham sayfalar [research/truba](research/truba/README.md) |
| [37_P2_LABELS](37_P2_LABELS.md) | varsayılan eğitim etiketi p2 (model mesh'inde tek çözüm): karar ve gerekçe, `model_sized_mesh` kuralı ve aile tablosu, p2 doğruluğu, `label_ok`, p4 referans aracı (`scripts/reference_check.py`), açık işler |
| [38_PROJECTION_FREE_TRAINING](38_PROJECTION_FREE_TRAINING.md) | `training.projection: false`: eğitim ve doğrulama çekirdek projeksiyonu ve Rayleigh–Ritz olmadan (Kp CG yok, G/Kp yok), model seçimi `val/span_loss` ile, test projekte; matematik (GᵀMT = 0, r_düz ≥ r_proj, gradyan-kör ortho, ölçek-bağımsız ölü sütun kuralı, çökmüş sütun), `grad_frac_lb` alt sınırı, kaybedilenler, aktarılmış etiket uyarısı, kullanım (config, TRUBA), `dataset.max_tets` ve V100 bellek ölçümü, ölçümler, operatör önbelleği biçimi (üst üçgen K/M, float32 nopot, zlib; 5,2× / 2,55× küçük, §9), Colab, verim (§11: `scripts/profile_loader.py`, `training.profile_steps`, host senkronizasyonlarının kaldırılması, `diag_metrics_every_n_steps`, `gpu_prefetch`, `cache_compress` / `recode_ops_cache.py`, `prefetch_factor`) |
| [39_ROADMAP](39_ROADMAP.md) | Uzun vadeli plan: kritik riskler (FE'ye göre gerçek kazanç, enerji normu ↔ frekans hatası ε_A², izdüşümsüz eğitimin bedeli, tahmin ağı kayması, OOD), model / veri ölçekleme deneyi, kayıp ve mimari kaldıraçlar, GPU veri yolu (matrissiz operatörler), aşamalar, karar ölçütleri, kaynak ihtiyacı |
| [research/2026-10_geometry](research/2026-10_geometry/README.md) | docs/33'ün ham malzemesi: dört araştırma raporu (İngilizce), kaynak listeleri ve sayıları üreten betikler |

2D model (GNOT, SpectralNO, 2D eigenspace operator) ve onun belgeleri (eski 00–17) ile H-alanı formülasyonu
kaldırıldı; `legacy-2d` (2D) ve `legacy-h` (H formülasyonu) dallarında duruyorlar. N0 kenar modeli
(`eigenspace3d`, PKL hattı, `convert_3d.py`, `eval_3d.py`) de kaldırıldı (git geçmişinde); belgeleri 19, 20, 25
tarihsel, 23, 24, 26, 29, 34'te ilgili kısımlar tarihsel.
