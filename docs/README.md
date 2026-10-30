# Belgeler

Güncel çalışma belgeleri (kod bunlara göre yazılır):

| belge | konu |
|---|---|
| [24_CAVITY_QOI](24_CAVITY_QOI.md) | kavite büyüklükleri (Q0, G, R/Q, R_sh, T, Epk/Eacc, Bpk/Eacc): sözleşme, fizik, doğrulama |
| [26_WEB_UI](26_WEB_UI.md) | web arayüzü: API, dataset gezgini, üretim, eğitim, Colab / telefon |
| [27_LABEL_BENCHMARK](27_LABEL_BENCHMARK.md) | etiket kalitesi: N0 ve NGSolve p2/p3 eğri eleman etiketleri, frekans + alan + QoI |
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
| [research/2026-10_geometry](research/2026-10_geometry/README.md) | docs/33'ün ham malzemesi: dört araştırma raporu (İngilizce), kaynak listeleri ve sayıları üreten betikler |

2D model (GNOT, SpectralNO, 2D eigenspace operator) ve onun belgeleri (eski 00–17) ile H-alanı formülasyonu
kaldırıldı; `legacy-2d` (2D) ve `legacy-h` (H formülasyonu) dallarında duruyorlar. N0 kenar modeli
(`eigenspace3d`, PKL hattı, `convert_3d.py`, `eval_3d.py`) de kaldırıldı (git geçmişinde); belgeleri 19, 20, 25
tarihsel, 23, 24, 26, 29, 34'te ilgili kısımlar tarihsel.
