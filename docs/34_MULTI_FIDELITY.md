# 34 — Çoklu doğruluk: düşük doğruluklu etiketle eğitim, yüksek doğruluklu veride çıkarım

> Durum: kod ve ölçümler hazır (dal `claude/field-model`, açık iş 13). **§8: eşit maliyetli deney** — LF ile eğitilen
> model, aynı ya da daha düşük etiket maliyetinde HF ile eğitilen kadar iyi (LF 40'ta biraz daha iyi); LF→HF boşluğu
> 0,02 puan. LF etiketleri artık HF'nin model mesh'inde üretiliyor (§8.1). Tüm eğitimler CPU'da, küçük model, tek
> aile (`pillbox_pipes`): sayılar yöntemin tavanını değil, etiket türünün etkisini ölçer.
> Bağlam: docs/28 (model), docs/29 (etiketler), docs/30 (uyarlamalı etiketler), docs/31 §3.

## 1. Soru ve tanımlar

- **HF etiket:** uyarlamalı p3 çözümü (`--labels field`, `--max_ndof 150000`); frekans ve QoI p3'ten, eğitim
  alanı p3 alanının model mesh'indeki p2'ye izdüşümü.
- **LF etiket:** model mesh'inde tek p2 çözümü (`--no_adapt --label_order 2 --model_order 2`). Izdüşüm hatası yok
  (`proj_err ~1e-7`), frekanslar modelin kendi uzayının ayrık özdeğerleri; Ritz değerleri bunlara tam
  ulaşabilir.
- Soru: LF ile eğitilen model HF verisinde (HF frekans etiketi, HF model mesh'i) iyi çıkarım yapar mı; Ritz
  katmanı hangi uzayda çalışıyorsa orada çalıştığına göre, daha ince mesh / p3 uzayında da çalışır mı?

## 2. Kod değişiklikleri

| dosya | değişiklik |
|---|---|
| `src/data/field_dataset.py` | `label_fidelity(attrs)`: `adapt_iters` var ya da `label_order > model_order` ise HF (1), değilse LF (0). Her örnek `Fidelity` taşır, `field_collate` istifler. `hf_repeat`: HF geometrileri train bölünmesinde k kez listeler. `train_extra`: ek H5 dosyaları (HF); geometrileri hep train'e gider, val/test'e gitmez. |
| `src/training/lightning_module.py` | `fidelity_weights=(LF, HF)`: span ve frekans terimlerinde örnek başına ağırlık, kayıp = ortalama_b(w_b · kayıp_b). `fidelity_freq_scale=(LF, HF)`: yalnız frekans terimine ek çarpan (HF frekansları p2 tabanının altında, bkz. §4). Batch'te her iki doğruluk varsa `{train,val,test}/freq_rel_err_{lf,hf}` ve `field_rel_l2_{lf,hf}` ayrı loglanır. Varsayılanlar eski davranışı verir. |
| `train.py` | `--init_from <ckpt>`: yalnız ağırlıkları yükler (optimizer / epoch sıfır), LF'den HF'ye ince ayar için. `dataset.hf_repeat`, `dataset.train_extra`, `training.fidelity_*` config'ten geçer. |
| `configs/field_3d.yaml` | yukarıdaki seçenekler, açıklamalı, varsayılanlar etkisiz. |
| `src/data_gen/dataset_generator_3d.py` | `--curvaturesafety` (varsayılan değişmez). Ölçümde mesh boyutunu değiştirmedi (§3), yine de ucuz LF için ileride gerekebilir. |
| `scripts/eval_fidelity.py` | Bir checkpoint'i H5'teki geometrilerde üç uzayda değerlendirir: `native` (saklı model mesh'i, HF etiketler), `p3` (aynı mesh, HCurl p3, p3 çözümüne karşı), `refine` (mesh bir kez düzgün inceltilir). Frekans hatası HF etikete ve değerlendirilen uzayın FE özdeğerine karşı, alan rel-L2, süreler. |
| `tests/test_field_labels.py` | `test_multi_fidelity_items_split_and_loss_weights`: sınıflandırma, `train_extra` / `hf_repeat`, ağırlıklı span terimi. |
| `src/data_gen/field_labels.py` | `start_mesh`: uyarlamalı yolun başlangıç mesh'i (`adapt_cs`, katlanırsa `curvaturesafety`); LF (`--no_adapt`, `label_order == model_order`) de bunu kullanıyor (§8.1). |
| `src/data/field_dataset.py` | Frekans normalizasyonu train bölünmesinden; bölünme boşsa (`train_ratio 0`) `train_extra` geometrilerinden (önceden val/test dosyasından alınıyordu). |

## 3. Veri ve maliyet

Aile `pillbox_pipes`, `--min_fillet 0.12`, `--sampling sobol --seed 0 --deform_prob 0`, kimlik 0–23.
Aynı kimlikler LF ve HF üretiminde aynı geometriyi verdi (`geom_params` birebir aynı, 4 örnekte denetlendi).
HF: kimlik 3, 5, 10, 13. Üretim: 2 işçi × 1 NGSolve iş parçacığı, 4 çekirdekli paylaşımlı VM.

| örnek | LF mesh + etiket [s] | LF tet | HF mesh + etiket [s] | HF model tet | HF etiket DOF |
|---|---|---|---|---|---|
| 3 | 1,4 + 24,7 | 6925 | 6,3 + 166,2 | 6925 | 186 576 |
| 5 | 1,0 + 20,5 | 6664 | 1,8 + 118,1 | 6061 | 163 176 |
| 10 | 2,5 + 33,8 | 9839 | 6,2 + 202,5 | 12 358 | 327 828 |
| 13 | 1,6 + 39,1 | 9234 | 1,0 + 124,7 | 8907 | 237 264 |
| **ortalama** | **31 s** | | **157 s** | | |

- Örnek başına maliyet oranı HF / LF ≈ **5**. Tüm 24 LF örnek 8 dk, 4 HF örnek 5,5 dk sürdü.
- HF'de `--max_ndof 150000` bütçesi ilk p3 çözümünden sonra doluyor (`adapt_iters = 1`, `adapt_stop = budget`):
  bu aile ve bu bütçede "HF" = başlangıç mesh'inde (curvaturesafety 1,5) p3 çözümü. Etiket mesh'i model mesh'i ile aynı.
- **Düzeltildi (§8.1):** aşağıdaki "mesh küçülmüyor" gözlemi yanlış teşhisti; LF yolu HF'nin başlangıç mesh'i yerine
  `curvaturesafety 2,0` kullanıyordu. **Eski gözlem:** LF "ucuz" olsun diye mesh küçülmüyor: `--curvaturesafety 1,0 / 1,5 / 2,0`, `--maxh_factor 3 / 8` aynı tet
  sayısını verdi; boyutu pervazlar ve borular belirliyor (docs/28 §4). `--min_fillet 0,05 → 0,12` yalnız
  8,2k–26,6k'yı 6,7k–15,6k tete indirdi. Kazanç p2 / p3 farkından geliyor (DOF ≈ 2,5 kat daha az, adaptif döngü yok).
  Bu yüzden LF'nin 0,05 pervazlı 16 örneklik ilk denemesi (8k–26k tet) örnek başına 123 s sürdü; ucuz LF için
  mesh'in kendisi küçülmeli (kaba mesh, daha yüksek pervaz tabanı).

### LF ve HF etiket farkı (aynı geometri, mod 0 QoI)

| örnek | f_LF/f_HF − 1, modlar 0–3 [%] | Q0 | R/Q | E_pk/E_acc | B_pk/E_acc | HF `proj_err` |
|---|---|---|---|---|---|---|
| 3 | +0,004 / +0,016 / +0,018 / +0,021 | −0,97 % | −0,005 % | +6,2 % | +8,1 % | 0,18–0,29 % |
| 5 | +0,007 / +0,021 / +0,021 / +0,020 | −0,82 % | +0,001 % | −28 % | +6,1 % | 0,31–0,42 % |
| 10 | +0,008 / +0,012 / +0,013 / +0,015 | −1,37 % | −0,039 % | −4,2 % | +5,4 % | 0,10–0,32 % |
| 13 | +0,004 / +0,011 / +0,011 / +0,016 | −0,76 % | +0,003 % | −13,6 % | +5,7 % | 0,14–0,26 % |

- Bu ailede LF frekansları HF'ye **1e-4 mertebesinde** yakın (üst sınır, hep pozitif); R/Q aynı; modun
  **duvara bağlı** büyüklükleri (Q0, G, R_sh −1 %, yüzey tepeleri %5–28) farklı.
- Yani frekans ve hacim alanı için LF yeterli; Q0 / tepe değerleri için HF etiket gerekir (bu çalışmada
  QoI ağırlığı kapalı: `qoi_weight = 0` field3d için).
- Alan karşılaştırması DOF düzeyinde yapılamaz: LF ve HF model mesh'leri farklı.

## 4. Deney: LF ile eğit, HF'de değerlendir

Model: `embed_dim 64`, 2 blok, 4 kafa, `n_basis 12`, `rff_dim 32`, 4 mod, batch 1,
`lr 1e-3` (30 epoch, 10 epoch sabit sonra kosinüs), 2 iş parçacığı. Eğitim: 14 LF geometri (kimlik 3, 5, 10, 13
**hariç** 20 LF'nin %70'i), val 3, test 3 (hepsi LF). Epoch ≈ 1,7–2,5 dk, toplam ≈ 65 dk. HF geometriler
eğitime hiç girmedi.

LF test (3 LF geometri, en iyi val epoch'u 24): alan rel-L2 0,182, frekans bağıl hata %2,07, span rel-L2 0,078.
Eğitim rel-L2 0,17, val 0,22–0,27: 14 örnekte belirgin aşırı uydurma / az eğitim.

### 4.1 HF geometrilerde (saklı HF model mesh'i, p2, HF frekans etiketi)

Frekans hatası f_model / f_HF − 1 [%], modlar 0 / 1 / 2 / 3; alan rel-L2 modlar 0 / 1 / 2 (mod 3 aşağıda):

| örnek | frekans hatası [%] | alan rel-L2 (mod 0–2) | FE tabanı (p2 FE / HF − 1) |
|---|---|---|---|
| 3 | +1,61 / +1,05 / +1,16 / +2,51 | 0,037 / 0,064 / 0,072 | ≤ 0,021 % |
| 5 | +1,29 / +0,63 / +0,68 / +1,89 | 0,026 / 0,043 / 0,037 | ≤ 0,033 % |
| 10 | +2,12 / +1,06 / +1,12 / +3,02 | 0,055 / 0,035 / 0,042 | ≤ 0,009 % |
| 13 | +2,06 / +1,05 / +1,10 / +2,83 | 0,038 / 0,056 / 0,059 | ≤ 0,017 % |

- Tüm hatalar pozitif ve değerlendirilen uzayın FE özdeğerinden büyük (en küçük fark +0,006 %): **Ritz üst
  sınırı korunuyor.**
- Hata p2 tabanından değil modelden geliyor: HF etiket ile p2 taban arasındaki fark ≤ 0,033 %, modelin hatası
  0,6–3 %. Bu ailede LF etiketin frekansı yanıltması söz konusu değil; darboğaz modelin doğruluğu.
- Mod 3'ün alan hatası 0,07 (örnek 10) ile 1,3 (örnek 13) arasında; örnek 3 ve 5'te 1,0 ve 0,58. Model bu modu
  bulamıyor ya da başka bir modla karıştırıyor (nedeni incelenmedi); 4 modlu bu küçük model için dördüncü mod
  henüz öğrenilmemiş.
- HF geometride LF-geometri performansına yakın (LF test frekans hatası %2,07 ve alan 0,182, HF'de mod 0–2
  ortalaması frekans %0,6–2,1 ve alan 0,035–0,058): LF'de eğitip HF'de değerlendirmenin **ek bir boşluğu
  görülmedi**, fakat bu, etiketlerin bu ailede zaten çok yakın olmasının sonucu.

### 4.2 Modelin başka uzayda çalıştırılması (aynı ağırlıklar)

Ritz katmanı yalnız K, M, G, Kp'yi kullandığı için model kod değişmeden ince mesh / p3 uzayına verildi.
Hedef: o uzayda ayrıca hesaplanan FE özdeğer / alanları.

| örnek, uzay | DOF | frekans hatası vs HF [%] modlar 0–3 | alan rel-L2 (0–2) | ops + ileri [s] | o uzayda FE çözümü [s] |
|---|---|---|---|---|---|
| 3, p2 (yerel) | 74 301 | +1,61 / +1,05 / +1,16 / +2,51 | 0,037 / 0,064 / 0,072 | 3,8 + 1,3 | 11,5 |
| 3, p3 aynı mesh | 186 576 | +2,29 / +1,39 / +1,51 / +3,23 | 0,038 / 0,065 / 0,073 | 12,6 + 5,1 | 48,1 |
| 5, p2 (yerel) | 64 965 | +1,29 / +0,63 / +0,68 / +1,89 | 0,026 / 0,043 / 0,037 | 1,7 + 1,2 | 6,4 |
| 5, p3 aynı mesh | 163 176 | +1,88 / +0,88 / +0,94 / +2,53 | 0,027 / 0,044 / 0,037 | 8,4 + 4,4 | 27,6 |
| 5, p2, mesh bir kez inceltilmiş (×8 tet) | 489 747 | +1,89 / +0,86 / +0,89 / +2,29 | 0,027 / 0,043 / 0,037 | 32,3 + 22,0 | 192 |

- **Çalışıyor:** ağırlıklar yeniden eğitilmeden hem p3'te hem 8 kat ince mesh'te çıkarım yapıyor; Ritz değerleri
  o uzayın FE özdeğerinin üstünde kalıyor (min. fark +0,009 %).
- **Doğruluk iyileşmiyor:** frekans hatası p2 yerel çıkarıma göre 0,2–0,6 puan *kötüleşiyor*, alan hatası aynı.
  Beklenen: uzay zenginleşiyor ama ψ aynı pürüzsüz alanlar; asıl hata ψ'nin (ağın) kalitesinde. Token yoğunluğu
  (p3'te 24, p2'de 14 nokta / tet) ve nokta-hacim özelliği eğitimdekinden farklı: küçük bir dağılım kayması.
- **Maliyet:** bu ölçekte (≤ 12k tet, p2) doğrudan FE çözümü (6–19 s) modelin ops + ileri süresiyle (3–8 s) aynı
  mertebede; model fark yaratmak için daha ince uzaylarda başlıyor (p3: ~2,5 kat, ×8 mesh: ~3,5 kat), ve oradaki
  hatası %1–2. Bu CPU, 2 iş parçacığı ölçümü; GPU'da ağın payı küçülür.

### 4.3 LF + az HF ince ayar (kontrollü)

HF kimlik 3, 5 ince ayara (6 HF öğe / epoch, `hf_repeat 3`, yani 14 LF + 6 HF), kimlik 10, 13 değerlendirmeye.
LF en iyi checkpoint'ten başlatıldı (`--init_from`), 8 epoch, `lr 3e-4`. **Kontrol:** aynı checkpoint'ten
yalnız LF ile aynı 8 epoch.

Ortalama |frekans hatası| (modlar 0–3, %) / alan rel-L2 (mod 0–2 ortalaması), HF değerlendirme geometrileri:

| model | örnek 10 | örnek 13 |
|---|---|---|
| LF-only (başlangıç) | 1,83 / 0,044 | 1,76 / 0,051 |
| LF-only +8 epoch (kontrol) | **1,50 / 0,041** | **1,50 / 0,050** |
| LF + 2 HF ince ayar +8 epoch | 1,73 / 0,045 | 1,50 / 0,048 |

HF'ye ince ayar yapılan geometrilerde (eğitim kümesi): kontrol 1,36 / 0,92 %, ince ayar 1,35 / 0,94 % (örnek
3 / 5): fark yok.

**Sonuç:** 2 HF örneğin LF'ye göre ölçülebilir bir kazancı yok; kazanç (1,8 → 1,5) ek eğitimden geliyor. Bu
beklenen: §3'te gösterildiği gibi bu ailede frekans ve alan etiketleri LF ve HF için zaten 1e-4 mertebesinde
aynı; HF'nin eklediği yalnız duvara bağlı QoI, ve onu eğiten bir terim bu çalışmada kapalı (`qoi_weight = 0`).
İnce ayarın (ya da `fidelity_*` ağırlıklarının) değeri, LF ve HF'nin gerçekten ayrıştığı durumda sınanmalı:
küçük pervazlı, tepe değerleri tekil ailelerde, ya da p2 tabanı %0,04–0,9 olan (docs/30) kaba mesh'lerde.

## 5. Dürüst sınırlar

- Model 14 geometride az eğitildi (eğitim 0,17 / val 0,22 rel-L2); mutlak hatalar (%1–3) ölçekli eğitimle
  düşecek. Yalnız göreli karşılaştırmalar (aynı model, farklı uzay / etiket) anlamlı.
- 4 HF geometri, tek aile (`pillbox_pipes`, pervaz tabanı 0,12). `reentrant` ve `elliptical` denenmedi
  (mesh'leri 10k–26k tet; CPU'da eğitilemez).
- HF = bütçeli p3 (tek adım); gerçekten uyarlamalı, daha ince etiket mesh'i olan örneklerde LF–HF farkı daha
  büyük olabilir.
- İnce mesh çıkarımında FE hedefi o uzayın kendi çözümü; "HF" etiketi değil.
- §3'teki HF üretim süreleri `datagen-speed` birleşmeden önce alındı (sonrasında ~%30 hızlı bekleniyor); LF/HF
  oranı bu yüzden biraz değişebilir.

## 6. Hibrit çıkarım (denenmedi) ve açık işler

- **Hibrit** (modelin alt uzayı → birkaç LOBPCG / Ritz iterasyonu, HF uzayında): kodlanmadı. Gerekçe: ölçülen
  doğrudan çözüm bu boyutta ucuz (p2'de 6–19 s, p3'te 28–48 s; 490k DOF'ta 192 s); modelin ops + ileri maliyeti de
  aynı mertebede. Yararı ancak çözücü süresi dakikalara çıktığında (docs/31 §3.4: 358k DOF'ta sıcak başlatma 10
  iterasyonda 36 s) ve modelin başlangıç hatası ≤ %0,1 olduğunda görülür. İlk sıra modelin doğruluğu.
- Eğitim: uygun ölçekte (yüzlerce LF + onlarca HF) `fidelity_freq_scale` ve `hf_repeat` süpürmesi; QoI kaybı
  field3d'de açılana kadar HF'nin Q0 / tepe bilgisi öğrenilemez.
- Ucuz LF için mesh'i küçülten bir ayar (üretici `--min_fillet` tabanı yükseltir; ya da p1 model uzayı, docs/29).
- Aynı eğitim kodu N0 (`eigenspace3d`) etiketini LF olarak kullanamaz: farklı DOF uzayı.

## 7. Yeniden üretme

```bash
G="--labels field --families pillbox_pipes --seed 0 --sampling sobol --deform_prob 0 --min_fillet 0.12 --n_workers 2 --threads 1"
python src/data_gen/dataset_generator_3d.py $G --no_adapt --label_order 2 --model_order 2 --n_total 24 --h5_filename lf.h5
python src/data_gen/dataset_generator_3d.py $G --max_ndof 150000 --ids_file hf_ids.txt --h5_filename hf.h5   # 3 5 10 13
python train.py --config configs/field_3d.yaml --override dataset.data_path=lf_train.h5 dataset.n_modes=4 \
    model.embed_dim=64 model.n_basis=12 model.num_field_modes=4 model.rff_dim=32 model.eigenspace.n_layers=2 \
    training.batch_size=1 training.num_workers=0 dataset.train_ratio=0.7 dataset.val_ratio=0.15
python train.py ... --init_from lf_best.ckpt --override dataset.train_extra=hf_ft.h5 dataset.hf_repeat=3 training.learning_rate=3e-4
python scripts/eval_fidelity.py --ckpt lf_best.ckpt --data hf.h5 --spaces native p3 refine --ids 1
```

`lf_train.h5` / `hf_*.h5`: `pp_lf.h5` ve `pp_hf.h5`'nin kimliğe göre bölünmüş kopyaları (h5py ile `copy`).

## 8. Eşit maliyetli deney: LF ile eğit, HF'de çıkarım (2026-10-24)

### 8.1 LF etiketleri HF'nin model mesh'inde

- Önceki LF yolu (`--no_adapt`) mesh'i `curvaturesafety 2,0` ile kuruyordu. Uyarlamalı (HF) yol ise `adapt_cs 1,5` ile
  başlıyor, eleman katlanırsa 2,0'a düşüyor. Sonuç: LF mesh'leri HF model mesh'lerinden 1,2–1,9 kat büyüktü (6 test
  geometrisinde 5,9k–12,1k tet, HF'de 4,8k–10,0k). LF hem pahalıydı hem de eğitimdeki mesh dağılımı çıkarımdakinden
  farklıydı.
- Yeni konteynerde (ngsolve 6.2.2608) kimlik 3 ve 24 için LF ve HF mesh'i yine aynı. Tet sayıları sürüm farkıyla biraz
  değişti (kimlik 3: 6754, önce 6925); LF etiketi 2,8–3,7 kat hızlı (eğitim yükü yokken).
- `--curvaturesafety 1,5` tek başına yetmiyor: 1,5'te katlanan geometrilerde LF yolu daha ince bir mesh'le yeniden
  deniyordu (ör. kimlik 3: 13,7k tet; HF 2,0'a düşüp 6,9k).
- **Değişiklik:** `field_labels.start_mesh` (uyarlamalı yolun başlangıç mesh'i seçimi, ayrı fonksiyon). `--no_adapt` ve
  `label_order == model_order` olduğunda LF de onu kullanıyor. Uyarlamalı yol ve uyarlamasız p3 değişmedi.
  Test: `test_low_fidelity_labels_start_from_the_adaptive_paths_mesh`.
- **Ölçüm:**
  - LF mesh'i, HF model mesh'iyle bayt düzeyinde aynı: 8/8 eğitim, 6/6 test geometrisi. HF'de `max_ndof 150000`
    bütçesi tek adımda doluyor, model mesh'i = başlangıç mesh'i. Daha büyük bütçede HF model mesh'i inceltilmiş bir
    seviye olabilir.
  - 40 LF örnekte ortalama 9404 → 7639 tet (−%19).
- **Maliyet** (aynı 8 mesh; 2 işçi × 1 iş parçacığı, 4 çekirdek, eşzamanlı eğitim yükü altında): HF 1242 s, LF 291 s,
  yani **HF/LF = 4,3**. 6 test geometrisinde HF 779 s, LF 190 s (4,1 kat). Eski LF mesh'leriyle oran 2,3'tü.
- **Etiket farkı, aynı mesh'te:**
  - frekans LF/HF − 1 = +0,004 – +0,047 %;
  - alan: HF modunun LF'nin 6 modluk uzayı dışında kalan kısmı, M-normunda, modlar 0–3: **%0,08–0,32**.
  - Modelin hatası bunun 10–50 katı.

### 8.2 Deney

- **Veri:** `pillbox_pipes`, `--min_fillet 0.12 --seed 0 --sampling sobol --deform_prob 0`.
  - HF eğitim: kimlik 0–7.
  - LF eğitim: kimlik 0–23 (LF 24) ve 0–23 + 30–45 (LF 40).
  - Test: kimlik 24–29, HF etiketli; aynı mesh'lerde LF etiketleri de var.
- **Model:** §4'teki model; `lr 1e-3`, `custom_cosine` (10 epoch sabit, sonra kosinüs), batch 1, 1 iş parçacığı.
- **Adım sayısı:** her kolda 600 adım. HF 8 için `hf_repeat 5`, yani epoch başına 40 öğe.
- **Değerlendirme:**
  - `last.ckpt` (checkpoint seçimi yok), `scripts/eval_fidelity.py`, saklı mesh, p2.
  - Val = test geometrilerinin 3'ü; yalnız izleme için kullanıldı.
  - Kurulum: `data_path` = test dosyası, `train_ratio 0`, eğitim verisi `train_extra` (glob ya da dizin).

| kol | eğitim verisi | etiket maliyeti [s] | adım | \|f hatası\| ort. / medyan / maks. [%] | alan rel-L2 (mod 0–2) |
|---|---|---|---|---|---|
| A | HF 8 | 1242 | 15 × 40 | 1,66 / 1,11 / 8,11 | 0,050 |
| B2 | LF 24 | 997 | 25 × 24 | 1,55 / 1,06 / 6,57 | 0,046 |
| C2 | LF 40 | 1704 | 15 × 40 | **1,37 / 1,00 / 5,30** | **0,045** |
| D | C2 + HF 8 ince ayar (5 × 40, `lr 3e-4`) | 1704 + 1242 | +200 | 1,13 / 0,82 / 4,41 | 0,039 |

- **Ana sonuç: LF ile eğitmek en az HF kadar iyi; daha çok LF geometri biraz daha iyi.** Her kol tek bir eğitim
  koşusu (tohum tekrarı yok); farklar küçük.
  - B2'nin etiket maliyeti A'dan %20 düşük; ortalama hatası A'dan düşük (%1,55 / %1,66), ama bu fark çoğunlukla aykırı
    geometri 27'den geliyor. 27 hariç ortalama: A %1,11, B2 %1,14, C2 %1,06.
  - C2 (maliyet 1,4 kat) A ile aynı öğrenme oranı programında koştu. 6 geometrinin 3'ünde (24, 27, 29) A'dan iyi,
    diğer üçünde 0,01–0,06 puan geride.
    Geometri başına ortalama hata: A 0,74 / 1,18 / 1,24 / 4,44 / 0,92 / 1,46; C2 0,66 / 1,23 / 1,25 / 2,93 / 0,98 / 1,17.
- **LF→HF boşluğu:** aynı tahminlerin LF etiketlerine göre hatası, HF'ye göre olandan yalnız 0,02 puan düşük
  (C2: %1,35 ve %1,37). LF ile eğitilen model HF verisinde, LF verisindeki kadar iyi.
- **Ritz üst sınırı korunuyor:** tüm tahminler, o uzayın FE özdeğerinin üstünde (en az +%0,47).
- **İnce ayar (D) açık bir soru:**
  - Testte C2'ye göre %1,37 → %1,13 iyileşti.
  - Kontrol D0 (aynı başlangıç, aynı 200 adım, yalnız LF 40) val'de 0,031 GHz / alan 0,046 verdi; D'de 0,029 / 0,052.
  - D0'ın test değerlendirmesi konteyner yeniden başlatılınca kayboldu. İyileşmenin HF'den mi ek eğitimden mi
    geldiği ayrılamadı (§4.3 de fark göstermemişti).
- **Geçersiz ve eksik kollar:**
  - LF 40 + HF 8 karışık 8 × 80 adım: öğrenme oranı hiç düşmedi (8 < 10 sabit epoch), %1,95; karşılaştırmaya alınmadı.
  - Aynı programla LF 32 + HF 8 kolu yeniden başlatmada epoch 6'da kayboldu.
- **Aykırı geometri:** örnek 27 her kolda en kötü (%2,9–4,4); nedeni incelenmedi.
- **Çıkarım süresi** (1 iş parçacığı): operatör 5 s + ileri geçiş 3,3 s; aynı uzayda FE çözümü 18 s.

### 8.3 Öneri

- **Veri:** HF bütçesini LF örnek sayısına çevirin. Aynı maliyete ~4 kat geometri düşer.
  Komut: `--labels field --no_adapt --label_order 2 --model_order 2`.
- **HF örnekleri** test ve doğrulama için ve QoI kaybı field3d'de açıldığında duvar büyüklükleri (Q0, tepe değerleri) için.
- **Ölçekte yapılacak:** karışık eğitim ve ince ayar, doğru programla (`max_epochs` > 10 ya da sabit faz kısaltılarak),
  bir kontrol koluyla birlikte; `fidelity_freq_scale` / `hf_repeat` taraması.
