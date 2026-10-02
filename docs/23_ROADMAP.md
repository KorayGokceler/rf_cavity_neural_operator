# 23 · Yol haritası: uzun vadeli ve genel hedefler

> Durum: **kayıt** — ileride üzerinde çalışılacak hedefler. Kısa notlar mevcut durumu ve ilgili
> dosyaları gösterir; henüz iş planı değildir.

## Genel hedef

RF kaviteleri için CST sınıfı eigenmode çözümüne yakın doğrulukta, çok daha hızlı bir yapay zekâ
modeli: rastgele geometriden modlar, frekanslar ve kaviteler için elzem değerler (Q faktörü, R/Q,
G, E_pk/E_acc, B_pk/E_acc, …).

## 1. Modeli mümkün olan en kısa sürede ölçeklemek

### 1.1 Yeterli bir dataset

- Dataset **farklı geometri sınıflarından** oluşacak.
- Genişletilebilir olmalı: **yeni sınıflar eklenebilmeli** ya da **mevcut sınıflarda geometri sayısı
  artırılabilmeli**.
- Bu nedenle **her sınıf ayrı üretilmeli** (sınıf başına ayrı parçalar / dosyalar, sonradan birleştirilebilir).

*Mevcut durum:* E formülasyonuna geçildi (`claude/3d-e-formulation`): kulplu kaviteler (spoke, HWR,
DTL) dahil her kapalı PEC kavite çalışır; H formülasyonu `--field H` ile duruyor. Aileler
`src/data_gen/cavity_shapes.py` içinde; üretici `--families` ile tek aile
üretebilir, `--start_id` parçaları ve `convert_3d.py` çok dosya birleştirmeyi destekler. Notebook aileleri karışık üretiyor; TRUBA'da her aile ayrı üretilir
(`cluster/truba/README.md`). Kapsama raporu:
`scripts/dataset_stats_3d.py`.

### 1.2 Model boyutu

- Modeli ölçeklemek için **model boyutu testleri** yapılmalı (genişlik, derinlik, baz sayısı; veri
  miktarıyla birlikte).

### 1.3 Her aşamada CST ile kıyas

1. **Birinci kıyas:** datasetteki sınıflarda model ve CST'nin performansı.
2. **İkinci kıyas:** dağılım dışı (out-of-distribution) geometriler.
3. **Üçüncü kıyas:** gerçek bir hızlandırıcı kavitesi (üretilmiş, gerçek bir tasarım).

*Mevcut durum:* ID test (notebook hücre 8), OOD test aileleri (`cavity_shapes.OOD_FAMILIES`, hücre
11), çözünürlük çalışması (`scripts/resolution_study.py`) var; karşılaştırma şu an kendi FE
çözücümüzle (N0). CST karşılaştırması ve gerçek kavite henüz yok.

## 2. RF kaviteleri için elzem değerler

- Q faktörü ve diğer kavite değerleri (R/Q, geometri faktörü G, shunt empedansı, E_pk/E_acc,
  B_pk/E_acc, hücreler arası kuplaj, HOM değerleri, …) hesaplanacak.

*Mevcut durum:* `claude/3d-cavity-qoi` — Q0, G, R/Q, R_sh, T, E_pk/E_acc, B_pk/E_acc tahmin edilen
alan + frekanstan hesaplanıyor (`src/qoi/`, docs/24_CAVITY_QOI.md): pillbox / kutu analitik doğrulaması,
PKL'de FE etiketleri, `scripts/eval_qoi.py` (model vs FE), isteğe bağlı eğitim terimi `--qoi_weight`.
Açık: E formülasyonunda eğri duvarlarda Q0/G birinci mertebe (~%3–6 ağ yanlılığı; kalıntı-akı
düzeltmesi §2.2), HOM / kuplaj değerleri, CST ile kıyas.

## 3. Altyapı

- Model ve dataset **TÜBİTAK TRUBA**'da üretilecek / eğitilecek.

*Mevcut durum:* çok GPU (tek düğüm DDP) var. TRUBA veri üretimi hazır (`cluster/truba/`, branch
`claude/truba-datagen`): aile başına SLURM iş dizileri, aileye özel kimlik blokları, `--resume`, aile
PKL'leri + `merge.sh` karışımları, `status.sh`. Gerekenler: çok düğümlü eğitim, parça bazlı tembel veri yükleme (tek büyük pickle ölçeklenmez), FP64
(Ritz/CG) performansı için GPU tipi seçimi.

## 4. Model düzgün hale geldiğinde

- **Modelin hangi datadan ne öğrendiği** üzerine çalışılacak (sınıf / parametre katkıları,
  genelleme).
- **Hız** üzerine çalışılacak:
  - **eğitim süresi**,
  - **inference süresi**.

*İlgili ölçümler:* inference, FE özdeğer çözümüne göre 3.5× (7k DOF) → 32× (52k DOF) daha hızlı
(CPU, tek geometri); Ritz katmanı inference süresinin ~%15–25'i. Olası hibrit kullanım (modeli
standart çözücüye başlangıç / ön koşullayıcı olarak vermek) not edildi.
