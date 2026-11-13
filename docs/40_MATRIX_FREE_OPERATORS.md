# 40 — Matris-serbest HCurl p2 operatörleri (`src/data/matfree.py`)

> 2026-11-12, dal `claude/matfree-operators` (`f9b386e` üzerine). Karar: sahip (3 fazlı değişiklik). **Faz 1** (§1–§9): çekirdek modül, testler, tablo betiği
> (commit `3747b72`). **Faz 2** (§10, commit `1fff0e2`; inceleme düzeltmeleri ve ilk GPU ölçümü §11): izdüşümsüz eğitim/doğrulama yoluna bağlandı (`dataset.operators: matfree`); varsayılan
> (`assembled`) bit düzeyinde değişmedi. GPU'da ölçülmedi.

## 1. Neden

`highorder.field_operators` her geometri için global seyrek K, M, C (+P) kurar: 23k tet'te 285 MB (nopot) – 594 MB (K+M+C+P), 47k tet'te 1,3 GB
(`ops_cache` ile 128–284 MB'a sıkıştırılıyor, docs/38 §9). Hepsi küçük bir kaydın fonksiyonudur; GPU'da asıl maliyet bu matrislerin
yüklenmesi/taşınması. `matfree.py` aynı sayıları eleman toplamlarıyla verir; **hiçbir global matris kurulmaz**.

## 2. Yöntem

1. **Sınıf.** HCurl p2 hiyerarşik tabanı bir elemana yalnız 4 global tepe numarasının *sıra permütasyonu* ile bağlıdır:
   `rk = argsort(argsort(tets[e]))`, sınıf = `itertools.permutations(range(4))` içindeki indeksi (24 sınıf; `tets` = `ngmesh.Elements3D()`).
   L2(2) tabanı da öyle (L2 DOF'ları eleman-yerel).
2. **Tablolar** (`reference_tables`), yalnız NGSolve'dan: sınıf başına referans şekiller ve rotasyonelleri (46 noktalı derece-8 TET kuralı =
   `volume_measure(2)`), L2 tabanı 46 ve 14 noktada (14 = token kuralı, derece 4 = `2·order`), ve ConvertOperator'ın dual fonksiyonelleri
   `T[24, 30, 138]` (çekilmiş alan Jᵀψ'ye, 46 nokta × 3 bileşen). `T`, sınıf başına 14 rastgele kübik deformasyonlu ayrık tet üzerinde en küçük
   kareler ile bulunur ve **başka tohumlu bağımsız bir mesh'te** denetlenir (hata > 1e-10 ise tablo üretilmez; ölçülen 4e-15).
   Rotasyoneller J = I olan *deforme edilmemiş* referans tet'lerde hesaplanır (gürültüsüz); rasyonelleştirme **yapılmadı** (noktadaki
   değerler irrasyonel; katsayıları rasyonelleştirmek ayrı bir iş, gerek yok: hata 1e-14 mertebesinde).
3. **Geometri.** Geometri derecesi ≤ 3 için (deformasyonlu da: generator'ın `smooth_deform` haritası mesh'e derece 3'te uygulanır) x(ξ) eleman başına
   *tam* kübiktir: `x(ξ) = x0 + Σ_m xc[m]·mono_m(ξ)`, 20 monom × 3. `xc`, 46 noktadaki x'ten (eleman ofseti x0 çıkarılmış; mutlak koordinatlar
   fit'te basamak kaybettirir: K hatası 1e-12) bulunur; kalıntı eleman boyutunun 1e-9'unu aşarsa `ValueError` (derece > 3). J[t,q,i,j] = Σ_m xc[t,m,i]·∂_j mono_m(ξ_q).
4. **Eleman matrisleri.** u = J⁻ᵀφ_ref, c = J·curl_ref/detJ (işaretli detJ), `Me = (1/s) Σ_q w_q|detJ| u_l·u_m`, `Ke = s Σ_q w_q|detJ| c_l·c_m`
   (s = `ops['scale']`; `field_operators`ın ξ-ölçeklemesi). K = Σ_e scatter(Ke), M = Σ_e scatter(Me). Kural ağırlıkları > 0 (kare kök kullanılır; kodda denetlenir).
   `apply` = gather → bmm → `index_add`; `gram` = Σ_e V_eᵀ A_e V_e (scatter yok).
5. **ψ → DOF.** `u = C·vec(Pψ)` şu zincirle yeniden üretilir: a = Pψ (token 14 nokta → L2 katsayıları; P = (ΦᵀWΦ)⁻¹ΦᵀW, Φ = L14[sınıf],
   W = |detJ|·w, token noktalarında), ψ(ξ_q46) = L46[sınıf]·a, pr = Jᵀψ (kovaryant geri çekme; |detJ| ağırlığı yok), c_loc = T[sınıf]·vec(pr),
   u = scatter_add(c_loc / n_d[dofs]) (n_d = DOF'u paylaşan eleman sayısı; **duvar DOF'ları dahil** — maskeyi model uygular; kullanılmayan DOF'lar için n_d = 0, hiç indekslenmez).
   Zincir doğrusal olduğundan eleman başına tek bir `Ce[30, 14, 3]` olur (`psi_map`); `psi_to_dofs` = bmm + `index_add`.
6. **Token.** `token_points` = (x0 + xc·mono14 − center)/scale = `ops['qx']`; `token_weights` = |detJ|·w14/scale³ = `ops['qw']`.

Hepsi float64 torch, cihazdan bağımsız, eleman döngüsü yok; `chunk` geçici belleği sınırlar. Kayıt gradyan taşımaz; V ve ψ taşır.

## 3. Kayıt düzeni (`build_record(mesh, order=2, scale=None, center=None)`, numpy)

| anahtar | tür | not |
|---|---|---|
| `dofs` | int32 [nt, 30] | `fes.GetDofNrs`, `fes = HCurl(mesh, order=2, dirichlet="wall")` |
| `cls` | uint8 [nt] | sınıf 0..23 |
| `xc` | float64 [nt, 20, 3] | **float32 saklanmaz** (J'nin basamakları gider) |
| `x0` | float64 [nt, 3] | eleman ofseti |
| `bnd` | bool [n_dof] | duvar DOF'ları |
| `n_dof, n_tets, scale, center, order, format` | skaler | `center`/`scale`: `field_operators` gibi (None → tepe ortalaması, max‖x − center‖); veri kümesi `field_labels.model_normalisation(mesh)` (hacim ağırlıklı merkez) verir, **aynısı verilmelidir** |

≈ 636 B/tet (21,6k tet: 13,7 MB; seyrek K+M+C+P: 594 MB). `bisect` edilmiş ve `highorder.single_level` geçmiş mesh'ler çalışır; `single_level` geçmemiş (hiyerarşi) mesh'te
kullanılmayan DOF'lar (n_d = 0) bulunur ve u'da 0 kalır (1097 tetlik bisect denemesinde 12120 DOF'un 219'u).

## 4. Tablolar, önbellek ve sürüm bağlaşımı

`reference_tables()` ilk kullanımda üretilir (≈ 0,6–1 s) ve `$RFCAV_CACHE` ya da `~/.cache/rfcav/` altına
`matfree_tables_f{FORMAT}_ngsolve_{sürüm}.npz` adıyla atomik yazılır (geçici dosya + `os.replace`). **Anahtar** = `matfree.FORMAT` + `ngsolve.__version__`:
farklı NGSolve (hiyerarşik taban, ConvertOperator ya da kural değişebilir) başka dosya, yeniden üretim demektir; bozuk dosya güvenilmez, yeniden üretilir.
İkili dosya commit edilmez. `scripts/make_matfree_tables.py [--rebuild] [--verify]`: üretir/doğrular (24 sınıflı sentetik mesh ve pillbox'ta `field_operators` ile karşılaştırır, `PASS`).
Yeni bir NGSolve sürümüne geçişte `--rebuild --verify` ve `tests/test_matfree.py` çalıştırılmalıdır.

## 5. Genel API (özet)

`reference_tables`, `build_record`, `to_torch(rec, device)`, `jacobians(rec, "mass"|"token")` [nt, nq, 3, 3] (fiziksel birim), `element_matrices(rec, chunk)` → `(Me, Ke)` [nt, 30, 30]
(1/s, s dahil), `apply(Ae, dofs, V)`, `gram(Ae, dofs, V, T=None)` (T verilirse [V|T] blok Gram'ı), `grams_plain(rec, V, T, mats=None)` (= `hcurl_grams_plain`in tek geometri çıktısı:
`G_M, G_A [m+n, m+n]`, `MT [Nd, n]`; kayıt listesi + [B, Nd, ·] girdi ile yığın), `psi_map(rec)` → `Ce`, `psi_to_dofs(rec, psi [nt,14,m,3], Ce=None)` (modelin `field_basis`i ile aynı
indeks sırası; DofMask/BndDof maskesinden **önce**), `token_points`, `token_weights`, `check_against_ngsolve(mesh)`.
Not: modelin yolunda `QP` float32'dir (`field_item`); matris-serbest yol P'yi float64 yeniden kurar (saklı P'den ≤ 1e-11 farkla; spike'ta 2,4e-13), yani biraz daha doğrudur.

## 6. Doğrulama (ölçülen; tek koşu, CPU, deterministik girdiler)

Karşılaştırma: `field_operators(mesh, 2, potential=False)`; bağıl hata = max|fark|/max|referans|. Test eşikleri: işlemler/Gram ≤ 1e-11, ψ→DOF ≤ 1e-10, gradyan ≤ 1e-9.

| mesh | tet | sınıf | K·V | M·V | VᵀKV | VᵀMV | ψ→DOF | qx | qw | `grams_plain` | ∂/∂V | ∂/∂ψ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pillbox (eğri 3) | 130 | 2 | 3e-15 | 4e-15 | 2e-15 | 2e-15 | 5e-15 | 4e-15 | 5e-15 | 4e-15 | 3e-15 | 8e-15 |
| + generator deformasyonu | 130 | 2 | 3e-15 | 3e-15 | 2e-15 | 2e-15 | 6e-15 | 4e-15 | 4e-15 | 4e-15 | 3e-15 | 7e-15 |
| bisect + `single_level` | 1867 | 24 | 4e-15 | 2e-15 | 7e-16 | 7e-16 | 7e-15 | 2e-15 | 7e-15 | 2e-15 | 3e-15 | 5e-15 |
| bisect, hiyerarşi (kullanılmayan DOF) | 1867 | 24 | 3e-15 | 3e-15 | 7e-16 | 6e-16 | 6e-15 | 3e-15 | 6e-15 | 3e-15 | 3e-15 | 7e-15 |
| sentetik 48 ayrık kübik tet | 48 | 24 | 5e-15 | 4e-15 | 2e-15 | 3e-15 | 4e-15 | 6e-15 | 5e-15 | 4e-15 | 6e-15 | 2e-14 |
| gerçek: pillbox (adapt_meas) | 10434 | 2 | 3e-15 | 3e-15 | 1e-15 | 2e-15 | 1e-14 | 1e-15 | 8e-15 | 3e-15 | – | – |
| gerçek: spoke | 8431 | 2 | 6e-15 | 3e-15 | 2e-15 | 2e-15 | 1e-14 | 1e-15 | 5e-15 | 3e-15 | – | – |
| gerçek: hwr (keskin, deforme) | 21596 | 2 | 4e-14 | 2e-15 | 9e-16 | 1e-15 | 3e-14 | 2e-15 | 7e-15 | 4e-15 | – | – |
| gerçek: ridged_box | 47466 | 24 | 3e-14 | 4e-15 | 3e-15 | 2e-15 | 2e-14 | 1e-15 | 5e-15 | 3e-15 | – | – |

Ek testler (`tests/test_matfree.py`, 32 test, ≈ 8 s): chunk'lı = chunk'sız (eleman matrisleri bit düzeyinde), kayıt boyutu ≤ 700 B/tet, derece-4 geometrinin reddi (`ValueError`), derece 2'nin kabulü,
tablo önbelleği (gidiş-dönüş, sürüm/FORMAT anahtarı, bozuk dosya), kural ağırlıkları > 0. Mevcut `tests/test_field_model.py` ve `tests/test_projection_free.py` ile birlikte 67 test geçti.

## 7. Süre ve bellek (21,6k tet, `hwr_3670021_adaptive.h5`, CPU, 4 torch iş parçacığı)

Donanım/yük: 4 çekirdek, 15 GB, GPU yok; `torch.get_num_threads()` = 4, NGSolve iş parçacığı = 1 (`ngsolve.GetNumThreads()`); ölçüm sırasında başka iş başlatılmadı, makine yük ortalaması 1–2.
Tekrarlar arası oynama %50'ye varabiliyor (örn. `element_matrices` 1,15–1,8 s); minimum-of-3 verilir. **Bunlar sağlık sayıları; hedef GPU.**

| işlem | seyrek yol | matris-serbest |
|---|---|---|
| kayıt / operatör kurulumu | `field_operators(potential=False)` 4,07 s | `build_record` 0,94 s (+ tablolar ilk kullanımda ≈ 1 s) |
| bellek | K+M+C+P 594 MB | kayıt 13,7 MB; `Me+Ke` 311 MB (isteğe bağlı önbellek), `Ce` 218 MB; geçici ≈ 0,12 MB/eleman·chunk |
| `element_matrices` (chunk 512/1024/2048) | – | 1,5 / 1,8 / 1,8 s (başka koşuda 1,15 / – / 1,48 s) |
| `psi_map` | – | 0,5–0,8 s |
| `grams_plain` ileri, m = 30, n = 4 | 146 ms | 214 ms (Me, Ke önbellekte); 1850 ms (eleman matrisleri dahil) |
| `grams_plain` ileri + geri | 393 ms | 656 ms (önbellekte); 2339 ms (dahil) |
| `psi_to_dofs` ileri, m = 30 | 217 ms | 222 ms (`Ce` önbellekte); 855 ms (anında) |
| `psi_to_dofs` ileri + geri | 444 ms | 670 ms (önbellekte); 1534 ms (anında) |

Yorum: CPU'da seyrek yol önbelleğe alınmış operatörle hâlâ hızlı; kazanç bellek/aktarım (kayıt 13,7 MB'a karşı yüzlerce MB) ve GPU'da `Me, Ke`nin adım içinde kurulabilmesidir. GPU'da ölçülmedi.

## 8. Yapılmayanlar ve riskler

- ~~Entegrasyon yok~~ — faz 2 ile yapıldı (§10): veri kümesi, collate, önbellek, model, kayıp; `QP` float32 → float64 farkı ölçüldü (§10.3), DofMask/BndDof maskesi ve B > 1 test edildi.
- **Kp / kernel projeksiyonu** (G, Kp, CG) kapsam dışı: projeksiyonlu yol hâlâ seyrek G, Kp ister (`training.projection: false` yolu bu modülle tam kapanır, docs/38).
- Yalnız HCurl **p2**, geometri derecesi **≤ 3**; farklı kural (`quad_order`) ya da derece için tablolar yeniden türetilmeli (`NotImplementedError` / `ValueError` ile reddedilir).
- Sınıf varsayımı (`Elements3D` tepe sırası = NGSolve eleman tepe sırası; hiyerarşi yalnız sıra permütasyonuna bağlı) 4 gerçek mesh'te (biri 24 sınıflı, 47k tet), bisect/`single_level` ve sentetik meshlerde doğrulandı; yeni NGSolve sürümünde `--verify` zorunlu. Gerçek meshlerin çoğunda yalnız 2 sınıf görünür (tepe numaralandırması neredeyse sıralı); 24 sınıfın tamamı ridged_box ve test meshlerinde sınanmıştır.
- `element_matrices` CPU'da bellek-bağlı (gather); sınıfa göre gruplayıp paylaşılan tablolarla GEMM yapmak CPU'yu hızlandırabilir (denenmedi). Eleman matrislerini chunk başına kurup hemen indirgeyen "kaynaşık" `apply`/`gram` (Me, Ke saklamadan) yapılmadı.
- GPU'da `index_add_` float64 atomikleri sıra-belirsizdir (bit düzeyinde tekrarlanabilirlik yok; hata düzeyi 1e-15).

## 9. Yeniden üretme

```bash
python scripts/make_matfree_tables.py --rebuild --verify        # tablolar + field_operators karşılaştırması (PASS)
python -m pytest -q tests/test_matfree.py tests/test_field_model.py tests/test_projection_free.py
```

## 10. Faz 2: izdüşümsüz eğitim/doğrulama yoluna entegrasyon

> 2026-11-12, aynı dal, commit `1fff0e2`. Ölçümler CPU'da (4 çekirdek, 15 GB, GPU yok; `torch.get_num_threads()` 4 ya da işçi koşullarında 1, `ngsolve.GetNumThreads()` 1; ölçüm sırasında başka iş yok, yük ort. ~1). **GPU sayıları sahipten gelecek.**

### 10.1 Seçenek ve ne nerede çalışır

`dataset.operators: assembled | matfree` (varsayılan `assembled`: bugünkü davranış, bit düzeyinde aynı, §10.4). `matfree`:

| | train / val | test |
|---|---|---|
| öğe | `field_item_matfree`: token dizileri (`X, Input_funcs, Area, QX`), etiketler, `BndDof`, `MF` = kayıt (dofs int32, cls uint8, xc/x0 float64, bnd, ölçek/merkez); **M, K, C, QP, Ct, G, Kp yok** | `assembled` (G, Kp, CG, Ritz; değişmedi) |
| collate | `field_collate_mf`: yalnız yoğun tensörler; `batch["MF"]` = örnek başına kayıt sözlüklerinin **listesi** (dolgu yok) | `field_collate` |
| V (taban) | `EigenspaceOperatorField._matfree_dofs`: örnek başına `matfree.psi_to_dofs` (float64), sonra aynı `DofMask & ~BndDof` maskesi ve `.float()` | `spmm(C, …)` |
| Gram'lar | `matfree.grams_plain(...)` (aynı blok düzeni; sonrası `_normalise_columns`, `span_residual`, enerji-Gram ortho, tanı metrikleri, `grad_frac_lb` aynı kod) | `hcurl_grams_plain` |

`matfree` yalnız `training.projection: false` ile (aksi `ValueError`, `train.py`); `FieldDataset(operators="matfree", build_potential=True)` da `ValueError`. Çıkarım yolları (`src/service/model.py`, `scripts/predict_geometry.py`, değerlendirme betikleri) değişmedi. `forward()` (Ritz) matfree batch'i reddeder.

**Öğe** (`FieldDataset.load_matfree`): `read_field_group` → `model_normalisation(mesh)` (merkez, ölçek: assembled yolla **aynı**) → `matfree.build_record(mesh, order, scale, center)` → token dizileri `highorder.token_inputs` ile. `field_operators` token özelliklerini artık `highorder.token_features` ile hesaplar ve `token_inputs` **aynı fonksiyonu** çağırır (formül kopyası yok); `field_operators`'ın çıktısı HEAD ile bit düzeyinde aynı (§10.4). `field_operators` çağrılmaz (testle korunur). Etiket doğrulaması (yᵀKy/yᵀMy ≈ `rq_model·s²`): assembled yol K, M ile her kurulumda yapar; matfree yol eleman matrisleriyle yapar (`check_labels_matfree`), ama bu eleman matrislerinin maliyetidir (21k tet: ~1,5 s 4 iş parçacığında, ~6 s tek iş parçacıklı işçide; aşağıdaki tabloda hwr, 1 iş parçacığı: 2,9 → 8,9 s) — bu yüzden `dataset.verify_labels` (null = açık: öğe her kurulduğunda, geometri ve süreç başına **bir kez**; önbellek isabetinde yok; true/false zorlar; §11.1).

**Önbellek**: `{tag}__{key}.mf1.pkl` (`ops_cache.MF_SUFFIX`; `.ops2.pkl` / `.nopot.ops2.pkl` ile **karışmaz**: ayrı ad, `kind: "matfree"` + `mf_format` + `matfree.FORMAT` + NGSolve sürümü alanı; başka biçim/sürüm/bozuk dosya uyarıyla yeniden kurulur ve üzerine yazılır). Ham dizilerin tek pickle'ı (zlib yok, indis farkı yok; atomik yazma `ops_cache.write`): kayıt + `qfeat` (float32 [Nt·14, 9]; `X` ve `QX` bunun ilk 3 sütunundan, bit düzeyinde aynı) + `qw` + etiketler (`u_model` float32 tam ise float32). Önbellek isabetinde mesh, uzay ve operatör kurulmaz (testle korunur); yalnız sürüm karşılaştırması için `import ngsolve` yapılır, yani NGSolve kurulu olmalıdır. `dataset.cache_dir` anlamı değişmedi.

**Adım içi** (`matfree.StepOperators`): `Me, Ke` (`element_matrices`) ve `Ce` (`psi_map`) aygıtta, `no_grad` altında, **adım başına bir kez** kurulur; V ve Gram'lar paylaşır. Kalıcı bellek ≈ 25 KB / (tet·örnek) (Me+Ke 14,4 KB, Ce 10 KB; 21,6k tet: 530 MB, 47k tet ve B = 2: ~2,4 GB) + geçici ≈ 0,12 MB × `training.matfree_chunk` (varsayılan 2048). Otograd grafiği Ce'yi (∂V/∂ψ) ve Ke/Me'yi (∂Gram/∂V) geri geçişe kadar tutar, sonra hepsi serbest kalır; `StepOperators.free()` kayıt/eleman matrisi referanslarını kayıp fonksiyonunun sonunda bırakır. Gram'ın geri geçişi için `matfree._GramFn` (özel autograd): saklanan yalnız W, Ae, dofs; `dW = A W (ḡ + ḡᵀ)` tek parçalı `apply` (eleman toplama grafiği Gram başına 2 × Nt·30·k çift = ~0,3 GB @ 21k tet, k = 30; iki Gram için ~0,6 GB: tutulmaz). Trunk `grad_checkpoint` etkilenmez (V hesabı `embed` dışında).

**Yükleyici** (`train.py`): matfree split'lerde `field_collate_mf` ve `pin_memory = torch.cuda.is_available()` (sparse tensör yok, doğrudan pinlenir); `gpu_prefetch` ile **çakışmaz**: `gpu_prefetch=true` yalnız test yükleyicisinin `SparseParts` collate'ini ve `transfer_batch_to_device`'ın yan-akım kopyasını etkiler; `join_sparse` liste/sözlük gezdiği için matfree batch de yan akımdan kopyalanabilir (CPU'da test edildi; CUDA kolları test edilemedi). `gpu_prefetch` matfree için gerekmez.

### 10.2 Boyutlar ve süreler (CPU; `hwr_3670021_adaptive.h5` 21 596 tet, `comp_adaptive.h5` 23 155 tet, tek örnek)

`FieldDataset(...)[0]` süreleri (min. ölçüm yok: tek koşu; ilk satırlar 1 iş parçacıklı işçi koşulu, son 4 iş parçacığı; `cache_compress` varsayılan true; assembled önbellek = `.nopot.ops2.pkl`):

| | hwr 1 ip. | comp 1 ip. | hwr 4 ip. |
|---|---|---|---|
| assembled, ilk epoch (kurulum, önbelleksiz) | 10,8 s | 12,2 s | 6,9 s |
| assembled, önbellek yazma dahil | 18,4 s | 19,4 s | 13,8 s |
| assembled, önbellekten okuma (zlib) | 2,0–2,6 s | 2,1–2,7 s | 2,0–2,4 s |
| assembled, önbellekten okuma (zlib yok; 308–328 MB dosya) | 1,2–1,6 s | 1,2–1,6 s | 1,1–1,6 s |
| **matfree, ilk epoch (kurulum, önbelleksiz)** | **2,9 s** | **3,4 s** | **3,3 s** |
| matfree, + `verify_labels` | 8,9 s | 9,1 s | 5,0 s |
| matfree, önbellek yazma dahil | 3,2 s | 3,4 s | 3,2 s |
| **matfree, önbellekten okuma** | **0,019–0,025 s** | **0,021–0,030 s** | **0,014–0,021 s** |

Önbellek dosyası: assembled 121 / 128 MB (zlib) ya da 309 / 328 MB; **matfree 35,1 / 37,1 MB** (≈ 1,6 KB/tet: kayıt 13,7 MB + qfeat 10,9 + qw 1,2 + etiketler ~9; hedef ≲ 30 MB'ı biraz aşıyor — etiketler (231k DOF × 10 mod float32) ve token özellikleri baskın). Dosya okuma (`pickle.load`) 5–8 ms, `unpack_mf` 2–3 ms; geri kalanı `field_item_matfree` + `from_numpy` kopyaları.
"~1 s" beklentisi yalnız `build_record` idi (1,0–1,1 s); ilk epoch öğesi: `read_field_group` 1,6 s (assembled'da da var) + normalizasyon 0,14 + `build_record` 1,1 + `token_inputs` 0,52 (NGSolve'da 14 noktalı kuralı yeniden değerlendirir; `build_record` 46 noktalıyı) = ~3,3 s; yani ilk epoch assembled'ın 0,3–0,5'i, sonraki epoch'lar ~60–100× hızlı (0,02 s'ye karşı 1,2–2,5 s).

Batch (B = 1): **43,1 MB** (hwr; matfree) / 45,7 MB (comp) — assembled **1119 MB** / 1191 MB (26×). `scripts/profile_loader.py` (hwr, 2 işçi, B = 1, `--step_time 0.3`, önbellek sıcak): matfree işçi süresi 0,05 s/örnek (assembled 3,0 s), ana süreç CPU 10 ms/batch (95), `/dev/shm` 43 MB/batch (1119). (Tek örnekli veri kümesinde her epoch yükleyiciyi yeniden başlatır: `wait` ~100 ms bunun etkisidir.)

Bir eğitim adımı (`scripts/profile_field_step.py --grad_checkpoint --threads 4 --reps 2`, config boyutları 0,854 M parametre, K = 6 mod, CPU; bellek = süreç tepe RSS'i, veri hazırlığı dahil, her mod ayrı süreçte):

| | tet | assembled `--no_projection` | `--matfree` |
|---|---|---|---|
| ileri+kayıp | 21,6k | 5,6–5,7 s | 8,7–9,4 s (eleman matrisleri + ψ haritaları: 3,0 s) |
| geri | 21,6k | 13,9–14,1 s | 13,8–14,3 s |
| tepe RSS | 21,6k | 5,51 GB | **4,46 GB** |
| ileri+kayıp | 23,2k | 5,8–6,4 s | 8,7–9,4 s (2,8–3,1 s) |
| geri | 23,2k | 14,5–16,5 s | 14,4–15,5 s |
| tepe RSS | 23,2k | 5,92 GB | **4,78 GB** |

CPU'da adım matfree ile ≈ %10–19 yavaş (bağımsız kontrol: ileri+geri %19, yalnız ileri+kayıp %55; §11.2) (her adım eleman matrislerini ve ψ haritasını yeniden kurar; ~3 s, CPU'da bellek-bağlı gather); bellek ~1–1,1 GB az. Hedef GPU'da bu kurulumlar bellek-bağlı küçük çekirdekler; **GPU'da ölçülmedi**. Assembled sütununun tepe RSS'i operatör kurulumunu ve collate'i içerir (matfree'de yok): bellek farkı bunu da yansıtır, GPU belleği farklı olacaktır.

### 10.3 Eşdeğerlik: ölçülen farklar

Aynı ağırlık (aynı tohum), aynı geometriler, B = 2 farklı boyutlu örnek (pillbox 57 tet + kutu 46 tet; dolgulu) ve B = 1. Karşılaştırılan: V (`forward_basis`), kayıp, **her** loglanan bileşen (12: loss, span, ortho, span_rel_l2, enerji, 4 mod, mass_diag_ratio, grad_frac_lb, grad_frac_lb_max), tüm parametre gradyanları; train (otograd) ve val (`no_grad`). Beklenen kaynak: assembled yolda P·ψ float32 ve float32 saklanan `QP` ile, matfree'de float64 ve yeniden kurulan P; çıkış `.float()` ile float32 olduğundan fark bir float32 yuvarlamasıdır (ε = 6e-8).

| geometri | V (max/max) | kayıp (bağıl) | loglanan bileşenler (en kötü bağıl) | gradyan (norm bağıl) |
|---|---|---|---|---|
| fixture B=2 (0,2), (3,1) | 7,5e-8 | 0 (float32 baskısında) | 9,2e-8 | 1,4–1,5e-7 |
| fixture B=1 | 1,1e-7 | 0 | 0 | 1,8e-7 |
| pillbox 10,4k + spoke 8,4k, B=2 (CPU) | 3,0e-7 | 0 | 0 | 9,4e-8 |

Test eşikleri (`tests/test_matfree_training.py`): V, kayıp, bileşenler, gradyan hepsi **1e-6** (ölçülenin ≥ 3×, en çok 10× üstü; bir float32 ulp'nin birkaç katı). Chunk 3 / 50 / tüm = 2048 ile 1e-6 içinde aynı (pillbox: bit düzeyinde aynı). `matfree.gram`'ın özel geri geçişi eski otograd yolu ile 1e-12 içinde aynı.

### 10.4 Varsayılan yolun değişmediğinin kanıtı

`highorder.field_operators` token özelliklerini artık `token_features` ile hesaplar; ve `field_collate` kuyruğu `_collate_labels` yardımcısına taşındı (saf taşıma). Kanıt: HEAD (`3747b72`) kaynak ağacı ile çalışma ağacı **ayrı süreçlerde** aynı pillbox örneğinde (10,4k tet, `build_potential` False ve True): `field_operators` çıktısı (qx, qw, qfeat, P, K, M, C, G, Kp: değer + indis özetleri), `field_collate` batch'inin tüm tensörleri (sparse: değer + indis), izdüşümsüz ve izdüşümlü kayıp, **tüm loglanan metrikler** ve tüm parametre gradyanları: **69 özetin 69'u bit düzeyinde aynı** (betik `scratchpad/matfree_p2/bitid.py`, depoda değil). Testte: `test_assembled_features_are_the_former_formula` (eski satırlar elle yeniden yazılmış, `np.array_equal`), `test_matfree_items_equal_the_assembled_items_apart_from_the_operators` (token/özellik/ağırlık/etiket bit düzeyinde).

### 10.5 Testler (`tests/test_matfree_training.py`, 24 test, ≈ 20 s CPU)

Öğe eşitliği (anahtar kümesi farkı tam `{QP, M, K, C, Ct}` / `{MF}`; sparse tensör yok; ≤ 700 B/tet), `field_operators` çağrılmaması, kayıt/token/etiket kontrolü (ve `verify_labels`'ın uyumsuz `rq_model`'i yakalaması), collate dolgusu, adım eşdeğerliği (train/val, B = 1, 2), chunk bağımsızlığı, `StepOperators` paylaşımı + serbest bırakma, `_GramFn` geri geçişi, önbellek (kendi adı, diğer önbellekle karışmama, ikinci epoch'ta NGSolve yok, bozuk/eski biçim/başka NGSolve → uyarı + yeniden kurma, atomik yazma), seçenek korumaları (`ValueError`'lar, config varsayılanları), yükleyici seçenekleri (`gpu_prefetch` ile ve onsuz, CUDA varmış gibi), 2 işçili DataLoader + pinleme + iki transfer yolu, Lightning `fit` (iki epoch, matfree ≈ assembled), `train.py --fast_dev_run` (train/val matfree, test assembled + Kp CG), 2 epoch + `--resume last.ckpt`, projeksiyonlu config ile reddetme, varsayılanın assembled kalması.

### 10.6 Hâlâ assembled / yapılmayanlar / belirsizlikler

- **Test bölümü** (Kp, G, CG, Ritz) ve tüm çıkarım yolları assembled; `forward()` matfree batch'i reddeder. Test split'inin önbelleği `.ops2.pkl` (284 MB @ 23k tet) kalır.
- Etiket doğrulaması önbellek isabetinde yapılmaz (assembled yol gibi); her kurulumda yapılır (§11.1).
- Adım başına eleman matrisi/ψ haritası yeniden kurulumu CPU'da ~3 s (adımın ~%12'si); GPU'da ölçülmedi. `Me, Ke`'yi epoch'lar arası saklamak bellek maliyetli (311 MB/örnek) ve yapılmadı; chunk'a bağlı "kaynaşık" `apply`/`gram` da yapılmadı (§8).
- `index_add_` float64 atomikleri (GPU) sıra-belirsiz: matfree adımı GPU'da bit düzeyinde tekrarlanabilir değil (1e-15 düzeyi).
- CUDA kolları (pinleme, yan akım, `record_stream`, `non_blocking` kopya) bu makinede çalıştırılamadı; `to_torch` aygıt taşımalı kayıt (CPU testi var) CUDA'da denenmedi.
- Bellek: B = 2 ve 47k tet'te kalıcı eleman matrisleri ~2,4 GB + ağ aktivasyonları; `max_tets` / `training.matfree_chunk` ile sınırlanır; V100 16 GB için ölçülmedi.
- `profile_loader.py`'de `verify_labels` config'ten geçer; `colab_train_noproj.ipynb` varsayılanı `OPERATORS = 'assembled'` (GPU ölçümünden sonra çevrilmeli); BRANCH ayarı bu değişikliği içeren dala işaret etmelidir.

### 10.7 Yeniden üretme

```bash
python -m pytest -q tests/test_matfree.py tests/test_matfree_training.py tests/test_field_labels.py tests/test_projection_free.py tests/test_throughput.py
python train.py --config configs/field_3d_noproj.yaml --override "dataset.data_path=data/field/*.h5" dataset.operators=matfree dataset.cache_dir=/scratch/ops_cache training.matfree_chunk=2048
python scripts/profile_field_step.py <h5> --sample 0 --matfree --grad_checkpoint --device cuda   # yan yana: --no_projection (ayrı süreçte)
python scripts/profile_loader.py --config configs/field_3d_noproj.yaml --override dataset.data_path=<h5> dataset.operators=matfree dataset.cache_dir=<dir> --workers 4 8 --step_time 0.3 --device cuda
```

## 11. Faz 3: bağımsız inceleme, düzeltmeler ve ilk GPU ölçümü

> 2026-11-13, aynı dal. **Sonuç: sayısal olarak doğru, ama ölçülen GPU'da adım ~2 kat YAVAŞ; bu hâliyle eğitim hızlandırması değildir.** Varsayılan `assembled` kalır.

### 11.1 Fizik incelemesi (taze bağlam) ve düzeltmeler

Yanlış sonuç ya da bozulan AGENTS.md değişmezi bulunmadı (Piola eşlemeleri, işaretli det, 1/s ve s ölçekleri, 24 sınıf varsayımı yerel inceltilmiş ağda, ters yönelimli eleman, `_GramFn` geri formülü simetrik olmayan üst gradyanla 2e-16, varsayılan yolun değişmediği). Düzeltilenler:

- **Etiket doğrulaması**: `cache_dir` yokken matfree yol etiketleri hiç denetlemiyordu (assembled yol her kurulumda denetler). `verify_labels: null` artık **açık** demektir: her kurulumda, geometri ve süreç başına bir kez; yalnız önbellek isabetinde atlanır. Test: varsayılanla, önbelleksiz, bozuk `rq_model` → `ValueError`.
- **Tablo önbelleği**: kesik `.npz` (`zipfile.BadZipFile`, `EOFError`) artık yeniden kurulur; yazılamayan önbellek dizini (`OSError`) uyarı verir ve tablolar bellekte tutulur (eğitim düşmez).
- Belgedeki "önbellek isabeti NGSolve'a dokunmaz" ifadesi düzeltildi (§10.1).

Açık bırakılanlar (küçük): `RULE_DEGREE["mass"] = 8` sabit, `highorder.volume_measure(2)`'den türetilmiyor (karşılaştırma testi bir sapmayı yakalar); `matfree_chunk=0` 2048'e çevrilir.

### 11.2 Bağımsız sayısal kontrol (CPU, kendi betikleriyle)

| kontrol | sonuç |
|---|---|
| tam takım / ruff / vulture | 379 geçti / temiz / temiz |
| operatörler, 13 aileden 14 gerçek örnek (8,4k–73,8k tet; ≥ 9'u deforme; 2 ya da 24 sınıf) | en kötü bağıl fark 6e-14 (`field_operators`'a karşı); negatif detJ yok |
| etiketlerin Rayleigh oranı (matfree K, M) / saklı frekans | 1e-13 … 1,2e-11; ‖GᵀMT‖/‖MT‖ 4,5e-8 … 4,9e-8 (float32 etiket gürültüsü, assembled M ile aynı) |
| eğitim adımı, assembled ↔ matfree, 5 durum (B = 1, 2; 24 sınıf dahil) | V ≤ 3,0e-7; kayıp bit-eşit; gradyan norm-bağıl ≤ 1,6e-7, maks-bağıl ≤ 2,1e-7 |
| `train.py` 3 ve 12 epoch, 75 TensorBoard etiketi | ≤ 1,3e-7 fark, kayma yok; test aşaması (izdüşümlü) aynı; önbellek yaz/oku ve `--resume` temiz |
| varsayılan yol, origin/main ↔ dal | 419 özetin 419'u bit-eşit |
| CPU maliyeti (hwr 21,6k tet) | öğe okuma 2,2 s → 0,014 s; batch 1119 → 43 MB; adım (ileri+geri) 17,3 → 20,6 s (+%19) |

### 11.3 İlk GPU ölçümü (sahip, Colab; GPU modeli kaydedilmedi, adım süreleri G4'ün ~6 katı: T4/L4 sınıfı)

`profile_field_step.py --device cuda --threads 4 --reps 2 --grad_checkpoint`, küçük model (0,854 M), ikinci tekrar:

| | 23 711 tet assembled | matfree | 51 038 tet assembled | matfree |
|---|---|---|---|---|
| ileri + kayıp | 0,53 s | 2,11 s | 1,17 s | 4,57 s |
| ↳ eleman matrisleri + ψ haritaları | – | 1,44 s | – | 3,09 s |
| geri | 1,14 s | 1,24 s | 2,55 s | 2,91 s |
| **adım** | **1,67 s** | **3,35 s** | **3,72 s** | **7,48 s** |
| GPU'ya taşıma | 0,33 s | 0,01 s | 0,68 s | 0,03 s |
| öğe + collate (işçi) | 1,68 s | 0,03 s | 4,44 s | 0,08 s |
| tepe GPU belleği | 4,74 GB | 3,65 GB | 10,18 GB | 7,82 GB |

Veri tarafı beklendiği gibi (taşıma ve collate ~0, GPU belleği −%23), ama adım başına yeniden kurulum (61 µs/tet) kazancı fazlasıyla geri alıyor; kurulum hariç adım 1,91 s (assembled'dan ~%15 yavaş, CPU'daki oranla aynı). Kurulumun aritmetiği küçüktür (birkaç GFLOP); süre uygulamadan geliyor olmalı (doğrulanmadı). Adaylar: afin (eğri olmayan) elemanlarda 46 noktalı kuadratür yerine sınıf başına 6 referans matrisin bileşimi; float64 ara dizilerin ([nt, 46, 30, 3]) küçültülmesi. Eleman matrislerini GPU'da saklamak üretim setine sığmaz (175–340 MB/örnek). G4'te ve çok epoch'lu eğitimde ölçülmedi.

