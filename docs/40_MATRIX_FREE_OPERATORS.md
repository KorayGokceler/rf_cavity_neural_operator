# 40 — Matris-serbest HCurl p2 operatörleri (`src/data/matfree.py`)

> 2026-11-12, dal `claude/matfree-operators` (`f9b386e` üzerine), **commit edilmemiş**. Karar: sahip (3 fazlı değişiklik; bu belge **faz 1**: çekirdek modül,
> testler, tablo betiği). **Veri kümesine, modele, eğitime entegre edilmedi** (faz 2–3, §8). Varsayılan hiçbir yol değişmedi.

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

- **Entegrasyon yok** (faz 2–3): `FieldDataset`/`field_collate` hâlâ seyrek matris taşır; model `spmm(C, …)` kullanır; eğitim adımı, `ops_cache` biçimi, `QP` float32 → float64 farkı, DofMask/BndDof maskesi, yığın dolgusu (B > 1; `grams_plain` listeyi destekler ama eğitimle denenmedi) yapılmadı.
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
