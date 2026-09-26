# 20 — 3B Maxwell Modeli: EigenspaceOperator3D (H alanı, Whitney N0)

> docs/18 §3'teki tasarımın uygulaması. Veri tarafı: docs/19 (`convert_3d.py` PKL sözleşmesi).
> Kod: `src/models/eigenspace_operator_3d.py`, `src/models/hcurl.py`, `src/data/dataset_3d.py`,
> `GNOTLightning(model_type='eigenspace3d')`, `configs/eigenspace_3d.yaml`, `scripts/eval_3d.py`,
> testler: `tests/test_eigenspace_3d.py` (+ sentetik veri `tests/maxwell3d_synth.py`).

## 1. Mimari

| Adım | Ne yapar |
|---|---|
| Gövde | 2D `EigenspaceOperator.embed` aynen kullanılır: RFF(xyz) ‖ F=9 düğüm özelliği → MLP → `n_layers` × kütle-farkında lineer attention. Ağırlıklar düğüm hacmidir (`node_volume` → `batch['Area']`). |
| Kenar tabanı | Kenar $e=(a\to b)$, düşük→yüksek indeks. Girdi simetriktir: $z_e=\tfrac12(h_a+h_b)\,\Vert\,\mathrm{RFF}(x_{mid})$. MLP orta noktada $m$ **vektör** alan verir, $\psi_j(x_{mid})\in\mathbb R^3$. DOF $V_{ej}=\psi_j(x_{mid})\cdot t_e$, $t_e=x_b-x_a$ olur; bu, $\int_e\psi\cdot t$'nin orta nokta kuralıdır. |
| Yön | Kenar ters çevrilince yalnız $t_e$ işaret değiştirir, dolayısıyla $V_e\to-V_e$. Bu yapısal olarak kesindir (test edildi). Sınır kapısı yoktur ($\mathbf H$'de iki PEC koşulu da doğaldır); dolgu kenarları 0 verir. |
| Ritz | Projekteli span üzerinde yapılır (§2); frekans $f=c\sqrt\lambda/(2\pi\,\mathrm{scale})$ (`_physics_freq_z`). |

Çıktılar: `basis` [B,Ne,m], `field` [B,Ne,K] (projekteli, birim $M$-normlu, işaret keyfi), `eigenvalues`, `freq` [B,K], `M_mat`=$M_{div}$, `L_mat`=$A_V$ [B,m,m]. Kayıplar için ayrıca `M_div`, `A_V`, `M_V`=$V^\top MV$ (float64) ve `Z`=$K_p^{-1}G^\top MV$ döner; böylece ikinci bir çözüm gerekmez.

## 2. Çekirdeğin çıkarılması

- $B=G^\top MV$, $Z=K_p^{-1}B$, $PV=V-GZ$ tanımlarıyla $A_V=V^\top KV$ olur ($KG=0$), $M_{div}=V^\top MV-B^\top Z$ ise Schur tümleyenidir.
- **$K_p$'nin sabit çekirdeği:** Sabitleme ve ridge kullanılmaz. $1^\top B=(G1)^\top MV=0$ olduğundan sistem tutarlıdır ve CG range'de yakınsar. Çözümün taşıdığı sabiti hem $GZ$ hem $B^\top Z$ yok eder. Tekil Neumann sisteminin koşul sayısı, sabitlenmiş sisteminkinden iyidir: $\lambda_{min}$ Fiedler değeridir, nokta kısıtının verdiği $O(h\lambda_2)$ değil. Sayısal olarak $Z$ ve sağ taraf her örnekte **ortalamasız** tutulur. Aksi halde Jacobi-PCG iterasyonları sabit yönünde kayar ve $1^\top G^\top(\cdot)\approx0$ yuvarlamasını büyütür. Bu kaymanın $M_{div}$'in gradyan bloğunda $10^{-6}$'lık sahte özdeğerler ürettiği ölçüldü.
- **Çözücü:** Batched Jacobi-PCG, float64. Her (örnek, kolon) çiftinin kendi adım boyu vardır. `KpSolve` bir `autograd.Function`'dır: $K_p$ sabit ve simetrik olduğundan geri geçiş, aynı operatörle yapılan bir çözümdür ($\bar B=K_p^{-1}\bar Z$). Tolerans `kp_tol=1e-8`. CPU'da ve GPU'da aynı kodla çalışır.
- **Rank-revealing Ritz (`projected_eigh`):** Saf gradyan yönlerinde $M_{div}$ ve $A_V$ birlikte sıfırlanır ve 0/0 çıkar. Ridge bu yönlere spektrumun ortasında keyfi bir $\theta$ verir (2D `_ritz`'te sütunların ortalama Rayleigh bölümü). Üstelik kayıplar gradyan içeriğini hiç görmez, dolayısıyla eğitim bu yönlerden uzak tutulmaz. Bunun yerine şu adımlar izlenir:
  1. Projeksiyonsuz kütleyle beyazlatılır: $DM_VD=CC^\top$.
  2. $\mu$ = eig$(C^{-1}DM_{div}DC^{-\top})\in[0,1]$ hesaplanır; bu, her yönün diverjanssız kütle payıdır.
  3. $\mu<$ `drop_tol` olan yönler atılır, yani $\theta=\infty$ alınır.
  4. Kalanlarla eig$(W^\top A_VW)$ çözülür.

  Kalan her $\theta$ sıfırdan farklı, diverjanssız bir alanın Rayleigh bölümüdür, dolayısıyla $\theta_k\ge\lambda_{h,k}$ olur. İki eigh de genişletilmiş (broadened) geri geçişle çalışır.

## 3. Kayıplar (2D `_eigenspace_terms` ile aynı terimler)

`hcurl_grams` $[PV\,|\,T]$ matrisinin Gram'larını kurar ($T$ = saklanan tüm modlar):
- Kütle Gram'ı: VV bloğu $M_{div}$, çapraz blok $(PV)^\top MT=V^\top MT-Z^\top(G^\top MT)$ (herhangi bir $T$ için kesin), TT bloğu $T^\top MT$.
- Curl–curl Gram'ı: projeksiyondan etkilenmez.

Terimler ortak `_span_terms` içinde hesaplanır:
- **span:** Ortalama $\sqrt{r_k}$; kütle ($M_{div}$) ve enerji (curl–curl, $\mathbf H$'de $\lVert\mathbf E-\Pi\mathbf E\rVert$) normlarında, tüm hedefler üzerinden (`span_norm: both`).
- **selfsup:** $-\log\mathrm{tr}(A_V^{-1}M_{div})$. Projeksiyonla sınırlıdır. Neredeyse-gradyan bir kolonda projeksiyonsuz compliance $>10^3\times$ patlar (test edildi).
- **ortho:** $M_{div}$'in köşegen dışı korelasyonları.
- **freq:** Ritz frekanslarının z-MSE'si (`freq_weight`).

`_jacobi_cholesky`'de "ölü kolon" eşiği göreli olarak `diag ≤ 1e-12·max diag` alındı. Saf gradyan kolonların $10^{-16}$'lık yuvarlama köşegenleri Jacobi ölçeklemesinde $O(1)$ gürültüye dönüşmemeli. 2D'de pratikte bir etkisi yoktur.

**Metrikler:**
- `mode_k_rel_l2`: $M$-normunda işaretten bağımsız; yakın-dejenere kümede alt uzay hatası olarak hesaplanır. Çıktı sınırını aşan bir küme varsa, hesaba `FreqNext` = $K+1$. modun frekansı da katılır; o modlar NaN olur ve ortalamaya girmez.
- `freq_mae_ghz`, `freq_rel_err`.
- `span_rel_l2(_energy)`, `span_mode_k_rel_l2`.
- `grad_frac`: ham tabanın gradyan kütle payı.
- `kp_cg_iters`.

## 4. Seyrek matrislerin batch'lenmesi

`maxwell3d_collate` her operatörü **tek bir blok-diyagonal torch COO** tensörüne çevirir (float64). Blok yerleşimi dolgulu indeks uzayındadır: örnek $b$'nin kenarı $e$ → satır $b\cdot N_e^{max}+e$, düğümü $v$ → sütun $b\cdot N_v^{max}+v$. Dolgu satır ve sütunları boştur. Böylece `spmm(A, X[B,N,m])` bütün batch için tek bir seyrek çarpım olur ve `EdgeMask`/`Mask` ile tutarlıdır. Anahtarlar `M, K, G, Gt, Kp` ve `Kp_diag` (Jacobi) şeklindedir.
- CPU'da COO, CSR'den ~8× hızlı çıktı.
- Seyrek tensörler DataLoader worker'larından sorunsuz geçer, ama pin edilemez (3D'de `pin_memory` kapalı).
- `--no_operators` PKL'lerinde operatörler `geometry_operators(X, tets)` ile yeniden kurulur ve worker başına önbelleğe alınır.

## 5. Adım maliyeti ve $N$

Ölçüm: CPU, 4 thread, B=2, $D=128$, 4 katman, $m=24$, K=6; kutu mesh'leri; tam eğitim adımı (ileri + kayıp + geri).

| $N_v$ | $N_e$ | gövde+başlık (ileri) | $K_p$-PCG ileri (iter.) | adım |
|---:|---:|---:|---:|---:|
| 378 | 2 053 | 0.03 s | 0.04 s (60) | 0.19 s |
| 1 144 | 6 731 | 0.06 s | 0.22 s (90) | 0.60 s |
| 2 618 | 16 093 | 0.10 s | 0.45 s (120) | 1.38 s |
| 4 641 | 29 184 | 0.14 s | 0.97 s (140) | 2.88 s |
| 7 500 | 47 919 | 0.31 s | 1.98 s (170) | 5.34 s |

- Attention $O(ND^2)$'dir.
- PCG iterasyon sayısı $\sim h^{-1}\sim N^{1/3}$ büyür, maliyeti $O(\mathrm{nnz}\cdot m\cdot N^{1/3})\approx O(N^{4/3})$ olur ve CPU'da baskındır; geri geçiş bir çözüm daha ekler. GPU'da seyrek çarpım ucuzdur.
- Sonraki adımlar: (i) aynı geometri için $Z$ ile sıcak başlangıç, (ii) çok düzeyli (AMG) ön koşullayıcı, (iii) CPU eğitiminde örnek başına önceden alınmış seyrek Cholesky.

## 6. Duman testi (gerçek veri, `smoke3d.pkl`)

Veri: 30 geometri (9 pillbox, 9 axisym_cell, 12 blob), $N_v$ 1.2–1.7k, $N_e$ 6.8–9.5k, K=6. Küçük model: $D=32$, 2 katman, $m=16$. B=2, 5 epoch (60 adım), CPU. Eğitim ~46 s sürdü (doğrulama dahil ≈ 0.65 s/adım). Tek adım ≈ 0.55 s: ileri+kayıp 0.28 s, geri 0.27 s, PCG 90–100 iterasyon.

| Ölçü (5 epoch sonra) | test (3 geo.) | tümü (30 geo.) |
|---|---|---|
| alan rel-L2 ($M$-normu, ortalama) | 0.32 | 0.60 |
| frekans göreli hata | 16 % | 23 % (axisym 15 %, pillbox 22 %, blob 31 %) |
| span rel-L2 kütle / enerji | 0.23 / 0.57 | 0.28 / 0.63 |
| `grad_frac` (ham tabanın gradyan kütle payı) | 0.59 | 0.59 |

- 180 tahminin **hepsi** $f_{pred}\ge f_{true}$ çıktı; en küçük fark +7.4 %. Model mesh'i etiket mesh'i olduğundan beklenen üst sınır sağlanıyor.
- Ham taban kütlesinin ~%60'ı gradyandır: kayıplar gradyan içeriğini görmez. Projeksiyon bunu tam olarak temizler; projeksiyon olmasaydı Ritz değerleri çökerdi (docs/18 E4).

## 7. Veri tarafından beklenenler (sözleşmeye ek)

- `edges` sözlük sırasında ve düşük→yüksek yönlü olmalı, DOF = düşük→yüksek çizgi integrali. `geometry_operators` yeniden kurulumu bu kenar sırasını birebir üretmeli (yükleyici kontrol eder).
- Mesh bağlantılı olmalı (tek bileşen; ortalamasız PCG bunu varsayar) ve topolojik top olmalı (kulp yok: harmonik alanlar çekirdeğe eklenmiyor).
- `Y` gradyanlara $M$-ortogonal olmalı (ölçek serbest; kayıp ve metrikler normalize eder).
- `freq_next` saklanmalı: bölünmüş son çiftin metrik dışı bırakılması buna dayanıyor.
- `feature_names`, `x,y,z`, üç `dir_*` ve `*volume*` sütunlarını içermeli (augmentasyon ve attention ağırlığı bunlarla çalışır).
