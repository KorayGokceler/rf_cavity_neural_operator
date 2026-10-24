# 35 — Numerical checks requested by the referee report (preprint, 2026-10-24)

> Status: the five checks below were run in one session on branch `claude/field-model`. Scripts: `scripts/paper/`.
> No file under `src/` or `train.py` was changed. Large outputs (H5, logs, JSON) are in the session scratchpad and are
> not committed; every table below is reproducible with the commands given.
> Referee items: M1 (assumption A), M3 (inexact CG), E7 / E8 (baselines). Context: docs/34 §3, §8; docs/31 §3.4–3.5;
> `paper/sections/A04_method.tex`.

**Headline results (the ones that touch claims of the paper):**

1. Assumption (A) holds in all 15 cases tested (5 meshes x p = 1, 2, 3): dim ker K = rank G = number of interior
   H1(p+1) DOFs + extra wall potentials (§1).
2. The upper bound θ_k ≥ λ_h,k is **not** exact for the inexact CG projection: with a gradient admixture it is
   violated by up to 1.5e-8 (relative) at the shipped τ = 1e-8 and by up to 8.0e-5 at τ = 1e-4 (γ = 0.99). Every
   violation is covered by the a posteriori bound θ̃ ≥ λ_h/(1+η̄) of the appendix (η̄ ≈ 1–3 τ here). With pure gradient
   columns and τ = 1e-4 the drop rule fails (4 of 8 gradient directions are kept, η̄ = 1, θ ≈ 0, margin −100 %); at
   τ ≤ 1e-6 all 8 are dropped (§2).
3. **A GP on the six geometry parameters is as accurate as the model in this family:** 1.01 % mean / 1.82 % max error
   with 40 LF samples (model C2: 1.37 % / 5.30 %), 1.64 % with 24 LF (model B2: 1.55 %); with 8 samples the model is
   better (GP 3.8 %, model A 1.66 %) (§3a).
4. **A direct FE solve dominates the model in time and accuracy:** p1 on the model mesh needs 1.7 s for 1.25 % mean
   error (model: 8.3 s for 1.37–1.66 %); p2 on a 48 % coarser curved mesh needs 5.7 s for 0.037 % (§3b).

## 1. Assumption (A): rank of the kernel (M1)

**Command.**

```bash
python scripts/paper/check_assumption_a.py --geoms pillbox shell annulus --pillbox_maxh 0.05 --json rank_pillbox.json
python scripts/paper/check_assumption_a.py --geoms pillbox --pillbox_maxh 0.04 --json rank_pillbox179.json
python scripts/paper/check_assumption_a.py --geoms pipes --pipes_h5 lf_test.h5 --pipes_key sample_0024 \
    --pipes_cs 0.5 --pipes_maxh 0.2 --max_dense 10000 --json rank_pipes.json
```

**Setup.** `highorder.field_operators(mesh, order=p)` (the operators of the model), free DOFs only, dense
generalised `scipy.linalg.eigh(K, M)`. "Kernel" = eigenvalues below 1e-10 · max. Meshes (curve 2): pillbox
r = h = 0.1 m with maxh 0.04 (179 tets; the test fixture of `tests/test_field_model.py` has 109 tets with maxh 0.05
in this netgen version) and 0.05; spherical shell (two wall shells, one extra potential); annular cylinder (solid
torus, b1 = 1); a coarse `pillbox_pipes` mesh (316 tets, curvaturesafety 0.5, rebuilt from the BREP of test sample 24).
rank G by SVD of the dense G on the free rows (threshold 1e-10 · σ_max), rank Kp = GᵀMG by `eigvalsh`. "dist" is the
largest M-norm of (N − G Kp⁻¹ GᵀM N) over the kernel vectors N: ker K lies in range G. Label = shift-invert
eigenvalue of `highorder.solve_modes` (the generator's solver), scaled to the normalised geometry.

| mesh | tets | p | n_free | dim ker K | rank G | interior H1(p+1) + extra | rank Kp | largest "zero" eig / max | smallest nonzero / max | dist(ker K, range G) | λ_min vs label | λ_min vs analytic TM010 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pillbox | 109 | 1 | 116 | 58 | 58 | 58 + 0 | 58 | 3.4e-16 | 3.3e-02 | 1.7e-15 | +2.0e-15 | +3.5e-02 |
| pillbox | 109 | 2 | 672 | 282 | 282 | 282 + 0 | 282 | 3.2e-16 | 7.5e-03 | 6.0e-15 | +4.7e-15 | +2.3e-03 |
| pillbox | 109 | 3 | 1996 | 781 | 781 | 781 + 0 | 781 | 3.2e-16 | 2.9e-03 | 1.1e-14 | +3.2e-14 | +3.1e-04 |
| pillbox | 179 | 1 | 220 | 113 | 113 | 113 + 0 | 113 | 3.6e-16 | 2.2e-02 | 2.4e-15 | -3.1e-15 | +3.0e-02 |
| pillbox | 179 | 2 | 1185 | 508 | 508 | 508 + 0 | 508 | 3.3e-16 | 5.2e-03 | 6.3e-15 | +7.1e-15 | +1.6e-03 |
| pillbox | 179 | 3 | 3436 | 1367 | 1367 | 1367 + 0 | 1367 | 3.4e-16 | 2.0e-03 | 1.4e-14 | -6.6e-15 | +1.2e-04 |
| shell | 139 | 1 | 184 | 93 | 93 | 92 + 1 | 93 | 3.9e-16 | 2.5e-02 | 4.2e-15 | +2.2e-15 | - |
| shell | 139 | 2 | 963 | 414 | 414 | 413 + 1 | 414 | 3.3e-16 | 5.7e-03 | 8.6e-15 | -1.5e-14 | - |
| shell | 139 | 3 | 2756 | 1103 | 1103 | 1102 + 1 | 1103 | 3.2e-16 | 2.1e-03 | 3.9e-14 | -5.3e-15 | - |
| annulus | 78 | 1 | 76 | 38 | 38 | 38 + 0 | 38 | 3.5e-16 | 7.7e-02 | 1.2e-15 | +2.0e-15 | - |
| annulus | 78 | 2 | 462 | 192 | 192 | 192 + 0 | 192 | 3.0e-16 | 1.3e-02 | 4.4e-15 | -9.9e-15 | - |
| annulus | 78 | 3 | 1392 | 540 | 540 | 540 + 0 | 540 | 2.2e-16 | 3.9e-03 | 1.1e-14 | +1.8e-15 | - |
| pillbox_pipes | 316 | 1 | 436 | 229 | 229 | 229 + 0 | 229 | 2.4e-16 | 4.2e-03 | 7.3e-15 | +4.0e-15 | - |
| pillbox_pipes | 316 | 2 | 2220 | 969 | 969 | 969 + 0 | 969 | 1.4e-16 | 7.7e-04 | 3.1e-14 | -8.0e-15 | - |
| pillbox_pipes | 316 | 3 | 6312 | 2547 | 2547 | 2547 + 0 | 2547 | 1.4e-16 | 2.7e-04 | 7.2e-14 | +9.8e-14 | - |

(The pillbox has 109 tets at maxh 0.05; rows 1-3 are that mesh. 15 rows = 5 meshes x 3 orders.)

**Reading.** In every case dim ker K equals rank G equals the number of columns of G (interior H1(p+1) DOFs plus one
potential per extra wall shell: the shell has 1 extra), and rank Kp is the same number, so G has full column rank and
range G = ker K. The zero cluster (≤ 4e-16 of the largest eigenvalue) is separated from the first nonzero eigenvalue
(≥ 2.7e-4 of the largest) by at least 12 orders of magnitude, so the cut at 1e-10 is not delicate. The kernel vectors
lie in range G to 7e-14 (M-norm). The smallest nonzero generalised eigenvalue equals the generator's shift-invert
eigenvalue to 1e-13, and for the pillbox it is above the analytic TM010 value (+3.5e-2 at p = 1 down to +1.2e-4 at
p = 3, so the discrete eigenvalue is an upper bound of the exact one on these meshes). The solid torus (b1 = 1) shows
no extra kernel in the E formulation, as stated in Section 03 of the paper. The statement can be written as the
referee proposes: "(A) was verified numerically: dim ker K = rank G = dim S_h + #extra wall components for p = 1, 2, 3
on five meshes (pillbox, spherical shell, annular cylinder, pipe pillbox)".

**Limitations.** Meshes are small (≤ 6312 free DOFs) because the check needs dense linear algebra; the real model
meshes (10⁴ tets, 5·10⁴–10⁵ DOFs) are not covered, only indirectly by `test_field_model.py` (KG = 0). Only curve 2
meshes; the production meshes use curve 3. The pipe mesh is a coarse remesh (min Jacobian ratio 0.04) of one sample.

## 2. Ritz-bound stress test (M3)

**Command.**

```bash
python scripts/paper/check_ritz_bound.py --data lf_test.h5 --key sample_0024 --seeds 3 --json ritz_bound_0024.json
python scripts/paper/check_ritz_bound.py --data lf_test.h5 --key sample_0027 --seeds 2 --json ritz_bound_0027.json
```

**Setup.** LF test geometry 24 (4818 tets, 51 105 DOFs, 18 662 potentials) and 27 (6315 tets, 69 645 DOFs); the stored
`u_model` are the exact discrete p2 eigenvectors, λ_h,k = `rq_model`·scale² (K = 4: 16.168, 21.414, 21.415, 32.046 for
geometry 24). Operators from `FieldDataset.load` + `field_item` + `field_collate`; the call is
`hcurl.hcurl_ritz(V, batch, K=4, mass_ridge=1e-8, drop_tol=1e-6, eig_broadening=1e-4, tol=τ, maxiter=3000)` with the
configuration-file defaults. m = 12 columns, wall rows 0, "random" = white noise per free DOF at unit M-norm. Bases:
(a) 4 exact eigenvectors + 8 random; (b) exact + relative noise ε (M-norm) + 8 random; (c) 12 random; (d) exact +
gradient G r of mass fraction γ (γ = ‖g‖²/(‖u‖²+‖g‖²), so the divergence-free fraction is 1 − γ) + 8 random; (g, extra)
4 exact + 8 pure gradient columns; γ = 0.9, 0.99 are extras. Diagnostics: margin_k = θ_k/λ_h,k − 1, a violation is
margin < −1e-12, η̄ = max |eig(Wᵀ D E D W)| with E = −sym(Zᵀρ), Z the exact solution of Kp Z = B (SuperLU), ρ = Kp Z̃ − B
the CG residual and W the kept whitened Ritz subspace (Wᵀ D M_div D W = I), "dropped" = whitened μ ≤ drop_tol,
"spurious" = Ritz values of span(V) without projection (VᵀKV c = θ VᵀMV c) below 0.5 λ_h,1. Note that the stored
eigenvectors are float32 (the H5 stores `u_model` in single precision, as does the model's V), i.e. they are already
"exact + 6e-8 noise"; this is why margins at ε = 0 are ±1e-14 and not exactly 0.

Min over modes k = 1..4 and seeds (3) of θ_k/λ_h,k − 1, with the number of violations out of 12 (4 modes x 3 seeds), geometry 24:

| basis | par | τ = 1e-4 | τ = 1e-6 | τ = 1e-8 (shipped) | τ = 1e-10 | dropped | spurious w/o projection |
|---|---|---|---|---|---|---|---|
| a exact + random | | -5.3e-15 / 0 | -1.6e-14 / 0 | -1.7e-15 / 0 | -3.7e-14 / 0 | 0 | 0 |
| b ε | 0 | -2.9e-14 / 0 | -4.2e-14 / 0 | -2.6e-14 / 0 | -7.4e-15 / 0 | 0 | 0 |
| b ε | 1e-8 | +2.1e-14 / 0 | +2.6e-14 / 0 | +1.3e-15 / 0 | +3.0e-14 / 0 | 0 | 0 |
| b ε | 1e-6 | +1.0e-10 / 0 | +1.0e-10 / 0 | +1.0e-10 / 0 | +1.0e-10 / 0 | 0 | 0 |
| b ε | 1e-4 | +1.0e-06 / 0 | +1.0e-06 / 0 | +1.0e-06 / 0 | +1.0e-06 / 0 | 0 | 0 |
| c random | | +2.8e+02 / 0 | +2.8e+02 / 0 | +2.8e+02 / 0 | +2.8e+02 / 0 | 0 | 0 |
| d γ | 1e-6 | -8.6e-13 / 0 | -6.4e-14 / 0 | -5.3e-14 / 0 | -1.0e-14 / 0 | 0 | 0 |
| d γ | 1e-4 | -7.3e-11 / 3 | -7.7e-13 / 0 | -4.5e-14 / 0 | -3.8e-13 / 0 | 0 | 0 |
| d γ | 1e-2 | -7.7e-09 / 8 | -6.6e-11 / 4 | -1.3e-12 / 1 | -2.7e-13 / 0 | 0 | 0 |
| d γ | 0.5 | -7.5e-07 / 7 | -2.6e-09 / 6 | -2.0e-11 / 4 | -6.1e-13 / 0 | 0 | 1 (θ_1 = 0.5 λ_h,1) |
| d γ (extra) | 0.9 | -6.2e-06 / 4 | -2.6e-08 / 6 | -6.1e-10 / 5 | -5.7e-12 / 2 | 0 | 4 |
| d γ (extra) | 0.99 | -7.4e-05 / 4 | -4.5e-07 / 3 | -6.6e-09 / 6 | -1.4e-10 / 5 | 0 | 4 |
| g pure gradient (extra) | | **-1.0 / 12** | +2.6e-14 / 0 | +2.5e-14 / 0 | +2.6e-14 / 0 | 4-8 (8 needed) at τ = 1e-4; 8 otherwise | 8 |

Geometry 27 (2 seeds) gives the same picture (γ = 0.5: -5.9e-8 / -1.0e-9 / -1.4e-10 / -8.5e-13; γ = 0.99: -8.0e-5 /
-9.0e-7 / -1.5e-8 / -1.7e-10; pure gradient at τ = 1e-4: -1.0, 8 violations; no case with a margin below −1e-12 at
ε ≤ 1e-4 for the exact-eigenvector bases a, b). Other columns, geometry 24 (27):

| τ | max η̄ (bases a-d) | max CG iterations | η̄ for pure-gradient basis g | unconverged CG (3000) |
|---|---|---|---|---|
| 1e-4 | 2.5e-4 (3.7e-4) | 80 (50) | **1.0** | 0 |
| 1e-6 | 2.5e-6 (2.0e-6) | 120 (90) | 3e-24 (all 8 dropped) | 0 |
| 1e-8 | 2.6e-8 (2.8e-8) | 160 (120) | 4e-25 | 0 |
| 1e-10 | 2.8e-10 (4.3e-10) | 200 (150) | 4e-25 | 0 |

Further facts: (i) with the EXACT projection (sparse direct Kp⁻¹) the minimum margin over all bases except random is
−2.3e-13 (geometry 24) and −2.9e-13 (geometry 27): no violation at the 1e-12 level. (ii) The a posteriori guarantee
θ̃_k ≥ λ_h,k/(1+η̄) holds in all 156 + 104 runs (geometry 24 + 27) where η̄ < 1, i.e. all except the 3 + 2 runs of the pure-gradient
basis at τ = 1e-4 (η̄ = 1.0, margin −1.0: θ̃ ≈ 0, the guarantee is void). (iii) The measured violation never exceeds η̄ and grows with
γ (at τ = 1e-8: 1.3e-12 at γ = 0.01, 2e-11 to 1.4e-10 at γ = 0.5, 6.6e-9 to 1.5e-8 at γ = 0.99); η̄ ≈ 1–3 τ at these
meshes (the constant C of the referee's estimate is ≈ 1–3 for the admixture bases and ≳ 10 for pure gradients). (iv) The drop rule (μ ≤ 1e-6) works when the residual gradient left by the
CG has mass fraction (Cτ)² < 1e-6: at τ = 1e-4 only 4–5 of the 8 pure-gradient directions are dropped, the rest has
μ̃ > 1e-6 and enters the Ritz problem with θ ≈ 0 (this is the regime the referee warned of); at τ = 1e-6 all 8 are
dropped. (v) Without the projection, a basis with 8 pure gradient columns has 8 Ritz values at θ ≈ 0 (< 0.5 λ_h,1);
the gradient admixture γ = 0.9 / 0.99 lowers the first four Ritz values by (1 − γ) (4 spurious); with projection the
count is 0 for τ ≤ 1e-6. **Random white-noise bases (c) and random padding (a, b) do not produce spurious modes without
the projection** (their curl energy is large): the projection matters for smooth gradient-rich bases, not for noise.
(vi) θ/λ_h − 1 of a random basis is +160 to +280.

**Reading.** At the shipped τ = 1e-8 the bound θ_k ≥ λ_h,k can be violated, but only for a basis whose
divergence-free fraction is far from 1 and then by at most 1.5e-8 (γ = 0.99; 1.4e-10 at γ = 0.5; 1.4e-12 at γ = 0.01),
orders of magnitude below the model error (0.5–3 %), and always within η̄ ≈ 2.8e-8. At τ = 1e-4 the violations reach
8.0e-5 and the drop rule can fail for nearly pure gradient directions. So the sentence "the bound is rigorous in
exact arithmetic and observed for the implemented solver" is true for exact arithmetic and, for τ ≤ 1e-6, up to a
relative error η̄ ≈ 1–3 τ; the appendix bound is confirmed numerically (it is never violated when η̄ < 1) and the
observation "which is why directions with tiny μ are dropped" is supported only for τ ≲ 1e-6. A trained model has
γ ≈ 0.5–0.6 before projection (docs/20 §6: gradient fraction 0.59), so the relevant rows are γ = 0.5: margin −2e-11 /
−1.4e-10 (geometry 24 / 27) at τ = 1e-8.

**Limitations.** Two geometries of one family, one CG preconditioner (Jacobi), maxiter 3000 never reached. Bases are
synthetic: trained bases have a continuum of μ values, the pure-gradient and exact-plus-gradient rows are the
adversarial extremes. η̄ is computed over the whole kept subspace (conservative: the Ritz vectors only see a subspace).
θ is compared to λ_h of the p2 space of the stored mesh; mode identity is not tested (index-wise bound only).

## 3. Baselines (E7, E8)

### 3a. Parametric regression on the geometry parameters

**Command.**

```bash
python scripts/paper/check_baseline_regression.py --test hf_test.h5 --train lf40=lf_train.h5 \
    --train lf24=lf_train.h5:0-23 --train lf8=lf_train.h5:0-7 --train hf8=hf_train.h5 --json baseline_regression.json
```

**Setup.** Inputs: the six continuous parameters R, Lc, rp, Lpipe, rho_iris, rho_out of `geom_params` (the four others
are constant 0 in this family), standardised; targets: the 4 lowest frequencies [GHz]. Variants: GP (scikit-learn 1.9,
`Constant·kernel + White`, ARD length scales, 8 optimiser restarts) with RBF or Matern-5/2; MLP (2x64 tanh, L-BFGS,
α = 1e-4, mean of 5 seeds); each also with log inputs and log targets. The variant reported is the one with the lowest
5-fold CV error on the TRAINING set (the test set is not used for selection). Training sets: LF 40 (ids 0-23, 30-45),
LF 24 (ids 0-23), LF 8 (ids 0-7), HF 8 (ids 0-7, the p3 labels). Test: the six HF test geometries (24-29), statistics of
the paper: |f/f_HF − 1| over 6 x 4 = 24 values. `scikit-learn` is not in the repository requirements; it was
installed with pip in the sandbox for this check.

| training set | n | selected variant | CV mean [%] | test mean [%] | median | max | per geometry 24..29 [%] | model (docs/34 §8.2) mean / median / max |
|---|---|---|---|---|---|---|---|---|
| LF 40 | 40 | GP Matern-5/2, log | 1.66 | **1.01** | 1.16 | **1.82** | 0.76 / 1.14 / 1.15 / 0.92 / 1.15 / 0.97 | C2: 1.37 / 1.00 / 5.30 |
| LF 24 | 24 | GP Matern-5/2, log | 1.91 | 1.64 | 1.85 | 4.53 | 1.13 / 1.36 / 1.85 / 2.15 / 1.61 / 1.75 | B2: 1.55 / 1.06 / 6.57 |
| LF 8 (ids 0-7) | 8 | GP RBF, log | 4.59 | 3.75 | 2.51 | 9.95 | 4.22 / 2.21 / 2.14 / 2.50 / 6.04 / 5.41 | - |
| HF 8 (ids 0-7) | 8 | GP RBF, log | 4.58 | 3.76 | 2.55 | 9.93 | 4.19 / 2.21 / 2.15 / 2.53 / 6.04 / 5.41 | A: 1.66 / 1.11 / 8.11 |

Other variants (test mean, LF 40): GP RBF 2.15, GP Matern 1.50, GP RBF log 2.14, MLP 2.55, MLP log 2.39 (LF 24: MLP 6.33,
4.14). The GP trained on HF 8 and on LF 8 are indistinguishable (3.76 vs 3.75 %): fidelity does not matter for a
regression at these sizes, the number of geometries does.

**Reading.** The best parametric baseline matches or beats the neural operator on every statistic except the median
when 24 or more geometries are available (LF 40: mean 1.01 against 1.37 %, max 1.82 against 5.30 %, median 1.16 against
1.00 %; LF 24: 1.64 against 1.55 %), and per geometry it is better on 4 of 6 against C2 (per geometry C2:
0.66 / 1.23 / 1.25 / 2.93 / 0.98 / 1.17). Geometry 27 is not an outlier for the GP (0.92 %) whereas it is for the model
(2.9-4.4 %). With 8 samples the model is about 2.3 times more accurate than the GP (arm A: 1.66 against 3.76 %), which is
the regime where a field-based model has an advantage over a 6-parameter regression. This check does not support a
claim that the model beats a parametric surrogate on this single low-dimensional family; the paper's argument has to
be generality (geometry given by a mesh, fields and figures of merit as outputs), not frequency accuracy.

**Limitations.** One family, six parameters; test geometries 24 and 27 lie slightly outside the training parameter box
(rho_out, Lc), 6 geometries, no seeds for the model. The GP is a frequency-only baseline: it gives no fields. The model
numbers are single runs from docs/34 §8.2, not re-run here (no checkpoint was available).

### 3b. Coarse FE cost-accuracy

**Command.**

```bash
python scripts/paper/check_baseline_coarse_fe.py --data hf_test.h5 --json coarse_fe_all.json
python scripts/paper/make_figures.py cost_accuracy --fe coarse_fe_all.json --out paper/figures
```

**Setup.** `highorder.solve_modes(mesh, 4, order=p)` (the generator's solver: shift-invert + kernel projection,
ARPACK tol 1e-10, 4 extra vectors), 1 NGSolve thread, `OMP_NUM_THREADS=1`, nothing else running; one run per
configuration. Times: assembly = `info["t_assemble"]` (space, K, M, Kp, shifted operator), solve = factorisation +
ARPACK + final Ritz (`t_factor + t_eig`); mesh = netgen generation + curving + repair for the coarse rungs (not for
the model mesh, which is read from the H5). Reference: stored HF frequencies (p3, one level). "Model mesh" = the stored
mesh of the sample (LF label mesh); p2 on it is the LF label. Coarse rungs: `netgen_mesh` from the stored BREP with
curvaturesafety 1.25 (curve 3 and curve 2) and 1.1 (curve 2 / 3), maxh of the sample (maxh x 1 and x 2 gave the same
mesh on geometry 24: the curvature controls the element size on these faces). curvaturesafety ≤ 1.0 folds elements for every maxh
(tested on geometry 24 with cs 1.0 / 0.75 / 0.5 / 0.35 and maxh x 1, x 2; netgen_mesh without retries), so no
much coarser valid curved mesh exists with netgen's mesher here;
cs 1.1 produced a valid mesh only for one geometry; cs 1.25 / curve 3 for four geometries (the others fold or are
not coarser). Lean settings (ARPACK tol 1e-6, 1 extra) are in the JSON and are 10–25 % faster, same errors.

Mean over the geometries for which the rung exists (`n_geo`); error = |f/f_HF − 1| over modes 0-3 (24 values for 6 geometries):

| mesh | p | n_geo | tets | DOF | mesh [s] | assembly [s] | solve [s] | total [s] | mean err [%] | median | max err [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model mesh | 1 | 6 | 6969 | 19 928 | - | 0.30 | 1.36 | 1.68 | 1.247 | 1.429 | 2.247 |
| cs 1.25, curve 3 | 1 | 4 | 5018 | 14 427 | 0.82 | 0.20 | 0.90 | 1.12 | 1.407 | 1.508 | 2.349 |
| cs 1.25, curve 2 | 1 | 6 | 3624 | 10 888 | 0.52 | 0.14 | 0.55 | 0.70 | 1.668 | 1.788 | 2.769 |
| cs 1.1, curve 2 | 1 | 1 | 2855 | 8516 | 0.44 | 0.10 | 0.31 | 0.42 | 1.702 | 1.915 | 2.101 |
| model mesh (= LF label) | 2 | 6 | 6969 | 75 180 | - | 1.22 | 12.48 | 13.90 | 0.019 | 0.019 | 0.045 |
| cs 1.25, curve 3 | 2 | 4 | 5018 | 54 315 | 0.82 | 0.87 | 7.96 | 8.96 | 0.026 | 0.025 | 0.071 |
| cs 1.25, curve 2 | 2 | 6 | 3624 | 40 382 | 0.52 | 0.66 | 4.89 | 5.65 | 0.037 | 0.032 | 0.111 |
| cs 1.1, curve 2 | 2 | 1 | 2855 | 31 653 | 0.44 | 0.47 | 3.43 | 3.97 | 0.059 | 0.048 | 0.098 |

Per geometry on the model mesh (sample, tets): p1 total 0.98 s (24, 4818), 3.08 (25, 10 030), 1.30 (26, 5844), 1.37 (27,
6315), 2.10 (28, 9390), 1.09 (29, 5419); p2 total 7.35 s, 21.8, 12.2, 10.4, 20.1, 10.4. The model's own cost (docs/34
§8.2: operators 5 s + forward 3.3 s = 8.3 s on one thread) was not re-measured here (no checkpoint) and refers to
another geometry; its errors are those of arms A (1.66 %) and C2 (1.37 %) of docs/34 §8.2. The p1 / p2 error is nearly
independent of which geometry (p1: 0.2-2.2 %, p2: 0.002-0.11 %).

**Reading.** On this family the direct FE solve is a better trade-off than the model at every point measured:
p1 on the model mesh costs 1.7 s on average (0.98 s for geometry 24) at a mean error of 1.25 % (against 8.3 s at 1.4-1.7 %
for the model), p1 on the cs 1.25 / curve 2 mesh 0.70 s at 1.67 %, and p2 on the cs 1.25 / curve 2 mesh 5.65 s at 0.037 %
(a 40x lower error than the model for 0.7x the time). The p2 solve on the model mesh took 13.9 s on average (assembly 1.2 s +
solve 12.5 s) and 7.4 s for geometry 24, i.e. less than the 8.3 s attributed to the model, so the "8 s against 18 s"
of Sections 01/07/08 (docs/34 §8.2; geometry and load not recorded there) is not reproduced on an idle machine: the
ratio is 1.7 on average with the model time taken from another geometry. More importantly, the model is dominated by
FE in a lower-order or coarser space, which the paper has to state (referee E7). The figure `cost_accuracy.pdf` shows this.

**Limitations.** Times from a single run on a shared 4-core VM with one worker (± 10 %), model time not re-measured
and from a different geometry/load; the model on GPU or with a larger batch was not considered (the paper's own
argument: benefits only for much finer spaces); only the pillbox-with-pipes family; netgen's coarsening range is
narrow (at most 48 % fewer tets than the model mesh, and not for every geometry), so "p2 on a much coarser mesh"
could not be tested; straight-sided meshes were not tried (their geometry error is not comparable to the labels).

## 4. Figures

All PDF (fonts embedded as Type 42), 6.5 in wide (text width of `paper/main.tex`), 8-8.5 pt text, Okabe-Ito
palette (blue #0072B2, sky #56B4E9, orange #E69F00, vermillion #D55E00, black), hatches so that the bars differ in
greyscale, no titles. A PNG preview next to each PDF is for a quick look only.

| file | content | command |
|---|---|---|
| `paper/figures/field_modes.pdf` | \|E\|/max on the y = 0 plane, modes 0 (TM010-like, 2.166 GHz) and 1 (TE111-like dipole, 2.492 GHz) of HF test geometry 24; stored p2 field evaluated with NGSolve on the rebuilt curved mesh; mode 1 belongs to a degenerate pair, so its orientation in the plane is arbitrary | `make_figures.py field_modes --data hf_test.h5 --key sample_0024` |
| `paper/figures/equal_cost_per_geometry.pdf` | grouped bars, mean \|f error\| over modes 0-3 for arms A (HF 8), B2 (LF 24), C2 (LF 40) on test ids 24-29; numbers of docs/34 §8.2 typed into the script | `make_figures.py equal_cost_per_geometry` |
| `paper/figures/cost_accuracy.pdf` | time per geometry (1 thread) against mean frequency error, log-log: FE p1 / p2 on the model mesh and on the coarser cs 1.25 / curve 2 mesh (the rungs that exist for all six geometries), and the model (8.3 s; arms A and C2) | `make_figures.py cost_accuracy --fe coarse_fe_all.json` |

Caption hints: `field_modes`: "geometry id 24, y = 0 plane, normalised by the maximum on the plane; the high values
near the iris fillets in mode 0 are the surface peaks"; `cost_accuracy`: "FE times are measured (1 thread, assembly +
solve, mean over six test geometries), the model time is the 8.3 s of docs/34 on one geometry".

## 5. Data (regenerated for these checks) and reproduction

```bash
G="--labels field --families pillbox_pipes --seed 0 --sampling sobol --deform_prob 0 --min_fillet 0.12 --n_workers 3 --threads 1"
python src/data_gen/dataset_generator_3d.py $G --no_adapt --label_order 2 --model_order 2 --ids_file ids_lf_train.txt --h5_filename lf_train.h5   # ids 0-23, 30-45
python src/data_gen/dataset_generator_3d.py $G --no_adapt --label_order 2 --model_order 2 --ids_file ids_test.txt --h5_filename lf_test.h5       # ids 24-29
python src/data_gen/dataset_generator_3d.py $G --max_ndof 150000 --ids_file ids_test.txt --h5_filename hf_test.h5
python src/data_gen/dataset_generator_3d.py $G --max_ndof 150000 --ids_file ids_hf_train.txt --h5_filename hf_train.h5                           # ids 0-7
```

Measured with 3 workers on the 4-core VM (one thread each): LF test 3.4 s mesh + 39.9 s label per sample, HF test
3.2 s + 170.3 s (ratio 4.0), LF 40 train 4.2 s + 47.5 s, HF 8 train 2.9 s + 133.1 s per sample; LF and HF meshes of
test geometries are identical (4818-10 030 tets), and the LF-HF frequency differences of the regenerated test labels
are +0.004-0.05 % (e.g. geometry 24: 2.16603 against 2.16561 GHz), in line with docs/34 §8.1.

Not done / not possible: a re-run of the neural-operator arms (no checkpoints; docs/34 §8.2 numbers are used), a
second geometry family, seeds for the model.
