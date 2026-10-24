# Editor notes — revision pass 3 (integrate the referee checks)

The checks the referee asked for are done: docs/35_PAPER_CHECKS.md (read it fully — commands, tables, readings,
limitations) and three figures in paper/figures/ (field_modes.pdf, equal_cost_per_geometry.pdf,
cost_accuracy.pdf; include with `\includegraphics[width=\linewidth]{figures/<name>}`). Replace every
`% TODO pass 3` in your section with the result, or delete the TODO if the result does not belong there.
Cite numbers as `% src: docs/35 §N`. BRIEF §4 rules still hold.

## What the checks found (all sections must agree)

1. **Assumption (A) holds** in all 15 cases checked (5 curved meshes incl. a two-wall shell, p = 1, 2, 3): dim ker K =
   rank G = rank Kp = interior H1(p+1) DOFs + extra wall potentials; kernel eigenvalues ≤ 4e-16 relative, first
   nonzero ≥ 2.7e-4; ker K ⊂ range G to 7e-14. Small meshes only (dense eigensolves).
2. **The upper bound under the implemented, inexact projection:** exact arithmetic — rigorous; implemented solver —
   holds up to a relative perturbation η̄ ≈ (1–3)·τ_CG (≈ 1e-8 at the shipped τ_CG = 1e-8), and the a posteriori
   guarantee θ̃ ≥ λ_h/(1+η̄) held in every run with η̄ < 1. Violations of the plain bound appear only with gradient
   admixture (up to −1.5e-8 relative at γ = 0.99, −1.4e-10 at γ = 0.5, shipped τ); none for exact eigenvectors with
   noise or for random bases. The drop rule protects only when τ_CG ≲ 1e-6 (pure-gradient columns at τ = 1e-4: η̄ = 1,
   θ ≈ 0). Without the projection, gradient-rich bases give spurious modes (8 → 0 with projection).
   → State the proposition for exact arithmetic and add "and, for the implemented CG projection, up to a relative
   error η̄ ≈ τ_CG, verified a posteriori" wherever the guarantee is stated (abstract, 01, 04, 08).
3. **Parametric-regression baseline (one family, 6 geometry parameters):** a Gaussian process on the geometry
   parameters reaches 1.01 / 1.16 / 1.82 % (mean / median / max) with LF 40 vs the model's 1.37 / 1.00 / 5.30 %;
   1.64 % with LF 24 vs 1.55 %; with 8 geometries the model is ~2.3× better (GP 3.75–3.76 %, model 1.66 %). The GP
   is insensitive to label fidelity (LF 8 ≈ HF 8) and gives no fields. → On a single low-dimensional family the
   model has no accuracy advantage over parametric regression once ≥ 24 geometries are available.
4. **Coarse-FE baseline (1 thread, idle):** p1 on the model mesh 1.68 s, 1.25 % mean error; p2 on a ~48 % coarser
   valid curved mesh 5.65 s, 0.037 %; p2 on the model mesh (= the LF label) 13.9 s, 0.019 %. The model path costs
   8.3 s (docs/34 §8.2; measured in an earlier run on one geometry, not re-measured) for 1.4–1.7 %. → Plain FE
   dominates the model in cost and accuracy at these sizes. The earlier "8.3 s vs 18 s" comparison is not
   reproduced (p2 solve 13.9 s mean, 7.4 s on geometry 24) — drop it; say the model has no speed advantage here.
5. **Label cost re-measured:** HF 173.5 s vs LF 43.3 s per sample on the 6 test geometries (3 workers × 1 thread),
   ratio 4.0 — add as a row of the cost-conditions table (06 / A06); the range "2.8–4.3×" stays.

## Framing (abstract, 01, 07, 08)

The paper is a method + data-pipeline paper with controlled small-scale evidence and an honest negative baseline
result. Contributions that stand: the construction and its discrete guarantee (plus the a posteriori treatment of
the inexact projection), space independence, the validated label generator and its cost reductions, LF labels on the
model mesh (2.8–4.3× cheaper, ≤ 0.047 % / ≤ 0.32 % from HF), and the finding that at this scale the number of training geometries,
not the label fidelity, decides the accuracy (for the model and for the GP). What is not shown: an accuracy or speed
advantage over parametric regression on one low-dimensional family or over direct FE at ≤ 10k tets. Where the
approach could pay off — geometry sets without a shared low-dimensional parametrisation (arbitrary CAD), predicted
fields and QoI with a guaranteed eigenvalue ordering, warm starts for large FE solves, larger meshes where FE cost
grows faster than the model's — is future work and must be worded as hypothesis.

## Assignments

- **01:** contribution list and Scope per the framing; bound wording per item 2; baselines in one sentence.
- **03:** one sentence after eq:03-A: verified numerically on 15 small cases (\cref to the appendix table owned by 04).
- **04 / A04:** item 1 as a short table in A04 + one sentence in 04; item 2: rewrite the inexact-CG remark and the
  appendix with the stress-test table (τ × γ worst margins, η̄ values, drop-rule failure at τ = 1e-4, spurious-mode
  counts with/without projection); the proposition text stays exact-arithmetic.
- **05:** field_modes.pdf as a figure (caption: |E| on the y = 0 cut, HF label of test geometry 24, modes 0 and 1;
  mode 1 belongs to a degenerate pair, orientation arbitrary); otherwise unchanged.
- **06 / A06:** item 5 (cost row); one sentence: the GP baseline is equally insensitive to label fidelity (LF 8 ≈ HF 8).
- **07 / A07:** new subsection "Baselines" with two compact tables (GP vs model; coarse FE vs model) and
  cost_accuracy.pdf; equal_cost_per_geometry.pdf in the equal-cost subsection (it may replace the per-geometry
  table, move that table to A07); remove the "about 2× faster" statements; limits of the baselines (GP: no fields, 2
  test geometries slightly outside the training box; FE: meshing time excluded on both sides? — state what each
  timing includes per docs/35); keep ≤ 4 pages.
- **08:** discussion and limitations per the framing; the structural-ceiling paragraph now has measured support.
- **Editor (abstract):** after the sections.

Compile in a private copy. Reply with the changes.
