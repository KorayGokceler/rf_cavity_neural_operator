# Editor notes — revision pass 2 (after the referee report)

Read the referee report: /tmp/claude-0/-home-user-rf-cavity-neural-operator/551360d9-0e42-536d-82da-64927523c26c/scratchpad/referee_report.md
(its line-level suggestions for your section, §1–§3 and the spot-check log). Apply every item that concerns your
section unless it conflicts with this file. The rules of BRIEF.md §4 and EDITOR_NOTES.md still hold.

New results (rank check of assumption (A), Ritz-bound stress test with a posteriori η̄, parametric-regression and
coarse-FE baselines, a field figure) are being produced now and will be added in pass 3; leave a LaTeX comment
`% TODO pass 3: <what>` where they belong, do not invent them.

## Decisions (all sections)

1. **Title** (editor): "...: Discrete Upper-Bound Eigenvalues on Curved High-Order Elements and Training with
   Low-Fidelity Labels". Everywhere: upper bounds of the *discrete* eigenvalues of the space the model runs in.
2. **LF vs HF claim:** "comparable" (not "at least as accurate"). State the confound (fidelity and number of
   geometries change together; LF 8 vs HF 8 was not run), single runs, no seeds, and that the LF/HF label
   difference (≤ 0.047 % frequency, ≤ 0.32 % field) is one to two orders of magnitude smaller than the model error (frequency 23–425×, field 14–63×; do not write "30–300×"), so the study does not
   yet probe the regime where label type matters. The clean, robust results are: LF labels on the HF model mesh
   are 2.8–4.3× cheaper and nearly identical to HF; the model's errors against LF and HF labels differ by 0.02 pp.
3. **HF in all experiments** = one p3 solve on the start mesh (the 150k-DOF budget is used by the first solve;
   label mesh = model mesh). The adaptive path and the label/model-mesh split are part of the generator but are
   not exercised by the training experiments. Say this in 05, 06, 07 (and the editor in the abstract).
4. **Cost ratio:** "2.8–4.3×" with conditions (4.3: 8 meshes under concurrent training load; 4.1: 6 test meshes,
   same load; 2.8–3.7: 2 geometries, no load, new container). 06 owns a small conditions table; others quote
   "2.8–4.3×".
5. **Ritz-bound margins:** quote only the measured minimum from the equal-cost study, +0.47 % (min over arms,
   geometries and modes of f_model / f_FE(space) − 1). The values "+0.006 %" and "+0.009 %" from docs/34 §4.1–§4.2
   are a units slip in the doc (fractions, i.e. ≈0.6 % and ≈0.9 %); write "all predicted eigenvalues lay above the
   FE eigenvalue of the evaluated space" for those studies, without a number.
6. **Space-transfer degradation:** 0.2–0.7 percentage points (not 0.2–0.6); the explanation is a hypothesis.
7. **Speed:** one wording everywhere: single thread, model path 8.3 s (operators 5.0 s + forward 3.3 s) vs 18 s for
   assembling and solving the FE problem in the same space (docs/34 §8.2); at these sizes the model is ~2× faster,
   not more; the larger ratios of docs/34 §4.2 (p3, refined mesh) are 2-thread measurements. No speed-advantage
   claim beyond that.
8. **Representation capacity:** call it "single-sample representation capacity" (fit of one 179-tet sample to its
   own p2 label). The N0 2.4e-3 is a space-vs-exact floor on a different sample — either drop the comparison or
   state that it is not like-for-like.
9. **Floor wording:** "the discrete p2 eigenvalue on the model mesh lies at most 0.047 % above the HF frequency on
   the six test geometries (0.033 % on four earlier geometries)".
10. **Fillet floor:** the experiments use a minimum fillet of 0.12 (the generator default for field labels is 0.05).
11. **K:** experiments predict K = 4 modes (labels store more; LF solves compute 6); frequency statistics include
    mode 3, field statistics modes 0–2. Say it where the statistics are defined (07).
12. **LOBPCG warm start** (08): totals including set-up and mode transfer: about 130 s warm vs 149 s cold vs 175 s
    direct (docs/31 §3.4) — a marginal gain on the label solver, not a model result.
13. **Penalty vs projection ablation** (docs/31 §3.5: penalty errors 4e-6, 6.7e-4, 6.7e-2, 6.7 at gradient
    fractions 0, 1e-6, 1e-4, 1e-2; projection stays at 5e-6): 07 adds it (short table or sentence), 04 refers to it.

## Notation (fix collisions in your own section to this table)

| symbol | meaning | owner |
|---|---|---|
| Z_h | discretely divergence-free subspace of V_h | 03 defines, 04 uses |
| 𝖶 | whitening matrix of the Ritz layer only | 04 |
| w_q | quadrature (mass) weight of token q | 04 |
| 𝒮 (\mathcal{S}) | attention score / kernel matrix (was A_v) | 04 |
| 𝖠_V | VᵀKV (Ritz stiffness) | 04 |
| μ_i | divergence-free mass fraction of a basis direction | 04 (μ₀ stays the permeability) |
| α_span, α_ss, α_ortho, α_freq | loss weights | 04, 06 |
| ω_LF, ω_HF | per-sample fidelity weights | 04, 06 (06 must use these) |
| β_LF, β_HF | fidelity frequency scale | 04, 06 |
| m, K | basis width, number of predicted modes | defined in 01 at first use |

Other single letters (𝖰, 𝖡, 𝖳, ρ, θ, s) — avoid reuse inside your section; if two meanings meet, rename the local one.

## Section-specific

- **01:** contribution 1 "discrete"; contribution 3 per decision 8; contribution 5 per decisions 2–4; define m and
  K at first use; decisions 5–7.
- **02:** add, only if verified by search: reduced-basis methods for parametric eigenproblems with output bounds
  (e.g. Machiels, Maday, Oliveira, Patera, Rovas 2000, C. R. Acad. Sci. Paris), reduced-basis for Maxwell /
  cavity eigenproblems if one exists, hard-constraint or divergence-free neural networks / operators, learned
  initial subspaces for iterative eigensolvers, and parametric-regression surrogates for cavity frequencies.
  Keep ≤ 1.75 pages.
- **03:** λ_h also depends on the explicit quadrature rule (degree 2p+4); state the order convention once
  (implementation order p: complete polynomials of degree p, p = 0 Whitney; gradients of H1 order p+1); Z_h.
- **04:** referee M1 (keep (A) explicit; `% TODO pass 3: rank check number`), M2 ("What the proposition does not
  give" paragraph: index-wise bounds, no mode identity, no accuracy bound, discrete not exact), M3 (rewrite the
  inexact-CG remark: with τ_CG = 1e-8 and τ_drop = 1e-6 the bound η ≲ C τ_CG / μ is not small a priori; the
  guarantee is exact-arithmetic; `% TODO pass 3: stress test and a posteriori η̄`), M4 (decision 5), M5 (all
  minor points; replace the "Schur-complement cancellation" attribution by the measured fact: float32 CG does not
  converge within its iteration cap, docs/21 B3), notation table above, decision 13 reference. Keep ≤ 3.8 pages.
- **05:** decisions 3, 10; the analytic validation is on a smooth pillbox (no pipes or fillets); the pipe pillbox's
  Bpk moved 4 % between its two finest levels; move details of families not used by any experiment to A05,
  keeping the list; ≤ 2.5 pages.
- **06:** decisions 2–4 (own the cost-conditions table), notation (ω, β instead of w, s, λ_f).
- **07:** decisions 2, 5, 6, 7, 11, 13; add per-geometry values for every arm to tab:equal-cost (or a second small
  table) and remove bold that suggests significance; one hyperparameter table (model sizes, optimiser, schedule,
  steps, threads, seeds = single run); `% TODO pass 3: baselines and field figure`.
- **08:** decisions 2, 7, 12; add to limitations: no parametric-regression or coarse-FE baseline yet (pass 3 may
  change this), HF = one-level p3, confound in the LF/HF study; ≤ 1.75 pages.

Compile in a private copy as before. Reply with the changes per referee item.
