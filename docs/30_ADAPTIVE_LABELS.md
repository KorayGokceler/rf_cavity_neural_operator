# 30 — Adaptive labels: CST-grade frequencies, fields and figures of merit, budgeted model mesh

> Status: implemented on `claude/field-model` (`src/data_gen/adaptive.py`, `field_labels.label_sample`,
> generator `--labels field`). Measured on the analytic copper pillbox and three generator families in a
> 15 GB sandbox. Large-scale references and the per-family cost distribution still have to come from a
> TRUBA trial run.
> Builds on docs/29 (field labels) and docs/28 (field model).

## 1. Target

The labels must have the quality of a converged CST eigenmode run. Errors are measured against a
converged reference (analytic where one exists):

| quantity | target |
|---|---|
| frequency | ≤ 1e-5 (stop criterion 1e-6) |
| Q0, G, R/Q | ≤ 0.2 % (stop criterion 0.1 %) |
| E_pk/E_acc, B_pk/E_acc | ≤ 1 % |

The training cost is bounded separately: the model works on a budgeted mesh (§5).

## 2. Error indicator

All quantities are per element T and summed over the K modes. The modes have unit L2 norm, so
∫|curl E|² = λ. R is averaging interpolation into VectorH1(p):

    η_vol,T²  = Σ_k ( ‖curl E_k − R curl E_k‖²_T + λ_k ‖E_k − R E_k‖²_T ) / λ_k
    η_wall,T² = Σ_k ( ∫_{∂T∩wall} |curl E_k − R curl E_k|² / ∫_wall |curl E_k|²  +  same for E_k )

**Volume term.** Σ η_vol² estimates the relative eigenvalue error.

**Wall term.** The figures of merit are first order in the field error near the wall:
- wall loss ∝ ∫|H_t|², hence Q0 and G;
- the surface peaks.

The frequencies, in contrast, are second order. That is why η_wall is a squared relative wall
error. At the targets both terms are about 1e-6 (f 1e-6 ↔ λ 2e-6; Q0 1e-3 ↔ 1e-6 squared), so
they add with weight 1.

Measured on the analytic copper pillbox (R = 40 mm, L = 50 mm, p3, six modes). The six frequencies
and the TM010 Q0, G, R/Q and peaks are known in closed form.

| refinement | tets | DOF | est. δf/f | f (max of 6) | Q0 | R/Q | E_pk/E_acc | B_pk/E_acc |
|---|---|---|---|---|---|---|---|---|
| uniform | 992 | 28k | — | 1.1e-5 | −1.9e-4 | 1.6e-4 | −3.6e-4 | 5.0e-4 |
| uniform | 7936 | 210k | — | 2.2e-7 | −7.5e-6 | 8.1e-7 | −1.9e-4 | 1.8e-4 |
| adaptive, volume term only | 6668 | 173k | 1.5e-6 | 1.9e-6 | −9.2e-5 | 1.6e-6 | −1.9e-4 | 1.2e-3 |
| adaptive, volume + wall | 5961 | 157k | 2.1e-6 | 6.1e-7 | −3.4e-5 | 1.2e-5 | −2.1e-4 | 3.9e-4 |

- **Wall term.** It cuts the Q0 and B_pk errors by 3× at equal DOF.
- **Estimator.** It tracks the true frequency error within 1.5× at every level:
  1.2e-3 / 1.1e-3, 6.0e-6 / 4.7e-6, 2.1e-6 / 6.1e-7.
- **Pillbox specifics.** The pillbox is smooth, so uniform refinement is near-optimal there.
  E_pk/E_acc stalls at −2e-4: that is the sampling of the maximum (wall integration points), not
  field error.
- **Cost of the targets.** All targets are met from about 22k DOF.

## 3. Geometry: no singular edges

The spoke family had no fillets at all. Its beam pipe and bore junctions are re-entrant edges, with
a vacuum angle of 270°. E and H are singular there. Between refinement levels:
- the surface peaks moved by 20–33 %;
- Q0 moved by 2.4 %.

CST would also return mesh-dependent peaks on such a geometry. Real cavities round these edges for
exactly this reason.

**Detection.** `cavity_shapes.round_reentrant_edges` marks an edge as re-entrant when more than 55 %
of a small circle around its midpoint lies in vacuum. For reference: a box edge gives 0.25, a smooth
or tangent edge 0.5, a 270° edge 0.75.

**Rounding.** All marked edges are filleted in one OCC call. The radius is MIN_FILLET × the
equivalent-sphere radius, halved on failure. This runs before the OCC healing: after healing,
`isInside` reports false re-entrant edges on `ridged_box` (24 per sample, fraction > 0.8).

| family (4 Sobol draws) | edges rounded | left sharp |
|---|---|---|
| reentrant, pillbox_pipes | 0 (already filleted) | 0 |
| spoke | 2–5 | 0 |
| dtl | 2 | 0 |
| elliptical (coupler port) | 0–6 | 0 |
| ridged_box (beam pipes) | 0–2 | 0 |
| composite | 1–4 | 1 sample of 4: 36 |
| hwr (transverse port) | 0–3 | 1 sample of 4: 13 |

When OCC cannot fillet (some BSpline–cylinder unions), the solid is kept as it is. Its labels then:
- store the surface peaks as NaN;
- leave the peaks out of the adaptive stop criterion.

Filleting edge by edge was tried and left a broken topology with more sharp edges than before.
`--no_round_edges` restores the old geometry.

## 4. Refinement: remeshing, not bisection

netgen's bisection of marked elements has a closure that explodes on curvature-graded meshes:

- **The failure.** Marking 200 of 8109 tets (pillbox with pipes) produced a 29k-tet mesh, i.e. 105
  new tets per marked one. This ran the sandbox out of memory twice.
- **The replacement.** Every iteration builds a new mesh from an error-equidistributing size field:

      h_new,T = h_T · (ρ · mean(η²) / η_T²)^(1/(2p)),   ρ = 0.25, clipped to [0.3, 1] · h_T

  (p is the polynomial order). It is passed to netgen as `RestrictH` at the element centroids.
- **No oscillation.** The restrictions accumulate over the iterations; the smallest request wins.
  Rebuilding from the latest field alone oscillates: a region refined at one step is coarsened again
  at the next.
- **Budget.** The predicted size Σ(h/h_new)³ matched netgen within 15 %. When the prediction
  exceeds the DOF budget, the field is scaled up uniformly.
- **Distorted elements.** Curved elements whose det-J ratio falls below 0.05 get a halved local size
  and the shape is remeshed.

The start mesh is coarser than before: curvaturesafety 1.5, falling back to 2 when 1.5 folds
elements (spoke and reentrant draws did). Pillbox with pipes, p3, six modes, compared with the
finest level (15.4k tets, 405k DOF):

| mesh | tets | f | Q0 | R/Q | E_pk | B_pk |
|---|---|---|---|---|---|---|
| old default (cs 2, one solve) | 8.1k | 3.8e-6 | 4.5e-4 | 2.6e-5 | 3.8e-3 | 1.2e-2 |
| adaptive from cs 1.5, level 2 | 9.0k | 4.8e-6 | 3.6e-4 | 2.8e-5 | 1.9e-3 | 2.0e-2 |
| adaptive from cs 1.5, level 3 | 13.8k | 2.1e-6 | 6.6e-5 | 3.4e-6 | 8.0e-4 | 3.9e-2 |

B_pk differs by 4 % between the two finest levels, so the reference itself has not converged for
B_pk at the sandbox's 400k-DOF limit. TRUBA runs with MAX_NDOF = 900k.

Elliptical (that draw had a sharp coupler port), compared with its finest level:

| mesh | tets | f | Q0 | R/Q |
|---|---|---|---|---|
| old default | 9.4k | 1.3e-5 | 1.3e-3 | 7.4e-5 |
| adaptive from cs 1.5 | 8.2k | 7.4e-6 | 5.8e-4 | 1.2e-5 |

Here the adaptive mesh is 2–6× more accurate with fewer elements. Its peaks stayed apart by a
constant 2–3 % because the port edge was still sharp in that run; §3 rounds it now.

## 5. Label mesh and model mesh

- **Label mesh.** The finest adaptive level: frequencies, figures of merit and the p3 fields.
- **Model mesh.** The finest level with at most `model_max_elements` tets (default 10000). The model
  trains on it, at HCurl(model_order = 2).
- **Transfer.** The p3 label fields are L2-projected onto the model space. NGSolve evaluates the
  fine fields at the model mesh's integration points (point search across meshes):

      M₂ u₂ = ∫ v₂ · E₃ dx₃

- **Stored per mode:** proj_err and rq_model. As before, FieldDataset repeats the Rayleigh-quotient
  check on the rebuilt model mesh.
- **When the meshes coincide:** if the start mesh already exceeds the budget, the model mesh is the
  start mesh and the label mesh is finer. If the final mesh fits the budget, both are the same mesh
  and the nested projection of docs/29 is used.
- **Attributes:** `n_tets` (model), `n_tets_label`, `n_tets_start`, `adapt_iters`, `adapt_stop`
  (converged | budget | max_iter), `est_rel_f`, `est_rel_wall`, `adapt_df`, `adapt_dq`,
  `sharp_edges`.

## 6. Defaults and cost

**Generator.** `--labels field` is adaptive by default. Options:
- `--tol_f 1e-6`, `--tol_q 1e-3`;
- `--model_max_elements 10000`;
- `--max_ndof 900000` (label space, about 12 GB at p3);
- `--no_adapt` (one solve, docs/29);
- `--no_round_edges`.

**Workers.** Field labels run one sample per worker process (`maxtasksperchild = 1`). Forking a
process that holds NGSolve / netgen state crashes the child, which is also why the adaptive tests
run after the generator tests.

**TRUBA.** `run.sh … LABELS=field ADAPT=1 TOL_F=… TOL_Q=… MODEL_MAX_ELEMENTS=… MAX_NDOF=…`; the
TRUBA page of the UI has the same fields.

**Cost.** About 3 refinement levels of a growing mesh, i.e. about 3–4× one solve (pillbox with pipes:
25 + 54 + 100 s on 4 threads against 43 s for the old one-solve label). On the training side the
model mesh is now ≤ 10k tets, against 8–90k for the old curvature-sized meshes, which bounds the p2
operators at about 100k DOF per geometry.

**TRUBA budget.**
- Memory: with `THREADS=8`, a 56-core job runs 7 samples at a time, at most about 7 × 12 GB ≈ 85 GB
  at the 900k-DOF budget. Lower `MAX_NDOF` (or raise `THREADS`) on a partition that gives less.
- Time: the cost model gives about 8.5 h for a 1024-geometry shard, against the 12 h `TIME`
  default. The UI warns (`shard_time`) when the estimate passes 80 % of `TIME`. A killed shard
  resumes on the next run, but an `all` run's training (afterany) could start on incomplete data.

## 7. Open items

- **TRUBA trial.** Run per-family convergence references (`TEST=1`, MAX_NDOF 900k), and measure the
  stop reasons and the time and memory per sample.
- **B_pk convergence.** It is the slowest quantity. A peak-targeted indicator, refining the faces
  around the current maximum, is the next lever if the TRUBA references show it is needed.
- **Coarsening.** Coarsening below the start mesh (grow > 1) needs a non-accumulating size field
  with damping.
- **Token subsampling (plan E).** The encoder can use a subset of the quadrature points; the Ritz
  layer keeps the full DOF space.
