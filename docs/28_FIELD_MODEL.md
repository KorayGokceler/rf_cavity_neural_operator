# 28 — Option B: learned-field model on a high-order H(curl) space

> Status: implemented (`claude/field-model`). The model, the operators and the batches work, and
> single-sample overfit converges. The data pipeline (generator labels, storage, dataset, Lightning
> `field3d`, TRUBA) and the curved-mesh size control are in docs/29.

## 1. Why

docs/27 §3.4: on the training mesh, the best possible N0 field still has a 12–21% E error. A model that
outputs N0 edge DOFs therefore cannot match CST fields, however good its labels are. Option B
keeps the trunk and the Ritz layer but makes the learned basis live in HCurl(p) (p = 2 or 3) on a
curved mesh.

## 2. Design

The N0 model already evaluates m learned vector fields at one point per edge (the midpoint):
V[e, j] = ψ_j(x_mid)·t_e. The field model evaluates the same ψ_j at the nq quadrature points of every
curved element and turns them into HCurl(p) DOFs with a fixed linear map:

    ψ_j(x_q) → per element: L2(p) coefficients  c = P_t ψ      (|J|-weighted least squares)
             → HCurl(p) DOFs  V = C · c                          (NGSolve ConvertOperator:
                                                                   projection-based interpolation)

- **Tokens:** the trunk's tokens are the quadrature points themselves. Their mass weights are the
  quadrature weights, so the attention integrals are the element quadrature. Their features are
  computed on the curved geometry (`highorder.point_features`): distance and direction to dense wall
  samples, point volume, and a torsion function from an H1(p+1) solve. Vertex features fail on
  coarse curved meshes, which can have no interior vertex at all (torsion 0/0).
- **Ritz:** `hcurl_ritz` is unchanged; it only needs K, M, G, Kp:
  - G = gradients of the interior H1(p+1) DOFs, plus one potential per extra wall shell;
  - Kp = GᵀMG formed exactly.
- **Symmetries:** ψ is a geometric field, so the DOFs transform correctly under mesh renumbering.
  Wall DOFs are set to zero (n × E = 0).
- **Weight transfer:** the edge head has the same shape as in the N0 model (embed ‖ RFF → 3m), so an
  N0 checkpoint can initialise it.

Code:
- `src/data_gen/highorder.py` — `field_operators`, `point_features`, `volume_measure`;
- `src/data/field_dataset.py` — `field_item`, `field_collate`;
- `src/models/eigenspace_operator_field.py`;
- `tests/test_field_model.py`.

Numerical note: on curved elements every form must use one explicit integration rule
(`volume_measure`). NGSolve otherwise picks each form's rule from its integrand, and
curl·curl − σ·u·v assembled as one form then differs from K − σM by 3e-4. That biased the
shift-invert eigenvalues by 4e-5. Fixed: the eigenvalues now match a dense solve to 1e-13.

## 3. Single-sample overfit (curved pillbox, 179 tets, span loss, 64-dim 2-layer trunk, m = 12, K = 4)

| version | p | DOF | steps | λ error (Ritz / label − 1) |
|---|---|---|---|---|
| vertex tokens + barycentric interpolation | 2 | 2280 | 2000 | 0.8–1.1e-4 |
| vertex tokens + barycentric interpolation | 3 | 5480 | 2000 | 2.3–3.0e-4 |
| **quadrature-point tokens** | 2 | 2280 | 1250 | **1e-6 – 9e-6** |

For comparison, the N0 model's frequency floor on the same kind of cavities is 2.4e-3 (docs/27).

## 4. Cost: mesh size is the critical issue

pillbox_pipes #0 (the current N0 label mesh has 10k tets / 13.5k edges):

| curved mesh | tets | p | DOF | potentials | tokens | nnz (K, M, G, Kp, C) | operator build (1 thread) |
|---|---|---|---|---|---|---|---|
| netgen, curvaturesafety 2 | 10.5k | 2 | 112k | 41k | 148k | 25M | 7–10 s |
| netgen, curvaturesafety 2 | 10.5k | 3 | 282k | 100k | 253k | 105M | 37 s |
| gmsh order 2, 6 elements per 2π of curvature | 8k | 2 | 83k | — | — | — | — |

- The element count is set by the **small fillets**, not by `maxh`. For example, `rho_iris` goes down
  to 0.05·rp, about 0.6% of the cavity radius.
- Both meshers resolve these fillets with elements of ~R/2. With less refinement, the coarse curved
  elements fold (docs/27 §3.5).
- At p2 a sample is therefore ~10× the N0 size; p3 is impractical for training.
- **Options:**
  1. Raise the generator's minimum fillet radius, e.g. ≥ 2% of the cavity size or ≥ 1.5 h.
     Expected: several times fewer elements, and B at p2 lands near today's N0 sizes.
  2. Accept p2 at ~10× cost: about 2–4 s per training step on an A100 and ~40 h for 150 epochs over
     1000 samples.
- **Storage:** netgen .vol + BREP reload + Curve reproduces the curved mesh exactly (eigenvalues
  1e-14) in 0.13 s, from 1.4 MB per sample. Operators are rebuilt per item, because storing them
  would take ~0.4 GB per sample at p2.

## 5. Remaining work

1. Generator: a `--labels field --order p` mode writing (.vol, BREP, deformation map, DOF
   eigenvectors, frequencies) to H5.
2. Dataset: rebuild `field_operators` per item (LRU cache).
3. Lightning: a `model_type: field3d` switch, and DOF masks instead of `EdgeMask` in the metrics.
4. QoI from high-order fields: a torch version of `highorder.mode_qoi`.
5. Size-controlled valid curved meshes for every family (geometry cleanup: §4, docs/27 §3.5).
