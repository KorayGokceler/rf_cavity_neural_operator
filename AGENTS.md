# AGENTS.md — instructions for every coding agent working on this repository

Read this file first, then `docs/31_SESSION_HANDOFF.md` (current state, measurements, open decisions).
How agents and models split the work (roles, branches, review, templates): `docs/32_AGENT_WORKFLOW.md`.
The owner works in Turkish: reply to them in Turkish; code, comments and commit messages are English.

## What this project is

A neural operator that predicts the resonant modes of 3D RF cavities from the geometry: the K lowest
eigenfrequencies, the E (and H) fields, and the cavity figures of merit (Q0, G, R/Q, R_sh, T,
Epk/Eacc, Bpk/Eacc). Problem: `curl curl E = k² E` in Ω, `n × E = 0` on the PEC wall.

Method (docs/20, docs/28): geometry → trunk (mass-aware linear attention over tokens) → m learned
vector fields ψ_j → fixed linear map to finite-element DOFs V → kernel projection PV = V − G Kp⁻¹GᵀMV
(CG) → Rayleigh–Ritz `VᵀKV c = θ M_div c` → frequencies f = c₀√θ/(2π·scale) and fields PV·c.

One model: `EigenspaceOperatorField` (model_type `field3d`): HCurl p2 on a curved mesh, tokens =
quadrature points (14 per tet), labels = adaptive p3 FE fields projected onto p2. The web UI's Predict,
`scripts/predict_geometry.py` and TRUBA `predict.sh` run it. The old Whitney-N0 edge model
(`eigenspace3d`), its PKL pipeline and the N0 label type were removed: field labels are the generator's only
label type (`--labels` is a hidden, deprecated no-op). The Whitney-N0 display / QoI layer was removed too
(docs/31 §6, item 15): the web UI draws the HCurl p2 field directly (`src/viz/hcurl_display.py`) and the figures
of merit come from the p2 fields (`highorder.mode_qoi`, contract and rescaling in `src/qoi/figures.py`).

## Map

| path | what |
|---|---|
| `src/data_gen/` | geometry families (`cavity_shapes.py`), generator (`dataset_generator_3d.py`), curved meshes + p3 solver (`highorder.py`), adaptive refinement (`adaptive.py`), field labels (`field_labels.py`) |
| `src/data/` | `field_dataset.py` (field3d items, operators rebuilt per item), `sparse.py` (block-diagonal batch operators), `mesh_features.py` (mesh helpers, boundary topology, the model's vertex features) |
| `src/models/` | `hcurl.py` (projection, Ritz, CG `KpSolve`), `eigenspace_operator_field.py`, `layers.py` |
| `src/training/lightning_module.py` | losses (span / frequency / self-supervised / ortho), metrics |
| `src/viz/` | `hcurl_display.py` (p2 field → display mesh, surface, axis; docs/26), `meshsample.py` (point location, cut planes of the input features), `vtk_export.py`, `geometry.py` |
| `src/qoi/` | figures of merit: `figures.py` (keys, conventions, beam axis, rescaling to U = 1 J), `analytic.py` (closed forms; docs/24) |
| `src/service/` | FastAPI backend of the web UI (`api.py`, `model.py`, jobs, datasets, TRUBA, training) |
| `web/` | React + vtk.js UI; `src/fieldlab.ts` computes cuts / 3D grid / phase client-side |
| `cluster/truba/` | SLURM scripts, `run.sh` one-line orchestrator |
| `scripts/` | validation, evaluation, `serve_web.py` |
| `docs/` | numbered design docs; `docs/README.md` is the index |

## Commands

```bash
python -m pytest -q tests/                                    # full suite; fast subsets: test_field_model.py test_service_api.py test_hcurl_display.py
python scripts/validate_field_labels.py --max_ndof 300000     # analytic pillbox check of the label solver (must print PASS)
cd web && npm run build                                       # tsc + vite; the UI must build cleanly
python scripts/serve_web.py --data <dataset root> --checkpoint_optional   # local UI on :8000
python train.py --config configs/field_3d.yaml --override "dataset.data_path=data/field/*.h5"
bash cluster/truba/run.sh all LABELS=field ... --dry-run      # always dry-run TRUBA commands first
```

## Invariants — do not break these

- E formulation only (the H formulation was removed: handle topologies, b1 > 0, create harmonic
  fields the gradient projection cannot remove; branch `legacy-h`).
- The kernel projection is required: gradients have zero curl and would give spurious θ ≈ 0 modes.
  Ritz values must stay upper bounds of the discrete eigenvalues (θ_k ≥ λ_h,k). A cheap diagonal
  penalty instead of the projection was measured and rejected (docs/31 §4).
- Kp CG runs in float64 (float32 does not converge, docs/21 B3).
- `import torch` before `ngsolve` in any process that uses both (segfault otherwise; `tests/conftest.py`).
- Curved meshes repaired by bisection must go through `highorder.single_level` before storage, or the
  stored DOFs do not match the rebuilt space (`field_labels._check_reload` guards this).
- Every form on a curved mesh uses one explicit integration rule (`highorder.volume_measure`).
- Labels: frequencies and QoI come from the p3 label mesh; the training field target is the p2
  projection on the model mesh (≤ `model_max_elements` tets).

## Working rules

- Verify before claiming done: the relevant tests, the analytic check for solver changes, `npm run build`
  and a browser check (Playwright, Chromium is preinstalled in the cloud sandbox) for UI changes.
  Report numbers, not impressions.
- Measure before optimising; keep before/after numbers in the docs.
- Do not change the data generator's behaviour without the owner's go-ahead.
- Never open public tunnels from a sandbox, never disable TLS verification, never commit data or
  checkpoints (`*.h5`, `*.pkl`, `*.ckpt` are ignored).
- One agent per branch or git worktree (`bash scripts/agent_worktree.sh <name>`); do not edit the same
  files in parallel. Claim an open item in docs/31 §6 (owner, branch) before starting it.
- Docs: a change in behaviour updates the matching `docs/NN_*.md`; new topics get the next number and
  an entry in `docs/README.md`.
- At the end of a session, update `docs/31_SESSION_HANDOFF.md` (what changed, numbers, open items) so
  the next agent — any model, any tool — starts from the repository, not from a chat log.
