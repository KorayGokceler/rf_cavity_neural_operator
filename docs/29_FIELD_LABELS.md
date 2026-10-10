# 29 — Field labels: CST-grade labels, a cheaper model space, one-line TRUBA runs

> Status: implemented on `claude/field-model` (smoke-tested: generator → H5 → dataset → `train.py
> --fast_dev_run`, TRUBA `--dry-run`). Not yet run at scale.
> Model: docs/28. Label benchmark that motivated this: docs/27.
> Adaptive refinement of the labels and the separate, budgeted model mesh: docs/30.

## 1. Pipeline

One curved mesh per geometry; two polynomial orders on it.

| step | where | what |
|---|---|---|
| geometry | `cavity_shapes` | the generator families, with a fillet floor `MIN_FILLET` (§2.1) |
| CAD | `dataset_generator_3d.mesh_sample` | gmsh OCC solid → BREP. `ridged_box` and `dtl` are OCC-healed first (§2.3) |
| curved mesh | `highorder.netgen_mesh` | netgen at maxh = 3·h, curvaturesafety 2, grading 0.5, curved to order 3, local repair of distorted elements (§2.2) |
| labels | `field_labels.label_sample` | HCurl **p3**: frequencies, fields, QoI computed directly from the p3 fields (`mode_qoi`) |
| model space | `field_labels.project_to_model` | M-orthogonal projection of the p3 fields onto HCurl **p2** of the same mesh (nested spaces, so this is the exact best approximation) |
| storage | `field_labels.write_field_group` | per geometry: netgen `.vol` + BREP of the meshed shape (+ deformation map), `freqs`, `u_model` (float32), `qoi`, `rq_model`, `proj_err` |
| dataset | `FieldDataset` | rebuilds the curved mesh exactly, assembles the p2 operators, checks the stored labels against them (§3.3) |
| model | `CavityLightning(model_type='field3d')` | `EigenspaceOperatorField` (docs/28), same losses and metrics as the N0 model |

Commands:

```bash
python src/data_gen/dataset_generator_3d.py --labels field --families elliptical spoke \
    --n_total 64 --threads 4 --h5_filename data/field/s0.h5
python train.py --config configs/field_3d.yaml --override "dataset.data_path=data/field/*.h5"
bash cluster/truba/run.sh all LABELS=field FAMILIES=train N_PER_FAMILY=1024 EXP=field_base   # TRUBA, §5
```

Generator options for field labels: `--min_fillet` (default 0.05), `--label_order 3`, `--model_order 2`,
`--curve 3`, `--maxh_factor 3`, `--field_max_elements 30000`, `--max_ndof 900000`, `--threads`,
`--no_heal`. The default `--labels n0` reproduces the v2 N0 datasets unchanged (fillet floor 0, no
healing). The discrete `freeform` family has no CAD solid and is skipped for field labels.

## 2. Curved meshes

### 2.1 Fillet floor

A fillet is the rounded edge between two faces (iris tip, nose cone, cell equator). netgen sizes
elements on a curved face as about radius / curvaturesafety, so a 1 mm fillet in a 57 mm cavity
fills the mesh with sub-millimetre tets. `fillet_floor(rho, L, cap)` raises every drawn radius to
`MIN_FILLET · L`, where L is the cavity size: R for the revolved families, min(a, b, d) for
`ridged_box`, Ro for `hwr`, Rt for `dtl`. The radius is never raised above the builder's own
feasible maximum (`cap`). With `MIN_FILLET = 0`, the old geometries are reproduced exactly.

Element count against the floor (pillbox with pipes, R = 57 mm, rp = 17 mm, curvaturesafety 2,
after the local repair of §2.2):

| floor | ρ_iris | tets |
|---|---|---|
| 0.02 | 1.2 mm | 45k |
| 0.04 | 2.3 mm | 30k |
| 0.06 | 3.4 mm | 25k |
| 0.08 | 4.5 mm | 18k |

The count drops more slowly than 1/ρ because the pipe radius and the other fillets also set the
size. The same geometry has 6k tets in the gmsh N0 mesh at 0.10.

### 2.2 Element quality decides the wall quantities

Coarse straight tets that are bent onto a small fillet distort; at worst they fold (det J < 0).
`netgen_mesh` computes min/max det J of every element. It bisects the elements below `fix_below`
and re-curves them (up to 3 passes, about +5 % elements). Only if that fails does it regenerate the
mesh with curvaturesafety doubled (about 3× the elements). Measured p3 labels on the same geometry
(8 % floor), compared with the repaired mesh:

| mesh | tets | min det-J ratio | frequencies | Q0 (TM010) | E_pk, B_pk |
|---|---|---|---|---|---|
| curvaturesafety 1.5 | 9k | 0.010 | 1e-5 | −50 % | 100–1600× off |
| curvaturesafety 2 | 16k | 0.069 | 1e-6 | 2e-6 | 2e-7 (isolated modes) |
| curvaturesafety 2 + repair | 18k | 0.35 | reference | reference | reference |
| curvaturesafety 1 | 4k | < 0 (folded) | — | — | — |

Frequencies are insensitive to element quality, but wall loss and surface peaks are not.
Defaults: curvaturesafety 2, grading 0.5 (0.8 folds elements), accept a mesh at a ratio of 0.05 or
more, repair below 0.1. A geometry whose mesh still fails, or that exceeds `--field_max_elements`,
is re-drawn like a failed boolean (`mesh_sample(post=…)`).

### 2.3 Healing

`gmsh occ.healShapes(sewFaces=False, makeSolids=False)` is applied to `ridged_box` and `dtl` only.
Their box booleans leave tiny edges and sliver faces, and before healing netgen could not mesh them
(docs/27). Sewing or makeSolids deletes the revolved solids. Even the gentle healing breaks the 1D
mesh of the `composite` cells, so healing is restricted to those two families.

### 2.4 Sizes per family

The table uses the 5 % floor, defaults, and 4 Sobol draws per family, without re-draws. "fold"
means the mesh was still folded after repair; the generator re-draws those geometries.

| family | curved tets |
|---|---|
| spoke | 5.2k, 5.2k, 5.5k, 7.8k |
| composite | 0.9k, 2.8k |
| pillbox_pipes | 8.3k, 9.8k, 25k, fold |
| elliptical | 9.2k, 30k, fold, fold |
| hwr | 9.1k, 14k, 27k, fold |
| reentrant | 13k, 28k, 38k, 42k |
| dtl | 15k, 27k, 43k, 47k |
| ridged_box | 38k, 55k, 62k, 89k |

`ridged_box` has fillets along every box edge. With the 30k budget, most of its draws are re-drawn,
which biases that family towards larger fillets, or the sample fails after 6 tries.

## 3. Labels and the model space

### 3.1 Why p3 labels and a p2 model

The pillbox with pipes, 16k curved tets, compared with p3:

| space | DOF | frequency error | best field error of the projected labels | Rayleigh quotient of the projection |
|---|---|---|---|---|
| p1 | 47k | 0.3–1 % | 1–2 % | +3 % |
| p2 | 176k | 1e-5–1.3e-4 | 0.1–0.2 % | +0.05–0.3 % |
| p3 (labels) | 438k | — | — | — |

The model's frequencies are Ritz values in its own space, so they cannot fall below that space's
discrete eigenvalue. p1 leaves about a 1 % frequency floor, no better than N0 (0.25–2.4 %). p2 has
both floors far below what a learned model reaches. The default is therefore `--model_order 2`;
`--model_order 1` is the cheap option.

### 3.2 Projection

u₂ = argmin ‖u₂ − u₃‖_M, computed with the label rule (dx₃) as M₂₂ u₂ = M₂₃ u₃, using NGSolve
sparse Cholesky on the model space (14 s for 176k DOF). `proj_err = √(1 − ‖u₂‖²/‖u₃‖²)`. The
targets are the stored p3 frequencies. The fields are the projections, wall rows exactly 0.

### 3.3 Consistency on load

The .vol + BREP round trip reproduces the curved mesh exactly: same vertex, element and DOF
numbering, with eigenvalues equal to 1e-14. `FieldDataset` compares the Rayleigh quotients of the
stored `u_model` under the rebuilt operators with `rq_model` from generation (rtol 1e-5). It raises
if they differ, for example because of a different curve order, deformation or DOF numbering;
the test permutes the rows. `check_labels` (docs/28) is not used here: the projections are not
eigenvectors of the p2 operators.

## 4. Cost

| | measured | at scale |
|---|---|---|
| label (18k tets, p3 495k DOF, 10 modes) | 140 s on 4 threads, 5.1 GB peak | ≈ 0.15 core-h per geometry |
| storage | `.vol` 2.7 MB (gzip ≈ 1 MB), `u_model` 0.8 MB per mode | ≈ 9 MB per geometry at p2, 10 modes |
| model operators (16k tets, p2) | 11 s to assemble, 0.46 GB (K, M, C dominate) | per item, in the DataLoader workers |
| N0 for comparison | 6k tets, 8k DOF | ~25× lighter per geometry |

On TRUBA (orfoz, 56 cores per job), `THREADS=8` runs 7 samples at a time, using about 35–60 GB.
A 1024-geometry shard takes about 4 h, within the 12 h default. Training with p2 at about 200k DOF
per geometry is the expensive part:

- Batch 2 per GPU.
- 14 loader workers: operator assembly is about 10 s per geometry.
- `CACHE_DIR` keeps the pickled operators, about 0.5 GB per geometry. Use it only when the scratch
  quota allows.

Knobs, from cheapest to most accurate:

| knob | effect |
|---|---|
| `MODEL_ORDER=1` | about 6× cheaper training; frequency floor about 1 % |
| larger `MIN_FILLET` | fewer tets; the geometry space excludes sharper features |
| `FIELD_MAX_ELEMENTS` | caps the cost; re-draws biased to larger features |

## 5. TRUBA in one line, and the UI page

`cluster/truba/run.sh <setup|dataset|train|all|status> KEY=VALUE … [--dry-run]`:

- Every `config.sh` variable can be set as KEY=VALUE, plus:
  - `FAMILIES`: train, ood, all, or a comma list;
  - `N_PER_FAMILY`;
  - `TRAIN_FAMILIES`;
  - `MIX`, the name of the n0 merged PKL.
- It creates the conda environment on first use (`setup_env.sh`, CUDA 12.4 torch, NGSolve, with a
  field-label self-check).
- It submits each family's missing shards, then the training with `--dependency=afterany:<jobs>`:
  - `LABELS=field`: generation → training on the H5 shard directories of the train-group families.
    OOD families are never trained on, and there is no conversion step.
  - `LABELS=n0`: generation → per-family conversion → `merge.sh` → training on `mix_<MIX>.pkl`.
- `--dry-run` prints every sbatch command with its dependencies and submits nothing
  (`dry_sbatch.sh`).
- Re-running is safe: finished shards are skipped, and the same `EXP` resumes from `last.ckpt`.
- The field TAG is `F_p3p2_ms0.10_k10_mf0.05_v1`, separate from the N0 TAG.

The web UI's **TRUBA** tab (`src/service/truba.py`, `/api/truba/config`, `/api/truba/command`)
turns a form into the line to paste on a login node:

```
[ -d ~/rf_cavity_neural_operator ] || git clone <repo> ~/rf_cavity_neural_operator; cd ~/rf_cavity_neural_operator \
  && git fetch -q origin && git checkout -q <branch> && git pull -q --ff-only origin <branch> \
  && bash cluster/truba/run.sh all N_PER_FAMILY=1024 LABELS=field EXP=field_base
```

- Every value is validated: numbers by range; names, times, branches, URLs and paths by pattern,
  with `$USER` and `$HOME` allowed in paths. Values are shell-quoted, and only those that differ
  from `config.sh` are written.
- The page shows the TAG, the output folder, a rough estimate (geometries, core-hours, disk) and
  warnings: model order 1, `ridged_box` cost, no operator cache.
- The command never contains credentials. For a private repo, register an SSH key on TRUBA and use
  the `git@github.com:…` URL.

## 6. Open items

- `scripts/eval_3d.py`, the prediction service and the QoI loss are N0-only. `field3d` checkpoints
  train and test through `train.py`, but `--then-eval` is skipped for them.
- No rotation augmentation for `field3d`: C and QP are built for the stored orientation.
- `field_operators` spends its 11 s in the point features, the scipy conversions and
  `ConvertOperator`; 2–3× faster looks possible.
- Netgen meshing is not bit-reproducible across runs (threaded surface meshing). The stored mesh is
  the record, so the labels are reproducible from the H5, not from the seed.
- The per-family numbers in §2.4 come from 4 draws each. The first TRUBA trial run (`TEST=1`) should
  record the real distribution and the re-draw rate.
