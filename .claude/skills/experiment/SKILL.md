---
name: experiment
description: Run a controlled training / evaluation experiment on the field3d model the reproducible way - question and arms written down first, fixed data split by geometry id, equal-cost or equal-step comparison, seeds, baselines (GP on geometry parameters, coarse FE), and results recorded in docs/34 or docs/35 with the commands. Use for "does X help", fidelity studies, ablations and model-size or data-size sweeps.
---
# Controlled experiment

## 1. Design (write it down before any run; get the owner's OK when it costs hours)

| item | content |
|---|---|
| question | one sentence, e.g. "do LF labels match HF labels at equal label cost?" |
| arms | each arm's data, labels, model size and steps |
| held fixed | geometry family, test geometry ids, split, config, threads, seed set |
| metric | frequency rel. error (mean / median / max over geometries × K modes; K = 4 unless stated), field rel-L2 (modes 0–2, degenerate pairs as subspaces), Ritz margin min(f_model/f_FE − 1), wall time |
| baselines | GP on the geometry parameters (`scripts/paper/check_baseline_regression.py`), p1/p2 FE on the model mesh (`scripts/paper/check_baseline_coarse_fe.py`) |
| decision | what each outcome changes (e.g. "LF ≥ HF → generate LF on TRUBA") |

Watch for confounds. In docs/34 §8, fidelity and the number of geometries changed together. Name every
variable that differs between arms.

## 2. Data

- Generate with the generator's own flags and record them. docs/34 §7 has the reference command lines:
  `--families pillbox_pipes --seed 0 --sampling sobol --deform_prob 0 --min_fillet 0.12`.
  - LF: `--no_adapt --label_order 2 --model_order 2`.
  - HF: `--max_ndof 150000`.
- Split by geometry id with `h5py` `copy` into train / val / test files. Write the id lists into the report.
  Test ids are never in any training file.
- Check the label quality of the test set: frequency tolerance and the adaptive stop reason
  (`adapt_stop` attr). Labels that stopped on budget get a note.

## 3. Runs

- Run `python train.py --config configs/field_3d.yaml --fast_dev_run --override ...` once per arm, then the
  real run in the background with a log file:
  `python train.py ... > <scratch>/<arm>_seed<s>.log 2>&1`.
- Use ≥ 3 seeds before calling a difference real, with `dataset.random_seed=<s>` (default 42).
  - That one seed drives the model init, the shuffling and the train/val/test split inside `data_path`
    (`train.py`).
  - So the test geometries always live in their own file and are evaluated with `eval_fidelity.py`.
    Otherwise a new seed silently changes the test set.
  - With one seed, label the result "single run".
- Record per run: steps or epochs, best val loss, wall time, and the threads (`nproc`, the torch and NGSolve
  threads). Note when other jobs ran at the same time.
- Evaluate each checkpoint the same way:
  `python scripts/eval_fidelity.py --ckpt <best.ckpt> --data <test.h5> --spaces native [p3 refine] --json <arm>.json`.

## 4. Report and record

1. A setup table, then a results table: arms × metrics as mean ± spread over seeds, plus the baselines in the
   same table.
2. What the result shows and does not show. Compare it with the size of the label error: the LF/HF difference
   is ≤ 0.047 % in frequency (docs/34 §8.1). A model error 100× larger cannot resolve label effects.
3. The exact commands, and the split id lists.
4. Add the section to docs/34 (fidelity, data) or docs/35 (paper checks) and a measurement row to docs/31 §3.
   If the paper quotes the affected numbers, update the paper through `docs-writer`.
5. Never commit checkpoints, logs or H5 files.
