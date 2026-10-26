---
name: measure
description: Measure speed or memory before and after a change the reproducible way - fixed inputs, warm-up, repeated runs, controlled threads, no concurrent load, peak RSS - and record the before/after table in the matching doc. Use before optimising anything (AGENTS.md "measure before optimising") and to back any speed or memory claim.
---
# Measure before and after

## 1. Fix the setup

- **Inputs.** Pin the inputs: geometry family and id, mesh size, DOF budget, K modes, model config, batch.
  Prefer an existing reference case:
  - training step: pillbox_pipes, 13 052 tets, 141 654 DOF, 4 modes (docs/31 §3.2);
  - label sample: test geometry 24 of the multi-fidelity set (docs/34);
  - UI: the docs/26 §2 interactions on a 48 852-tet elliptical.
- **Threads.** Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `torch.set_num_threads(n)` and
  `ngsolve.SetNumThreads(n)` explicitly, and record `nproc`.
- **Load.** Check that the machine is idle (`uptime`, and `ps` for other python/node processes). A measurement
  under load is labelled "under load".
- **Baseline.** Measure the old code in a worktree at the base commit
  (`bash scripts/agent_worktree.sh bench-base <base>`), and measure both versions in the same session.
  Remove the worktree at the end.

## 2. Run

- One warm-up run (imports, JIT, caches, the first NGSolve assembly), then ≥ 5 timed runs. Interleave
  old/new when possible (A B A B …) so drift hits both.
- Time the part being changed, not only the total:
  - `time.perf_counter()` around the code;
  - the generator's own `t_mesh` / `t_solve` / `t_labels` attrs;
  - Lightning step times;
  - in the browser, Playwright `performance.now()` around the interaction, waiting for the next rendered
    frame.
- **Memory:**
  - peak RSS via `/usr/bin/time -v` (Maximum resident set size);
  - `torch.cuda.max_memory_allocated()` on GPU;
  - `tracemalloc` for Python-side growth.

  For the generator, measure per worker: the field_plan model is ≈ 1 GB + 12 GB · max_ndof / 900k.
- **Correctness alongside speed.** Check that the outputs match: frequencies to ~1e-12 relative for a pure
  refactor, the same test results, the same rendered image for UI work.

## 3. Report and record

| part | before | after | change |
|---|---|---|---|
| … | median (min–max) over n runs | … | ×speed-up or % |

- Give the setup line: inputs, threads, hardware (CPU model from `lscpu`, GPU if any), load, commit hashes.
- Report the median and the range, not the best run. With fewer than 5 runs, say so.
- Put the table in the matching doc: docs/26 §2 (UI), docs/29 or docs/30 (labels), docs/31 §3 (training
  step, data preparation, solver, CG). Supersede older numbers explicitly.
- A speed claim without a table like this does not go into docs, commit messages or the paper.
