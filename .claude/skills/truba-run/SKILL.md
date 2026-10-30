---
name: truba-run
description: Prepare a TRUBA (SLURM) dataset or training run for the owner - choose LABELS, families, counts, tag and resources from measured costs, dry-run every command, hand over the exact login-node commands, then read the status and logs the owner brings back and propose fixes. Use whenever work moves to the cluster.
---
# Prepare and follow a TRUBA run

The sandbox cannot reach TRUBA. The deliverable is commands that the owner runs on the login node, checked
with `--dry-run` here. The `truba-ops` subagent can do the whole skill in its own context.

Platform facts (queues, GPU access, quotas, storage, modules) are in `docs/36_TRUBA_REFERENCE.md`; check it,
especially `GPU_PARTITION` access and the cores-per-GPU rule, before choosing resources.

## 1. What to run

Agree with the owner on the following:

- **Action:** `dataset`, `train` or `all` (data, then training with a dependency). Use `setup` once per
  account: it builds the Apptainer image (`rfcav.def`) on a debug node, ~15-25 min.
- **LABELS:** `field` (the only label type: curved mesh, p3 labels → p2 model mesh, H5 only); `config.sh` rejects anything else.
- **FAMILIES:** `train`, `ood`, `all`, or a comma list. Also choose N_PER_FAMILY.
  - The default is `cluster/truba/families.tsv`.
  - Blocks are fixed, and shard_size is never changed for existing shards.
- **Label settings:** MESH_SIZE, N_STORE (modes stored), MIN_FILLET, LABEL_ORDER and MODEL_ORDER, the adaptive
  budget, and the tolerances.
  - The TAG follows from these.
  - The UI TRUBA page builds the same command (`src/service/truba.py`).
- **Training:** EXP, MODEL size, EPOCHS, BATCH, LR, TRAIN_FAMILIES.

## 2. Resources

- **Field labels, per worker:**
  - memory ≈ 1 GB + 12 GB · max_ndof / 900k;
  - THREADS NGSolve threads;
  - workers × THREADS ≤ CPUS.
- **Time per sample:** take it from measured numbers (docs/29, docs/34; e.g. HF ~174 s and LF ~43 s per
  pillbox_pipes sample at 150k DOF with 1 thread). Use the slowest family. Then compute
  TIME = 1.5 × samples per worker × time per sample.
- A shard that cannot finish in the partition limit is made smaller (a new family's shard_size), or relies on
  `--resume`.
- **Training:** GPU partition and GPUS. Steps per epoch follow from the number of geometries and the batch.

## 3. Dry-run

```bash
bash cluster/truba/run.sh <action> LABELS=field FAMILIES=... N_PER_FAMILY=... [KEY=VALUE ...] --dry-run
```

- Check the printed sbatch lines: partition, CPUS, memory, TIME, array sizes, `--dependency=afterok:…`,
  output paths under `$DATA_ROOT/$TAG/`, and the generator arguments.
- If a script was edited:
  - run `bash -n` on it and `python -m pytest -q tests/test_service_truba.py`;
  - dry-run `dataset`, `train` and `all` with `LABELS=field`; `LABELS=n0` must fail with a clear message.

## 4. Hand over

- Give the exact commands in order: `setup` if needed, then the run, then
  `bash cluster/truba/run.sh status LABELS=field` to check it.
- Explain in Turkish what will happen: jobs, expected time, and where the data lands.
- Say that re-running is safe: finished shards are skipped, and the same EXP resumes training.
- Never write account names, credentials or hostnames into the repository.

## 5. Follow up

From the `status` output and `slurm-*.out` the owner brings back, report:

- jobs done, failed or pending;
- failed sample ids and their errors;
- timeouts;
- OOM kills;
- the projected finish.

Propose the smallest fix for each: resubmit, more TIME, smaller shards, or fewer workers. Update docs/29 §5 or
the TRUBA README when behaviour changed, and record measured per-sample times in docs/31 §3.
