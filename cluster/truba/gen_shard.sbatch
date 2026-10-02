#!/bin/bash
# One shard of one geometry family (SLURM array task = shard index).
# Submitted by submit_family.sh (partition / cores / time / array come from config.sh there).
# Env from sbatch --export: TRUBA_DIR, FAMILY, plus anything overriding config.sh.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=rfgen
set -euo pipefail
source "$TRUBA_DIR/config.sh"
family_row "$FAMILY"
eval "$ENV_ACTIVATE"
# one BLAS thread per worker process: the parallelism is the generator's process pool
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

SHARD=${SLURM_ARRAY_TASK_ID:?run as an array job}
START_I=$((SHARD * FAM_SHARD))
N=$((FAM_N - START_I)); [ "$N" -gt "$FAM_SHARD" ] && N=$FAM_SHARD
[ "$N" -gt 0 ] || { echo "shard $SHARD is past n_total=$FAM_N, nothing to do"; exit 0; }
START_ID=$((FAM_BLOCK * ID_BLOCK + START_I))
OUT=$(printf "%s/%s/%s_s%05d.h5" "$H5_DIR" "$FAMILY" "$FAMILY" "$SHARD")
mkdir -p "$(dirname "$OUT")"
if [ -f "$OUT" ]; then echo "exists, skipped: $OUT"; exit 0; fi

echo "$(date -Is) $FAMILY shard $SHARD: ids $START_ID … $((START_ID + N - 1)) on $(hostname), ${SLURM_CPUS_PER_TASK:-$CPUS} workers"
cd "$REPO_DIR"
python src/data_gen/dataset_generator_3d.py \
  --families "$FAMILY" --field "$FIELD" --n_total "$N" --start_id "$START_ID" --seed "$SEED" \
  --n_eigen_modes "$N_STORE" --mesh_size "$MESH_SIZE" --sampling "$SAMPLING" \
  --deform_prob "$DEFORM_PROB" --deform_max "$DEFORM_MAX" \
  --n_workers "${SLURM_CPUS_PER_TASK:-$CPUS}" --sample_timeout "$SAMPLE_TIMEOUT" \
  --h5_filename "$OUT" --resume
echo "$(date -Is) done: $OUT"
