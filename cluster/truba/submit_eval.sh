#!/bin/bash
# Evaluate a trained run on the test split and the OOD set (GPU, ~minutes):
#   EXP=base cluster/truba/submit_eval.sh          (DEP=afterok:<jobid> to wait for a training job)
set -euo pipefail
TRUBA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TRUBA_DIR
source "$TRUBA_DIR/config.sh"
mkdir -p "$LOG_DIR"
D=(); [ -n "${DEP:-}" ] && D=(--dependency="$DEP")
JID=$(sbatch --parsable $(sbatch_common "$GPU_PARTITION") --gres=gpu:1 -c "$GPU_CPUS" -t 0-06:00:00 ${D[@]+"${D[@]}"} \
      --job-name="eval_$EXP" -o "$LOG_DIR/eval_${EXP}_%j.out" --export=ALL "$TRUBA_DIR/eval.sbatch")
echo "evaluation of $EXP: job $JID → $RUN_DIR/$EXP/eval"
