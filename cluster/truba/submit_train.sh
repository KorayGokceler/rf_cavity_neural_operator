#!/bin/bash
# Submit a training run (GPU). Every config.sh variable can be set on the command line:
#   cluster/truba/submit_train.sh                                  # EXP=base MODEL=base on mix_train.pkl
#   EXP=large_q MODEL=large QOI_WEIGHT=0.1 cluster/truba/submit_train.sh
#   GPUS=4 GPU_CPUS=16 EXP=xl MODEL=xl BATCH=2 cluster/truba/submit_train.sh
#   TRAIN_EXTRA="training.patience=30 model.num_field_modes=8" cluster/truba/submit_train.sh
# Re-running with the same EXP continues from its last.ckpt. Prints the job id; `--then-eval`
# also submits the evaluation (submit_eval.sh) to run after it.
set -euo pipefail
TRUBA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TRUBA_DIR
source "$TRUBA_DIR/config.sh"
model_overrides "$MODEL" > /dev/null
[ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$LOG_DIR" "$RUN_DIR"
D=(); [ -n "${DEP:-}" ] && D=(--dependency="$DEP")       # run.sh all: wait for the data jobs
if [ "$LABELS" = field ]; then        # train.sbatch reads the H5 shard directories of these families
  TRAIN_FAMILIES=$(family_list "${TRAIN_FAMILIES:-train}")
  export TRAIN_FAMILIES
  DATA_DESC="field H5 of: $TRAIN_FAMILIES"
else
  DATA_DESC=$TRAIN_PKL
fi
JID=$($SBATCH_CMD --parsable $(sbatch_common "$GPU_PARTITION") --gres=gpu:"$GPUS" --ntasks-per-node="$GPUS" \
      -c "$GPU_CPUS" -t "$GPU_TIME" ${D[@]+"${D[@]}"} --job-name="train_$EXP" -o "$LOG_DIR/train_${EXP}_%j.out" \
      --export=ALL "$TRUBA_DIR/train.sbatch")
echo "training $EXP (LABELS=$LABELS, MODEL=$MODEL, $GPUS GPU, data $DATA_DESC): job $JID → $RUN_DIR/$EXP"
[ -n "${JOBID_FILE:-}" ] && echo "$JID" >> "$JOBID_FILE"
if [ "${1:-}" = --then-eval ]; then
  if [ "$LABELS" = field ]; then
    echo "  --then-eval: scripts/eval_3d.py evaluates N0 (eigenspace3d) runs only; skipped for field3d"
  else
    DEP="afterok:$JID" "$TRUBA_DIR/submit_eval.sh"
  fi
fi
