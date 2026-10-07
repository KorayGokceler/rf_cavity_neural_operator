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
mkdir -p "$LOG_DIR" "$RUN_DIR"
JID=$(sbatch --parsable $(sbatch_common "$GPU_PARTITION") --gres=gpu:"$GPUS" --ntasks-per-node="$GPUS" \
      -c "$GPU_CPUS" -t "$GPU_TIME" --job-name="train_$EXP" -o "$LOG_DIR/train_${EXP}_%j.out" \
      --export=ALL "$TRUBA_DIR/train.sbatch")
echo "training $EXP (MODEL=$MODEL, $GPUS GPU, data $TRAIN_PKL): job $JID → $RUN_DIR/$EXP"
if [ "${1:-}" = --then-eval ]; then
  DEP="afterok:$JID" "$TRUBA_DIR/submit_eval.sh"
fi
