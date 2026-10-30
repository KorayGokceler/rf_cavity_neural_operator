#!/bin/bash
# Submit a training run (GPU). Every config.sh variable can be set on the command line:
#   LABELS=field cluster/truba/submit_train.sh                     # EXP=base MODEL=base, train-group H5
#   LABELS=field GPUS=4 GPU_CPUS=16 EXP=xl MODEL=xl BATCH=1 cluster/truba/submit_train.sh
#   LABELS=field TRAIN_EXTRA="training.patience=30 model.num_field_modes=8" cluster/truba/submit_train.sh
# Re-running with the same EXP continues from its last.ckpt. Prints the job id.
set -euo pipefail
TRUBA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TRUBA_DIR
source "$TRUBA_DIR/config.sh"
model_overrides "$MODEL" > /dev/null
[ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$LOG_DIR" "$RUN_DIR"
D=(); [ -n "${DEP:-}" ] && D=(--dependency="$DEP")       # run.sh all: wait for the data jobs
TRAIN_FAMILIES=$(family_list "${TRAIN_FAMILIES:-train}")   # train.sbatch reads these families' H5 shard dirs
export TRAIN_FAMILIES
DATA_DESC="field H5 of: $TRAIN_FAMILIES"
JID=$($SBATCH_CMD --parsable $(sbatch_common "$GPU_PARTITION") --gres=gpu:"$GPUS" --ntasks-per-node="$GPUS" \
      -c "$GPU_CPUS" -t "$GPU_TIME" ${D[@]+"${D[@]}"} --job-name="train_$EXP" -o "$LOG_DIR/train_${EXP}_%j.out" \
      --export=ALL "$TRUBA_DIR/train.sbatch")
echo "training $EXP (LABELS=$LABELS, MODEL=$MODEL, $GPUS GPU, data $DATA_DESC): job $JID → $RUN_DIR/$EXP"
if [ -n "${JOBID_FILE:-}" ]; then echo "$JID" >> "$JOBID_FILE"; fi
