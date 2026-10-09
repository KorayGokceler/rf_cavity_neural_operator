#!/bin/bash
# Training of the 3D eigenspace model on TRAIN_PKL (submitted by submit_train.sh: GPU queue, one
# task per GPU). Resumes from $RUN_DIR/$EXP/last.ckpt when it exists.
#SBATCH -N 1
#SBATCH --job-name=rftrain
set -euo pipefail
source "$TRUBA_DIR/config.sh"
eval "$ENV_ACTIVATE"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
cd "$REPO_DIR"
[ -f "$TRAIN_PKL" ] || { echo "no training PKL: $TRAIN_PKL (merge.sh mix_train … or TRAIN_PKL=…)"; exit 1; }
CKPT=$RUN_DIR/$EXP/last.ckpt
RESUME=(); [ -f "$CKPT" ] && RESUME=(--resume "$CKPT") && echo "resuming from $CKPT"
QOI=(); [ "$QOI_WEIGHT" != 0 ] && QOI=(--qoi_weight "$QOI_WEIGHT")
[ "$QOI_METRICS" = 1 ] && QOI+=(--qoi_metrics)
STRAT=training.strategy=auto; [ "$GPUS" -gt 1 ] && STRAT=training.strategy=ddp
nvidia-smi --query-gpu=name,memory.total --format=csv || true
srun python train.py --config configs/eigenspace_3d.yaml ${RESUME[@]+"${RESUME[@]}"} ${QOI[@]+"${QOI[@]}"} \
  --override dataset.data_path="$TRAIN_PKL" dataset.cache_operators=false \
    training.log_dir="$RUN_DIR" training.exp_name="$EXP" training.max_epochs="$EPOCHS" \
    training.batch_size="$BATCH" training.learning_rate="$LR" training.num_workers="$NUM_WORKERS" \
    "$STRAT" $(model_overrides "$MODEL") $TRAIN_EXTRA
