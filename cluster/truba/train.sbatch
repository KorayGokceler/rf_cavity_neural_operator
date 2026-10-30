#!/bin/bash
# Training of the field3d model on the field-label H5 shards of TRAIN_FAMILIES (submitted by
# submit_train.sh: GPU queue, one task per GPU). Resumes from $RUN_DIR/$EXP/last.ckpt when it exists.
#SBATCH -N 1
#SBATCH --job-name=rftrain
set -euo pipefail
source "$TRUBA_DIR/config.sh"
eval "$ENV_ACTIVATE"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
cd "$REPO_DIR"
CONFIG=${CONFIG:-configs/field_3d.yaml}
DIRS=()                               # docs/29: the H5 shard directories of TRAIN_FAMILIES
for fam in ${TRAIN_FAMILIES:-$(family_list train)}; do
  compgen -G "$H5_DIR/$fam/*.h5" > /dev/null && DIRS+=("\"$H5_DIR/$fam\"")
done
[ ${#DIRS[@]} -gt 0 ] || { echo "no field-label shards under $H5_DIR (run.sh dataset LABELS=field …)"; exit 1; }
DATA=("dataset.data_path=[$(IFS=,; echo "${DIRS[*]}")]")
[ -n "$CACHE_DIR" ] && DATA+=("dataset.cache_dir=$CACHE_DIR")
CKPT=$RUN_DIR/$EXP/last.ckpt
RESUME=(); [ -f "$CKPT" ] && RESUME=(--resume "$CKPT") && echo "resuming from $CKPT"
STRAT=training.strategy=auto; [ "$GPUS" -gt 1 ] && STRAT=training.strategy=ddp
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "config $CONFIG, data: ${DATA[*]}"
srun python train.py --config "$CONFIG" ${RESUME[@]+"${RESUME[@]}"} \
  --override "${DATA[@]}" \
    training.log_dir="$RUN_DIR" training.exp_name="$EXP" training.max_epochs="$EPOCHS" \
    training.batch_size="$BATCH" training.learning_rate="$LR" training.num_workers="$NUM_WORKERS" \
    "$STRAT" $(model_overrides "$MODEL") $TRAIN_EXTRA
