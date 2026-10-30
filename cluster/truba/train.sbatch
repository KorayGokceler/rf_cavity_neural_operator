#!/bin/bash
# Training of the field3d model on the field-label H5 shards of TRAIN_FAMILIES (submitted by
# submit_train.sh: GPU queue, one task per GPU). Resumes from $RUN_DIR/$EXP/last.ckpt when it exists.
# Python runs inside the Apptainer image (rfcav.def), --nv gives it the GPUs; srun starts one container per task (DDP).
#SBATCH -N 1
#SBATCH --job-name=rftrain
set -euo pipefail
source "$TRUBA_DIR/config.sh"
command -v module > /dev/null && module purge || true      # no host modules leaking into the container
container_prefix --nv
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
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
srun "${CX[@]}" python train.py --config "$CONFIG" ${RESUME[@]+"${RESUME[@]}"} \
  --override "${DATA[@]}" \
    training.log_dir="$RUN_DIR" training.exp_name="$EXP" training.max_epochs="$EPOCHS" \
    training.batch_size="$BATCH" training.learning_rate="$LR" training.num_workers="$NUM_WORKERS" \
    "$STRAT" $(model_overrides "$MODEL") $TRAIN_EXTRA
