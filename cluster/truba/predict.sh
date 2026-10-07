#!/bin/bash
# Pure-model prediction for ONE new cavity (CAD / mesh / generated), as a short GPU job:
#   EXP=base cluster/truba/predict.sh --step $HOME/tesla.step --unit mm --vol_div 9 --fe
#   EXP=base cluster/truba/predict.sh --family elliptical --id 524300 --fe
# Arguments go to scripts/predict_geometry.py; results → $RUN_DIR/$EXP/predict/<name>/.
set -euo pipefail
TRUBA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$TRUBA_DIR/config.sh"
NAME=${NAME:-pred_$(date +%Y%m%d_%H%M%S)}
OUT=$RUN_DIR/$EXP/predict/$NAME
mkdir -p "$OUT" "$LOG_DIR"
srun $(sbatch_common "$GPU_PARTITION") --gres=gpu:1 -c "$GPU_CPUS" -t 0-01:00:00 --job-name="pred_$EXP" \
  bash -c "$ENV_ACTIVATE && cd '$REPO_DIR' && python scripts/predict_geometry.py --checkpoint '$RUN_DIR/$EXP' --out_dir '$OUT' $*"
echo "→ $OUT (prediction.json, modes.vtu)"
