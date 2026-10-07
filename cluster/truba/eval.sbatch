#!/bin/bash
# Evaluation of $RUN_DIR/$EXP (submitted by submit_eval.sh): in-distribution test split of TRAIN_PKL
# and the OOD families (OOD_PKL, every geometry): fields + frequencies (eval_3d), figures of merit
# (eval_qoi), field plots + ParaView files (plot_3d). Results → $RUN_DIR/$EXP/eval/.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=rfeval
set -euo pipefail
source "$TRUBA_DIR/config.sh"
eval "$ENV_ACTIVATE"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
cd "$REPO_DIR"
CK=$RUN_DIR/$EXP
OUT=$CK/eval
mkdir -p "$OUT"
run_set() {   # name pkl split
  local name=$1 pkl=$2 split=$3
  [ -f "$pkl" ] || { echo "skip $name: no $pkl"; return 0; }
  echo "── $name ($pkl, split $split)"
  python scripts/eval_3d.py --checkpoint "$CK" --data_path "$pkl" --split "$split" --csv "$OUT/${name}_eval.csv" \
    | tee "$OUT/${name}_eval.txt"
  python scripts/eval_qoi.py --checkpoint "$CK" --data_path "$pkl" --split "$split" --deg_rel 1e-3 \
    --csv "$OUT/${name}_qoi.csv" --summary_csv "$OUT/${name}_qoi_summary.csv" | tee "$OUT/${name}_qoi.txt"
  python scripts/plot_3d.py --checkpoint "$CK" --data_path "$pkl" --split "$split" --n 6 \
    --out_dir "$OUT/${name}_viz" > "$OUT/${name}_viz.txt"
}
run_set test "$TRAIN_PKL" test
run_set ood "$OOD_PKL" all
echo "results in $OUT"
