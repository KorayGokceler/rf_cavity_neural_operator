#!/bin/bash
# Generation progress per family + your queued / running jobs.
#   cluster/truba/status.sh          (TEST=1 cluster/truba/status.sh for the trial run)
set -euo pipefail
TRUBA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$TRUBA_DIR/config.sh"
eval "$ENV_ACTIVATE"
T=(); [ "$TEST" = 1 ] && T=(--test --test_n "$TEST_N")
python "$TRUBA_DIR/status.py" --out_dir "$OUT_DIR" --families_tsv "$FAMILIES_TSV" ${T[@]+"${T[@]}"}
command -v squeue > /dev/null && squeue -u "$USER" -o "%.18i %.12P %.20j %.8T %.10M %.6C %R" || true
