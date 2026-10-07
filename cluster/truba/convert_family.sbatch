#!/bin/bash
# All finished shards of one family → $PKL_DIR/<family>.pkl (N0 operators + QoI labels).
# Submitted by submit_family.sh (after the shards) or convert_family.sh by hand.
# Env: TRUBA_DIR, FAMILY.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=rfconv
set -euo pipefail
source "$TRUBA_DIR/config.sh"
family_row "$FAMILY"
eval "$ENV_ACTIVATE"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p "$PKL_DIR"
shopt -s nullglob
H5S=("$H5_DIR/$FAMILY/${FAMILY}"_s*.h5)
[ ${#H5S[@]} -gt 0 ] || { echo "no finished shards in $H5_DIR/$FAMILY"; exit 1; }
EXPECTED=$(( (FAM_N + FAM_SHARD - 1) / FAM_SHARD ))
[ ${#H5S[@]} -eq "$EXPECTED" ] || echo "WARNING: ${#H5S[@]} of $EXPECTED shards finished — converting what exists"
LEAN=(); [ "$PKL_LEAN" = 1 ] && LEAN=(--no_operators)
OUT=$PKL_DIR/$FAMILY.pkl
cd "$REPO_DIR"
python convert_3d.py --h5_filepath "${H5S[@]}" --output_path "$OUT.tmp" ${LEAN[@]+"${LEAN[@]}"}
mv "$OUT.tmp" "$OUT"
echo "$(date -Is) wrote $OUT from ${#H5S[@]} shards"
