#!/bin/bash
# Mixed training PKL from several families' H5 shards: env MIX_NAME, MIX_SPEC = "fam:n_shards …"
# (n_shards = first n shards of that family; "all" = every finished shard). Submitted by merge.sh.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=rfmerge
set -euo pipefail
source "$TRUBA_DIR/config.sh"
eval "$ENV_ACTIVATE"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
shopt -s nullglob
H5S=()
for spec in $MIX_SPEC; do
  fam=${spec%%:*}; n=${spec#*:}; [ "$n" = "$spec" ] && n=all
  files=("$H5_DIR/$fam/${fam}"_s*.h5)
  [ ${#files[@]} -gt 0 ] || { echo "no shards for $fam"; exit 1; }
  [ "$n" = all ] || files=("${files[@]:0:$n}")
  echo "$fam: ${#files[@]} shards"
  H5S+=("${files[@]}")
done
LEAN=(); [ "$PKL_LEAN" = 1 ] && LEAN=(--no_operators)
mkdir -p "$PKL_DIR"
OUT=$PKL_DIR/mix_$MIX_NAME.pkl
cd "$REPO_DIR"
python convert_3d.py --h5_filepath "${H5S[@]}" --output_path "$OUT.tmp" "${LEAN[@]}"
mv "$OUT.tmp" "$OUT"
echo "$(date -Is) wrote $OUT (${#H5S[@]} shards: $MIX_SPEC)"
