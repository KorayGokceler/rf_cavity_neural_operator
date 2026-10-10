#!/bin/bash
# Submit one geometry family: an array job over its MISSING shards, then its conversion to
# <family>.pkl (runs after the shards, whatever their exit status — re-run this script to fill
# shards that failed / hit the wall time; finished shards are never redone).
#
#   cluster/truba/submit_family.sh elliptical            # shards + conversion
#   cluster/truba/submit_family.sh elliptical --no-convert
#   cluster/truba/submit_family.sh elliptical --convert-only
#   TEST=1 cluster/truba/submit_family.sh elliptical     # 112-geometry trial under <TAG>_test
#   CPUS=112 TIME=1-00:00:00 cluster/truba/submit_family.sh spoke   # any config.sh variable
set -euo pipefail
TRUBA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TRUBA_DIR
source "$TRUBA_DIR/config.sh"
FAMILY=${1:?usage: submit_family.sh <family> [--no-convert|--convert-only]}
MODE=${2:-}
export FAMILY
family_row "$FAMILY"
[ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$LOG_DIR" "$H5_DIR/$FAMILY"

N_SHARDS=$(( (FAM_N + FAM_SHARD - 1) / FAM_SHARD ))
MISSING=()
for ((s = 0; s < N_SHARDS; s++)); do
  [ -f "$(printf "%s/%s/%s_s%05d.h5" "$H5_DIR" "$FAMILY" "$FAMILY" "$s")" ] || MISSING+=("$s")
done
echo "$FAMILY  (block $FAM_BLOCK, ${FAM_N} geometries, ${N_SHARDS} shards of $FAM_SHARD, group $FAM_GROUP)"
echo "  TAG $TAG → $OUT_DIR"
echo "  shards missing: ${#MISSING[@]}"

# field labels (LABELS=field) are trained from the H5 shards directly: nothing to convert
if [ "$LABELS" = field ]; then
  [ "$MODE" = --convert-only ] && { echo "  LABELS=field: no conversion step"; exit 0; }
  MODE=--no-convert
fi
DEP=()
if [ "$MODE" != --convert-only ] && [ ${#MISSING[@]} -gt 0 ]; then
  ARRAY=$(IFS=,; echo "${MISSING[*]}")
  JID=$($SBATCH_CMD --parsable $(sbatch_common "$PARTITION") -c "$CPUS" -t "$TIME" \
        --array="${ARRAY}%${MAX_PARALLEL}" --job-name="gen_$FAMILY" \
        -o "$LOG_DIR/gen_${FAMILY}_%a_%A.out" --export=ALL "$TRUBA_DIR/gen_shard.sbatch")
  echo "  generation: job $JID (array $ARRAY, ≤ $MAX_PARALLEL at a time)"
  DEP=(--dependency="afterany:$JID")
  [ -n "${JOBID_FILE:-}" ] && echo "$JID" >> "$JOBID_FILE"     # run.sh: later steps wait for these
fi
PKL=$PKL_DIR/$FAMILY.pkl
if [ "$MODE" = "" ] && [ ${#MISSING[@]} -eq 0 ] && [ -f "$PKL" ] && \
   [ -z "$(find "$H5_DIR/$FAMILY" -name "${FAMILY}_s*.h5" -newer "$PKL" | head -1)" ]; then
  echo "  $PKL is up to date (use --convert-only to rebuild it)"
  exit 0
fi
if [ "$MODE" != --no-convert ]; then
  CJ=$($SBATCH_CMD --parsable $(sbatch_common "$CONVERT_PARTITION") -c "$CONVERT_CPUS" -t "$CONVERT_TIME" \
       ${DEP[@]+"${DEP[@]}"} --job-name="conv_$FAMILY" -o "$LOG_DIR/conv_${FAMILY}_%j.out" \
       --export=ALL "$TRUBA_DIR/convert_family.sbatch")
  echo "  conversion: job $CJ → $PKL"
  [ -n "${JOBID_FILE:-}" ] && echo "$CJ" >> "$JOBID_FILE"
fi
