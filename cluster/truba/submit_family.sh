#!/bin/bash
# Submit one geometry family: an array job over its MISSING shards (re-run this script to fill
# shards that failed / hit the wall time; finished shards are never redone).  The model trains on
# the H5 shards directly (LABELS=field): there is no conversion step.
#
#   cluster/truba/submit_family.sh elliptical
#   TEST=1 cluster/truba/submit_family.sh elliptical     # 112-geometry trial under <TAG>_test
#   CPUS=112 TIME=1-00:00:00 cluster/truba/submit_family.sh spoke   # any config.sh variable
set -euo pipefail
TRUBA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TRUBA_DIR
source "$TRUBA_DIR/config.sh"
FAMILY=${1:?usage: submit_family.sh <family>}
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

if [ ${#MISSING[@]} -gt 0 ]; then
  ARRAY=$(IFS=,; echo "${MISSING[*]}")
  JID=$($SBATCH_CMD --parsable $(sbatch_common "$PARTITION") -c "$CPUS" -t "$TIME" \
        --array="${ARRAY}%${MAX_PARALLEL}" --job-name="gen_$FAMILY" \
        -o "$LOG_DIR/gen_${FAMILY}_%a_%A.out" --export=ALL "$TRUBA_DIR/gen_shard.sbatch")
  echo "  generation: job $JID (array $ARRAY, ≤ $MAX_PARALLEL at a time)"
  if [ -n "${JOBID_FILE:-}" ]; then echo "$JID" >> "$JOBID_FILE"; fi     # run.sh: later steps wait for these
fi
