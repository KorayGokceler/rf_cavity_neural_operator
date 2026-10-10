#!/bin/bash
# Mixed training PKL from per-family H5 shards (the per-family PKLs stay as they are).
#   cluster/truba/merge.sh mix9 elliptical:4 reentrant:4 hwr:all …   # first 4 shards / all shards
# → $DATA_ROOT/$TAG/pkl/mix_mix9.pkl
set -euo pipefail
TRUBA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TRUBA_DIR
source "$TRUBA_DIR/config.sh"
MIX_NAME=${1:?usage: merge.sh <name> fam[:n_shards] …}
shift
[ $# -gt 0 ] || { echo "give at least one fam[:n_shards]"; exit 1; }
export MIX_NAME MIX_SPEC="$*"
[ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$LOG_DIR"
D=(); [ -n "${DEP:-}" ] && D=(--dependency="$DEP")       # e.g. afterany:<conversion jobs> (run.sh all)
MJ=$($SBATCH_CMD --parsable $(sbatch_common "$CONVERT_PARTITION") -c "$CONVERT_CPUS" -t "$CONVERT_TIME" ${D[@]+"${D[@]}"} \
  --job-name="merge_$MIX_NAME" -o "$LOG_DIR/merge_${MIX_NAME}_%j.out" --export=ALL "$TRUBA_DIR/merge.sbatch")
echo "merge $MIX_NAME ($MIX_SPEC): job $MJ → $PKL_DIR/mix_$MIX_NAME.pkl"
[ -n "${JOBID_FILE:-}" ] && echo "$MJ" >> "$JOBID_FILE"
exit 0
