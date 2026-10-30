#!/bin/bash
# One-line TRUBA entry point — the web UI's TRUBA page writes these commands (docs/29 §5):
#
#   bash cluster/truba/run.sh setup                                   # conda env (once, login node)
#   bash cluster/truba/run.sh dataset LABELS=field FAMILIES=train N_PER_FAMILY=2048
#   bash cluster/truba/run.sh train   LABELS=field EXP=field_base MODEL=base EPOCHS=150
#   bash cluster/truba/run.sh all     LABELS=field FAMILIES=elliptical,spoke EXP=f1   # data, then training
#   bash cluster/truba/run.sh status  LABELS=field
#
# KEY=VALUE: any config.sh variable (PARTITION, CPUS, TIME, ACCOUNT, MESH_SIZE, N_STORE, MIN_FILLET,
# LABEL_ORDER, MODEL_ORDER, THREADS, TAG, GPU_PARTITION, GPUS, MODEL, EPOCHS, BATCH, LR, …) plus
#   FAMILIES        train | ood | all | comma list (default train)
#   N_PER_FAMILY    geometries per family (default: families.tsv)
#   TRAIN_FAMILIES  families the model trains on (default: the dataset's train-group FAMILIES)
# Training (train / all) needs LABELS=field (model field3d); LABELS=n0 only generates N0-label H5.
# --dry-run: print every sbatch command (with its dependencies), submit nothing.
# Re-running is safe: finished shards are skipped, a training run with the same EXP resumes.
set -euo pipefail
TRUBA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TRUBA_DIR
USAGE="usage: run.sh <setup|dataset|train|all|status> [KEY=VALUE …] [--dry-run]"
ACTION=${1:-}
case "$ACTION" in setup|dataset|train|all|status) shift ;; *) echo "$USAGE" >&2; exit 2 ;; esac
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *=*)
      key=${arg%%=*}
      [[ $key =~ ^[A-Z][A-Z0-9_]*$ ]] || { echo "bad variable name '$key' in '$arg'" >&2; exit 2; }
      export "$arg" ;;
    *) echo "unknown argument '$arg'" >&2; echo "$USAGE" >&2; exit 2 ;;
  esac
done
export DRY_RUN
if [ "$DRY_RUN" = 1 ]; then
  export SBATCH_CMD="$TRUBA_DIR/dry_sbatch.sh" DRY_COUNTER
  DRY_COUNTER=$(mktemp)
fi
source "$TRUBA_DIR/config.sh"
FAMILIES=${FAMILIES:-train}

ensure_env() {   # the conda env of setup_env.sh (created on first use; GPU torch for training)
  if [ -x "$CONDA_HOME/bin/conda" ] && "$CONDA_HOME/bin/conda" env list 2>/dev/null | grep -q '^rfcav '; then
    return 0
  fi
  if [ "$DRY_RUN" = 1 ]; then echo "[dry-run] would run: bash $TRUBA_DIR/setup_env.sh"; return 0; fi
  echo "conda env 'rfcav' not found: running setup_env.sh (once, ~10 min)"
  TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu124} bash "$TRUBA_DIR/setup_env.sh"
}

submit_dataset() {   # every family's missing shards
  local fams
  fams=$(family_list "$FAMILIES")
  [ -n "$fams" ] || { echo "no families in '$FAMILIES'" >&2; exit 2; }
  echo "== dataset: LABELS=$LABELS, TAG=$TAG, families: $fams"
  for fam in $fams; do "$TRUBA_DIR/submit_family.sh" "$fam"; done
}

train_families() {   # the train-group families among FAMILIES (OOD families never train the model)
  local f out=""
  for f in $(family_list "$FAMILIES"); do
    [ "$(awk -v f="$f" '$1 == f {print $5}' "$FAMILIES_TSV")" = train ] && out="$out $f"
  done
  [ -n "$out" ] || { echo "no train-group family in FAMILIES='$FAMILIES'" >&2; exit 2; }
  echo $out
}

need_field() {   # the model trains on field labels only (the N0 model eigenspace3d was removed)
  [ "$LABELS" = field ] || { echo "LABELS=$LABELS cannot train: the model trains on field labels (LABELS=field);" \
    "LABELS=n0 generates N0-label H5 only (run.sh dataset)" >&2; exit 2; }
}

dependency() {   # afterany:<every job id written to JOBID_FILE so far> (empty: none)
  local ids
  ids=$(paste -sd: "$JOBID_FILE" 2>/dev/null || true)
  [ -n "$ids" ] && echo "afterany:$ids" || true
}

case "$ACTION" in
  setup)
    if [ "$DRY_RUN" = 1 ]; then echo "[dry-run] would run: bash $TRUBA_DIR/setup_env.sh"; exit 0; fi
    TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu124} bash "$TRUBA_DIR/setup_env.sh" ;;
  status)
    ensure_env
    "$TRUBA_DIR/status.sh" ;;
  dataset)
    ensure_env
    submit_dataset ;;
  train)
    need_field
    ensure_env
    TRAIN_FAMILIES=${TRAIN_FAMILIES:-$(train_families)}; export TRAIN_FAMILIES
    echo "== training: LABELS=$LABELS, EXP=$EXP, MODEL=$MODEL"
    "$TRUBA_DIR/submit_train.sh" ;;
  all)
    need_field
    ensure_env
    JOBID_FILE=$(mktemp); export JOBID_FILE
    submit_dataset
    TRAIN_FAMILIES=${TRAIN_FAMILIES:-$(train_families)}; export TRAIN_FAMILIES
    echo "== training after the data jobs: EXP=$EXP, MODEL=$MODEL"
    DEP=$(dependency) "$TRUBA_DIR/submit_train.sh"
    rm -f "$JOBID_FILE" ;;
esac
[ "$DRY_RUN" = 1 ] && rm -f "$DRY_COUNTER"
T=""; [ "$TEST" = 1 ] && T=" TEST=1"
echo "done ($ACTION). Progress: bash cluster/truba/run.sh status LABELS=$LABELS$T"
