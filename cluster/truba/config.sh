# TRUBA dataset generation settings — sourced by every script in cluster/truba/.
# Edit this file (or export the variables before calling a script) — nothing else needs changing.

# ── paths ──────────────────────────────────────────────────────────────────
# git clone of this repo (default: the clone these scripts live in, else ~/rf_cavity_neural_operator)
if [ -z "${REPO_DIR:-}" ] && [ -n "${TRUBA_DIR:-}" ] && [ -f "$TRUBA_DIR/../../train.py" ]; then
  REPO_DIR=$(cd "$TRUBA_DIR/../.." && pwd)
fi
REPO_DIR=${REPO_DIR:-$HOME/rf_cavity_neural_operator}
DATA_ROOT=${DATA_ROOT:-/arf/scratch/$USER/rfcav3d}           # large files: scratch, not $HOME
# command that activates the Python environment on a compute node (setup_env.sh creates it)
CONDA_HOME=${CONDA_HOME:-$HOME/miniforge3}                 # setup_env.sh installs Miniforge here
ENV_ACTIVATE=${ENV_ACTIVATE:-"source $CONDA_HOME/bin/activate rfcav"}

# ── SLURM (check the current limits: `sinfo`, TRUBA docs "Kuyruk Bilgisi") ─────
PARTITION=${PARTITION:-orfoz}          # orfoz: 112 cores/node, jobs in multiples of 56 cores
CPUS=${CPUS:-56}                       # cores per shard job = generator worker processes
TIME=${TIME:-0-12:00:00}               # wall time per shard job (killed jobs resume, see --resume)
ACCOUNT=${ACCOUNT:-}                   # -A <account> if your project needs one (empty: none)
MAX_PARALLEL=${MAX_PARALLEL:-10}       # shard jobs of ONE family running at the same time
CONVERT_PARTITION=${CONVERT_PARTITION:-$PARTITION}
CONVERT_CPUS=${CONVERT_CPUS:-$CPUS}    # conversion is single-process; the cores give it the node's RAM
CONVERT_TIME=${CONVERT_TIME:-0-12:00:00}

# ── physics / generator (one TAG = one consistent dataset; change TAG when you change these) ─
MESH_SIZE=${MESH_SIZE:-0.10}           # tet size relative to the (per-cell) characteristic length
N_STORE=${N_STORE:-10}                 # eigenmodes stored per geometry
SAMPLING=${SAMPLING:-sobol}
DEFORM_PROB=${DEFORM_PROB:-0.5}
DEFORM_MAX=${DEFORM_MAX:-0.5}
SEED=${SEED:-0}

# ── labels (docs/29) ─────────────────────────────────────────────────────────
# n0:    Whitney N0 labels on the gmsh mesh → per-family PKL → model eigenspace3d (configs/eigenspace_3d.yaml)
# field: NGSolve HCurl p3 labels on a curved netgen mesh, projected onto the model's p2 space → H5
#        only (no conversion) → model field3d (configs/field_3d.yaml)
LABELS=${LABELS:-n0}
LABEL_ORDER=${LABEL_ORDER:-3}          # field: label order (frequencies, QoI)
MODEL_ORDER=${MODEL_ORDER:-2}          # field: model space order (2: ~0.2 % field / 1e-4 freq floor; 1: ~6× cheaper)
MIN_FILLET=${MIN_FILLET:-}             # fillet floor / cavity size (empty: 0 for n0, 0.05 for field)
THREADS=${THREADS:-8}                  # field: NGSolve threads per sample; workers per job = CPUS / THREADS (~5 GB each at 18k tets)
FIELD_MAX_ELEMENTS=${FIELD_MAX_ELEMENTS:-30000}   # field: larger curved meshes are re-drawn
MAXH_FACTOR=${MAXH_FACTOR:-3.0}        # field: netgen maxh = MAXH_FACTOR × N0 mesh size
if [ "$LABELS" = field ]; then
  SAMPLE_TIMEOUT=${SAMPLE_TIMEOUT:-3600}
  TAG=${TAG:-F_p${LABEL_ORDER}p${MODEL_ORDER}_ms${MESH_SIZE}_k${N_STORE}_mf${MIN_FILLET:-0.05}_v1}
else
  SAMPLE_TIMEOUT=${SAMPLE_TIMEOUT:-900}  # s; a stuck mesh/solve is skipped
fi
TAG=${TAG:-E_ms${MESH_SIZE}_k${N_STORE}_v1}
SBATCH_CMD=${SBATCH_CMD:-sbatch}       # run.sh --dry-run swaps in dry_sbatch.sh (prints, submits nothing)
# TEST=1: a small trial run (TEST_N geometries per family, one shard) under <TAG>_test
TEST=${TEST:-0}
TEST_N=${TEST_N:-112}
[ "$TEST" = 1 ] && TAG=${TAG%_test}_test

# ── conversion ─────────────────────────────────────────────────────────────
# 1 (default): --no_operators — M/K/G/Kp are NOT stored, the dataset rebuilds them per item
# (train with dataset.cache_operators: false). Measured at mesh 0.10, 10 modes: full PKL 5–14 MB per
# geometry (~9 × the H5), lean 0.7–2 MB. 0 only for small sets.
PKL_LEAN=${PKL_LEAN:-1}

# ── training / evaluation (GPU; check `sinfo` for the GPU queues and their cores-per-GPU rule) ──
# The Ritz layer and the Kp CG run in float64: pick GPUs with real FP64 (A100 / V100 / H100),
# not consumer cards.
GPU_PARTITION=${GPU_PARTITION:-palamut-cuda}
GPUS=${GPUS:-1}                        # >1: one DDP rank per GPU (srun), training.strategy=ddp
GPU_CPUS=${GPU_CPUS:-16}               # cores per GPU (DataLoader workers rebuild lean operators)
CONFIG=${CONFIG:-}                     # training config (empty: eigenspace_3d.yaml / field_3d.yaml by LABELS)
TRAIN_FAMILIES=${TRAIN_FAMILIES:-}     # field: families whose H5 shards train the model (empty: group "train")
CACHE_DIR=${CACHE_DIR:-}               # field: on-disk operator cache (~0.5 GB per geometry at p2; empty: none)
GPU_TIME=${GPU_TIME:-3-00:00:00}       # killed runs resume from last.ckpt when re-submitted
TRAIN_PKL=${TRAIN_PKL:-}               # default: $PKL_DIR/mix_train.pkl
OOD_PKL=${OOD_PKL:-}                   # default: $PKL_DIR/mix_ood.pkl (merge.sh ood …)
EXP=${EXP:-base}                       # run name → $OUT_DIR/runs/$EXP
MODEL=${MODEL:-base}                   # small 0.9M | base 2.8M | large 6.5M | xl 22M (see model_overrides)
EPOCHS=${EPOCHS:-150}
LR=${LR:-3.0e-4}
if [ "$LABELS" = field ]; then
  BATCH=${BATCH:-2}                    # p2 on ~20k curved tets: ~200k DOF per geometry
  NUM_WORKERS=${NUM_WORKERS:-14}       # the loader workers assemble the p2 operators (~10 s per geometry)
else
  BATCH=${BATCH:-4}
  NUM_WORKERS=${NUM_WORKERS:-6}        # per GPU; each worker sees the whole PKL (fork, copy-on-write)
fi
QOI_WEIGHT=${QOI_WEIGHT:-0}            # >0: + figures-of-merit loss (docs/24 §5)
QOI_METRICS=${QOI_METRICS:-0}          # 1: log qoi_*_rel_err during training (QoI operators per item: slower)
TRAIN_EXTRA=${TRAIN_EXTRA:-}           # more key=value overrides for train.py --override

# model size presets: embed_dim n_heads eigenspace.n_layers n_basis
model_overrides() {
  case "$1" in
    small) echo "model.embed_dim=128 model.n_heads=4 model.eigenspace.n_layers=4 model.n_basis=24" ;;
    base)  echo "model.embed_dim=192 model.n_heads=4 model.eigenspace.n_layers=6 model.n_basis=32" ;;
    large) echo "model.embed_dim=256 model.n_heads=8 model.eigenspace.n_layers=8 model.n_basis=48" ;;
    xl)    echo "model.embed_dim=384 model.n_heads=8 model.eigenspace.n_layers=12 model.n_basis=64" ;;
    *) echo "unknown MODEL '$1' (small|base|large|xl)" >&2; return 1 ;;
  esac
}

# ── ids: geometry id = block · ID_BLOCK + i  (block per family in families.tsv) ─────
# Unique ids across families (shards of different families can be merged into one PKL) and a
# disjoint Sobol range per family. 2^19 = 524288 geometries per family; ids stay < 2^24, exact in
# the float32 Theta column of the PKL as long as block ≤ 31.
ID_BLOCK=524288

# ── derived ────────────────────────────────────────────────────────────────
OUT_DIR=$DATA_ROOT/$TAG
H5_DIR=$OUT_DIR/h5
PKL_DIR=$OUT_DIR/pkl
LOG_DIR=$OUT_DIR/logs
TRUBA_DIR=$REPO_DIR/cluster/truba
FAMILIES_TSV=${FAMILIES_TSV:-$TRUBA_DIR/families.tsv}
RUN_DIR=$OUT_DIR/runs
TRAIN_PKL=${TRAIN_PKL:-$PKL_DIR/mix_train.pkl}
OOD_PKL=${OOD_PKL:-$PKL_DIR/mix_ood.pkl}

# family row from families.tsv → FAM_BLOCK FAM_N FAM_SHARD FAM_GROUP (exit 1 if unknown)
# N_PER_FAMILY (optional): geometries per family instead of families.tsv's n_total
family_row() {
  local row
  row=$(awk -v f="$1" '$1 == f && $1 !~ /^#/ {print $2, $3, $4, $5}' "$FAMILIES_TSV")
  [ -n "$row" ] || { echo "family '$1' not in $FAMILIES_TSV" >&2; return 1; }
  read -r FAM_BLOCK FAM_N FAM_SHARD FAM_GROUP <<< "$row"
  [ -n "${N_PER_FAMILY:-}" ] && FAM_N=$N_PER_FAMILY
  if [ "$TEST" = 1 ]; then FAM_N=$TEST_N; FAM_SHARD=$TEST_N; fi
}

# families of a group (train | ood | all) or a comma list → space separated; field labels need a
# CAD solid (the discrete freeform family is skipped)
family_list() {
  local spec=${1:-train} out
  case "$spec" in
    train|ood|all) out=$(awk -v g="$spec" '$1 !~ /^#/ && NF >= 5 && (g == "all" || $5 == g) {print $1}' "$FAMILIES_TSV") ;;
    *) out=${spec//,/ } ;;
  esac
  [ "$LABELS" = field ] && out=$(for f in $out; do [ "$f" = freeform ] || echo "$f"; done)
  echo $out
}

sbatch_common() {   # common sbatch flags (partition / account)
  echo "-p $1 ${ACCOUNT:+-A $ACCOUNT}"
}
