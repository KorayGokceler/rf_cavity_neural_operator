#!/bin/bash
# One-time setup on a TRUBA login node: build the Apptainer image of the Python stack (rfcav.def) on a `debug`
# compute node and check it with a 2-geometry calibration run. TRUBA forbids conda / pip installs under /arf and
# builds on the login nodes (docs/36 §5), so nothing is installed on the login node and the whole stack ends up in
# ONE file, $CONTAINER (default ~/containers/rfcav.sif, CUDA 12.4 torch, NGSolve, gmsh, scikit-fem, h5py).
#   bash cluster/truba/setup_env.sh                  # build if missing, then check   (~15-25 min, needs internet)
#   FORCE=1 bash cluster/truba/setup_env.sh          # rebuild (after requirements.txt changed)
#   ACCOUNT=<hesap> BUILD_TIME=1:30:00 bash cluster/truba/setup_env.sh
# Alternative without TRUBA compute-node internet: build the same file on any Linux machine with Apptainer
#   apptainer build --fakeroot rfcav.sif cluster/truba/rfcav.def     (from the repo root)
# and copy it to $CONTAINER (rsync -avP over the VPN); then run this script again for the check only.
# The build runs under srun, so keep the terminal open (tmux / screen).
set -euo pipefail
TRUBA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$TRUBA_DIR/config.sh"
FORCE=${FORCE:-0}
BUILD_TIME=${BUILD_TIME:-1:00:00}        # debug queue: 4 h at most
BUILD_CPUS=${BUILD_CPUS:-8}
REQ_SUM_FILE=$CONTAINER.requirements.sha256
mkdir -p "$(dirname "$CONTAINER")"

build() {
  # Apptainer's cache and temp files go to scratch / the node's NVMe /tmp: its default cache is ~/.apptainer (inode quota)
  local cache=${APPTAINER_CACHEDIR:-/arf/scratch/$USER/apptainer_cache}
  mkdir -p "$cache"
  echo "building $CONTAINER on a debug node (torch from $TORCH_INDEX) …"
  srun $(sbatch_common debug) -N 1 -n 1 -c "$BUILD_CPUS" --time="$BUILD_TIME" --job-name=rfcav_build \
    env APPTAINER_CACHEDIR="$cache" APPTAINER_TMPDIR="/tmp/$USER-apptainer" bash -c \
    "mkdir -p /tmp/$USER-apptainer && cd '$REPO_DIR' && '$APPTAINER' build --fakeroot --force \
       --build-arg TORCH_INDEX='$TORCH_INDEX' '$CONTAINER.tmp' cluster/truba/rfcav.def"
  mv "$CONTAINER.tmp" "$CONTAINER"
  sha256sum "$REPO_DIR/requirements.txt" | cut -d' ' -f1 > "$REQ_SUM_FILE"
}

if [ "$FORCE" = 1 ] || [ ! -f "$CONTAINER" ]; then build; else echo "image exists: $CONTAINER (FORCE=1 rebuilds)"; fi

# check on a debug node as well (login nodes take short jobs only, and the NGSolve calibration needs a few cores)
TMP=/tmp/$USER-rfcav-check
srun $(sbatch_common debug) -N 1 -n 1 -c 4 --time=0:20:00 --job-name=rfcav_check bash -c "
  set -e
  export TRUBA_DIR='$TRUBA_DIR'; source \"\$TRUBA_DIR/config.sh\"; container_prefix
  rm -rf '$TMP'; mkdir -p '$TMP'
  # field labels (docs/29): gmsh CAD → netgen curved mesh → NGSolve p3 → p2 projection, read back
  OMP_NUM_THREADS=1 \"\${CX[@]}\" python src/data_gen/dataset_generator_3d.py --mode calibration --n_total 2 \
    --n_workers 1 --mesh_size 0.2 --n_eigen_modes 4 --h5_filename '$TMP/field.h5'
  \"\${CX[@]}\" python -c \"from src.data.field_dataset import FieldDataset; d = FieldDataset('$TMP/field.h5', train_ratio=1.0, val_ratio=0.0); d[0]; print('field data OK')\"
  rm -rf '$TMP'
"
echo "Environment OK: $CONTAINER"
