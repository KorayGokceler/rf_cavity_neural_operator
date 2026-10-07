#!/bin/bash
# One-time setup on a TRUBA login node (needs internet): Miniforge + env "rfcav" (CPU torch by
# default — enough for data generation / conversion; TORCH_INDEX=… for a CUDA build), gmsh from conda-forge (brings its own GL/X libs, the
# PyPI wheel needs system libGLU / libXrender …), the rest from requirements.txt. Ends with a
# 2-geometry generation check (a few seconds, fine on the login node).
#   bash cluster/truba/setup_env.sh
set -euo pipefail
TRUBA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$TRUBA_DIR/config.sh"
CONDA_HOME=${CONDA_HOME:-$HOME/miniforge3}
if [ ! -x "$CONDA_HOME/bin/conda" ]; then
  curl -fsSL -o /tmp/miniforge_$USER.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge_$USER.sh -b -p "$CONDA_HOME"
fi
source "$CONDA_HOME/bin/activate"
conda env list | grep -q '^rfcav ' || conda create -y -n rfcav -c conda-forge python=3.11 gmsh python-gmsh
conda activate rfcav
# CPU torch is enough for generation / conversion; for GPU training use a CUDA wheel, e.g.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash cluster/truba/setup_env.sh
pip install --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}" torch
grep -v '^gmsh' "$REPO_DIR/requirements.txt" > "/tmp/req_rfcav_$USER.txt"
pip install -r "/tmp/req_rfcav_$USER.txt"
python -c "import gmsh, skfem, h5py, scipy, torch; print('gmsh', gmsh.__version__, '| torch', torch.__version__, '| CUDA build', torch.version.cuda)"
cd "$REPO_DIR"
TMP=$(mktemp -d)
OMP_NUM_THREADS=1 python src/data_gen/dataset_generator_3d.py --families elliptical --n_total 2 \
  --n_workers 1 --mesh_size 0.2 --h5_filename "$TMP/check.h5"
python convert_3d.py --h5_filepath "$TMP/check.h5" --output_path "$TMP/check.pkl"
rm -rf "$TMP"
echo "Environment OK. Activate with: $ENV_ACTIVATE"
