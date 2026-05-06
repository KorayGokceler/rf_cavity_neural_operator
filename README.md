# RF Cavity Neural Operator

A physics-informed neural operator for predicting resonant mode shapes and frequencies of 2D RF cavities. The model learns the mapping from cavity geometry to electromagnetic field distributions by solving the Helmholtz eigenvalue problem — without re-running FEM at inference time.

---

## What It Does

Given an arbitrary 2D RF cavity shape, the model predicts:
- **Field distribution** $E(x, y)$ for the first 3 resonant modes
- **Resonant frequency** $f$ (GHz) for each mode

Inference is ~1000× faster than FEM while maintaining competitive accuracy.

---

## Architecture

The model is a **Geometric Neural Operator Transformer (GNOT)** with three key design choices motivated by the physics:

```
Cavity Geometry (mesh nodes + 8 geometric features)
         │
         ├── Random Fourier Features  →  coordinate encoding φ(x,y)
         │   [Frozen Gaussian kernel, Bochner's theorem]
         │
         └── Input function encoder  →  geometry embedding
                    │
             Mode embedding injection  (monopole / dipole / quadrupole)
                    │
         ┌──────────▼──────────┐
         │   Shared Trunk       │  6 × GNOTBlock
         │   [geometry learning] │     ├─ CrossAttention  (query geometry)
         └──────────┬──────────┘     ├─ Global context injection
                    │                 ├─ LinearAttention   (O(Nd²) not O(N²))
         ┌──────────▼──────────┐     └─ GeometricGatingFFN (dense MoE, 4 experts)
         │  Mode-Specific Heads │
         │  Mode 0 → field + freq
         │  Mode 1 → field + freq
         │  Mode 2 → field + freq
         └─────────────────────┘
```

**~22M parameters.** Key components:

| Component | Description |
|---|---|
| **RFF** | Random Fourier Features encode (x,y) as Gaussian kernel approximation |
| **LinearAttention** | ELU+1 kernel trick — O(Nd²) instead of O(N²) for variable mesh sizes |
| **GeometricGatingFFN** | Dense MoE with 4 experts; position-gated via temperature-0.5 softmax |
| **AttentionPool** | Learned global pooling into a single context vector per geometry |
| **Mode branches** | 3 independent physics branches after shared trunk; each predicts its own frequency |

---

## Data Pipeline

```
FEM Solver  ──►  H5 file  ──►  Feature extraction  ──►  PKL dataset  ──►  Training
(gmsh + skfem)              (geometry features)       (normalized)
```

**Physics:** Helmholtz eigenvalue problem on 2D PEC cavities:

$$\nabla^2 E + k^2 E = 0, \quad E|_{\partial\Omega} = 0$$

Solved with P2 finite elements (6 nodes/triangle). Three geometry types:
- **Sharp**: random polygons (7–12 vertices)
- **Smooth**: random Fourier series perturbations of a disk
- **Calibration**: square, circle, annulus (analytically verifiable)

---

## Quick Start

### Installation

```bash
git clone https://github.com/koraygokceler/rf_cavity_neural_operator
cd rf_cavity_neural_operator
pip install -r requirements.txt
```

### Full Pipeline (Data → Train)

```bash
python run_pipeline.py --config configs/default.yaml
```

Or step by step:

```bash
# Step 1 — Generate FEM dataset (5000 geometries → H5)
python src/data_gen/dataset_generator.py --n_total 5000

# Step 2 — Convert to training format (H5 → PKL with geometric features)
python convert.py --h5_filepath rf_cavity_5000_dataset.h5 --output_path data/gnot_dataset_5k.pkl

# Step 3 — Train
python train.py --config configs/default.yaml
```

### Fast Verification (1 iteration)

```bash
python train.py --config configs/default.yaml --override training.fast_dev_run=true
```

### Inference

```bash
python infer.py --config configs/default.yaml
```

---

## Google Colab

```python
!git clone https://github.com/koraygokceler/rf_cavity_neural_operator
%cd rf_cavity_neural_operator
!pip install -r requirements.txt

# Skip data generation if you already have an H5 file
!python convert.py --h5_filepath rf_cavity_5000_dataset.h5 --output_path data/gnot_dataset_5k.pkl
!python train.py --config configs/default.yaml
```

Set `num_workers: 0` in `configs/default.yaml` if you hit DataLoader errors on Colab.

---

## Configuration

All parameters live in `configs/default.yaml`. Any value can be overridden from CLI:

```bash
python train.py --config configs/default.yaml \
  --override model.embed_dim=128 training.batch_size=64 training.max_epochs=100
```

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `model.embed_dim` | 256 | Hidden dimension |
| `model.n_shared_layers` | 6 | Shared trunk depth |
| `model.n_mode_layers` | 1 | Per-mode branch depth |
| `model.rff_dim` | 64 | Random Fourier Features output dim |
| `model.rff_length_scale` | 0.1 | Gaussian kernel bandwidth |
| `training.learning_rate` | 1e-4 | Adam base LR |
| `training.scheduler` | custom_cosine | 10-epoch warmup + cosine decay |
| `training.freq_weight` | 0.5 | Weight of frequency loss term |
| `training.mode_loss_weights` | [1, 2, 2] | Per-mode loss scaling |
| `training.permutation_invariant_dipole` | true | Swap mode 1/2 if lower loss |
| `dataset.max_nodes` | 1024 | Node sub-sampling cap (VRAM limit) |

Full reference: [`docs/08_CONFIG_REFERENCE.md`](docs/08_CONFIG_REFERENCE.md)

---

## Loss

$$\mathcal{L} = \underbrace{\text{RelL2}(\hat{E}, E) + 0.1\,\|{\hat{E} - E}\|_1}_{\text{field}} + \alpha \underbrace{\|\hat{f} - f\|^2}_{\text{frequency}} + \lambda \underbrace{\|\hat{E}_{\text{bnd}}\|^2}_{\text{boundary}}$$

- **Permutation-invariant dipole:** for near-degenerate mode pairs (modes 1 & 2), both assignment orders are tried per batch; the lower-loss assignment is kept.
- **Sign realignment:** ground-truth sign is flipped if −E is closer to the prediction (eigenfunction sign is arbitrary).
- Default: α=0.5, λ=0.0 (boundary term disabled).

---

## Project Structure

```
rf_cavity_neural_operator/
├── configs/
│   ├── default.yaml           # Main config (all parameters)
│   └── kaggle_2gpu.yaml       # Multi-GPU config
├── src/
│   ├── models/gnot.py         # GNOTModel, RFF, LinearAttention, GeometricGatingFFN
│   ├── data/
│   │   ├── dataset.py         # GNOTDataset, gnot_collate_fn
│   │   └── dataset_converter.py # H5 → PKL feature extraction
│   ├── data_gen/
│   │   └── dataset_generator.py # FEM solver (gmsh + skfem)
│   ├── training/
│   │   ├── lightning_module.py  # Loss, optimizer, scheduler
│   │   └── callbacks.py         # Field visualization callback
│   └── config.py              # Config loader with CLI override support
├── tests/                     # 74 unit tests (pytest)
├── docs/                      # Architecture & physics documentation
├── train.py                   # Training entry point
├── infer.py                   # Inference entry point
├── convert.py                 # H5 → PKL conversion
└── run_pipeline.py            # Full pipeline orchestrator
```

---

## Tests

```bash
pip install pytest
python -m pytest tests/ --tb=short
```

74 tests covering: RFF kernel properties, dataset splits (no geometry leakage), loss correctness (sign realignment, permutation invariance), model forward shapes, config system, and end-to-end integration.

---

## Documentation

Detailed docs in [`docs/`](docs/):

| Doc | Contents |
|---|---|
| [`01_DATA_GENERATION.md`](docs/01_DATA_GENERATION.md) | FEM solver, geometry types, mesh strategy |
| [`02_FEATURE_ENGINEERING.md`](docs/02_FEATURE_ENGINEERING.md) | 8 geometric input features |
| [`03_DATASET_LOADER.md`](docs/03_DATASET_LOADER.md) | Dataset class, geometry-based splitting |
| [`04_MODEL_ARCHITECTURE.md`](docs/04_MODEL_ARCHITECTURE.md) | Full architecture with math |
| [`05_TRAINING_SYSTEM.md`](docs/05_TRAINING_SYSTEM.md) | Loss, scheduler, mode handling |
| [`08_CONFIG_REFERENCE.md`](docs/08_CONFIG_REFERENCE.md) | All config parameters |
| [`09_PHYSICS_BACKGROUND.md`](docs/09_PHYSICS_BACKGROUND.md) | Helmholtz equation, FEM details |

---

## References

- [GNOT: A General Neural Operator Transformer for Operator Learning](https://arxiv.org/abs/2302.14376) — Hao et al., 2023
- [Random Features for Large-Scale Kernel Machines](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) — Rahimi & Recht, 2007
