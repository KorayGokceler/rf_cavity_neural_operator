# 24 · Cavity figures of merit (QoI): Q0, G, R/Q, R_sh, E_pk/E_acc, B_pk/E_acc

> Status: **contract** (branch `claude/3d-cavity-qoi`). §0 is the binding interface shared by
> `src/qoi/operators.py` (numpy, reference), `src/data/*` (labels, batching), `src/qoi/torch_qoi.py`
> (differentiable, training). Sections §1–§5 are filled in by the implementation.

## 0. Contract

### 0.1 Idea

Every CST-style eigenmode post-processing value is a **ratio of quadratic / linear functionals of
the mode field**, evaluated with fixed per-geometry sparse operators:

| quantity | functional of the DOF vector u (normalised mesh) |
|---|---|
| stored energy U | uᵀ M u (volume mass, already in the PKL) |
| wall loss P_c | uᵀ S u (S = tangential-trace "wall loss" matrix on ∂Ω) |
| voltage V_acc | \|Σ_p q_p (A_z u)_p e^{j k_b s ζ_p}\| (axis evaluation matrix A_z, quadrature q_p) |
| E_pk, B_pk | max over surface points of \|E_surf u\|, \|H_surf u\| (row-triplets) |

So QoI are **derived from the predicted (Ritz) field + predicted frequency** — no extra network
head; they are cheap (sparse mat-vecs), differentiable, and consistent with the FE labels because
the same operators are applied to the FE field.

### 0.2 Units, normalisation, conventions

- Mesh: normalised coordinates ξ = (x − center)/scale (PKL `X`); `scale` s [m], `center` [m] (3,).
- DOFs u: N0 edge DOFs on the normalised mesh, arbitrary amplitude and sign (labels have
  uᵀMu = 1). The physical field is defined as F(x) = F̃(ξ) (amplitude is arbitrary; every
  reported value is normalised to **U = 1 J**, like CST).
- Peak-amplitude (standing-wave) convention: U = ½ε0∫|E|² dV = ½μ0∫|H|² dV,
  P_c = ½ R_s ∮|H_t|² dS, Q0 = ωU/P_c.
- **E primary** (field 'E', PEC wall DOFs = 0): H = (j/(ωμ0)) curl E ⇒ |H| = |curl_ξ Ẽ|/(ωμ0 s).
  - U = ½ ε0 s³ · uᵀMu
  - P_c = ½ R_s /(ωμ0)² · uᵀ S u, with S_ij = ∮_∂Ω̃ (n×curl w_i)·(n×curl w_j) dS̃
    (curl of the boundary tet; normalised surface)
  - V = s · |Σ_p q_p Ẽ_z(ξ_p) e^{j ω s ζ_p /(βc)}|
- **H primary** (field 'H'): E = (1/(jωε0)) curl H ⇒ |E| = |curl_ξ H̃|/(ωε0 s).
  - U = ½ μ0 s³ · uᵀMu
  - P_c = ½ R_s s² · uᵀ S u, with S_ij = ∮_∂Ω̃ (n×w_i)·(n×w_j) dS̃
  - V = |Σ_p q_p (curl_ξ H̃)_z(ξ_p) e^{j ω s ζ_p /(βc)}| / (ωε0)
- Beam axis: the physical line x = y = 0 along z (all cavity families are built on z = beam
  axis), i.e. ξ_xy = −center_xy/s. Axis points ζ_p (normalised axial coordinate) with midpoint
  quadrature weights q_p (normalised length); points outside the mesh get q_p = 0.
- R_s: default copper σ = 5.8e7 S/m, R_s(f) = √(π f μ0/σ) at the mode's own frequency (predicted
  frequency for the model, FE frequency for labels); or a fixed `Rs` [Ω] (e.g. SRF Nb).
- β = 1 default. Accelerating length `L_acc` [m]: given, else the axis chord length inside Ω.
- Convention `'linac'` (default): R/Q = V²/(ωU), R_sh = V²/P_c. `'circuit'`: both halved.

### 0.3 numpy API — `src/qoi/operators.py` (exported from `src/qoi/__init__.py`)

```python
build_qoi_operators(X, tets, edges, scale, center, field, M=None, n_axis=401,
                    axis_xy=(0.0, 0.0)) -> dict
    # keys (all for the NORMALISED mesh, numpy / scipy CSR float64):
    'field'   'E' | 'H'
    'scale'   float s [m]
    'M'       CSR [Ne×Ne]   volume mass (assembled if M is None)
    'S'       CSR [Ne×Ne]   wall-loss quadratic form (§0.2)
    'Az'      CSR [P×Ne]    E-primary: Ẽ_z at axis points; H-primary: (curl H̃)_z
    'zeta'    float [P]     axial coordinate of the points (normalised, relative to center)
    'q'       float [P]     quadrature weights (normalised length), 0 outside Ω
    'L_axis'  float         chord length of the axis inside Ω (normalised)
    'Esurf'   CSR [3Nf×Ne]  rows 3i..3i+2 = Cartesian components at surface point i of
                            E-primary: Ẽ;          H-primary: curl H̃
    'Hsurf'   CSR [3Nf×Ne]  E-primary: curl Ẽ;     H-primary: H̃
    'face_area' float [Nf]  (surface points = boundary-face centroids, one per boundary face)

qoi_from_dofs(ops, U, f_hz, Rs=None, sigma=5.8e7, beta=1.0, L_acc=None,
              convention='linac') -> dict[str, ndarray [K]]
    # U [Ne] or [Ne,K]; f_hz scalar or [K]. Keys (U = 1 J normalisation):
    'f_Hz','Rs_ohm','U_J','P_c_W','Q0','G_ohm','V_acc_V','T_transit','R_over_Q_ohm',
    'R_sh_ohm','L_acc_m','E_acc_Vm','E_pk_Vm','B_pk_T','Epk_Eacc','Bpk_Eacc_mT_per_MVm'

cavity_qoi(geom, U, f_ghz, field=None, **kw) -> dict
    # geom = PKL geometry_pool entry (X, tets, edges, scale, center[, M]); field default
    # geom.get('field', 'H'); builds the operators (cache the result yourself if reused).

QOI_LABELS = ('Q0', 'G_ohm', 'R_over_Q_ohm', 'R_sh_ohm', 'T_transit', 'Epk_Eacc',
              'Bpk_Eacc_mT_per_MVm')    # what the PKL stores per sample (§0.4)
```

T_transit = V / (s·Σ_p q_p |Ẽ_z|) (E-primary; analogous for H with curl).

### 0.4 Data contract (converter / dataset / collate)

- PKL `samples[i]['qoi']`: dict {name: float} for `QOI_LABELS`, computed with the FE field and FE
  frequency, copper R_s, β = 1, 'linac'; `metadata['qoi']` = {'labels', 'sigma', 'beta',
  'convention', 'axis': 'x=y=0, z', 'L_acc': 'axis chord'}.
- Dataset item: `Y_qoi` float32 [K_data, n_qoi] (NaN where missing), `ds.qoi_names`.
  With `Maxwell3DDataset(..., qoi_ops=True)` the item also carries `qoi_ops` (dict above; cached
  per geometry).
- Collate (only when every item has `qoi_ops`): batch keys
  - `Y_qoi` [B, K_max, n_qoi] (NaN padding)
  - `QoI_S` block-diagonal COO on the padded edge space [B·Ne_max × B·Ne_max]
  - `QoI_Az` block-diagonal COO [B·P_max × B·Ne_max], `QoI_zeta`, `QoI_q` [B, P_max] (q = 0 pad)
  - `QoI_Esurf`, `QoI_Hsurf` block-diagonal COO [B·3Nf_max × B·Ne_max], `QoI_area` [B, Nf_max],
    `QoI_SurfMask` [B, Nf_max] bool
  - `QoI_scale` [B], `QoI_Laxis` [B] (normalised)
  Same block layout as M (sample b's edge e ↔ row b·Ne_max + e).

### 0.5 torch API — `src/qoi/torch_qoi.py`

```python
qoi_torch(batch, F, f_ghz, Rs=None, sigma=5.8e7, beta=1.0, convention='linac',
          peak_p=None) -> dict[str, Tensor [B,K]]
    # F [B, Ne_max, K] DOFs (any amplitude), f_ghz [B,K]; same keys/values as qoi_from_dofs
    # (peaks exact max when peak_p is None, else a p-norm soft max for gradients).
```
Training term (`GNOTLightning(qoi_weight=0.0, qoi_terms=('Q0','R_over_Q_ohm','G_ohm'))`, off by
default): mean squared log-ratio of predicted vs FE QoI over isolated (non-cluster) modes.
