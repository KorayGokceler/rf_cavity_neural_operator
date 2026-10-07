# 24 · Cavity figures of merit (QoI): Q0, G, R/Q, R_sh, E_pk/E_acc, B_pk/E_acc

> Status: **implemented** (branch `claude/3d-cavity-qoi`). §0 is the binding interface shared by
> `src/qoi/operators.py` (numpy, reference), `src/data/*` (labels, batching), `src/qoi/torch_qoi.py`
> (differentiable, training). §1–§3 physics, discretisation, validation; §4 data & evaluation; §5 training.

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
- Beam axis: the physical line x = y = 0 along z (the cavity families are built on z = beam
  axis), i.e. ξ_xy = −center_xy/s. **Exception hwr**: its coaxial line runs along z and the beam
  crosses it along x at mid-length, so `qoi_operators_of` (labels, dataset, eval, prediction) uses
  `beam_axis(shape_type, …)` → axis_dir 'x' through (0, 0, z_mid). General axes: `axis_dir` /
  `axis_point` of `build_qoi_operators`. Shapes without a beam (ridged_box, composite, freeform)
  keep the nominal z line: their "R/Q" is a defined functional, not an accelerator figure. Axis points ζ_p (normalised axial coordinate) with midpoint
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

Implementation notes (clarifications of the above, no change of names / keys / shapes):
- A 1-D `U` gives arrays of shape `[1]` (K = 1), never 0-d; `float(v[0])` per value.
- `Esurf` / `Hsurf` hold only the components the exact PEC wall field has — the normal part
  n nᵀE of E and the tangential part (I − n nᵀ)H of H, n the face normal — so ‖row triplet‖ is the
  wall |E| / |H|. E is the value of the face's own tet at the centroid, H the nodally recovered
  (vertex-averaged) field interpolated to the centroid (§2.4; `surface_operators(..., method=)`
  gives the other variants). Still one point per boundary face, rows 3i..3i+2.
- `edges` may be in any row order (PKL sorted order or skfem DOF order); `M` may be a CSR
  matrix or the PKL CSR tuple `(indptr, indices, data)`. V = 0 (no E_z on the axis) gives
  E_acc = 0 and `inf` peak ratios, T = NaN only if E_z ≡ 0.
- Extra public helpers: `surface_resistance(f, sigma)`, `QOI_KEYS` (the 16 output keys),
  `SIGMA_CU`; closed forms + reference meshes in `src/qoi/analytic.py`.

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

## 1. Physics & definitions

### 1.1 Stored energy, wall loss, Q0 and G

Time-harmonic fields with **peak** (not RMS) phasor amplitudes, e^{jωt}, PEC walls for the eigenmode
and a small surface resistance R_s for the losses (perturbation: the lossless field is used):

- stored energy (time-averaged electric + magnetic, equal for an eigenmode)
  U = ¼ε0∫|E|² + ¼μ0∫|H|² = ½ε0∫|E|² dV = ½μ0∫|H|² dV;
- wall loss P_c = ½ R_s ∮|H_t|² dS (H_t = n × H, the surface current density);
- unloaded quality factor Q0 = ωU / P_c; geometry factor G = Q0·R_s = ωμ0∫|H|²dV / ∮|H_t|²dS —
  independent of the wall material and of the size (scale invariant), so the natural shape label.

**R_s model.** Normal-conducting skin effect R_s = √(ωμ0/(2σ)) = √(π f μ0/σ), evaluated at the mode's
own frequency (FE frequency for labels, predicted frequency for the model) with σ_Cu = 5.8·10⁷ S/m;
`Rs=` overrides it with a fixed value (e.g. SRF niobium, R_s ≈ 10 nΩ at 1.3 GHz / 2 K). Q0 ∝ 1/R_s,
G, R/Q and the peak ratios are independent of R_s.

### 1.2 E or H as the primary field (N0 DOFs on the normalised mesh)

The physical field is F(x) = F̃(ξ), ξ = (x − center)/s, so ∇_x = ∇_ξ/s, dV = s³dṼ, dS = s²dS̃.
Maxwell: curl E = −jωμ0 H, curl H = jωε0 E.

- **E primary** (PKL field 'E'): H = (j/(ωμ0)) curl_x E ⇒ |H| = |curl_ξ Ẽ|/(ωμ0 s).
  U = ½ε0 s³ uᵀMu; P_c = ½R_s s² ∮|n×curl_ξẼ|²/(ωμ0 s)² dS̃ = ½R_s/(ωμ0)² uᵀSu.
- **H primary** (field 'H'): E = curl_x H/(jωε0) ⇒ |E| = |curl_ξ H̃|/(ωε0 s).
  U = ½μ0 s³ uᵀMu; P_c = ½R_s s² uᵀSu.

U is taken from the primary field only (uᵀMu). For a discrete eigenpair this equals the energy of
the derived field exactly: e.g. E primary, ½μ0∫|H|² = ½μ0 s³ uᵀKu/(ωμ0 s)² = ½ s uᵀKu/(ω²μ0) and
ω² = c²λ/s² with λ = uᵀKu/uᵀMu ⇒ = ½ε0 s³ uᵀMu (same for H primary). For a predicted (Ritz) field
the two differ by the Rayleigh-quotient mismatch of the predicted frequency — the reason the
frequency enters the ratios explicitly. Every value is reported for **U = 1 J** (amplitude
1/√U_raw), like CST; the amplitude and sign of u drop out.

### 1.3 Accelerating voltage, transit-time factor, E_acc

V_acc = |∫ E_z(0, 0, z) e^{jωz/(βc)} dz| on the beam axis (physical x = y = 0, every family in
`cavity_shapes.py` is built around it). The modulus makes V independent of the time origin and of
the axial origin (ζ is measured from the volume centroid). Transit-time factor
T = V_acc / ∫|E_z| dz (= the classical sin x/x for a constant on-axis field; ∫|E_z| rather than
|∫E_z| keeps T ∈ [0, 1] also for modes whose E_z changes sign). E_acc = V_acc / L_acc with
L_acc = the chord of the axis inside Ω by default (pillbox: its length; with beam pipes the pipes
count — pass `L_acc` explicitly for a CST-like N·βλ/2 definition). β = 1 default.

### 1.4 R/Q and shunt impedance — linac vs circuit

`'linac'` (default; the accelerator convention):
R/Q = V²/(ωU), R_sh = V²/P_c = (R/Q)·Q0. `'circuit'` (RF-engineering convention, parallel-RLC
P = V²/(2R)): both halved. Q0, G, T and the peak ratios do not depend on the convention.

### 1.5 Peak surface fields

E_pk = max_∂Ω |E|, B_pk = μ0 max_∂Ω |H| (on a PEC wall E is normal and H tangential). Reported as
E_pk/E_acc (dimensionless) and B_pk/E_acc in mT/(MV/m) (= 10⁹·B_pk[T]/E_acc[V/m]).

### 1.6 Closed-form references (`src/qoi/analytic.py`, derivations in its docstring)

Pillbox TM010, radius R, length L (j01 = 2.40483, J1(j01) = 0.51915, η = √(μ0/ε0)):
k = j01/R; G = η j01 L/(2(R + L)); Q0 = G/R_s; T = sin(kL/2β)/(kL/2β);
R/Q = 2ηLT²/(π j01 R J1²(j01)); E_pk/E_acc = 1/T (end-cap centre);
B_pk/E_acc = J1(j'11)/(cT) = 0.58187/(cT) (end caps at r = 0.766 R).
(Check: kL = π ⇒ R/Q = 195.9 Ω, the textbook value.)

Box a × b × d (x, y, z), fundamental with E ∥ z = beam axis (d < a, b; "TE101" in waveguide
naming, TM110 w.r.t. z): k² = (π/a)² + (π/b)²; Q0 = k³ηabd/(4R_s[abk²/2 + π²(ad/b² + bd/a²)])
(= Pozar's TE101 formula, test-checked); T = sin(kd/2β)/(kd/2β); R/Q = 8ηdT²/(kab);
E_pk/E_acc = 1/T; B_pk = πE0/(ω min(a, b)) ⇒ B_pk/E_acc = π/(kc·min(a, b)·T).

## 2. Discretisation (N0 / Whitney, `src/qoi/operators.py`)

Basis w_e = λ_a∇λ_b − λ_b∇λ_a (a < b global), curl w_e = 2∇λ_a×∇λ_b constant per tet; DOF lookup
via `nedelec.tet_dofs` (any edge row order). Everything is vectorised over tets / faces / points
(no Python loops over elements); build time ≈ 0.3 s at 20k edges, 0.8 s at 54k edges.

### 2.1 M
The closed-form N0 mass of `dataset_converter_3d.assemble_n0` (passed through when the PKL has it).

### 2.2 Wall-loss form S
Boundary faces = faces of exactly one tet (same rows/order as `boundary_faces`), each with its tet,
the opposite local vertex, unit outward normal n and area A. With P = I − nnᵀ ((n×a)·(n×b) = a·Pb):

- **E primary**: S_ij = Σ_f A_f c_i·P c_j with c_i the constant curl of the tet's local basis
  function i — exact for the discrete field. (n·curl Ẽ = 0 on a wall face anyway: it is the surface
  curl of the zero tangential trace.)
- **H primary**: n × w_i is linear on the face; with ∫_F λ_kλ_l dS = A(1 + δ_kl)/12 for face
  vertices (0 for the opposite one), S_ij = A Σ (∫λλ)(∇λ·P∇λ) — the face analogue of the closed-form
  mass matrix, exact.

Accuracy (§3): H primary Q0/G converge fast (pillbox ≤ 0.05 % from Ne = 7.7k). E primary is fine on
flat walls (box +0.1 %) but **first order on curved walls**: pillbox Q0 −6.4 % → −2.9 % for
Ne 3k → 54k (≈ h¹). A per-region breakdown (h = 0.014) shows the end caps exact and the excess
loss on the faceted side wall (+11.6 % local |H_t|²) and the rim tets (+8.8 %); the H formulation
on the same mesh is within 0.2 % there. This is consistent with the polyhedral wall: n × E = 0 on
two facets meeting at a slightly convex crease forces E → 0 along every crease, a boundary layer
that the one-tet-thick constant curl turns into an O(h/R) overestimate of |H_t|. Nodal recovery of
curl Ẽ does not help (+4–5 %). The **consistent boundary flux** does: by Green's identity
g_e = ((λM − K)u)_e on the wall edges is the moment ∮(n × curl Ẽ)·w_e, and ∮|n×curl Ẽ|² ≈ gᵀM_Γ⁻¹g
(M_Γ = the H-primary S restricted to the wall edges); it gives Q0 −1.1 % → −0.15 % (column
"resid. flux" in §3). It needs λ and a sparse solve with M_Γ (lumping M_Γ destroys it: +4…+20 %),
so it is not a fixed sparse S and is **not** in the §0 contract; it is implemented in
`scripts/qoi_validation.py:residual_flux_q0` as a candidate follow-up for E-formulation labels.
For training consistency the bias is benign (labels and predictions go through the same S), but
E-primary Q0/G labels of curved cavities carry a few-% mesh bias at dataset resolution.

### 2.3 Beam axis
(General axis d through a point p: the same along the mesh extent of X·d, A rows = basis · d; the
default below is d = z.) n_axis = 401 midpoints of equal cells over the mesh z-extent on (ξx, ξy) = (axis_xy − center_xy)/s,
located with `nedelec.locate`; q_p = Δζ inside, 0 outside, L_axis = Σq. A_z row p: the z-component of
the Whitney basis at the point (E primary, piecewise linear along the axis) or of the constant curl
of the containing tet (H primary). The quadrature error (O(Δζ²) per smooth piece, ≤ Δζ per wall
crossing in L_axis) is negligible against the field error; for the pillbox / box the end walls sit
on the grid ends, so L_axis is exact. V, T, R/Q converge like the field (≈ h¹…h², §3).

### 2.4 Surface points and peak evaluation
One surface point per boundary face (its centroid; rows 3i..3i+2 of Esurf/Hsurf). Two evaluations
were compared for each of E and H (table "surface-peak methods" in §3):

- *centroid*: the value of the face's own tet (Whitney value with λ_opp = 0, or the constant curl);
- *recovered*: nodal (volume-weighted) average over every tet sharing each boundary vertex, as
  CST-like field viewers do, linearly interpolated to the centroid.

Findings, consistent over both cavities, both formulations and all meshes:

- **E: centroid wins** (pillbox |err| ≤ 0.6 %, box ≤ 0.9 % at Ne ≈ 3k; recovered −1.8…−3.3 %). The
  peak of E sits on a broad maximum (end-cap centre); smoothing has a systematic negative O(h²)
  bias there, while the normal component at a wall face is accurate (H primary: n·curl H̃ is the
  face flux, single-valued across the face; E primary: the normal Whitney component).
- **H: recovered wins** (pillbox ≤ 0.8 %, box ≤ 2 % at Ne ≈ 3k, → 0.1 % / 0.06 % at 54k; centroid
  +1.5…+5.7 %). The piecewise-constant curl (E primary) and the lowest-order tangential trace
  (H primary) carry O(h) element-to-element noise and the max over thousands of faces picks the
  positive excursions; nodal averaging removes the noise.

Default (`SURFACE_METHOD = ('centroid', 'recovered')`): E from the face's own tet, H recovered.
Both are reduced to the exact wall components (n nᵀE, (I − nnᵀ)H); this changes the peaks by
< 0.01 % on these cavities (for E primary + centroid it is an identity), but keeps them physical
on coarse / curved meshes. Peaks at sharp re-entrant edges are field singularities and do not
converge with any method (CST included) — the families round their irises. The max is not
differentiable at ties; `qoi_torch(peak_p=…)` provides a p-norm soft max.

## 3. Validation

`python scripts/qoi_validation.py --md out.md` (gmsh OCC meshes as in the generator, generator E / H
eigensolvers, converter normalisation; lowest mode; copper, β = 1, 'linac'). Relative error vs the
closed forms of §1.6. Total run ≈ 3 min on one core. "resid. flux": E-primary Q0 with the
consistent boundary flux of §2.2 (not the contract S).

**Pillbox** R = L = 0.1 m — analytic f = 1.14743 GHz, Q0 = 25629, G = 226.49 Ω, R/Q = 222.75 Ω,
T = 0.7759, Epk/Eacc = 1.2889, Bpk/Eacc = 2.5016 mT/(MV/m).

| field | h [m] | Ne | f | Q0 | G | R/Q | T | Epk/Eacc | Bpk/Eacc | Q0 (E, resid. flux) |
|---|---|---|---|---|---|---|---|---|---|---|
| E | 0.0200 | 3062 | -0.36% | -6.44% | -6.61% | -1.98% | +0.13% | +0.57% | -0.40% | -1.12% |
| E | 0.0140 | 7671 | -0.19% | -5.03% | -5.12% | -0.66% | +0.13% | -0.46% | -0.28% | -0.57% |
| E | 0.0100 | 19669 | -0.09% | -3.91% | -3.95% | -0.44% | +0.06% | -0.15% | -0.13% | -0.38% |
| E | 0.0070 | 54318 | -0.04% | -2.85% | -2.87% | -0.19% | +0.03% | -0.10% | -0.09% | -0.15% |
| H | 0.0200 | 3062 | +0.76% | -0.04% | +0.34% | -0.41% | -0.37% | +0.22% | +0.13% | – |
| H | 0.0140 | 7671 | +0.38% | -0.01% | +0.18% | -0.32% | -0.19% | -0.10% | +0.75% | – |
| H | 0.0100 | 19669 | +0.19% | +0.01% | +0.11% | -0.36% | -0.11% | +0.20% | +0.77% | – |
| H | 0.0070 | 54318 | +0.09% | +0.00% | +0.05% | -0.11% | -0.05% | +0.03% | +0.38% | – |

**Box** 0.10 × 0.08 × 0.06 m (E ∥ z) — analytic f = 2.39951 GHz, Q0 = 18664, G = 238.52 Ω,
R/Q = 196.71 Ω, T = 0.6615, Epk/Eacc = 1.5116, Bpk/Eacc = 3.9373 mT/(MV/m). (gmsh gives the same box
mesh for every h ≳ 0.014, hence the finer sizes.)

| field | h [m] | Ne | f | Q0 | G | R/Q | T | Epk/Eacc | Bpk/Eacc | Q0 (E, resid. flux) |
|---|---|---|---|---|---|---|---|---|---|---|
| E | 0.0100 | 3712 | -0.30% | +0.44% | +0.29% | -1.14% | +0.51% | -0.94% | -2.05% | -1.61% |
| E | 0.0070 | 10332 | -0.11% | +0.26% | +0.21% | -0.54% | +0.16% | -0.31% | -0.53% | -0.88% |
| E | 0.0050 | 23969 | -0.06% | +0.13% | +0.10% | -0.55% | +0.08% | -0.15% | -0.21% | -0.45% |
| E | 0.0038 | 55691 | -0.03% | +0.11% | +0.09% | -0.24% | +0.03% | -0.05% | -0.06% | -0.25% |
| H | 0.0100 | 3712 | +0.08% | +0.72% | +0.76% | -1.23% | -0.11% | +0.13% | -0.31% | – |
| H | 0.0070 | 10332 | +0.01% | +0.47% | +0.47% | -0.65% | -0.06% | +0.32% | -0.04% | – |
| H | 0.0050 | 23969 | +0.01% | +0.24% | +0.24% | -0.09% | +0.01% | +0.08% | -0.18% | – |
| H | 0.0038 | 55691 | -0.01% | +0.17% | +0.16% | -0.18% | +0.01% | +0.04% | -0.03% | – |

Surface-peak methods (relative error of Epk/Eacc for "E …", of Bpk/Eacc for "B …"; the default is
E centroid + B recovered, projected):

| case | h [m] | Ne | E centroid | E recovered | B centroid | B recovered | B recovered, no proj. |
|---|---|---|---|---|---|---|---|
| pillbox E | 0.0200 | 3062 | +0.57% | -1.77% | +5.72% | -0.40% | -0.40% |
| pillbox E | 0.0140 | 7671 | -0.46% | -1.86% | +3.64% | -0.28% | -0.28% |
| pillbox E | 0.0100 | 19669 | -0.15% | -0.61% | +3.17% | -0.13% | -0.13% |
| pillbox E | 0.0070 | 54318 | -0.10% | -0.38% | +1.56% | -0.09% | -0.09% |
| pillbox H | 0.0200 | 3062 | +0.22% | -3.03% | +3.83% | +0.13% | +0.13% |
| pillbox H | 0.0140 | 7671 | -0.10% | -1.94% | +3.35% | +0.75% | +0.75% |
| pillbox H | 0.0100 | 19669 | +0.20% | -0.71% | +2.67% | +0.77% | +0.77% |
| pillbox H | 0.0070 | 54318 | +0.03% | -0.39% | +1.81% | +0.38% | +0.38% |
| box E | 0.0100 | 3712 | -0.94% | -2.71% | +1.53% | -2.05% | -2.05% |
| box E | 0.0070 | 10332 | -0.31% | -1.14% | +1.31% | -0.53% | -0.53% |
| box E | 0.0050 | 23969 | -0.15% | -0.77% | +0.77% | -0.21% | -0.21% |
| box E | 0.0038 | 55691 | -0.05% | -0.28% | +0.42% | -0.06% | -0.06% |
| box H | 0.0100 | 3712 | +0.13% | -3.29% | +1.00% | -0.31% | -0.31% |
| box H | 0.0070 | 10332 | +0.32% | -1.35% | +0.63% | -0.04% | -0.04% |
| box H | 0.0050 | 23969 | +0.08% | -0.93% | +0.23% | -0.18% | -0.18% |
| box H | 0.0038 | 55691 | +0.04% | -0.43% | +0.21% | -0.03% | -0.03% |

Summary. Frequencies converge ≈ h² (E from below, H from above; ≤ 0.1 % at Ne ≈ 20–50k). At
dataset-like resolution (Ne ≈ 3–20k) R/Q, T and the peak ratios are within ≈ 0.2–2 % for both
formulations, H-primary Q0/G within 0.1–0.8 %, E-primary Q0/G within 0.5 % on flat walls but
−4…−6 % on the curved pillbox wall (first order, §2.2). Errors below ~0.1 % are at the level of
mesh-to-mesh noise (gmsh meshes are not nested), so not every column decreases monotonically.

Tests (`tests/test_qoi.py`, ≈ 8 s): closed forms (textbook R/Q, Pozar Q), exactness of S / A_z /
surface operators for constant and rotational fields on a jittered box, FE vs analytic for pillbox
and box in both formulations (h = 0.02 / 0.01), E vs H agreement, sign/amplitude invariance, U = 1 J
(independent axis integral, direct E_pk), linac = 2 × circuit, R_s and L_acc overrides, β, vectorised
[Ne, K] = per-mode loop, edge-order independence.

## 4. Data & evaluation

> **Deviations from §0.4 (Agent 2):**
> 1. `Y_qoi` is collated whenever any item carries it (every `Maxwell3DDataset` item does — NaN
>    when the PKL has no labels), not only when every item has `qoi_ops`. Superset of the contract.
> 2. All `QoI_*` dense tensors (`QoI_zeta`, `QoI_q`, `QoI_area`, `QoI_scale`, `QoI_Laxis`) are
>    **float64** (like the sparse operators and `M`); `QoI_area` is padded with 0.
> 3. The `QoI_*` sparse operators are torch **COO** (coalesced, float64) as §0.4 says, while the
>    existing `M, K, G, Gt, Kp` stay CSR. `hcurl.spmm` / `torch.sparse.mm` accept both.
> 4. Extra (non-contract) `metadata['qoi']['n_failed']` = geometries whose labels failed (NaN).


### 4.1 Labels in the PKL (`src/data/dataset_converter_3d.py`, `convert_3d.py`)

- `RFCavity3DConverter.convert_dataset(..., compute_qoi=True)` (CLI default; `--no_qoi` turns it
  off) builds the QoI operators **once per geometry** —
  `build_qoi_operators(X, tets, edges, scale, center, field, M=M)` on the stored normalised mesh
  (float32 `X` as stored, the converter's float64 `M`) — and evaluates `qoi_from_dofs` on the
  **stored** float32 DOFs `Y` at the FE frequency, copper (σ = 5.8e7 S/m, R_s(f)), β = 1, `'linac'`
  (`QOI_LABEL_SETTINGS`). This happens before `--no_operators` drops M/K/G/Kp, so lean PKLs get
  the same labels (tested bit-identical).
- `samples[i]['qoi'] = {name: float for name in QOI_LABELS}`;
  `metadata['qoi'] = {'labels', 'sigma', 'beta', 'convention', 'axis': 'x=y=0, z',
  'L_acc': 'axis chord', 'n_failed'}`. E and H PKLs, multi-shard conversion and `--modes` subsets
  all work (labels are computed for exactly the stored modes).
- A geometry whose QoI evaluation raises gets NaN labels and a warning (the conversion
  continues); counted in `n_failed`. `src.qoi` missing → the converter fails early (use
  `--no_qoi`).
- Back-fill an existing PKL (e.g. the Drive data converted before this branch):
  `python convert_3d.py --add_qoi data.pkl [--output_path out.pkl]` (in place by default, atomic
  rename) = `dataset_converter_3d.attach_qoi_labels(data)`; lean PKLs: M is rebuilt from
  (X, tets). Not needed for evaluation (`eval_qoi.py` recomputes the truth), only for label-based
  training.
- Cost: the smoke run (12 E cavities, Ne 4–24k) converted in 6.6 s total including QoI labels.

### 4.2 Dataset items (`src/data/dataset_3d.py`)

- `item['Y_qoi']` float32 [K_data, n_qoi], columns `ds.qoi_names` (= `metadata['qoi']['labels']`,
  else `src.qoi.QOI_LABELS`, else a copy of the contract tuple when `src.qoi` is not importable),
  rows in the item's mode order (ascending frequency, like `Y_field`). NaN where a sample has no
  label (old PKLs: all NaN; `ds.has_qoi` tells).
- `Maxwell3DDataset(..., qoi_ops=True)`: `item['qoi_ops']` = the §0.3 dict of the **stored**
  (un-augmented) mesh, M = the dataset's (stored or rebuilt) M. Built lazily on first access,
  cached per geometry when `cache_operators=True` (the default; ≈ a few MB / geometry at mesh 0.10
  per DataLoader worker — set `cache_operators=False` for very large train sets).
  Rotation augmentation does not invalidate them: N0 DOFs are invariant under an orthogonal map of
  mesh + field, so the operators of the stored mesh still apply (the beam axis rotates with the
  cavity). `ds.qoi_operators(g_id)` is the public accessor.
- `item_from_geometry(g, ..., qoi_ops=True)` (label-free inference) adds `qoi_ops` too (no
  `Y_qoi`).

### 4.3 Batch keys (`maxwell3d_collate`)

| key | shape / type | padding |
|---|---|---|
| `Y_qoi` | float32 [B, K_max, n_qoi] — whenever items carry it | NaN |
| `QoI_S` | COO float64 [B·Ne_max × B·Ne_max] block-diagonal | empty rows/cols |
| `QoI_Az` | COO float64 [B·P_max × B·Ne_max] | empty |
| `QoI_zeta`, `QoI_q` | float64 [B, P_max] | 0 (q = 0 ⇒ no contribution) |
| `QoI_Esurf`, `QoI_Hsurf` | COO float64 [B·3Nf_max × B·Ne_max], rows b·3Nf_max + 3i + c | empty |
| `QoI_area` | float64 [B, Nf_max] | 0 |
| `QoI_SurfMask` | bool [B, Nf_max] | False |
| `QoI_scale`, `QoI_Laxis` | float64 [B] | — |

The `QoI_*` keys appear only when **every** item has `qoi_ops`; a batch's `qoi_ops` must share the
batch field. Same block layout as M (sample b's edge e ↔ column b·Ne_max + e), so
`spmm(batch['QoI_S'], F)` etc. serve the whole batch; tested against per-item scipy products and,
with the real operators, `qoi_torch(batch, …)` == per-geometry `qoi_from_dofs` (rtol 1e-6, E and H).

### 4.4 `scripts/eval_qoi.py`

```
python scripts/eval_qoi.py --checkpoint CKPT_OR_DIR --data_path data.pkl [--split test|val|train|all]
    [--csv qoi.csv] [--summary_csv qoi_summary.csv] [--max_geoms N] [--rq_floor 0.01]
    [--sigma 5.8e7 | --Rs 1e-8] [--beta 1] [--convention linac|circuit] [--n_axis 401]
    [--deg_rel 1e-3] [--device cpu] [--verbose]
```

Per geometry: `src/viz/predict.py: predict()` (aligned Ritz fields, `f_pred`, `f_true`, clusters,
split), one `build_qoi_operators` call (timed: `t_ops_s`), then `qoi_from_dofs` on the predicted
field at `f_pred` (timed: `t_qoi_s`, all K modes) and on the FE field at `f_true` — same operators.
Python API: `evaluate_qoi(lm, ds, ..., deg_rel=None) → rows`, `summarize_qoi(rows, rq_floor) →
DataFrame`, `geometry_rows(ops, predict_out, labels)`, `rel_gap_clusters`, `regroup`.

**Degenerate modes.** A mode inside a near-degenerate cluster (size > 1, `lm._clusters_3d`) or of a
pair split by the last output gets NaN relative errors (`degenerate = True`): its individual value
depends on the arbitrary rotation inside the eigenspace (both for FE and for the Ritz output; the
aligned prediction there is the projection of the target onto the predicted cluster span, i.e.
oracle-rotated). Their `_true` / `_pred` values are still in the CSV.

By default the clusters are the model's own (`near_deg_rel_threshold` of the checkpoint, 2 % in
configs/eigenspace_3d.yaml — the same rule as eval_3d / predict). Multicell elliptical passbands
(modes ~0.2 % apart, π mode included) then count as clusters and are excluded. `--deg_rel r`
replaces the rule with "relative gap of the FE frequencies < r" over all stored modes + `freq_next`
(so a pair cut by the last output is still flagged); the modes this releases are evaluated with
their **raw** Ritz column (QoI are amplitude/sign invariant), not the oracle-rotated projection.
Caveat: at coarse meshes the mesh-induced splitting of true dipole pairs (~1e-4–3e-3) overlaps the
passband spacing, so no threshold separates the two cleanly.

**Accelerating mode** (`accel = True`, one per geometry) = the non-degenerate output with the
largest **true** R/Q. `accel_all` marks the argmax over all outputs; the printout counts the
geometries where it lies inside a cluster (excluded).

CSV — one row per (geometry, output mode): `geom_id, shape_type, field, n_edges, mode, f_true_GHz,
f_pred_GHz, f_rel_err, rel_l2, degenerate, accel, accel_all, rq_frac` (true R/Q / the geometry's accelerating
R/Q), for each q in QOI_LABELS `q_true, q_pred, q_rel` (signed (pred − true)/|true|, NaN if
degenerate) and `q_label` (stored PKL label, NaN if none or other settings), `label_rel_diff`
(max over q of |label − true|/|true|), `t_ops_s, t_qoi_s`.

Summary (printed, `--summary_csv`): index (group = `all` and every shape_type, qoi = `f` + QOI_LABELS),
columns `n_modes, modes_median, modes_mean` (|rel err| over isolated modes) and `n_accel,
accel_median, accel_mean` (accelerating mode only). In the `modes` selection the voltage-based QoI
(R/Q, R_sh, T, Epk/Eacc, Bpk/Eacc) count only modes with `rq_frac ≥ --rq_floor` (default 1 %):
non-accelerating modes have V ≈ 0 and their ratios are meaningless; Q0, G, f count every isolated
mode. The printout also gives the median QoI post-processing time per geometry and the label check
(stored labels vs recomputed truth, flagged above 1e-4 rel; only when the settings equal the label
settings).

Notebook: `Colab_Maxwell3D.ipynb` cell "12b. Kavite değerleri" runs it on the test split and shows
the `all` summary and the accelerating-mode table.

## 5. Training & model

### 5.1 Why the QoI come from the field and not from a regression head

The figures of merit are fixed functionals of (field, frequency) (§0.1). The
EigenspaceOperator3D already predicts both: the Ritz field `out['field']` (unit M-norm, sign
arbitrary) and the frequency `out['freq']`. So `src/qoi/torch_qoi.qoi_torch` evaluates the QoI
from those outputs with the same per-geometry sparse operators that produced the FE labels. A
separate MLP head that regresses Q0, R/Q, ... was not used, because:

- **Consistent by construction.** Q0 = G/R_s, R_sh = (R/Q)·Q0, Epk/Eacc = E_pk/(V/L) all hold
  exactly for the predicted values. Independent heads would contradict each other and the
  predicted field.
- **No new failure modes.** Q0 and G do not depend on amplitude, and every quantity is invariant
  to the sign of u (|V|, uᵀSu, uᵀMu, max|·|). The arbitrary Ritz sign and normalisation therefore
  need no alignment.
- **Accuracy follows the field.** A field with M-norm error ε gives errors of O(ε²) in the
  quadratic ratios (Q0, G) and O(ε) in the voltage and peak values. A good field therefore gives
  good QoI for free, and the QoI loss only reweights *which* field errors matter (wall tangential
  field, on-axis E_z).
- **Nothing extra to store or learn.** It works for any mesh, β, R_s model (copper σ or a fixed
  SRF R_s) and convention at evaluation time (`qoi_torch(..., Rs=, beta=, convention=)`). These
  are things a regression head would have to be retrained for.

### 5.2 Loss

`GNOTLightning(qoi_weight=0.0, qoi_terms=('Q0','R_over_Q_ohm','G_ohm'), qoi_peak_p=None,
qoi_rq_floor=1e-2)`, used only by `model_type='eigenspace3d'`. It is **off by default**: with
`qoi_weight = 0` the loss is bit-identical to before.

For a batch that carries the QoI operators (`QoI_*` keys, §0.4):

    f̂ = freq_stats-denormalised out['freq'] [GHz]        f = de-normalised Y_freq[:, :K]
    q̂ = qoi_torch(batch, out['field'], f̂, peak_p)         (with gradient)
    q  = qoi_torch(batch, Y_field[..., :K], f, peak_p)     (no grad, same operators)
    L_qoi = mean_{t ∈ qoi_terms}  mean_{(b,k) valid for t} (log q̂_t − log q_t)²
    total += qoi_weight · L_qoi

- The references q are recomputed on the fly from the FE targets with the FE frequency. The
  stored `Y_qoi` labels are **not** used in the loss, so the loss stays consistent with
  `qoi_peak_p`, any R_s / β choice and the float32 targets, and it works for PKLs that have no
  labels.
- The log-ratio makes the loss scale-free across quantities that span many decades (Q0 ~ 10⁴,
  R_sh ~ 10⁶ Ω, T ~ 0.5). Each term is O(rel. error²).
- R_s(f) is evaluated at the **predicted** frequency for q̂, so frequency errors propagate into
  Q0 = ωU/P_c (∝ √f for copper), into the transit phase and into R/Q. Gradients reach both the
  field and the frequency.
- Peaks: by default the loss uses the exact max (gradient through the arg-max point). Setting
  `qoi_peak_p = p` uses the p-norm (Σ_i |x_i|^p)^{1/p} for E_pk / B_pk in **both** q̂ and q. This
  spreads the gradient over the near-peak wall points and is useful when `Epk_Eacc` or
  `Bpk_Eacc_mT_per_MVm` is a term. The metrics always use the exact max.
- Each term is a mean over its own valid entries; a term with no valid entries contributes 0.
  The loss is the mean over terms, so `qoi_weight` does not depend on how many terms are chosen.

### 5.3 Masking rules (loss and metrics)

An entry (sample b, output k) counts only if:

1. **Isolated mode.** k is a singleton cluster of `_clusters_3d` (same near-degeneracy rule and
   thresholds as the `mode_k_rel_l2` metric: all stored frequencies + `FreqNext`). Inside a
   near-degenerate cluster the per-mode QoI depend on the arbitrary rotation within the
   eigenspace, so the predicted and FE "mode k" are not comparable. A cluster that the K-th
   output splits is excluded as well.
2. **Accelerating mode** (voltage-based quantities only: `V_acc_V, T_transit, R_over_Q_ohm,
   R_sh_ohm, E_acc_Vm, Epk_Eacc, Bpk_Eacc_mT_per_MVm`). The FE R/Q must satisfy
   `R/Q_bk > max(qoi_rq_floor · max_k' R/Q_bk', 1e-6 Ω)`, taken over the K outputs of the same
   geometry. Modes with E_z ≈ 0 on the axis (TE-like, dipoles) have V ≈ 0. Their R/Q is
   discretisation noise (≈1e-4 of the TM010 value on the synthetic boxes), log R/Q is
   ill-conditioned, and Epk/Eacc → ∞. The 1e-6 Ω absolute floor covers geometries where no mode
   accelerates. Q0 and G are well defined for every mode and are not masked by this rule.
3. **Finite and positive** FE value.

`|V|` is computed as √(Re² + Im² + 1e-300), so masked V ≈ 0 entries still have finite (bounded)
gradients. The masked branch is excluded with `torch.where`, so it does not produce NaN.

### 5.4 Metrics

Whenever a batch carries the operators, the module logs, for every label quantity in
`QOI_LABELS`, `{prefix}/qoi_<name>_rel_err` = mean |q̂/q − 1| over the valid entries (rules
above, exact peaks, no grad). It does this even with `qoi_weight = 0` (use `--qoi_metrics`).

These metrics follow the NaN-safe pattern of `mode_k_rel_l2`: every key is logged on every step,
with `batch_size` equal to its number of valid entries. A batch with none logs 0 with weight 0
and never NaN, so EarlyStopping, ModelCheckpoint and the DDP key sets stay intact.

With the loss on, the module also logs `{prefix}/qoi_loss` and `{prefix}/qoi_<term>_log_mse`.

### 5.5 Cost

Per batch, each QoI evaluation costs one product each with M, S, A_z, E_surf and H_surf on the
K fields (block-diagonal, one sparse product per operator for the whole batch, as `hcurl.spmm`),
plus O(B·K·(P + N_f)) elementwise work. There is no solve and no eigh. The loss evaluates twice
(prediction with grad, FE reference without). The metrics add one more evaluation when
`qoi_peak_p` is set. On the synthetic 8³ boxes (2 × 2.8k edges, K = 6), forward + backward of
the QoI is about 14 ms against about 120 ms for the model step on CPU. Real meshes are dominated
by the Kp CG of the Ritz projection.

The dataset builds the operators once per geometry (`Maxwell3DDataset(..., qoi_ops=True)`,
cached), so data loading pays for them only on first use.

### 5.6 When to turn it on

- Start without it (`qoi_weight = 0`, but `--qoi_metrics`) to see how well a field-only model
  already reproduces the QoI. Usually Q0 and G are within a few % once the field rel-L2 is small.
  R/Q and the peak ratios are more sensitive.
- Turn it on (e.g. `--qoi_weight 0.1`) for fine-tuning once the span loss has converged and the
  isolated modes are matched. Early in training the Ritz mode k and FE mode k may be different
  modes, and the QoI term then mostly adds noise.
- Choose terms by application. Use `Q0,G_ohm` for wall-loss accuracy (tangential H at the wall)
  and `R_over_Q_ohm` for the on-axis field. Add `Epk_Eacc,Bpk_Eacc_mT_per_MVm` with
  `qoi_peak_p ≈ 8–16` for peak-field optimisation.

CLI:

    python train.py --config configs/eigenspace_3d.yaml --qoi_weight 0.1 \
        --qoi_terms Q0,G_ohm,R_over_Q_ohm          # loss + metrics
    python train.py ... --qoi_metrics               # metrics only

These are equivalent to `training.qoi_weight`, `training.qoi_terms` (a list or 'a,b' string),
`training.qoi_metrics`, `training.qoi_peak_p` and `training.qoi_rq_floor`. Either
`training.qoi_weight > 0` or `training.qoi_metrics` makes train.py build the 3D datasets with
`qoi_ops=True`, and records `data_cfg['qoi_ops']`. `qoi_weight > 0` on a batch without
operators raises an error rather than silently skipping the term.
