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
n_axis = 401 midpoints of equal cells over the mesh z-extent on (ξx, ξy) = (axis_xy − center_xy)/s,
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

*(placeholder — converter labels, dataset / collate integration, evaluation scripts; owned by the
data agent.)*

## 5. Training

*(placeholder — `qoi_torch`, the optional QoI loss term and its effect; owned by the training
agent.)*
