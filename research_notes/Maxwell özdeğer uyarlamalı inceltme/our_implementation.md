# Our implementation: adaptive p3 label refinement for the Maxwell eigenproblem (repository state)

Scope: branch `claude/field-model` at commit `a95d56a` (read-only inspection, no runs), plus the opt-in prototype
`afc80c3` on branch `adaptive-stop-rule` (worktree `.claude/worktrees/agent-a3bd8a8d617369abd`).
All "sources" are repository files; links are repo-relative paths with line numbers. Line numbers refer to
`a95d56a` unless marked "worktree".

Short version of the loop as it runs by default (`label_sample` → `adaptive_modes(..., remesh=mesh_remesher(...))`):

    for it in 0..7:
        SOLVE    HCurl p3 on a curve-3 netgen mesh, shift-invert ARPACK (tol 1e-10) + exact gradient projection, K+1 modes
        QoI      mode_qoi (Q0, G, R/Q, T, Epk, Bpk) per mode → cluster-invariant table
        ESTIMATE recovery (ZZ-type) indicator: volume + wall terms, summed over the K modes
        STOP?    Δ(cluster mean f) < 1e-6 and Δ(Q0, G, R/Q, Epk, Bpk) < 1e-3 between two successive levels
        REFINE   NEW netgen mesh from an accumulated RestrictH size field (no marking, no bisection)
                 budget checked on the PREDICTED element count only

---

## 1. SOLVE step (element order, curve order, eigensolver, tolerance, number of modes)

### Takeaway
Each level is a fresh, from-scratch solve: full HCurl order-3 space (NGSolve), mesh curved to geometric order 3,
ARPACK `eigsh` in shift-invert mode with a negative shift, an exact M-orthogonal gradient (kernel) projection after
every operator application (both factorisations are direct sparse Cholesky, not CG), ARPACK `tol=1e-10`, a final
Rayleigh–Ritz of the converged block, K+1 physical modes kept (K+5 computed); no warm start between levels.

### Cited Findings
- Order and curving: `ho.solve_modes(mesh, k + 1, order=order)` with `order=s["label_order"]` = 3 and `curve=s["curve"]` = 3 — [src/data_gen/adaptive.py:192](src/data_gen/adaptive.py), [src/data_gen/field_labels.py:47](src/data_gen/field_labels.py), [src/data_gen/field_labels.py:264](src/data_gen/field_labels.py). The remesher curves each new mesh with `mesh.Curve(int(curve))` and re-applies the smooth deformation map (`ho.set_deformation`) — [src/data_gen/adaptive.py:143-145](src/data_gen/adaptive.py).
- Space: `fes = HCurl(mesh, order=order, dirichlet="wall")` (PEC on all of "wall"); the docstring says it is the FULL HCurl space "with its own discrete gradient (fes.CreateGradient, H1 of order p + 1). nograds=True with a P1-gradient projection gives ~2 % errors for p ≥ 2" — [src/data_gen/highorder.py:13-14](src/data_gen/highorder.py), [src/data_gen/highorder.py:273](src/data_gen/highorder.py), [src/data_gen/highorder.py:282](src/data_gen/highorder.py).
- One explicit integration rule for all forms: `volume_measure(order, bonus=4)` = tet rule of degree `2·order + bonus` (degree 10 for p3); reason given: on curved elements "curl·curl − σ·u·v assembled as one form then differs from K − σM (pillbox p = 2: 3e-4), which biases the shift-invert eigenvalues" — [src/data_gen/highorder.py:257-263](src/data_gen/highorder.py), used at [src/data_gen/highorder.py:277-293](src/data_gen/highorder.py).
- Shift: `lam_scale = _diag_mean(K.mat, free) / _diag_mean(M.mat, free)`; `sigma = -1e-2 * lam_scale` (σ < 0 so K − σM is SPD) — [src/data_gen/highorder.py:289-291](src/data_gen/highorder.py), [src/data_gen/highorder.py:15](src/data_gen/highorder.py).
- Shift-invert operator with static condensation: `A = BilinearForm((curl(u) * curl(v) - sigma * u * v) * dx, symmetric=True, condense=True)`, inverted by `_condensed_inverse` (sparse Cholesky on the coupling DOFs + harmonic extension + inner solve: "for p ≥ 3 most DOFs are interior, so fill and time drop by an order of magnitude") — [src/data_gen/highorder.py:246-254](src/data_gen/highorder.py), [src/data_gen/highorder.py:293-296](src/data_gen/highorder.py).
- Kernel projection: `Kp` = ∇φ·∇ψ on the H1(p+1) gradient space (`fes.CreateGradient()`), factorised with `Kp.Inverse(fesh1.FreeDofs(), inverse="sparsecholesky")`; `project(vec)`: "vec ← (I − G Kp⁻¹ GᵀM) vec, in place", applied after every `Ainv` application inside `opinv` and to the random start vector `v0` — [src/data_gen/highorder.py:282-288](src/data_gen/highorder.py), [src/data_gen/highorder.py:297](src/data_gen/highorder.py), [src/data_gen/highorder.py:305-317](src/data_gen/highorder.py), [src/data_gen/highorder.py:330-333](src/data_gen/highorder.py). (The float64 CG `KpSolve` of AGENTS.md is the MODEL's projection, `src/models/hcurl.py`; the label solver uses a direct factorisation.)
- ARPACK call: `eigsh(..., k=nev, M=..., sigma=sigma, OPinv=..., which="LM", v0=v0, tol=tol, ncv=max(2 * nev + 1, 30))` with `nev = k + n_extra`, `n_extra=4`, `tol=1e-10` (default of `solve_modes`) — [src/data_gen/highorder.py:266](src/data_gen/highorder.py), [src/data_gen/highorder.py:334-338](src/data_gen/highorder.py).
- Post-processing: "final Rayleigh–Ritz of the converged block with K, M (cheap; also rotates degenerate pairs into a K/M-orthonormal basis)" (`sla.eigh(sym(vecs.T @ KV), sym(vecs.T @ MV))`); spurious zero modes of isolated conductors dropped by `phys = vals > 1e-6 * lam_scale`; fields normalised to unit M-norm; f = c₀√λ/(2π) — [src/data_gen/highorder.py:340-360](src/data_gen/highorder.py).
- Number of modes: the adaptive loop solves `k + 1` modes (the extra one gives `freq_next`), so ARPACK computes K+5; indicator and QoI use only the first k — [src/data_gen/adaptive.py:192-195](src/data_gen/adaptive.py), [src/data_gen/field_labels.py:306](src/data_gen/field_labels.py). K = `--n_eigen_modes`, CLI default 6 — [src/data_gen/dataset_generator_3d.py:76](src/data_gen/dataset_generator_3d.py); UI default `n_modes: int = Field(default=10, ...)` — [src/service/api.py:531](src/service/api.py).
- No DOF guard inside the adaptive solves: `adaptive_modes` calls `solve_modes` without `max_ndof` (the guard `raise RuntimeError(... > max_ndof)` exists only when passed, e.g. in the non-adaptive path) — [src/data_gen/adaptive.py:192](src/data_gen/adaptive.py), [src/data_gen/highorder.py:274-275](src/data_gen/highorder.py), [src/data_gen/field_labels.py:279](src/data_gen/field_labels.py).
- Cost measurements: 118k DOF p3 pillbox: sparsecholesky + ARPACK 32 s / 2.2 GB; 358k DOF: 175 s / 8.4 GB; LOBPCG + BDDC cold 73 s / 286 s; warm-started LOBPCG from the previous level converges in 10–20 iterations (36–65 s, eigenvalue error 1.9e-8–2.3e-11) but transfer of the modes by NGSolve `Set` cost 68 s, so not adopted; direct-solver memory at 900k DOF could reach ~25 GB ("docs/30 says ~12 GB") — [docs/31_SESSION_HANDOFF.md:77-101](docs/31_SESSION_HANDOFF.md).
- Per-level breakdown (pillbox_pipes, max_ndof 450k, 2 threads): level 0 (235k DOF) assembly 11.5 / factorisation 19.9 / ARPACK 64.5 = 98 s; level 1 (430k DOF) 26.3 / 106 / 141 = 277 s; `mode_qoi` + `indicator` 1.3 + 11.4 s and 1.9 + 17.5 s; after optimisations the solver is "90 % of a sample" (433 s total, 400 s in the two level solves) — [docs/31_SESSION_HANDOFF.md:295-298](docs/31_SESSION_HANDOFF.md), [docs/30_ADAPTIVE_LABELS.md:243-246](docs/30_ADAPTIVE_LABELS.md).
- ARPACK tolerance experiments: `tol` 1e-10 → 1e-8: 78 → 74 iterations, → 1e-6: 60; `n_extra` 4 → 1: 72; σ = −0.03·λ₁: 61 iterations; none adopted. "ARPACK tolerance 1e-10 determines the eigenvector to ~1e-5" (u_model subspace difference 8e-6 between runs) — [docs/31_SESSION_HANDOFF.md:285-348](docs/31_SESSION_HANDOFF.md) (§3.8).

### Inferences
- The algebraic tolerance is fixed (1e-10 on Ritz values, direct solves for both A and Kp) at every level, independent of the current discretisation error (est. 1e-3 on coarse levels); there is no balancing of algebraic vs discretisation error. The projection is exact up to the Cholesky round-off, so Ritz values are upper bounds of the discrete eigenvalues on each mesh.
- Because each level is a new non-nested mesh (§4), nothing is reused between levels (no warm start, no previous eigenvectors as `v0`).

### Gaps
- Whether NGSolve's `HCurl(order=3)` here is the full-polynomial (type II) Nédélec space is not stated in code; it is NGSolve's default HCurl, I did not verify the family.
- No measured memory at 900k DOF on TRUBA exists yet (docs/31 §6 item 4, open).

---

## 2. ESTIMATE step (indicator terms, weights, h-powers, combination over modes, divergence / normal jumps)

### Takeaway
The indicator is a gradient-recovery (Zienkiewicz–Zhu-type, "flux recovery" in the code's words) estimator, not a
residual estimator: per element it measures how far E_k and curl E_k are from their averaged continuous
VectorH1(p) interpolants, in the volume and on the wall facets. There are no h-weights, no element residual
(curl curl E − λE), no explicit tangential/normal jump terms and no divergence term; the indicators of the K modes
are summed with weights 1/λ_k (volume) and 1/‖·‖²_wall (wall); volume and wall are added with weight 1.

### Cited Findings
- Formula (module docstring): "Indicator (flux recovery; unit-L2 modes, so ∫|curl E|² = λ): η_vol,T² = Σ_k ( ‖curl E_k − R curl E_k‖²_T + λ_k ‖E_k − R E_k‖²_T ) / λ_k ; η_wall,T² = Σ_k ( ∫_{∂T∩wall} |curl E_k − R curl E_k|² / ∫_wall |curl E_k|² + same for E_k ). R = averaging interpolation into VectorH1(p): the jumps of the discrete fields are the error." — [src/data_gen/adaptive.py:11-16](src/data_gen/adaptive.py); same text in [docs/30_ADAPTIVE_LABELS.md:24-28](docs/30_ADAPTIVE_LABELS.md).
- Implementation: `V = VectorH1(mesh, order=order)`; `rB.Set(curl(g))`, `rE.Set(g)`; `dB, dE = curl(g) - rB, g - rE`; volume `Integrate(InnerProduct(dB, dB) + float(lk) * InnerProduct(dE, dE), mesh, VOL, element_wise=True, order=2 * order + 2)`, then `ev += np.asarray(e) / float(lk)` — [src/data_gen/adaptive.py:43-57](src/data_gen/adaptive.py). Recovery order = solution order p (3).
- Wall term: indicator `chi = GridFunction(FacetFESpace(mesh, order=0)); chi.Set(1.0, definedon=mesh.Boundaries("wall"))`, measure `dx(element_boundary=True, bonus_intorder=4)`; `wb = Integrate(chi * InnerProduct(dB, dB) * deb, ..., element_wise=True)`, same for `we`; normalisers `nb = ∫ |BoundaryFromVolumeCF(curl g)|² ds("wall")`, `ne_ = ∫ |BoundaryFromVolumeCF(g)|² ds("wall")`; `ew += wb / max(nb, 1e-300) + we / max(ne_, 1e-300)` — [src/data_gen/adaptive.py:46-49](src/data_gen/adaptive.py), [src/data_gen/adaptive.py:58-64](src/data_gen/adaptive.py).
- Combination: `eta = ev + s["wall_weight"] * ew` with `wall_weight` 1.0 (DEFAULTS) — [src/data_gen/adaptive.py:213](src/data_gen/adaptive.py), [src/data_gen/adaptive.py:33-35](src/data_gen/adaptive.py). Justification: "Σ η_vol² estimates the relative eigenvalue error (λ_h − λ ≈ ‖curl e‖² − λ‖e‖², second order in the field error); the figures of merit (wall loss, R/Q, surface peaks) are FIRST order in the field error near the wall, hence the squared relative wall term. At the targets (f 1e-6 ↔ λ 2e-6, Q0 1e-3 ↔ 1e-6 squared) both are ~1e-6, so they add with weight 1." — [src/data_gen/adaptive.py:17-20](src/data_gen/adaptive.py).
- Global estimates recorded per level: `"est_rel_f": 0.5 * float(ev.sum()) / k` and `"est_rel_wall": float(np.sqrt(ew.sum() / k))` — [src/data_gen/adaptive.py:196-197](src/data_gen/adaptive.py). They are stored as attributes `est_rel_f`, `est_rel_wall` — [src/data_gen/field_labels.py:270-272](src/data_gen/field_labels.py).
- Measured effect of the wall term (analytic copper pillbox, R = 40 mm, L = 50 mm, p3, six modes): volume only 6668 tets / 173k DOF: f 1.9e-6, Q0 −9.2e-5, B_pk 1.2e-3; volume + wall 5961 tets / 157k DOF: f 6.1e-7, Q0 −3.4e-5, B_pk 3.9e-4 ("cuts the Q0 and B_pk errors by 3× at equal DOF") — [docs/30_ADAPTIVE_LABELS.md:40-50](docs/30_ADAPTIVE_LABELS.md).
- Cost: indicator 9–18 s per level (11.4 s at 235k, 17.5 s at 430k DOF); replacing the wall integral by `ds("wall")` would cost 0.03 s instead of 0.65 s but changes the rule (3.5e-5 deviation) and therefore the size field and the mesh, so not adopted — [docs/31_SESSION_HANDOFF.md:296-298](docs/31_SESSION_HANDOFF.md), §3.8 "Denenip alınmayanlar".
- Docs list a peak-targeted indicator as an open lever: "B_pk convergence. It is the slowest quantity. A peak-targeted indicator, refining the faces around the current maximum, is the next lever" — [docs/30_ADAPTIVE_LABELS.md:264-265](docs/30_ADAPTIVE_LABELS.md).

### Inferences
- No h-powers appear anywhere in the indicator: the recovery error itself scales like the true error, so no h-weighting is needed (standard for ZZ-type estimators), but it also means there is no reliability constant / guaranteed upper bound.
- Divergence and jumps are only implicit: E ∈ HCurl has continuous tangential but discontinuous normal components, so ‖E − R E‖ (R continuous in all components) picks up the normal jumps of E, i.e. the discrete divergence defect [n·E]; curl E ∈ HDiv has continuous normal but discontinuous tangential components, so ‖curl E − R curl E‖ picks up the tangential jumps of curl E (the facet residual of residual estimators). The element residual curl curl E_h − λ_h E_h and the volume divergence term h‖div E_h‖ of the classical residual estimators (Monk / Beck–Hiptmair–Hoppe–Wohlmuth type) are absent.
- The sum over modes of quadratic per-mode quantities is invariant under an orthogonal rotation inside an exactly degenerate pair (for equal λ_k), so the indicator is cluster-invariant for exact degeneracies; it is not a cluster-robust estimator in the sense of the literature (no explicit treatment of the cluster gap, the (K+1)-th mode is not included).
- The wall term is a heuristic proxy for QoI error (normalised squared wall recovery error), not a goal-oriented (dual-weighted) estimator.
- `est_rel_f` is the MEAN over the K modes of the relative λ-estimate, halved; the validation script compares it with the MAX over modes of the true frequency error (§7), so the two are not like-for-like.

### Gaps
- The exact NGSolve semantics of `GridFunction.Set` into VectorH1 (local projection followed by averaging of shared DOFs) are taken from the code comment ("averaging interpolation"); not verified against NGSolve source here.

---

## 3. MARK step

### Takeaway
In the default (remeshing) path there is no marking step at all: every element receives a target size from a
continuous equidistribution formula. Dörfler marking (θ = 0.5) exists only in the bisection path, which is
disabled in `label_sample` (and in the prototype's stall fallback).

### Cited Findings
- `mark_dorfler(eta, theta)`: "Smallest set of elements carrying the fraction theta of Σ η²" (sort descending, cumulative sum, `searchsorted(c, theta * c[-1]) + 1`) — [src/data_gen/adaptive.py:68-75](src/data_gen/adaptive.py). θ default 0.5 — [src/data_gen/adaptive.py:33](src/data_gen/adaptive.py), [src/data_gen/field_labels.py:52](src/data_gen/field_labels.py).
- It is called only after the `if remesh is not None: ... continue` block, i.e. only when `remesh=None` — [src/data_gen/adaptive.py:214-230](src/data_gen/adaptive.py). `label_sample` always passes `remesh=remesh` (a `mesh_remesher`) — [src/data_gen/field_labels.py:262-264](src/data_gen/field_labels.py); so does `scripts/validate_field_labels.py` — [scripts/validate_field_labels.py:63-65](scripts/validate_field_labels.py).
- Bisection path budget: Dörfler set truncated to the `n_allowed` largest η, `n_allowed = int((s["max_ndof"] * mesh.ne / rec["ndof"] - mesh.ne) / growth)`, `growth` initial 6.0 ("new elements per marked element (bisection closure); measured"), updated as `max(1.0, 1.2 * (mesh.ne - ne0) / max(nm, 1))`; stop "budget" when `n_allowed < 0.02 * mesh.ne` — [src/data_gen/adaptive.py:189](src/data_gen/adaptive.py), [src/data_gen/adaptive.py:231-243](src/data_gen/adaptive.py).
- Why bisection was abandoned: "Marking 200 of 8109 tets (pillbox with pipes) produced a 29k-tet mesh, i.e. 105 new tets per marked one. This ran the sandbox out of memory twice." — [docs/30_ADAPTIVE_LABELS.md:125-129](docs/30_ADAPTIVE_LABELS.md); remesher docstring: "its closure grows a graded mesh 30–100 new tets per marked one" — [src/data_gen/adaptive.py:124-125](src/data_gen/adaptive.py).

### Inferences
- The default loop is "estimate → size field → remesh" (metric-based / equidistribution adaptivity as in remeshing codes), not the "SOLVE–ESTIMATE–MARK–REFINE" AFEM loop whose convergence theory needs bulk (Dörfler) marking plus a refinement that actually refines the marked set.

### Gaps
- None in code; θ is passed through `label_sample` settings but has no effect in the remeshing path.

---

## 4. REFINE step (size field, remeshing vs bisection, curved geometry, budget prediction)

### Takeaway
Each level builds a completely new netgen mesh of the CAD shape with `MeshingParameters.RestrictH` point
restrictions at the centroids of the current elements, accumulated over all levels; the requested size is
h·(ρ·mean(η²)/η_T²)^(1/(2p)) with ρ = 0.25, clipped to [0.3, 1]·h. Geometry is exact through re-meshing the OCC
shape and curving to order 3. The element budget is checked only on the PREDICTED count Σ(h/h_new)³; the mesh
netgen actually returns is never compared with the request, so a remesh that does not refine (or even shrinks)
goes undetected.

### Cited Findings
- Size field: `fac = (rho * eta.mean() / eta) ** (1.0 / (2 * order))`; `return h * np.clip(fac, shrink, grow)` with `rho=0.25, shrink=0.3, grow=1.0`; docstring: "The error of the region of element T scales as (h_new/h)^(2p) (energy norm², p = polynomial order) ... (grow > 1 would coarsen where the error is small; with the accumulating remesher it has no effect)" — [src/data_gen/adaptive.py:110-117](src/data_gen/adaptive.py). The p in the exponent is the FE order `order` (3), not the curve order.
- Element size measure: `h = (6·√2·vol)^(1/3)` ("edge of the regular tet of the same (curved) volume"), centroids of the straight tets — [src/data_gen/adaptive.py:100-107](src/data_gen/adaptive.py).
- Remesher: `mp = MeshingParameters(maxh=float(maxh), curvaturesafety=float(curvaturesafety), grading=float(grading))`; `for c, hh in zip(cen, hn): mp.RestrictH(x=..., y=..., z=..., h=float(hh))`; `mesh = Mesh(OCCGeometry(shape).GenerateMesh(mp=mp))`; `mesh.Curve(int(curve))`; optional deformation — [src/data_gen/adaptive.py:138-145](src/data_gen/adaptive.py). Parameters from `label_sample`: `maxh = maxh_factor·h` (3·h), `curve=3`, `curvaturesafety=cs` (the start mesh's value, 1.5 or 2), `grading=0.5`, `min_jacobian_ratio=0.05` — [src/data_gen/field_labels.py:262-263](src/data_gen/field_labels.py), [src/data_gen/field_labels.py:47-55](src/data_gen/field_labels.py).
- Accumulation: `acc["cen"] = np.vstack([acc["cen"], cen]); acc["h"] = np.r_[acc["h"], hn]` — "The restrictions ACCUMULATE over the calls (RestrictH is an upper bound, so the smallest request wins): remeshing from scratch with only the latest field oscillates — a region refined at one iteration is coarsened again at the next. The coarse end comes from the coarse start mesh (low curvaturesafety), not from coarsening." — [src/data_gen/adaptive.py:126-137](src/data_gen/adaptive.py); [docs/30_ADAPTIVE_LABELS.md:134-136](docs/30_ADAPTIVE_LABELS.md).
- Distorted curved elements: if `r.min() <= min_jacobian_ratio` (det-J min/max ratio from `element_jacobian_ratio`, order-6 rule), the centroids of the bad elements are added with `0.5 * hb[bad]` and the shape is remeshed, up to `tries=3`, else `RuntimeError("remeshed curved mesh still has distorted elements ...")` — [src/data_gen/adaptive.py:146-154](src/data_gen/adaptive.py), [src/data_gen/highorder.py:206-211](src/data_gen/highorder.py). (The start mesh additionally repairs elements below `fix_below` = 0.1 by bisection + `single_level`; the remesher does not — [src/data_gen/highorder.py:140-150](src/data_gen/highorder.py).)
- Budget prediction and scaling: `dof_per_el = rec["ndof"] / mesh.ne`; `ne_next = float(np.sum((h / hn) ** 3))`; `ne_max = s["max_ndof"] / dof_per_el / 1.25  # 25 % margin (grading, curvature)`; if `ne_next > ne_max`: `hn *= (ne_next / ne_max) ** (1 / 3)`; `over = ne_next < 1.05 * mesh.ne and ne_max < 1.3 * mesh.ne   # budget reached: no real refinement left`; then `mesh = remesh(cen, hn)` with no check of the returned `mesh.ne` — [src/data_gen/adaptive.py:214-229](src/data_gen/adaptive.py). Docs: "The predicted size Σ(h/h_new)³ matched netgen within 15 %. When the prediction exceeds the DOF budget, the field is scaled up uniformly." — [docs/30_ADAPTIVE_LABELS.md:137-138](docs/30_ADAPTIVE_LABELS.md).
- Element budget numbers: `max_ndof` 900 000 (label space, "about 12 GB at p3"), the start mesh must have ≤ `max_elements` = 30 000 tets (`TooManyElements` is raised right after netgen generation) — [src/data_gen/field_labels.py:49](src/data_gen/field_labels.py), [src/data_gen/highorder.py:134-136](src/data_gen/highorder.py), [docs/30_ADAPTIVE_LABELS.md:217-222](docs/30_ADAPTIVE_LABELS.md).
- Start mesh: `start_mesh(cad_path, h, deform, s, (s["adapt_cs"], s["curvaturesafety"]))` = curvaturesafety 1.5, falling back to 2 when 1.5 folds elements; `retries: 0`; "a finer value is not tried after a too-large mesh" — [src/data_gen/field_labels.py:74-88](src/data_gen/field_labels.py), [src/data_gen/field_labels.py:253](src/data_gen/field_labels.py), [docs/30_ADAPTIVE_LABELS.md:142-143](docs/30_ADAPTIVE_LABELS.md).
- Bisection alternative (unused by default): `mesh.SetRefinementFlag(...); mesh.Refine(); mesh.Curve(int(curve))` ("netgen Refine: new boundary points on the CAD"), then up to `fix_passes=2` extra bisections of elements with det-J ratio < 0.05 — [src/data_gen/adaptive.py:6](src/data_gen/adaptive.py), [src/data_gen/adaptive.py:78-97](src/data_gen/adaptive.py). Meshes with a bisection hierarchy must go through `ho.single_level` (a .vol round trip) before storage because "the spaces and DOF numbering differ from those of the same mesh saved and loaded again" — [src/data_gen/highorder.py:191-203](src/data_gen/highorder.py), [src/data_gen/field_labels.py:291-293](src/data_gen/field_labels.py).
- Measured remeshing behaviour (pillbox with pipes, p3, six modes, errors vs the finest level, 15.4k tets / 405k DOF): old default cs 2 one solve 8.1k tets: f 3.8e-6, Q0 4.5e-4, B_pk 1.2e-2; adaptive level 2 9.0k tets: 4.8e-6 / 3.6e-4 / 2.0e-2; level 3 13.8k tets: 2.1e-6 / 6.6e-5 / 3.9e-2; "B_pk differs by 4 % between the two finest levels, so the reference itself has not converged for B_pk at the sandbox's 400k-DOF limit" — [docs/30_ADAPTIVE_LABELS.md:144-153](docs/30_ADAPTIVE_LABELS.md). Elliptical (sharp port) adaptive 8.2k tets: f 7.4e-6, Q0 5.8e-4 vs old 9.4k tets 1.3e-5 / 1.3e-3 — [docs/30_ADAPTIVE_LABELS.md:155-163](docs/30_ADAPTIVE_LABELS.md).
- RestrictH at folded fillet elements does not fix them ("local RestrictH at the folded elements or at the BREP vertices (down to 0.1 × fillet radius) | still 10–30") — [docs/30_ADAPTIVE_LABELS.md:101-102](docs/30_ADAPTIVE_LABELS.md).

### Inferences
- The per-level refinement request is mild: for an element with η_T² = mean(η²), fac = 0.25^(1/6) ≈ 0.79, i.e. the target size is only 21 % below the current size (≈ 2× more elements for a uniform error distribution); the full clip 0.3 is reached only where η_T² ≥ 0.25/0.3⁶ ≈ 343 × mean; elements with η_T² ≤ 0.25·mean get h_new = h (a restriction at their current size). On a coarse mesh where netgen's own curvature/maxh sizing already gives ≈ these sizes, the requests can be absorbed without change — which is consistent with the false-convergence observations of §8 (my arithmetic, not a repository measurement).
- Meshes are non-nested between levels (new triangulation each time), so the discrete spaces are not nested and Ritz values need not decrease monotonically from level to level; the docs' "no oscillation" argument (accumulated RestrictH) is about the requested sizes, not a guarantee on the generated mesh.
- The equidistribution formula targets each OLD element's region to ρ·mean(η²), not a global tolerance; there is no link from tol_f/tol_q to the size field.
- Coarsening is impossible by construction (grow = 1, accumulating minima), docs/30 §7 lists coarsening as open — [docs/30_ADAPTIVE_LABELS.md:266-267](docs/30_ADAPTIVE_LABELS.md).

### Gaps
- No repository measurement of how often netgen honours / ignores RestrictH requests as a function of mesh coarseness; only the two anecdotal sequences of §8.
- The "within 15 %" prediction accuracy is stated without the underlying table or number of cases.

---

## 5. STOP rules (tol_f, tol_q, successive differences, est_rel_f, budget, iteration caps, exact `done`)

### Takeaway
Convergence is declared purely from successive-level relative differences of cluster-invariant quantities:
max over clusters of Δ(mean f) < tol_f = 1e-6 AND max over clusters of Δ(mean Q0, mean G, max Epk, max Bpk, and
Σ R/Q of accelerating clusters) < tol_q = 1e-3, from the second solve on. The error estimate `est_rel_f` is computed
and stored but never used to stop. Precedence of the reported stop reason: converged > budget > max_iter (8 solves).
There is no check that the mesh actually got finer between the two compared levels.

### Cited Findings
- Cluster-invariant table: `clusters(f, rel=1e-3)` groups consecutive frequencies with `abs(f[i] / f[i - 1] - 1) < rel`; `invariant_qoi` = "[n_clusters, 6] in QNAMES order: mean f, mean Q0, mean G, Σ R/Q, max E_pk, max B_pk (U = 1 J)" — [src/data_gen/adaptive.py:158-174](src/data_gen/adaptive.py), [src/data_gen/adaptive.py:24-25](src/data_gen/adaptive.py). Computed on `lab["f_hz"][:k]` only — [src/data_gen/adaptive.py:194](src/data_gen/adaptive.py).
- Differences: only `if prev is not None and prev["inv"].shape == inv.shape`; `rel = |inv − prev| / |prev|`; `acc = inv[:, 3] > s["rq_floor"] * inv[:, 3].max()` (rq_floor 1e-2); `rec["df"] = float(rel[:, 0].max())`; `cols = [1, 2, 4, 5] if s["peaks"] else [1, 2]  # sharp edges left: peaks are not converging`; `rec["dq"] = float(max(rel[:, cols].max(), rel[acc, 3].max() if acc.any() else 0.0))` — [src/data_gen/adaptive.py:198-204](src/data_gen/adaptive.py). `peaks` = `sharp_edges == 0` in `label_sample` — [src/data_gen/field_labels.py:267](src/data_gen/field_labels.py).
- Exact condition: `done = "df" in rec and it + 1 >= s["min_iter"] and rec["df"] < s["tol_f"] and rec["dq"] < s["tol_q"]` (min_iter 2) — [src/data_gen/adaptive.py:212](src/data_gen/adaptive.py), [src/data_gen/adaptive.py:33](src/data_gen/adaptive.py).
- Exit (remesh path): `if done or over or it == s["max_iter"] - 1: stop = "converged" if done else ("budget" if over else "max_iter")`; return the CURRENT level's mesh, labels and QoI — [src/data_gen/adaptive.py:224-226](src/data_gen/adaptive.py). `max_iter` = `adapt_max_iter` 8 — [src/data_gen/field_labels.py:52](src/data_gen/field_labels.py), [src/data_gen/field_labels.py:266](src/data_gen/field_labels.py).
- Stored diagnostics: `adapt_iters`, `adapt_stop`, `n_tets_start`, `adapt_cs`, `est_rel_f`, `est_rel_wall`, `adapt_df`, `adapt_dq` — [src/data_gen/field_labels.py:270-272](src/data_gen/field_labels.py).
- docs/30 targets vs stop criteria: frequency target ≤ 1e-5 (stop 1e-6); Q0, G, R/Q ≤ 0.2 % (stop 0.1 %); peaks ≤ 1 % — [docs/30_ADAPTIVE_LABELS.md:14-18](docs/30_ADAPTIVE_LABELS.md).
- docs/31 §6 item 14 (open, found 2026-10-25 with `scripts/cst_calibration_set.py`): "Uyarlamalı etiketlerde sahte yakınsama: yeniden mesh'leme istenen inceltmeyi uygulamayınca (aynı mesh) ardışık seviye farkı ~1e-15 çıkıyor ve döngü `converged` diyor. Görüldüğü yer: kalibrasyon pillbox'ı (`--mode calibration`, 96 → 91 tet, 2.864 DOF, kendi hata tahmini 1,6e-3, TE111 hatası +1,8e-4). Öneri: `converged` için tet / DOF sayısının gerçekten büyümesini ya da `est_rel_f < tol_f` şartını da istemek. Üreteç davranışı değişeceği için sahibin onayı gerekir." — [docs/31_SESSION_HANDOFF.md:432](docs/31_SESSION_HANDOFF.md).

### Inferences
- A level difference below tol is evidence of convergence only if the second level is a genuine refinement of the first; the code never checks `mesh.ne` (or ndof) growth before `done`, and `over` uses the predicted `ne_next`, not the returned mesh. When netgen returns (almost) the same mesh, Δ ≈ 1e-15 and `done` is true regardless of `est_rel_f` (here 1.6e-3, i.e. 1600 × tol_f).
- When the requested refinement is small (`ne_next < 1.05·ne`) but the budget is far (`ne_max ≥ 1.3·ne`), `over` is false and the loop keeps remeshing; this is precisely the regime where the stall → false "converged" occurs.
- If the number of clusters changes between levels (a pair splitting by more than 1e-3 on a coarse mesh), `df` is skipped and that level cannot stop; a cluster cut by the K boundary (K-th mode = one member of a degenerate pair) makes Σ R/Q and max peaks not rotation-invariant, because the (K+1)-th mode is excluded from `invariant_qoi`.
- "budget" and "max_iter" stops return unconverged labels with no extrapolation; the only quality evidence is the stored `est_rel_f` / `adapt_df` / `adapt_dq`.
- The stop is on relative CHANGE (Cauchy-type) with the same tolerance as the per-step change, so with a convergence rate r per level the remaining error is ≈ Δ·r/(1−r); with the mild per-level refinement (§4 inference: ≈ 2× elements) r is not small, so Δ < tol does not imply error < tol.

### Gaps
- No repository statistics on the distribution of `adapt_stop` reasons over a real dataset (TRUBA `TEST=1` reference runs are open, docs/30 §7, docs/31 §6 item 4).

---

## 6. Label mesh / model mesh split and the snapshot logic

### Takeaway
Labels (frequencies, QoI, p3 fields) come from the final adaptive level; the model trains on the last level
visited with ≤ `model_max_elements` = 10 000 tets (or the start mesh if none fits), at HCurl p2, with the p3 fields
L2-transferred by one point search when the meshes differ, or nested M-projection when they coincide.

### Cited Findings
- Snapshot callback: `def keep_model_level(m, lab_, q_, rec):  # finest level within the model budget` — `if m.ne <= s["model_max_elements"] or not snap: snap.update(vol=_mesh_bytes(m), ne=int(m.ne))`; called after every solve via `on_iter` — [src/data_gen/field_labels.py:259-261](src/data_gen/field_labels.py), [src/data_gen/adaptive.py:207-208](src/data_gen/adaptive.py).
- Transfer choice: `if s["adaptive"] and snap["ne"] != mesh.ne:` → reload snap mesh, `transfer_to_model` (L2 transfer across meshes); `elif mesh.levels > 1:` → `ho.single_level` + `transfer_to_model`; `else:` → `project_to_model(..., unit_norm=True)` (exact nested projection) — [src/data_gen/field_labels.py:287-296](src/data_gen/field_labels.py).
- Transfer method: model mesh integration points (216 per tet) located once in the label mesh, values on an `IntegrationRuleSpace`, `M₂ u₂ = ∫ v₂ · E₃ dx₃`; 253 s → 29 s per 430k-DOF sample, same projection to 3e-14 — [src/data_gen/field_labels.py:134-184](src/data_gen/field_labels.py), [docs/30_ADAPTIVE_LABELS.md:165-180](docs/30_ADAPTIVE_LABELS.md).
- Guard: `_check_reload` rebuilds the stored model mesh and compares HCurl(p2) ndof — [src/data_gen/field_labels.py:203-212](src/data_gen/field_labels.py), [src/data_gen/field_labels.py:303](src/data_gen/field_labels.py).
- Model-space floor: p2 Rayleigh quotients on pillbox_pipes model meshes of 11.3k / 13k tets give frequency floors 2.4e-3 … 3.7e-4 and 8.6e-3 … 4.9e-4, proj_err 1.8–5.4e-3 ("about 0.04–0.9 %") — [docs/30_ADAPTIVE_LABELS.md:202-213](docs/30_ADAPTIVE_LABELS.md); docs/31 §1 repeats "%0,04–0,9" — [docs/31_SESSION_HANDOFF.md:12](docs/31_SESSION_HANDOFF.md).

### Inferences
- "Finest level within the budget" is implemented as "LAST level within the budget"; with non-monotone meshes (e.g. 1324 → 1232 → 1257 tets, §8) the two differ.
- The label accuracy (1e-6) and the model's own floor (1e-4–1e-2 relative in frequency) differ by 2–4 orders of magnitude, so label tolerances far below the model floor buy nothing for training (docs/31 §5 already suggests separate loose/tight tolerances for train vs test sets — [docs/31_SESSION_HANDOFF.md:398-411](docs/31_SESSION_HANDOFF.md)).

### Gaps
- None for the code path; the effect of the transfer (snap ≠ final) on the training targets beyond proj_err is not measured in the docs.

---

## 7. Defaults (generator CLI, field_settings, UI, TRUBA) and the analytic validation

### Takeaway
Defaults are consistent across generator, UI and TRUBA: p3 labels on a curve-3 mesh, p2 model, tol_f 1e-6,
tol_q 1e-3, max_ndof 900k (UI: largest of 900k/600k/300k/150k that fits 85 % of the machine memory), model mesh
≤ 10 000 tets, 8 levels max, adaptive on; ρ, θ, clip limits, cluster_rel, rq_floor are not exposed.

### Cited Findings
- `adaptive.DEFAULTS = {"theta": 0.5, "tol_f": 1e-6, "tol_q": 1e-3, "max_iter": 8, "max_ndof": 900_000, "min_iter": 2, "wall_weight": 1.0, "cluster_rel": 1e-3, "rq_floor": 1e-2, "min_jacobian_ratio": 0.05, "peaks": True, "rho": 0.25}` — [src/data_gen/adaptive.py:33-35](src/data_gen/adaptive.py).
- `field_labels.DEFAULTS`: `label_order 3, model_order 2, curve 3, maxh_factor 3.0, curvaturesafety 2.0, grading 0.5, fix_passes 3, fix_below 0.1, min_jacobian_ratio 0.05, retries 1, max_elements 30000, max_ndof 900_000, adaptive True, adapt_cs 1.5, tol_f 1e-6, tol_q 1e-3, adapt_max_iter 8, theta 0.5, model_max_elements 10000` — [src/data_gen/field_labels.py:47-55](src/data_gen/field_labels.py).
- Generator CLI (`parse_args`): `--n_eigen_modes` 6, `--label_order` 3, `--model_order` 2, `--curve` 3, `--maxh_factor` 3.0, `--curvaturesafety` None ("default 2.0; adaptive labels start at 1.5"), `--field_max_elements` 30000, `--max_ndof` 900000, `--no_adapt`, `--tol_f` 1e-6 ("relative frequency change between refinements"), `--tol_q` 1e-3 ("relative change of Q0 / G / R/Q / surface peaks"), `--model_max_elements` 10000 — [src/data_gen/dataset_generator_3d.py:76](src/data_gen/dataset_generator_3d.py), [src/data_gen/dataset_generator_3d.py:114-139](src/data_gen/dataset_generator_3d.py).
- `field_settings(args)` maps these 1:1; an explicit `--curvaturesafety` sets BOTH `curvaturesafety` and `adapt_cs` — [src/data_gen/dataset_generator_3d.py:769-777](src/data_gen/dataset_generator_3d.py).
- UI (`GenReq`): `n_modes 10, label_order 3, model_order 2, min_fillet 0.05, maxh_factor 3.0, field_max_elements 30000, adapt True, tol_f 1e-6 (1e-9…1e-2), tol_q 1e-3 (1e-6…0.2), model_max_elements 10000, max_ndof None → machine default, sample_timeout 7200` — [src/service/api.py:525-556](src/service/api.py); `_default_ndof`: first of `(900_000, 600_000, 300_000, 150_000)` with `J.field_mem_gb(n) <= 0.85 * host['mem_gb']`, else 60 000 — [src/service/api.py:565-569](src/service/api.py); memory model `FIELD_MEM_BASE_GB, FIELD_MEM_GB_PER_NDOF = 1.0, 12.0 / 900_000` — [src/service/jobs.py:63-66](src/service/jobs.py); the job passes `--max_ndof --tol_f --tol_q --model_max_elements` to the generator — [src/service/jobs.py:125-145](src/service/jobs.py).
- TRUBA: `TOL_F` 1e-6, `TOL_Q` 1e-3, `MODEL_MAX_ELEMENTS` 10000, `MAX_NDOF` 900000 defaults — [src/service/truba.py:169-172](src/service/truba.py), [src/service/truba.py:281](src/service/truba.py).
- Analytic validation script: copper pillbox R = 40 mm, L = 50 mm, netgen `Cylinder`, `mesh_remesher(cyl, maxh=0.012, curve=3, curvaturesafety=1.5)`, start mesh = `remesh(np.zeros((0, 3)), np.zeros(0))`, K = 6, `max_ndof` default 300 000, tol_f 1e-6, tol_q 1e-3, max_iter 8; PASS targets on the FINAL level only: `TARGETS = {"f": 1e-5, "Q0": 2e-3, "G": 2e-3, "R/Q": 2e-3, "Epk/Eacc": 1e-2, "Bpk/Eacc": 1e-2}`; f error = max over the six analytic frequencies (TM010, TE111×2, TM011, TM110×2), QoI errors for TM010 (mode 0) — [scripts/validate_field_labels.py:26-47](scripts/validate_field_labels.py), [scripts/validate_field_labels.py:49-70](scripts/validate_field_labels.py). The stop reason is printed but does not affect PASS — [scripts/validate_field_labels.py:84-85](scripts/validate_field_labels.py).
- Reported validation result: "Analitik pillbox kontrolü geçiyor: 157k DOF'ta f 6e-7, Q0 3e-5, R/Q 1e-5, Epk 2e-4, Bpk 4e-4" — [docs/31_SESSION_HANDOFF.md:11](docs/31_SESSION_HANDOFF.md); matches docs/30 §2 row "adaptive, volume + wall | 5961 | 157k | 2.1e-6 | 6.1e-7 | −3.4e-5 | 1.2e-5 | −2.1e-4 | 3.9e-4" — [docs/30_ADAPTIVE_LABELS.md:48](docs/30_ADAPTIVE_LABELS.md). Uniform references: 992 tets / 28k DOF: f 1.1e-5, Q0 −1.9e-4, B_pk 5.0e-4; 7936 tets / 210k DOF: f 2.2e-7, Q0 −7.5e-6, R/Q 8.1e-7, E_pk −1.9e-4, B_pk 1.8e-4 — [docs/30_ADAPTIVE_LABELS.md:45-46](docs/30_ADAPTIVE_LABELS.md). "All targets are met from about 22k DOF"; "The pillbox is smooth, so uniform refinement is near-optimal there"; "E_pk/E_acc stalls at −2e-4: that is the sampling of the maximum (wall integration points), not field error" — [docs/30_ADAPTIVE_LABELS.md:53-56](docs/30_ADAPTIVE_LABELS.md).

### Inferences
- On the analytic pillbox, uniform refinement at 210k DOF beats the adaptive 157k-DOF result in every column except E_pk (f 2.2e-7 vs 6.1e-7, Q0 7.5e-6 vs 3.4e-5, R/Q 8.1e-7 vs 1.2e-5, B_pk 1.8e-4 vs 3.9e-4); the validation thus shows the loop meets loose targets, not that it is efficient.
- The validation's PASS criteria are 10× (f) and 2× (Q0, G, R/Q) looser than the stop tolerances, and it accepts any stop reason, so a false "converged" at a coarse level would still FAIL only if the final error exceeds 1e-5 — it does not test the stop rule itself.

### Gaps
- The validation run's per-level table (DOF sequence, timings, stop reason) for the 157k-DOF result is not stored in the repository; only the final-level numbers are.

---

## 8. Measured behaviour and the opt-in prototype `--strict_stop` (worktree, commit afc80c3)

### Takeaway
The prototype makes `converged` require real DOF growth (≥ 1.05× the previous level) and replaces a stalled
remesh (< 1.05× the current tets) by Dörfler bisection within the budget; default off. It documents the
false-convergence sequence 96 → 88 → 91 → 91 → 91 tets on the calibration pillbox and 1324 → 1232 → 1257 on the
60k validation run. Several of the prototype agent's other numbers (5× estimator over-prediction, the 31k–36k DOF
levels, flat B_pk) are not recorded anywhere in the repository and are unverified here; one is indirectly
supported: docs/30's own numbers contradict its "within 1.5×" claim.

### Cited Findings
- Diff `a95d56a..afc80c3`: 5 files, +71/−5 (docs/30, adaptive.py, dataset_generator_3d.py, field_labels.py, tests/test_field_adaptive.py) — `git -C .claude/worktrees/agent-a3bd8a8d617369abd diff --stat a95d56a..afc80c3`.
- New defaults `"strict_stop": False, "min_growth": 1.05` — worktree [src/data_gen/adaptive.py:42](.claude/worktrees/agent-a3bd8a8d617369abd/src/data_gen/adaptive.py).
- Rule 1: `if done and s["strict_stop"] and prev is not None and rec["ndof"] < s["min_growth"] * prev["ndof"]: done = False  # no real refinement since the last level: no evidence; rec["no_growth"] = True` — worktree [src/data_gen/adaptive.py:220-223](.claude/worktrees/agent-a3bd8a8d617369abd/src/data_gen/adaptive.py).
- Rule 2: `new = remesh(cen, hn)`; `if s["strict_stop"] and new.ne < s["min_growth"] * mesh.ne:   # remesher did not refine: bisect instead` → `mark_dorfler(eta, s["theta"])`, budget `n_allowed` as in the bisection path (stop `"budget"` if `n_allowed < 0.02 * mesh.ne`), `refine(mesh, flags, curve, deform, ...)`, `rec["stalls"] = 1`; else `mesh = new` — worktree [src/data_gen/adaptive.py:238-252](.claude/worktrees/agent-a3bd8a8d617369abd/src/data_gen/adaptive.py).
- Plumbing: `--strict_stop` CLI flag ("`converged` needs real DOF growth since the previous level, and a remesh that does not refine is replaced by bisection"), `field_settings` → `"strict_stop": bool(...)`, `field_labels.DEFAULTS["strict_stop"] = False`, passed to `adaptive_modes` — worktree dataset_generator_3d.py (+4 lines near 137 and 778), field_labels.py:52 and 267.
- New test `test_strict_stop_does_not_call_a_stalled_remesh_converged`: a remesher that always returns `_pillbox(0.08)`; default rule → `res["stop"] == "converged" and res["hist"][-1]["df"] < 1e-12` while `est_rel_f > 100 * tol_f`; with strict_stop the final ndof is > 1.5× the first and some level has `stalls` — worktree [tests/test_field_adaptive.py:272-287](.claude/worktrees/agent-a3bd8a8d617369abd/tests/test_field_adaptive.py).
- Worktree docs/30 §4a: "On a coarse mesh netgen can return the same mesh for every `RestrictH` request (the calibration pillbox, `--mode calibration`: 96 → 88 → 91 → 91 → 91 tets), so Δf and Δq are 0 and the loop reports `converged` while its own estimate is 1.6e-3 (true errors: TE111 +1.8e-4, TM110 +1.2e-3). The same, milder, shows in `validate_field_labels.py --max_ndof 60000`: the last levels go 1324 → 1232 → 1257 tets and 'converge'." Strict result: "Calibration pillbox, max_ndof 60k, 1 thread: default stops 'converged' at 2,864 DOF (TM110 +1.2e-3); strict_stop: 96 → 613 → 1626 tets, stops `budget` at 43k DOF with all six frequencies within 7e-6." Suggested filter for existing datasets: `n_tets_label < 1.05 * n_tets_start` or `est_rel_f > 100 * tol_f` — worktree [docs/30_ADAPTIVE_LABELS.md:165-182](.claude/worktrees/agent-a3bd8a8d617369abd/docs/30_ADAPTIVE_LABELS.md).
- Cross-check with docs/31 item 14: same case "96 → 91 tet, 2.864 DOF, kendi hata tahmini 1,6e-3, TE111 hatası +1,8e-4" — [docs/31_SESSION_HANDOFF.md:432](docs/31_SESSION_HANDOFF.md). The TM110 +1.2e-3 figure appears only in the worktree's docs/30 §4a, consistent with item 14 (which reports only TE111).
- Estimator calibration claim vs its own data: docs/30: "It tracks the true frequency error within 1.5× at every level: 1.2e-3 / 1.1e-3, 6.0e-6 / 4.7e-6, 2.1e-6 / 6.1e-7" — [docs/30_ADAPTIVE_LABELS.md:51-52](docs/30_ADAPTIVE_LABELS.md); module docstring same claim — [src/data_gen/adaptive.py:21-22](src/data_gen/adaptive.py).

### Inferences
- The third pair quoted in docs/30 (2.1e-6 / 6.1e-7) is a ratio of 3.4×, not ≤ 1.5×; the "1.5×" claim is contradicted by the doc's own numbers at the finest level. The prototype agent's "about 5× (8.9e-6 est vs 1.7e-6 true)" is in the same direction (over-prediction growing as the mesh is refined) but from a run whose table is not in the repository. Also note the aggregation mismatch (mean-over-modes estimate vs max-over-modes true error, §2).
- Code confirms the agent's mechanism: only `ne_next` (prediction) is checked ([src/data_gen/adaptive.py:218-223](src/data_gen/adaptive.py)) and the returned mesh size is never compared ([src/data_gen/adaptive.py:227](src/data_gen/adaptive.py)); that netgen "ignores RestrictH on coarse meshes" is an empirical observation (the tet sequences), not something the code proves; the mild size requests (§4 inference) are an alternative/complementary explanation.
- Prototype limitations visible in the code: (a) rule 1 compares with the previous level's ndof only (a level pair 1324 → 1232 → 1257 has the last step +2 %, < 5 %, so it would now be blocked, as intended); (b) the bisection fallback re-introduces the closure blow-up risk documented in docs/30 §4 (105 new tets per marked one on graded meshes), bounded only by the predicted `growth` (initial 6.0) and `n_allowed`; (c) a bisected mesh has `levels > 1`, handled downstream by `single_level` + transfer ([src/data_gen/field_labels.py:291-293](src/data_gen/field_labels.py)); (d) it does not use `est_rel_f` in the stop rule (the other option in item 14) and does not change the size field or the indicator; (e) default off, so existing datasets keep false `converged` flags (the doc gives a filter).
- On the ownership rule: AGENTS.md requires the owner's go-ahead for generator behaviour changes, which is why it is opt-in ("Üreteç davranışı değişeceği için sahibin onayı gerekir", docs/31 item 14).

### Gaps
- Not found in any repository file (only in the caller's summary of the prototype agent's findings), therefore unverified: the 5× over-prediction run (8.9e-6 vs 1.7e-6); "levels from 31k to 36k DOF bring no gain but take about 60 % of the time"; "B_pk error flat at 4e-4 to 1.2e-3 while the level differences stay below tol_q". The repository only has the related docs/30 §4 observation that B_pk "differs by 4 % between the two finest levels" on pillbox_pipes and that B_pk is "the slowest quantity".
- I did not run the prototype test or the validation (read-only, shared machine).

---

## 9. Gaps against the standard AFEM loop (nestedness, guaranteed refinement, stop rule, clusters, QoI, algebraic error)

### Takeaway
Compared with the textbook SOLVE–ESTIMATE–MARK–REFINE loop for Maxwell eigenproblems, the default implementation
replaces MARK+REFINE by non-nested remeshing with advisory size requests, uses a recovery indicator without a
reliability bound, stops on successive differences without checking refinement or using its own estimator, handles
degenerate modes only through cluster-invariant aggregates with a fixed 1e-3 gap threshold, treats QoI by a
heuristic wall term (no goal-oriented estimator), and runs a fixed-tolerance direct eigensolve at every level.

### Cited Findings
- Nestedness: each level is `Mesh(OCCGeometry(shape).GenerateMesh(mp=mp))` — a new triangulation — [src/data_gen/adaptive.py:142](src/data_gen/adaptive.py); nested bisection exists but is off by default — [src/data_gen/adaptive.py:179-182](src/data_gen/adaptive.py) ("remesh=None: bisection of the Dörfler-marked elements of `mesh` (modified in place; on graded meshes the closure explodes)").
- Guaranteed refinement of marked elements: none in the default path (no marking; `RestrictH` requests; returned mesh unchecked) — [src/data_gen/adaptive.py:214-229](src/data_gen/adaptive.py); observed non-refinement 96 → 88 → 91 → 91 → 91 — worktree [docs/30_ADAPTIVE_LABELS.md:167-171](.claude/worktrees/agent-a3bd8a8d617369abd/docs/30_ADAPTIVE_LABELS.md).
- Stop rule: Cauchy differences only, `est_rel_f` unused — [src/data_gen/adaptive.py:212](src/data_gen/adaptive.py).
- Clusters: consecutive-ratio threshold `cluster_rel` 1e-3, invariants on the first K modes only, indicator summed over modes, final Rayleigh–Ritz rotation inside the ARPACK block — [src/data_gen/adaptive.py:158-174](src/data_gen/adaptive.py), [src/data_gen/adaptive.py:194-195](src/data_gen/adaptive.py), [src/data_gen/highorder.py:340-347](src/data_gen/highorder.py).
- QoI: wall term as first-order proxy; peaks excluded when sharp edges remain (`peaks: sharp_edges == 0`, stored as NaN); R/Q only for clusters with R/Q > 1 % of the max — [src/data_gen/adaptive.py:17-20](src/data_gen/adaptive.py), [src/data_gen/adaptive.py:201-204](src/data_gen/adaptive.py), [src/data_gen/field_labels.py:282-286](src/data_gen/field_labels.py); E_pk/B_pk = max over wall integration points of order 8 (`wall_order=8`) — [src/data_gen/highorder.py:403-442](src/data_gen/highorder.py).
- Algebraic vs discretisation: ARPACK tol 1e-10, direct factorisations at every level, no warm start; warm-start LOBPCG measured but not adopted — [src/data_gen/highorder.py:266](src/data_gen/highorder.py), [docs/31_SESSION_HANDOFF.md:91-99](docs/31_SESSION_HANDOFF.md).

### Inferences
- Nestedness: without nested spaces, the monotonicity of Ritz values across levels and the contraction/quasi-optimality arguments of AFEM theory do not apply; successive differences can be small (or zero) without the error shrinking.
- Guaranteed refinement: the AFEM convergence proofs need every marked element refined (e.g. newest-vertex bisection with closure); here the mesh generator may ignore the request, which is the root of item 14.
- Stop rule: standard practice is estimator-based (η ≤ tol, ideally with an efficiency/reliability calibration) or a difference test only between genuinely refined levels; the repository has the estimator but does not use it; the prototype adds the growth condition only.
- Clusters: the 1e-3 relative-gap threshold is fixed and independent of the discretisation error; a cluster split near the K/K+1 boundary is not handled; the literature's cluster-robust estimators (whole invariant subspace, gap to the next eigenvalue) are not used.
- QoI: Q0/G (wall loss) and R/Q are first-order functionals of the field; a dual-weighted / goal-oriented indicator would weight the wall recovery error by an adjoint, rather than a fixed weight 1; B_pk/E_pk as maxima are not smooth functionals at all, which matches their slow convergence.
- Algebraic error: solving to 1e-10 at a level whose discretisation error is 1e-3 is wasted work; the docs' own cost tables show the solves are ~90 % of the sample.

### Gaps
- No repository document compares the implementation with specific literature or production codes on these points (docs/31 §4 mentions Palace as an independent reference with AMR, without details of its loop); that comparison is the other researchers' scope.
