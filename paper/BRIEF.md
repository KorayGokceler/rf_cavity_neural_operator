# Paper brief — shared by every section author

Target: an arXiv preprint (cs.LG + physics.acc-ph / math.NA). It has to pass arXiv moderation and read as a
careful methods paper: precise mathematics, honest scope, every number traceable to a measurement in this
repository. Language: English, academic, concise. LaTeX, `article` class, `paper/main.tex`.

## 1. Thesis (what the paper argues)

We predict the K lowest resonant modes of 3D RF cavities (frequencies, E fields, figures of merit) from the
geometry with a neural operator whose output is *not* the fields themselves but a learned basis of m vector
fields. The basis is mapped into a conforming high-order H(curl) finite-element space on a curved mesh,
projected onto the discretely divergence-free subspace (kernel projection with a float64 CG solve), and a
Rayleigh–Ritz step returns frequencies and fields. Consequences that the paper develops and measures:

1. **Guarantee:** every predicted eigenvalue is an upper bound of the corresponding discrete eigenvalue of the
   space it runs in (min–max), and no spurious zero modes appear (gradients are projected out).
2. **Space independence:** the same weights run on any conforming space (p2 or p3, a refined mesh); the bound
   still holds there (measured, docs/34 §4.2).
3. **Accuracy floor:** quadrature-point tokens reach 1e-6–9e-6 relative eigenvalue error in single-sample
   overfit (docs/28 §3) vs 2.4e-3 for a lowest-order (Whitney N0) model.
4. **Data:** an adaptive p3 label generator on curved meshes (volume + wall error indicator, label/model
   mesh split), validated against analytic pillbox solutions; measured cost reductions.
5. **Low-fidelity training:** one p2 solve on the model mesh (LF) costs 4.3× less than an adaptive p3 label
   (HF), differs from it by 0.004–0.047 % in frequency and 0.08–0.32 % (M-norm) in field, and a model trained
   on LF labels is at least as accurate on held-out HF geometries as one trained on HF labels of equal or higher
   label cost, and slightly better with more LF geometries (docs/34 §8; single runs, small differences; per
   geometry LF 40 beats HF 8 on 3 of 6, excluding the outlier geometry 27: HF 8 1.11 %, LF 24 1.14 %, LF 40 1.06 %). The LF→HF gap of the model is 0.02 percentage points.

Honest scope (must be stated, not hidden): all training runs are small (CPU, ≤40 training geometries, one
geometry family `pillbox_pipes`, a 64-dim 2-block trunk); absolute errors (1–2 % frequency) reflect an
under-trained model, not the method's ceiling; the QoI loss is disabled for the field model; fine-tuning on
HF and mixed LF+HF training were inconclusive (lost control run, schedule error). Present the work as a
method + data pipeline + controlled small-scale evidence.

## 2. Sections, files and owners

Each section author writes ONLY its own files: `paper/sections/NN_name.tex` and `paper/bib/NN_name.bib`.
Do not edit `main.tex`, `macros.tex`, another section or anything outside `paper/`.

| NN | file | scope | primary sources (repository) | length |
|---|---|---|---|---|
| 00 | `00_abstract.tex` | written last by the editor | — | 200 words |
| 01 | `01_introduction.tex` | motivation (cavity design loop, cost of 3D eigen-solves, CST/ACE3P-grade accuracy demands), why naive field regression fails (spurious modes, no guarantees, fixed resolution), our approach in one paragraph, explicit contribution list (5 items above), paper outline | AGENTS.md, docs/31 §1 §5, docs/28 §1, docs/34 §8, docs/24 §1 | 1–1.5 pages |
| 02 | `02_related_work.tex` | neural operators (FNO, DeepONet, graph/transformer operators such as GNOT, Galerkin/linear attention, Transolver, geometry-aware operators); learning eigenvalue problems and subspaces (neural eigensolvers, spectral/Ritz-type learning); structure-preserving / FE-coupled learning (physics-informed, finite-element neural networks); Maxwell eigenproblems with Nédélec elements and spurious modes; RF cavity simulation and design (SUPERFISH, CST, ACE3P, Palace) and ML surrogates for accelerator cavities; multi-fidelity learning | docs/15–17 on branch history are not available; use the web. Every reference must be verified to exist | 1–1.5 pages |
| 03 | `03_formulation.tex` | the cavity eigenproblem (curl curl E = k²E, n×E=0), weak form in H0(curl), Nédélec spaces of order p on curved (isoparametric) meshes, generalized eigenproblem K x = λ M x, discrete kernel = gradients of S_h, potentials G, Kp = GᵀMG, the discretely divergence-free subspace, frequencies f = c₀√λ/(2π·scale), figures of merit (stored energy, wall loss, Q0, G, R/Q, R_sh, T, Epk/Eacc, Bpk/Eacc, beam axis, L_acc) with normalisation | docs/20 §2 §8, docs/24 §0–§1, docs/28 §2, docs/29 §2–3, docs/19 §6, src/qoi/figures.py, src/data_gen/highorder.py | 1.5–2 pages |
| 04 | `04_method.tex` | the model: geometry tokens = quadrature points (14/tet, features), mass-aware linear attention trunk, m vector fields ψ_j, fixed linear map to DOFs (element L2 fit + projection-based interpolation C), wall DOFs zero, kernel projection PV = V − G Kp⁻¹ GᵀMV (batched Jacobi-PCG float64, autograd via a second solve), rank-revealing Ritz (projected_eigh), **Proposition: Ritz values ≥ discrete eigenvalues, with proof**, losses (span in mass+energy norm, self-supervised compliance, ortho, frequency, fidelity weights), resolution/order transfer remark, complexity and the measured step cost; a TikZ pipeline figure | docs/20 §1–§3 §8, docs/28 §2 §4, docs/34 §2, src/models/eigenspace_operator_field.py, src/models/hcurl.py, src/models/layers.py, src/training/lightning_module.py, configs/field_3d.yaml | 2.5–3 pages |
| 05 | `05_data.tex` | geometry families and parameter ranges (published elliptical half-cells accepted: TESLA, ILC-LL, ICHIRO …), fillet floor, curved netgen meshes and element quality, adaptive p3 labels (error indicator: volume + wall; remeshing not bisection; stopping), label mesh vs model mesh (≤ model_max_elements), projection/transfer of labels onto p2, reload consistency check, QoI computation (beam axis, accelerating length), analytic validation (pillbox: f 6e-7, Q0 3e-5, R/Q 1e-5, Epk 2e-4, Bpk 4e-4 at 157k DOF), cost per sample and the measured generator speed-ups (transfer 253→29 s, pillbox 450k DOF 664→433 s, elliptical ~108→75 s, labels unchanged to 1e-15–1e-9) | docs/19, docs/29, docs/30, docs/33 §12, docs/24 §3, docs/31 §1 §3.8, src/data_gen/*.py | 2–2.5 pages |
| 06 | `06_multifidelity.tex` | LF/HF definitions (LF: one p2 solve on the model mesh; HF: adaptive p3, projected to p2), why the Ritz layer makes LF labels natural (LF frequencies are the discrete eigenvalues of the model space itself; HF frequencies lie below the p2 floor), the shared start mesh (`start_mesh`), cost ratio and label differences (frequency, M-norm field subspace distance, wall QoIs Q0 −0.8 to −1.4 %, peaks up to −28 %), training mechanics (per-sample fidelity, fidelity_weights, fidelity_freq_scale, train_extra, hf_repeat, init_from) | docs/34 §1–§3 §8.1, src/data/field_dataset.py, src/training/lightning_module.py, src/data_gen/field_labels.py | 1.5 pages |
| 07 | `07_experiments.tex` | (a) single-sample overfit table (docs/28 §3); (b) first LF→HF study (docs/34 §4.1); (c) space transfer p2→p3 and refined mesh (docs/34 §4.2), with costs; (d) the equal-cost experiment (docs/34 §8.2) — main table, per-geometry results, transfer gap, Ritz bound check; (e) inference vs FE time; (f) training and loader cost reductions (docs/28 §4, docs/31 §3.7). Setup paragraph: hardware (4-core CPU VM), model size, optimiser, schedule, steps | docs/28, docs/34, docs/31 §3 | 2.5–3 pages |
| 08 | `08_discussion.tex` | discussion (structural ceiling: model cannot beat the p2 FE solution of its own space; where speed comes from; when the method pays off), limitations (scale, one family, QoI loss off, inconclusive fine-tune / mixed runs, mesh generation still per geometry, GPU path untested), future work (hybrid warm-started LOBPCG with model subspace, scaled LF+HF training, QoI loss, Palace cross-validation, more families), conclusion | docs/31 §4 §5 §6, docs/34 §5 §6 §8, docs/33 | 1–1.5 pages |

## 3. Notation (use `paper/macros.tex`)

Ω cavity, ∂Ω PEC wall, E electric field, k wavenumber, λ = k² (scaled), V_h = HCurl(p) Nédélec space on the
curved mesh, S_h = H¹₀ potential space of order p+1, n DOFs, m basis width, K modes, 𝖪 curl–curl stiffness,
𝖬 mass, 𝖦 discrete gradient, 𝖪_p = 𝖦ᵀ𝖬𝖦, 𝖵 ∈ ℝ^{n×m} learned basis (DOFs), 𝖯𝖵 = 𝖵 − 𝖦𝖪_p⁻¹𝖦ᵀ𝖬𝖵,
𝖠_V = 𝖵ᵀ𝖪𝖵, 𝖬_div = (𝖯𝖵)ᵀ𝖬(𝖯𝖵), θ_k Ritz values, λ_{h,k} discrete eigenvalues. LF / HF for the two label
fidelities. Macros: `\Kmat \Mmat \Gmat \Kp \Vmat \Pmat \Mdiv \Av \Vh \Sh \lamh \Qz \RoQ \Epk \Bpk \Eacc \Lacc \LF \HF`.
Label prefixes: `sec:`, `eq:`, `tab:`, `fig:`, `prop:`; prefix your labels with your section number
(e.g. `sec:04-ritz`, `eq:03-weak`). You may reference labels of other sections only with these names:
`sec:introduction sec:related sec:formulation sec:method sec:data sec:multifidelity sec:experiments
sec:discussion` (each section's top-level `\section{}` carries its label), `prop:upper-bound` (owned by 04),
`tab:equal-cost` (owned by 07), `fig:pipeline` (owned by 04).

## 4. Rules

1. **No invented numbers.** Every number comes from the repository docs (or code defaults). Put the
   source next to it as a LaTeX comment: `% src: docs/34 §8.2`. If a number you want does not exist, write
   the sentence without it. Do not round in a way that changes a claim.
2. **No invented references.** Only cite works you have verified exist (web search: title, authors, year,
   venue or arXiv id). Write full BibTeX entries in your own `paper/bib/NN_name.bib`; key style
   `lastnameYEARfirstword` (e.g. `li2021fourier`). If you cannot verify a work, do not cite it.
3. Precise claims: say "on one geometry family", "on 6 held-out geometries", "upper bound of the discrete
   eigenvalue" (not of the exact eigenvalue — the latter needs an additional argument; say so if relevant).
4. The repository docs are mostly in Turkish; the paper is in English. Translate the meaning, not literally.
5. No AI-model names, no internal branch names, no file paths in the body text (code names such as
   `start_mesh` may appear only in a footnote or not at all; describe the algorithm instead). Software that
   was used (NGSolve/Netgen, PyTorch, Lightning, gmsh, OpenCascade) may be named and cited.
6. Use `booktabs` tables, `\cref`, SI units with `siunitx` where natural. Equations numbered only if referenced.
7. Write the section so it compiles inside `main.tex` (no preamble, no `\begin{document}`). Begin with
   `\section{...}\label{sec:...}` using the label from §3.
8. Do not run long computations. Reading code and docs is the job. You may run `grep`, `sed -n`, `python -c`
   for small checks.
9. When done, reply with: the files written, a list of every number used with its source, the references
   added, and any claim you were unsure about.
