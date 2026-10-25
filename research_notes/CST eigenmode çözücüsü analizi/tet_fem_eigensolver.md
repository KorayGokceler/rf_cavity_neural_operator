# CST Studio Suite tetrahedral FEM eigenmode solver: discretisation and solution (public sources)

Method note for the reader: WebFetch was blocked in this sandbox (DNS failure for every host; the proxy also returned 403 for arxiv.org, doi.org, crossref). So I could not open any page. Every finding below comes from the web-search tool's summaries of pages (CST 2013/2016 help mirrors, vendor press releases, JACoW/CERN slides, arXiv, forums), not from reading the full documents. Wording in quotation marks is what the search summaries quoted. Treat exact numbers as "reported by search summary of source X", and re-verify against the primary page before the report relies on them. Most sources are CST 2011-2018; I found no CST 2020-2025 eigenmode-specific text.

## 1. Tetrahedral mesh generation and adaptation (settings, h-/p-adaptation, monitored quantity, passes)

### Takeaway
The eigenmode solver's adaptation is h-type mesh refinement driven by the maximum eigenfrequency change between consecutive passes (Delta F), with user-set accuracy, minimum and maximum pass counts. The first pass mesh comes from CST's automatic "expert system". I found no documentation of p-adaptation for the eigenmode solver, and no documented default pass counts or default threshold.

### Cited Findings
- CST help "Adaptive Mesh Refinement (Eigenmode)" (CST 2013 mirror): the refinement improves the mesh until a given accuracy of the eigenmode frequency variation is reached, set by the Accuracy value in the Eigenmode Solver Parameters dialog. An example value of 0.01 means stop once the maximum frequency variation between two passes is below that limit (an example, not a stated default). — [CST2013 help mirror, Adaptive Mesh Refinement (Eigenmode)](http://www.mweda.com/cst/cst2013/mergedprojects/cst_microwave_studio/special_solvopt/special_solvopt_mesh_adaption_properties_sap.htm)
- The convergence error Delta F is the largest change in eigenmode frequencies between two successive passes. The 2016 documentation calls the stopping criterion "Maximum frequency variation". — same page; [CST2013 Adaptive Tetrahedral Mesh Refinement (Frequency Domain), related page](http://www.mweda.com/cst/cst2013/mergedprojects/cst_microwave_studio/special_solvopt/special_solvopt_adaptive_mesh_refinement_frequency_domain.htm)
- Minimum passes are always performed even if frequencies stop changing; maximum passes caps the run even if not converged. — [CST2013 help mirror, Adaptive Mesh Refinement (Eigenmode)](http://www.mweda.com/cst/cst2013/mergedprojects/cst_microwave_studio/special_solvopt/special_solvopt_mesh_adaption_properties_sap.htm)
- The initial mesh is generated automatically by the "expert system" and used for the first pass; refinement then adjusts expert-system settings until the frequency variation is acceptable. The "Curved elements" mesh setting should be "Automatic". — same page
- CERN Accelerator School 2023 (M. Wendt): adaptive refinement increases mesh density in high-field areas until the change in resonance frequency of the eigenmodes is below the given error value. — [CAS 2023 simulations slides](https://indico.cern.ch/event/1294829/contributions/5442426/attachments/2698007/4683942/CAS2023_simulations_wCover.pdf)
- A CST tutorial says adaptive refinement is on by default for tetrahedral meshes, but that text concerns the frequency-domain S-parameter solver, whose default (2%) does not apply to eigenmode. — [MWS tutorials (Rose-Hulman)](https://www.rose-hulman.edu/class/ee/HTML/ECE340/PDFs/MWS_Tutorials.pdf)
- CST "True Geometry Adaptation" (2009, frequency-domain tetrahedral solver, not specifically eigenmode): nodes of a newly created tetrahedral mesh are projected back onto the input model rather than placed on edges of the old mesh. — [Microwave Journal, CST frequency domain mesh adaptation](https://www.microwavejournal.com/articles/7178-cst-represents-breakthrough-in-frequency-domain-mesh-adaptation)
- Benchmark user setting (TESLA cavity, CST MWS 2018): 200,771 curved tetrahedra of curved order 5, solver order 3 (constant), desired accuracy 1e-12. — [arXiv 2001.09686](https://arxiv.org/pdf/2001.09686)
- ICAP 2012 CST slides: tetrahedral eigenmode mesh converges quickly (nine modes in under two minutes reported by the summary). — [ICAP2012 moaai2 talk](https://epaper.kek.jp/ICAP2012/talks/moaai2_talk.pdf)

### Inferences
- Because only Delta F of frequencies is monitored (not a field/residual estimator), CST's adaptation is likely a heuristic h-refinement; mirroring it in NGSolve would mean refining on the frequency change of the K lowest modes, optionally with a field-based estimator (inference, no source says what indicator CST refines on).
- Order/curvature is not adapted per pass in the documented description (p-adaptation not described); order is a fixed user choice. (Absence of evidence, not a documented negative.)
- The quoted steps-per-wavelength, min/max edge length and surface-tolerance settings for the eigenmode tetrahedral mesh were not found; they likely belong to the general tetrahedral Mesh Properties dialog (inference).

### Gaps
- Default values for Accuracy (Delta F), minimum/maximum passes, and whether refinement is per-element by error indicator or global: not found.
- "Steps per wavelength", "surface mesh tolerance", min/max edge length semantics for the eigenmode tet mesh: not found.
- Whether the fraction of elements refined per pass is documented: not found.

## 2. Element type and order (Nedelec family, orders, curved elements)

### Takeaway
The tetrahedral solver uses H(curl)-conforming higher-order elements with a user-selectable solver order 1-3 (second order default), on curved tetrahedra whose geometric order is chosen to follow the solver order. Curved elements for the eigenmode solver were introduced in CST MWS 2011 (SP3).

### Cited Findings
- CST announced a FEM eigenmode solver with curved elements of arbitrary order in CST MWS 2011 (SP3); curved elements give a conformal geometry representation and better accuracy. — [Microwave Journal, curved finite elements for the eigenmode solver](https://www.microwavejournal.com/articles/11362-cst-announces-curved-finite-elements-now-also-for-the-eigenmode-solver); [Microwave Journal, MWS 2011](https://www.microwavejournal.com/articles/14593-cst-introduces-curved-finite-element-for-eigenmode-solver-in-cst-microwave-studio-2011)
- Help "Eigenmode Solver Specials (Tetrahedral)" (CST 2013 mirror): choice of first, second or third order; second order is default; first order suits geometrically complex structures, higher order suits structures with large voids and fewer geometric details (fewer cells needed for accuracy). — [CST2013 help mirror](http://www.mweda.com/cst/cst2013/mergedprojects/cst_microwave_studio/special_solvopt/special_solvopt_eigenmode_solver_specials_tetrahedral.htm)
- Same page: increase the curved element order (a mesh property) when the solver order is increased; the curvature order is usually chosen automatically to fit the solver order; linear elements are curved elements of order one. — same page
- ICAP 2012 CST slides: curved tetrahedra available up to third order (per search summary). — [ICAP2012 moaai2 talk](https://epaper.kek.jp/ICAP2012/talks/moaai2_talk.pdf)
- PETRA 7-cell paper (DESY/TU Darmstadt): the real-valued CST eigenmode solver uses basis functions up to third order on curved tetrahedral elements (per search summary); curvilinear tetrahedral elements model curved geometry. — [Numerical Calculation of Eigenmodes in PETRA 7-cell](https://inspirehep.net/files/f5fa81124143df6be5126be6e243c694)
- A user in arXiv 2001.09686 used curved element order up to 5 ("user defined") with solver order 3 (constant) in CST MWS 2018; the authors report CST's solution stagnating for high order and small h. — [arXiv 2001.09686](https://arxiv.org/pdf/2001.09686)
- The terms "mixed order" and "constant order": only "Solver order: 3rd (constant)" appeared, in the arXiv benchmark settings, suggesting a constant/mixed order option exists in the dialog. — same arXiv paper

### Inferences
- "Up to third order" for the basis (HCurl p3-like, whichever Nedelec family) matches the p3 level of this project's label solver; CST "second order" default probably corresponds to the full-degree-2 or second-kind-like space, but no source specifies first vs second kind or the DOF count (unknown).
- Geometric order up to 5 in 2018 vs "up to 3" in 2012 suggests the curved order cap rose between versions (inference from two sources of different dates, CST release not confirmed).

### Gaps
- Nedelec family type (first/second kind), hierarchical vs interpolatory basis, quadrature: not found.
- Exact meaning of "mixed order" and the dialog options in versions after 2018: not found.

## 3. Eigenvalue algorithms, shift/target, convergence, static and spurious modes, losses, boundaries

### Takeaway
For tetrahedral meshes CST implements a single "general purpose" method with no user choice. AKS and JDM are the hexahedral choices. The tetrahedral solver accuracy is a relative residual of the eigensystem. I found no public statement of how the gradient null space is suppressed in the tetrahedral solver.

### Cited Findings
- CST product page: Eigenmode Solver is a 3D solver for resonant structures incorporating the Advanced Krylov Subspace method (AKS) and the Jacobi-Davidson method (JDM); typical uses: high-Q accelerator cavities, filters, slow-wave structures. — [3DS solver overview](https://www.3ds.com/products/simulia/cst-studio-suite/electromagnetic-simulation-solvers)
- Help (as quoted by forum/mirror): with a tetrahedral mesh "one general purpose method is implemented and no choice of the method is to be made"; hexahedral offers AKS and JDM. — [CST2013 help mirror, tet specials](http://www.mweda.com/cst/cst2013/mergedprojects/cst_microwave_studio/special_solvopt/special_solvopt_eigenmode_solver_specials_tetrahedral.htm); [EDAboard thread quoting help](https://www.edaboard.com/threads/when-to-use-tetrahedral-or-hexahedral-mesh-in-cst-eigenmode-solver.349439/)
- Tetrahedral solver accuracy is defined as the relative residual of the eigensystem. — [CST2013 Eigenmode Solver Parameters](http://www.edatop.com/cst/CST2013/30591.html) (per search summary)
- AKS uses a special filter polynomial to suppress unwanted higher modes and needs an estimate of the highest eigenfrequency of interest (default 0 = unknown). JDM can choose the number of modes automatically for a band fmin..fmax. — [CST2013 Eigenmode Solver Overview](http://www.mweda.com/cst/cst2013/mergedprojects/cst_microwave_studio/special_overview/special_overview_eigenmodesolverover.htm); [CST2013 Eigenmode Solver Parameters](http://www.edatop.com/cst/CST2013/30591.html)
- Number of modes: the solver always returns lowest-frequency modes; time grows with mode count. A "frequencies above" option calculates modes above a given frequency in ascending order. — [CST2013 Eigenmode Solver Parameters](http://www.edatop.com/cst/CST2013/30591.html)
- Losses: by default ignored in the eigenmode calculation; "Consider losses in postprocessing only" gives Q via perturbation; if disabled, lossy/dispersive materials are evaluated at one fixed frequency (default centre frequency) giving complex eps/mu; this slows the solve significantly. External Q (waveguide ports) requires the postprocessing option. — [CST MWS manual via manualzz](https://manualzz.com/doc/o/n27e0/cst-microwave-studio-eigenmode-solver); [CST2013 Eigenmode Solver Parameters](http://www.edatop.com/cst/CST2013/30591.html)
- The older CST overview says the (new) eigenmode solver can compute modes in lossy structures directly; lossy JDM requires frequency-independent complex permittivity validity. — [MWS tutorials](https://www.rose-hulman.edu/class/ee/HTML/ECE340/PDFs/MWS_Tutorials.pdf); [manualzz](https://manualzz.com/doc/o/n27e0/cst-microwave-studio-eigenmode-solver)
- Periodic boundaries: phase shifts per axis; non-zero phase gives travelling-wave modes (complex-valued fields); the band-diagram workflow sweeps phase over the Brillouin zone. — [3DS community, band diagram analysis](https://3dswym.3dexperience.3ds.com/post/simulia-community/band-diagram-analysis-cst-studio-suite-approach_2baxVWubTQmeJkx5vMTVQg)
- PETRA 7-cell paper: commercial eigenmode solvers (including CST's) are mostly real-valued (lossless); a complex-valued solver is needed for the coupler/loss case. — [PETRA 7-cell paper](https://inspirehep.net/files/f5fa81124143df6be5126be6e243c694)
- Related research code (not CST): TU Darmstadt 2013 slides refer to controlling a Jacobi-Davidson solver with eigenvalues selected in a region around a complex target. Does not establish CST's algorithm. — [DESY TEMF 2013-08-09](https://www.desy.de/fel-beam/data/talks/files/DESY_TEMF_WA%202013-08-09.pdf)

### Inferences
- A tetrahedral "general purpose" method with a relative-residual accuracy is consistent with a shift-and-invert / Krylov-type or JD-type iteration on a generalized problem, but no source names the algorithm; do not state it as CST fact.
- Real-valued, lossless default plus perturbative Q means the discrete problem is the same real symmetric generalized eigenproblem as this project's `VᵀKV c = θ M c`, with Q computed afterwards from wall loss (consistent with `src/qoi/`).
- Gradient/static modes: for Nedelec elements the zero-eigenvalue null space (dimension = number of interior nodes in lowest order) is physically spurious; CST must either shift past it or project it out. This is a general FEM fact, not a CST statement; how CST does it is unknown.

### Gaps
- Algorithm name of the tetrahedral general solver, shift/target selection, preconditioner/linear solver, convergence tolerance default, static mode filtering: not found.
- Surface impedance (wall loss) treatment in the tetrahedral eigenmode solver specifically: not found.
- Any CST 2019-2025 changes to the tetrahedral eigenmode solver: not found (the 2024 What's New result in search did not show eigenmode specifics).

## 4. Tetrahedral vs hexahedral for accelerator cavities and reported accuracy

### Takeaway
Recent CERN training material and older cavity studies say tetrahedral (curved, higher order) is preferred and converges much faster for curved SRF geometries; hexahedral is the fast-setup alternative. Quantitative head-to-head numbers inside CST were not found.

### Cited Findings
- CAS 2023: CST eigenmode supports tetrahedral and hexahedral; tetrahedral preferred as more precise and flexible for representing a geometry. — [CAS 2023 slides](https://indico.cern.ch/event/1294829/contributions/5442426/attachments/2698007/4683942/CAS2023_simulations_wCover.pdf)
- A 2012 CST talk: tetrahedral mesh converges quickly; default for hexahedral solvers is now an enhanced fast perfect-boundary approximation. — [ICAP2012 moaai2 talk](https://epaper.kek.jp/ICAP2012/talks/moaai2_talk.pdf)
- A study compared hex-mesh (time/frequency domain) and tetrahedral higher-order curvilinear approaches on the CERN SPL 704.4 MHz 5-cell cavity (TEMF, IPAC 2011 WEPC093). Search summaries differed on its conclusion (one said hex convergence was too slow to be practical and tetra converged much faster); I could not read the paper and it may use TEMF in-house codes rather than CST. — [IPAC2011 WEPC093](https://accelconf.web.cern.ch/ipac2011/papers/wepc093.pdf); [IPAC2012 MOPPC059](https://proceedings.jacow.org/IPAC2012/papers/moppc059.pdf)
- Multipacting study (spoke cavity): far fewer high-order tets than hexahedra needed for given accuracy; recommended curved tets (it used CST hex and Omega3P tets). — [arXiv 1303.6731](https://arxiv.org/pdf/1303.6731)
- Accuracy against a reference: TESLA 1-cell cavity in CST MWS 2018 (tet, order 3, curved order 5, accuracy 1e-12) agreed with an isogeometric BEM reference to relative error 4.018e-08 with f about 1.2767 GHz; CST stagnates at high orders/small h. — [arXiv 2001.09686](https://arxiv.org/pdf/2001.09686)
- Elliptical cavity perturbation study (IPAC) used CST MWS 2012 tet mesh with curved elements for unperturbed modes. — [ResearchGate: Eigenmode computation for elliptical cavities](https://www.researchgate.net/publication/272475903_EIGENMODE_COMPUTATION_FOR_ELLIPTICAL_CAVITIES_SUBJECT_TO_GEOMETRIC_VARIATIONS_USING_PERTURBATIVE_METHODS)
- Forum (user opinion, low weight): beginners advised to use tetrahedral; if both converge, mesh type should not matter. — [EDAboard](https://www.edaboard.com/threads/when-to-use-tetrahedral-or-hexahedral-mesh-in-cst-eigenmode-solver.349439/)

### Inferences
- A frequency relative error around 1e-8 is probably the practical floor for CST curved-tet p3 on a smooth cavity (single data point; the stagnation is the authors' remark). That bounds how well a p3 NGSolve label pipeline can be "validated against CST" for fine frequencies.
- Match CST's recommended practice (tets, solver order 2-3, curved order >= solver order, adaptation on the maximum frequency change) as the reference configuration for comparisons.

### Gaps
- Direct tet-vs-hex accuracy/time tables within CST for the same cavity: not found.
- Current (2024-2025) vendor recommendation text: not found.

## 5. Sensitivity settings (accuracy, losses, lowest frequency above, symmetry)

### Takeaway
Documented knobs: Accuracy (relative residual for the tet solver; Delta F for adaptation), number of modes, "frequencies above", "consider losses in postprocessing only", AKS/JDM options (hex only), solver order and curved order (tet). Symmetry handling for the eigenmode solver was not found.

### Cited Findings
- Accuracy target is a relative residual of the eigensystem for the tetrahedral solver (per CST2013 parameters page via search summary); the same Accuracy value also sets the adaptive-refinement frequency tolerance per the adaptation page. A Rose-Hulman tutorial gives a -40 dB default accuracy, but that describes the time-domain solver and is not applicable. — [CST2013 Eigenmode Solver Parameters](http://www.edatop.com/cst/CST2013/30591.html); [CST2013 adaptation page](http://www.mweda.com/cst/cst2013/mergedprojects/cst_microwave_studio/special_solvopt/special_solvopt_mesh_adaption_properties_sap.htm); [MWS tutorials](https://www.rose-hulman.edu/class/ee/HTML/ECE340/PDFs/MWS_Tutorials.pdf)
- "Frequencies above" / lowest-frequency option: eigenmodes above a given frequency are calculated in ascending order; JDM supports automatic mode count in a band. — [CST2013 Eigenmode Solver Parameters](http://www.edatop.com/cst/CST2013/30591.html)
- Benchmark accuracy used by a researcher: 1e-12. — [arXiv 2001.09686](https://arxiv.org/pdf/2001.09686)
- Student/learning edition lacks the eigenmode solver (user report). — [3DS community](https://3dswym.3dexperience.3ds.com/question/3dexperience-edu-students/eigenmode-analysis-in-cst-studio-suite_mxp-8iB8SwqKnzQqCn5QHw)

### Inferences
- Setting both "frequencies above" and a mode count is probably implemented as a shift for a shift-invert iteration (inference only), analogous to the NGSolve shift in this project.

### Gaps
- Symmetry planes (PEC/PMC) effect on the eigenmode solver, its default accuracy value, and 2020+ option names: not found.
