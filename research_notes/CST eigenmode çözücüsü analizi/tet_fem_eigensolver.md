# CST Studio Suite tetrahedral FEM eigenmode solver: discretisation and solution (public sources)

Provenance (revision 2): in revision 1 WebFetch was blocked and I relied on search-tool summaries. In this revision I downloaded and read the full text (curl + pdftotext) of: arXiv 2001.09686 (Bembel IGA-BEM paper with CST benchmarks), IPAC2011 WEPC093, ICAP2015 WEP36 (PETRA), the CERN CAS 2023 CST tutorial (4005 lines of text), the ICAP2012 CST AG slides (MOAAI2), the ICAP2012 TEMF slides (MOADI1), the Imperial/IEEE THz paper, the 2011 CST press releases, the goengineer CST 2024 What's New, the Dassault 2022 blog and the 3ds.com solver page. The CST 2013/2016 help pages (mweda.com, edatop.com, manualzz, edaboard) could NOT be fetched (connection closed / 403 / DNS), so every statement attributed to "CST help" below is still second-hand via search-tool summaries and is marked "(help text via search summary)". Treat those as unverified until someone opens the help in an installed CST.

Corrections to revision 1 (all verified in the full texts):
- The 4.018e-8 figure is the relative difference between CST and the authors' own Bembel TESLA-cell reference; it is not "CST's error", and the "stagnation" in arXiv Fig. 6 is the Bembel-vs-CST-reference difference stalling, which the authors read as Bembel being the more accurate one. It is not shown that CST itself stagnates.
- The Table 2 CST numbers (order-3 solver, curved up to 5, accuracy 1e-12) are for the unit SPHERE, not the TESLA cell.
- IPAC2011 WEPC093 compares CST MWS 2010 HEXAHEDRAL eigenmode with a TEMF in-house tetrahedral curvilinear code. It is NOT evidence about CST's tetrahedral solver.
- In the PETRA paper only the real-valued runs use CST MWS 2015; the complex-valued solver is TEMF's own code.

## 1. Tetrahedral mesh generation and adaptation (settings, h-/p-adaptation, monitored quantity, passes)

### Takeaway
CST's eigenmode adaptation is automatic h-refinement of the tetrahedral mesh, stopped on the maximum eigenfrequency change between passes. The first-pass mesh comes from the generic automatic tetrahedral mesher. I found no p-adaptation, and no documented default for threshold or pass counts, in any source I could open.

### Cited Findings
- Practical CST workflow (CAS 2023 tutorial, pillbox and TESLA mid-cell): leave "Adaptive mesh refinement" ticked; "the refinement will increase the mesh density in, e.g. high field areas until the variation in resonance frequency of the Eigenmodes is less than the error value given". For the periodic TESLA cell the user also ticks "Refine tetrahedral mesh". The solver reports, under 1D Results, "Adaptive Meshing", the mode frequencies and the number of tetrahedra for each mesh pass. — [CERN CAS 2023, M. Wendt, CST tutorial](https://indico.cern.ch/event/1294829/contributions/5442426/attachments/2698007/4683942/CAS2023_simulations_wCover.pdf)
- Help text via search summary (CST 2013 mirror, not opened): the convergence error Delta F is the largest eigenfrequency change between two successive passes; Accuracy in the Eigenmode Solver Parameters sets it (0.01 was an example, not a default); minimum passes are always run, maximum passes caps the run; the first-pass mesh comes from the automatic "expert system"; "Curved elements" in Mesh Properties should be "Automatic". — [CST2013 help mirror "Adaptive Mesh Refinement (Eigenmode)"](http://www.mweda.com/cst/cst2013/mergedprojects/cst_microwave_studio/special_solvopt/special_solvopt_mesh_adaption_properties_sap.htm) (unreachable here)
- CST 2024 What's New lists, for the Eigenmode Solver, a "multilevel mesh refinement option to force stronger refinement in the first passes" and "thermal loss export for the General (Lossy) method". — [GoEngineer, CST Studio Suite 2024 What's New](https://www.goengineer.com/blog/simulia-cst-studio-suite-2024-whats-new)
- CST 2011 SP3 introduced the FEM eigenmode solver with "curved elements of arbitrary order", familiar from the general-purpose FEM frequency-domain solver and combined with "True Geometry Approximation" (nodes of refined tets projected back to the model); CST called the performance gain in its tests "extraordinary". — [Microwave Journal, 7 June 2011](https://www.microwavejournal.com/articles/14593-cst-introduces-curved-finite-element-for-eigenmode-solver-in-cst-microwave-studio-2011)
- CST AG slide (Wolfheimer, ICAP 2012): eigenmode solver "Curved tetrahedral mesh available (up to 3rd order)"; "Tetrahedral mesh: quick convergence, < 2 min to calculate 9 modes" (model courtesy of Lancaster University). — [ICAP2012 MOAAI2 slides](https://epaper.kek.jp/ICAP2012/talks/moaai2_talk.pdf)
- Observed user settings in published runs: unit sphere, MWS 2018, accuracy 1e-12, solver order 3 (constant), curved elements "up to order 5 (user defined)": 1089 tets gave error 1.86e-7 (21,597 DOFs), 5292 tets 5.42e-9 (101,997 DOFs), 9191 tets 1.95e-9 (175,818 DOFs) on the smallest eigenvalue. TESLA one-cell: 200,771 curved tets of order 5, CST f = 1.27666401260 GHz. — [arXiv 2001.09686, Table 2 and Sec. 7.2](https://arxiv.org/pdf/2001.09686)
- PETRA 7-cell (MWS 2015, PMC symmetry planes, 307,200 tets): second-order basis on second-order curved tets (cyan) versus third-order basis on fourth-order curved tets (red) both reproduce frequencies and R/Q; mesh studies with 117,012 / 347,427 / 1,145,408 curvilinear tets are shown for the complex solver. — [ICAP2015 WEP36](https://inspirehep.net/files/f5fa81124143df6be5126be6e243c694)
- Mesh-size anchor for tet vs hex: the TEMF tetrahedral code reached the 704.4 MHz design value of the SPL 5-cell with 6.2 million tets, where CST MWS 2010 hex needed 100 million cells (see Q4). — [IPAC2011 WEPC093](https://accelconf.web.cern.ch/ipac2011/papers/wepc093.pdf)

### Inferences
- Curved geometric order cap grew over time in public statements: "up to 3rd" (2012 slide), 4 used (MWS 2015 PETRA), 5 used (MWS 2018 benchmark). The cap is a version-dependent fact; I cannot say when it changed.
- Ratio DOFs/tet in the sphere table (21,597/1089 = 19.8; 175,818/9191 = 19.1) is consistent with a full third-degree first-kind Nédélec space (asymptotic about 7/6 edges x 3 + 2 faces x 6 + 3 interior = 18.5 DOF per tet, my own count). That suggests CST "third order" = 45-dim first-kind tetrahedron, same size as NGSolve HCurl order 2 (full first kind) or order 3 (with the higher-order hierarchical space). This is arithmetic on public numbers, not a CST statement; check by comparing with NGSolve ndof on the same mesh.
- Only the frequency change is documented as the stop criterion, so CST appears to refine on a frequency-sensitivity or field-gradient indicator but report on Delta F. What marker selects elements is not public.

### Gaps
- Default Accuracy (Delta F), default minimum and maximum passes, fraction of elements refined, surface-mesh tolerance, steps per wavelength, min/max edge length for the eigenmode tet mesh: not found in any openable source.
- Whether solver order can change between passes (p-adaptation): no evidence either way.
- The CST help pages themselves: unreachable (mweda.com closes the tunnel, edatop.com and edaboard/manualzz not retrievable).

## 2. Element type and order (Nédélec family, orders, curved elements)

### Takeaway
The tetrahedral solver uses H(curl)-conforming higher-order tetrahedral elements of user-selectable solver order 1-3 on curved tetrahedra whose geometric order can be set separately (published runs use 2, 4 and 5). Second order is the default per the help text (via search summary). The Nédélec kind and basis type are not documented publicly.

### Cited Findings
- Help text via search summary (CST 2013 mirror "Eigenmode Solver Specials (Tetrahedral)", not opened): solver order 1st/2nd/3rd, second default; first order for complex geometries, higher order for structures with large voids; curved element order is a mesh property that should be raised along with the solver order and is usually chosen automatically to fit it; linear elements are curved order one; for tet meshes "one general purpose method is implemented and no choice of the method is to be made". — [CST2013 help mirror](http://www.mweda.com/cst/cst2013/mergedprojects/cst_microwave_studio/special_solvopt/special_solvopt_eigenmode_solver_specials_tetrahedral.htm) (unreachable here)
- PETRA paper, full text: the real-valued CST eigenmode solver "utilizes the basis functions up to third order on curved tetrahedral elements"; runs used (2nd order basis, 2nd order curved) and (3rd order basis, 4th order curved). — [ICAP2015 WEP36](https://inspirehep.net/files/f5fa81124143df6be5126be6e243c694)
- arXiv 2001.09686 full text: "Solver order: 3rd (constant)" and "Curved elements: up to order 5 (user defined)" in MWS 2018. The word "constant" implies a mixed/variable-order alternative in the dialog, but the paper does not say so. — [arXiv 2001.09686](https://arxiv.org/pdf/2001.09686)
- Convergence of the general kind is shown for TEMF's own tetrahedral FEM code (not CST) with hierarchical H(curl) bases (Ingelström): on a sphere with planar elements the error rate stays near 2 even for the high-order basis (slide labels 2.0 and 2.1), whereas curvilinear elements allow the high-order basis to reach rate 4.0 (labels read from the plot; my reading of the slide). — [ICAP2012 MOADI1 (Ackermann, Weiland)](https://proceedings.jacow.org/ICAP2012/talks/moadi1_talk.pdf)
- 2011 press release: "curved elements of arbitrary order … enable a conformal representation of the geometry which improves the simulation accuracy". — [Microwave Journal](https://www.microwavejournal.com/articles/14593-cst-introduces-curved-finite-element-for-eigenmode-solver-in-cst-microwave-studio-2011)

### Inferences
- For this project: CST's best-practice pairing (solver order 3 with geometric order >= solver order, often higher) is the same design as curved HCurl p3 on an order-3+ curved mesh; the TEMF finding says geometry order is what unlocks high-order convergence, which supports using one explicit curved integration rule (invariant in AGENTS.md).
- First-kind vs second-kind not stated; DOF count (above) points to first kind, full degree. Unverified.

### Gaps
- Nédélec kind, hierarchical vs interpolatory basis, quadrature order, geometry mapping (isoparametric vs interpolation of CAD) for CST: not found.
- Meaning of "mixed order": not found.

## 3. Eigenvalue algorithms, shift/target, convergence, static and spurious modes, losses, periodic and impedance boundaries

### Takeaway
AKS and JDM are the hexahedral-mesh methods; for tetrahedral meshes there is one undisclosed "general purpose" method (the 2024 notes also name a "General (Lossy)" method). Accuracy is a relative residual. How CST removes the gradient null space and how it picks its shift is not public in anything I could open.

### Cited Findings
- Vendor product page: Eigenmode Solver "incorporating the Advanced Krylov Subspace method (AKS), and the Jacobi-Davidson method (JDM)", for high-Q accelerator cavities, filters and slow-wave structures; supports sensitivity analysis (direct detuning from structural deformation). — [3DS CST solver overview](https://www.3ds.com/products/simulia/cst-studio-suite/electromagnetic-simulation-solvers)
- Method names in release notes: 2024 What's New mentions the "General (Lossy)" method of the Eigenmode Solver; a Dassault community post says the General (Lossy) solver includes dispersive lossy materials, open boundaries and ports. — [GoEngineer CST 2024](https://www.goengineer.com/blog/simulia-cst-studio-suite-2024-whats-new); [3DS community, band diagrams](https://3dswym.3dexperience.3ds.com/post/simulia-community/band-diagram-analysis-cst-studio-suite-approach_2baxVWubTQmeJkx5vMTVQg)
- Help text via search summary (not opened): tetrahedral accuracy = relative residual of the eigensystem; AKS needs an estimate of the highest frequency (0 = unknown) and uses a filter polynomial; JDM supports lossy structures and can find all modes in fmin..fmax; "Frequencies above" computes modes above a threshold in ascending order; always the lowest modes are returned; "Consider losses in postprocessing only" (default; losses by perturbation, required for external Q); if off, lossy/dispersive materials are evaluated at one fixed frequency (default centre frequency) and the solve is much slower. — [CST2013 Eigenmode Solver Parameters mirror](http://www.edatop.com/cst/CST2013/30591.html), [CST MWS manual copy](https://manualzz.com/doc/o/n27e0/cst-microwave-studio-eigenmode-solver) (both unreachable here)
- Lossy walls, full text of a 2012 IEEE THz paper: with CST MWS (then) the walls are PEC because the eigenmode solver "does not support lossy metals"; unloaded Q is extracted post-processing by entering conductivity, and CST reports only the ideal resonant frequency, not the complex eigenfrequency. The paper's own method integrates the field over the cavity volume and the tangential field over the wall surface (volume and surface integrals of the lossless solution). This predates the General (Lossy) method. — [Episkopou et al., IEEE TTST 2012](https://www.imperial.ac.uk/media/imperial-college/faculty-of-engineering/electrical-and-electronic-engineering/public/optical-and-semiconductor-devices/pubs/2012_09_TTST.pdf)
- CAS 2023 tutorial: "For Eigenmode and similar simulations of beam-pipe components … fill the background with PEC. Losses can easily be introduced in a post-processing step by a perturbation method based on the lossless solution." — [CERN CAS 2023](https://indico.cern.ch/event/1294829/contributions/5442426/attachments/2698007/4683942/CAS2023_simulations_wCover.pdf)
- Periodic boundaries: Z-planes periodic, a phase advance per cell (e.g. 180 deg) is set in Boundary Conditions, Phase shifts tab; TESLA mid-cell is the tutorial example; phase sweep builds dispersion diagrams. — [CERN CAS 2023](https://indico.cern.ch/event/1294829/contributions/5442426/attachments/2698007/4683942/CAS2023_simulations_wCover.pdf); [3DS community](https://3dswym.3dexperience.3ds.com/post/simulia-community/band-diagram-analysis-cst-studio-suite-approach_2baxVWubTQmeJkx5vMTVQg)
- PETRA paper: commercial eigenmode solvers are "mostly real-valued"; TEMF's own complex-valued solver (not CST) handles coupler ports with second-order bases and MPI. — [ICAP2015 WEP36](https://inspirehep.net/files/f5fa81124143df6be5126be6e243c694)
- Context only, not CST: TEMF's code writes the problem as a generalised (quadratic when ports are present) eigenproblem with hierarchical bases; JD with a complex target region is used there. — [ICAP2012 MOADI1](https://proceedings.jacow.org/ICAP2012/talks/moadi1_talk.pdf)
- Third-party MCP tool schema (glama.ai) lists "accuracy default 1e-6, solver order 1-3, frequency estimate" for CST eigenmode; this is a hobbyist wrapper, NOT CST documentation, and is not used as a source for defaults. — [glama.ai listing](https://glama.ai/mcp/servers/ismailakdag/cst-studio-mcp/tools/cst_configure_eigenmode_advanced)

### Inferences
- A real-valued eigenmode solve plus perturbative Q (default) matches this project's `VᵀKV c = θ M c` plus wall-loss QoI in `src/qoi/`. The frequency-independent perturbation Q requires only the lossless H field on the wall.
- The gradient null space cannot be left in a FEM Nédélec eigenproblem; CST must shift past it (a target/shift above zero, consistent with the "frequencies above" and "estimate of highest frequency" options) or project it out. No source says which. The project's kernel projection is a legitimate stronger alternative, not a CST copy.

### Gaps
- Algorithm of the tetrahedral "general purpose" method, preconditioner, linear solver, shift/target choice, static-mode filtering and any Hodge/gradient projection: searched three times (search summaries, forum, CAS slides, 2006 CST overview PDF not reachable), found nothing public.
- Default accuracy value for the tetrahedral solver: not found (help page not reachable).
- Surface impedance in the tet eigenmode solver: a search summary of a recent CST overview lists surface-impedance models (tabulated surface impedance, Ohmic sheet, lossy metal, corrugated wall, coating, roughness) for the Eigenmode Solver on hexahedral or tetrahedral meshes; I could not open that page, so it is unconfirmed which of these apply to tet.

## 4. Tetrahedral vs hexahedral for accelerator cavities, and reported accuracy

### Takeaway
CST AG and CERN training both prefer the tetrahedral eigenmode solver for cavities; the only quantitative public comparison I could read pits CST MWS 2010 hex against a TEMF tetrahedral code, not against CST tet. One published CST tet result for a TESLA cell agrees with an independent reference to about 4e-8.

### Cited Findings
- CAS 2023: "The Eigenmode solver of CST supports both tetrahedral and hexahedral mesh elements, of which the tetrahedral are preferred as they are more precise and flexible for representing a geometry." For a multipacting PIC hand-off the tutorial switches to hex because exporting fields from the tet mesh gives "too large file sizes". — [CERN CAS 2023](https://indico.cern.ch/event/1294829/contributions/5442426/attachments/2698007/4683942/CAS2023_simulations_wCover.pdf)
- IPAC2011 WEPC093, SPL 704.4 MHz beta=1 five-cell: CST MWS 2010 hex eigenmode converges slowly: TM010 pi mode 701.3 to about 704.0 MHz with 100 million cells (704.006 MHz at 100 M cells), 111 h for 61 M cells up to 275 h for 100 M cells on a Xeon X5472 with 64 GB, with 31-51 GB of result storage. The TEMF tetrahedral code with curvilinear higher-order elements gave 704.398 MHz with 6.2 M tets (design 704.4). Conclusion: hex frequency-domain "is not an ideal approach" and the tetrahedral approach "is highly efficient". The tet code required tets symmetric about the beam axis, otherwise a spurious transverse Ey on axis spoiled shunt impedance (so far only for TM010). — [IPAC2011 WEPC093](https://accelconf.web.cern.ch/ipac2011/papers/wepc093.pdf)
- CST AG (2012): enhanced fast PBA (EFPBA) is the new default for hex solvers; curved tet eigenmode "quick convergence, < 2 min to calculate 9 modes". — [ICAP2012 MOAAI2](https://epaper.kek.jp/ICAP2012/talks/moaai2_talk.pdf)
- TESLA one-cell, MWS 2018, solver order 3, 200,771 order-5 curved tets: f = 1.27666401260 GHz vs Bembel reference 1.276664064 GHz, relative difference 4.018e-8. Sphere: errors listed in Q1. The authors state the IGA-BEM needs far fewer DOFs, and in their Fig. 6 the Bembel-vs-CST difference stalls at high order and small h, which they attribute to the Bembel reference being more accurate "provided the convergence of the contour integral approach behaves as observed". — [arXiv 2001.09686](https://arxiv.org/pdf/2001.09686)
- PETRA: in the 1st monopole passband the MAFIA hex FIT results approach the CST FEM tet values as hex cells grow (94,231 and 698,412 cells), while CST tet used 307,200 tets. — [ICAP2015 WEP36](https://inspirehep.net/files/f5fa81124143df6be5126be6e243c694)

### Inferences
- CST tet p3 on a smooth cavity delivers relative frequency differences of order 1e-8 to 1e-9 on sphere/TESLA with 5k-200k tets and accuracy set to 1e-12. That is the right order of magnitude for validating a p3 NGSolve pipeline against CST; differences below about 1e-7 on frequency would not be meaningful because the reference itself is uncertain.
- The need for beam-axis-symmetric meshes in the TEMF tet code is relevant to field quality metrics (R/Q) on unstructured meshes: field errors near the axis can matter more than frequency errors. Not shown for CST.

### Gaps
- No like-for-like CST hex vs CST tet table; no CST 2020+ accuracy study.
- Current (2024-2025) vendor guidance text: not found.

## 5. Sensitivity settings (accuracy, losses, lowest frequency above, symmetry)

### Takeaway
Documented knobs are Accuracy (relative residual; also the adaptation Delta F threshold), number of modes, frequency target options, loss handling, solver order and curved order, and PEC/PMC symmetry planes or periodic phase shifts. Defaults could not be read from primary help.

### Cited Findings
- Symmetry: the CAS tutorial applies "magnetic symmetry in the YZ- and XZ-planes" to the pillbox eigenmode run, which reduces tets; for periodic structures it advises: "It would be best to repeat the simulation later with different symmetry planes to ensure you get all modes", since a symmetry plane removes modes of the other symmetry class. — [CERN CAS 2023](https://indico.cern.ch/event/1294829/contributions/5442426/attachments/2698007/4683942/CAS2023_simulations_wCover.pdf)
- PETRA run uses PEC on the whole surface and PMC symmetry planes in CST MWS 2015 tet solves. — [ICAP2015 WEP36](https://inspirehep.net/files/f5fa81124143df6be5126be6e243c694)
- The tutorial sets number of modes by hand ("3 in Modes field") and leaves the adaptive refinement tick on. — [CERN CAS 2023](https://indico.cern.ch/event/1294829/contributions/5442426/attachments/2698007/4683942/CAS2023_simulations_wCover.pdf)
- Help text via search summary: Accuracy = relative residual for tet; "Frequencies above" and "all modes between fmin..fmax" (JDM); "Consider losses in postprocessing only" default; fixed-frequency evaluation of lossy materials otherwise. — [CST2013 Eigenmode Solver Parameters mirror](http://www.edatop.com/cst/CST2013/30591.html) (unreachable here)
- Published values actually used: accuracy 1e-12 with solver order 3 for benchmark-grade numbers. — [arXiv 2001.09686](https://arxiv.org/pdf/2001.09686)
- The eigenmode solver is not in the free Learning Edition (user report). — [3DS community Q&A](https://3dswym.3dexperience.3ds.com/question/3dexperience-edu-students/eigenmode-analysis-in-cst-studio-suite_mxp-8iB8SwqKnzQqCn5QHw)

### Inferences
- Symmetry planes change which modes appear (they filter by symmetry class); when mirroring CST for a K-lowest-modes comparison, compare on the full cavity, or on the same symmetry class.
- Setting both "frequencies above" and mode count most likely implements a shift/target for a shift-invert-type iteration; unconfirmed.

### Gaps
- Default Accuracy for the tet eigenmode solver, default number of modes, effect of each setting on run time: not found.
- Any treatment of spurious or static modes visible to the user (e.g. warnings, filtering): not found.
