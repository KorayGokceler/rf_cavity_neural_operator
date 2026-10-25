# Published accuracy benchmarks of the CST Studio Suite eigenmode solver for RF cavities

Scope: public sources only. Every number below was read from the primary PDF (downloaded and text-extracted) unless flagged "snippet only". "Calculated" means I derived it from the quoted numbers (not stated by the source). CST solver families: **FIT/hex** = Finite Integration Technique eigenmode solver on a hexahedral grid with PBA (Perfect Boundary Approximation) or FPBA; **FEM/tet** = curved-element tetrahedral eigenmode solver, introduced in CST MWS 2011 SP3 ([Microwave Journal, CST announcement](https://www.microwavejournal.com/articles/11362-cst-announces-curved-finite-elements-now-also-for-the-eigenmode-solver)).

Important attribution warning: several TEMF Darmstadt papers (IPAC11/12/13/14) use CST only for geometry/tetrahedral meshing and then solve with the group's own in-house FEM code. Those "tetrahedral" columns are NOT CST's own tet solver results. This is flagged wherever it matters.

## 1. Pillbox and other analytic cases: CST frequency/Q error vs analytic, at stated mesh density, hex and tet, convergence with mesh

### Takeaway
No public source gives a clean CST-vs-analytic TM010 pillbox table with mesh counts (searched; see Gaps). The best analytic evidence is: (a) a unit sphere with CST MWS 2018 FEM/tet, curved elements order 5, solver order 3: eigenvalue error 1.9e-7 at 21.6k DOF falling to 1.95e-9 at 176k DOF; (b) CST MWS 2009 FIT/hex PBA on 10 cm spheres at 2 mm mesh: relative frequency errors of order 1e-4 at 2 GHz growing to 1e-3...1e-2 near 9 GHz as cells per wavelength drop, with surface-field errors one to two orders worse; (c) a 100 mm cylinder with CST MWS (mesh type not explicit): relative frequency deviation < 1.6e-4 up to 35.7 GHz. Unlike a high-order FEM, hex FIT errors are O(1e-4) at best on curved geometry and degrade with frequency.

### Cited Findings

**CST MWS 2018, FEM/tet, unit sphere (analytic reference)**
- Setting: "Mesh: Tetrahedral, Desired accuracy: 1e-12, Solver order: 3rd (constant), Curved elements: up to order 5 (user defined)". Quantity: smallest eigenvalue of the unit sphere vs the closed-form (first root of a spherical Bessel function); Fig. 4 defines the error as the mean absolute difference of the computed first eigenvalue to the analytic one (it is the wavenumber-type eigenvalue, around 2.7 for the unit sphere; the table column is labelled only "error"). — [Kurz, Schöps, Unger, Wolf, arXiv:2001.09686, Table 2 and Sec. 7.1](https://arxiv.org/pdf/2001.09686)
  - 1089 elements, SD 6, 21,597 DOFs: error 1.8638e-07
  - 5292 elements, SD 10, 101,997 DOFs: error 5.4204e-09
  - 9191 elements, SD 12, 175,818 DOFs: error 1.9524e-09
  - For comparison in the same paper: volume IGA (GeoPDEs) p=3, 84,560 DOFs: 1.10e-08; p=5, 69,600 DOFs: 2.15e-12; the IGA boundary-element method reaches 7.17e-12 with 1,728 DOFs. The authors state the difference to commercial tools "is even more pronounced". — same source, Tables 1-2
- Run times/memory not given in that paper.

**CST MWS 2018, FEM/tet, one-cell TESLA cavity (no analytic solution; independent high-order reference)**
- Setting: solver order 3 (constant), mesh of 200,771 curved tetrahedral elements of order 5. CST result 1.27666401260 GHz; the authors' contour-integral IGA-BEM reference (p=5, h=1/8) gives 1.276664064 GHz; stated relative difference 4.018e-08. The paper says stagnation w.r.t. the CST solution at higher order / small h in its Fig. 6 "suggests that the Bembel reference solution is more accurate" (authors' interpretation, conditional on their method's convergence behaviour). — [arXiv:2001.09686, Sec. 7.2](https://arxiv.org/pdf/2001.09686)
- Calculated: 4.0e-8 x 1.2767 GHz is about 51 Hz difference, i.e. CST frequency is converged to well below 1 kHz at this extreme setting.

**CST MWS 2008/2009, FIT/hex PBA, spheres (analytic Borgnis solutions), compared with Omega3P 7.2.1, HFSS 11.0, ANSYS 11.0, VORPAL 4.0**
- Models: single sphere a = 10 cm and "double sphere" (sphere minus two cones, b = 10 cm, d = 5 cm); all modes up to 10 GHz, azimuthally independent, 18-degree sector (1/40 or 1/20 of the sphere). — [Tian et al., JLab, PAC09 TH5PFP093](https://accelconf.web.cern.ch/pac2009/papers/th5pfp093.pdf)
- MWS settings: 1 CPU, 3 GB RAM, run in several frequency bands; "2mm mesh size was fixed by setting the upper frequency limit"; AKS eigensolver for the lower band and JDM for the higher band; CPU time "from a few minutes to 6 hours"; accuracy of each solution checked after the run via the Maxwell equations. MWS (and HFSS) could not solve the double sphere above about 6 GHz in reasonable CPU time. Same source.
- Comparison settings: Omega3P with 2 mm unstructured 10-node tets from CUBIT, residual < 1e-7; HFSS 2 mm max mesh length, adaptive passes with 0.01% max frequency change per pass. Error definition: |X_sim - X_analytic| / |X_analytic|. Same source.
- Reported (text): Omega3P "consistently more accurate", "well below 1e-5"; "with increasing frequency, the mesh number per wavelength decreases, therefore relative errors for most codes increase as expected"; ANSYS roughly constant at about 2e-5. Same source.
- Read off Fig. 3 by eye (log plot, approximate, factor ~2-3 uncertainty; these are my readings, not numbers stated by the authors): MWS frequency error on the single sphere is about 1e-4 to 3e-4 at 2 GHz and climbs to about 1e-3 to 1e-2 at 8-9 GHz; on the double sphere about 1e-4 at 1 GHz rising to roughly 1e-3 near 5-6 GHz. HFSS sits near 1e-4 to 1e-3 and Omega3P near 1e-7 to 1e-5.
- Surface-field accuracy (Figs. 4-5): text says Omega3P best, ANSYS inconsistent, "VORPAL is still comparable with MWS, but HFSS shows very poor accuracy for all cases". The plotted MWS errors for maximum surface magnetic and electric fields are visibly at the 1e-2 to 1e-1 level (read off by eye, approximate), i.e. 1-3 orders worse than the frequency error. Same source.

**CST MWS (version not stated), 100 mm cylinder**
- Model: cylindrical cavity R = 100 mm, L = 100 mm, fundamental 1.15 GHz; TM(2,n,2) modes up to 35.71 GHz computed with "CST MWS Eigenmode Solver". Reported: "relative deviation is less than 1.6e-4 inside a very large frequency range (up to 35.71 GHz)", and "accuracy degrades with increasing frequency due to the fact that the number of simulated mesh cells is limited by the available memory"; Fig. 4 plots "intended steps per wavelength" which "reduce from 12 GHz on due to full memory usage". The text of the paper recommends FEM with curved tets for curved cavities, but the "steps per wavelength" language suggests the hex/FIT solver was used for this cylinder; the paper does not state the mesh type for the frequency claim explicitly. — [Brackebusch and van Rienen, ICAP2012 MOADC2](https://accelconf.web.cern.ch/ICAP2012/papers/moadc2.pdf)
- Same paper, field-near-wall quality on the R = 100 mm cylinder: FEM-computed fields have a defective boundary range of 0.3 micrometre vs 70 micrometre for FIT-computed fields (FIT fields are allocated on the dual grid at a distance from the boundary); "FEM combined with a tetrahedral mesh and curved elements proved to reproduce the boundary much more precisely than the hexahedral dual grid of FIT". Same source.

**Pillbox with CST including beam pipes (not an eigenmode accuracy test of CST itself)**
- A 2026 arXiv paper builds a CST pillbox-like cavity (Table I: cavity length g = 260 mm, radius R = 210 mm; beam pipe length L = 200 mm, radius r = 10 mm) and compares analytic pillbox R/Q and frequency with fitted wakefield results and CST eigenmode results. Frequencies 0.546 GHz (TM010) and 0.794 GHz agree with analytic to three digits; deviations grow with frequency because beam pipes perturb the ideal closed-cavity modes. Table II compares wake-fit with theory, not CST-eigenmode with theory; CST-eigenmode vs theory appears only graphically in Fig. 4. No mesh settings, so it is not a usable error benchmark. — [Wang et al., arXiv:2607.04312](https://arxiv.org/pdf/2607.04312)

**Reference FEM-vs-analytic mesh convergence (COMSOL, NOT CST; shown as a yardstick for what a 2nd-order curved FEM does)**
- COMSOL RF cavity-resonator model, TM010 of a cylindrical cavity (analytic f = 9.412 GHz, Q = 8065): h_max 9.458 GHz / Q 7891; h_max/2 9.419 / 8004; h_max/4 9.411 / 8056; h_max/8 9.411 / 8065. Rectangular TE101 (analytic 9.273 GHz, Q 7770): 9.706/7039, 9.283/7687, 9.273/7765, 9.273/7770. Spherical TM011 (analytic 9.698 GHz, Q 14594): 9.752/14121, 9.723/14430, 9.701/14616, 9.697/14641. The document notes the mesh represents curved surfaces by second-order polynomials by default, and convergence is "fastest for the rectangular cavity and slowest for the spherical cavity". — [COMSOL, Computing Q-factors and resonant frequencies of cavity resonators](https://doc.comsol.com/6.0/doc/com.comsol.help.models.rf.cavity_resonators/models.rf.cavity_resonators.pdf)
- Calculated: coarsest COMSOL TM010 error +0.49%, then +0.07%, then -0.01%; Q error -2.2%, -0.8%, -0.1%, 0.0%.

### Inferences
- Frequency error of CST FEM/tet with curved order-5 elements and solver order 3 is in the 1e-7 to 1e-9 range on smooth analytic geometry (sphere) and about 4e-8 on a TESLA cell relative to an independent reference; CST FIT/hex PBA is about three to four orders worse at comparable geometry/wavelength regimes (1e-4 to 1e-3 at moderate frequency). These come from different papers, geometries and CST versions (2009 vs 2018), so the gap is indicative, not a controlled comparison.
- A "match to CST" tolerance for frequency of 1e-4 relative is realistic if CST was run in FIT mode with a sensible mesh, and 1e-6 or better only if CST was run in FEM mode with high order and converged mesh. Mesh-converge your own solver first: if your solver is at 1e-6 and CST at 1e-4, the difference you observe is CST's discretization error, not yours.
- Surface-field quantities (Epk, Bpk) on hex FIT are one to two orders worse than frequency (from the sphere benchmark), so matching them to better than percent level is unlikely.

### Gaps
- No public CST-vs-analytic TM010 pillbox table with cells-per-wavelength and relative error was found. The GSI page "cavity simulations in CST Studio Suite" (https://web-docs.gsi.de/~ssanjari/cavity-simulations-in-cst-studio-suiter.html) reportedly compares CST with analytic and Superfish for a pillbox, but returned HTTP 404 when fetched, and the search snippet had no numbers.
- Q0 accuracy vs analytic for CST (e.g. pillbox with finite conductivity, or spherical lossy cavity) was not found; the Papantonis and Lucyszyn lossy-sphere paper (Imperial College, PIER 2015) tested HFSS and COMSOL, not CST.
- No CST-vendor application note with quantitative accuracy was found in public search; the only vendor material found is the 2011 curved-element press release (qualitative).
- CST version is not stated in the Brackebusch cylinder result and the mesh type behind the 1.6e-4 claim is ambiguous.

## 2. TESLA / elliptical SRF cavities and other accelerator cavities: CST vs SUPERFISH, ACE3P/Omega3P, HFSS, COMSOL, Palace, GdfidL, MAFIA, measurement

### Takeaway
Direct public CST-vs-ACE3P/Omega3P or CST-vs-Palace tables for TESLA/elliptical cavities were not found. Available hard numbers: CST vs SUPERFISH vs HFSS on a quarter-wave resonator (frequency -8 kHz / -8e-5; Hpk +0.1%; Epk +1.7%; Q0 +0.4%), CST FIT/hex 80 M cells vs a curvilinear-tet FEM on the 704 MHz 5-cell SPL cavity (CST lower by 30-37 kHz, about 5e-5, at 80 M cells and about 670 h), CST vs HFSS on 4 GHz crab-cavity HOMs (frequency 1e-5 to 1e-4, Q differences 12% to 41%), and CST vs HFSS/MAFIA on the 3.9 GHz cavity (within 10 MHz).

### Cited Findings

**HIE-ISOLDE quarter-wave resonator, 101.7 MHz: CST MWS vs SUPERFISH (reference) vs HFSS**
- Simplified model without beam ports (azimuthally symmetric so SUPERFISH is exact; SUPERFISH mesh step 2.5 mm, finer mesh does not change results). CST: one quarter modelled, mesh refinement steps up to "about 1,000,000 meshcells" (CST version not stated; arXiv 2009; mesh type not stated but cell language implies hex). HFSS: 1/16 slice, surface mesh refinement 2 micrometre, about 110,000 tetrahedra. Fields normalised to 1 J stored energy. — [D'Elia, Jones, Pasini, arXiv:0910.0409, Table 1](https://arxiv.org/pdf/0910.0409)

| quantity | SUPERFISH | CST | HFSS | CST-SF | HFSS-SF |
|---|---|---|---|---|---|
| f (MHz) | 101.674 | 101.666 | 101.674 | -8 kHz (calculated, -7.9e-5) | 0 |
| Hpeak (kA/m) | 16.711 | 16.733 | 16.763 | +0.1% | +0.3% |
| Epeak (MV/m) | 11.38 | 11.57 | 11.6 | +1.7% | +1.9% |
| Q0 | 11795 | 11844 | 11746 | +0.4% | -0.4% |

- With beam ports added (no SUPERFISH possible) CST Epeak and Hpeak stayed consistent with HFSS but CST Q0 came out "overestimated, about 14000" versus about 11700 from HFSS; the authors attribute this "probably" to a geometry problem in the CST model around the racetrack-shaped noses, because changing the noses to a circular shape made it compatible again. Same source. (Authors' own explanation; not proven.)

**CERN SPL beta=1, 704.4 MHz 5-cell: CST FIT/hex vs a curvilinear-tet FEM (in-house TEMF code; CST used only for geometry/mesh)**
- CST MWS 2011 eigenmode solver, hex PBA, 80 million mesh cells, vs in-house FEM with 6.2 million curvilinear tets (Table 1 of [Liu et al., IPAC2012 MOPPC059](https://accelconf.web.cern.ch/IPAC2012/papers/moppc059.pdf)):

| mode | CST hex f (MHz) | R (ohm) | FEM tet f (MHz) | R (ohm) | diff f (calculated) |
|---|---|---|---|---|---|
| TM010 1/5 pi | 692.416 | 0.002 | 692.446 | 0.002 | -30 kHz |
| 2/5 pi | 695.643 | 0.037 | 695.676 | 0.037 | -33 kHz |
| 3/5 pi | 699.710 | 0.010 | 699.744 | 0.011 | -34 kHz |
| 4/5 pi | 703.065 | 0.071 | 703.101 | 0.067 | -36 kHz |
| 5/5 pi | 704.361 | 565.468 | 704.398 | 565.452 | -37 kHz (-5.3e-5); R: +2.8e-5 |

- Design frequency 704.4 MHz. The hex solver took "almost 670 hours" for 80 M cells and the authors call the convergence "not very satisfied". The FEM converged to 704.397 MHz "very fast" with 6.0 M cells (Fig. 5), with MPI parallelisation. A time-domain CST Transient run with 49,630,173 FPBA cells gave only 703.755 MHz for the pi mode (calculated: -643 kHz, -9.1e-4 vs 704.398), taking about 50 hours on a GPU cluster. Same source.
- Earlier version of the hex convergence plot (CST MWS 2010; Xeon X5472 3.0 GHz, 64 GB DRAM): TM010 frequency vs mesh cells up to 100 M cells; Table 2 gives eigenmode solve time and disk: 61 M cells: 111 h, 31 GB; 71 M: 238 h, 36 GB; 88 M: 256 h, 40 GB; 100 M: 275 h, 51 GB. — [Liu et al., IPAC2011 WEPC093](https://jacow.org/IPAC2011/papers/WEPC093.pdf)
- Caveat: the lower-frequency hex result is compared with a different code's FEM result, which is assumed converged (704.397/704.398 near the 704.4 MHz design value); the paper does not give an independent analytic reference.

**CST vs HFSS on a 4 GHz crab-cavity module HOMs (Elettra 2.0), CST eigenmode (version not stated) vs HFSS eigenmode with PML**
- Same model; mesh not stated beyond "the discrepancy in Q ... can be attributed mainly to the way mesh generation and Q factor calculation are performed in HFSS and CST". Table 3 (f in GHz / Q): — [Shafqat et al., Elettra/CERN, JACoW THP2044](https://indico.jacow.org/event/95/papers/14374/files/5595-THP2044.pdf)
  - CST 4.00339 / 6.45e5 vs HFSS 4.00347 / 5.45e5 (calculated: df = -80 kHz, 2e-5; CST Q +18%)
  - 4.00341 / 6.45e5 vs 4.00349 / 5.46e5 (df = -80 kHz; CST Q +18%)
  - 4.01279 / 2.33e5 vs 4.01318 / 2.08e5 (df = -390 kHz, 9.7e-5; CST Q +12%)
  - 4.01345 / 5.14e5 vs 4.01384 / 3.87e5 (df = -390 kHz; CST Q +33%)
  - 4.04001 / 3.84e8 vs 4.03997 / 6.54e8 (df = +40 kHz, 1e-5; CST Q -41%)
- Same paper: CST eigenmode vs CST wakefield (time-domain) frequencies agree "very well"; impedance magnitudes differ for Q >= 1e5 because truncated wake potentials underestimate high-Q peaks.

**CLIC 12 GHz crab/structure, CST MWS vs HFSS (slide from CERN review of codes, Grudiev, c. 2012)**
- MWS (PEC walls, Perfect Boundary Approximation, 134,912 hexahedra per quarter, lines per wavelength = 40, lower mesh limit 40, mesh line ratio limit 40) vs HFSS (copper walls, 8,223 tetrahedra per quarter, surface approximation 5 mm, aspect ratio 5): f 11.9941 vs 11.9959 GHz (calculated: -1.8 MHz, -1.5e-4); Q 6395 vs 6106 (but PEC vs Cu wall model differs); Rt/Q 54.65 vs 53.78 ohm (+1.6%); Esurf/Et 3.43 vs 3.28 (+4.6%); Hsurf/Et 0.0114 vs 0.0106 (+7.5%). The slide itself lists as a CST shortcoming "Cartesian mesh: especially in FD can result in less accurate calculations of frequency, Q-factor, surface fields compared to tetrahedral mesh (HFSS, COMSOL, ACE3P)" and "of three eigenmode solvers only one takes into account losses but it is iterative and very slow". — [Grudiev, Simulation packages and review of codes, CERN indico 164082](https://indico.cern.ch/event/164082/contributions/246170/attachments/195880/274792/Simulation_packages_and_Review_of_Codes_AG.pdf). Presenter opinion plus one case; date/CST version not shown on the extracted slide.

**3.9 GHz third-harmonic cavity (FLASH/XFEL): CST hex vs MAFIA and HFSS**
- CST MWS Eigenmode solver, hex mesh, a quarter structure with symmetry planes, EE and MM boundary conditions, solver accuracy 1e-6 relative residual, max mesh step 1.1 mm (about 2.1 million cells per quarter) for the accelerating mode. "The differences are within 10 MHz for both boundary conditions from both simulation codes" (CST vs MAFIA, CST vs HFSS; HFSS values taken from another group's work). — [Zhang, Baboi, Jones, arXiv:1206.2782](https://arxiv.org/pdf/1206.2782). CST version not stated. The 10 MHz is an envelope over all modes up to the 8 GHz sextupole band (so at most about 1e-3 relative at multi-GHz); not a TM010 figure.

**ILC cavity coupler RF kick: CST hex vs HFSS vs COMSOL vs Omega3P (an integrated field quantity, not eigenfrequency)**
- Fully symmetric hexahedral mesh locked to the axis for CST; HFSS with gradual three-zone mesh; highly regularized COMSOL tets. All codes gave nearly the same electric field while "OMEGA3P gives about 20% lower magnetic field" (an independent SLAC run, different mesh), and estimated RF-kick convergence errors were "better than 10% for HFSS and less than 5% for CST and COMSOL". — [Lunin et al., IPAC10 WEPE034](https://accelconf.web.cern.ch/IPAC10/papers/wepe034.pdf)

**CST Qext via waveguide ports vs absorber boundary (SRF cavities, curved-tet eigenmode solver)**
- LCLS-II cavity, 446,220 mesh cells, 1 h 13 min with absorbers vs 435,658 cells, 2 h 12 min with waveguide ports (factor ~1.8 longer); with ports, two modes came out with identical fl and Qext; Qext values from perturbation and from lossy-absorber methods differ strongly (e.g. mode 1: 9.45e3 vs 6.88e3; mode 2: 2.90e4 vs 2.11e4; mode 3: 1.33e4 vs 9.65e3). CST 2015 SP4; a mode-number sorting bug was found in CST 2014. "In each case, inaccuracies usually rise with an increase of the Qext-value." Computing all HOMs up to 3.2 GHz in the full 3D LCLS-II cavity with fine HOM-coupler meshing (300-450 curved surface tets) "required a few working days" on a 2-processor workstation (Xeon E5-2670 v3, 128 GB). — [Marhauser, SRF2015 THPB003](https://accelconf.web.cern.ch/SRF2015/papers/thpb003.pdf). The method also states results vary "significantly depending on symmetry planes and/or the total number of modes used" for high-Qext FPC optimisation.

**Higher-order-mode R/Q etc. for beta=0.63 650 MHz cavity**
- Paper reports CST 3D design values (R/Q 333.5 ohm, G 192.7 ohm, Epk/Eacc 2.34, Bpk/Eacc 4.63 mT/(MV/m)) but no cross-code table. — [Wen et al., arXiv:1505.02885](https://arxiv.org/pdf/1505.02885). Not a benchmark.

**Palace/COMSOL/HFSS (non-CST context)**
- For a superconducting-circuit CPW resonator (not an accelerator cavity): Palace AMR O2 9.996 GHz, COMSOL 9.991 GHz, Ansys HFSS 10.004 GHz; differences to Palace 0.020% and 0.110%. No CST. (Snippet only; the PDF was not opened.) — [Open-source highly parallel EM simulations for superconducting circuits, arXiv:2511.01220](https://arxiv.org/pdf/2511.01220)

### Inferences
- Frequency agreement between CST and other codes at converged settings is typically 1e-5 to 1e-4 relative (kHz to a few hundred kHz at 0.1-4 GHz): -8 kHz at 102 MHz, -30 to -37 kHz at 704 MHz, -40 to -390 kHz at 4 GHz, -1.8 MHz at 12 GHz (the last with different wall models and a coarse HFSS mesh, so not a clean comparison).
- Peak-field ratios between CST and an independent code are 0.1-2% on a well-meshed axisymmetric-like case (Hpk better than Epk), and 5-8% on a coarser 12 GHz case; Q0 differs 0.4% when conditions match, but 12-41% for high-Q HOMs with different port/PML treatments, and can be off by +20% (14000 vs 11700) if geometry detail is poorly resolved.
- Because Q and Qext depend on the wall-loss model (PEC vs Cu, perturbation vs lossy-material, ports vs absorbers), a "matches CST" Q target needs the same loss model; tolerance of 10-20% is more realistic for high-Q modes, ~0.5% for Q0 of a simple cavity.
- The shunt impedance/R/Q of the fundamental agrees between hex FIT and curvilinear FEM at the 3e-5 level (565.468 vs 565.452 ohm for the pi mode) when the hex mesh is huge (80 M cells).

### Gaps
- No public side-by-side CST vs SUPERFISH vs ACE3P (Omega3P) for the TESLA 9-cell/1-cell with R/Q, G, Epk/Eacc and Bpk/Eacc was found. The JAEA ADS paper (JPS Conf. Proc. 1350 / BB2019-0054) reportedly benchmarked SUPERFISH vs CST; the PDF fetched was not parsable text and no numbers were extracted.
- No CST vs Palace, CST vs NGSolve, CST vs GdfidL eigenmode tables were found. EPAC2006 WEPCH109 (MAFIA, GdfidL, MWS, HFSS on the 500 MHz EU cavity) was read in full: it compares MAFIA vs GdfidL time-domain HOM spectra against measurement and gives no MWS eigenmode numbers.
- No published CST-vs-measurement frequency table for an SRF cavity with kHz-level numbers was found (the HIE-ISOLDE paper has room-temperature prototype frequency measurements but I did not extract the CST-vs-measurement figure).
- A 2012 TEMF statement that CST/Matlab are "clearly separated" in solve time on a 9-cell TESLA cavity (ICAP2012 FRSAC2) concerns timing (CST tet mesh, eigenvalue accuracy 1e-9), not result accuracy; no run times were quoted in text.

## 3. Reported sensitivities: mesh density, PBA vs staircase, curved tets and order, symmetry, adaptive refinement, run time and memory

### Takeaway
Reported sensitivities are large for the hex FIT solver on curved SRF cavities (tens of millions of cells and hundreds of hours for 5e-5 relative frequency accuracy), small for curved-tet FEM (order up to 5 with solver order 3 gives 1e-7 to 1e-9 on spheres). Symmetry planes and boundary conditions affect Qext in CST. Specific "lines per wavelength" figures are rare: only the CLIC example (40 lines/wavelength, hex PBA) and a 12 steps-per-wavelength-limited cylinder case.

### Cited Findings
- Hex mesh sensitivity (SPL 5-cell, 704 MHz, CST MWS 2010/2011, PBA): TM010 pi-mode frequency converges slowly with cell count (convergence plotted from 0.1 M to 80-100 M cells); 80 M cells gave 704.361 vs the FEM 704.398 (design 704.4). Solve time rose from 111 h at 61 M cells to 275 h at 100 M cells (MWS 2010) and 670 h at 80 M cells (MWS 2011; different machine/mesh scheme). "A lower number of hexahedral elements cannot represent the contours of the elliptical resonator precisely. This can result in a slow convergence." — [IPAC2012 MOPPC059](https://accelconf.web.cern.ch/IPAC2012/papers/moppc059.pdf), [IPAC2011 WEPC093](https://jacow.org/IPAC2011/papers/WEPC093.pdf)
- PBA vs FPBA and cell numbers: the same mesh-based approach gave an FPBA transient frequency error of -9e-4 relative with 49.6 M cells (calculated), much worse than the eigenmode solver at 80 M cells, i.e. a time-domain peak-picked frequency is far less accurate than the eigenmode value. Same IPAC2012 source.
- Number of mesh cells vs frequency: on the 10 cm spheres, relative frequency errors for MWS increase with frequency because "mesh number per wavelength decreases" (2 mm fixed mesh via upper frequency limit); and on the cylinder test the intended steps per wavelength were reduced above 12 GHz due to full memory. — [PAC09 TH5PFP093](https://accelconf.web.cern.ch/pac2009/papers/th5pfp093.pdf), [ICAP2012 MOADC2](https://accelconf.web.cern.ch/ICAP2012/papers/moadc2.pdf)
- Curved boundary representation: FIT dual-grid fields have a 70 micrometre defective boundary zone vs 0.3 micrometre for FEM with curved tets (R = 100 mm cylinder). Same ICAP2012 source. Hex grids with mesh lines chosen to match iris and equator radii exactly were used for the 3.9 GHz cavity (about 2.1 M cells per quarter, 1.1 mm max step). — [arXiv:1206.2782](https://arxiv.org/pdf/1206.2782)
- Tet order/curved elements: CST MWS 2018 on the unit sphere with curved elements up to order 5 and solver order 3: error drops from 1.9e-7 to 1.95e-9 as elements go from 1089 to 9191 (DOF 21.6k to 176k): convergence rate between the first and last points is about 2 orders in about 8x the DOFs (calculated). A TESLA cell at 200,771 order-5 curved tets: 4e-8 vs an independent reference, with stagnation toward the CST value at high order and small h. — [arXiv:2001.09686](https://arxiv.org/pdf/2001.09686)
- CST version/feature timeline: FEM eigenmode solver with curved elements of arbitrary order released with MWS 2011 SP3 ([Microwave Journal](https://www.microwavejournal.com/articles/11362-cst-announces-curved-finite-elements-now-also-for-the-eigenmode-solver)); vendor wording: "True Geometry Approximation" improves accuracy. The announcement gives no numbers.
- Symmetry and mode count: for fundamental-power-coupler Qext optimisation in CST (curved tets), "results have been found to vary significantly depending on symmetry planes and/or the total number of modes used"; recommended to compute 4-10 modes sequentially; solver may converge slowly. — [Marhauser, SRF2015 THPB003](https://accelconf.web.cern.ch/SRF2015/papers/thpb003.pdf)
- Boundary-condition (EE/MM) choice matters for dipole-mode R/Q comparisons: in the Darmstadt third-harmonic study "large deviations for different boundary conditions" were observed. — [arXiv:1206.2782](https://arxiv.org/pdf/1206.2782) (qualitative)
- Adaptive refinement: the HIE-ISOLDE CST study used a mesh refinement sequence (up to about 1 M cells) and plotted peak E and H vs step (Fig. 4), so the peak fields converge with refinement; the HFSS comparison used a 2 micrometre surface approximation (about 110k tets). Specific CST adaptive-refinement numbers per pass were not extractable from the figure. — [arXiv:0910.0409](https://arxiv.org/pdf/0910.0409)
- Run time and memory: MWS 2009 sphere runs on 1 CPU / 3 GB: minutes to 6 h per band; 2012 hex SPL runs: 670 h; LCLS-II HOM runs: "a few working days"; SPL FEM with 6.2 M tets using MPI is far cheaper (no hour figure given). Disk: 31 to 51 GB for 61-100 M cells. See citations above.
- Unrelated Q sensitivity: in a ferrite-loaded low-Q (Q around 0.7 to 0.9) cavity, CST Q and shunt impedance differed from the analytic model by 50-90% and 50-70%; the cited text says analytic and simulated results agree very well for Q > 2 but not for Q < 2. Not an SRF case. — [arXiv:1504.02257](https://arxiv.org/pdf/1504.02257)

### Inferences
- A rule of thumb from these data: hex FIT needs on the order of 1e7-1e8 cells for ~5e-5 relative frequency accuracy on a 5-cell elliptical cavity at 700 MHz, i.e. roughly 1e6 cells per cavity cell, and still lands 30-40 kHz low; curved-tet FEM reaches the same accuracy with ~6 M elements in other codes, and CST's own FEM solver demonstrates 1e-8 on a TESLA cell with 2e5 order-5 elements.
- "Mesh lines locked to iris and equator radii" is a standard trick to improve hex FIT results.
- Symmetry planes with CST affect high-Qext results; for a team validating against CST, avoid comparing Qext with symmetry/mode-count variations unreported.

### Gaps
- No published study varies CST FEM solver order systematically on an accelerator cavity (only the 2018 sphere table by element count/SD).
- Few "lines per wavelength" or "steps per wavelength" values are published with accompanying error; only the CLIC example (40 lines per wavelength) and the 12 steps-per-wavelength remark. No curve of error vs lines per wavelength was found.
- Adaptive-mesh (CST) pass-by-pass frequency changes were not published in extractable text.
- Palace/NGSolve/GdfidL sensitivity comparisons with CST are absent.

## 4. Known systematic biases (frequency always above/below, peak fields overestimated on hex meshes, Q bias)

### Takeaway
The data show CST hex FIT frequencies below the converged value (low by 30-37 kHz on SPL; -8 kHz and -1.5e-4 vs references in two other cases) and CST peak E exceeding the SUPERFISH reference by 1.7% in one case; but there are too few cases for a general rule, and the sign of FIT error may be geometry dependent.

### Cited Findings
- CST FIT hex frequencies were below the curvilinear-tet FEM values for all five TM010 modes of the SPL cavity (30-37 kHz low, i.e. -4e-5 to -5e-5 relative), and the time-domain value was low by 0.64 MHz (-9e-4). — [IPAC2012 MOPPC059](https://accelconf.web.cern.ch/IPAC2012/papers/moppc059.pdf)
- CST was below SUPERFISH by 8 kHz in the HIE-ISOLDE comparison (101.666 vs 101.674 MHz), while peak E from CST was +1.7% above SUPERFISH and peak H +0.1% above; HFSS gave the same sign (+1.9%, +0.3%). — [arXiv:0910.0409](https://arxiv.org/pdf/0910.0409)
- CST MWS was below HFSS by 1.8 MHz (11.9941 vs 11.9959 GHz, -1.5e-4) in the CLIC example, with CST surface-field ratios 4.6% (E) and 7.5% (H) higher than HFSS; walls differ (PEC vs Cu). — [Grudiev slides](https://indico.cern.ch/event/164082/contributions/246170/attachments/195880/274792/Simulation_packages_and_Review_of_Codes_AG.pdf)
- CST FIT frequency error rises with frequency (fixed 2 mm mesh): from about 1e-4 at 2 GHz to 1e-3 or worse near 9 GHz in the sphere benchmark (the sign is not readable from the log plot, which shows absolute relative error). — [PAC09 TH5PFP093](https://accelconf.web.cern.ch/pac2009/papers/th5pfp093.pdf)
- CST Q: slightly above SUPERFISH by 0.4% in the matching case; strongly overestimated (+about 20%) when the geometry around racetrack noses was poorly modelled; CST Q higher than HFSS for four of five HOMs in a crab-cavity module (+12 to +33%) but lower (-41%) for one very high-Q mode (3.8e8). — [arXiv:0910.0409](https://arxiv.org/pdf/0910.0409), [THP2044](https://indico.jacow.org/event/95/papers/14374/files/5595-THP2044.pdf)
- Ordering and port bugs: CST 2014 had a mode-numbering bug in sorting unloaded vs loaded frequency; with waveguide ports in CST 2015 SP4 two modes had identical fl and Qext. — [Marhauser SRF2015](https://accelconf.web.cern.ch/SRF2015/papers/thpb003.pdf)
- Another group's comment (Arxiv 2305.05918 IERD paper): "for modes above fc, CST doesn't yield accurate resonance frequencies" when waveguide ports are present, while Omega3P agrees better with their integral-equation solver; no numbers. — [IERD resonance-detection paper, arXiv:2305.05918](https://arxiv.org/pdf/2305.05918)

### Inferences
- Frequency from hex FIT is likely biased low (not high) on elliptical cavities; this is supported by three cases but not proven (two of them against other codes' values, not against analytic), and I found no source that states a bias direction explicitly.
- Peak surface E on CST in the HIE-ISOLDE case was above the exact reference; HFSS did the same, so this may be a general "pointwise maxima converge from different sides" effect rather than a hex-mesh bias. Two codes with different mesh families showing the same sign argues against a hex-specific overestimation claim.
- For validating an NGSolve solver: pick CST in FEM mode with order >= 3 and 1e5+ tets for a frequency reference, and expect field-maximum agreement of only 1-2%.

### Gaps
- No source explicitly documents "CST hex overestimates peak fields due to edge singularities"; a search for this found nothing definite (the Fermilab paper by Lunin et al., ICAP2012 FRAAI1, mentions less than 1% surface-field error for HFSS with an artificial surface mesh, not CST).
- The Cornell lecture (Hoffstaetter, 08S_688) reportedly quotes CST's marketing claim of an order-of-magnitude improvement from PBA; the PDF could not be fetched (SSL error) so this is unverified and not used above.
- No statistics on bias sign from a larger sample of cavities.
