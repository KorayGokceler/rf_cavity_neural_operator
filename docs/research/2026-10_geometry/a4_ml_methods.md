# A4: Data-driven and computational RF cavity design, compared with our generator

*Area: ML surrogates and learned solvers for RF cavities, classical shape optimisation, the accuracy designers need,
manufacturing shape errors, figure-of-merit (FoM) conventions. Covers all families. Date: 2026-10-22.*

## 0. How to read this report (method and reliability)

- **Access problem.** WebFetch failed on every host with a DNS error (`getaddrinfo ENOTFOUND`, even for
  en.wikipedia.org). Direct downloads through the sandbox proxy returned HTTP 403 for every host tried (arxiv.org,
  jacow.org, osti.gov, cds.cern.ch, journals.aps.org, semanticscholar.org, openalex.org). So **every statement about
  a source comes from web-search result excerpts**: titles, abstract fragments and sentences quoted by the search
  engine. None comes from a full text. Treat every number as abstract-level. It is quoted as the excerpt gave it,
  with its URL, and was not checked against the paper's tables.
- **Budget.** I ran about 75 searches. Then the shared web-search budget ran out (200 calls per turn, shared by
  every agent), so some leads stayed open (section 7).
- **Markers.** [Q] means the source gives a number used here. *(estimate)* marks my own arithmetic.
  *(inference)* marks my interpretation. "Excerpt" means text the search engine returned.
- **Our side.** Facts about our pipeline come from `docs/24`, `docs/28`, `docs/30`, `docs/31`,
  `src/data_gen/dataset_generator_3d.py` and `src/data_gen/highorder.py`. One measurement (deformation magnitudes,
  §4.3) used a copy of `deform_map` in the scratchpad
  (`scripts/deform_mag.py`).

## 1. Scope and sources

### A. ML and data-driven RF cavity design
1. **S1 [Q]** Fermilab (2026), *Neural-Network Inverse Design of SRF Cavities and Transmons for Bosonic Quantum
   Computation*, arXiv 2607.02289, https://arxiv.org/abs/2607.02289. Secondary summaries:
   https://pith.science/paper/2607.02289 and https://quantumzeitgeist.com/fermi-accelerator-laboratory-inverse-design-srf/.
   A DNN inverse map from passband width and the axial E_z profile to spline control points, trained on
   2000 + 2000 COMSOL geometries, reaching about 5 %.
2. **S2 [Q]** Y. Wang, Y. Tang, C.-F. Wu, G. Feng, *Multiobjective Bayesian optimization for the shape design of rf
   cavity in particle accelerators*, Phys. Rev. Accel. Beams (2026), DOI 10.1103/mrr8-z48f,
   https://journals.aps.org/prab/abstract/10.1103/mrr8-z48f. Data and code: https://zenodo.org/records/18829772.
   Gaussian-process MOBO with equality constraints on a 499.65 MHz cavity; a Pareto front within 1000 simulations.
3. **S3** A. H. Shali, M. Fukuda, H. Kanda, T. Yorita (Osaka), *Multi-objective optimization of ring cyclotron RF
   cavity using neural network ensembles with uncertainty quantification*, IPAC'25 WEPS066,
   https://indico.jacow.org/event/81/contributions/8517/. Thesis abstract:
   https://www.phys.sci.osaka-u.ac.jp/daigakuin/2026/abstracts/shali.pdf. NURBS shape, HFSS labels, an NN ensemble
   surrogate with uncertainty, MOGA, and Monte Carlo over manufacturing deviations.
4. **S4** G. Diaz-Aylwin, J. Neighbor, A. Malkani Talwar, R.-Y. Zhang, H. B. Moss, *Example-driven
   Parametrisations for Bayesian Shape Optimisation* (BOLD), arXiv 2610.11984 (2026),
   https://arxiv.org/abs/2610.11984. A PCA of deformations between example designs gives a linear search space,
   used with GP Bayesian optimisation on a 3D RF-dipole cavity.
5. **S5** M. Kranjčević, A. Adelmann, P. Arbenz, A. Citterio, L. Stingelin, NIM A 920 (2019) 106–114,
   DOI 10.1016/j.nima.2018.12.066, arXiv 1810.02990, https://arxiv.org/abs/1810.02990. A parallel evolutionary
   algorithm with a fast 2D mixed-FE eigensolver (symmetric Jacobi–Davidson) and a penalty on the frequency
   constraint, applied to an SLS-2 cavity.
6. **S6** M. Kranjčević, S. Gorgi Zadeh, A. Adelmann, P. Arbenz, U. van Rienen, PRAB 22, 122001 (2019),
   DOI 10.1103/PhysRevAccelBeams.22.122001, arXiv 1905.13693. Constrained multi-objective optimisation with
   robustness against geometric perturbations (manufacturing, cryogenic) and a global sensitivity analysis to
   shrink the search space; the case is FCC-ee Z.
7. **S7** P. Putek, S. Gorgi Zadeh, U. van Rienen, J. Comput. Phys. 513 (2024) 113125,
   https://www.sciencedirect.com/science/article/pii/S0021999124003747. Robust multi-objective optimisation of
   TESLA-like cavities with polynomial chaos expansion, stochastic collocation and shape derivatives.
8. **S8** A. Ziegler, M. Hahn, J. Isensee, A. Nguyen, S. Schöps, IET Sci. Meas. Technol. 18(9) 462–470 (2024),
   DOI 10.1049/smt2.12190, arXiv 2310.15751. Gradient-based eigenvalue optimisation with closed-form shape
   derivatives and eigenvalue matching, applied to the 9-cell TESLA cavity.
9. **S9** Kappesser, Ziegler, Schöps, *Reduced Basis Approximation for Maxwell's Eigenvalue Problem and
   Parameter-Dependent Domains*, arXiv 2308.02259 (IEEE document 10414166). Follow-up with a tree–cotree gauge:
   arXiv 2406.11276. A reduced basis for parametric Maxwell eigenproblems that removes spurious modes; the test
   case morphs TESLA into a pillbox.
10. **S10 [Q]** J. Villarreal, D. Winklehner, D. Koser, J. M. Conrad, *Neural Networks as Effective Surrogate
    Models of RFQ Particle Accelerator Simulations*, Mach. Learn.: Sci. Technol. (2024), arXiv 2210.11451. Beam
    dynamics rather than EM: 200,000 samples, MAPE < 6 %.
11. **S11 [Q]** *Neural Network Model of the PXIE RFQ Cooling System and Resonant Frequency Response*,
    arXiv 1612.07237. Predicts the resonant frequency under thermal load; MAE 346 Hz on the test set.
12. **S12 [Q]** Chalmers thesis, *Data-Efficient AI-Driven RF Circuit Design via Multi-Fidelity Surrogates*,
    https://odr.chalmers.se/items/5026f31b-5064-4b6c-80b6-e33623222bd8. Multi-fidelity training cut the
    high-fidelity samples from 10k to 200 and the data generation time from 83.3 h to 5.2 h.

### B. ML analogues (eigenproblems, photonics, geometry-aware operators)
13. **S13 [Q]** T. Asano, S. Noda, Opt. Express 26, 32704 (2018), DOI 10.1364/OE.26.032704 (arXiv 1808.05722).
    A CNN predicts the Q of photonic-crystal nanocavities from 1000 samples with a 13 % standard-deviation error.
14. **S14 [Q]** T. Asano, S. Noda, Nanophotonics (2019), DOI 10.1515/nanoph-2019-0308 (arXiv 1908.03702). An
    iterative DNN loop over 8070 structures in 101 steps reached Q > 1.1e7.
15. **S15 [Q]** T. Christensen et al., Nanophotonics 9, 4183 (2020), DOI 10.1515/nanoph-2020-0197,
    https://dspace.mit.edu/handle/1721.1/132457. A CNN predicts the band structures of 20,000 2D photonic-crystal
    cells with about 0.5–0.6 % mean relative error.
16. **S16** C. Loh et al. (SIB-CL), Nat. Commun. (2022), DOI 10.1038/s41467-022-31915-y (arXiv 2110.08406).
    Surrogate data plus invariances cut the number of labels needed "by orders of magnitude".
17. **S17 [Q]** Z. Li et al., *Geometry-Informed Neural Operator* (GINO), NeurIPS 2023, arXiv 2309.00583.
    About 500–600 3D car geometries, 7–8 % relative L2 error on pressure, and drag 26,000× faster than OpenFOAM.
18. **S18** H. Wu et al., *Transolver*, ICML 2024, arXiv 2402.02366. Physics-attention over learned slices of mesh
    points, at linear cost.
19. **S19** S. Cao, *Choose a Transformer: Fourier or Galerkin*, NeurIPS 2021, arXiv 2105.14995. Softmax-free
    linear attention is comparable to a Petrov–Galerkin projection.
20. **S20** V. Fanaskov et al., *Deep Learning for Subspace Regression*, ICLR 2026, arXiv 2509.23249. Predicting a
    larger subspace than needed improves accuracy; a predicted eigenspace used as the LOBPCG start cuts iterations.
21. **S21** D. Pfau et al., *Spectral Inference Networks*, ICLR 2019, arXiv 1806.02215, and Z. Deng et al.,
    *NeuralEF*, ICML 2022, arXiv 2205.00165. Neural eigenfunctions learned by trace or Rayleigh-type objectives.
22. **S22** C. Rowan, J. Evans, K. Maute, A. Doostan, arXiv 2506.04375 (2025). Neural eigenfunctions with the
    Rayleigh quotient and Gram–Schmidt. It reports that strong-form-residual methods struggle to reach higher modes.
23. **S23** L. Zhao, M. Fogler, *Can a neural network hear the shape of a drum?*, arXiv 2203.08073 (PRR 5, 013110).
    The inverse spectral problem (eigenvalues to shape), on pentagons.
24. **S24** *Computation of waveguide eigenmodes by physics-informed neural networks*,
    https://iopscience.iop.org/article/10.1088/3049-4761/ae510e. A label-free PINN for Helmholtz waveguide
    eigenmodes.

### C. Classical design practice, solvers, conventions
25. **S25 [Q]** B. Aune et al., *Superconducting TESLA cavities*, PRSTAB 3, 092001 (2000),
    DOI 10.1103/PhysRevSTAB.3.092001, arXiv physics/0003011. Design parameters, trimming, weld-shrinkage
    allowances and Lorentz-force detuning.
26. **S26** H. Padamsee, *Design Topics for Superconducting RF Cavities and Ancillaries*, arXiv 1501.07129. The
    FoM definitions (G, R/Q, Epk/Eacc, Bpk/Eacc, k_cc).
27. **S27** V. Shemelin, *Optimal choice of cell geometry for a multicell superconducting cavity*, PRSTAB 12,
    114701 (2009), DOI 10.1103/PhysRevSTAB.12.114701. An optimisation algorithm at fixed aperture, Epk/Eacc and
    wall slope; re-entrant inner cells.
28. **S28** V. Shemelin, S. Gorgi Zadeh, J. Heller, U. van Rienen, PRAB 19, 102002 (2016),
    DOI 10.1103/PhysRevAccelBeams.19.102002. A systematic scan of elliptic shapes for β = 0.4–1.
29. **S29** V. Shemelin, G. Hoffstaetter, *Analytical relationships between elliptic accelerating cavity shape and
    fields*, https://www.researchgate.net/publication/276105835. A semi-analytic polynomial field expansion.
30. **S30 [Q]** V. Shemelin, V. Yakovlev, *Equidistant optimization of elliptical SRF standing wave cavities*,
    PRAB 26, 092001 (2023), DOI 10.1103/PhysRevAccelBeams.26.092001, https://www.osti.gov/servlets/purl/2007218.
    Minimises the larger of the two normalised peak ratios; minimal Epk/Eacc ≈ 1.54 at a 35 mm aperture
    (as excerpted).
31. **S31 [Q]** N. Juntong, R. M. Jones, *High-gradient SRF cavity with minimized surface e.m. fields and superior
    bandwidth for the ILC*, arXiv 0910.1305, and arXiv 0906.3592. Superfish and HFSS differ by 1.7 / 8.2 / 1.3 %;
    frequency tolerance 100 kHz; the NLSF shape has 8.5 % lower Bs/Ea than TESLA.
32. **S32 [Q]** *ILC Technical Design Report, Vol. 3.I*, arXiv 1306.6353. FoM table for the TESLA, LL/ICHIRO, RE
    and LSF shapes.
33. **S33 [Q]** Omega3P (SLAC ACE3P). Sources: https://confluence.slac.stanford.edu/spaces/AdvComp/pages/59146985/Omega3P;
    ICAP 2006 talk https://proceedings.jacow.org/icap06/TALKS/WEA4IS01_TALK.PDF; Project-X Main Injector cavity,
    SLAC-PUB-16642, https://www.slac.stanford.edu/pubs/slacpubs/16500/slac-pub-16642.pdf; SLAC-PUB-11914,
    https://www.slac.stanford.edu/pubs/slacpubs/11750/slac-pub-11914.pdf. Curved quadratic tetrahedra, p up to 6,
    adaptive refinement, about 400k elements to converge.
34. **S34 [Q]** *High-β cavity design* (SLANS), SRF2005 SuA02, https://proceedings.jacow.org/SRF2005/papers/SuA02.pdf.
    SLANS uses biquadratic quadrilateral elements and agrees with published values within +2…−1 %.
35. **S35** cavsim2d (an open axisymmetric code; author handle "Dark-Elektron", Rostock),
    https://github.com/Dark-Elektron/cavsim2d and https://pypi.org/project/cavsim2d/. Epk/Eacc does not converge at
    a corner singularity; gives the TESLA mid-cell parameters.
36. **S36** A. Grudiev (CERN), *Simulation packages and review of codes*,
    https://indico.cern.ch/event/164082/contributions/246170/attachments/195880/274792/Simulation_packages_and_Review_of_Codes_AG.pdf.
    Cartesian (FD/FIT) meshes are less accurate for surface fields than tetrahedral ones.
37. **S37 [Q]** M. Dauge, *Benchmark computations for Maxwell equations*,
    https://perso.univ-rennes1.fr/monique.dauge/benchmax.html. Reference values are reproduced in arXiv 2512.15314;
    an NGSolve check is at https://forum.ngsolve.org/t/maxwell-eigenvalues-fichera-corner/3861. Fichera corner
    λ1 = 3.21987401386, against 3.21967 from NGSolve.
38. **S38** Shunt-impedance conventions: *Low-beta structures*, arXiv 1201.2593, and a USPAS standing-wave linac
    lecture, https://uspas.fnal.gov/materials/09UNM/Unit_4_Lecture_10_Standing_Wave_Linacs.pdf. The linac
    convention is V²/P; the circuit convention is V²/(2P).
39. **S39** Eacc length conventions: *High Gradient Tests of the Fermilab SSR1 Cavity*, arXiv 1301.5657;
    V. Yakovlev, USPAS 2024 lecture 13,
    https://uspas.fnal.gov/materials/24RohnertPark/PILA/Lecture%2013%20-%20Yakovlev.pdf; J. Delayen, USPAS 2015,
    https://uspas.fnal.gov/materials/15Rutgers/3_Cavity_Fundamentals.pdf. The definitions in use are N·βλ/2, βλ
    and 2/3·βλ (ANL).
40. **S40** Transverse-impedance definitions: V. Yakovlev, FERMILAB-FN-1234-TD,
    https://lss.fnal.gov/archive/test-fn/1000/fermilab-fn-1234-td.pdf; EIC crab-cavity HOM budget,
    https://inspirehep.net/files/27f618852246bb31bff75f9f4d96d2aa; IHEP02 HOMs, arXiv 1502.01771.
41. **S41** R. Wanzenberg, *HOMs in the TESLA 9-cell cavity*,
    https://indico.cern.ch/event/57247/contributions/1212325/attachments/988671/1405921/TESLAcavityHOM.pdf.
    Defines R = V²/2P.
42. **S42** On the validity of Slater's theorem: Particle Accelerators 46 (1994) 217–224,
    https://cds.cern.ch/record/251929/files/p217.pdf.
43. **S43** *Efficient Design Scheme of Superconducting Cavity*, arXiv physics/0008208. The equator radius is the
    frequency-tuning knob, with negligible influence on the other parameters.

### D. Manufacturing, tuning, shape errors, Lorentz-force detuning
44. **S44** W. Singer et al., *Production of superconducting 1.3-GHz cavities for the European XFEL*, PRAB 19,
    092001 (2016), DOI 10.1103/PhysRevAccelBeams.19.092001. More than 800 cavities; field flatness > 90 %.
45. **S45 [Q]** European XFEL RF QA, SRF2015 WEBA02. Paper: https://proceedings.jacow.org/SRF2015/papers/weba02.pdf;
    talk: https://accelconf.web.cern.ch/SRF2015/talks/weba02_talk.pdf. π-mode at 2 K 1299.7 ± 0.1 MHz, field
    flatness > 90 %, cell shape ±0.2 mm.
46. **S46** W. Singer, *SRF Cavity Fabrication and Materials*, arXiv 1501.07142. Deep drawing, springback,
    criticality of the equator welds.
47. **S47 [Q]** Deep-drawn 1.3 GHz half-cells, SRF2009 THPPO071, https://epaper.kek.jp/SRF2009/papers/thppo071.pdf.
    0.4 mm is the realistic tolerance for fine-grain niobium.
48. **S48 [Q]** *Shape evolution of C75 large-grain niobium half-cells*, SRF2021 MOPCAV002,
    https://proceedings.jacow.org/srf2021/papers/mopcav002.pdf. About 63 % of scanned points within ±0.1 mm.
49. **S49 [Q]** European XFEL eccentricity, SRF2015 THPB023 / THPB033,
    https://accelconf.web.cern.ch/SRF2015/papers/thpb023.pdf. Average maximum eccentricity 0.2 mm, "twice
    straighter" than the specification.
50. **S50 [Q]** RRCAT. A 1.3 GHz copper single cell (InPAC 2011):
    https://www.iuac.res.in/iuacoldwebsite/event/InPAC11/proceedings/InPAC2011%20Proceedings/Oral%20%20and%20posters%20contributory/12_Bagre.pdf
    (equator weld shrinkage 0.90 mm). The DAE-BRNS17 fabrication talk:
    https://www.rrcat.gov.in/symposiums/conf/DAE-BRNS17/talks/TT-11_SCRF%20cavity%20fabrication_Avinash_20-07_final.pdf
    (K_tune +2.8 MHz/mm, K_trim −2.7 MHz/mm).
51. **S51 [Q]** *SRF cavity design, RF measurements and tuning*, USPAS 2017, Fermilab,
    https://uspas.fnal.gov/materials/17NIU/SRF%20Cavity.pdf. Sensitivities of 14 MHz/mm at the iris and
    −55 MHz/mm at the equator; the excerpt does not say which cavity.
52. **S52 [Q]** Fermilab SSR1 (325 MHz). Production, FERMILAB-CONF-07-241:
    https://lss.fnal.gov/archive/2007/conf/fermilab-conf-07-241-td.pdf. Tuner, FERMILAB-CONF-09-424:
    https://lss.fnal.gov/archive/2009/conf/fermilab-conf-09-424-td.pdf. RRCAT SSR dressing:
    https://www.rrcat.gov.in/symposiums/conf/DAE-BRNS17/talks/TT-12_dressing%20SSR_V%20K%20Mishra_20_07_17_final.pdf.
    PIP-II SSR2: arXiv 2307.08852. Gives the weld shortening, trimming sensitivity, tuner range and assembly budget.
53. **S53 [Q]** *Tuning of the TESLA superconducting cavities*, https://inspirehep.net/files/5839f1ac457f6d405cfc3406d3e7f458.
    Etching shifts the frequency by −10 kHz/µm; flatness reaches 95 % after correcting 3–4 cells.
54. **S54 [Q]** D. Zu, J. Chen, *Study on TESLA Cavity Shape*, PAC93, https://epaper.kek.jp/p93/PDF/PAC1993_1095.PDF.
    Tuning sensitivity 2880 Hz/µm.
55. **S55 [Q]** Fermilab field-flatness definition, FERMILAB-PUB-22-008-TD,
    https://lss.fnal.gov/archive/2022/pub/fermilab-pub-22-008-td.pdf. Flatness = min/max of the π-mode cell peaks;
    > 98 % bare, > 90 % dressed.
56. **S56 [Q]** P. Kneisel, *State of the art of multicell SC cavities and perspectives*, EPAC02 TUXGB001,
    https://proceedings.jacow.org/e02/PAPERS/TUXGB001.pdf. Unflatness factors N²/(βk).
57. **S57** *Project X Accelerator Reference Design*, arXiv 1306.5022. Flatness sensitivity δE/E ~ (n−1)²/k.
58. **S58 [Q]** L. Xiao et al., *Modeling imperfection effects on dipole modes in TESLA cavity*, SLAC-PUB-12634,
    https://www.slac.stanford.edu/pubs/slacpubs/12500/slac-pub-12634.pdf. A mesh-distortion imperfection model
    with Omega3P; cell-length sensitivity 5.4 MHz/mm.
59. **S59** V. Akçelik et al., *Shape determination for deformed electromagnetic cavities*, J. Comput. Phys. 227
    (2008), https://www.researchgate.net/publication/222398307. An inverse problem that recovers the deformed shape
    from measured frequencies.
60. **S60 [Q]** J. Corno et al., *Uncertainty modeling and analysis of the European XFEL cavities manufacturing
    process*, NIM A (2020), DOI 10.1016/j.nima.2020.164135, arXiv 1906.09151. UQ and global sensitivity analysis
    of the manufacturing chain; targets of flatness > 90 % and π-mode error < 100 kHz.
61. **S61** J. Dölz, D. Ebert, S. Schöps, A. Ziegler, *Shape uncertainty quantification of Maxwell eigenvalues and
    -modes with application to TESLA cavities*, CMAME 428 (2024) 117108, arXiv 2401.11890. Code:
    https://zenodo.org/records/11032300. A Karhunen–Loève model with M = 7 terms from the DESY database.
62. **S62** N. Georg, W. Ackermann, J. Corno, S. Schöps, CMAME 350 (2019), arXiv 1802.02978. Isogeometric
    analysis with mode tracking and a KL model from measured cavities.
63. **S63 [Q]** Lorentz-force detuning (LFD) and pressure sensitivity: arXiv 1511.00456 (IHEP 325 MHz HWR),
    arXiv 1604.02799 (β = 0.09 HWR), arXiv 1307.3354 (IMP HWR), arXiv 1309.6035 (squeezed HWR), PAC03 TPAB043
    https://proceedings.jacow.org/p03/PAPERS/TPAB043.PDF (spoke), SRF01 SA010
    https://proceedings.jacow.org/srf01/papers/SA010.pdf (SNS), and a 325 MHz spoke tuner test
    https://s3.cern.ch/inspire-prod-files-2/26814b21b9a5da63c4937a4a594cd4c3.
64. **S64 [Q]** Frequency budgets: the PIP-II tuner, arXiv 2208.06315 (2 K landing frequency); a JLab SRFI trip
    note, BNL C-A/AP/207, https://www.agsrhichome.bnl.gov/AP/ap_notes/ap_note_207.pdf; FRIB low-β process,
    https://inspirehep.net/files/1d9f1f8131731e3ae7791492be63b894 (0.854 kHz/µm, within 6 kHz); HIAT2012 PO16,
    https://proceedings.jacow.org/HIAT2012/papers/po16.pdf (±120 kHz); the GW-detector SRF cavity, arXiv 2601.18719
    (40 kHz/µm measured against 20 kHz/µm simulated).
65. **S65 [Q]** HOM spreads: the multi-pass BBU study arXiv 1305.6119 (dipole HOM spread of order 10 MHz from
    fabrication); degenerate HOMs in HB650, https://pith.science/paper/2607.15206 (a passband splits under
    realistic tolerances).

Thirty of the 65 entries carry quantitative results ([Q]).

## 2. State of the art in ML for RF cavities

### 2.1 Works (RF cavities first, then the closest analogues)

| # | year | cavity / domain | geometry parametrisation | dataset | model | targets | reported accuracy | source |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026 | SRF 3-cell multicell + cylindrical cavities (qubit hosts) | spline control points, "sampled from a uniform distribution around the values of the initial curve" | 2000 multicell + 2000 cylindrical COMSOL runs, ~2 min each | DNN, **inverse** (targets → shape) | in: passband width, axial E_z profile; out: control points | re-simulated designs within ~5 % (cavities), ~2 % (transmons); passband error ≤ ~5 % | S1 (secondary summary) |
| 2 | 2026 | 499.65 MHz normal-conducting cavity (USTC) | design variables of a published cavity (not retrieved) | ≤ 1000 simulations | multi-objective BO, GP; two-stage acquisition (vector AF + MOEA, then scalar AF) | several objectives, "extremely strict equality constraints" | "high-quality Pareto-optimal solutions within 1000 simulations"; better than penalty AF | S2 |
| 3 | 2025 | ring-cyclotron cavity (PSI-type), Osaka | NURBS control points + weights | HFSS runs, periodic retraining (count not retrieved) | NN ensemble surrogate + MOGA, ensemble UQ, Monte Carlo over manufacturing deviations | efficiency vs peak-field limits | not retrieved | S3 |
| 4 | 2026 | 3D RF-dipole (crab) cavity, plus aerofoils and wings | PCA of deformations between example shapes (linear, low-dimensional) | "modest numbers of examples" | GP Bayesian optimisation | single- and multi-objective | "significant improvements over a carefully hand-crafted baseline" (authors' claim) | S4 |
| 5 | 2019 | SLS-2 main cavity (axisymmetric, NC) | cross-section parameters | EA population (not retrieved) | EA + 2D mixed FE (no ML) | multi-objective, frequency penalty | — | S5 |
| 6 | 2019 | FCC-ee Z SRF cavity | elliptic-arc cell parameters, reduced by global sensitivity analysis | not retrieved | EA + axisymmetric solver + frequency tuning, robust to perturbations | fundamental-mode and HOM FoMs | — | S6 |
| 7 | 2023–24 | TESLA morphed to pillbox | parameter-dependent domain | greedy reduced-basis snapshots | reduced basis for Maxwell eigenproblems; spurious modes removed (tree–cotree) | eigenpairs, mode tracking | matches analytic solutions (no number in excerpt) | S9 |
| 8 | 2024 | RFQ (IsoDAR), beam dynamics (analogue) | RFQ design parameters | 200,000 simulations | MLP surrogate | 6 beam outputs | MAPE < 6 % | S10 |
| 9 | 2016 | PXIE RFQ, frequency under cooling (analogue) | thermal and operating inputs | measured data | NN | resonant frequency | MAE 346 Hz on test | S11 |
| 10 | 2018 | 2D photonic-crystal nanocavity (analogue) | random air-hole displacements | 1000 cavities | 4-layer CNN | Q factor | 13 % (standard deviation) | S13 |
| 11 | 2019 | same, iterative | same | 8070 structures over 101 iterations | DNN + active loop | Q | best Q > 1.1e7 | S14 |
| 12 | 2020 | 2D photonic crystals (analogue) | 32×32 permittivity image (level set of Fourier sums) | 20,000 cells | CNN | first 6 TM / TE bands on a 23×23 k-grid | mean relative error ~0.5–0.6 % | S15 |
| 13 | 2022 | 2D photonic crystals (analogue) | same | 20,000, data-scarce regimes | contrastive pretraining + surrogate data + invariances | DOS, bands | "orders of magnitude" fewer labels for the same accuracy | S16 |
| 14 | 2023 | 3D car aerodynamics (analogue) | SDF + point cloud | ~500–600 geometries | GINO (graph + Fourier neural operator) | surface pressure, drag | 7.29 % (ShapeNet car) / 8.31 % (Ahmed) relative L2 on pressure; drag 26,000× faster than OpenFOAM | S17 |
| 15 | 2025 | RF circuits (analogue) | circuit geometry | high-fidelity samples 10k → 200 | multi-fidelity surrogate | S-parameters | target accuracy kept; data generation 83.3 h → 5.2 h | S12 |

No search found an RF-cavity work that uses a mesh, point cloud, image or graph as input, or that predicts 3D
fields. Null results came from three searches: "graph neural network … cavity eigenfrequency", "CNN … RF cavity
… resonant frequency" and "neural operator resonant cavity eigenmodes 3D". The closest learned-solver works for
Maxwell eigenproblems are classical model reduction (S9) and perturbation-based UQ (S61, S62), not neural
networks. Learned eigen-subspaces exist in ML (S20–S22), but none of them work on Maxwell problems with a kernel
projection.

### 2.2 Answer to (a): how other ML works parametrise and sample cavities, and dataset size against accuracy

- **Parametrisation.** Every RF-cavity ML or optimisation work found feeds a **low-dimensional CAD parameter
  vector** to the model:
  - seven elliptic-arc parameters (S5, S6, S27, S28, S35);
  - spline control points (S1), "a less common alternative" chosen because "splines offer more shape variety,
    which matters for DNN training, at the cost of direct control over parameters like iris radius or cell
    width";
  - NURBS control points and weights (S3);
  - PCA modes of deformations between example designs (S4);
  - for uncertainty quantification, Karhunen–Loève modes fitted to measured cavities (S61, S62).
- **Sampling.**
  - Uniform around a nominal design (S1).
  - Evolutionary populations (S5, S6).
  - GP acquisition (S2, S4).
  - Ensemble-driven retraining, a form of active learning (S3).
  - Iterative active learning in photonics (S14: 8070 structures in 101 rounds).
  - Dimension reduction by global sensitivity analysis before optimising (S6, S60).
- **Dataset size and accuracy.**
  - RF cavities: 1000 simulations for a GP Pareto front (S2), 4000 COMSOL runs for ~5 % inverse-design accuracy
    (S1).
  - Photonic analogues: 1000 samples gave 13 % on Q (S13); 20,000 gave ~0.5 % on 2D band frequencies (S15).
  - Accelerator beam dynamics: 200,000 samples gave MAPE < 6 % (S10).
  - 3D neural operators: ~500–600 geometries gave 7–8 % field L2 error (S17).
  - No error-against-N learning curve for RF cavities was found. In photonics, surrogate pretraining (S16) and
    multi-fidelity training (S12) cut the high-fidelity labels needed by 10–100×.

## 3. Classical design practice relevant to our labels

### 3.1 Parametrisations and optimisation habits

- **The elliptic half-cell** (equator ellipse A, B; iris ellipse a, b; Riris, Req, half-cell length L) is the
  working parametrisation of SRF optimisation (S27, S28, S35).
  - The cavsim2d example gives the TESLA mid-cell as [A, B, a, b, Ri, L, Req] = [42, 42, 12, 19, 35, 57.7,
    103.353] mm (S35). Our anchor uses Req 103.3 mm, a match to 0.05 %.
  - The TESLA end cups in the Aune table (S25, excerpt): iris 39, arc radius 40.3, a 10, b 13.5, length 56.0 mm.
- **Standard optimisation problem.** Minimise Bpk/Eacc at fixed Epk/Eacc, aperture and **wall slope angle**,
  scanned over β = 0.4–1 (S27, S28).
  - Shemelin reports that lower wall slope angles give **re-entrant inner cells** and that these also help end
    cells (S27).
  - The "equidistant" method minimises the larger of the two normalised peak ratios; minimal Epk/Eacc ≈ 1.54 at a
    35 mm aperture (S30, excerpt).
  - Designers usually fix the frequency by tuning one parameter. The equator radius is used because "its effect on
    the resonance frequencies is large and its influence on the other cavity parameters is negligible" (S43).
    Frequency is linear in the equator parameter B, and designs were tuned to within 100 kHz (S31).
- **Analytic and semi-analytic tools.** Shemelin & Hoffstaetter relate shape and fields through a truncated
  polynomial field expansion (S29); the excerpt gives no accuracy figure. The ML and optimisation papers above do
  not use such formulas as features.
- **Splines, NURBS, Bezier.** Used for diversity (S1, S3). The search found no head-to-head Bpk comparison between
  Bezier and elliptic profiles.
- **Real-design reference values** (ILC TDR table, S32, excerpt):

| shape | Epk/Eacc | Bpk/Eacc [mT/(MV/m)] | G·R/Q [Ω²] |
|---|---|---|---|
| TESLA | 1.98 (Aune: 2.0) | 4.15 (Aune: 4.26) | 30,840 |
| LL / ICHIRO | 2.36 | 3.61 | 37,970 |
| RE (re-entrant) | 2.28 | 3.54 | 41,208 |
| LSF | 1.98 | 3.71 | 36,995 (k_cc 1.27 %) |

  The TESLA values from S25 (excerpt) are G = 270 Ω, Epk/Eacc 2.0, Bpk/Eacc 4.26, k_cc 1.87 %. NLSF has 8.5 % lower
  Bs/Ea than TESLA (S31).

### 3.2 Solvers and convergence checks

- **SLANS** uses biquadratic quadrilateral elements for axisymmetric monopole modes (SLANS2 also does
  multipoles). It agrees with published values within "+2…−1 %" (S34, S35).
- **SUPERFISH** is used as the reference for axisymmetric structures in an HIE-ISOLDE comparison with HFSS and
  CST (excerpt, arXiv 0910.0409).
- **CST** uses the Finite Integration Technique with Perfect Boundary Approximation meshing (S35). A CERN review
  says Cartesian meshes are less accurate for frequency, Q and surface fields than tetrahedral ones (S36).
- **Omega3P** (S33) uses curved second-order tetrahedra with hierarchical Nédélec bases up to p = 6.
  - Adaptive refinement "until a converged result is obtained" (SLAC-PUB-11914).
  - About 400k curved elements with second-order bases were needed to converge the Project-X Main Injector
    cavity.
  - Strong scaling on 1.5M high-order tetrahedra (LCLS gun) on 4096 CPUs.
  - The stated accuracy target is "10s kHz mode separation out of GHz", about 1e-5.
- **Code-to-code scatter.**
  - Superfish and HFSS differ by **1.7 % (k_c), 8.2 % (Es/Ea) and 1.3 % (Bs/Ea)** for the same ILC shape (S31).
  - The TESLA peak ratios differ by 1–2.6 % between S25 and S32 (2.0 / 1.98; 4.26 / 4.15).
- **Singularities.**
  - cavsim2d notes that Epk/Eacc "does not converge under mesh refinement" at a corner singularity (S35).
  - On the Dauge benchmark, the Fichera-corner reference λ1 = 3.21987401386 and an NGSolve forum value of 3.21967
    differ by 6e-5 *(estimate)* (S37). Even good codes converge slowly at re-entrant edges.
- **Our label solver (docs/31 §1).** The analytic pillbox check at 157k DOF gives f 6e-7, Q0 3e-5, R/Q 1e-5,
  Epk 2e-4 and Bpk 4e-4. That is better than the published scatter, and it follows the Omega3P recipe: curved
  high-order elements with adaptive refinement.

### 3.3 Accuracy designers actually need

| quantity / practice | value | relative | source |
|---|---|---|---|
| European XFEL π-mode frequency at 2 K | 1299.7 ± 0.1 MHz | ±7.7e-5 | S45 |
| European XFEL π-mode error target | < 100 kHz | 7.7e-5 | S60 |
| design tuning (NLSF) | within 100 kHz at 1.3 GHz | 7.7e-5 | S31 |
| Omega3P target for HOM separation | "10s kHz out of GHz" | ~1e-5 | S33 |
| FRIB QWR before final tuning | within ~6 kHz of target at 80.5 MHz | 7.5e-5 | S64 |
| SSR1 tuner range | ~200 kHz at 325 MHz (TRIUMF spec ≥ 180 kHz) | 6e-4 | S52 |
| SSR1 helium-vessel assembly budget | ≤ 300 kHz | 9e-4 | S52 |
| TESLA equator trimming allowance | 1 mm extra length × 2.7 MHz/mm ≈ 2.7 MHz | ~2e-3 *(estimate)* | S25, S50 |
| field flatness | > 90 % dressed, > 98 % bare; ≥ 95 % after tuning | — | S45, S55, S53 |
| unflatness factor N²/(βk) | TESLA ~4050; SNS β 0.61: 3934; β 0.81: 2964 | a 1e-5 cell-frequency error gives a few % field tilt *(estimate)* | S56 |
| published scatter of TESLA peak ratios | 2.0 vs 1.98; 4.26 vs 4.15 | 1 %, 2.6 % | S25, S32 |
| Superfish vs HFSS | k_c 1.7 %, Es/Ea 8.2 %, Bs/Ea 1.3 % | — | S31 |
| differences between competing designs | Bpk/Eacc −11…−15 %, G·R/Q +20…+34 % vs TESLA | — | S32 |

### 3.4 Figure-of-merit conventions in use

| FoM | conventions in the literature | our label today | note |
|---|---|---|---|
| R/Q, R_sh | linac V²/(ωU), V²/P vs circuit V²/(2ωU), V²/(2P), a factor 2 (S38). DESY HOM work uses R = V²/2P (S41). The Aune excerpt lists **518 Ω** for the 9-cell TESLA (S25). The TDR's G·R/Q = 30,840 Ω² with G = 270 Ω implies **≈114 Ω per cell** *(estimate)*, matching a "113.8 Ω per inner cell" excerpt; 9 × 114 ≈ 1030 Ω, so 518 Ω is the circuit value *(inference)*. | linac, β = 1, V over the whole axis including pipes | store the convention (done) and per-cell R/Q |
| Eacc length | elliptical: N·β_g·λ/2 (S39, PIP-II). Spoke / HWR: βλ for 2 gaps, generally n·βλ/2 for n gaps (Fermilab SSR1); the old ANL definition is 2/3·βλ (S39). Delayen: "no universally adopted definition" for β < 1 (S39). | **axis chord inside Ω, beam pipes included** (`highorder.mode_qoi`, docs/24 §1.3) | not comparable, §5.3 |
| Epk/Eacc, Bpk/Eacc | inherit the Eacc length; Bpk/Eacc in mT/(MV/m) | chord-based | rescale |
| G | Q0·R_s [Ω], universal (S26) | yes | — |
| Q0 | copper or SRF R_s; SRF papers quote G | copper σ 5.8e7 | G is the transferable number |
| T | V / ∫‖E_z‖ at β; β_opt or β_g for low-β | β = 1 | add β |
| k_cc | 2(f_π − f_0)/(f_π + f_0) (S60) | not stored | add |
| field flatness | min/max of the π-mode cell peaks (S55) | not stored | add |
| G·R/Q | per-cell product in Ω² (S32) | derivable | add |
| ZT² per length | linac effective shunt impedance (S38) for DTL-type structures | not stored | add for dtl |
| transverse R/Q | Ω/m, Ω/m² or at an offset; differs by lab (S40); budgets such as (R/Q)⊥·Q·k < 5e5 Ω/m (EIC crab) | not stored | add, with the convention |

## 4. Manufacturing and shape-error magnitudes and how they are modelled

### 4.1 Magnitudes
Percentages are relative to a TESLA-size cell (Req 103.3 mm, full-cell length 115.4 mm) *(estimate)*.

| error source | magnitude | % of size | frequency effect | source |
|---|---|---|---|---|
| cell shape (inner surface) specification, European XFEL | ±0.2 mm | ±0.19 % of Req | — | S45 (talk) |
| same, percentile form | ±0.2 mm for 90 % of measured data, ±0.25 mm for the rest | 0.19–0.24 % | — | excerpt citing an SRF fabrication source; a phrase search did not pin it (S46–S48) |
| deep-drawn fine-grain half-cells | 0.4 mm "realistic achievable tolerance" | 0.39 % | — | S47 |
| large/medium-grain half-cells (C75) | ~63 % of points within ±0.1 mm | 0.1 % | — | S48 |
| forming method | 200 µm (electrohydraulic) vs 600 µm (traditional) | 0.19 % / 0.58 % | — | excerpt, ResearchGate entry on electrohydraulic forming |
| eccentricity, European XFEL production | average max 0.2 mm, "twice straighter than required" | specification ≈ 0.4 mm *(inference)* | splits dipole HOMs | S49 |
| equator weld shrinkage (Cu 1.3 GHz single cell) | 0.90 mm | 0.8 % of cell length | corrected by machining | S50 |
| SSR1 end-wall welds | ~1.4 mm axial shortening | — | trim 702.5 kHz/mm/side (theory 735); SSR1-02: 376 kHz/mm | S52 |
| TESLA trimming allowance | 1 mm extra at the equator, set after the dumbbell frequency measurement | 0.87 % of cell length | — | S25 |
| chemistry (BCP) | ~200 µm (in the SSR1 frequency budget) | 0.19 % of Req | TESLA −10 kHz/µm; measured 40 kHz/µm vs 20 kHz/µm simulated (GW cavity); FRIB QWR 0.854 kHz/µm | S52, S53, S64 |
| cell length / tuning | 1.3 GHz 9-cell: K_tune +2.8 MHz/mm, K_trim −2.7 MHz/mm; TESLA measured 5.4 MHz/mm; PAC93: 2.88 MHz/mm | — | — | S50, S58, S54 |
| iris / equator radius | 14 MHz/mm (iris), −55 MHz/mm (equator), cavity not identified | — | — | S51 |
| HOMs from fabrication | dipole HOM spread ~10 MHz; TDR cavities: shifts and splittings of hundreds of kHz to a few MHz, against tens of kHz ideal | — | — | S65, S33 (ICAP06) |
| LFD | TESLA ~1 Hz/(MV/m)², ~600 Hz at 25 MV/m, dynamic ±120…±300 Hz, halved by stiffening. HWR −2.1 (ports fixed) to −12.7 (free); squeezed HWR −16.9 / −10.2 stiffened; spoke −2 (double), −3.5 (single); SNS specification < 3 Hz/(MV/m)² | 600 Hz / 1.3 GHz = 4.6e-7 *(estimate)* | coupled structural problem | S25, S63 |
| pressure | 325 MHz spoke: −280…−360 Hz/torr unloaded, −137…−210 with the tuner | — | — | S63 |
| frequency budget | PIP-II sets a 2 K "landing frequency" above nominal, because the tuner only compresses; JLab: budget for final BCP / HPR; SSR1 room-temperature target 324.535 MHz includes vacuum, cool-down and 200 µm BCP | — | — | S64, S52 |

### 4.2 How shape errors are modelled in the literature

- **Statistical models fitted to real cavities.**
  - A truncated Karhunen–Loève expansion with M = 7 terms "derived … from real-world data from the DESY
    database", describing misaligned cells from dumbbell production and equator welding (S61, code on Zenodo).
  - KL modes from measured resonators combined with mode tracking (S62).
  - Normally distributed tolerances along the whole European XFEL manufacturing chain (10 sub-components welded
    at the equators), compared with measured statistics (S60).
- **Robust optimisation.** Perturbations from manufacturing and cryogenic conditions inside a constrained
  multi-objective optimisation (S6); polynomial chaos expansion and stochastic collocation with shape derivatives
  (S7).
- **Perturbation methods.** For small perturbations, perturbation approximations of mean and covariance are
  "faster and more accurate than sampling or surrogate model strategies" (S61). Closed-form shape derivatives are
  in S8.
- **Physical imperfection models.** Mesh distortion plus Omega3P, compared with TTF Module 5 dipole data (S58).
  Inverse shape determination from measured frequencies by Gauss–Newton with truncated SVD (S59).
- **Slater's theorem** is the first-order tool: δf/f = ∫ΔV (μ0 H² − ε0 E²) dV / (4U). A CERN note warns that it
  assumes an n × E = 0 wall and is wrong for displacements of an n × H = 0 boundary (S42). Our walls are all PEC,
  so it applies to our labels.

### 4.3 Calibration of our deformation option (measured)

- **Our map.** x → x + Σ a_m sin(ω_m·x + φ_m) with 3–8 plane waves, **wavelengths 0.25–1.5 × the bounding-box
  diagonal**, and Lipschitz constant lip ∈ [0.3, 1]·`deform_max` (default 0.5)
  (`dataset_generator_3d.deform_map`). It is a global, long-wavelength bending.
- **Measurement.** I sampled 400 draws of the same map on a TESLA-size single cell with pipes (Req 103.3 mm,
  L_cell 115.4 mm, pipe radius 35 mm, 70 mm pipes; bounding diagonal 388 mm):

| `deform_max` | median max displacement | 5–95 % | % of diagonal | % of Req |
|---|---|---|---|---|
| 0.5 (default) | 8.4 mm | 3.9–15.5 mm | 2.2 % | 8.2 % |
| 0.05 | 0.84 mm | 0.39–1.55 mm | 0.22 % | 0.82 % |
| 0.01 | 0.17 mm | 0.08–0.31 mm | 0.04 % | 0.16 % |
| 0.002 | 0.03 mm | 0.02–0.06 mm | 0.01 % | 0.03 % |

- **Conclusion.** The default is a **design-diversity augmentation**, about 40× the European XFEL ±0.2 mm shape
  tolerance and 20× the 0.4 mm deep-drawing tolerance.
  - Fabrication-scale amplitudes need `deform_max` ≈ 0.01–0.02 (about 0.17–0.34 mm median maximum).
  - Even then the error **pattern** is wrong. Real errors are local to half-cells and welds: per-cell radius and
    length errors, rigid cell offsets (eccentricity 0.2–0.4 mm) and nearly uniform removal by chemistry
    (~0.1–0.2 mm). The single-cell measurement above does not show this, but for multi-cell cavities the KL modes
    of S61 describe cell misalignment.

## 5. Comparison with our generator and model

### 5.1 Aligned with practice
- **Label solver.** Adaptive high-order H(curl) on curved tetrahedra (p3) with an analytic check is the Omega3P
  recipe (S33). Its pillbox errors (f 6e-7, Epk 2e-4) are below the published code-to-code scatter (1–8 %, S31,
  S34). This beats the CST-style FIT that docs/31 benchmarks against for surface fields (S36).
- **Elliptical family.** It uses the community's seven-parameter elliptic half-cell. The TESLA anchor matches the
  literature values (S25, S35). Anchors around real designs, perturbed ±10 %, mirror the "uniform around the
  initial curve" sampling of S1. Scrambled Sobol sampling is standard design-of-experiments practice.
- **Rounded re-entrant edges.** Rounding them, and flagging `sharp_edges` for out-of-distribution families,
  matches the known non-convergence of peak fields at corners (S35, S37).

### 5.2 Ahead of the RF state of the art
- **Outputs.** No RF-cavity ML work found predicts 3D vector fields, several modes, or FoMs computed from those
  fields. All of them map a few parameters to scalars or to an axial profile (S1–S4).
- **Variational structure.** A learned basis with kernel projection and Rayleigh–Ritz gives upper-bound
  eigenvalues and a basis for mode tracking. In RF, comparable structure exists only in classical reduced-basis
  and perturbation methods, each for a single parametric family (S9, S61, S62).
- **ML design choices with external support.**
  - Predicting more basis vectors than modes (m = 24 for K = 4) matches S20, which found that predicting a larger
    subspace than needed improves accuracy.
  - Using the predicted subspace as the start of a LOBPCG solve (the hybrid mode of docs/31 §5) is also supported
    by S20.
  - Mass-weighted linear attention over quadrature tokens is a quadrature form of Galerkin-type linear attention
    (S19) and of Transolver's physics attention (S18) *(inference)*.
- **Breadth and scale.**
  - Eight training families plus five out-of-distribution families, against one family per paper in the
    literature.
  - Labels cost ~0.5 core-hour, so 10k geometries cost 5k core-hours (1 % of the 500k budget) and 100k cost 10 %.
    The largest RF-cavity ML datasets found have ≤ 4000 geometries (S1) *(estimate)*.

### 5.3 Behind, missing or not comparable (families, sampling, labels)

1. **The Eacc length makes our Epk/Eacc and Bpk/Eacc incomparable with published values.**
   - The labels use L_acc = the axis chord inside Ω, beam pipes included (`mode_qoi`). The literature uses
     N·β_g·λ/2 (elliptical), the gap or cell length (pillbox, nose-cone, DTL cells) or βλ (2-gap spoke / HWR)
     (S39).
   - Size of the effect *(estimate from our ranges)*: for a TESLA-like single cell with 35 mm pipes of
     1.2–2.5 × radius, the chord is 199–291 mm against L_cell = 115 mm, so our Epk/Eacc and Bpk/Eacc are
     **1.7–2.5× too high**. For 5 cells the factor is 1.15–1.3. For pillbox_pipes it can be 1.5–3×.
   - A TESLA cell would read Epk/Eacc ≈ 3.5–5 instead of 2.0.
   - R/Q, T, G and Q0 are unaffected, because V is integrated over the whole axis, which is standard.
   - The chord also makes these labels depend on the arbitrary pipe length. The model has to learn a dependence
     with no physical meaning *(inference)*.
2. **β = 1 for every family.** Published R/Q, T and Eacc of reduced-β cavities are quoted at β_opt or β_g, with
   their own L_eff (S39).
   - Our 25 % reduced-β elliptical cells (β 0.5–0.95) and the spoke, hwr and dtl families therefore carry FoMs
     that cannot be compared.
   - For dtl, β_cell = Lc/λ ≈ 0.2–0.35 from our ranges *(estimate)*, so a β = 1 transit factor has no design
     meaning.
3. **Design-space coverage of the elliptical family.**
   - The generator **rejects re-entrant walls**, but the RE shape holds the low-Bpk, high-G·R/Q corner of the
     published table (Bpk/Eacc 3.54 mT/(MV/m), G·R/Q 41,208 Ω², S32), and Shemelin finds re-entrant cells
     beneficial (S27).
   - Our own LL anchor (Req 98.6, A 50.1, B 34.2, a 7.6, b 10.0, L 57.7 mm) lies **outside the random ranges**:
     A/L = 0.87 > 0.85, B/A = 0.68 < 0.85, a/L = 0.13 < 0.15. Only Req 98.58 and Riris 30 were confirmed by the
     NLSF tables (S31); A, B, a and b remain unverified.
   - So the low-loss and low-surface-field region, which designers actually use, is sampled only by the 30 %
     anchor draws.
   - The TESLA end cup (S25: a 10, b 13.5 against 12, 19) is outside our ±15 % end-cell range for b (−29 %) and,
     marginally, for a (−17 %).
4. **Missing multi-cell labels.** Designers' multi-cell numbers are not stored: k_cc, field flatness, per-cell
   R/Q, G·R/Q and the passband. They are cheap to derive when K ≥ N_cells.
5. **Missing dipole and HOM conventions.** No transverse R/Q and no off-axis voltages, so the dipoles among the K
   modes carry only an R/Q that is noise (docs/24 already masks them with `rq_floor`).
6. **Fabrication errors are not modelled** (§4.3). The deformation option is 20–40× larger and long-wavelength.
7. **Families with literature interest that we lack.**
   - RF-dipole / crab cavities (S4, S40).
   - Photoinjector gun cavities (APEX2 design by genetic algorithm, arXiv 1905.10619, excerpt only).
   - Storage-ring normal-conducting cavities (S2's 499.65 MHz test case, with data on Zenodo).
   - Ring-cyclotron cavities (S3).
   - Couplers, external Q, multipacting and LFD are outside a closed-PEC eigenproblem. They need port boundary
     conditions, particle tracking or a structural solve.
8. **Unproven results.** The field model has not been trained at scale (docs/31 §1). There is no learning curve,
   no comparison with a simple parametric baseline, and no community benchmark yet.
9. **Inference speed.** Literature surrogates evaluate in microseconds and run inside 10³–10⁵-evaluation
   optimisers (S2, S5, S6). Our model needs meshing (~8 s), operators (~4 s) and a forward pass (~6–9 s) per
   geometry on CPU (docs/31 §3.6). That is fine as a solver accelerator, slow as a multi-objective inner loop.
10. **Fillet floor (5 % of size).** It probably excludes realistic small nose-cone and drift-tube nose radii,
    which control Epk. This survey found no source for typical radii *(inference, unverified)*.

### 5.4 Answers to (b) and (c)

**(b) Representation and families.**
- Our representation is **beyond** the published RF-ML state of the art:
  - it predicts fields on a curved FE mesh across families, through a variational Ritz layer;
  - published RF-ML works use parameter vectors with MLP, NN-ensemble or GP models, for one family at a time.
- It is **in line** with geometry-aware neural operators and learned-subspace methods in ML (S17–S20), and with
  high-order curved FE practice in RF (S33).
- The family set is broader than any single paper. It misses the **re-entrant and low-loss elliptical region**
  and the **crab and gun cavity** types.
- No shared ML benchmark exists for RF cavities. One search for open eigenmode datasets found none. Each paper
  makes its own data and validates by:
  - a held-out split;
  - re-simulating predicted designs (S1);
  - Pareto quality per number of simulations (S2).
- The de facto reference cases for solvers and designs are:
  - the analytic pillbox;
  - the **TESLA cell and 9-cell** with the Aune parameters (S25), the TDR shapes (S32), and measured
    TTF / European XFEL cavities (S33, S58, S60, S61);
  - **Dauge's Maxwell benchmarks** for singular domains (S37);
  - S2's released 499.65 MHz data.

**(c) Accuracy needed, against our model-space floor of 0.05–0.9 %.**

The floor in docs/30 §5 is 3.7e-4 – 8.6e-3, a relative error of the Rayleigh quotient, which is the eigenvalue
k². The frequency error is about half: 0.02–0.45 % *(inference; check the definition)*.

- **Final design and fabrication** need 1e-5 – 1e-4: 100 kHz at 1.3 GHz, 6 kHz at 80.5 MHz, "10s kHz out of GHz"
  (S31, S33, S45, S64). The floor is 5–90× too coarse. The model cannot replace the solver there. The hybrid
  warm-start mode (S20; docs/31 §5) is the route to that accuracy.
- **Correctable by standard practice.**
  - Low-β tuner ranges and assembly budgets are 6e-4 – 9e-4 (S52).
  - The TESLA trimming allowance is ≈ 2e-3 (1 mm × 2.7 MHz/mm) *(estimate)*.
  - So a model at the good end of the floor (≤ 0.2 %) gives a frequency that ordinary trimming can absorb. The bad
    end (0.9 %) cannot be absorbed.
  - In dimensions, 0.05–0.9 % on Req ≈ 100 mm is 0.05–0.9 mm, the same order as the ±0.2–0.4 mm fabrication
    tolerance.
- **Screening and Pareto exploration.** Frequency is a scale: a frequency error is removed by rescaling or by
  retuning the equator (S43), with negligible effect on the dimensionless FoMs. What matters is the **accuracy of
  G, R/Q, Epk/Eacc and Bpk/Eacc**.
  - ~1–2 % equals the scatter between published values and codes (1–2.6 %, up to 8 % for Epk, S25 / S31 / S32).
  - It is far below the 10–35 % differences between competing designs (S32).
  - **≤ 2 % on FoMs is a useful target, even with a 0.5 % frequency error.**
- **Multi-cell field flatness** needs cell frequencies to ~1e-5, because the unflatness factor is ~3000–4000
  (S56). This is out of reach for the model. Multi-cell field amplitudes predicted by the model will be tilted
  *(inference)*.
- **HOM bands.** Fabrication spreads HOMs by hundreds of kHz to ~10 MHz (S65, S33). A 0.1–0.5 % error on a
  ~2–3 GHz HOM is 2–15 MHz, the same order as the fabrication spread *(estimate)*.

## 6. Recommendations

Cost basis: 500k core-hours; labels ~0.5 core-hour per geometry; a CPU training step ~24 s per sample at 4
threads (docs/31 §3.2). So 10k samples × 150 epochs ≈ 40k core-hours per training run *(estimate)*.

### High priority

**H1. Store the standard Eacc normalisation, and make existing labels comparable.**
- *What.*
  - Keep V (already implied by R/Q and U = 1 J).
  - Store the standard lengths: elliptical N·2L, with β_g = 4L/λ for half-cell length L; the cell or gap length
    for pillbox, reentrant and dtl; βλ for 2-gap spoke and hwr.
  - Store Epk/Eacc and Bpk/Eacc for each convention.
  - For existing H5 files the rescaling Epk/Eacc_std = Epk/Eacc_chord × L_std / L_chord is exact, because Epk and
    V do not change.
- *Benefit.* Removes the 1.15–3× bias of §5.3-1 and a pipe-length nuisance dependence, so our numbers can be put
  next to S25, S32 and S39.
- *Cost.* **Cheap**: post-processing, no new solves; the chord is recomputable from the stored mesh.
- *Sources.* S39, S32, S25; docs/24 §1.3.

**H2. Store the on-axis complex E_z(z) of every mode and derive β-dependent quantities.**
- *What.* Store V(β), T(β), R/Q(β), β_opt and β_g.
- *Benefit.* Comparable R/Q and T for reduced-β elliptical, spoke, hwr and dtl (S39, S38), and any future L_acc
  or β choice without re-solving.
- *Cost.* **Cheap**. `axis_samples` is already evaluated in `mode_qoi`; 2001 complex points × K = 8 is ~0.25 MB
  per sample. For old data, u_model (p2) gives an approximation within its proj_err of 0.2–0.5 %.
- *Note.* This changes the generator's output, which needs the owner's go-ahead (AGENTS.md).

**H3. Add a fixed community benchmark set.**
- *What.*
  - The TESLA mid-cell and 9-cell (Aune: G 270 Ω, Epk/Eacc 2.0, Bpk/Eacc 4.26, k_cc 1.87 %).
  - LL / ICHIRO, RE and LSF from the ILC TDR table, which needs their dimensions (section 7) and re-entrant
    support for RE.
  - Dauge's Fichera corner (λ1 = 3.21987401386).
  - One or two low-β cavities (Fermilab SSR1, IHEP 325 MHz HWR), once their dimensions are collected.
- *Benefit.*
  - Validates the label pipeline against published numbers.
  - Exposes convention mismatches (H1, and the R/Q 518 against ~1036 Ω question).
  - Gives an out-of-distribution test the community recognises.
- *Cost.* **Cheap**: ~20–30 geometries, ≤ 15 core-hours.
- *Sources.* S25, S32, S37, S52.

**H4. Report a parametric baseline next to the field model.**
- *What.* An MLP or GP from each family's own parameter vector to f·scale, G, R/Q, Epk/Eacc and Bpk/Eacc.
- *Benefit.* This is the literature's standard surrogate (S2, S3, S5, S6, S10). Showing that the field model
  matches or beats it in-family, and generalises across families where it cannot, is the evidence a "beyond state
  of the art" claim needs. It also gives microsecond FoM estimates for optimisation loops (§5.3-9).
- *Cost.* **Cheap**: uses existing labels; minutes of CPU per family.

**H5. Widen the elliptical design space to the designs people use.**
- *What.*
  - Allow moderately re-entrant walls.
  - Extend B/A down to ~0.65, a/L down to ~0.12 and A/L up to ~0.9, which contains the LL anchor.
  - Widen the end-cell b range to about −30 %.
- *Benefit.* Covers the low-Bpk, high-G·R/Q part of published Pareto fronts (S27, S32).
- *Cost.* **Cheap** in compute; labels cost the same. Overhanging walls need a meshing and torsion-feature check.
- *Note.* Changes the generator: the owner's go-ahead is needed.

**H6. Split the deformation option into "diversity" and "fabrication" presets.**
- *What.*
  - Keep today's map as "diversity": median maximum displacement 8 mm, 8 % of Req (§4.3).
  - Add a fabrication preset with amplitudes 0.1–0.4 mm (`deform_max` ≈ 0.01–0.02 with the current map, plus
    cell-scale wavelengths), per-cell rigid offsets of 0.2–0.4 mm, per-cell Req and L errors with σ ~0.1–0.2 mm,
    and an optional uniform inward offset of 0.1–0.2 mm. The data-driven alternative is the KL modes of S61
    (code on Zenodo).
  - Generate the fabrication cases as **pairs** (nominal, perturbed) on the same mesh topology, so the frequency
    shift is resolved better than the absolute label error *(inference)*.
- *Benefit.* Realistic tolerance and UQ data, comparable with S6, S60 and S61. It also tells us honestly that
  ~1e-3 fabrication shifts are at or below the model floor, so the model alone cannot resolve them.
- *Cost.* **Cheap**: 1000 pairs ≈ 1000 core-hours (0.2 %).
- *Sources.* S45, S47–S50, S52, S61.
- *Note.* Changes the generator: the owner's go-ahead is needed.

### Medium priority

**M1. Shape-sensitivity labels by Slater's theorem.**
- *What.* Store wall ε0‖E‖² and μ0‖H‖² at the wall quadrature points, or their integrals against the normal
  velocity of each CAD parameter. That gives δf/f for any small normal wall displacement, and the radiation
  pressure for LFD studies.
- *Benefit.* First-order tolerance analysis without new solves (S42, S61, S8, S59). It could also feed derivative
  terms in training *(inference)*.
- *Cost.* **Cheap to moderate**: the fields exist. Wall values take ~1–5 MB per sample; parameter velocities need
  CAD re-evaluation, which is geometry work, not a solve.

**M2. Multi-cell FoMs.**
- *What.* k_cc, field flatness, per-cell R/Q, G·R/Q and passband frequencies. Ensure K ≥ N_cells + 1 for
  elliptical draws.
- *Benefit.* The numbers designers compare (S56, S55, S60, S32).
- *Cost.* **Cheap** to derive. **Moderate** if K must grow (more eigenpairs per adaptive p3 solve; cost not
  measured).

**M3. Mid-cell symmetry labels as a validation set.**
- *What.* A single cell with magnetic walls (n × H = 0) at the iris planes for the π-mode and electric walls for
  the 0-mode. That gives the infinite-chain f_π, f_0, k_cc and mid-cell Epk/Bpk, the objects SLANS-style
  optimisation reports (S27, S28, S35). In the E formulation a magnetic wall is the natural boundary condition
  (standard theory, not from a source here).
- *Benefit.* Direct comparison with the optimisation literature.
- *Cost.* **Cheap** labels (one cell). Using them for **training** needs a boundary-type token feature, which is a
  moderate model change.

**M4. A learning curve before a large production.**
- *What.* 300 / 1000 / 3000 samples on 1–2 families, the error-versus-N fits that S15–S17 report. This is
  docs/31 §6 item 1.
- *Why.* Labels are not the binding constraint; training is. 10k samples cost ~40k core-hours per CPU run
  *(estimate)*, and operator caches take 0.33–0.38 GB per geometry (3.5 TB for 10k).
- *Option.* Pretrain on cheap labels (`--no_adapt` or N0) and fine-tune on p3. The literature reports 10–100×
  label savings (S12, S16).
- *Cost.* **Moderate**, mostly training.

**M5. Ensemble or active sampling.** Use `scripts/active_sampling.py`, or an ensemble disagreement score, to place
labels where the model is uncertain (S3, S14, S2). **Moderate**: needs several trained models.

**M6. Dipole conventions.** Store off-axis V at r0 on the x and y lines and a transverse R/Q with an explicit
convention (S40). **Cheap**.

**M7. Data-driven shape modes.** Collect a small library of published elliptical designs (TESLA, LL, RE, LSF,
NLSF, PIP-II, SNS, FCC) and sample PCA modes of their differences, as BOLD does (S4). This places training data on
the realistic design manifold. **Moderate**: the library must be assembled by hand.

### Low priority

- **L1. A spline- or NURBS-profile elliptical-like family** for more diversity (S1, S3). It must respect the
  curvature floor (≥ 5 % fillet). **Moderate** development, cheap compute.
- **L2. An axisymmetric 2D solver** (monopoles and dipoles via the azimuthal order) for axisymmetric families, as
  a cross-check and cheap multi-fidelity label (S5, S34, S35; docs/31 §5). **Moderate** development, compute
  ~100× cheaper *(estimate)*.
- **L3. New families** with literature interest: RF-dipole crab (S4), photoinjector gun, NC storage-ring cavity
  with S2's open data as an external test. **Moderate** each; closed-PEC compatible.
- **L4. LFD coefficients.** **Infeasible** in the current pipeline: it needs a structural model with wall
  thickness, stiffening and boundary conditions, and published values span −1 to −17 Hz/(MV/m)² depending on
  constraints (S63). Store only the wall pressure (M1).
- **L5. External Q, couplers and multipacting.** **Infeasible** with a closed-PEC real eigenproblem: they need
  port or absorbing boundaries (a complex eigenproblem) or particle tracking. Out of scope.

## 7. Open questions and what I could not verify

1. **Every number is excerpt-level** (§0). Full texts were unreachable. Before anything is quoted in docs/, verify
   the key values against the papers: the Aune table, the ILC TDR table, S1's dataset and accuracy, S31's code
   comparison and the S45 tolerances.
2. **TESLA R/Q.** The Aune excerpt lists 518 Ω for the 9-cell cavity. The commonly quoted ~1036 Ω was not seen in
   any excerpt. The "518 = circuit convention" reading (§3.4) is my inference from the TDR G·R/Q.
3. **±0.2 mm for 90 % / ±0.25 mm for 10 %.** The exact source was not pinned. The ±0.2 mm cell-shape requirement
   itself is in the DESY QA talk (S45).
4. **S51's 14 MHz/mm (iris) and −55 MHz/mm (equator):** the cavity is not identified. −55 MHz/mm is about 4× the
   f/Req scaling for 1.3 GHz, so it may refer to the 3.9 GHz cavity *(inference)*.
5. **Our floor.** Is the docs/30 §5 floor a relative eigenvalue (k²) error or a frequency error? It matters by a
   factor 2 in §5.4.
6. **LL, RE, LSF, ICHIRO and NLSF dimensions** (A, B, a, b) were not retrieved; only LL's Req 98.58 and Riris 30
   were confirmed (S31). Our "ILC low-loss" anchor remains unverified.
7. **Missing details.** Not retrieved: S2's design variables, S3's dataset size and errors, S6's perturbation
   magnitudes, the statistics in S60, the KL amplitudes in S61, the RB snapshot counts and speed-ups in S9, and the
   accuracy figures of S22.
8. **Possible gaps.** Searches did not find RF-cavity ML works using CNNs, GNNs, point clouds or neural operators,
   or any public RF-cavity eigenmode dataset. With the search budget exhausted, Chinese-language journals (NST,
   CPC), IPAC'26 / LINAC'26 and very recent arXiv preprints may hold works this survey missed.
9. **Typical small radii** (nose-cone tips, drift-tube noses, spoke bases) in real designs were not surveyed, so
   the impact of our 5 % fillet floor on Epk realism is unquantified.
10. **Reduced-β estimates.** The dtl β ≈ 0.2–0.35 and the chord inflation factors (1.15–3×) are my estimates from
    our parameter ranges. They should be checked on actual H5 samples: the stored chord, the cell count and
    `deform_max_disp`.
