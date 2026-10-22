# A1 — Elliptical SRF cavities (TM010-type, β ≈ 0.5–1) and their end groups: literature survey and evaluation of our generator

*Research report, 2026-10-22. Families concerned: `elliptical`, `elliptical_long` (OOD), `pillbox_pipes`,
`pillbox_port` (OOD) of `src/data_gen/cavity_shapes.py` (branch `claude/field-model`). No repository file was
edited. The numbers tagged [R] come from scripts in
`scripts/` (files `a1_*.py`;
standalone copies of our half-cell construction and sampler; they import nothing from the repository).*

## 0. Method, limits and provenance tags (read first)

- **No paper could be opened in this sandbox.** `WebFetch` failed with DNS errors for every host, and `curl`
  through the agent proxy got 403 for arxiv.org, accelconf.web.cern.ch, proceedings.jacow.org, journals.aps.org,
  link.aps.org, osti.gov and cds.cern.ch. Every literature number below therefore comes from the excerpts that the
  WebSearch tool returned for the cited document. Excerpts of tables can mis-align columns; where the excerpt
  itself warned about this I say so. The shared WebSearch budget ran out after about 20 queries; about 10 more
  were run after it reset. §6 lists what is still unverified and the searches that would close the gaps.
- Every number carries a provenance tag:
  - **[S]** from a search excerpt of the cited source (full text not opened);
  - **[A]** abstract-level statement only;
  - **[C]** value in an open-source cavity-design code, GitHub `Dark-Elektron/CavityDesignHub` (Rostock/CERN
    FCC-ee cavity-design work), read through GitHub code-search fragments. Secondary, not a publication. Where a
    design is also in an [S] source (TESLA, LL, ICHIRO), the two agree;
  - **[D]** derived by me from tagged inputs, with the formula given;
  - **[E]** my estimate or physical inference;
  - **[U]** widely known fact that I did not re-verify in this session (used only for cell counts and qualitative
    statements);
  - **[R]** our repository (docs, code) or a script run on a standalone copy of our sampler.
- **Geometry convention** (identical to our `half_cell_wall`): half-cell from the iris plane z = 0 to the equator
  plane z = L; iris ellipse with semi-axes a (along z) and b (along r) centred at (0, Ri + b); equator ellipse
  with A (z) and B (r) centred at (L, Req − B); the two are joined by their common tangent. For the accelerating
  π-mode, L = βλ/4.
- **Wall angle θ** = angle between the connecting straight line and the +z (beam) axis, going from iris to
  equator: θ < 90° normal, 90° vertical, θ > 90° re-entrant. This is the convention of `our_generator.md`.
  Shemelin's α = 180° − θ, and a wall with α < 90° is re-entrant [S, src 17]. TESLA mid-cell: θ = 76.7° [R], i.e.
  13.3° from vertical (as in docs/18 [R]) and α = 103.3°; the SLANS-tuned TESLA cell in [C] reports α = 103.17°.
- **R/Q convention.** Ours is the linac one, V²/(ωU). The TESLA paper's 518 Ω is the circuit value V²/(2ωU), i.e.
  1036 Ω in our convention [D]. This is consistent with the per-cell TESLA value of src 4, 113.8 Ω
  (9 × 113.8 = 1024 Ω) [D]. For the other R/Q values the excerpts did not state the convention; I say so where it
  matters.

## 1. Scope and sources

Scope: elliptical TM010 cavities with β ≈ 0.5–1 (1.3 GHz ILC/XFEL-type cells; CEBAF 1497 MHz; β < 1 proton linacs
at 650–805 MHz; large-aperture storage-ring and collider cavities at 400–800 MHz; 3.9 GHz harmonic cavities), their
end cells, beam pipes and couplers. Not covered: crab cavities (the HL-LHC ones, DQW and RFD, are not elliptical
[U]) and spoke / HWR / QWR structures.

**Shapes for 1.3 GHz, β = 1**

1. B. Aune et al., *Superconducting TESLA cavities*, Phys. Rev. ST Accel. Beams 3, 092001 (2000), DESY 00-031,
   DOI 10.1103/PhysRevSTAB.3.092001 — https://arxiv.org/abs/physics/0003011 ,
   https://link.aps.org/pdf/10.1103/PhysRevSTAB.3.092001 , https://arxiv.org/html/physics/0003011 .
   Table III, half-cell shapes of mid-cup and end-cups 1 and 2 [S]; RF table (G, R/Q, Epk/Eacc, Bpk/Eacc, kcc,
   active length) [S]; coaxial FPC (Qext 3·10⁶), two HOM couplers per cavity with a 1.3 GHz notch set by a 1.9 mm
   gap [S].
2. H. Padamsee, *Design topics for superconducting RF cavities and ancillaries*, CAS 2013, CERN-2014-005 —
   https://arxiv.org/pdf/1501.07129 . TESLA figures of merit [S]; aperture trade-off (small aperture lowers peak
   ratios, large aperture lowers wakefields) [S]; advanced shapes (RE, LL, ICHIRO, LSF) lower Bpk/Eacc by roughly
   10–15 % [S].
3. J. Sekutowicz, *Superconducting elliptical cavities*, CAS 2010 — https://arxiv.org/pdf/1201.2598 . TESLA and
   LL inner-cell kcc and Epk/Eacc [S]; the original CEBAF (OC) cell has a very high kcc, and the LL shape a
   dissipation factor 37 % above OC [S].
4. J. Sekutowicz et al., *Design of a low loss SRF cavity for the ILC*, PAC 2005, TPPT056, pp. 3342–3344,
   DOI 10.1109/PAC.2005.1591465, SLAC-PUB-11730 — https://accelconf.web.cern.ch/AccelConf/p05/PAPERS/TPPT056.PDF ,
   https://inspirehep.net/files/a37adb02247bba6b0b8d23cfebb18180 . TESLA vs LL inner-cell table (iris, kcc,
   Epk/Eacc, Bpk/Eacc, R/Q per cell) [S]; cryogenic loss 23 % below TESLA; main drawback the higher Epk/Eacc [S].
5. Z. Li et al., *Optimization of the low loss SRF cavity for the ILC*, PAC 2007, SLAC-PUB-13088 —
   https://www.slac.stanford.edu/pubs/slacpubs/13000/slac-pub-13088.pdf , https://www.osti.gov/biblio/921642 .
   LL cell: 30 mm iris, a more upright side wall than TESLA, > 10 % lower Bpk/Eacc, 15 % higher R/Q and G;
   multipacting found in the end groups, end groups optimised for HOM damping [A].
6. N. Juntong, R. M. Jones, *High-gradient SRF cavity with minimized surface e.m. fields and superior bandwidth
   for the ILC*, SRF 2009, Cockcroft-09-35 — https://arxiv.org/abs/0910.1305 ,
   https://epubs.stfc.ac.uk/manifestation/4216/Cockcroft-09-35.pdf . Table 1: Ri, Req, A, B, a, b of TESLA, LL,
   ICHIRO, NLSF, NLSF-A, NLSF-RE [S, partially; values for LL and ICHIRO confirmed, see §3].
7. N. Juntong, R. M. Jones, *SRF cavity geometry optimization for the ILC with minimized surface e.m. fields and
   superior bandwidth* — https://arxiv.org/pdf/0906.3592 . ICHIRO Req 98.14 mm, Ri 30 mm, A 50.052 mm; full-cell
   length fixed at λ/2 = 115.304 mm [S].
8. N. Juntong, R. M. Jones, IPAC'10 WEPEC053 — https://arxiv.org/pdf/1006.0086 ,
   https://proceedings.jacow.org/IPAC10/papers/wepec053.pdf . NLSF bandwidth about 26.5 % wider than LSF; NLSF takes
   the iris thickness of LSF and Req of LL [S].
9. A. Yamamoto (ILC GDE), *Superconducting RF cavity development for the ILC*, ASC 2008, 2LX01 —
   https://agenda.linearcollider.org/event/2845/contributions/9740/attachments/7585/12687/ASC-2LX01-080812.pdf .
   Table TESLA / LL / RE: iris aperture, R/Q per cell, G [S]; Epk/Eacc of the alternative shapes 10–20 % above
   TESLA [S].
10. R. L. Geng et al., *World record accelerating gradient achieved in a superconducting niobium RF cavity*,
    PAC 2005, ROAC009 — https://proceedings.jacow.org/p05/PAPERS/ROAC009.PDF . Re-entrant (RE) single cell with a
    70 mm aperture; Hpk/Eacc and kcc vs TESLA [S; the excerpt said the table was garbled].
11. CERN Courier, *Records fall at Cornell* — https://cerncourier.com/a/records-fall-at-cornell/ . RE-70 single
    cell: 46 MV/m at Q0 > 10¹⁰, 47 MV/m pulsed; a 60 mm aperture would lower Bpk further [S].
12. R. L. Geng et al., *High gradient studies for ILC with single-cell re-entrant shape and elliptical shape
    cavities made of fine-grain and large-grain niobium*, PAC 2007, WEPMS006 —
    https://epaper.kek.jp/p07/PAPERS/WEPMS006.PDF . RE-60: Hpk/Eacc 15 % below TESLA; 52 MV/m [S].
13. H. Padamsee, *History of gradient advances in SRF* — https://arxiv.org/pdf/2004.06720 . RE: Hpk/Eacc
    37.8 Oe/(MV/m) vs 42 for TESLA, Epk/Eacc 2.4 [S].
14. *The International Linear Collider: Report to Snowmass 2021* — https://arxiv.org/pdf/2203.07622 . RE (60 mm
    aperture) Epk/Eacc 2.28 vs 2.0 for TESLA (70 mm); G·R/Q of RE about 34 % above TESLA [S].
15. *ILC Technical Design Report, Volume 3.I* — https://arxiv.org/pdf/1306.6353 . Three alternative shapes (RE,
    LL, LSF): up to 15 % higher effective gradient, at the cost of lower kcc and higher HOM loss factors [S].
16. V. Shemelin, *Optimal choice of cell geometry for a multicell superconducting cavity*, PRST-AB 12, 114701
    (2009), DOI 10.1103/PhysRevSTAB.12.114701 —
    https://www.classe.cornell.edu/rsrc/Home/Research/SRF/2009/Optimal_choice_of_cell_geometry_for_a_multicell_superconducting_cavity.pdf .
    Inner cells optimised for minimal losses or minimal Bpk at given aperture, Epk/Eacc and wall slope angle; lower
    wall angles (re-entrant cells) also benefit the end cells; end cells optimised with and without end irises
    [A].
17. V. Shemelin, V. Yakovlev et al., *Optimization of a traveling wave superconducting rf cavity for upgrading the
    ILC*, PRAB 25, 021001 (2022), DOI 10.1103/PhysRevAccelBeams.25.021001 —
    https://journals.aps.org/prab/pdf/10.1103/PhysRevAccelBeams.25.021001 , https://arxiv.org/pdf/2105.12276 .
    Definition: a cell with wall angle α below 90° is re-entrant [S].
18. S. Gorgi Zadeh et al., *Constrained multiobjective shape optimization of superconducting rf cavities
    considering robustness against geometric perturbations*, PRAB 22, 122001 (2019) —
    https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.22.122001 . Title only (not read).
19. H. Weise, TTC meeting summary, FNAL 2007 —
    https://agenda.linearcollider.org/event/1539/contributions/3423/attachments/2545/4303/TTC_Meeting_Summary.pdf .
    KEK LL single cell Hp/Eacc = 35 Oe/(MV/m) [S].
20. K. Sakai et al. (KEK), *Development of a 1.3 GHz 9-cell superconducting cavity* (KEK-ERL) —
    https://indico.cern.ch/event/15495/contributions/203911/attachments/165501/233340/SAKAI_SC.pdf . TESLA kcc 1.9 %,
    Hp/Eacc 42.6 Oe/(MV/m) [S].
21. ICFA Beam Dynamics Newsletter, *The baseline configuration for the ILC* —
    https://labcit.ligo.caltech.edu/~BCBAct/talks06/ICFA%20Beam%20Dynamics%20Newletter%2008-06/The%20Baseline%20Configuration%20for%20the%20ILC.pdf .
    TESLA shape: favourable low Epk/Eacc, acceptable kcc and wakefield loss factors [S, qualitative].

**CEBAF (1497 MHz)**

22. *First cryogenic tests with JLab's new upgrade cavities*, LINAC 2004, MOP84 —
    https://proceedings.jacow.org/l04/PAPERS/MOP84.PDF . Inner cells OC / HG / LL: equator and iris diameters, kcc
    [S; the excerpt gave the values as "roughly"]; HG optimised for Epk/Eacc, LL for R/Q·G [S].
23. JLab, *CEBAF at 12 GeV* — https://www.jlab.org/IR/pubs/CEBAFat12GeV/CEBAF_at_12_GeV(39).pdf . C100 cavities
    designed for CW operation at up to 19.3 MV/m [S].
24. Figure captions (ResearchGate): *LL seven cell cavity* —
    https://www.researchgate.net/figure/LL-seven-cell-cavity_fig3_238705314 ; *CEBAF 5-cell OC cavity with the cold
    HOM loads inserted in two HOM waveguides* —
    https://www.researchgate.net/figure/CEBAF-5-cell-OC-cavity-with-the-cold-HOM-loads-inserted-in-two-HOM-waveguides_fig1_266006724
    [S, captions only].

**β < 1 proton linacs**

25. SNS — *Physics of the SNS SRF cavity*, APAC 2004, MOM402, https://proceedings.jacow.org/a04/PAPERS/MOM402.PDF ;
    *SNS superconducting linac*, https://www.osti.gov/etdeweb/servlets/purl/20226960 ; G. Ciovati et al.,
    *Superconducting prototype cavities for the SNS project*,
    https://www.researchgate.net/publication/3939076_Superconducting_prototype_cavities_for_the_Spallation_NeutronSource_SNS_project .
    805 MHz 6-cell, β 0.61 / 0.81; peak fields at design gradient; inter-cell coupling 1.6 % [S; the excerpt did
    not say which of these three documents holds the parameter table].
26. ESS — *ESS prototype cavities developed at CEA Saclay*, SRF 2019, THP096,
    https://proceedings.jacow.org/srf2019/papers/thp096.pdf ; *Vertical test results on ESS medium beta elliptical
    cavity prototype*, https://inspirehep.net/files/15374358f0fe9a8e19f3ceea4c485dbe ; *ESS medium and high beta
    cavity prototypes*, https://inspirehep.net/files/5de42373ca48508a2730fc479d08b7c8 . 704.42 MHz, β 0.67 6-cell
    and β 0.86 5-cell; Epk/Eacc, Bpk/Eacc, R/Q, G, gradients; kcc 1.2 % (medium β) [S].
27. PIP-II — *650 MHz elliptical superconducting RF cavities for PIP-II project*, LINAC 2016, THPLR041,
    https://proceedings.jacow.org/linac2016/papers/thplr041.pdf . LB650 (β 0.61) and HB650 (β 0.92), both 5-cell,
    shapes revised for pulsed and CW operation [S].
28. *An 8 GeV linac as the Booster replacement in the Fermilab power upgrade*, Snowmass 2021 white paper —
    https://arxiv.org/pdf/2203.05052 . HB650 iris aperture 118 mm, chosen as a compromise of peak field ratios,
    shunt impedance, kcc and handling during surface processing; FPC Qext 1.2·10⁷ [S].
29. *Q-factor optimization for high-beta 650 MHz cavities for PIP-II* — https://arxiv.org/pdf/2111.01894 ,
    https://www.osti.gov/servlets/purl/1832813 . HB650 design point 18.8 MV/m, Bpk 73.1 mT, Q0 3·10¹⁰ [S].
30. *Electromagnetic design of βg = 0.9, 650 MHz superconducting radiofrequency cavity* —
    https://arxiv.org/pdf/1208.4245 . Title only (not read).

**Storage rings and circular colliders (1–5 cells, large apertures)**

31. CEPC 650 MHz 2-cell — *Mechanical design of a 650 MHz superconducting RF cavity for CEPC*, SRF 2017, TUPB038,
    https://proceedings.jacow.org/srf2017/papers/tupb038.pdf ; *Development and vertical tests of 650 MHz 2-cell
    superconducting cavities with higher order mode couplers*, NIM A (2021),
    https://www.sciencedirect.com/science/article/abs/pii/S0168900221000772 ; *HOM damping requirements and
    measurements for CEPC 650-MHz 2-cell cavity*, https://www.researchgate.net/publication/337275468 . R/Q, G,
    Epk/Eacc, Bpk/Eacc, length, equator diameter; beam-tube cut-offs; two coaxial HOM couplers on the beam pipe;
    trapped modes and multipacting in a first design with four identical half-cells [S].
32. KEKB — *Status of KEKB superconducting cavities and study for future SKEKB*,
    https://www.researchgate.net/publication/242226325 ; *Operating experience with β = 1 high current ...*
    (Cornell), https://www.classe.cornell.edu/public/SRF/2003/SRF031215-19/SRF031215-19.pdf . 509 MHz single cell,
    220 mm iris diameter, ferrite HOM absorbers on both sides, 150 mm beam ducts, HOM power up to 2.5 kW per module
    [S].
33. CESR-B — H. Padamsee et al., *Accelerating cavity development for the Cornell B-factory, CESR-B*, PAC 1991,
    https://accelconf.web.cern.ch/accelconf/p91/PDF/PAC1991_0786.PDF ,
    https://www.researchgate.net/publication/224635914 . 500 MHz single cell; large beam pipes (240 mm diameter on
    one side, excerpt garbled), the other pipe fluted to lower its TE11 cut-off to 570 MHz; ferrite loads; HOM
    Q ≈ 100; waveguide FPC with Qext down to 5·10⁴ through a C-slot iris [S].
34. FCC-ee — *SRF cavity development for the FCC-ee at 400/800 MHz*, https://meow.elettra.eu/89/pdf/FRA05.pdf ;
    *Progress and R&D challenges for FCC-ee SRF*, EPJ Tech. Instrum. (2023),
    https://epjtechniquesandinstrumentation.springeropen.com/articles/10.1140/epjti/s40485-023-00094-5 ;
    S. Gorgi Zadeh, *Cavity design approaches and HOM damping for FCC-ee*,
    https://indico.cern.ch/event/556692/contributions/2484333/attachments/1466381/2267103/Cavity_Design_approaches_and_HOM_damping_for_FCC-ee_-_Shahnam_Gorgi_Zadeh.pdf .
    400 MHz Nb/Cu single cells at the Z, 4-cell (some documents: 2-cell) 400 MHz Nb/Cu at W and ZH, 5-cell bulk-Nb
    800 MHz at tt̄ [S]. No geometry in the excerpts.
35. *The Large Hadron–Electron Collider at the HL-LHC* — https://arxiv.org/pdf/2007.14491 . 802 MHz 5-cell
    prototype, equator diameter 327.95 mm [S].

**3.9 GHz harmonic cavities**

36. Fermilab-TM-2210, *Harmonic cavity with an increased end-cup iris* —
    https://lss.fnal.gov/archive/test-tm/2000/fermilab-tm-2210.pdf . Mid-cell iris radius 15 mm, end-cell iris
    20 mm, equator radius 35.787 mm [S]. Also FERMILAB-CONF-03-480-TD, *Development of the third harmonic SC
    cavity at Fermilab*, https://lss.fnal.gov/archive/2003/conf/fermilab-conf-03-480-td.pdf (title only).

**Couplers, Qext, LL-type variants**

37. External Q with closed eigensolvers — *Simulations and measurements of a heavily HOM-damped multi-cell SRF
    cavity*, PAC 2007, WEPMS070, https://proceedings.jacow.org/p07/PAPERS/WEPMS070.PDF ; F. Marhauser,
    *Calculations for RF cavities with dissipative material*, SRF 2015, THPB003,
    https://proceedings.jacow.org/SRF2015/papers/thpb003.pdf ; *Evaluation of external Q using Kroll-Yu method
    with Microwave Studio*, https://www.osti.gov/etdeweb/biblio/21098056 ; *Circuit-model representation of
    external-Q calculation*, https://www.researchgate.net/publication/243478065 ; N. Juntong, R. M. Jones, *HOM and
    FP coupler design for the NLSF high gradient SC cavity*, https://arxiv.org/pdf/1110.1213 . Balleyguier method
    (two eigen-runs with electric and magnetic terminations of the coupler line), Kroll–Yu method, CST's modified
    Balleyguier macro, lossy-absorber ports [S].
38. *Stop band characteristics of a TESLA cavity coaxial-type HOM coupler* —
    https://www.academia.edu/7069947/ACCELERATORS_Stop_band_characteristics_of_a_TESLA_cavity_coaxial_type_HOM_coupler .
    Notch at 1.3 GHz with S21 −62 dB, notch tuning about 175 MHz/mm, bandwidth ±10 MHz [S].
39. *Studies of LL-type 500 MHz 5-cell superconducting cavity at SINAP* — https://arxiv.org/pdf/1407.3003 .
    LL-type cell with an elliptical contour and a straight section at the equator [S].

**Code and our repository**

40. `Dark-Elektron/CavityDesignHub` (GitHub) — https://github.com/Dark-Elektron/CavityDesignHub . Files
    `cli/test_notebook.ipynb`, `manual_run/abci/abciWakefield.py`, `manual_run/slans/slansEigen.py`,
    `analysis_modules/eigenmode/customEig/run_field_solver.py`, `utils/write_geometry_boundary_multipac.py`,
    `analysis_modules/eigenmode/SLANS/MidCellTune/tuned.tgp`: half-cell sets [A, B, a, b, Ri, L, Req] of TESLA
    (published and SLANS-tuned), ILC-LL, ICHIRO, NLSF, NLSF-RE, Cornell ERL main linac ("ERL_MA"), FCC-ee "C3794"
    (400 MHz), "C3795" (800 MHz), "FCCUROS5", TESLA scaled to 800 MHz [C].
41. Our repository — docs/18 (axisymmetric TESLA check: kcc 1.886 % vs 1.87 % design), docs/19 (elliptical v2),
    docs/29 (curved tets per family), docs/30 (port-junction folding, sharp fallback),
    `src/data_gen/cavity_shapes.py`, `src/data_gen/dataset_generator_3d.py` (`--n_eigen_modes` default 6) [R].

Of these 41 entries, 26 give numerical geometry or figures of merit (1–4, 6, 7, 9–14, 19, 20, 22, 25, 26, 28, 29,
31–33, 35, 36, 38, 40).

## 2. Cavity types in this area

### 2.1 Canonical cell and how designers parametrise it

- **Seven numbers per half-cell** (Req, Ri, L, A, B, a, b) define the TESLA-type cell; the wall angle follows
  from the common tangent. This is exactly our parametrisation (`half_cell_wall` [R]) and the input set of the
  design codes in [C] (`[A, B, a, b, Ri, L, Req]`, with α as an output of the tuning step).
- **Design flow described in the sources.** L is fixed by β and the frequency (L = βλ/4; for 1.3 GHz, β = 1 the full
  cell is λ/2 = 115.304 mm [S, src 7]). The aperture Ri is chosen first: a small aperture lowers Epk/Eacc and
  Hpk/Eacc, a large one lowers wakefields [S, src 2]; HB650's 118 mm aperture is a compromise of peak-field
  ratios, shunt impedance, kcc and processing [S, src 28]. The ellipses and the wall angle then trade Epk/Eacc
  against Bpk/Eacc and G·R/Q (CEBAF: HG optimised for Epk/Eacc, LL for R/Q·G [S, src 22]; Shemelin optimises at a
  given aperture, Epk/Eacc and wall slope angle [A, src 16]). Finally Req is tuned to put the π-mode on frequency:
  the SLANS-tuned TESLA cell in [C] has Req 103.353 mm and L 57.6524 mm at 1300.000 MHz, against the rounded
  103.3 / 57.7 mm of the published table.
- **End groups.** The end half-cell iris equals the beam-pipe radius, larger than the mid iris (TESLA 39 vs 35 mm
  [S, src 1]). The end half-cells are reshaped, and the two ends of TESLA differ from each other: a 12 → 10 / 9 mm,
  b 19 → 13.5 / 12.8 mm, A = B 42 → 40.3 / 42 mm, L 57.7 → 56.0 / 57.0 mm for end-cups 1 / 2 [S, src 1]. Shemelin
  optimises end cells separately, with and without end irises [A, src 16]. CEPC's first 2-cell design used four
  identical half-cells to save tooling; it had two trapped modes and a wide multipacting band (10–19 MV/m) and was
  changed [S, src 31].
- **Coupling and field flatness.** kcc = 2(fπ − f0)/(fπ + f0) (docs/18 [R]). In the coupled-oscillator picture
  the field-profile sensitivity to cell-frequency errors grows like N²/kcc [E: standard result, not re-verified in
  this session].
- **Wall angles of real cells** (θ, our convention, computed with our construction from the §3 geometries [R]):
  TESLA 76.7° (end-cups 73.3° and 72.3°); Cornell ERL 85.0°; FCC-ee 800 MHz "C3795" 85.5°, "FCCUROS5" 79.9°,
  400 MHz "C3794" 72.8°; LL, ICHIRO and NLSF 89.9°, i.e. a vertical side wall ("a nearly parallel cavity side"
  [S, src 4]); NLSF-RE 101.1° (re-entrant); the Cornell RE cells are re-entrant [S, src 10–14; angle not
  retrieved].

### 2.2 Classes: frequency, β, cell count, typical ratios and figures of merit

Ranges are taken over the designs of §3 (provenance per row there).

| class | examples (cells) | f (MHz), β | Ri/Req (mid) | L/Req | A/L, B/A, a/L, b/a | θ (°) | Epk/Eacc | Bpk/Eacc mT/(MV/m) | kcc % | G (Ω) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.3 GHz linac | TESLA / XFEL / ILC (9), LCLS-II (9, TESLA shape [U]); LL, ICHIRO (9); RE (single-cell tests); NLSF; Cornell ERL (7 [U]) | 1300, β = 1 | 0.30–0.35 | 0.56–0.59 | 0.73–0.87, 0.665–1.00, 0.13–0.22, 1.31–1.67 | 76.7–89.9; RE > 90 | 1.98–2.4 | 3.61–4.26 | 1.52–2.4 | 270–285 |
| CEBAF | OC (5), HG, LL = C100 (7) | 1497, β = 1 | 0.305–0.374 | 0.535–0.575 [D] | not retrieved | — | — | — | 1.49–3.29 | — |
| β < 1 proton linacs | SNS (6), ESS medium (6) / high (5), PIP-II LB650 / HB650 (5) | 805, 704.42, 650; β 0.61–0.92 | not retrieved (HB650: Ri 59 mm) | ≈ (0.54–0.57)·β [E] | not retrieved | — | 2.2–2.72 | 3.89–5.74 | 1.2–1.6 | 196.6–241 |
| storage rings, circular colliders | KEKB (1), CESR-B (1), CEPC (2), FCC-ee 400 MHz (1; 2 or 4), FCC-ee 800 MHz (5), LHeC 802 MHz (5) | 400–802, β = 1 | 0.36–0.42 [C]; pipe radius / Req 0.38–0.47 | 0.506–0.57 | A/L 0.39–0.72, B/A 0.85–1.79, a/L 0.23–0.57, b/a 0.77–1.65 [C] | 72.8–85.5 [C] | 2.38 (CEPC) | 4.17 (CEPC) | — | 284.1 (CEPC) |
| 3.9 GHz harmonic | FNAL / FLASH / XFEL (9 [U]) | 3900, β = 1 | 0.419 (end iris 0.559) | 0.537 [D] | not retrieved | — | — | — | — | — |

Couplers in this class (what they add to the RF volume):

- **TESLA:** one coaxial FPC on a beam tube, Qext = 3·10⁶ at 25 MV/m, tunable (a range from 10¹⁰ down to 3·10⁶ is
  also quoted), > 200 kW [S, src 1]; two HOM couplers per 9-cell cavity, a loop antenna coupling to H for TE modes
  and to E for TM modes, with the 1.3 GHz notch formed by the loop inductance and the capacitance of a 1.9 mm gap
  between loop and wall; flanged and welded variants [S, src 1]. The coaxial HOM coupler's notch: S21 −62 dB at
  1.3 GHz, tuning sensitivity about 175 MHz/mm, bandwidth ±10 MHz [S, src 38].
- **CEPC 2-cell:** two detachable coaxial HOM couplers on the beam pipe, 1 kW each; they must damp HOMs between
  780 and 1471 MHz; 1.95 kW of HOM power per cavity at the Z pole [S, src 31].
- **KEKB:** 220 mm iris and ferrite HOM absorbers on both beam pipes, which propagate the HOMs out [S, src 32].
  **CESR-B:** large beam pipes, one fluted to lower its TE11 cut-off to 570 MHz, ferrite loads, HOM Q ≈ 100,
  waveguide FPC (C-slot iris, Qext down to 5·10⁴) [S, src 33].
- **CEBAF OC:** waveguide HOM couplers with cold loads [S, src 24, caption].
- **Simulation practice for Qext:** closed eigen-solves with two terminations of the coupler line (Balleyguier:
  one electric-type and one magnetic-type termination; the two partial Q's combine into Qext independently of the
  phase), the Kroll–Yu method (also eigenmode based), CST's macro (a modified Balleyguier method), or a lossy
  absorber / port at the coupler end, which gives Qext and the loaded frequency in one (complex) eigen-run
  [S, src 37].

## 3. Reference designs

### 3.1 Geometry (mm) and dimensionless ratios, tested against our sampler

"Outside" lists the random ranges of `_draw_half` that the cell violates (Ri/Req 0.25–0.42, L/Req 0.45–0.65 at
β = 1, A/L 0.6–0.85, B/A 0.85–1.25, a/L 0.15–0.30, b/a 1.0–1.9). "`_valid_half`" is the outcome of our validity
test (non-overlap rule Ri + b < Req − B, tangent solve, monotone wall) on the exact cell, run on a standalone copy
of the code [R]. End cells are compared with their mid cell in §4.1.

| # | design | f (MHz), β, cells | Ri | Req | L | A | B | a | b | Ri/Req | L/Req | A/L | B/A | a/L | b/a | θ (°) | outside | `_valid_half` | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TESLA mid-cup | 1300, 1, 9 | 35 | 103.3 | 57.7 | 42 | 42 | 12 | 19 | 0.339 | 0.559 | 0.728 | 1.000 | 0.208 | 1.583 | 76.7 | none | accepted | [S] src 1 Table III; same in [C] |
| 2 | TESLA end-cup 1 | 1300, 1, 9 | 39 | 103.3 | 56.0 | 40.3 | 40.3 | 10 | 13.5 | 0.378 | 0.542 | 0.720 | 1.000 | 0.179 | 1.350 | 73.3 | (end cell) | accepted | [S] src 1; [C] |
| 3 | TESLA end-cup 2 | 1300, 1, 9 | 39 | 103.3 | 57.0 | 42 | 42 | 9 | 12.8 | 0.378 | 0.552 | 0.737 | 1.000 | 0.158 | 1.422 | 72.3 | (end cell) | accepted | [S] src 1; [C] |
| 4 | LL mid (ILC low-loss) | 1300, 1, 9 | 30 | 98.58 | 57.7 | 50.052 | 36.50 | 7.6 | 10 | 0.304 | 0.585 | 0.867 | 0.729 | 0.132 | 1.316 | 89.9 | A/L, B/A, a/L | accepted (θ at the limit) | [S] src 6, 7; same in [C] |
| 5 | ICHIRO mid (KEK, LL-type) | 1300, 1, 9 | 30 | 98.14 | 57.7 | 50.052 | 34.222 | 7.6 | 9.945 | 0.306 | 0.588 | 0.867 | 0.684 | 0.132 | 1.309 | 89.9 | A/L, B/A, a/L | accepted (θ at the limit) | [S] src 6, 7; same in [C] |
| 6 | our `ilc_ll` anchor | — | 30 | 98.6 | 57.7 | 50.1 | 34.2 | 7.6 | 10.0 | 0.304 | 0.585 | 0.868 | 0.683 | 0.132 | 1.316 | 90.0 | A/L, B/A, a/L | accepted (θ at the limit) | [R] `ANCHORS` |
| 7 | NLSF mid | 1300, 1 | 32 | 98.58 | 57.7 | 47.152 | 31.35 | 10.5 | 15.5 | 0.325 | 0.585 | 0.817 | 0.665 | 0.182 | 1.476 | 89.9 | B/A | accepted | [C] (design of src 6) |
| 8 | NLSF-RE mid | 1300, 1 | 32 | 98.58 | 57.7 | 49 | 35.30 | 10.5 | 17 | 0.325 | 0.585 | 0.849 | 0.720 | 0.182 | 1.619 | 101.1 | B/A | rejected: re-entrant | [C] (design of src 6) |
| 9 | Cornell ERL main linac, mid | 1300, 1, 7 [U] | 35 | 101.205 | 57.652 | 43.99 | 35.06 | 12.53 | 20.95 | 0.346 | 0.570 | 0.763 | 0.797 | 0.217 | 1.672 | 85.0 | B/A | accepted | [C] |
| 10 | Cornell ERL, end (left) | 1300, 1 | 39 | 101.205 | 59.988 | 50.9 | 45.3 | 8.4 | 11.5 | 0.385 | 0.593 | 0.849 | 0.890 | 0.140 | 1.369 | 85.0 | (end cell) | accepted | [C] |
| 11 | Cornell ERL, end (right) | 1300, 1 | 37 | 101.205 | 62.665 | 52.1 | 47.9 | 9.9 | 11.3 | 0.366 | 0.619 | 0.831 | 0.919 | 0.158 | 1.141 | 84.9 | (end cell) | **rejected: solver tolerance** (valid wall exists) | [C] |
| 12 | FCC-ee 400 MHz "C3794" mid | 400.8, 1, 2 ([C] label "2-Cell") | 150 | 369.63 | 187 | 73.52 | 131.75 | 106.25 | 118.7 | 0.406 | 0.506 | 0.393 | 1.792 | 0.568 | 1.117 | 72.8 | A/L, B/A, a/L | **rejected: Ri + b ≥ Req − B** (a monotone wall exists) | [C] |
| 13 | FCC-ee 800 MHz "C3795" mid | 801.6, 1, 5 | 72 | 171.2 | 93.5 | 62.22 | 66.13 | 30.22 | 23.11 | 0.421 | 0.546 | 0.665 | 1.063 | 0.323 | 0.765 | 85.5 | Ri/Req (marginal), a/L, b/a | accepted | [C] |
| 14 | FCC-ee 800 MHz "C3795" end | 801.6, 1, 5 | 80 | 171.2 | 93.795 | 62.58 | 57.54 | 17.21 | 12.0 | 0.467 | 0.548 | 0.667 | 0.919 | 0.183 | 0.697 | 67.6 | (end cell) | **rejected: solver tolerance** (valid wall exists) | [C] |
| 15 | 800 MHz "FCCUROS5" mid | 801.6, 1, 5 | 60 | 166.591 | 93.5 | 67.72 | 57.45 | 21.75 | 35.95 | 0.360 | 0.561 | 0.724 | 0.848 | 0.233 | 1.653 | 79.9 | B/A (marginal) | accepted | [C] |
| 16 | CEBAF OC inner cell | 1497, 1, 5 | 35.0 | 93.5 | 50.07 [D] | — | — | — | — | 0.374 | 0.535 | — | — | — | — | — | none of the two | — | [S] src 22 (Ø 70.0 / 187.0) |
| 17 | CEBAF HG inner cell | 1497, 1 | 30.7 | 90.25 | 50.07 [D] | — | — | — | — | 0.340 | 0.555 | — | — | — | — | — | none of the two | — | [S] src 22 (Ø 61.4 / 180.5) |
| 18 | CEBAF LL inner cell (C100) | 1497, 1, 7 [S, src 24] | 26.5 | 87.0 | 50.07 [D] | — | — | — | — | 0.305 | 0.575 | — | — | — | — | — | none of the two | — | [S] src 22 (Ø 53.0 / 174.0) |
| 19 | 3.9 GHz (FNAL) | 3900, 1, 9 [U] | 15 (end 20) | 35.787 | 19.22 [D] | — | — | — | — | 0.419 (end 0.559) | 0.537 | — | — | — | — | — | none (Ri/Req at the limit; end iris ×1.333) | — | [S] src 36 |
| 20 | CEPC 650 MHz | 650, 1, 2 | pipe 78.0 [D] | 205 | 115.3 [D] | — | — | — | — | pipe 0.380 | 0.562 | — | — | — | — | — | — | — | [S] src 31 (Ø 410 mm; pipe from TM01 / TE11 cut-offs 1471 / 1126 MHz [D]) |
| 21 | LHeC 802 MHz | 802, 1, 5 | — | 163.98 | 93.5 [D] | — | — | — | — | — | 0.570 | — | — | — | — | — | — | — | [S] src 35 (Ø 327.95 mm) |
| 22 | PIP-II HB650 | 650, 0.92, 5 | 59 | — | 106.1 [D] | — | — | — | — | — | — | — | — | — | — | — | — | — | [S] src 28 (aperture 118 mm) |
| 23 | KEKB single cell | 509, 1, 1 | 110 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | [S] src 32 (iris Ø 220 mm) |
| 24 | CESR-B single cell | 500, 1, 1 | pipe 120 (one side; excerpt garbled) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | [S] src 33 |

L for rows 16–22 is βλ/4 [D]; for the single cells of rows 23–24 the cell length need not be λ/2, so I give none.
Reproduce rows 1–15 with `scripts/a1_ratios.py`.

### 3.2 Figures of merit

| design | f (MHz) | β | cells | R/Q (Ω), convention | G (Ω) | Epk/Eacc | Bpk/Eacc mT/(MV/m) | kcc (%) | Q0 / other | source |
|---|---|---|---|---|---|---|---|---|---|---|
| TESLA cavity | 1300 | 1 | 9 | 518 circuit = 1036 linac [D] | 270 | 2.0 | 4.26 | 1.87 | active length 1.038 m; FPC Qext 3·10⁶ at 25 MV/m | [S] src 1, 2 |
| TESLA inner cell | 1300 | 1 | per cell | 113.8 (114) per cell; linac by consistency with the 9-cell value [D] | 271 | 1.98 | 4.15 | 1.9 | iris Ø 70 mm; Hp/Eacc 42.6 Oe/(MV/m) in the KEK-ERL table | [S] src 4, 9, 3, 20 |
| LL inner cell | 1300 | 1 | per cell | 133.7 (134; the ASC table also shows 138 for a second LL-type variant) | 284 (285) | 2.36 | 3.61 | 1.52 | iris Ø 60 mm (61 for the second variant); cryogenic loss −23 % vs TESLA; KEK LL single cell Hp/Eacc 35 Oe/(MV/m) | [S] src 4, 9, 3, 19 |
| RE-70 (Cornell) | 1300 | 1 | 1 (tests) | — | — | 2.4 | 3.78 (37.8 Oe/(MV/m)) | 2.4 (TESLA 1.9 in the same table) | 46 MV/m at Q0 > 10¹⁰, 47 MV/m pulsed | [S] src 10, 11, 13 |
| RE, 66 mm aperture (ASC table) | 1300 | 1 | per cell | 127 | 277 | — | — | — | — | [S] src 9 |
| RE-60 | 1300 | 1 | 1 | — | — | 2.28 | 15 % below TESLA | — | G·R/Q about 34 % above TESLA; 52 MV/m | [S] src 12, 14 |
| CEBAF OC / HG / LL inner cells | 1497 | 1 | 5 / – / 7 | — | — | (HG optimised for it) | — | 3.29 / 1.72 / 1.49 | LL dissipation factor 37 % above OC; C100 CW up to 19.3 MV/m | [S] src 22, 3, 23 |
| SNS medium / high | 805 | 0.61 / 0.81 | 6 | — | — | 2.72 / 2.24 [D] | 5.74 / 4.87 [D] | 1.6 | Epk 27.5 / 35 MV/m and Bpk 58 / 76 mT at Eacc 10.1 / 15.6 MV/m; 33 medium + 60 high planned, 81 built (48 high) | [S] src 25 |
| ESS medium / high | 704.42 | 0.67 / 0.86 | 6 / 5 | 367 / 435, convention not stated | 196.6 / 241 | 2.36 / 2.2 | 4.79 / 4.3 | 1.2 (medium) | Eacc 16.7 / 19.9 MV/m; 36 + 84 cavities | [S] src 26 |
| PIP-II HB650 | 650 | 0.92 | 5 | — | — | — | 3.89 at the design point [D] | — | 18.8 MV/m with Bpk 73.1 mT, Q0 3·10¹⁰; FPC Qext 1.2·10⁷ | [S] src 28, 29 |
| CEPC | 650 | 1 | 2 | 212.7, convention not stated | 284.1 | 2.38 | 4.17 | — | length 1060 mm; Q0 3.1·10¹⁰ at 20 MV/m with HOM couplers | [S] src 31 |
| KEKB | 509 | 1 | 1 | — | — | — | — | — | HOM power ≤ 2.5 kW per module in the ferrites | [S] src 32 |
| CESR-B | 500 | 1 | 1 | — | — | — | — | — | HOM Q ≈ 100 with ferrite; FPC Qext down to 5·10⁴ | [S] src 33 |

Units: 1 Oe/(MV/m) corresponds to 0.1 mT/(MV/m) in vacuum (B = μ0H).

### 3.3 Which designs can be rebuilt exactly as validation geometries

- **Complete from a publication [S]:** the **TESLA 9-cell** cavity: mid-cup plus end-cups 1 and 2, beam tubes of
  radius 39 mm (= end iris) [S, src 1]. Pipe lengths and ports are not needed for a closed check; use PEC caps at
  ≥ 3 pipe radii (the TM010 field there is about 1.5·10⁻³ of its iris value [D, §4.1]). The table values are
  rounded: the SLANS-tuned set in [C] (Req 103.353, L 57.6524, end-cup L 55.716 / 56.815, end-cup 1 A = B = 40.34)
  is at 1300.000 MHz, and our 2.5D solver gives fπ = 1.30098 GHz with the rounded values (docs/18 [R]). Targets:
  kcc 1.87 %, Epk/Eacc 2.0, Bpk/Eacc 4.26 mT/(MV/m), G 270 Ω, R/Q 1036 Ω (linac) [S/D].
- **LL and ICHIRO mid-cells [S, src 6, 7]:** complete mid-cells; their end groups were not retrieved, so use them as
  single cells or as periodic π-mode checks (LL targets: kcc 1.52 %, Epk/Eacc 2.36, Bpk/Eacc 3.61, R/Q 133.7 Ω per
  cell, G 284 Ω [S, src 4, 9]).
- **Complete in [C] only** (check against the publications before use): NLSF, NLSF-RE, Cornell ERL main linac (mid
  and both ends), FCC-ee "C3794" (mid), "C3795" (mid and end), "FCCUROS5" (mid).
- **Partial:** CEBAF OC / HG / LL (Ri, Req, kcc), 3.9 GHz (mid and end iris, Req), CEPC (Req, pipe radius [D],
  figures of merit), HB650 (Ri), KEKB (Ri), CESR-B (one pipe), LHeC (Req).
- **Not retrieved:** RE, LSF, SNS, ESS, LB650, LHC 400 MHz, the 3.9 GHz ellipses, LCLS-II (TESLA shape [U]).

## 4. Comparison with our generator

### 4.1 `elliptical`

**(a) Are the half-cell ratio ranges realistic, and what fraction of real designs falls outside?**

- **Ri/Req and L/Req: realistic for β = 1.** Every β = 1 mid cell with known Ri and Req lies inside
  0.25–0.42 (TESLA 0.339, LL 0.304, ICHIRO 0.306, NLSF 0.325, Cornell ERL 0.346, FCCUROS5 0.360, C3794 0.406,
  CEBAF 0.305–0.374, 3.9 GHz 0.419), except C3795 (0.421, marginal) [S/C/D]. L/Req at β = 1 is 0.506–0.588,
  inside 0.45–0.65. The 1.3 GHz linac cells use only the band 0.30–0.35 of Ri/Req; the upper part (0.38–0.42)
  belongs to large-aperture ring cells, 800 MHz collider cells and the 3.9 GHz cell.
- **Ellipse ratios: too narrow, and centred on TESLA.** Of the 9 complete β = 1 mid cells (rows 1, 4, 5, 7, 8, 9,
  12, 13, 15 of §3.1), only TESLA lies inside all six random ranges; 8 of 9 violate at least one (7 of 9 if the
  marginal FCCUROS5 B/A = 0.848 counts as inside). Among the three [S]-confirmed mid cells (TESLA, LL, ICHIRO),
  2 of 3 are outside. The main misses:
  - B/A < 0.85 for the whole post-TESLA low-loss / low-surface-field family (LL 0.729, ICHIRO 0.684, NLSF 0.665,
    NLSF-RE 0.720, Cornell ERL 0.797): their equator ellipse is flatter than a circle;
  - a/L < 0.15 (LL and ICHIRO 0.132) and A/L > 0.85 (LL and ICHIRO 0.867): thin iris, long equator ellipse;
  - b/a < 1 (FCC-ee 800 MHz: 0.765 mid, 0.697 end) and a/L > 0.30 (0.323) [C];
  - the FCC-ee 400 MHz cell (A/L 0.393, B/A 1.79, a/L 0.568) sits in a different corner altogether [C].

  The sample is small, biased towards 1.3 GHz ILC R&D shapes, and 6 of its 9 mid cells are [C]. Read "8 of 9" as
  "most post-TESLA designs", not as a population statistic [E].
- **β < 1 (25 % of draws).** L ∝ β is the right physics (L = βλ/4; TESLA gives L/Req = 0.559 at β = 1 [D]). I
  could not retrieve SNS, ESS or LB650 / HB650 cell geometry, so whether Ri/Req ≥ 0.25 and the ellipse ranges
  fit low-β cells is unverified (§6). These cells have clearly higher Bpk/Eacc (SNS β 0.61: 5.74 [D]; ESS β 0.67:
  4.79 [S]) and Epk/Eacc 2.2–2.72 [S/D].

**Wall angles.** Accepted mid-cell wall angles of our generator (main-session measurement, 3000 draws including
anchors): p1 61.5°, p10 68.7°, median 78.9°, p90 87.1°, p99 89.7° [R]. My run of the random branches alone: β = 1
median 75.6°, p95 87.0°; β < 1 median 83.5°, p95 89.2° [R]. Real β = 1 cells span 67.6°–89.9° without
re-entrant designs and reach 101.1° with them (§2.1). LL, ICHIRO and NLSF (θ = 89.9°) and our `ilc_ll` anchor
(θ = 90.0° exactly, because its L = A + a) sit on the rejection limit.

**Validity test and rejections.**

- About 50 % of random β = 1 half-cells are rejected by `_valid_half` (main session [R]; my run: 49.9 %, of which
  26.5 points from the non-overlap rule, 20.7 "no tangent", 2.7 re-entrant [R]).
- **Solver tolerance (new finding).** `half_cell_wall` calls `least_squares` with its default tolerances
  (xtol = ftol = gtol = 1e-8) but accepts a solution only if max |residual| ≤ 1e-9. This rejects valid walls: in
  3000 random β = 1 draws, 222 (7.4 %) came back "no tangent" although a tangent of the right type (the two
  ellipse centres on opposite sides of the line, as for TESLA) with a monotone wall exists. Passing
  xtol = ftol = gtol = 1e-15 from the same start recovers all 222 [R, `solverfix.py`]. Their wall angles
  (p5 / p50 / p95: 68.6° / 78.8° / 88.2°) are ordinary, so the loss is mostly waste with a mild bias. The same
  issue removes 8.9 % of perturbed-TESLA and 6.7 % of perturbed-`ilc_ll` draws, and two real end cells (Cornell ERL
  right end, C3795 end; rows 11 and 14) [R, `notangent3.py`].
- **The non-overlap rule Ri + b < Req − B is stricter than needed:** the FCC-ee 400 MHz cell violates it but has a
  valid monotone wall (θ = 72.8°) [R/C]. With the tolerance fix and the rule replaced by a direct check that r is
  monotone along the wall, the acceptance of the current ranges rises from 50.1 % to 75.9 % (with re-entrant walls
  up to θ = 105° allowed behind a web guard, 5.4 % of the accepted cells are re-entrant) [R, `widened.py`].
- **Anchors:** of the perturbed `ilc_ll` draws, 33 % are rejected as re-entrant and 25 % as "no tangent", 43 % are
  accepted; for TESLA, 58 % are accepted [R, `mc_generator.py`]. The accepted `ilc_ll` neighbourhood is one-sided
  (θ ≤ 90° only).

**(b) The `ilc_ll` anchor.** Published values (arXiv:0910.1305 Table 1, src 6; Req and A also in src 7; identical
in [C]):

| cell | Ri | Req | L | A | B | a | b |
|---|---|---|---|---|---|---|---|
| LL (ILC low-loss) | 30 | 98.58 | 57.7 | 50.052 | 36.50 | 7.6 | 10.0 |
| ICHIRO (KEK) | 30 | 98.14 | 57.7 | 50.052 | 34.222 | 7.6 | 9.945 |
| our `ilc_ll` | 30 | 98.6 | 57.7 | 50.1 | 34.2 | 7.6 | 10.0 |

Our anchor mixes the two designs: Req and b of LL, B of ICHIRO. LL figures of merit: iris Ø 60 mm, kcc 1.52 %,
Epk/Eacc 2.36, Bpk/Eacc 3.61 mT/(MV/m), R/Q 133.7 Ω per cell, G 284 Ω [S, src 4, 9]. Because the anchors are
perturbed by ±10 % and every label comes from the FE solve, no label is wrong; but the anchor is not the LL design,
and its comment "approximate, unverified" can now be replaced by the values above.

**End half-cells.** Our end iris of ×1.0–1.35 the mid iris covers TESLA (×1.114), Cornell ERL (×1.057 and ×1.114),
C3795 (×1.111) and the 3.9 GHz cavity (×1.333, near the limit) [D]. The ±15 % perturbation of L, A, B, a and b is
too narrow for the iris ellipse; real end cells shrink a and b much more [D, from §3.1]:

| end / mid ratio | A | B | a | b | Ri | L |
|---|---|---|---|---|---|---|
| TESLA end-cup 1 | 0.96 | 0.96 | 0.833 | 0.711 | 1.114 | 0.971 |
| TESLA end-cup 2 | 1.00 | 1.00 | 0.75 | 0.674 | 1.114 | 0.988 |
| Cornell ERL left end | 1.157 | 1.292 | 0.67 | 0.549 | 1.114 | 1.041 |
| Cornell ERL right end | 1.184 | 1.366 | 0.79 | 0.539 | 1.057 | 1.087 |
| FCC-ee C3795 end | 1.006 | 0.87 | 0.569 | 0.519 | 1.111 | 1.003 |

All five real end cells have a and b outside ±15 %, and the Cornell ERL ends also A and B. Also, 40 % of our
cavities use end half-cells identical to the mid cell. That is legitimate (TESLA end-cup 2 keeps A = B = 42 mm),
but CEPC's chain of identical half-cells produced trapped modes [S, src 31], so a fully identical chain is a
special case, not the norm [E].

**Beam pipes and caps.** Our pipes are 1.2–2.5 pipe radii long and end in PEC caps. For a TESLA-size pipe (39 mm)
the TM010 field decays into the pipe as exp(−√(kc² − k²)·z) with a 1/e length of 18.1 mm, so at the cap it is
7.5 % (1.2 rp) to 0.45 % (2.5 rp) of its iris value; for the TE111 band (≈ 1.7 GHz) the 1/e length is 32.3 mm and
the cap amplitude 23 % to 4.9 % [D]. With K = 6 stored modes, cavities with fewer than 6 cells carry dipole modes in
their labels, and those depend on the cap position far more than the fundamental does [E]. Pipe cut-offs relative
to f0 [D]: TE11 1.73 f0 and TM01 2.26 f0 for TESLA and CEPC proportions; 1.46 / 1.91 f0 for the FCC-ee 400 MHz
pipe; 1.13 / 1.47 f0 for the 3.9 GHz end pipe. In large-aperture designs part of the K = 6 spectrum lies above the
pipe cut-off, where the closed-cap labels are pipe resonances rather than cavity modes [E].

**Real features we do not model:** re-entrant walls (RE, NLSF-RE); an equator straight section (SINAP LL-type cell
[S, src 39]); the large-aperture ring-cell shape (C3794); asymmetric pipe treatments (a fluted pipe [S, src 33],
ferrite dampers [S, src 32, 33], pipe transitions such as KEKB's 220 mm iris to 150 mm ducts [S, src 32]); coaxial
antennas, HOM loops and hooks with their 1.9 mm notch gaps [S, src 1]; waveguide couplers (CEBAF OC, CESR-B
[S, src 24, 33]).

### 4.2 `elliptical_long` (OOD, 6–9 cells) — question (d)

- **Real cell counts:** 9 (TESLA / XFEL / ILC, LCLS-II [U]; 3.9 GHz [U]); 7 (C100 [S, src 24]; Cornell ERL [U]);
  6 (SNS, ESS medium β [S]); 5 (ESS high β, PIP-II, FCC-ee 800 MHz, LHeC, CEBAF OC [S]); 2 (CEPC [S]; FCC-ee
  400 MHz in some documents [S]); 1 (KEKB, CESR-B, FCC-ee 400 MHz at the Z [S]). The counts of most built linac
  cavities (6, 7, 9) are therefore only in the OOD set; the training range covers the ring and collider counts
  (1, 2, 5) and the 5-cell β < 1 structures.
- **With K = 6 stored modes** (`--n_eigen_modes` default 6 [R]) the operating π-mode of a cavity with N ≥ 7 cells
  is not among the labels: the 6 lowest modes are the passband members π/N … 6π/N [D]. For N = 6 the π-mode is
  the 6th and last mode stored. The OOD test thus never asks for the accelerating mode of 7–9-cell cavities.
- **The passband gets denser with N.** The π-mode's separation from its neighbour is f·(kcc/2)·(1 − cos(π/N)):
  2.32 MHz (1.8·10⁻³ relative) for 5 cells, 1.20 MHz for 7 and 0.73 MHz (5.6·10⁻⁴) for 9 TESLA-like cells, and
  0.60 MHz for 9 LL-like cells [D, coupled-oscillator estimate]. The field-flatness sensitivity factor N²/kcc grows
  from 13 (5 cells) to 43 (9 cells, TESLA) and 53 (9 cells, LL) [D]. Going from 5 to 9 cells is a harder test than
  the cell count alone suggests.
- **Mesh cost.** The mesh size is set from the per-cell volume (`_vol_div = n_cells` [R]), so the tet count grows
  roughly linearly with N [E]. Elliptical draws already give 9.2k, 30k, fold and fold curved tets at ≤ 5 cells
  (docs/29 [R]); a 9-cell structure at the same resolution would usually exceed 30k → costly [E].

### 4.3 `pillbox_pipes`

Not an SRF design family, but a sensible smooth baseline. For a λ/2 gap, R = 2.405c/(2πf) = 0.383λ gives
Lc/R = 1.31β, so Lc/R 0.4–1.6 spans β ≈ 0.3–1.2 [D]. rp/R 0.12–0.35 overlaps only the lower part of the real
iris ratios of elliptical cells (0.30–0.42) [D]. No change is needed; to resemble storage-ring single cells,
rp/R up to about 0.45 would be closer [E].

### 4.4 `pillbox_port` (OOD) and the coupler stubs of `elliptical` — question (e)

- **Where couplers are.** In every SRF elliptical design retrieved, couplers sit on the beam pipes, not on the
  cells (TESLA FPC and HOM couplers [S, src 1]; CEPC coaxial HOM couplers on the beam pipe [S, src 31]; KEKB and
  CESR-B dampers on the pipes [S, src 32, 33]). Radial ports on the cell body belong to normal-conducting HOM-damped
  cavities [U]. `pillbox_port` is a valid OOD symmetry-breaking test, but not representative of SRF elliptical
  cavities.
- **What our hollow stubs are.** The `elliptical` stubs are hollow tubes of 0.25–0.6 pipe radius, closed by PEC
  caps. For a TESLA-size pipe their TE11 cut-off is 2.9–6.9 f0 [D], so every stored mode is evanescent in them.
  Physically this is a port with its antenna removed (or a pick-up or pumping port): a small local perturbation
  that breaks the axial symmetry, not a coupler [E]. It is not free: the port-junction fillet folds the curved
  mesh. Before the sharp fallback, 7 of 12 ported draws were rejected; now ported samples fall back to sharp edges
  and store NaN peaks (docs/30 [R]).
- **Why not add an antenna under a closed cap.** A coax line shorted at the cap and open at the antenna tip is a
  λ/4 resonator, f ≈ c/(4l). With our stub lengths (l ≈ 1.3–2.5 pipe radii ≈ 51–98 mm at TESLA size) that is
  0.77–1.48 GHz [D/E], on top of the 1.3 GHz TM010 band: the K lowest modes would include coupler-line resonances
  that do not exist in the matched real line (TESLA FPC Qext 3·10⁶ [S]). The same holds for HOM loops and hooks,
  whose 1.9 mm notch gap [S] is far below our 5 % fillet floor.
- **If coupler physics is wanted later:** Qext from two closed eigen-solves with an electric- and a magnetic-type
  termination of the coupler line (Balleyguier), or the Kroll–Yu method [S, src 37]. Both keep our real symmetric
  problem and the Ritz upper bounds, at 2 or more label solves per sample plus the coax mesh. A port or absorbing
  boundary (a complex eigenproblem) would break our invariants, since the Ritz upper bounds need the Hermitian
  closed problem [E]; infeasible now.

## 5. Recommendations

Each item: what to change, expected benefit, cost under our constraints (fillet floor 5 %, curved mesh ≤ 30k tets,
closed PEC cavity), supporting source. Every item that changes the generator's output needs the owner's go-ahead
(AGENTS.md: "Do not change the data generator's behaviour without the owner's go-ahead").

### High priority

- **H1. Replace the `ilc_ll` anchor by the published LL mid-cell and add ICHIRO as a third anchor.**
  LL: Ri 30, Req 98.58, L 57.7, A 50.052, B 36.50, a 7.6, b 10.0 mm; ICHIRO: Ri 30, Req 98.14, L 57.7, A 50.052,
  B 34.222, a 7.6, b 9.945 mm. *Benefit:* the anchors become real designs. *Cost:* none. *Source:* src 6, 7, [C].
  Both cells have θ = 89.9°, so H2 decides how their neighbourhoods are sampled.
- **H2. Allow mildly re-entrant walls, θ ≤ 100–105° (Shemelin α ≥ 75–80°).** Replace the monotone-z test by:
  r monotone along the wall, no self-intersection, and a web guard for walls that turn back towards the iris
  plane, e.g. min z of the wall beyond the iris arc ≥ 0.05 Req (half the metal web between neighbouring cells;
  real values of L − A: LL 7.65 mm, NLSF-RE 8.7 mm, TESLA 15.7 mm [D]). *Benefit:* covers RE and NLSF-RE and
  removes the one-sided truncation around LL, ICHIRO and NLSF, all at θ = 89.9°; nothing physical happens at 90°,
  so a hard cut there creates an artificial edge exactly where the modern high-gradient shapes sit [E]. *Cost:*
  cheap — the profile stays one smooth spline with no new edges or fillets; the vacuum mesh near the thin web is
  unaffected [E]. *Sources:* RE results (src 10–14), re-entrant definition and optimisation with the wall angle
  (src 16, 17), NLSF-RE [C], ILC TDR alternative shapes (src 15).
- **H3. Fix the tangent solve and relax the non-overlap rule.** Pass xtol = ftol = gtol = 1e-15 (or keep the
  defaults and use a multi-start) in `half_cell_wall`; replace `Ri + b < Req − B` by a direct check that r is
  monotone along the wall. *Benefit:* +7.4 % valid random draws from the solver fix alone; with both changes the
  acceptance of the current ranges goes from 50.1 % to 75.9 % [R]; real cells now rejected (Cornell ERL right end,
  C3795 end, FCC-ee 400 MHz) become reachable. *Cost:* trivial; the geometry stream of a given seed changes
  (geometry ids are not reproducible across the change). *Source:* [R] scripts `solverfix.py`, `widened.py`.
- **H4. Widen the β = 1 ellipse ranges:** B/A 0.85–1.25 → 0.6–1.3; a/L 0.15–0.30 → 0.12–0.35; A/L 0.6–0.85 →
  0.6–0.9; b/a 1.0–1.9 → 0.65–1.9. *Benefit:* covers every complete β = 1 cell of §3.1 except the FCC-ee 400 MHz
  cell (see M2). *Cost:* cheap in geometry. With the H2/H3 rules the acceptance is 65.3 % (13.9 % of accepted cells
  re-entrant; θ p1 / p50 / p99 = 60.9° / 78.8° / 102.2°), against 41.8 % with today's rules [R, `widened.py`]. Add
  a guard on the iris-tip radius, a²/b ≥ 0.05 Req, to stay consistent with the fillet floor: real cells have
  0.059 Req (LL, ICHIRO) to 0.074 Req (TESLA, ERL) [D], while today's ranges already allow 0.036 Req and the
  widened ones 0.028 Req [D]; small tip radii drive the curved mesh towards the 30k limit [E]. *Sources:* §3.1 rows
  4–15 (src 6, 7, [C]).
- **H5. Make the multi-cell OOD test meaningful.** For N-cell samples store K ≥ N + 2 modes (the whole TM010
  passband plus the first dipoles), or at least label the passband index; add a small validation set of exactly
  rebuildable real structures (TESLA 9-cell with its two end-cups; LL and ICHIRO single cells; the [C] designs
  after checking them against their publications). *Benefit:* the accelerating π-mode of 7–9-cell cavities enters
  the test; validation against published kcc, Epk/Eacc, Bpk/Eacc, G and R/Q. *Cost:* medium to high — more
  eigenpairs per solve, and 9-cell meshes exceed 30k tets at today's resolution (§4.2). *Sources:* src 1, 4, 6,
  9; docs/18, docs/29 [R].
- **H6. Couplers: keep closed PEC caps; no antennas under closed caps; decide what the hollow stubs are for.**
  Either drop the stubs from `elliptical` (port physics stays in the OOD families) or keep them explicitly as a
  symmetry-breaking perturbation and say so in the docs. If Qext is wanted later, use the Balleyguier
  two-termination method. *Benefit:* no spurious coax-line modes among the K lowest; fewer folded meshes and NaN
  peaks. *Cost:* none (dropping) or 2× label solves (Balleyguier). *Sources:* src 1, 31, 37; docs/30 [R]; §4.4.

### Medium priority

- **M1. Real end groups.** For the TESLA anchor use its published end-cups 1 and 2 (asymmetric) instead of random
  ends; widen the random end-cell perturbation to a ×0.5–1.05, b ×0.5–1.05, A ×0.9–1.2, B ×0.85–1.4, keep L ±10 %
  and the end iris ×1.0–1.35. *Benefit:* realistic end cells and field flatness, which matter for the multi-cell
  modes. *Cost:* cheap (smooth profiles). *Sources:* §4.1 end/mid table (src 1, [C]); CEPC trapped modes (src 31).
- **M2. A large-aperture ring and collider sub-family:** 1–2 cells (and 5 at 800 MHz), Ri/Req 0.38–0.47,
  B/A up to 1.8, a/L up to 0.6, θ 65–90°, C3794 / C3795-like, with long pipes. *Benefit:* covers KEKB, CESR-B,
  CEPC, FCC-ee and LHC-type cavities. *Cost:* cheap geometry (needs H3), but more modes above the pipe TE11
  cut-off (1.46 f0 for the FCC-ee 400 MHz pipe [D]), whose closed-cap labels depend on the pipe length; pair with
  M3. *Sources:* [C]; src 31–35.
- **M3. Store per-sample cut-offs** (TE11 and TM01 of each beam pipe and each port) and flag modes above them.
  *Benefit:* separates cavity modes from pipe resonances in training and evaluation. *Cost:* trivial (metadata).
  *Source:* §4.1 [D].
- **M4. Verify the β < 1 ranges against SNS, ESS and PIP-II geometry** before trusting the 25 % low-β branch. The
  λ/4 scaling is right; check whether Ri/Req ≥ 0.25 is too high for β ≈ 0.6 cells and whether short cells need
  B/A > 1 [E]. *Cost:* a literature check only. *Sources to open:* src 25–27, 30.
- **M5. Sample the wall angle directly**, e.g. θ ~ U(65°, 105°), and solve for one dependent ellipse parameter
  (A or a), so that rejection does not shape the θ distribution [E]. *Cost:* cheap.
- **M6. Pipe length of validation geometries ≥ 3 pipe radii**, where the TM010 field at the cap is about 1.5·10⁻³
  of its iris value for TESLA proportions [D]. The training range 1.2–2.5 rp can stay (it is part of the geometry
  the model sees), but comparisons with published designs should not depend on the cap.

### Low priority

- **L1. Equator straight section** (a cylindrical flat of length d at Req), as in the SINAP LL-type cell
  [S, src 39]. Cheap.
- **L2. Rectangular waveguide stubs on the beam pipes** (CEBAF OC HOM waveguides, CESR-B FPC [S, src 24, 33]) as a
  more realistic symmetry breaker than round hollow stubs. Medium cost: rectangular junction edges need fillets,
  and fillets at port junctions fold the curved mesh (docs/30 [R]).
- **L3. Leave out deliberately:** HOM-coupler internals (loops, hooks, 1.9 mm notch gaps ≪ 5 % floor), ferrite
  dampers (lossy material, not PEC), fluted pipes, stiffening rings (outside the RF volume). Infeasible or
  irrelevant under our constraints [S/E].
- **L4. Per-cell Req jitter** (a fraction of a per cent) to emulate tuning and field-flatness errors in multi-cell
  samples; cheap, and useful for the 9-cell OOD test [E].

## 6. Open questions and what I could not verify

- **Method.** No full text was opened (WebFetch and direct downloads blocked). All [S] numbers come from search
  excerpts. Column alignment was explicitly doubtful for the Geng 2005 table (src 10: RE kcc 2.4 %, Hpk/Eacc 37.8)
  and the ASC 2008 table (src 9: which LL-type variant has 61 mm, 138 Ω and 285 Ω), and the CEBAF diameters
  (src 22) were given as "roughly". The [C] geometries (NLSF, NLSF-RE, Cornell ERL, FCC-ee C3794 / C3795 /
  FCCUROS5) come from an unpublished code and must be checked against their publications (Juntong & Jones; the
  Cornell ERL cavity papers; the Rostock / CERN FCC-ee cavity papers) before use as validation targets.
- **Not retrieved:** cell geometry of SNS (β 0.61 / 0.81), ESS (0.67 / 0.86), PIP-II LB650 / HB650 (only the
  HB650 aperture), the Cornell RE and LSF shapes, the CEBAF ellipse parameters, the LHC 400 MHz cavity, KEKB and
  CESR-B Req, the 3.9 GHz ellipses and figures of merit, LCLS-II's changes to the TESLA design, the published
  FCC-ee 400 / 800 MHz cells, the R/Q conventions of ESS (367 / 435 Ω) and CEPC (212.7 Ω), and the TESLA 8π/9–π
  spacing (I give only a coupled-oscillator estimate, 0.73 MHz).
- **HL-LHC:** its new 400 MHz cavities are crab cavities (DQW, RFD), not elliptical [U]. That the LHC main 400 MHz
  system uses superconducting single cells is stated in the excerpt of the storage-ring search that returned
  src 32 and 33; its geometry I did not retrieve.
- **Not measured, cheap to measure with our pipeline** (about 10 draws each): the label change caused by a hollow
  stub (f and QoI with vs without the stub); the cap-position sensitivity of the fundamental and dipole modes
  (pipe 1.2 vs 2.5 vs 4 rp); the mesh cost of re-entrant walls and of LL-like small iris-tip radii.
- **Field-flatness scaling N²/kcc** is the standard coupled-oscillator result; I did not verify it in a source in
  this session.
- **Searches that would close the gaps** (in a follow-up session with working fetch): ESS cavity design (CEA /
  INFN cell table with Ri, Req, A, B, a, b, α); SNS cavity design (Ciovati / Kneisel / Sang-ho Kim); PIP-II
  LINAC 2016 THPLR041 Table 1 and arXiv:1208.4245; Geng PAC 2005 ROAC009 RE table; Li & Adolphsen LINAC 2008 (LSF);
  the full CEBAF MOP84 table; the FCC-ee 400 / 800 MHz cavity papers; the FNAL 3.9 GHz parameter table; the LHC
  400 MHz system (LHC Project Report 316); KEKB and CESR-B cell dimensions.

### Scripts used

All standalone, in `scripts/` (the copies there carry an `a1_` prefix):

- `halfcell.py` — copy of our tangent construction, plus the wall angle;
- `ratios.py` — §3.1 ratios, wall angles and `_valid_half` outcome;
- `mc_generator.py` — acceptance and wall-angle distribution of our sampler and of the perturbed anchors;
- `notangent3.py`, `solverfix.py` — the tangent-solver tolerance finding;
- `widened.py` — acceptance and θ spread for current vs widened ranges and rules;
- `derived.py` — end/mid ratios, iris-tip radii, passband spacing, cut-offs, decay at the caps, coax λ/4 estimate.
