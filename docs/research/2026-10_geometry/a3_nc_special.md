# A3: normal-conducting cavities, accelerating structures and special cavities, compared with our generator

Area: normal-conducting (NC) single cavities and structures, and special cavities. Our families concerned:
`reentrant`, `pillbox_pipes`, `ridged_box`, `composite`, and the OOD families `box`, `junction`, `pillbox_port`
(src/data_gen/cavity_shapes.py, branch claude/field-model). Date 2026-10-22.

## 0. How to read this report (access limits, tags)

- **No full text was read.** WebFetch failed for every host (`getaddrinfo ENOTFOUND`). The agent proxy refused
  curl to jacow.org, accelconf.web.cern.ch, proceedings.jacow.org, epaper.kek.jp, arxiv.org, cds.cern.ch,
  osti.gov and journals.aps.org (HTTP 403, policy). So every number credited to a source comes from the
  passages that the WebSearch tool quoted from that source. Read each such number as "search extract".
  - "(attribution uncertain)" marks an extract that mixed several documents.
  - Scanned 1990s tables were sometimes garbled; such numbers are flagged.
  - The shared web-search budget ran out after about 45 queries. Two more were made after the reset.
- **[calc]** marks my own computation. I wrote an axisymmetric TM0 eigen-solver (NGSolve: H_phi formulation,
  H1 p = 3, curved 2D mesh). It solves the generator's own `reentrant` and `pillbox_pipes` profiles,
  reusing `cavity_shapes.build_*` and `filleted` with the 5 % fillet floor, plus three rebuilt real geometries.
  - Validation against the analytic pillbox (R = d = 5 cm): f to 3e-11, R/Q to 3e-4, G to 5e-10 relative error.
  - Mesh check (h/2.5 at p4): f within 1.3e-5, R/Q within 0.7 %.
  - Scripts and data are in the scratchpad `lit/` folder: `a3_axisym.py`, `a3_family_stats.py` (400 draws per
    family → `a3_reentrant_stats.json`, `a3_pillbox_pipes_stats.json`), `a3_scan.py`, `a3_gun.py`, `a3_slac.py`.
- **[inference]** marks a conversion or deduction that I made, not a statement of the source.
- **[code]** marks an observation from reading the repository, not from the literature.
- **R/Q conventions.** Our labels use R/Q = V²/(ωU), the "linac" convention at β = 1. Storage-ring papers often
  quote the shunt impedance as V²/(2P), the "circuit" convention, which is half the linac value.
  Every conversion is tagged.
- **Dimensionless ratios** use R = cavity (equator) radius, λ = free-space wavelength at the operating
  frequency, Lc = cell length, g = gap, rp = beam-pipe radius, a = iris radius.

## 1. Scope and sources

The list has 78 entries. More than 40 of them give numerical dimensions or figures of merit; at least 20 give
dimensions (S6, S12, S17, S29–S31, S37–S39, S44–S47, S51, S54, S56–S59, S62). In brackets: what each one gives.
Every entry was read through search extracts only (see section 0).

**Storage-ring / re-entrant cavities**

1. **S1** PEP-II RF cavity, EPAC'92 p. 1289. https://epaper.kek.jp/e92/PDF/EPAC1992_1289.PDF
   [HER table: Ks = V²/2P = 3.5 MΩ, 0.93 MV, 122 kW, Q0 ≈ 31 000 with ports at 40 °C; OCR partly garbled]
2. **S2** PEP-II cavity, EPAC'94 p. 2101. https://www.jacow.org/proceedings/e94/PDF/EPAC1994_2101.PDF
   [0.93 MV needs ~122 kW; nose pieces brazed into the bowls; frequency tuned by trimming the nose length
   (attribution between S2 and S3 uncertain)]
3. **S3** PEP-II high-power cavity, EPAC'96 WEP070L / WEP069L. https://epaper.kek.jp/e96/PAPERS/WEPL/WEP070L.PDF
   [CW, up to 150 kW wall loss; 3 rectangular waveguides (WG) + broadband loads act as high-pass filters;
   strongest HOMs damped by more than 3 orders of magnitude]
4. **S4** PEP-II HOM impedances, calculated vs bench vs beam-sampled, PRST-AB (Oct 2000).
   https://doaj.org/article/cd9fde24144b47d88424ccdcb32cbf3d (also https://www.osti.gov/biblio/887420)
   [agreement for the strongest HOMs]
5. **S5** R. Rimmer, HOM10 talk. https://www.classe.cornell.edu/Events/HOM10/rsrc/LEPP/Events/HOM10/Agenda/TA1_Rimmer.pdf
   [PEP-II typically ~850 kV and 103 kW; PEP-II and DAΦNE = Cu cavities with rectangular HOM WGs and
   broadband loads (attribution vs PAC'99 MOP103 https://jacow.org/p99/papers/MOP103.pdf uncertain)]
6. **S6** BESSY, "HOM damped 500 MHz cavity design for 3rd generation SR sources", PAC'01 MPPH033.
   https://cds.cern.ch/record/555828/files/mpph033.pdf
   [design variables: cavity length and radius, nose-cone angle, gap length, WG cutoff; beam pipe Ø 74 mm;
   WG cutoffs 650 / 670 MHz for the cylindrical / spherical candidate]
7. **S7** "First tests of a HOM-damped high power 500 MHz cavity", EPAC'04 TUPKF011.
   https://accelconf.web.cern.ch/e04/papers/tupkf011.pdf
   [cylindrical cavity with nose cones and 3 double-ridged WGs; Rsh 3.1 MΩ; longitudinal HOM ≤ 4.8 kΩ
   (from a mixed extract)]
8. **S8** "Impedance measurements of a HOM-damped low power model", EPAC'02 TUPLE084.
   https://proceedings.jacow.org/e02/PAPERS/TUPLE084.pdf
   [615 MHz WG cutoff as a compromise; homogeneous WGs could give ≈ 1.7 kΩ (attribution uncertain)]
9. **S9** "Status of the European HOM damped normal conducting cavity", EPAC'08 THXM03.
   https://proceedings.jacow.org/e08/papers/thxm03.pdf
   [3 circular double-ridge WG dampers with NiZn ferrite; HOM < 5 kΩ / 200 kΩ/m; tile brazing qualified
   to 14 W/cm²; a port–ridge gap creates TM011 impedance]
10. **S10** ESRF, "Status of HOM damped room-temperature cavities for the ESRF storage ring", EPAC'08 MOPP108.
    https://proceedings.jacow.org/e08/papers/mopp108.pdf
    [single cell without dampers optimised to Rsh 6.1 MΩ, R/Q 148.5 Ω, Q0 41 100; dampers cover
    0.4–4 GHz; residual HOMs a factor 2 below threshold]
11. **S11** V. Serrière, ESLS-RF 2008 slides. https://intranet.cells.es/ESLS-RF/2008/Presentations/5b_Serriere.pdf/
    [Al model: Q0 ≈ 48 000, Rs/Q0 ≈ 150 Ω, Rs ≈ 7.2 MΩ (Cu); < 100 W in the ferrites; damper cutoff 452 MHz
    (attribution uncertain)]
12. **S12** ESRF Upgrade "Cavity design report" D13.1.
    https://www.esrf.fr/Apache_files/Upgrade/ESRFUP-deliverables/ESRFUP_D13.1.pdf
    [352 MHz version of the 500 MHz EU design; first model beam pipe Ø 76 mm, Ø 100 mm specified]
13. **S13** Serrière et al., "352.2 MHz HOM damped NC ESRF cavity: design and fabrication", IPAC'11 MOPC004.
    https://www-linac.kek.jp/mirror/IPAC2011/papers/mopc004.pdf
    [derived from the BESSY/ALBA cavity, with modifications]
14. **S14** D'Elia, Jacob, Serrière, "ESRF-EBS 352 MHz HOM damped RF cavities", IPAC'21 MOPAB333,
    doi:10.18429/JACoW-IPAC2021-MOPAB333. https://proceedings.jacow.org/ipac2021/papers/mopab333.pdf
    [13 cavities; the smaller top absorber was not installed (no effect below 1 GHz); damper-ridge/port gaps
    removed]
15. **S15** "ESRF-EBS 352.37 MHz radio frequency system", IPAC'21 MOPAB108.
    https://proceedings.jacow.org/ipac2021/papers/mopab108.pdf
    [per cavity R/Q 145, Q0 35 700, Rs 5 MΩ, 500 kV nominal / 750 kV maximum]
16. **S16** ALBA RF, IPAC'11 MOPC045 (https://accelconf.web.cern.ch/ipac2011/papers/mopc045.pdf) and EPAC'06 TUPCH141
    (https://accelconf.web.cern.ch/e06/PAPERS/TUPCH141.PDF)
    [DAMPY = EU cavity: 500 ± 1 MHz, 0.5 m insertion, 700 kV, 80 kW, Rsh 3.3 MΩ (spec > 3.1 MΩ);
    E011 ≈ 11 kΩ in the pre-series]
17. **S17** MAX IV RF, IPAC'11 MOPC051. https://accelconf.web.cern.ch/IPAC2011/papers/mopc051.pdf
    [capacity-loaded Cu cavity: inner radius 410 mm, rod radius 60 mm, gap 50 mm for 300 kV; Rsh 3.5 MΩ
    theory (3.2 practical); capacity-loaded 3rd-harmonic cavities 5 MΩ]
18. **S18** L. Malmgren, 19th ESLS-RF (2015).
    https://www.cells.es/en/media/workshops/european-synchrotron-rf-meetings/doc_19thesls_lmalmgren.pdf/@@download/file/DOC_19thESLS_LMalmgren.pdf
    [measured Rsh 3.3–3.45 MΩ (linac definition) vs 3.52 MΩ theory; Q0 19 700–20 500 vs 20 923 theory]
19. **S19** HarmonLIP 2025 slides (Cullinan or Alves deck).
    https://indico.cells.es/event/1662/contributions/3278/attachments/1961/4373/07-FCullinan_HarmonLIP2025.pdf
    [MAX IV 100 MHz: Rs0 1.71 MΩ, Q0 20 284]
20. **S20** ALS digital LLRF, arXiv:1910.07175. https://arxiv.org/pdf/1910.07175
    [499.64 MHz, Q0 28 670, QL 167, two NC cavities, ~1.3 MV in total]
21. **S21** ALS 500 MHz cavity, PAC'89 p. 121. https://epaper.kek.jp/p89/PDF/PAC1989_0121.PDF
    [frequency very sensitive to the nose-cone dimensions; URMEL/MAFIA tuning]
22. **S22** Elettra RF cavities, PAC'89 p. 220. https://epaper.kek.jp/p89/PDF/PAC1989_0220.PDF
    [1 MV per cavity, 3.3 MV/m; at 500 MHz an ultra-relativistic beam makes a "wide gap cavity" the natural
    choice]
23. **S23** KEK ARES: PAC'97 9C006 (https://epaper.kek.jp/pac97/papers/pdf/9C006.PDF), PAC'05 TPPT010
    (https://epaper.kek.jp/p05/PAPERS/TPPT010.PDF), SAST2003 2CA-2
    (https://ibic12.kek.jp/mirror/www.pasj.jp/web_publish/sast2003/2CA-2.pdf), KEK MAC 2014
    (https://www-kekb.kek.jp/MAC/2014/Report/Abe.pdf)
    [three-cavity π/2 system (accelerating + coupling + TE013 storage cavity); accelerating cavity with 4 WGs
    and grooved beam pipes, SiC loads; 508.887 MHz; damping-ring variant Rsh/Q0 150 Ω, Q0 ~30 000;
    ~95 kW HOM power per cavity at 9.4 A]
24. **S24** DAΦNE cavity: EPAC'96 WEP032L (https://accelconf.web.cern.ch/e96/PAPERS/WEPL/WEP032L.PDF) and PAC'93
    p. 778 (https://epaper.kek.jp/p93/PDF/PAC1993_0778.PDF)
    [368.26 MHz NC single cell, rounded, no nose cones, long tapers; 5 WGs (3 on the cell, 2 on the
    tapers); ~250 kV per ring]
25. **S25** E. Jensen, "RF cavity design", CAS Darmstadt 2009 slides.
    https://pdfs.semanticscholar.org/6235/268345920c0ebdebc0608892e137da145652.pdf
    [KEK PF 500 MHz, optimised vs original: R/Q 111 vs 107.5 Ω, Q 44 270 vs 41 630; nose cones raise T,
    a round outer shape minimises losses]
26. **S26** E. Jensen, "RF Cavity Design", arXiv:1601.05230. https://arxiv.org/pdf/1601.05230
    [TM010 is the fundamental for ℓ < 2.03 R; the beam pipe is a waveguide below cutoff; rounding against
    field enhancement; nose cone and round profile come out of R/Q optimisation]

**Klystrons**

27. **S27** PEP-II 476 MHz klystron, PAC'93 p. 1259. https://accelconf.web.cern.ch/p93/PDF/PAC1993_1259.PDF
    [cavity R/Q 100–108 Ω, transit angles 0.95–1.14 rad; cavity 1: 476.3 MHz, R/Q 108 Ω, Qext 1000]
28. **S28** High-efficiency klystron design, IPAC'16 (MOPMY014 or MOPMY027; the extract does not say which).
    https://accelconf.web.cern.ch/ipac2016/papers/mopmy014.pdf
    [large gap for the gain cavities, small gap for the output cavity; normalised drift-tube radius 0.67 and
    beam radius 0.43 in a SLAC design]

**Proton-linac nose-cone cells**

29. **S29** LIBO 3 GHz proton linac booster, arXiv:0712.0830. https://arxiv.org/pdf/0712.0830
    [2998 MHz side-coupled linac (SCL); nose cones added to a pillbox reduce the gap and raise T;
    nose-angle optimum 20°; web 4 mm; 16 MV/m; measured shunt impedance within 3 % of the computed value]
30. **S30** ACLIP 3 GHz SCL. https://proceedings.jacow.org/c07/PAPERS/172.pdf [cell aperture radius 4 mm]
31. **S31** Paramonov, annular coupled structure (ACS) optimisation, arXiv:1306.6287. https://arxiv.org/pdf/1306.6287
    [30° cone angle as a compromise between shunt impedance and heat removal; the 972 MHz ACS keeps the
    transverse dimensions of the L-band ACS]
32. **S32** XiPAF nose-cone buncher optimisation, LINAC'18 THPO042.
    https://proceedings.jacow.org/linac2018/papers/thpo042.pdf
    [multi-objective optimisation over cavity radius, gap length and cone angle at fixed frequency]
33. **S33** "Accelerating structures" (CAS), arXiv:1804.08525. https://arxiv.org/pdf/1804.08525
    [3 GHz, β = 0.4: 1 mm smaller bore radius gives +10 % ZT²; the septum thickness acts similarly]
34. **S34** Linac4 PIMS design, PAC'09 FR5REP051. https://proceedings.jacow.org/PAC2009/papers/fr5rep051.pdf
    [352.2 MHz, 12 cavities × 7 cells, ZTT 23.1–26 MΩ/m, k 5.6–5.0 %, Es 1.8 Kilpatrick, 3.9 MV/m]
35. **S35** Linac4 design report CERN-2020-006. https://inspirehep.net/files/69f25ed1fd4f6ab625ef53a83e6cd1a4
    [the PIMS gap length tunes the frequency at constant cell inner radius; a scaled LEP NC structure]
36. **S36** Billen & Young, "Poisson/Superfish on PC compatibles", PAC'93 p. 790.
    https://proceedings.jacow.org/p93/PDF/PAC1993_0790.PDF
    [CCLFISH tunes cell diameter, septum, gap or cone angle; symmetric half cells]
37. **S37** SCL design study for an ADS (700 MHz). https://www.osti.gov/etdeweb/biblio/20146433
    [bore radii 2.00, 2.25, 2.50 cm, SUPERFISH-optimised]

**RF guns**

38. **S38** LCLS-TN-04-4, "Design considerations for the LCLS RF gun".
    https://www.slac.stanford.edu/pubs/slactns/tn04/slac-tn-05-019.pdf
    [BNL/SLAC/UCLA prototype dimensions; π 2856.0 MHz, 0-mode 2852.5 MHz]
39. **S39** LCLS-TN-05-3, "RF design of the LCLS gun". https://www-ssrl.slac.stanford.edu/lcls/technotes/lcls-tn-05-3.pdf
    [racetrack full cell, half-cell dimensions, coupling-slot width; Δf 15 MHz; 0-mode at the cathode < 3 %]
40. **S40** "Overview of the RF systems for LCLS", PAC'05 WPAT043. https://proceedings.jacow.org/p05/PAPERS/WPAT043.PDF
    [gun: 49 MΩ/m, Q0 12 000, QL 4000, 120 MV/m]
41. **S41** LCLS gun RF measurements, PAC'07 TUPMS047. https://accelconf.web.cern.ch/p07/PAPERS/TUPMS047.PDF
    [Q0 13 960 design / 13 900 measured; Δf 15 / 15.17 MHz]
42. **S42** "The development of the LCLS RF gun", arXiv:1503.05877. https://arxiv.org/pdf/1503.05877
    [cathode field 120 MV/m (Gun III) / 140 MV/m (LCLS); racetrack = two circles offset by 5.9 mm]
43. **S43** PITZ L-band gun, arXiv:1803.05540. https://arxiv.org/pdf/1803.05540
    [1.3 GHz π mode, coaxial coupler + doorknob; ~60 MV/m at the cathode, ~6.5 MW]

**Disk-loaded structures**

44. **S44** "The Stanford Two-Mile Accelerator", ch. 6. https://www.slac.stanford.edu/spires/hep/HEPPDF/twomile/Chapter_6.pdf
    [2856 MHz, 2π/3, constant gradient by tapering; disk thickness 0.230 in; 53 MΩ/m in a table]
45. **S45** SLAC/AP-26 (https://osti.gov/servlets/purl/6140649) or SLAC/AP-86
    (https://www.slac.stanford.edu/pubs/apnotes/slac-ap-086.pdf); the extract does not say which.
    [constant-gradient stack tests, "cavity 79": 2a = 1.99 cm, 2b = 8.192 cm]
46. **S46** SuperB progress reports, arXiv:1009.6178. https://arxiv.org/pdf/1009.6178
    [S-band 2π/3 constant-gradient section: period 3.499 cm, 86 cells]
47. **S47** Grudiev & Wuensch, CLIC CDR main-linac structure, LINAC'10 MOP068 (https://jacow.org/LINAC2010/papers/MOP068.pdf;
    extract via https://www.researchgate.net/publication/47549721)
    [CLIC_G: <a>/λ 0.11, iris radius 3.15 / 2.35 mm, iris thickness 1.67 / 1.00 mm, Q 5536 / 5738,
    R' 81 / 103 MΩ/m, Es,max 230 MV/m, 26 regular cells]
48. **S48** Grudiev et al., LINAC'08 THP062. https://proceedings.jacow.org/LINAC08/papers/thp062.pdf
    [100 MV/m, 27.7 % RF-to-beam efficiency]

**Choke-mode cavities**

49. **S49** T. Shintake, KEK preprint 92-66. https://lss.fnal.gov/archive/other1/kek-preprint-92-66.pdf
    [the choke is a notch filter that traps the accelerating mode; HOMs propagate to the loads]
50. **S50** High-power test of a choke-mode structure, LINAC'94 MO-91. https://proceedings.jacow.org/l94/papers/mo-91.pdf
    [50 MV/m at S-band]
51. **S51** Zha et al., CLIC choke-mode structure, IPAC'12 TUPPR015. https://epaper.kek.jp/IPAC2012/papers/tuppr015.pdf
    [24 cells, iris radius 2.35 / 3.15 mm, radial-line and choke gap 1 mm; radial choke because a 2π/3 cell is
    too short for a coaxial one; same extract: IPAC'17 single cells reached 75 MV/m, limited by choke
    breakdown]
52. **S52** Choke-mode RF optimisation, IPAC'13 TUPME026. https://proceedings.jacow.org/IPAC2013/papers/tupme026.pdf
    [model: radial line + choke → coaxial line used as a waveguide port; absorber in the outer radial line]
53. **S53** PRAB 22, 031001 (2019), doi:10.1103/PhysRevAccelBeams.22.031001
    [high field and small choke gap → breakdown in the choke (from a mixed extract)]

**Crab / deflecting cavities**

54. **S54** Verdú-Andrés et al., DQW prototypes for HL-LHC, PRAB 21, 082002 (2018),
    doi:10.1103/PhysRevAccelBeams.21.082002; arXiv:1805.08123 https://arxiv.org/pdf/1805.08123
    [400.79 MHz, R_T/Q 429.3 Ω, G 87 Ω, Ep 37.6 MV/m and Bp 73 mT at 3.4 MV, first HOM 567–570 MHz,
    envelope dimensions]
55. **S55** Xiao et al., compact SC DQW crab cavity, arXiv:1502.04088. https://arxiv.org/pdf/1502.04088
    [two quarter-wave resonators sharing a load capacitor; PoP R/Q 406 Ω, G 85 Ω (differs from S54)]
56. **S56** De Silva et al., RFD prototypes for HL-LHC AUP, SRF'23 TUPTB046 / arXiv:2307.06943.
    https://proceedings.jacow.org/srf2023/papers/tuptb046.pdf
    [two ridged poles in a square tank, TE11-like mode; 281 mm; 3.34 MV, Ep 35.0 MV/m, Bp 55.1 mT,
    R_T ≈ 431 Ω; tuned by length and outer rounding; pole length and end gap optimised; dipole/acc. HOMs at
    636 / 699 MHz]
57. **S57** De Silva & Delayen, PoP rf-dipole cryogenic test, PRSTAB 16, 082001 (2013).
    https://journals.aps.org/prab/abstract/10.1103/PhysRevSTAB.16.082001
    [400 MHz PoP: diameter 339.9 mm, length 542.4 mm]
58. **S58** Delayen & De Silva, FPC and hi-pass-filter HOM coupler for the RFD.
    https://inspirehep.net/files/6a9bf98e5fc2c42207a2388b0311e6a7
    [rounded-square tank 281 × 281 mm, iris aperture Ø 84 mm]
59. **S59** De Silva, Delayen, Rimmer, ILC RFD update (July 2023).
    https://agenda.linearcollider.org/event/10110/sessions/5518/attachments/39011/61466/ILC%20RFD%20CC%20Update%20Jul%2019%202023.pdf
    [1.3 GHz: pole separation 25 mm, aperture 50 mm, pole length 85 mm, pole height 31.5 mm, pole angle 22.5°]
60. **S60** De Silva & Delayen, "Design evolution and properties of superconducting parallel-bar rf-dipole
    deflecting and crabbing cavities", PRSTAB 16, 012004 (2013).
    https://doaj.org/article/439ff5c948fd481eb22fbce3319b3df7 (OSTI https://www.osti.gov/biblio/1061472)
    [abstract and talk only: started from straight cylindrical loading elements in a rectangular outer
    conductor; trapezoidal elements + cylindrical conductor give balanced low peak fields and high R_T;
    fundamental-mode separation 18 → 130 MHz, Bp −20 %]
61. **S61** KEKB crab cavities: arXiv:0706.3248 (https://arxiv.org/pdf/0706.3248) and SRF'07 MO405
    (https://accelconf.web.cern.ch/srf2007/PAPERS/MO405.pdf)
    [508.9 MHz, 1.41 / 1.44 MV; the squashed (racetrack) cell pushes the unwanted TM110 polarisation to
    ~700 MHz; a coaxial coupler extracts the ~430 MHz TM010; Esp/Vkick 14.4 (MV/m)/MV]
62. **S62** Lancaster 4-rod crab cavity: IPAC'12 WEPPC032 (https://accelconf.web.cern.ch/IPAC2012/papers/weppc032.pdf)
    and NAPAC'16 talk (https://accelconf.web.cern.ch/napac2016/talks/web2io02_talk.pdf)
    [two pairs of opposing λ/4 rods; transverse radius < 142 mm; [R/Q]_T 915 Ω, G 62.8 Ω, Ep* 4.0 MV/m
    and Bp* 7.56 mT per MV; at 3 MV Ep 27.7 MV/m, Bp 63.9 mT]
63. **S63** KEK cERL deflector, IPAC'21 MOPAB330. https://epaper.kek.jp/ipac2021/papers/mopab330.pdf
    [NC single-cell TM110 pillbox, no nose cone, 5.2 GHz]
64. **S64** Matsuba, Honda, Miyajima, 2.6 GHz single-cell deflecting cavity, IPAC'10 MOPE002.
    https://accelconf.web.cern.ch/IPAC10/papers/mope002.pdf
    [NC 2.6 GHz deflector; "rectangular, Q 15 300, slightly longer vertically" came from an extract not
    pinned to this paper]

**Ports and boundary conditions**

65. **S65** Kroll & Yu, external Q of waveguide-loaded cavities, SLAC-PUB-5171 (Part. Accel. 1990).
    https://www.slac.stanford.edu/pubs/slacpubs/5000/slac-pub-5171.pdf
    [eigen-solves with the WG shorted at several positions, fitted for Qext and the loaded frequency;
    Kroll–Lin SLAC-PUB-5296 halves the runs]
66. **S66** Balleyguier, Part. Accel. 57, 113 (1997) (cited in the extract) and APT coupler paper LINAC'98 MO4037.
    https://epaper.kek.jp/l98/PAPERS/MO4037.PDF
    [single run with electric/magnetic termination of the coupler line; < 15 % error for Qext 6e6 … 12]
67. **S67** Omega3P documentation (SLAC ACE3P). https://confluence.slac.stanford.edu/spaces/AdvComp/pages/59146985/Omega3P
    [waveguide / absorbing port BCs → complex eigenvalues; nonlinear problem if the port cutoffs differ]
68. **S68** TESLA beam-pipe modes, simulation vs measurement. https://inspirehep.net/files/f7fdb230efdcaa2068e0903054dec596
    [beam-pipe modes at 2.288 GHz with magnetic-type ends vs a dipole at 2.169 GHz with electric ends]
69. **S69** Third-harmonic cavity eigenmode compendium, arXiv:1204.0391. https://arxiv.org/pdf/1204.0391
    [higher-band modes propagate out; radiation BCs more appropriate]
70. **S70** Trapped modes in vacuum chambers, arXiv:acc-phys/9504002. https://arxiv.org/pdf/acc-phys/9504002
    [a trapped mode sits slightly below the pipe cutoff; quasi-trapped modes above it couple weakly
    (attribution vs PRAB 20, 111005 uncertain)]

**Shape families and ML**

71. **S71** "Neural-network inverse design of SRF cavities and transmons", arXiv:2607.02289. https://arxiv.org/pdf/2607.02289
    [spline profiles sampled uniformly around an initial curve, 5–6 DOF per cell, COMSOL eigenmodes; designs
    recovered within 5 %]
72. **S72** Multi-objective shape optimisation of RF cavities (evolutionary), arXiv:1810.02990. https://arxiv.org/pdf/1810.02990
    [axisymmetric parametrised profiles]
73. **S73** Constrained multi-objective SRF shape optimisation with robustness, arXiv:1905.13693.
    https://arxiv.org/pdf/1905.13693 [circles instead of ellipses, to cut the number of variables]
74. **S74** Cornell ERL cavity optimisation, PAC'07. https://www.classe.cornell.edu/rsrc/Home/Research/ERL/ErlPubs2007/PAC076.pdf
    [freeing the wall angle led to re-entrant shapes]
75. **S75** Christensen et al., Nanophotonics 9, 4183 (2020), doi:10.1515/nanoph-2020-0197.
    https://dspace.mit.edu/handle/1721.1/132457 [20 000 synthetic 2D photonic-crystal unit cells with band
    structures; generation method not confirmed]
76. **S76** Statistics of eigenfrequencies of 3D microwave cavities, PRE 52, 1146 (1995).
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.52.1146
    [abstract: irregular 3D cavities follow Wigner statistics]
77. **S77** "A ridged circular waveguide ferrite load for cavity HOM damping", EPAC'06 TUPCH114.
    https://proceedings.jacow.org/e06/PAPERS/TUPCH114.pdf [title only: ridged circular WG loads]
78. **S78** Cornell SRF meeting slides 2007-11-20. https://www.classe.cornell.edu/public/SRF/2007/SRF071120-03/SRF071120-03.pdf
    [the replaced DORIS cavities had very low HOM Qext, 300–10 000]

## 2. Cavity types in the area

For each type: canonical geometry, how designers parametrise it, dimensionless ranges, typical frequency and
figures of merit (FoM). λ values used below: 100 MHz → 2998 mm; 352.37 MHz → 851 mm; 400.79 MHz → 748 mm;
476 MHz → 630 mm; 500 MHz → 600 mm; 1.3 GHz → 231 mm; 2856 MHz → 105.0 mm; 2998 MHz → 100.0 mm;
11.994 GHz → 25.0 mm.

### 2.1 Re-entrant (nose-cone) single cells for β = 1 storage rings

**Geometry**
- A pillbox-like body, either cylindrical or a rounded/"spherical" outer wall (S6 optimised both).
- Two nose cones around the beam pipe, with a gap g between the nose tips; beam pipes, often tapered to the
  vacuum chamber.
- Ports: input coupler (loop or waveguide), plunger tuner, pick-up and, in damped versions, 3–4 HOM waveguides.

**Examples**
- PEP-II, 476 MHz (S1–S5).
- EU/BESSY/ALBA "DAMPY", 500 MHz (S6–S9, S16).
- ESRF-EBS, 352 MHz (S10–S15).
- ALS (S20, S21), Elettra (S22) and KEK PF (S25), all 500 MHz.
- KEK ARES accelerating cavity, 508.9 MHz (S23).
- DAΦNE, 368 MHz (S24), is a rounded cell **without** nose cones.

**How designers parametrise it**
- BESSY: "length and radius of the cavity, nose-cone angle, accelerating gap length, waveguide
  cutoff-frequency" (S6).
- CCLFISH: cell diameter, septum, gap, cone angle (S36).
- The XiPAF buncher: radius, gap length, cone angle at fixed frequency (S32).
- The usual procedure (S29): start from a pillbox, add nose cones to shorten the gap and raise T, fix the gap
  and nose geometry by maximising r/Q, then refine the outer shape. A round outer wall lowers the losses (S25).
- The frequency is very sensitive to the nose dimensions; the ALS cavity is tuned by them (S21) and PEP-II by
  trimming the nose length (S2).

**Dimensionless values found**

| quantity | value | source |
|---|---|---|
| beam-pipe radius, EU 500 MHz | rp/λ = 0.062 (Ø 74 mm) | S6 |
| beam-pipe radius, ESRF 352 MHz | rp/λ = 0.045 (first model) → 0.059 (specified) | S12 |
| nose / cone angle | ~20° optimum (LIBO, 3 GHz); 30° compromise (ACS) | S29, S31 |
| gap, Elettra | "wide gap" at 500 MHz | S22 (no number) |

- The angle convention (from the axis or from the radial direction) is not given in the extracts.
- No extract gave gap, cell length or nose radius for a storage-ring cavity. The ratios in section 4.1 come from
  my axisymmetric scan [calc].

**Figures of merit**

| cavity | as published | linac R/Q [inference] | G from Q0, ideal Cu [inference] |
|---|---|---|---|
| PEP-II | Ks = V²/2P = 3.5 MΩ, Q0 ≈ 31 000 (S1) | 226 Ω (113 Ω circuit) | 176 Ω |
| EU undamped optimum cell | R/Q 148.5 Ω, Q0 41 100 (S10) | ≈ 297 Ω, if circuit | 240 Ω |
| ESRF-EBS | R/Q 145 Ω, Q0 35 700, Rs 5 MΩ (S15) | ≈ 290 Ω, if circuit | 175 Ω |
| KEK PF | R/Q 111 Ω, Q 44 270 (S25) | 223 Ω, if circuit | 258 Ω |
| ALS | Q0 28 670 (S20) | not given | 167 Ω |

- Voltage 0.25–1 MV per cavity; wall loss 80–150 kW (S1, S3, S16, S22, S24).
- Why the EU, ESRF and KEK PF values are probably circuit:
  - 111 Ω matches the analytic optimum of a pipe-less pillbox in the circuit convention: ≈ 111.5 Ω, half of the
    linac optimum of ≈ 223 Ω, reached at d ≈ 0.97 R [calc: 222.8 Ω at d = R].
  - ALBA's 3.3 MΩ at 700 kV and 80 kW matches V²/2P = 3.06 MΩ (S16).
  - An optimised nose-cone cell below a pipe-less pillbox in R/Q would be implausible.
- The damped EU cavities quote Rsh 3.1–3.3 MΩ (S7, S16), against 6.1 MΩ for the undamped optimised cell (S10).
  The sources do not say whether this factor ≈ 2 is a damping penalty or a change of convention (section 6).

### 2.2 Capacity-loaded low-frequency cavities

MAX IV, 100 MHz main cavities (S17–S19): a re-entrant cavity with a long central "rod" (nose) on each side.

| quantity | value |
|---|---|
| inner radius | 410 mm → R/λ = 0.137, against 0.383 for a pillbox |
| rod radius | 60 mm → 0.146 R |
| gap | 50 mm → g/R = 0.122, g/λ = 0.017 |
| shunt impedance | 3.52 MΩ (linac, theory); 3.3–3.45 MΩ measured |
| Q0 | 20 923 theory; 19 700–20 500 measured |
| derived [inference] | linac R/Q ≈ 168 Ω, G ≈ 55 Ω |
| cavity length, pipe radius, nose shape | not in the extracts |

- S19 lists 1.71 MΩ, which is half of the measured 3.42 MΩ, i.e. the circuit convention [inference].
- The 3rd-harmonic Landau cavities are also capacity-loaded (5 MΩ, S17).

### 2.3 Klystron and IOT cavities

- Re-entrant cavities with a narrow gap around a drift tube; the beam is not relativistic.
- PEP-II 476 MHz klystron: cavity R/Q 100–108 Ω, transit angles θ = 0.95–1.14 rad, Qext = 1000 (S27).
  The R/Q convention is not stated.
- A SLAC design used a normalised drift-tube radius of 0.67 and a beam radius of 0.43. Gain cavities use large
  gaps; the output cavity uses a small gap (S28).
- [inference] g/λ = θ βe/(2π) ≈ 0.15–0.18 βe, i.e. 0.08–0.13 for βe = 0.5–0.7. The normalised drift-tube
  radius 0.67 gives a/λ = 0.67 βe/(2π) ≈ 0.05–0.07. βe is not in the extracts.

### 2.4 Proton-linac nose-cone cells (CCL/SCL, ACS, PIMS)

- Cell length βλ/2 with nose cones. Coupling is through side cells (SCL), annular cells (ACS) or on-axis slots
  (PIMS).

| quantity | value | source |
|---|---|---|
| bore radius at 3 GHz | 4 mm (a/λ = 0.040) | S30 |
| bore radius at 700 MHz | 20–25 mm (a/λ = 0.047–0.058) | S37 |
| web (septum) at 3 GHz | 4 mm | S29 |
| cone angle | 20° (LIBO); 30° (ACS) | S29, S31 |
| bore sensitivity | 1 mm less bore → +10 % ZT² (3 GHz, β = 0.4) | S33 |
| Linac4 PIMS | 7 cells, ZTT 23.1–26 MΩ/m, k 5.0–5.6 %, Es 1.8 Kilpatrick | S34 |

- Linac4 tunes the PIMS with the gap length at constant inner radius (S35).
- These cells are β < 1, so their T and R/Q are defined at β < 1.

### 2.5 HOM-damped NC cavities

How the HOMs are damped:
- **On-cell waveguides.**
  - PEP-II: 3 rectangular WGs with broadband loads; strongest HOMs damped by more than 3 orders (S3).
  - DAΦNE: 5 WGs, 3 on the cell and 2 on the tapers (S24).
  - KEK ARES: 4 WGs plus grooved beam pipes and SiC (S23).
- **Circular double-ridge WGs with NiZn ferrite.** EU, BESSY, ALBA and ESRF (S6–S11).
  - Cutoff 615–670 MHz at 500 MHz, i.e. fc/f0 = 1.23–1.34 (S6, S8).
  - ESRF damper cutoff 452 MHz at 352 MHz, fc/f0 = 1.28 (S11, attribution uncertain).
  - Ridged circular ferrite loads also exist (S77).
- **Long tapers** lower the HOM R/Q at the cost of fundamental shunt impedance (S24).

Results and side effects:
- Residual impedances: longitudinal ≤ 4.8–5 kΩ, transverse ≤ 200 kΩ/m (S7, S9). The ALBA pre-series had an
  E011 mode at ≈ 11 kΩ (S16).
- The gaps between damper ridges and port sleeves caused a TM011 impedance and heating (S9, S14).
- [inference] The waveguide size follows from fc/f0. A rectangular TE10 guide with fc/f0 = 1.23–1.34 is
  w = 0.37–0.41 λ wide, about 1.0–1.3 R for a nose-cone cell. Hence the ridged guides of the EU design.

### 2.6 Choke-mode cavities

- A radial line opens in the cell wall. Shintake (1992): a λ/4 "choke" in it acts as a notch filter that traps
  the accelerating mode, while HOMs propagate radially into SiC loads (S49).
- S-band high-power test: 50 MV/m (S50). C-band (5712 MHz) choke structures are used at SPring-8/SACLA
  (secondary extract, S53).
- CLIC variant (S51):
  - radial-line and choke gap 1 mm (g/λ = 0.040);
  - iris radius 2.35–3.15 mm, 24 cells;
  - a radial choke, because a 2π/3 cell is too short for a coaxial choke;
  - single cells reached 75 MV/m, limited by breakdown in the choke.
- Design model (S52): radial line + choke → coaxial port, absorber at the outer end.
- [inference] With a cell radius of ~10 mm (not in the extract), the 1 mm gap is ≈ 0.1 R.

### 2.7 RF photo-injector guns (1.6-cell)

- A half cell closed by the flat cathode wall, a full cell, and an exit iris with the beam pipe. Coupling is by
  side waveguides (LCLS, dual feed with a racetrack full cell) or coaxial (PITZ).

| design | absolute dimensions | ratios |
|---|---|---|
| BNL/SLAC/UCLA prototype, 2856 MHz (S38) | full cell L 32.5 mm, D 83.5 mm; half cell L 22.8 mm, D 83.0 mm; iris D 25.0 mm with 9.5 mm radius of curvature | L_half/L_full = 0.70; L_full/R = 0.78; L_half/R = 0.55; a/R = 0.30; ρ/R = 0.23; R/λ = 0.40 |
| LCLS gun (S39) | half-cell radius 42.15 mm, length 24.26 mm; racetrack full cell arc radius 41.43 mm, arc separation 2.8 mm, length 34.01 mm; coupling slot 13.2 mm | L_half/L_full = 0.713; L_full/R = 0.82; separation/R = 0.068 |

- S42 describes the racetrack as two circles offset by 5.9 mm (0.14 R). This may be a different definition from
  the 2.8 mm arc separation of S39.

**Figures of merit**

| quantity | value | source |
|---|---|---|
| 0–π separation | 3.5 MHz (prototype) → 15 / 15.17 MHz (LCLS) | S38, S39, S41 |
| relative separation | Δf/f = 1.2e-3 – 5.3e-3 | [inference] |
| Q0 | 12 000 (S40) or 13 960 design / 13 900 measured (S41) | S40, S41 |
| G from Q0 | 195 Ω | [inference] |
| shunt impedance | 49 MΩ/m | S40 |
| cathode / accelerating field | 120–140 MV/m (S-band); ~60 MV/m (PITZ L-band, 6.5 MW) | S40, S42, S43 |

### 2.8 Disk-loaded travelling-wave (and standing-wave) cells

- Periodic cells: iris radius a, cell radius b, period d (d = λ/3 at 2π/3), disk thickness t, iris edge round or
  elliptical.

| structure | dimensions | ratios | FoM |
|---|---|---|---|
| SLAC, 2856 MHz 2π/3 (S44–S46) | t = 5.84 mm, d = 34.99 mm; "cavity 79": a = 9.95 mm, b = 40.96 mm | a/λ 0.095, b/λ 0.390, t/λ 0.056; a/b 0.243, t/b 0.143, d/b 0.854 | 53 MΩ/m (S44) |
| CLIC_G, 11.994 GHz (S47, S48) | a 3.15 → 2.35 mm, t 1.67 → 1.00 mm, 26 cells | a/λ 0.126 → 0.094, t/λ 0.067 → 0.040, <a>/λ 0.11 | Q 5536 / 5738; R' 81 / 103 MΩ/m; Es,max 230 MV/m; 100 MV/m |

- CLIC_G: b is not in the extract; G from Q ≈ 158–164 Ω [inference].
- [calc] Three SLAC periods closed by electric walls at the cell mid-planes reproduce the 2π/3 mode at
  2855.2 MHz (published 2856 MHz; iris assumed full-round). The whole passband is 2837.5–2861.1 MHz.

### 2.9 Coupled-cell structures

- **Side-coupled** (SNS CCL 805 MHz, LIBO 2998 MHz). Scaling the SNS coupling slots gives k_c = 4.32 % (extract
  of an SCS comparison; attribution uncertain).
- **Annular-ring coupled** (J-PARC ACS, 972 MHz, adapted from a 1296 MHz design, S31).
- **On-axis-coupled PIMS** (Linac4, S34, S35).
- All of them need coupling slots and a second cell type (non-axisymmetric, small features).

### 2.10 Crab and deflecting cavities

These quote the **transverse** R_T/Q = Vt²/(ωU), not the longitudinal R/Q of our labels.

**DQW (BNL/CERN)** (S54, S55)
- Two quarter-wave resonators sharing a load capacitor; the beam crosses the capacitive gap between the two
  posts.
- 400.79 MHz; R_T/Q 429.3 Ω (S54) or 406 Ω for the PoP (S55); G 87 / 85 Ω.
- At 3.4 MV: Ep 37.6 MV/m, Bp 73 mT. First HOM 567–570 MHz (1.42 f0).
- Envelope: 352 × 288 × 286 mm (0.47 / 0.39 / 0.38 λ) in one source; flange-to-flange 659.6 mm, port-to-port
  495.7 mm, width 329.6 mm in another (S54).

**RFD, rf-dipole (ODU/JLab)** (S56–S60)
- Two trapezoidal "ridged" poles in a rounded-square or cylindrical tank; the operating mode is TE11-like.
- LHC: tank 281 mm (0.376 λ), aperture Ø 84 mm (0.112 λ; 0.30 of the tank).
- At Vt = 3.34 MV: Ep 35.0 MV/m, Bp 55.1 mT, R_T ≈ 430 Ω.
- Lowest dipole HOM 636 MHz (1.59 f0); lowest accelerating HOM 699 MHz (1.74 f0).
- Tuning by cavity length and outer rounding radius. Pole length and end gap are optimised; pole corners are
  rounded; a curved gap compensates the sextupole.
- PoP: Ø 339.9 mm (0.454 λ), length 542.4 mm (0.724 λ), L/D = 1.60.
- ILC 1.3 GHz: pole separation 25 mm (0.108 λ), aperture 50 mm (0.217 λ), pole length 85 mm (0.369 λ), pole
  height 31.5 mm (0.137 λ), pole angle 22.5°.
- History (S60): the parallel-bar ancestor had straight cylindrical bars in a rectangular outer conductor.

**KEKB crab** (S61; superconducting, but the shape is what matters here)
- Squashed (racetrack) cell. The unwanted TM110 polarisation sits at ~700 MHz (1.38 f0), above the cutoff of
  the large pipe.
- The TM010 LOM at ~430 MHz (0.845 f0) is extracted by a coaxial coupler.
- Vkick 1.41 / 1.44 MV; Esp/Vkick = 14.4 (MV/m)/MV.

**4-rod (Lancaster)** (S62)
- Two pairs of longitudinally opposing λ/4 rods; the frequency is set by the rod length, not by the radius.
- Transverse radius < 142 mm (0.19 λ).
- [R/Q]_T 915 Ω, G 62.8 Ω; Ep* 4.0 MV/m and Bp* 7.56 mT per MV.

**NC TM110 deflectors**
- KEK cERL, 5.2 GHz: single-cell pillbox without nose cone (S63).
- KEK 2.6 GHz single cell (S64).
- The dipole polarisations are split by a geometric asymmetry (squash), as in S61.

### 2.11 Rectangular and ridged cavities

- **Ridged geometry** appears in three places:
  - the RFD / parallel-bar family (S56, S60);
  - HOM damper waveguides: circular double-ridge guides of the EU cavity (S9) and ridged circular ferrite loads
    (S77);
  - H-mode (IH) structures and the 4-vane RFQ. **Not verified** in this survey: that search failed.
- **Rectangular NC cavities** appear mainly as test and analytic cases, and possibly as deflectors (S64, shape
  not confirmed).
- The analytic box spectrum makes our `box` a solver and model test, not a design family.

### 2.12 Ports and waveguides in closed eigenmode studies (question e)

**What designers do**
- **Shorted or open port planes.** The standard practice is an eigen-solve with the port plane terminated by an
  electric wall (PEC, "short") or a magnetic wall (PMC, "open"). Port loading is then recovered by:
  - Kroll–Yu (S65): several short positions, fitted to an analytic model; less accurate at high Qext.
  - Balleyguier (S66): one run with electric/magnetic terminations of the coupler line; < 15 % error for
    Qext = 6e6 … 12.
- **Complex eigen-solves** with waveguide or absorbing port BCs, as in Omega3P (S67). They give frequency and
  Qext together; the problem becomes nonlinear when the port cutoffs differ.
- **Radiation BCs for propagating modes.** For modes above the pipe cutoff, closed-wall results depend on the
  termination: TESLA beam-pipe modes at 2.288 GHz with one termination vs a dipole at 2.169 GHz with the other
  (S68). Radiation BCs are the appropriate choice there (S69). Trapped modes sit just below a cutoff, and
  quasi-trapped modes above it couple weakly (S70).
- **Fundamental vs HOMs in damped cavities.** The damping WGs are high-pass filters (fc/f0 ≈ 1.2–1.35) that
  "perturb the fundamental relatively lightly" (S3, S6). A closed-PEC model of such a cavity therefore gets the
  fundamental right. Every mode above fc becomes a resonance of the PEC-capped guide, whose frequency depends on
  the arbitrary stub length; in reality these HOMs have low Q (DORIS: HOM Qext 300–10 000, S78).

**Our pipeline** (PEC caps; pipes 1.5–4 rp long) [calc]

Field amplitude left at the cap, exp(−αLp), for a TM010-like mode (k R = 2.405):
- through TM01, the only pipe mode a monopole couples to: 0.03 at Lp = 1.5 rp down to 1e-4 at Lp = 4 rp, for
  any rp/R in 0.12–0.35.

For dipoles and higher modes (TE11, the lowest pipe mode they couple to):

| mode | rp/R = 0.35, Lp = 1.5 rp | rp/R = 0.35, Lp = 4 rp |
|---|---|---|
| TM110-like (k R = 3.832) | 0.15 | 6e-3 |
| k R = 5 | 0.42 | 0.10 |

- TE11 cutoff / TM010 = 0.7655 R/rp: 2.19 at rp/R = 0.35, 3.06 at 0.25.
- So the monopole fundamental is insensitive to the caps, but the dipole and higher modes of large-pipe draws
  carry a cap-dependent shift.

## 3. Reference designs

All values are from search extracts of the cited sources unless tagged. "Rebuild" says whether the published
data suffice to rebuild the geometry as a closed-PEC validation case:
- **YES**: rebuilt and checked [calc].
- **partial**: main dimensions known, some missing.
- **no**: only figures of merit.

G values are derived from Q0 with ideal copper (σ = 5.8e7 S/m) [inference]. A measured Q0 makes such a G a
lower bound.

| # | design (f) | published dimensions (ratios) | published FoM (convention) | derived [inference] | sources | rebuild |
|---|---|---|---|---|---|---|
| 1 | SLAC two-mile 2π/3 cell "cavity 79" (2856 MHz) | a 9.95, b 40.96, t 5.84, d 34.99 mm (a/b 0.243, t/b 0.143, d/b 0.854, b/λ 0.390) | structure r = 53 MΩ/m | — | S44–S46 | **YES**: 3 periods with electric walls at the mid-planes give 2855.2 MHz at 2π/3 vs 2856 [calc]; iris edge assumed full-round |
| 2 | CLIC_G (11.994 GHz) | a 3.15 → 2.35, t 1.67 → 1.00 mm, 26 regular cells (a/λ 0.126 → 0.094) | Q 5536 / 5738; R' 81 / 103 MΩ/m; Es,max 230 MV/m | G 158–164 Ω | S47, S48 | partial (b, iris shape missing) |
| 3 | BNL/SLAC/UCLA 1.6-cell gun (π 2856.0, 0 2852.5 MHz) | full cell L 32.5, D 83.5; half cell L 22.8, D 83.0; iris D 25.0, ρ 9.5 mm (L_h/L_f 0.70, a/R 0.30, ρ/R 0.23) | Δf(0–π) 3.5 MHz | — | S38 | partial: with an assumed 19 mm full-round iris, 2859.1 / 2879.0 MHz (+0.1 %) but separation 19.9 MHz, not 3.5 [calc]; needs the real iris and cell drawing |
| 4 | LCLS gun (2856 MHz) | half cell R 42.15, L 24.26; full cell racetrack arc R 41.43, separation 2.8 (or 5.9 offset, S42), L 34.01; slot 13.2 mm | Q0 13 960 design / 13 900 meas. (12 000 in S40); 49 MΩ/m; Δf 15 (15.17) MHz; 120–140 MV/m | G ≈ 195 Ω | S39–S42 | partial (iris missing; racetrack and slots are 3D) |
| 5 | PITZ 1.6-cell (1.3 GHz) | — | ~60 MV/m, ~6.5 MW; coaxial coupler | — | S43 | no |
| 6 | PEP-II cavity (476 MHz) | 3 rectangular WGs, nose cones | Ks = V²/2P 3.5 MΩ (HER); 0.93 MV; 122 kW; Q0 ≈ 31 000 with ports at 40 °C (OCR); ~850 kV / 103 kW typical | R/Q 113 Ω circuit = 226 Ω linac; G 176 Ω | S1–S5 | no |
| 7 | EU HOM-damped cavity, BESSY / ALBA DAMPY (500 MHz) | beam pipe Ø 74 mm (rp/λ 0.062); 3 circular double-ridge WGs, fc 615–670 MHz (fc/f0 1.23–1.34); 0.5 m insertion | undamped optimum cell: Rsh 6.1 MΩ, R/Q 148.5 Ω, Q0 41 100; damped: Rsh 3.1–3.3 MΩ, HOM ≤ 4.8 kΩ / 200 kΩ/m; 700 kV, 80 kW | R/Q ≈ 297 Ω linac if circuit; G 240 Ω (undamped cell) | S6–S10, S16 | no |
| 8 | ESRF-EBS (352.37 MHz) | beam pipe Ø 100 mm (rp/λ 0.059); 3 dampers (2 installed) | R/Q 145, Q0 35 700, Rs 5 MΩ; 500 / 750 kV | R/Q ≈ 290 Ω linac if circuit; G 175 Ω | S12–S15 | no |
| 9 | MAX IV 100 MHz capacity-loaded | inner radius 410, rod radius 60, gap 50 mm (R/λ 0.137, rod/R 0.146, g/R 0.122) | Rsh 3.52 MΩ linac (theory), 3.3–3.45 meas.; Q0 20 923 theory, 19 700–20 500 meas.; 300 kV; 1.71 MΩ in S19 | R/Q ≈ 168 Ω linac; G ≈ 55 Ω | S17–S19 | partial (length, nose shape, pipe missing) |
| 10 | ALS main cavity (499.64 MHz) | — | Q0 28 670, QL 167 | G ≈ 167 Ω | S20, S21 | no |
| 11 | KEK PF 500 MHz (Jensen's example) | — | R/Q 111 Ω (optimised) vs 107.5 Ω; Q 44 270 vs 41 630 | 111 Ω ≈ analytic optimum of a pipe-less pillbox in the circuit convention (111.5 Ω; 223 Ω linac); G 258 Ω | S25 | no |
| 12 | KEK ARES accelerating cavity (508.887 MHz) | 4 WGs + grooved beam pipes, SiC | Rsh/Q0 150 Ω, Q0 ~30 000 (damping-ring variant) | — | S23 | no |
| 13 | DAΦNE (368.26 MHz) | rounded cell, no nose cones, long tapers, 5 WGs | ~250 kV per ring | — | S24 | no |
| 14 | Linac4 PIMS (352.2 MHz) | 12 cavities × 7 equal cells | ZTT 23.1–26 MΩ/m; k 5.0–5.6 %; Es 1.8 Kilpatrick; 3.9 MV/m | — | S34, S35 | no |
| 15 | LIBO / ACLIP SCL (2998 MHz) | web 4 mm, aperture radius 4 mm (a/λ 0.040), nose angle 20° | 16 MV/m; shunt impedance within 3 % of computed | — | S29, S30 | no |
| 16 | PEP-II klystron cavities (476 MHz) | — | R/Q 100–108 Ω; θ 0.95–1.14 rad; Qext 1000 | g/λ ≈ 0.16 βe | S27 | no |
| 17 | HL-LHC DQW, SPS series (400.79 MHz) | body 352 × 288 × 286 mm (0.47 / 0.39 / 0.38 λ) or envelope 659.6 × 495.7 × 329.6 mm | R_T/Q 429.3 Ω; G 87 Ω; Ep 37.6 MV/m, Bp 73 mT at 3.4 MV; first HOM 567–570 MHz | HOM/f0 1.42 | S54, S55 | no (posts and hat not in the extracts) |
| 18 | HL-LHC RFD (400.79 MHz) | tank 281 × 281 mm (0.376 λ), aperture Ø 84 mm (0.30 of the tank) | Vt 3.34 MV; Ep 35.0 MV/m; Bp 55.1 mT; R_T ≈ 430 Ω; HOMs 636 (dipole), 699 MHz (accelerating) | HOM/f0 1.59 / 1.74 | S56, S58 | no (pole geometry missing) |
| 19 | RFD PoP (400 MHz) | Ø 339.9, L 542.4 mm (0.454 / 0.724 λ) | — | — | S57 | no |
| 20 | ILC RFD (1.3 GHz) | pole separation 25, aperture 50, pole length 85, pole height 31.5 mm, pole angle 22.5° | — | — | S59 | partial |
| 21 | KEKB crab (508.9 MHz) | squashed cell; LOM 0.845 f0, other TM110 1.38 f0 | Vkick 1.41 / 1.44 MV; Esp/Vkick 14.4 (MV/m)/MV | — | S61 | no |
| 22 | Lancaster 4-rod crab (400 MHz) | transverse radius < 142 mm (0.19 λ) | [R/Q]_T 915 Ω; G 62.8 Ω; Ep* 4.0 MV/m/MV; Bp* 7.56 mT/MV; at 3 MV: 27.7 MV/m, 63.9 mT | — | S62 | no |
| 23 | CLIC choke-mode structure (~12 GHz) | iris 2.35–3.15 mm; radial-line and choke gap 1 mm (g/λ 0.04); 24 cells | 75 MV/m (single cells, choke-limited) | — | S51 | no |
| 24 | KEK cERL deflector (5.2 GHz) | NC TM110 pillbox, no nose | — | — | S63 | no |

**Validation geometries we can build now** (section 5, H5 and M1)
- **#1** exactly, as a closed 3-period segment.
- **#3** for the frequency, not for the 0–π splitting.
- **#9** only qualitatively.

## 4. Comparison with our generator

Effective distributions come from 400 random draws per family through the generator's own builders, with the
5 % fillet floor, solved axisymmetrically for the lowest TM0 (accelerating) mode [calc]. The 3D label
pipeline's mode set also contains dipoles etc.; those do not enter these numbers.

### 4.1 `reentrant` (nose-cone cell)

| quantity | our draw | effective, min / median / p95 / max [calc] | literature |
|---|---|---|---|
| Lc/R | 0.4–1.0 | 0.40 / 0.67 / 0.96 / 1.00 | R/Q-optimised β = 1 cells need Lc/R ≳ 1.0–1.4 [calc scan]; not given in extracts |
| rp/R | 0.10–0.25 | 0.10 / 0.175 / 0.24 / 0.25 | rp/λ = 0.045–0.062 (S6, S12) → rp/R ≈ 0.15–0.2 at R/λ ≈ 0.3–0.35 [inference] |
| g/Lc | 0.15–0.6 | 0.15 / 0.39 / 0.58 / 0.60 | "wide gap" at β = 1 (S22) |
| g/λ | derived | 0.015 / 0.074 / 0.16 / 0.20 | klystron ≈ 0.08–0.13 [inference from S27]; capacity-loaded 0.017 (S17); β = 1 optimum 0.14–0.33 [calc] |
| tip outer radius rt/R | rp + (0.08–0.2) R | 0.19 / 0.32 / 0.40 / 0.44 | MAX IV rod 0.146 R (S17) |
| cone half-angle | −25…35° | median 8.7° | 20° (S29), 30° (S31); convention unverified |
| R/λ | — | 0.216 / 0.302 / 0.346 / 0.359 | MAX IV 0.137 (S17); pillbox 0.383 |
| R/Q (linac, β = 1) | — | 74 / 161 / 255 / 321 Ω | PEP-II 226; EU / ESRF ≈ 290–300 if circuit; MAX IV 168; KEK PF 223 [inference] |
| G | — | 74 / 127 / 174 / 200 Ω | 167–258 Ω (storage rings), 55 Ω (MAX IV) [inference] |
| T | — | 0.87 / 0.96 / 0.985 / 0.99 | β = 1 optimised cells T ≈ 0.83–0.94 [calc scan] |

**What our ranges cover**
- Narrow-gap, moderately loaded nose-cone cells: 71 % of the draws have g/λ ≤ 0.1, the klystron / IOT / buncher
  regime (S27, S28, S32).
- A correct qualitative trend: R/Q rises with Lc/R and g/R. A linear fit of R/Q on the eight ratios has
  R² = 0.98; the largest coefficients are Lc/R and g/R. G rises with g/R (correlation 0.94) [calc].

**What they miss**
1. **R/Q- and G-optimised β = 1 storage-ring cells** (PEP-II, EU / ALBA / ESRF, ALS, KEK PF, Elettra).
   - Only 13 % of our draws reach R/Q ≥ 226 Ω, 0.5 % reach ≥ 290 Ω, 5 % reach G ≥ 175 Ω, and 4.3 % reach both
     R/Q ≥ 226 Ω and G ≥ 175 Ω.
   - The scan [calc] (rp/R 0.15, tip 0.1 R, 20° cone) shows where they sit:
     - Lc/R = 1.2–1.4 with g/Lc = 0.4–0.6 gives R/Q 278–315 Ω and G 157–212 Ω (g/λ 0.14–0.28);
     - g/Lc = 0.75 at Lc/R = 1.0–1.4 gives G 229–254 Ω at R/Q 250–259 Ω.
   - Our upper bounds Lc/R = 1.0 and g/Lc = 0.6 cut exactly this region away.
2. **Strongly capacity-loaded cavities** (MAX IV; also harmonic cavities, S17).
   - Ours: R/λ ≥ 0.216 and G ≥ 74 Ω. MAX IV: R/λ = 0.137, G ≈ 55 Ω.
   - With MAX IV's rod and gap ratios (rt/R 0.146, g/R 0.122), R/λ falls to 0.157 only at Lc/R = 2 [calc].
     This is indicative only: the MAX IV length, nose shape and pipe radius are not in the extracts, and the
     model rod (rp = 0.06 R) gives R/Q 290–440 Ω, far above MAX IV's ≈ 168 Ω.
   - Our nose radius floor rt/R ≥ 0.19 excludes the MAX IV value of 0.146.
3. **β < 1.** Proton and klystron nose-cone cells (S27, S29–S35) are designed and judged at β < 1. Our
   geometry range covers many of them, but our FoM labels are β = 1 only.

**Real features we do not model**
- Ports on the cell: couplers, plunger tuners, pick-ups.
- HOM waveguides (S3, S6, S23, S24) and tapers to the chamber (S24).
- Grooved beam pipes (S23).
- Nose cones with separate inner and outer nose radii. Our profile has one tip thickness with a rounding of at
  most t_tip/2; 43 % of the tip fillets sit at the 5 % floor, i.e. full rounds. That is compatible with real
  rounded noses.

**Cost**
- docs/29 §2.4: 13k, 28k, 38k, 42k curved tets for 4 draws; half of them are over the 30k budget.
- Small gaps drive this (g/R down to 0.067). Moving probability towards wider gaps (H3 below) should lower the
  cost per draw [inference].

### 4.2 `pillbox_pipes`

| quantity | our draw | effective [calc] | literature |
|---|---|---|---|
| Lc/R | 0.4–1.6 | 0.41–1.60 | gun cells 0.55–0.82 (S38, S39); SLAC cell d/b 0.854 (S45) |
| rp/R | 0.12–0.35 | 0.12–0.35 | gun iris a/R 0.30 (S38); SLAC a/b 0.243 (S45) |
| iris rounding ρ/rp | 0.05–0.4 (≥ 0.05 R floor) | 0.14–0.42; 53 % at the floor | gun ρ/a = 0.76 (S38); SLAC/CLIC full-round t/2 |
| R/λ | — | 0.384–0.410 | pillbox-type cells 0.39–0.40 (S38, S39, S45) |
| R/Q (linac) | — | 91 / 149 / 196 / 205 Ω (min / median / p95 / max) | pipe-less pillbox optimum 223 Ω; nose cones are needed above |
| G | — | 139 / 247 / 306 / 323 Ω | KEK PF 258, LCLS gun 195 Ω [inference] |
| T | — | 0.39–0.93 | — |

**What it covers**
- The single cells of guns, TM110 deflectors (S63) and disk-loaded structures. The ratio ranges contain the
  published pillbox-type cells I could check: the gun cells and the SLAC cell. CLIC_G cannot be checked, since
  its b is missing.

**What it misses**
- Thick full-round irises between cells: gun ρ/R = 0.23, while ours is ≤ 0.4 rp ≈ 0.14 R.
- Multi-cell chains (guns, disk-loaded segments).
- A closed cathode wall (a half cell without a pipe).
- Squashed / racetrack cells (S39, S42, S61).

### 4.3 `ridged_box`

**What it is**
- A ridged-waveguide resonator: a box with 1–2 full-length ridges attached along their whole length.
- 40 % of the draws have beam pipes along the ridge axis (x) through the gap.

**Closest real cavity type: the RF-dipole / parallel-bar crab cavity**
- The parallel-bar cavity started as straight loading elements in a rectangular outer conductor (S60).
- The RFD has two ridged poles in a square tank, with the beam along the poles (S56, S58).

How our ridged_box differs:

| feature | our `ridged_box` | RFD |
|---|---|---|
| ridge / pole length | full length, ridges touch both end walls | pole length and end gap are design parameters (S56) |
| ridge / pole cross-section | thin rectangles, width 0.15–0.45 b | trapezoidal poles, flank angle 22.5° in the ILC design (S59) |
| gap | 0.40–0.76 d (double ridge), 0.60–0.88 d (single) | pole separation ≈ 0.30 of the tank (84 / 281 mm, S58) |
| beam pipe | rp = (0.2–0.4)·min(gap, w), small | aperture ≈ pole separation (S58) |
| outer conductor | all box edges filleted with small radii | the outer rounding radius is a tuning knob (S56); PoP is cylindrical (S57) |
| figure of merit | longitudinal R/Q on z (see the defect below) | transverse R_T/Q (S54, S56, S62) |

Other possible relatives, not verified: the IH (H-mode) structures, which are ridge-loaded tanks. With 1–2
ridges our box is not an RFQ.

**[code] Label defect**
- `qoi.operators.beam_axis` returns `{}` for every family except `hwr`.
- So `field_labels._beam_axis` and `highorder.mode_qoi` sample the line through (0, 0, 0) along z, in
  physical label-mesh coordinates. The centre/scale normalisation exists only in `field_operators`.
- `build_ridged_box` puts the box at [0, a] × [0, b] × [0, d] and fillets every edge. That line is therefore a
  rounded corner edge of the box: outside the vacuum, or on the wall.
- As a result L_acc ≈ 0 and V ≈ 0, so R/Q, T and Epk/Eacc of `ridged_box` are not meaningful. With pipes, the
  physical beam axis is the x line through (y = b/2, z = zc).
- docs/25 already says R/Q has no accelerator meaning for this family. The point here is stronger: its axis does
  not even pass through the cavity.

**Cost**
- The worst family: 38k–89k curved tets for 4 draws (docs/29 §2.4). Most draws are re-drawn, which biases the
  family towards large fillets.

**Verdict**
- As it stands, `ridged_box` approximates no production cavity. It is a degenerate RFD (zero end gap, thin
  ridges, wide gap).
- Replace it by a parametric RFD family (H2) and keep the present form at most as an OOD variant.

### 4.4 `composite` (and `freeform`)

**Nothing in the accelerator literature looks like a CSG blob.** I found no accelerator cavity, dataset or
shape-optimisation study that uses random unions of cylinders and ellipsoids.
- Cavity shape optimisation and ML work use parametrised realistic profiles:
  - elliptical arcs, or circles instead of ellipses (S72, S73);
  - free wall angles, which give re-entrant shapes (S74);
  - spline control points sampled around a realistic initial curve (S71).

**Close analogues outside accelerators:**
- Photonic-crystal ML uses synthetic unit-cell datasets of 20 000 cells (S75).
- Wave-chaos physics studies irregular 3D microwave cavities on purpose; they follow Wigner level statistics
  (S76).

So synthetic shapes are a recognised device for diversity in ML [inference from S75]. They are just not
accelerator cavities.

**Real non-axisymmetric features that composite crudely imitates**
- Attached cylinders resemble coupler, pick-up and damper ports.
- Ellipsoidal dents resemble plunger tuners that protrude into the cell (cylindrical in practice).
- A realistic variant would put these features at realistic sizes and positions on real cell bodies (M4).

**Cost:** low (0.9k–2.8k tets, docs/29 §2.4). The family is cheap diversity, but 1 of 4 draws kept 36 sharp
edges (docs/30).

### 4.5 OOD families: `box`, `junction`, `pillbox_port`

**`box`** (sharp rectangular)
- No production NC cavity in my sources is a bare box. Rectangular-like single-cell deflectors may exist (S64,
  shape not confirmed).
- Its value is the analytic spectrum and the sharp-edge singularities. Keep it as the analytic OOD test.

**`junction`** (L / T / X box arms)
- Its real analogues are waveguide junctions in RF distribution and the cavity–damper-guide junction region,
  not cavities.
- I found no cavity design of this type. Keep it as an OOD topology test.

**`pillbox_port`** (1–2 radial round ports on the cell)
- Real analogues: coupler, tuner and pick-up ports, and the HOM waveguides on the cell (PEP-II, DAΦNE, EU,
  ARES).
- Port radius (0.12–0.3)·min(R, Lc) puts the port TE11 cutoff at ≥ 2.55 f0 (up to ≈ 10 f0 for small ports)
  [calc, from 0.7655 R/rs]. The lowest modes, up to ≈ 2.5 f0, are therefore evanescent in all our ports.
- Real damping guides have fc/f0 = 1.23–1.34 (S6, S8, S11): every mode above that propagates into them.
- So `pillbox_port` tests symmetry breaking, but not the regime that matters physically for damped cavities.
  It is a fair OOD test of the "small port" kind only.

## 5. Recommendations

Every item that changes the generator or the labels needs the owner's go-ahead (AGENTS.md). Mesh costs for new
families are **estimates by analogy** with docs/29 §2.4 (spoke 5–8k, hwr 9–27k, dtl 15–47k, pillbox_pipes
8–25k curved tets). They should be measured with 4 Sobol draws each before adoption.

### High priority

**H1. Fix the `ridged_box` QoI axis and add transverse R/Q for deflecting families** (label correctness)
- Change:
  - `beam_axis` should return the pipe axis for `ridged_box` (x line through y = b/2, z = zc) when pipes exist;
    without pipes, store R/Q, T and Epk/Eacc as undefined (NaN).
  - For every deflecting family (RFD, DQW, squashed crab cells), add R_T/Q = |V_z(x0)|² / ((k x0)² ω U)
    (Panofsky–Wenzel), from V_z on a line offset by a small x0 in the deflecting plane.
- Benefit: removes labels with no meaning today [code, §4.3]. Crab cavities are specified by R_T/Q, Ep/Vt and
  Bp/Vt (S54, S56, S62).
- Cost: trivial; one extra axis integral per mode.

**H2. Replace `ridged_box` by a parametric RF-dipole (RFD) family**
- Proposed parameters (lengths over the tank width D):

| parameter | proposed range | anchor (source) |
|---|---|---|
| tank corner rounding ρ_t/D | 0.1–0.5 (0.5 = cylinder) | LHC rounded square, PoP cylinder (S56–S58) |
| tank length Lt/D | 1.2–2.0 | PoP 1.60 (S57) |
| pole separation g_p/D | 0.2–0.4 | LHC 0.30 (S58) |
| pole length Lp/Lt | 0.5–0.9, i.e. a non-zero end gap | S56 says both are optimised; values not in the extracts, so this range is my guess |
| pole flank angle | 10–30° | ILC 22.5° (S59) |
| pole tip width w_p/D | 0.15–0.4 | guess |
| roundings | pole tips and corners ≥ the 5 % floor | — |
| beam pipes | through the end plates on the tank axis, rp = (0.4–0.5)·g_p, length 1.5–3 rp | aperture ≈ pole gap (S58) |

- With Lp = Lt and thin poles the family degenerates into the present ridged box and the parallel-bar ancestor
  (S60). Keep that corner at low probability, or keep the present `ridged_box` as an OOD variant.
- Benefit: a real, current cavity type (HL-LHC AUP, EIC, ILC designs). Its lowest mode is a deflecting TE11-like
  mode, with HOMs at 1.59 / 1.74 f0 (S56). The model gets a new mode-ordering regime in which the fundamental is
  not TM010-like.
- Cost: medium. Estimate 10–30k tets: the smallest features are the pole gap (≥ 0.2 D), the end gaps and the
  tip roundings. The present `ridged_box` needs 38–89k, because every box and ridge edge carries a small fillet.
  The mesh needs OCC booleans plus fillets; the RFD has two mirror planes, but the pipeline cannot use symmetry.

**H3. Widen the `reentrant` ranges so that real β = 1 shapes are in the bulk, not the tail**
- Change:
  - Lc/R from 0.4–1.0 to 0.4–1.6, and g/Lc from 0.15–0.6 to 0.15–0.75. Keep the existing validity checks.
  - Optionally a "storage-ring" sub-mode (~30 % of draws): Lc/R 1.0–1.6, g/Lc 0.4–0.7, rp/R 0.13–0.2, cone
    10–30°.
  - Optionally a "capacity-loaded" sub-mode (≤ 10 %): Lc/R 1.2–2.0, g/R 0.08–0.15, rp/R down to 0.05 with
    t_tip = 0.1 R, so that rt/R ≈ 0.15 (MAX IV rod 0.146 R, S17).
- Benefit:
  - Today R/Q ≥ 226 Ω in 13 % of draws, ≥ 290 Ω in 0.5 %, G ≥ 175 Ω in 5 % [calc].
  - PEP-II, EU / ALBA / ESRF, ALS and KEK PF sit at R/Q ≈ 223–300 Ω (linac) and G ≈ 167–258 Ω [inference].
    The scan reaches that region at Lc/R 1.2–1.4 and g/Lc 0.4–0.75 [calc].
  - The capacity-loaded sub-mode extends R/λ from ≥ 0.216 down to ≈ 0.16 [calc].
- Cost:
  - Cheap for the main widening: wider gaps should lower the tet count per draw [inference]. The present family
    has 2 of 4 draws above 30k (docs/29 §2.4), driven by small gaps.
  - The capacity-loaded sub-mode is costly: small gap, thin nose, long cell. A t_tip below 0.1 R also caps the
    tip fillet below the 5 % floor (cap = t_tip/2), so keep t_tip ≥ 0.1 R there.

**H4. Add a double-quarter-wave (DQW) family as a variant of `hwr` / `coax_qw`**
- Geometry:
  - outer conductor of height H along the post axis y, either a box with rounded corners or a cylinder;
  - two posts of radius ri from opposite end plates, with a capacitive gap g at mid-height;
  - optional "hat", i.e. post ends widened to radius rh ≥ ri;
  - beam pipe along z through the gap.
- Ratios: Lz : W : H ≈ 1.23 : 1.0 : 1.0 at the SPS-series body (352 × 288 × 286 mm, S54); g/H and ri/W
  unknown, so vary them broadly (e.g. 0.1–0.4 and 0.15–0.35).
- Benefit: the second HL-LHC crab type (R_T/Q 429 Ω, G 87 Ω, first HOM 1.42 f0; S54, S55). It bridges the OOD
  `coax_qw` (one quarter wave) and `hwr` (inner conductor touching both plates).
- Cost: cheap to medium, like `hwr` (9–27k tets). Without the through-conductor the vacuum region is simply
  connected (b1 = 0), so it is easier than `hwr`.

**H5. Add a 1.6-cell RF-gun family (axisymmetric)**
- Geometry:
  - cathode half cell with a flat back wall and no pipe;
  - thick full-round iris; full cell; exit iris and beam pipe;
  - options: an extra full cell (2.6-cell), and a racetrack full cell (offset 0–0.15 R, S39, S42; makes it 3D).
- Ratios: L_half/L_full 0.6–0.75 (0.70, 0.713); L_full/R 0.7–0.9 (0.78, 0.82); a/R 0.25–0.35 (0.30); iris
  thickness t/R 0.3–0.5 with ρ = t/2 (ρ/R 0.23 in S38); exit pipe radius = a, length 1.5–3 a.
- Benefit:
  - The standard NC photo-injector (S38–S43).
  - Its 0/π pair is split by only Δf/f = 1.2e-3–5e-3, a sharp test of the operator's resolution of
    near-degenerate modes.
  - The ratio of the two cell fields is a direct field-balance check.
  - Validation geometry: the BNL/SLAC/UCLA table (#3) gives the frequency to +0.1 % [calc].
- Cost: cheap; two pillbox-like cells, estimate 8–20k tets.

### Medium priority

**M1. Add a closed disk-loaded standing-wave segment**
- Geometry:
  - N = 2–5 periods, ended by PEC planes through the cell centres (electric-wall symmetry), so the modes are the
    passband points φ = nπ/N;
  - no beam pipes needed;
  - a/b 0.22–0.33; t/b 0.08–0.17; d/b 0.6–1.3 (2π/3 → 0.85, π → 1.28 [inference]);
  - iris edge full-round or elliptical.
- Benefit: the most common NC accelerating structure. Its dense passband (0.8 % wide for the SLAC cell [calc])
  is a hard multi-mode test.
- Exact validation geometry: SLAC "cavity 79", #1, reproduced to 0.03 % at 2π/3 [calc].
- Cost: medium. Thin disks × N; estimate 4–6k tets per period, so N ≤ 4–5 under 30k.
- An alternative is to let the `elliptical` family reach pillbox-like equators (A/L down to ~0.1). That belongs
  to the SRF report.

**M2. Add squashed / racetrack cells**
- Change: apply a transverse scale s ∈ [0.8, 1.0], or a racetrack offset 0–0.2 R, to pillbox / reentrant /
  elliptical single cells before fusing the pipes.
- Benefit: realistic non-axisymmetric cells (KEKB crab with the other TM110 polarisation at 1.38 f0, S61; LCLS
  racetrack, S39, S42; TM110 deflectors, S63). They teach polarisation splitting of degenerate dipoles.
- Cost: cheap; one smooth cell.

**M3. Add a HOM-damped cell with 2–4 waveguide stubs** (rectangular or double-ridged, PEC-capped). Use with
care.
- Geometry: guide cutoff fc/f0 1.2–1.4 (S6, S8, S11). A rectangular guide is then w ≈ 0.36–0.42 λ ≈ 1.0–1.3 R
  wide [inference]; ridged guides are narrower. Positions: 3 at 120° (EU, PEP-II) or 4 above and below (ARES).
- Caveat [calc]: in a closed PEC model the stubs must be 0.8–1.0 λ long for the fundamental to decay to 1 % at
  the cap (fc/f0 1.34–1.23). The model then has many stub resonances above fc, which are physically damped
  (S3, S78).
- Benefit: real geometry for PEP-II / DAΦNE / EU / ARES-type cavities, with the fundamental realistic.
- Cost: costly in volume and in junction fillets. The HOM labels would be PEC-stub artefacts, not the real
  damped modes. Keep it OOD, or adopt it only once port BCs are available (S65–S67).

**M4. Make most of `composite` realistic**
- Change:
  - Draw a real single-cell body (pillbox / reentrant / elliptical) plus 1–3 realistic attachments: coaxial or
    loop-coupler ports (radius 0.1–0.25 R, on the cell or on a pipe), cylindrical plunger tuners protruding
    0–0.3 R into the cell (a realistic "dent"), small pick-up ports, waveguide stubs.
  - Keep 10–20 % of pure CSG composites for diversity.
- Benefit: the same topological diversity as now, at physically meaningful sizes and positions. Synthetic shapes
  remain as a minority regulariser, the way synthetic unit cells are used in photonics ML (S75).
- Cost: cheap; reuse the `composite` and `pillbox_port` code.

**M5. Add β-dependent figures of merit**
- Change: T(β) and R/Q(β) at e.g. β = 0.5 and 0.75, from the same fields.
- Benefit: makes the `reentrant` labels meaningful for klystron, IOT, buncher and proton cells (S27–S35).
- Cost: cheap; QoI only.

**M6. Reduce the cap sensitivity of the labels**
- Change:
  - Raise the minimum pipe length to ~2.5 rp when rp/R > 0.25. The TE11 amplitude left at the cap for a
    TM110-like mode is 0.15 at 1.5 rp and 6e-3 at 4 rp [calc].
  - Run a diagnostic on a few samples per family with magnetic (PMC) caps instead of PEC (S68), and flag the
    modes whose frequency moves.
- Benefit: dipole and higher labels depend less on the arbitrary pipe length.
- Cost: cheap; the pipes mesh coarsely.

### Low priority, or deliberately left out

- **L1. Choke-mode cell.** Costly / infeasible.
  - The 1 mm radial-line and choke gap at 12 GHz is ≈ 0.04 λ ≈ 0.1 R [inference]. Under the 5 % floor the slot
    corners become full rounds and the slot needs many elements.
  - A closed PEC radial line without its SiC load gives radial-line resonances (S49, S51, S52).
  - At most one OOD case, once the cell radius is known.
- **L2. Side-coupled / ACS / PIMS chains with coupling slots.** Costly: two cell types and thin slots; more than
  ~2 cells will not fit 30k tets [estimate]. At most one accelerating + one coupling cell as OOD (S29, S31,
  S34).
- **L3. 4-rod crab cavity.** Costly: long thin rods (S62). Skip.
- **L4. Leave out deliberately:**
  - lossy absorbers (NiZn ferrite, SiC): not PEC;
  - loop / antenna / hook couplers: thin inner conductors that also add handles;
  - grooved beam pipes (ARES), cooling channels, tuner mechanics;
  - multi-cavity systems (ARES).
  - Reason: non-PEC physics, or features far below the 5 % floor.
- **L5. Keep `box`, `junction` and `pillbox_port` as OOD.** Optionally add a "wide-port" OOD variant of
  `pillbox_port` with port cutoff 1.2–1.5 f0, to probe the damper regime; same caveat as M3.

## 6. Open questions and what I could not verify

1. **No full text was read** (section 0). These numbers should be checked against the PDFs before they are used
   as targets:
   - PEP-II Q0 ≈ 31 000: garbled OCR (S1).
   - The R/Q convention of the EU and ESRF values (S10, S15). I inferred "circuit".
   - The ESRF damper cutoff of 452 MHz (S11, attribution uncertain).
   - Whether "R_T 431 Ω" in S56 is R_T/Q.
   - Which SLAC note gives "cavity 79" (S45).
2. **Storage-ring nose-cone dimensions.** No extract gave g, Lc or the nose radii of a real β = 1 storage-ring
   cavity. My "where real cavities sit" statements (§4.1, H3) rest on published R/Q and G plus my own
   axisymmetric scan. The scan uses one nose shape, no ports and no dampers.
3. **Cone-angle convention.** The 20° and 30° optima (S29, S31) may be measured from the axis, as our α is, or
   from the radial direction. Check the CCLFISH manual.
4. **Shunt-impedance factor of 2.** Is the drop from 6.1 MΩ (undamped cell, S10) to 3.1–3.3 MΩ (damped
   cavities, S7, S16) a damping penalty or a change of convention?
5. **MAX IV geometry.** Cavity length, nose shape and pipe radius are unknown. The capacity-loaded scan is
   indicative: its R/Q (290–440 Ω) is far above MAX IV's ≈ 168 Ω.
6. **RFD and DQW parameters.** The RFD pole length and end gap and the DQW post, hat and gap dimensions are not
   in the extracts; the H2 and H4 ranges are partly guesses. Sources to read: PRSTAB 16, 012004 (S60), the
   De Silva dissertation (https://casa.jlab.org/publications/CASA_PhDs/De%20Silva_Dissertation.pdf), IPAC'12
   WEPPC102 (https://epaper.kek.jp/IPAC2012/papers/weppc102.pdf), and arXiv:1502.04088 (S55).
7. **CLIC_G.** The cell radius b and the iris ellipticity were not in the extract, so CLIC_G cannot be rebuilt.
8. **LCLS racetrack definition.** The 2.8 mm arc separation (S39) and the 5.9 mm circle offset (S42) may describe
   different design versions.
9. **Klystron gaps.** βe was not in the extract, so the g/λ conversion for klystron cavities is indicative.
10. **Not verified at all:**
    - whether the KEK 2.6 GHz deflector is rectangular (S64);
    - ridge-loaded H-mode (IH) and RFQ cavities as relatives of `ridged_box` (that search failed);
    - typical ACS, SNS-CCL and side-coupled cell dimensions.
11. **Scope of my numbers.**
    - The [calc] distributions are for the lowest TM0 (monopole) mode only. In the 3D labels a dipole-type
      TE111 mode can come below TM010 only for long cells (pillbox: ℓ > 2.03 R, S26), which is outside our
      ranges.
    - The proposed mesh costs are analogies, not measurements.
