# A2 — Low-β TEM-class and drift-tube structures: literature survey and comparison with our generator

Scope: quarter-wave (QWR), half-wave (HWR), single / double / triple spoke, CH, IH, Alvarez DTL, short RFQ note.
Our families concerned: `hwr`, `spoke`, `dtl` (training) and `coax_qw` (OOD), as implemented in
`src/data_gen/cavity_shapes.py` on branch `claude/field-model` (summary: `lit/our_generator.md`).
No repository file was edited.

## 0. How to read this report

**Access limits (important).** I could not open a single full text. From this sandbox, WebFetch failed with
DNS errors (`getaddrinfo ENOTFOUND`), and the egress proxy refused (HTTP 403) arxiv.org,
accelconf.web.cern.ch, proceedings.jacow.org, jacow.org, epaper.kek.jp, osti.gov, cds.cern.ch,
indico.cern.ch, indico.fnal.gov, indico.triumf.ca, journals.aps.org, link.aps.org, inspirehep.net,
lss.fnal.gov, uspas.fnal.gov, researchgate.net, publikationen.ub.uni-frankfurt.de, doi.org and the other
literature hosts I tried. **Every statement below comes from search-engine extracts of the cited pages.**
These are abstracts plus the table snippets the search tool quoted. Treat each number as "read in an
extract", not "checked in the full paper". The web-search budget is shared by all agents (200 calls per
turn). It ran out after 37 of my queries, so several planned look-ups were never run (§6).

**Tags.**
- **(derived):** computed by me from the source's own numbers, e.g. Epk/Eacc = Ep / (Va / Leff).
- **(inference):** my reasoning or a TEM-line estimate; no source states it.
- **(extract unclear):** the extract did not make the attribution or the column alignment unambiguous.

**Units and notation.**
- Bpk/Eacc is in mT/(MV/m); 1 G = 0.1 mT. λ = c/f.
- R/Q is quoted in the source's own convention. Where a source names the convention it is "linac",
  V²/(ωU), which is also our default.
- Ratios use our notation: Ro / ri are the outer / inner conductor radii, D = 2Ro (or 2Rt) the outer
  diameter, Rt the tank radius, Rd the drift-tube outer radius, rb the bore radius, L / Lt / H the
  lengths, Lc the cell length.

## Answers at a glance

- **(a) Realistic shape ratios.** Details in §2; numbers per design in §3.
  - **QWR:** D/λ ≈ 0.05–0.13 and rb/Ro ≈ 0.11–0.14. No QWR height was found. With H ≲ λ/4, H/D ≈ 2–5
    (upper-bound estimate, inference). Inner/outer ratio: no data. The TEM-line optimum is
    ri/Ro = 1/e ≈ 0.37 (inference).
  - **HWR:** IFMIF has Ro/λ = 0.053, rb/Ro = 0.22 and D/βλ = 1.1. This implies Ro/L ≈ 0.1 at β ≈ 0.1
    (inference).
  - **Spoke:** D/λ = 0.53–0.69 and rb/Rt = 0.06–0.09. Interior or overall length / Rt: ≈ 1.9 (single
    spoke, β ≈ 0.5), 3.0–4.1 (double), 3.7 (triple).
  - **DTL:** D/λ = 0.58–0.61, d/D = 0.17–0.27, rb/Rd = 0.14–0.22 (rb/Rt ≈ 0.04). Lc = βλ ≈ 3.3 β·Rt, so
    Lc/Rt ≈ 0.26–1.36 over 3–90 MeV (derived). A tank holds 23–75 drift tubes. Nose and corner radii are
    0.4–4 % of Rt. The face angle is the per-cell tuning knob. No sourced g/βλ value was obtained.
  - **CH:** 19 gaps of βλ/2 in L/λ ≈ 1.26.
  - **IH:** L/D ≈ 11 (HICAT).
- **(b) What we miss.**
  - HWR: low-β slender proportions; racetrack or ring sections of the inner conductor at the beam port.
  - QWR: a realistic one does not exist in training.
  - Spoke: shaped end walls (re-entrant, convex, "balloon"); a base-to-centre spoke taper; β ≥ 0.4
    lengths; triple spokes.
  - DTL: half drift tubes on the end walls, small bores, face angles, a cell-length ramp, post couplers,
    tuners and low-β cells (Lc < 0.5 Rt).
  - Whole types: CH and IH.
- **(c) QWR:** yes. Add a filleted, ported, realistically proportioned `qwr` training family. Keep the
  sharp-edged `coax_qw` as the OOD probe.
- **(d) Conventions.**
  - **What designers report.** R/Q = V²/(ωU) at β_opt (maximum transit-time factor) and Eacc = V/Leff.
  - **Leff choices:**
    - βλ for two-gap QWR / HWR / single spoke;
    - N_gaps·βλ/2 at ESS and for the CH;
    - (N_gaps − 1)·βλ/2 for the ANL triple spoke;
    - ad hoc values elsewhere (ISAC-II: 18 cm).
    - Effect: the same field gives Epk/Eacc values that differ by up to 4/3 between conventions.
  - **Problem in our labels.**
    - Labels use β = 1 and L_acc = the axis chord (pipes included).
    - On a two-gap cavity with β_opt ≈ 0.1, β = 1 cuts R/Q to ≈ 2–3 % of the design value (inference).
  - **Beam axis.** Transverse for HWR and QWR, along the tank axis for spoke and DTL.
- **(e) Meshability under our constraints** (30k curved tets, fillets ≥ 5 % of size, closed PEC).
  - **Cheap:** spoke variants (currently 5–8k tets), a QWR at HWR cost, a DTL unit cell.
  - **Costly:**
    - slender HWRs or QWRs;
    - DTL bores near real size, post couplers;
    - triple-spoke or CH sections with 3–4 crossbars.
  - **Infeasible:**
    - full DTL / CH / IH tanks: more than ~4 cells with drift tubes, while the 2–4-cell `dtl` already
      needs 15–47k tets;
    - RFQ vane tips.

## 1. Scope and sources

Each item gives the URL and what it provides. All were read as search extracts (§0).

**General and reviews**
1. [S1] M. Vretenar, "Low-beta structures", CAS 2010, arXiv:1201.2593, https://arxiv.org/abs/1201.2593 —
   QWR is the most widespread low-β heavy-ion resonator; HWR is used when even small on-axis dipole
   components must be avoided; spoke principle. The extract did not name the paper, so the attribution is
   unclear.
2. [S2] A. Facco, SRF2015 Tutorial "Non-elliptical resonators" I+II,
   https://indico.triumf.ca/event/0/contributions/2/attachments/9/10/SRF2015_Tutorial_-_Non-Elliptical_resonators_-_Part_I_and_Part_II.pdf
   — an HWR is equivalent to 2 QWRs facing each other and needs ≈ 2× the power for the same voltage.
3. [S3] M. Kelly, PAC2009 TU3RAI02 talk, https://accelconf.web.cern.ch/pac2009/talks/tu3rai02_talk.pdf —
   table of QWR linacs (frequency, β, number of cavities).
4. [S4] R. Laxdal, "Challenges for low beta SCRF", TTC 2008,
   https://indico.desy.de/event/946/contributions/89881/attachments/60392/73263/Challenges_for_Low_beta_SCRF-Laxdal.pdf
   — examples per type; ISAC-II QWR design values.

**FRIB QWR / HWR**

5. [S5] Popielarski et al., LINAC2010 THP039, https://accelconf.web.cern.ch/LINAC2010/papers/thp039.pdf —
   FRIB parameter table (β = 0.041 QWR); design rule Ep ≤ 31.5 MV/m and Bp ≤ 77 mT.
6. [S6] Facco et al., IPAC2012 MOOAC03 talk, https://accelconf.web.cern.ch/IPAC2012/talks/mooac03_talk.pdf —
   FRIB 4-cavity table with Va, Ep, Bp, R/Q, G, aperture and "Leff ≡ βλ"; operating limits.
7. [S7] "The FRIB project", HIAT2012 MOB01, https://proceedings.jacow.org/HIAT2012/papers/mob01.pdf —
   upgraded HWR parameters (R/Q, G, Ep, Bp, aperture).
8. [S8] FRIB, LINAC2014 THIOA02, https://accelconf.web.cern.ch/linac2014/papers/thioa02.pdf — QWR R/Q
   402 / 455 Ω, G 15.3 / 22.3 Ω, aperture 36 mm.
9. [S9] FRIB, NAPAC2016 WEB3IO01, https://accelconf.web.cern.ch/napac2016/papers/web3io01.pdf — QWR β
   listed as 0.043 / 0.086, aperture 36 mm.
10. [S10] Popielarski et al., SRF2011 TUPO060, https://epaper.kek.jp/SRF2011/papers/tupo060.pdf — ≈ 100
    β = 0.085 QWRs for FRIB; 2 K operation (ReA3 at 4.5 K).
11. [S11] "Electro-magnetic optimization of a half-wave resonator", SRF2011 MOPO044,
    https://proceedings.jacow.org/SRF2011/papers/mopo044.pdf — earlier β = 0.29 HWR: R/Q ≈ 196 Ω,
    Epk/Eacc ≈ 4.8, Bpk/Eacc ≈ 77 G/(MV/m).
12. [S12] Holzbauer et al., PAC2011 TUP090, https://jacow.org/PAC2011/papers/TUP090.pdf — second-generation
    β = 0.29 HWR: tapered inner conductor (lower Bpk), straight outer tube (stiffness); aperture 30 mm with
    a 32 mm port (variant 40 / 42 mm).
13. [S13] Xu, Facco et al., SRF2011 MOPO046, https://accelconf.web.cern.ch/SRF2011/papers/mopo046.pdf —
    third-generation HWR: aperture 30 → 40 mm; beam-port flange design.
14. [S14] Miller et al., SRF2011 design-for-manufacture paper — Va 1.9 / 3.7 MV, Bp 54 / 77 mT, Ep 30 MV/m
    for β 0.29 / 0.53. (extract unclear: MOPO045 https://jacow.org/SRF2011/papers/MOPO045.pdf or MOPO051
    https://accelconf.web.cern.ch/SRF2011/papers/mopo051.pdf)
15. [S15] J. Holzbauer, MSU thesis, https://d.lib.msu.edu/etd/1534 — has a section "Cavity geometrical
    features and parameterization" for the 322 MHz β = 0.29 HWR. Table of contents only.
16. [S16] "SRF development and cryomodule production for the FRIB linac",
    https://www.researchgate.net/publication/323792596_SRF_DEVELOPMENT_AND_CRYOMODULE_PRODUCTION_FOR_THE_FRIB_LINAC
    — β = 0.53 production Ep/Eacc 4.3 → 3.6; tuner df/dx 248 kHz/mm (β 0.29) and 101 kHz/mm (β 0.53).
17. [S17] J. Holzbauer, USPAS 2012 HWR design slides, https://uspas.fnal.gov/materials/12MSU/JPH_HWR_Design.pdf
    — β = 0.53 design Bpk ≈ 75 mT; measured quench 90–110 mT.

**Other QWRs**

18. [S18] "Niobium quarter-wave resonator development for the Rare Isotope Accelerator",
    arXiv:physics/0412031, https://arxiv.org/abs/physics/0412031 — 80.5 MHz β = 0.085 and 161 MHz
    β = 0.16. Outer conductor Ø 240 mm (ALPI: 180 mm); aperture 30 mm; tests reached Ep > 30 MV/m
    (β 0.085) and Ep 20 MV/m at Q0 2.5e9 (β 0.16).
19. [S19] ATLAS 72.75 MHz β = 0.077 QWR — SRF2009 TUPPO016 https://jacow.org/SRF2009/papers/TUPPO016.pdf,
    LINAC2010 MOP049 https://accelconf.web.cern.ch/LINAC2010/papers/mop049.pdf, PAC2011 TUP046
    https://jacow.org/PAC2011/papers/TUP046.pdf, SRF2013 THIOC01
    https://accelconf.web.cern.ch/srf2013/papers/thioc01.pdf.
    - Taper on both inner and outer conductors; two gaps for 0.06 < β < 0.12; 2.5 MV design.
    - Eacc uses leff = βλ.
    - Measured > 7.4 MV with peak fields of 164 mT and 117 MV/m.
    - 72.75 MHz was chosen over 60.625 MHz to reduce microphonics.
20. [S20] SPIRAL2 QWR design, SRF2003 TUP06, https://accelconf.web.cern.ch/SRF2003/papers/tup06.pdf —
    Table 2: β 0.07 and 0.12 values (Epk/Eacc, Bpk/Eacc, r/Q, beam tube, external diameter); limits
    Ep 40 MV/m, Bp 80 mT.
21. [S21] G. Devanz, SRF2005 MOP02 (+ talk), https://accelconf.web.cern.ch/SRF2005/papers/mop02.pdf —
    - updated β = 0.07 values: Q0 2.2e9 at Rs = 10 nΩ;
    - β = 0.12 prototype: 4 mm walls, 3 mm stem, 36 mm beam tubes;
    - both prototypes reached 11 MV/m.
22. [S22] "Cryogenics of a superconducting LINAC: SPIRAL2", arXiv:2502.01196,
    https://arxiv.org/abs/2502.01196 — f = 88.0525 MHz; Eacc = Vacc/βλ, at most 6.5 MV/m in operation.
23. [S23] ISAC-II medium-β QWR — SRF2001 TL025 https://jacow.org/srf01/papers/TL025.pdf and PAC2001 MPPH134
    https://jacow.org/p01/papers/MPPH134.pdf.
    - Geometry: 106.08 MHz, β = 0.072; ID 180 mm; full bore 20 mm; gap 40 mm; "inner tube 60 mm".
    - Operating point: 6 MV/m over an 18 cm effective length.
    - Ratios: Ep/Ea 4.6; Hp/Ea 103 G/(MV/m); U/Ea² 0.09 J/(MV/m)²; Rs·Q 19.1 Ω.
    - "No agreed upon definition of cavity length within the heavy ion community."
24. [S24] HIE-ISOLDE, SRF2017 WEYA03 talk, https://accelconf.web.cern.ch/srf2017/talks/weya03_talk.pdf —
    high-β QWR, 101.28 MHz, β_opt 10.9 %:
    - R/Q 553 Ω; G 30.34 Ω; Epk/Eacc 5.0; Bpk/Eacc 96 G/(MV/m);
    - the slide carries "300 mm" and "950 mm" labels whose meaning the extract does not give.
25. [S25] D'Elia et al., "HIE-ISOLDE high beta cavity study and measurements", arXiv:0910.0409,
    https://arxiv.org/abs/0910.0409 — comparison table with TRIUMF and SPIRAL2:
    - HIE-ISOLDE: β 11.4 %, Epk/Eacc 5.4, Bpk/Eacc 96 G/(MV/m), Rsh/Q 554 Ω, Lnorm 300 mm;
    - geometric β 6.3 / 10.3 % (another extract: 6.26 / 10.84 %).
26. [S26] Fraser, Pasini, D'Elia, Jones, arXiv:0909.4930 (SRF2009 THPPO026),
    https://arxiv.org/abs/0909.4930
    - A racetrack aperture reduces the field asymmetry above 3.7 MeV/u (β = 0.09).
    - An inner-conductor modification symmetrises the fields at a cost of +20 % Epk.
27. [S27] Facco & Zviagintsev, PRST-AB 14, 070101 (2011), https://doaj.org/article/c9ccd288075a4351937a1f40c87405a1
    — analytic QWR steering formula with correction methods (abstract only).
28. [S28] HIAF QWR design: drift-tube face tilts of ≈ 1.5° and ≈ 0.5° compensate steering. (extract
    unclear: probably SRF2017 MOPB007, https://accelconf.web.cern.ch/srf2017/papers/mopb007.pdf)
29. [S29] LINAC2016 THPLR040, https://accelconf.web.cern.ch/linac2016/papers/thplr040.pdf — β ≈ 0.08
    SC-QWR prototype: Epk/Eacc 6.2, Bpk/Eacc 9.7, Rsh/Q0 587 Ω. The extract does not name the laboratory.
30. [S74] INFN-LNL / Facco QWR and HWR work, EPAC2004 author index,
    https://accelconf.web.cern.ch/e04/html/AUTH3260.HTML
    - 352 MHz HWRs, β 0.17 and 0.31 (two gaps);
    - 161 MHz β 0.16 QWR with a shaped drift tube against steering;
    - an 80 MHz β 0.085 variant.

**HWR (non-FRIB)**

31. [S30] IFMIF HWR — PAC2009 TU5PFP066 https://accelconf.web.cern.ch/pac2009/papers/tu5pfp066.pdf and SRF2009
    THPPO015 https://epaper.kek.jp/SRF2009/papers/thppo015.pdf.
    - Table: 175 MHz, β 0.094, aperture radius 20 mm, βλ 161.04 mm, cavity radius 90 mm,
      Epk/Eacc 4.42, Bpk/Eacc 10.12, Lcav = Ngaps·βλ/2 with Ngaps = 2.
    - The central electrode is racetrack-shaped at the beam, with a short transition to the round part.
    - The resonant line is conical, for stiffness and lower Bpk.
    - Coupler at mid-plane. The capacitive plunger tuner was replaced by wall deformation.
32. [S31] PIP-II HWR (ANL) — SRF2015 WEBA05 https://accelconf.web.cern.ch/SRF2015/papers/weba05.pdf and SRF2017
    WEYA05 https://jacow.org/srf2017/papers/WEYA05.pdf.
    - Design: 162.5 MHz, β = 0.11, ≈ 2 MV at 2 K; 8 HWRs in the cryomodule, 2.1 → 10.3 MeV.
    - Tests: > 3.5 MV; peak fields of 134 MV/m and 144 mT reached (LINAC2016 report in the extract).
33. [S72] "A ring-shaped center conductor geometry for a half-wave resonator", IPAC2012 WEPPC037,
    https://proceedings.jacow.org/IPAC2012/papers/weppc037.pdf — title only.

**Spoke**

34. [S32] P. Sha, "Progresses on China ADS superconducting cavities" (2013),
    https://indico.classe.cornell.edu/event/17/contributions/508/attachments/357/454/Progresses_on_China_ADS_Superconducting_Cavities.pdf
    — Spoke012, Spoke021 and Spoke040 tables (Epk/Eacc, Bpk/Eacc, G, R/Q, aperture).
35. [S33] SRF2015 WEBA01 talk, https://accelconf.web.cern.ch/SRF2015/talks/weba01_talk.pdf — HWR010,
    Spoke012 and Spoke021 values with "Eacc normalised with β0λ"; they differ from [S32].
36. [S34] Spoke040, SRF2013 THP028 https://accelconf.web.cern.ch/srf2013/papers/thp028.pdf and CPC
    arXiv:1312.5086 https://arxiv.org/abs/1312.5086.
    - A convex end wall replaces the initial flat one.
    - Targets: Ep < 32.5 MV/m and Bp < 65 mT at 7.7 MV/m.
    - Arc-shaped and plate-shaped end groups are compared.
37. [S35] Spoke012, arXiv:1310.1445 (CPC 38, 2014), https://arxiv.org/abs/1310.1445 — 12 Spoke012 cavities
    in Injector I (abstract).
38. [S36] "Design of a double spoke cavity for C-ADS", arXiv:1310.2693, https://arxiv.org/abs/1310.2693 —
    325 MHz β 0.37: Epk/Eacc 4.39, Bpk/Eacc 7.15.
39. [S37] FNAL SSR1, SRF2009 THPPO011, https://epaper.kek.jp/SRF2009/papers/thppo011.pdf — G 84 Ω,
    R/Q0 242 Ω, βg 0.21, beam pipe 30 mm, shell ID 492 mm.
40. [S38] Rowe, SRF2017 MOYA05 talk, https://accelconf.web.cern.ch/srf2017/talks/moya05_talk.pdf — SSR1
    β_g 0.215, β_opt 0.22, Leff = βλ ≈ 203 mm.
41. [S39] Project X reference design, arXiv:1306.5022, https://arxiv.org/abs/1306.5022 — SSR1 β_G 0.186,
    β_opt 0.222.
42. [S40] V. Yakovlev, PIP-II collaboration meeting slides,
    https://indico.fnal.gov/event/10586/contributions/3692/attachments/2477/2960/PIP-II_Collab_meeting_-_VYakovlev.pdf
    — SSR1 at 10 MV/m: Bp 58.1 mT, Ep 38.4 MV/m, 2.05 MeV gain.
43. [S41] Berrutti, "SSR2 RF design" (2023),
    https://indico.fnal.gov/event/58629/contributions/261007/attachments/164381/217995/SSR2_RF_design_PIP-II_Berrutti.pdf
    — SSR2 v3.1: aperture 40 mm, β_opt 0.47, Leff 0.436 m, Epk/Eacc 3.46 / 3.51, Bpk/Eacc 6.79 / 6.75.
44. [S42] Earlier SSR2 table (INSPIRE file),
    https://s3.cern.ch/inspire-prod-files-6/6a8c013479d3e27bd18abcffa2b063ab — aperture 50 mm, β_opt 0.514,
    D 560.8 mm, L 537.2 mm, Leff 475.3 mm, R/Q 275.9 Ω, G 118.6 Ω, Epk/Eacc 5.66, Bpk/Eacc 6.25.
45. [S43] SSR2 redesign, SRF2019 TUP066, https://accelconf.web.cern.ch/srf2019/papers/tup066.pdf — elliptical
    end walls (TRIUMF balloon concept) to mitigate multipacting.
46. [S44] TRIUMF/PAVAC balloon SSR, β 0.30, 325 MHz (RISP), LINAC2016 MOPLR041,
    https://accelconf.web.cern.ch/linac2016/papers/moplr041.pdf — the balloon shape removes the central
    spool.
47. [S45] ESS double spoke, HB2016 WEAM4Y01, https://accelconf.web.cern.ch/hb2016/papers/weam4y01.pdf —
    994 mm long, internal Ø 481 mm, 3 gaps, Lacc = 0.639 m.
48. [S46] ESS double spoke, SRF2017 MOYA01 talk, https://accelconf.web.cern.ch/srf2017/talks/moya01_talk.pdf
    — G 133 Ω, r/Q 427 Ω, Epk/Eacc 4.3, Bpk/Eacc 6.8; at 9 MV/m: 61 mT, 38 MV/m.
49. [S47] IPN Orsay, SRF2013 FRIOC01, https://accelconf.web.cern.ch/SRF2013/papers/frioc01.pdf — Lacc defined
    as (n+1)βλ/2 with n = number of spoke bars; 8 MV/m.
50. [S48] ANL triple spoke, PAC2005 TPPT100 https://accelconf.web.cern.ch/p05/PAPERS/TPPT100.PDF and TPPT099
    https://accelconf.web.cern.ch/p05/PAPERS/TPPT099.PDF — TSR500 and TSR630 tables and dimensions; the
    accelerating mode is the lowest one, with adjacent spokes in antiphase.
51. [S49] Hopper & Delayen, PRST-AB 16, 102001 (2013), DOI 10.1103/PhysRevSTAB.16.102001,
    https://journals.aps.org/prab/abstract/10.1103/PhysRevSTAB.16.102001 (open access)
    - Spoke cavities had been built up to β ≈ 0.6.
    - Elliptical and racetrack spoke bases do not differ significantly in peak fields or shunt impedance.
    - The paper studies the base length at a fixed 200 mm base width (325 MHz double spoke).
52. [S50] Delayen, De Silva, Hopper, "Design of superconducting spoke cavities for high-velocity applications",
    https://www.researchgate.net/publication/241971096_Design_of_Superconducting_Spoke_Cavities_for_High-velocity_Applications
    — 352 MHz β 0.82 double spoke: Ø 584.4 mm, iris-to-iris 877 mm, overall 1057 mm, aperture 50 mm,
    R/Q 545 Ω.
53. [S51] MYRRHA spoke — J-L. Biarrotte, SLHiPP-2 (2012)
    https://indico.ess.eu/event/13/contributions/863/attachments/745/1169/Jean-Luc_Biarrotte_SLHiPP2_may12.pdf
    and LINAC2014 TUPP082 https://jacow.org/LINAC2014/papers/TUPP082.pdf — β 0.37, 352 MHz single spoke.
    - First generation: 4.7 and 12.8 mT/(MV/m), wall-to-wall ≈ 36 cm.
    - Second generation: 4.4 and 8.3 mT/(MV/m), ≈ 42 cm.
54. [S71] Korean SSR1 design (KNS preprint), https://www.kns.org/files/pre_paper/55/26S-436-고은솔.pdf — Leff
    298.2 mm, R/Q 233 Ω. The extract does not confirm the project or the β.
55. [S73] C. Hopper, ODU PhD dissertation, https://digitalcommons.odu.edu/physics_etds/62/ — high-velocity
    single and double spoke development (abstract).

**CH, IH, DTL, RFQ**

56. [S52] Dziuba et al., PRST-AB 13, 041302 (2010),
    https://publikationen.ub.uni-frankfurt.de/opus4/files/44218/PhysRevSTAB.13.041302.pdf — first
    superconducting multicell drift-tube cavity (H21 mode), 7 MV/m. Related extracts: 360 MHz 19-cell
    β 0.1 → 5.6 MV; 325 MHz 7-gap; 217 MHz β 0.059 15-cell.
57. [S53] IAP Frankfurt superconducting CH, SRF2007 MO403, https://jacow.org/srf2007/papers/MO403.pdf and talk
    https://accelconf.web.cern.ch/srf2007/TALKS/MO403_TALK.pdf — 360 MHz, β 0.1, 19 gaps, L 1048 mm,
    Ep/Ea 5.2, Bp/Ea 5.7, Ra/Q ≈ 3180 Ω, G ≈ 56 Ω. The extract notes a garbled table.
58. [S54] HICAT IH-DTL, 216.816 MHz: L/D ≈ 11, three internal triplets, 4 KONUS sections, 0.4 → 7 MeV in
    3.76 m, 5.5 MV/m. (extract unclear: one of IPAC2016 THPMB032
    https://accelconf.web.cern.ch/IPAC2016/papers/thpmb032.pdf, SAP2017 MOPH12
    https://accelconf.web.cern.ch/sap2017/papers/moph12.pdf, arXiv:1311.3204)
59. [S55] KHIMA IH-DTL, https://scholar.korea.ac.kr/handle/2021.sw.korea/91874 — 200 MHz, 2 tanks, 56 gaps,
    4 triplets; "IH-DTL widely used for heavy ions: high shunt impedance at β < 0.1".
60. [S56] Linac4 DTL, LINAC08 MOP049, https://jacow.org/LINAC08/papers/MOP049.pdf — 352.2 MHz, 3–50 MeV H⁻:
    - geometry: tank Ø 520 mm, drift tubes Ø 90 mm, aperture 20 mm;
    - layout: 3 tanks, 18.7 m in total, 35 / 41 / 29 drift tubes; post couplers; PMQs;
    - field: E0 3.2 MV/m.
61. [S57] Linac4 DTL, LINAC2010 TUP066, https://epaper.kek.jp/LINAC2010/papers/tup066.pdf — E0 3.1 MV/m in
    tank 1 and 3.3 MV/m in tanks 2–3; φs −35° → −24° in tank 1.
62. [S58] M. Vretenar, "Linear accelerators", CAS, arXiv:1601.05210, https://arxiv.org/abs/1601.05210 —
    Linac4 DTL: 120 cells in 19 m.
63. [S59] J-PARC DTL — PAC2003 RPAB049 https://jacow.org/p03/papers/RPAB049.pdf (also LINAC04 TUP90 and
    LINAC06 THP069) and the MHI delivery record https://www.mhi.com/products/industry/pdf/15009_delivery_record_j-parc.pdf.
    - Geometry: 324 MHz; tank ID 560 mm; drift tubes Ø 140 mm with EMQs.
    - Tank 1: 75 drift tubes, 37 post couplers, 10 fixed tuners. Tank 2: 42 drift tubes, 42 post couplers,
      10 fixed tuners.
    - E0 2.5–2.9 MV/m. Which of these pages gives which number is not clear from the extract.
64. [S60] SNS DTL, PAC1999 FRA86 https://jacow.org/p99/papers/FRA86.pdf (with FRA83 and PAC2001 MPPH139) —
    402.5 MHz:
    - tank Ø "slightly larger than 43 cm" (early baseline);
    - 6 tanks, 210 drift tubes, 2/3 of them with PMQs;
    - Epk below 1.3 Kilpatrick, with Kp = 19.47 MV/m at 402.5 MHz.
65. [S61] ESS DTL — IPAC2014 THPME041 https://jacow.org/IPAC2014/papers/THPME041.pdf and the TAC13 review
    https://indico.ess.eu/event/406/contributions/3216/attachments/3098/4444/ESS_DTL_TAC13_.pdf.
    - Layout: 352.21 MHz, 3.62 → 90 MeV, 5 tanks of ≈ 7.6–7.8 m with 61 / 34 / 29 / 26 / 23 cells.
    - Bore radius 10–12 mm.
    - Fields: E0 3.00–3.16 MV/m, Emax/Ek 1.55, Ep < 29 MV/m.
    - DTL1: 60 drift tubes, 23 fixed and 3 movable tuners, 24 post couplers.
66. [S62] "ESS DTL RF modelization: field tuning and stabilization",
    https://inspirehep.net/files/222d961f21b9d1aca6dd17da5d357e64 — the face angle αf is varied with
    DTLfish to reach the target frequency in every cell.
67. [S63] Pramana 79(2), 263 (2012), https://www.ias.ac.in/article/fulltext/pram/079/02/0263-0274 —
    352.2 MHz, 10 MeV DTL:
    - tank Ø 52.0 cm; drift tube Ø 14.0 cm; bore radius 1.0 cm;
    - corner radius 1.0 cm; inner nose radius 0.25 cm;
    - g/L optimised for ZTT.
68. [S64] KOMAC 200 MeV upgrade plan, https://inspirehep.net/files/dce84876009eda7768f4cd45ff1d323b — drift
    tube Ø 140 mm, outer nose radius 18 mm, "45 mm inner nose radius" (sic, probably garbled), 1 mm flat,
    1 mm corner radius, "70° face angle" (definition not given).
69. [S65] Fermilab TM-1541, https://lss.fnal.gov/archive/test-tm/1000/fermilab-tm-1541.pdf — a smaller stem
    radius raises the shunt impedance (0.5 cm chosen); the tank radius scales as 1/f and its optimum
    depends on β.
70. [S66] SSC DTL physics design, LINAC1990 TU418, https://epaper.kek.jp/l90/papers/tu418.pdf — tank and drift
    tube diameters, nose and corner radii and g/L are chosen together as a compromise.
71. [S67] Billen & Young, "Poisson/Superfish on PC compatibles", PAC1993,
    https://proceedings.jacow.org/p93/PDF/PAC1993_0790.PDF — DTLFISH adjusts the tank diameter, drift-tube
    diameter, gap or face angle to hit a target frequency.
72. [S68] M. Vretenar, "The radio-frequency quadrupole", arXiv:1303.6762, https://arxiv.org/abs/1303.6762 —
    RFQ lecture (abstract).
73. [S69] CAS Zeegse 2005 RFQ lecture, https://cas.web.cern.ch/sites/default/files/lectures/zeegse-2005/rfq.pdf
    — the 4-vane RFQ suits frequencies of order 200 MHz and above (the tank grows too big below); TE210 /
    TE110 mixing.
74. [S70] Window-type RFQ at 162.5 MHz: more compact and better mode separation than a 4-vane, without
    π-mode loops. (extract unclear: probably IPAC2016 MOPMW012
    https://accelconf.web.cern.ch/IPAC2016/papers/mopmw012.pdf)

**Internal references (repository, read directly)**
- [R1] `docs/29_FIELD_LABELS.md` §2.4 — curved tets per family (5 % floor, 4 Sobol draws): spoke 5.2–7.8k;
  hwr 9.1k / 14k / 27k / fold; dtl 15k / 27k / 43k / 47k.
- [R2] `docs/24_CAVITY_QOI.md` §0.2 and §1.3 — beam axis z (hwr: x at mid-length), β = 1, L_acc = the axis
  chord inside Ω.
- [R3] `docs/19_3D_DATA_PIPELINE.md` family table — frequency ranges: hwr 0.77–1.58 GHz; spoke
  0.77–1.49 GHz; dtl stem modes at 0.40–0.77 GHz below the TM010-like mode.
- [R4] `docs/30_ADAPTIVE_LABELS.md` §3 — automatic rounding of re-entrant edges.

Count: 74 numbered entries, some grouping 2–3 papers. At least 30 give numerical geometry or figures of
merit: S5–S9, S11–S14, S16, S18–S25, S29–S32, S36–S38, S40–S42, S45, S46, S48, S50, S51, S53, S56,
S57, S59–S61, S63 and S64.

## 2. Cavity types in the area

### 2.1 Quarter-wave resonator (QWR)

**Canonical geometry.**
- A coaxial line, shorted at one end plate (high current, peak B) and open at the other.
- Near the open end the inner conductor ends in a drift tube.
- The beam crosses transversally, through the outer-wall ports and the drift tube: two gaps.
- The TEM λ/4 resonance is lowered by the tip capacitance.
- The QWR is the most widespread low-β heavy-ion resonator [S1]. Its asymmetry produces on-axis dipole
  (steering) fields. Steering comes mainly from the transverse magnetic field of the TEM mode and grows
  with β_opt [S27]. The HWR exists to remove it [S1].

**Typical frequency and β (sources here).**
- 72.75 MHz, β 0.077 (ATLAS [S19]).
- 80.5 MHz, β 0.041 / 0.085 (FRIB [S5, S6]).
- 88 MHz, β 0.07 / 0.12 (SPIRAL2 [S20]).
- 101.28 MHz, β ≈ 0.06 / 0.11 (HIE-ISOLDE [S24, S25]).
- 106 MHz, β 0.057 / 0.071, and 141.4 MHz, β 0.11 (TRIUMF [S4, S25]).
- 161 MHz, β 0.16 (MSU/RIA [S18]).
- Kelly's table adds New Delhi (97 MHz) and Legnaro (80 / 160 MHz) [S3].

**How designers parametrise it.**
- Conductors: outer conductor diameter D; inner conductor diameter d. Tapers appear on the inner conductor
  only or on both conductors (ATLAS [S19]).
- Length and tip: height H, tip-to-end-plate distance, drift-tube length and shape, gap widths, beam
  aperture.
- Steering corrections:
  - drift-tube face tilt, ≈ 0.5–1.5° (HIAF [S28]);
  - a shaped drift tube (LNL/RIA [S74]);
  - a racetrack aperture or a modified inner conductor (HIE-ISOLDE [S26]);
  - a cavity offset in the cryomodule.

**Dimensionless ranges.**

| design | f [MHz] | D [mm] | D/λ | bore | rb/Ro | H/D if H = λ/4 (upper bound, inference) |
|---|---|---|---|---|---|---|
| ISAC-II medium β [S23] | 106.08 | 180 (ID) | 0.064 | Ø 20 mm | 0.11 | 3.9 |
| SPIRAL2 β 0.07 [S20] | 88.05 | 220 (external) | 0.065 | 30 mm (taken as Ø) | 0.14 | 3.9 |
| SPIRAL2 β 0.12 [S20] | 88.05 | 370 (external) | 0.109 | 50 mm (Ø) | 0.14 | 2.3 |
| MSU/RIA β 0.085 [S18] | 80.5 | 240 | 0.064 | 30 mm | 0.125 | 3.9 |
| MSU/RIA β 0.16 [S18] | 161 | 240 | 0.129 | 30 mm | 0.125 | 1.9 |
| ALPI [S18] | 80 / 160 | 180 | 0.048 / 0.096 | — | — | 5.2 / 2.6 |
| HIE-ISOLDE [S24] | 101.28 | "300 mm" label (meaning unclear) | (0.10) | — | — | (2.5) |

- **Height.** No QWR height came through in the extracts. Real heights are below λ/4 because of the tip
  loading, so the H/D column is an upper bound.
- **ri/Ro.** No source value. For a TEM line with fixed outer radius and fixed gap voltage, the peak
  surface E and the peak surface H on the inner conductor are both minimised at ln(Ro/ri) = 1, i.e.
  ri/Ro = 1/e ≈ 0.37 (derivation, inference). Our ranges 0.2–0.45 (`coax_qw`) and 0.2–0.5 (`hwr`)
  bracket it.
- **Leff and gaps.**
  - Leff = βλ at FRIB [S6], ATLAS [S19] and SPIRAL2 [S22].
  - ISAC-II uses 18 cm < βλ = 20.4 cm (derived ratio 0.885) [S23].
  - HIE-ISOLDE quotes "Lnorm 300 mm" [S25]. That is ≈ β_geom λ = 0.103 × 2.96 m = 305 mm (inference).
  - ISAC-II gap 40 mm: gap/D ≈ 0.22 (derived).

**Figures of merit (§3).**
- R/Q 402–639 Ω.
- G 15–30 Ω; ISAC-II Rs·Q 19.1 Ω.
- Epk/Eacc 4.6–6.2.
- Bpk/Eacc 7.0–12.6 mT/(MV/m).
- Peak-field limits:
  - FRIB design rule Ep ≤ 31.5 MV/m and Bp ≤ 77 mT [S5]; "reliable operation" Ep ≤ 35 MV/m and
    Bp ≤ 70 mT [S6];
  - SPIRAL2 40 MV/m and 80 mT [S20];
  - ISAC-II 30 MV/m and 60 mT [S4].

### 2.2 Half-wave resonator (HWR)

**Canonical geometry.**
- A coaxial λ/2 line shorted at both ends.
- The beam crosses at mid-length (the voltage maximum), through the outer conductor and the inner
  conductor: two gaps.
- The design is symmetric, so there is no on-axis dipole [S1]. It is "equivalent to 2 QWRs facing each
  other" and needs about twice the power for the same voltage [S2].
- Frequency and β in the sources: 162.5 MHz β 0.10–0.11 (PIP-II [S31], China-ADS HWR010 [S33]); 175 MHz
  β 0.094 (IFMIF [S30]); 322 MHz β 0.29 / 0.53 (FRIB [S6, S7]); 352 MHz β 0.17 / 0.31 (LNL [S74]).

**How designers parametrise it.**
- Ro and L.
- The inner-conductor profile:
  - conical "resonant line" for stiffness and lower Bpk [S30];
  - FRIB second generation: tapered inner conductor with a straight outer tube [S12].
- The inner conductor at the beam port: racetrack-shaped central electrode with the shortest possible
  transition to the round part [S30]; ring-shaped centre conductor (ANL, title only [S72]).
- The beam aperture (FRIB 30 → 40 mm, beam-port flange design [S12, S13]) and the drift-tube region.
- Ports and tuners:
  - coupler port at the mid-plane [S30];
  - tuner: IFMIF's capacitive plunger in the E region was abandoned for wall deformation [S30];
  - FRIB tuner sensitivities 248 kHz/mm (β 0.29) and 101 kHz/mm (β 0.53) [S16].

**Dimensionless ranges.**
- IFMIF [S30]: Ro = 90 mm at λ = 1713 mm.
  - Ro/λ = 0.0525, D/λ = 0.105.
  - rb/Ro = 20/90 = 0.22.
  - D/βλ = 180/161 = 1.12 (derived).
  - The cavity length is not in the extract. L ≲ λ/2 = 857 mm gives Ro/L ≈ 0.10–0.13 (inference).
- FRIB: aperture 40 mm; Leff = βλ = 270 mm (β 0.29) and 503 mm (β 0.53) at λ = 931 mm [S6, S7].
- For larger β the two gap centres must be ≈ βλ/2 apart, so D grows with βλ and Ro/L grows roughly in
  proportion to β (inference).

**Figures of merit.**
- R/Q 148–230 Ω: FRIB 224 / 230 Ω, China-ADS HWR010 148 Ω.
- G 28.5–107 Ω.
- Epk/Eacc 3.5–5.9 and Bpk/Eacc 7.7–11.8 mT/(MV/m) (§3).

### 2.3 Spoke cavities (single, double, triple)

**Canonical geometry.**
- A cylindrical shell whose axis is the beam axis, crossed by 1–3 spokes. Each spoke is a λ/2 TEM line
  shorted at the shell.
- The beam bore runs through the spoke centres.
- The end walls are re-entrant, convex or elliptical ("balloon").
- The accelerating mode is the lowest TEM-like mode, with adjacent spokes in antiphase [S48].
- Spokes are compact and insensitive to vibration [S1].
- Frequency in the sources: 325–352 MHz (345 MHz at ANL).
- β 0.12–0.63 for built cavities; design studies reach 0.82 and 1 [S49, S50].

**How designers parametrise it.**
- Shell diameter D and interior length.
- Spoke base and spoke centre cross-sections:
  - elliptical and racetrack bases give no significant difference in peak fields and shunt impedance
    [S49];
  - one design fixes the base width at 200 mm at 325 MHz, i.e. 0.22 λ; the extract does not say along
    which axis [S49].
- End-wall shape:
  - flat → convex for mechanical reasons (Spoke040 [S34]);
  - elliptical end walls against multipacting (SSR2 [S43]);
  - balloon shell without the central spool (TRIUMF [S44]).
- Aperture.
- For FNAL SSR1 the RF shape was optimised first; stiffeners were added later without changing it [S71
  extract group].

**Dimensionless ranges.**

| design | D/λ | other ratios |
|---|---|---|
| SSR1 [S37] | 0.533 (492 mm, 325 MHz) | rb/Rt = 15/246 = 0.061 |
| SSR2, earlier [S42] | 0.608 (560.8 mm) | Lt/Rt = 537.2/280.4 = 1.92; rb/Rt = 0.089 |
| ESS double spoke [S45] | 0.565 (481 mm, 352.21 MHz) | 994/240.5 = 4.13 (the 994 mm may include beam tubes) |
| ANL TSR630 [S48] | 0.527 (458 mm, 345 MHz) | interior Lt/Rt = 850/229 = 3.71; rb/Rt = 0.087 |
| ODU β 0.82 double spoke [S50] | 0.686 (584.4 mm, 352 MHz) | iris-to-iris / Rt = 3.00; overall / Rt = 3.62; rb/Rt = 0.086 |
| MYRRHA single spoke [S51] | — | wall-to-wall / βλ = 360/315 = 1.14 (first gen) → 420/315 = 1.33 (second gen) |

- MYRRHA: lengthening the cavity lowered Bpk/Eacc from 12.8 to 8.3 mT/(MV/m) [S51].
- Range summary: D/λ ≈ 0.53–0.69 and rb/Rt ≈ 0.06–0.09.
- Lt/Rt ≈ 1.9 for a β ≈ 0.5 single spoke, ≈ 3–4 for double and triple spokes.
- From the length / Leff pairs above, Lt ≈ (1.0–1.35) × (N_gaps − 1)·βλ/2 to N_gaps·βλ/2 (inference).

**Figures of merit.**
- R/Q 142–549 Ω: single spoke 142–276 Ω; double spoke 427–545 Ω; triple spoke 492–549 Ω.
- G 43–133 Ω.
- Epk/Eacc 2.8–5.0. The 5.66 in [S42] looks out of family; verify.
- Bpk/Eacc 5.8–12.8 mT/(MV/m).
- The Leff conventions differ (§2.8).

### 2.4 CH (crossbar H-mode) cavities

**Canonical geometry.**
- An H21-mode multicell drift-tube cavity in which the stems ("crossbars") alternate in orientation.
- It was the first superconducting multicell drift-tube cavity for low and medium energy [S52].

**IAP superconducting CH [S53].**
- 360 MHz, β 0.1, 19 gaps, L = 1048 mm, so L/λ = 1.26 (derived).
- Ep/Ea 5.2; Bp/Ea 5.7 mT/(MV/m).
- Ra/Q ≈ 3180 Ω, transit-time factor included; G ≈ 56 Ω.
- The cavity reached 7 MV/m ↔ 5.6 MV, so Leff = 0.8 m ≈ 19·βλ/2 = 0.79 m (derived) [S52].

**Other CH designs in the extracts.** 325 MHz with 7 gaps; 217 MHz β 0.059 with 15 cells [S52]. Tank
diameter: not obtained.

### 2.5 IH-DTL

**Canonical geometry.**
- An H11-mode structure. Drift tubes sit on stems fixed alternately to two girders.
- It has high shunt impedance at β < 0.1 [S55].

**Designs.**
- HICAT IH-DTL [S54]: 216.816 MHz; 3.76 m; L/D ≈ 11; 4 KONUS sections; 3 internal quadrupole triplets;
  0.4 → 7 MeV; 5.5 MV/m effective.
- That implies D ≈ 0.34 m and D/λ ≈ 0.25 (inference from L/D). This is much slimmer than an Alvarez tank
  (D/λ ≈ 0.6).
- KHIMA [S55]: 200 MHz; 2 tanks; 56 gaps.

### 2.6 Alvarez DTL

**Canonical geometry.**
- The TM010 0-mode tank: drift tubes on stems, cells of length βλ, quadrupoles inside the drift tubes.
  Linac4 [S56] and ESS [S61] use PMQs; J-PARC uses EMQs [S59].
- Post couplers stabilise the field [S56, S59, S61]. Fixed and movable tuners trim it [S59, S61].
- The face angle tunes each cell to frequency (DTLfish) [S62, S67].
- The tank and drift-tube diameters, nose and corner radii and g/L are chosen together [S66].
- Smaller corner radii raise ZTT until the peak field jumps to the bore-nose corner [S63 extract group].
- Smaller stems raise the shunt impedance [S65].
- Frequency in the sources: 324–402.5 MHz. β 0.08–0.41 (3–90 MeV), derived from the energies:
  3 MeV H⁻ → 0.080, 50 MeV → 0.314, 3.62 MeV p → 0.088, 89.9 MeV → 0.409.

**Dimensionless ranges.**
- Tank: D/λ = 0.611 (Linac4 520 mm [S56]; Pramana 520 mm [S63]), 0.605 (J-PARC 560 mm at 324 MHz [S59]),
  ≈ 0.58 (SNS early, > 43 cm at 402.5 MHz [S60]). So Rt/λ ≈ 0.29–0.31.
- Drift tube: d/D = 0.173 (Linac4), 0.25 (J-PARC), 0.269 (Pramana).
- Bore:
  - rb/Rd = 0.22 (Linac4) and 0.143 (Pramana);
  - rb/Rt ≈ 0.038 (Linac4, Pramana);
  - ESS bore radii are 10–12 mm [S61]; its tank diameter was not obtained.
- Radii: corner radius 1.0 cm and inner nose radius 0.25 cm on Rt = 26 cm, i.e. 0.038 and 0.0096 Rt
  (derived, Pramana). KOMAC: outer nose 18 mm (0.26 Rd), corner 1 mm [S64]. These are mostly below our
  5 % floor.
- Cell length: Lc = βλ ⇒ Lc/Rt = β/(Rt/λ) ≈ 3.3 β ≈ 0.26–1.03 (Linac4, derived). For ESS it is
  ≈ 0.29–1.34 if its tank also has Rt/λ ≈ 0.3 (inference).
- Cells per tank: 23–75 drift tubes (ESS 61 / 34 / 29 / 26 / 23 cells; J-PARC tank 1: 75 drift tubes;
  Linac4: 35 / 41 / 29).
- Post couplers: one per drift tube (J-PARC tank 2: 42 for 42 drift tubes) or about every other one
  (J-PARC tank 1: 37 for 75; ESS DTL1: 24 for 60).
- g/βλ: no sourced value was obtained (§6).
- **What DTL designers report:** E0, ZTT (MΩ/m), Emax/Kilpatrick (1.3–1.55) and φs. They do not report
  R/Q or G per cavity.

### 2.7 RFQ (short note)

- **Mode and structure.**
  - A 4-vane cavity operating in a TE210-like quadrupole mode.
  - Two TE110 dipole modes sit just below it and need stabilisation: vane-coupling rings, π-mode loops or
    dipole rods. Undercuts and tuners trim the field [S69 group].
  - The 4-vane suits frequencies of order 200 MHz and above; below that the tank becomes too large and
    4-rod structures are used [S69].
  - A 162.5 MHz window-type variant has larger mode separation without loops [S70].
- **Place in the linac.** The RFQ feeds the DTL: 3 MeV (Linac4 [S56]) and 3.62 MeV (ESS [S61]).
- **Fit for us.** There is no on-axis accelerating TM field, so our R/Q, T and Eacc are not meaningful for
  it. The vane-tip radius r0 is a few % of the tank radius (inference). Not a training candidate (§5).

### 2.8 Conventions designers use (answer to (d))

- **Beam axis.**
  - QWR and HWR: transverse to the coaxial axis, through the gap region (QWR: through the drift tube near
    the open end; HWR: at mid-length) [S1, S2, S30].
  - Spoke, CH and DTL: along the tank axis.
- **β.**
  - Designers quote β_opt, the β that maximises the transit-time factor (FRIB β_opt [S6]; SSR1 β_opt 0.22
    [S38, S39]).
  - Many also quote a "geometric" β_g = 2 × (gap-centre spacing)/λ, which differs:
    - SSR1: 0.186–0.215 [S37–S39];
    - HIE-ISOLDE: β_geom 10.3 % against β_opt 10.9–11.4 % [S24, S25].
  - R/Q and the peak ratios are quoted at β_opt.
- **R/Q.** Linac definition V²/(ωU) where stated ("Ra/Q0, linac definition" [S5]). Most extracts do not
  state it. The CH Ra/Q includes T [S53]. G = Rs·Q0 does not depend on any convention.
- **Eacc = V/Leff.** The Leff definitions found:

  | Leff | where |
  |---|---|
  | βλ | FRIB "Leff ≡ βλ" [S6]; ATLAS leff = βλ [S19]; SPIRAL2 Vacc/βλ [S22]; IFMIF Ngaps·βλ/2 with Ngaps = 2 [S30]; SSR1 ≈ 203–205 mm [S38]; SSR2 0.436 / 0.475 m [S41, S42]; China-ADS 2015 "β0λ" [S33] |
  | N_gaps·βλ/2 | ESS "(n+1)βλ/2" for 3 gaps [S47, S45]; CH 19·βλ/2 (derived [S52, S53]) |
  | (N_gaps − 1)·βλ/2 | ANL triple spoke "L (3βλ/2)" for 4 gaps [S48] |
  | ad hoc | ISAC-II 18 cm (< βλ), "no agreed upon definition" [S23]; HIE-ISOLDE "Lnorm 300 mm" [S25] |

  - The ANL and ESS-type definitions differ by 4/3 for a triple spoke. TSR500 Epk/Eacc = 2.79 would read
    3.72 under N_gaps·βλ/2 (derived).
  - The same China-ADS cavities appear with different numbers in 2013 and 2015 [S32, S33]:
    Spoke012 G 61 vs 43 Ω, Epk/Eacc 4.54 vs 5.0. So published FoMs carry 10–30 % convention or version
    scatter.
- **What designers report.** R/Q [Ω], G [Ω], Epk/Eacc, Bpk/Eacc [mT/(MV/m)], sometimes U/Eacc² [J/(MV/m)²]
  (ISAC-II), Va per cavity [MV], Leff [mm], aperture [mm], β_opt and β_g, Q0 at a stated Rs, and the
  peak-field limits.
- **Contrast with our labels [R2].**
  - Our labels use β = 1, L_acc = the axis chord inside Ω including the beam pipes, and axis z (x for
    hwr).
  - Two-gap estimate: gaps at ±V_g spaced β_opt·λ/2 give V(β)/V(β_opt) = |sin(π β_opt/(2β))| (inference,
    thin-gap estimate). At β = 1:
    - β_opt = 0.1 → V ratio 0.156 → R/Q ratio 0.024;
    - β_opt = 0.3 → V ratio 0.45 → R/Q ratio 0.21.
  - So our R/Q, T and Epk/Eacc labels for hwr / spoke / dtl are not comparable with published values. On
    the lowest-β samples they may also fall under the `rq_floor` (1 %) filter of the QoI metrics (docs/24
    §4; inference; check).

## 3. Reference designs table

Ratios are derived from the dimensions given; "—" means not obtained.
**Rebuild:**
- **N:** too little geometry in the extract.
- **P:** partial — enough for an approximate anchor once the missing dimensions are found in the full
  text.

**No design in the extracts is specified completely enough to rebuild exactly.**

| # | design | f [MHz] | β | geometry (abs → ratio) | R/Q [Ω] | G [Ω] | Epk/Eacc | Bpk/Eacc [mT/(MV/m)] | Leff | Q0 | rebuild | src |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FRIB QWR β0.041 | 80.5 | 0.041 (0.043 [S9]) | aperture 34 mm (30 [S5], 36 [S8]) | 402 (Ra/Q0 433, linac, earlier [S5]) | 15 (15.3) | 6.1 (derived: Ep 31, Va 0.81 MV) | 10.9 (derived: Bp 55) | βλ = 160 mm | 5e8 design [S5] | N | S5, S6, S8, S9 |
| 2 | FRIB QWR β0.085 | 80.5 | 0.085 (0.086) | aperture 34 / 36 mm | 452 (455) | 22 (22.3) | 5.87 (derived: Ep 33, Va 1.8 MV); 5.9 in a FRIB table (extract unclear: SRF2011 MOPO031 / MOPO049) | 12.4 (derived: Bp 70); 12.6 in the same table | βλ = 320 mm | — | N | S6, S8, S10 |
| 3 | FRIB HWR β0.29 | 322 | 0.29 | aperture 40 mm (30 in 2nd gen [S12]) | 224 (≈ 196, earlier geometry [S11]) | 78 | 4.26 (derived: Va 1.9 MV, Ep 30 [S14]); ≈ 4.8 earlier [S11] | 7.67 (derived: Bp 54); ≈ 7.7 earlier | βλ = 270 mm | — | N | S6, S7, S11, S12, S14 |
| 4 | FRIB HWR β0.53 | 322 | 0.53 | aperture 40 mm | 230 | 107 | 3.53 (derived: Ep 26, Va 3.7 MV); production 4.3 → 3.6 [S16] | 8.56 (derived: Bp 63) | βλ = 503 mm | — | N | S6, S7, S16 |
| 5 | ISAC-II medium-β QWR | 106.08 | 0.072 | ID 180 mm (D/λ 0.064); bore Ø 20 (rb/Ro 0.11); gap 40 mm | 540 (derived from U/Ea² = 0.09 J and V = 0.18 MV, linac) | 19.1 | 4.6 | 10.3 | 18 cm (0.885 βλ) | — | P | S23 |
| 6 | HIE-ISOLDE high-β QWR | 101.28 | β_g 0.103 / β_opt 0.109–0.114 | "300 mm", "950 mm" labels; racetrack aperture option [S26] | 553 / 554 | 30.34 | 5.0 [S24] / 5.4 [S25] | 9.6 | Lnorm 300 mm | — | N | S24–S26 |
| 7 | TRIUMF high-β QWR | 141.4 | 0.112 | — | — | — | 4.9 | 9.9 | — | — | N | S25 |
| 8 | SPIRAL2 QWR β0.07 | 88.05 | 0.07 | external Ø 220 (0.065); beam tube 30 mm (rb/Ro 0.14) | 639 [S20] / 632 [S21] | 22 (derived: Q0 2.2e9 at Rs 10 nΩ) | 5.5 / 5.0 | 8.9 / 8.75 | Vacc/βλ | 2.2e9 @ 10 nΩ | P | S20–S22 |
| 9 | SPIRAL2 QWR β0.12 | 88.05 | 0.12 | external Ø 370 (0.109); beam tube 50 mm (36 mm in the 2005 prototype) | 507 | — | 4.9 | 9.0 | βλ | — | P | S20, S21 |
| 10 | MSU/RIA QWRs | 80.5 / 161 | 0.085 / 0.16 | outer conductor Ø 240 (0.064 / 0.129); aperture 30 mm | — | — | — | — | — | β0.16: 2.5e9 at Ep 20 MV/m (test) | N | S18 |
| 11 | ATLAS QWR | 72.75 | 0.077 | tapered inner and outer conductors | — | — | ≈ 5.0 (derived: 117 MV/m at > 7.4 MV, βλ = 317 mm) | ≈ 7.0 (derived: 164 mT) | βλ | — | N | S19 |
| 12 | QWR prototype (lab not named) | — | ≈ 0.08 | — | 587 (Rsh/Q0) | — | 6.2 | 9.7 | — | — | N | S29 |
| 13 | IFMIF HWR | 175 | 0.094 | Ro 90 (Ro/λ 0.0525); aperture radius 20 (rb/Ro 0.22); racetrack centre electrode; conical line | — | — | 4.42 | 10.12 | Ngaps·βλ/2 = 161 mm | — | P | S30 |
| 14 | China-ADS HWR010 | 162.5 | 0.10 | aperture 40 mm | 148 | 28.5 | 5.9 | 11.8 | β0λ | — | N | S33 |
| 15 | PIP-II HWR | 162.5 | 0.11 | — | — | — | — | — | — | test > 3.5 MV; 134 MV/m, 144 mT | N | S31 |
| 16 | PIP-II SSR1 | 325 | β_g 0.21 (0.186, 0.215); β_opt 0.22 | shell ID 492 (0.533); aperture 30 (rb/Rt 0.061) | 242 | 84 | 3.84 (derived: 38.4 MV/m at 10 MV/m) | 5.81 (derived) | βλ ≈ 203–205 mm | — | P | S37–S40 |
| 17 | PIP-II SSR2 v3.1 | 325 | β_opt 0.47 | aperture 40 mm; elliptical end walls [S43] | — | — | 3.46 (CST) / 3.51 (COMSOL) | 6.79 / 6.75 | 0.436 m (= βλ) | — | N | S41, S43 |
| 18 | SSR2, earlier version | 325 | β_opt 0.514 | D 560.8 (0.608); L 537.2 (Lt/Rt 1.92); aperture 50 (rb/Rt 0.089) | 275.9 | 118.6 | 5.66 (as extracted; check) | 6.25 | 475.3 mm (= βλ) | — | P | S42 |
| 19 | China-ADS Spoke012 | 325 | 0.12 | aperture 35 mm | 142 / 150 | 61 / 43 | 4.54 / 5.0 | 6.37 / 6.9 | — / β0λ | — | N | S32, S33 |
| 20 | China-ADS Spoke021 | 325 | 0.21 | aperture 40 mm | 206 / 191 | 87 / 71 | 3.88 / 4.4 | 8.13 / 9.4 | — / β0λ | — | N | S32, S33 |
| 21 | China-ADS Spoke040 | 325 | 0.40 | aperture 50 mm; convex end wall | 244 | 104 | 3.30 | 8.34 | — | — | N | S32, S34 |
| 22 | MYRRHA single spoke | 352.2 | β_g 0.35, β 0.37 | wall-to-wall ≈ 36 → 42 cm (1.14 → 1.33 βλ) | — | — | 4.7 → 4.4 | 12.8 → 8.3 | — | — | N | S51 |
| 23 | ESS double spoke | 352.21 | 0.50 | ID 481 (0.565); length 994 mm (perhaps flange to flange); 3 gaps | 427 | 133 | 4.3 | 6.8 | 3βλ/2 = 0.639 m | — | P | S45–S47 |
| 24 | C-ADS double spoke | 325 | 0.37 | — | — | — | 4.39 | 7.15 | — | — | N | S36 |
| 25 | ANL TSR500 | 345 | 0.50 | aperture Ø 40; interior length 67 cm | 492 | 88.5 | 2.79 | 8.6 | 3βλ/2 = 65.2 cm (4 gaps) | — | P | S48 |
| 26 | ANL TSR630 | 345 | 0.63 | D 45.8 cm (0.527); interior 85 cm (Lt/Rt 3.71); shell OD 46.6; overall 101.6 cm; aperture Ø 40 | 549 | 93 | 2.93 | 9.0 | 3βλ/2 = 82 cm | — | P (best spoke anchor) | S48 |
| 27 | ODU double spoke | 352 | 0.82 | D 584.4 (0.686); iris-to-iris 877 (3.00 Rt); overall 1057; aperture 50 | 545 | — | — | — | — | — | P | S50 |
| 28 | IAP superconducting CH | 360 | 0.10 | 19 gaps; L 1048 mm (L/λ 1.26) | Ra/Q ≈ 3180 (T included) | ≈ 56 | 5.2 | 5.7 | ≈ 19·βλ/2 = 0.8 m (derived) | BCS 1.5e9 @ 4.2 K | N | S52, S53 |
| 29 | Linac4 DTL | 352.2 | 0.080–0.314 | D 520 (0.611); d 90 (d/D 0.173); aperture Ø 20 (rb/Rd 0.22); 35 / 41 / 29 drift tubes; post couplers | — | — | — | — | E0 3.1–3.3 MV/m | — | P (gaps missing) | S56, S57 |
| 30 | J-PARC DTL | 324 | — | ID 560 (0.605); d 140 (0.25); tank 1: 75 drift tubes, 37 post couplers, 10 fixed tuners | — | — | — | — | E0 2.5–2.9 MV/m | — | N | S59 |
| 31 | SNS DTL | 402.5 | — | D > 43 cm (≈ 0.58, early); 6 tanks, 210 drift tubes | — | — | Epk ≤ 1.3 Kp (Kp = 19.47 MV/m) | — | — | — | N | S60 |
| 32 | ESS DTL | 352.21 | 0.088–0.409 | bore radius 10–12 mm; 61 / 34 / 29 / 26 / 23 cells; DTL1: 24 post couplers, 23 + 3 tuners | — | — | Emax/Ek 1.55 | — | E0 3.0–3.16 MV/m | — | N | S61, S62 |
| 33 | Pramana DTL cell | 352.2 | (10 MeV) | D 52.0 cm; d 14.0 (0.269); rb 1.0 cm (0.143 Rd, 0.038 Rt); corner radius 1.0 cm; inner nose radius 0.25 cm | — | — | — | — | — | — | P (no g/L, face angle) | S63 |
| 34 | KOMAC DTL | — | — | d 140 mm; outer nose radius 18 mm; corner 1 mm; flat 1 mm; "face angle 70°" | — | — | — | — | — | — | N | S64 |
| 35 | HICAT IH-DTL | 216.816 | p 0.4–7 MeV | L 3.76 m; L/D ≈ 11 | — | — | — | — | 5.5 MV/m effective | — | N | S54 |

**Closest to validation geometries.**
- **ANL TSR630:** D, interior length, shell OD, overall length and aperture are given; spoke shape and end
  walls are missing.
- **IFMIF HWR:** Ro, aperture and βλ are given; length and inner-conductor profile are missing.
- **SSR1 / old SSR2:** D, length (SSR2), aperture and Leff are given; spoke dimensions are missing.
- **Linac4 / Pramana DTL cells:** D, d, bore and nose radii are given; g/L, cell length and face angle are
  missing.

All need the full texts: open-access PRST-AB papers, the MSU thesis [S15], the ANL PAC05 papers and the
Superfish cell tables in the DTL design reports.

## 4. Comparison with our generator

Our families are scale-free (the model sees x/scale), so only the shape ratios below matter. Mesh costs
are from [R1].

### 4.1 `hwr` — Ro 3–6 cm, L 8–20 cm, ri/Ro 0.2–0.5, 50 % taper (rm/ri 0.6–1.5), 50 % transverse port

**Our ranges.**
- **Ro/L** = 0.15–0.75. The TEM relation f ≈ c/2L gives 0.77–1.58 GHz [R3].
- **Bore.** rb = (0.45–0.7)·min(ri, rm), so rb/Ro ≈ 0.07–0.35.
- **Beam port.** At z = L/2 along x; the QoI axis follows it [R2].

**Covered.**
- ri/Ro brackets the TEM optimum 0.37 (inference).
- The IFMIF bore ratio (0.22) lies inside our range.
- The taper covers both senses: thick-at-ends conical lines (IFMIF [S30]) and tapered FRIB inner conductors
  [S12].
- The beam crosses at mid-length, as in all real HWRs.
- A rough two-gap estimate puts our β_opt at ≈ (Ro + ri)/L ≈ 0.2–1.0 (inference). So the family looks like
  medium- and high-β HWRs (FRIB β 0.29–0.53).

**Missed.**
- **Slender low-β HWRs.** At β ≈ 0.1, Ro/L ≈ 0.1 (IFMIF; also PIP-II and China-ADS HWR010 at 162.5 MHz).
  This is the most numerous HWR class in the sources; we sit at or above its upper edge (inference).
- **Shaped inner conductor at the beam port.** Racetrack centre electrode with a short transition (IFMIF
  [S30]); ring-shaped or donut centre conductor (ANL [S72], title only). Our round inner conductor with a
  round bore gives curved gap faces.
- **Straight vs tapered conductors.** FRIB keeps a straight outer tube and tapers the inner one [S12], which
  we cover. ATLAS-type outer-conductor tapers are a QWR feature [S19].
- **Ancillaries.** Coupler port at the mid-plane [S30] and tuners (plunger or end-wall deformation [S30,
  S16]) are absent.

**QoI.** Correct axis; wrong β (1 instead of β_opt) and a different L_acc: the chord ≈ 2Ro plus the pipes
against βλ (§2.8).

### 4.2 `spoke` — Rt 5–10 cm; 1 spoke (Lt/Rt 0.9–1.4) or 2 spokes (1.5–2.2); uniform elliptic spoke

**Our spoke cross-section.** Half-width along the beam az/Rt = 0.15–0.28. Across the beam, ay/az = 1.0–1.6
(wider across the beam when bored).

**Our bore.**
- 60 % of samples: a bore along z through the tank and every spoke, rp/Rt ≈ 0.07–0.23.
- 40 % of samples: pipe stubs only, so the beam line through the gaps is blocked by the spoke.

**Our end walls.** Flat end plates. Only re-entrant edges are rounded [R4].

**Covered.**
- Single-spoke proportions of β ≈ 0.2–0.3 cavities such as SSR1, Spoke021 and the Korean SSR1 [S37, S32,
  S71]. With D/λ ≈ 0.55, Lt/Rt ≈ 0.9–1.4 maps to β_opt ≈ 0.25–0.39 (inference).
- The bored variant, and parallel or 90°-rotated second spokes. A rotated pair is CH-like.

**Missed.**
- **Tank length at higher β.** β ≥ 0.4 single spokes have Lt/Rt ≈ 1.9 (SSR2 [S42]; MYRRHA 1.3 βλ
  [S51]). β ≈ 0.5 double spokes have Lt/Rt ≈ 3–4 (ESS [S45], ODU [S50]). Triple spokes have 3.7 (TSR630
  [S48]). Our two-spoke range stops at 2.2.
- **End walls.** Re-entrant, convex, elliptical or balloon end walls (Spoke040 [S34], SSR2 [S43], TRIUMF
  [S44]). Our flat plates give a less loaded spoke mode. Our family gives D/λ ≈ 0.50 [R3] against real
  0.53–0.69 (inference): real designs carry more capacitive loading.
- **Spoke taper.** Real spokes are not uniform: the base (racetrack or ellipse [S49]; one design fixes the
  base width at 0.22 λ) differs from the centre around the bore. Our elliptic cylinder is uniform across
  the full diameter.
- **Junction blends.** Real spoke-to-shell junctions are large blends; we have only the automatic
  re-entrant rounding at the 5 % floor.
- **Small bores.** Real rb/Rt is 0.06–0.09; ours is 0.07–0.23, at or above the top of the real range.
- **Triple spokes and CH-style 3–4 crossbar sections** are absent.

**QoI.** Correct axis (z); β = 1 instead of β_opt ≈ 0.25–0.4 (inference). The no-bore samples have no beam
path, so their "R/Q" is only a defined functional.

### 4.3 `dtl` — Rt 5–10 cm, 1–3 drift tubes, equal cells

**Our ranges.**
- Cells: Lc/Rt 0.5–0.9.
- Drift tubes: Rd/Rt 0.2–0.3; rb/Rd 0.35–0.5; ld/Lc 0.35–0.6, i.e. g/Lc 0.4–0.65.
- Shape details: rounded noses at the 5 % floor; stems along +y with rs/Rt ≈ 0.03–0.16; beam pipes at
  both ends.
- Mesh cost: 15–47k curved tets; 2 of 4 draws exceed 30k [R1].

**Covered.**
- Rd/Rt matches J-PARC (0.25) and Pramana (0.27). Linac4 (0.173) lies just below our range.
- Cells of β ≈ 0.15–0.27, if our tanks have Rt/λ ≈ 0.3 like real ones (inference).

**Missed.**
- **Low-β cells.** Lc/Rt 0.26–0.5 for β 0.08–0.15. These are most of the real cells: ESS tank 1 has 61
  cells from 3.62 to 21 MeV [S61].
- **High-β cells.** Lc/Rt 0.9–1.36.
- **Small bores.** Real rb/Rd is 0.14–0.22 and rb/Rt ≈ 0.04 [S56, S63]; ours is 0.35–0.5. The real bore
  radius (≈ 0.04 Rt) is below our fillet floor (0.05 Rt), so the real bore cannot be drawn with a floored
  edge fillet.
- **Sharp noses.** Real corner and nose radii are 0.4–4 % of Rt, against our ≥ 5 %. Our Epk is therefore
  probably low and T different (inference).
- **Real tank ends.** Real end walls carry half drift tubes. Without them our end cells are pillbox-like
  and the 0-mode field is not flat (inference).
- **Face angle.** It is the per-cell tuning knob in real tanks [S62, S67].
- **Cell-length ramp.** Real Lc = βλ grows along the tank. Inference from ESS tank 1 (β 0.088 → 0.21 over
  61 cells): ≈ 1.5 % per cell.
- **Field stabilisation and tuning.** Post couplers (rods from the wall, 0.5–1 per drift tube [S59, S61])
  and slug tuners [S59, S61] are absent.
- **Tank length.** Real tanks hold 23–75 drift tubes; ours 1–3.
- **Stems.** Ours are thicker than real stems [S65].

**QoI.** Correct axis; β = 1 instead of Lc/λ ≈ 0.15–0.3 (inference); the chord L_acc includes the pipes.

### 4.4 `coax_qw` (OOD)

**Our geometry.** Ro 2–5 cm; L/Ro 1.5–4, i.e. H/D = 0.75–2.0; ri/Ro 0.2–0.45; axial tip gap/L 0.08–0.3;
sharp edges; no beam port. The QoI axis defaults to z, along the coaxial axis. That is physically
meaningless for a QWR: the beam crosses transversally.

**Real QWRs.**
- H/D ≈ 2–5 (upper-bound estimate from D/λ ≈ 0.05–0.13 and H ≲ λ/4, inference).
- A transverse beam through a drift tube at the tip; rb/Ro ≈ 0.11–0.14.
- Tapered conductors (ATLAS [S19]) and steering corrections [S26, S28].

So `coax_qw` is a deliberately unrealistic probe, which is fine for OOD, but no realistic QWR exists in
training.

### 4.5 Types we do not model

- **Realistic QWR:** see §5 H2.
- **CH:** H21 mode with alternating crossbars and drift tubes. Our 90°-rotated two-spoke draw is the
  nearest relative.
- **IH:** two girders with alternating stems. The ridged box is only a distant relative.
- **DTL unit cell:** the standard Superfish-type design object [S67].
- **RFQ.**

## 5. Recommendations

Cost labels, under our constraints (curved mesh ≤ 30k tets, fillet ≥ 5 % of size, closed PEC):
- **cheap:** no new feature below the current smallest; tets similar to the current family.
- **costly:** new features near the fillet floor or more gaps; tet counts near 30k, so more re-draws and a
  bias towards easy draws.
- **infeasible:** over 30k by construction.

Tet anchors [R1]: spoke 5–8k, hwr 9–27k, dtl 15–47k. Every generator change needs the owner's go-ahead
(AGENTS.md).

### High priority

**H1. Use designer conventions for the low-β QoI. Cheap: post-processing; the axis operator already takes
any axis.**
- **Change:**
  - For hwr, spoke, dtl and a new qwr, compute V(β) = |∫E_z e^{iωz/(βc)} dz| on a β grid (e.g. 0.02–1)
    and store β_opt = argmax V(β). The maximum of V and of T coincide, because ∫|E_z| does not depend
    on β.
  - Report R/Q = V(β_opt)²/(ωU) (linac) and T(β_opt).
  - Report Eacc = V/Leff with Leff = β_opt λ for two-gap structures (QWR, HWR, single spoke) and
    N_gaps·β_opt λ/2 otherwise. Store the convention in the metadata.
  - Store the convention-free values too: V, U, Epk/V and Bpk/V. Then the ANL (N_gaps − 1)·βλ/2, ISAC-II
    or β_g conventions can be reproduced later.
  - The QWR axis must be transverse through the drift-tube centre. The builder exports z_b and
    `beam_axis()` reads it.
- **Benefit:**
  - R/Q, T and the peak ratios become comparable with every published design in §3.
  - It avoids the β = 1 cancellation: on two-gap β_opt ≈ 0.1 structures, R/Q drops to ≈ 2–3 % of the
    design value (inference).
  - The low-β samples may then stop falling under `rq_floor` (to check).
- **Sources:** S6, S19, S22, S23, S30, S33, S45–S48.

**H2. Move a realistic QWR into training as a new `qwr` family; keep the sharp `coax_qw` as OOD. Moderate
cost (comparable with hwr ported: 9–27k tets).**
- **Parameters:**
  - Height and conductors: H/D ≈ 1.5–3.5 at first (real ≈ 2–5, upper-bound estimate); ri/Ro 0.25–0.45.
    Measure the tet counts, then widen.
  - Tip: an inner-conductor end ("drift tube"), straight or enlarged (rdt/ri 1.0–1.5, inference); a tip
    to bottom-plate distance.
  - Beam port: a transverse beam port through the outer wall and the drift tube, rb/Ro 0.1–0.15 [S18, S20,
    S23], with PEC-capped pipes (b1 = 1).
  - Optional: inner and/or outer conductor taper (ATLAS [S19]); fillets at the floor.
- **Benefit:**
  - Covers the most common low-β superconducting structure: QWR linacs at FRIB, ATLAS, ISAC-II, SPIRAL2,
    HIE-ISOLDE, Legnaro and New Delhi [S1, S3, S10]. FRIB alone uses ≈ 100 β = 0.085 QWRs [S10].
  - Its physics differs from the HWR: asymmetric fields and steering [S1, S27]; E concentrated at the
    tip; B at the shorted top.
  - `coax_qw` stays a clean generalisation test: sharp edges, extreme ratios, no port.
- **Cost:** the long annulus and the small bore features. Expect more tets than an unported hwr; risk of
  re-draw bias above H/D ≈ 3 (inference).

**H3. Make the spoke family realistic. Cheap to moderate: the family has a 4× tet margin.**
- **(a) End walls.** Re-entrant noses around the bore, or convex / elliptical end walls [S34, S43, S44].
  The `reentrant` nose-cone builder can be reused. Cheap.
- **(b) Tank length.** Extend Lt/Rt to 0.9–2.0 for one spoke (SSR2 1.92) and 1.5–3.2 for two (ODU 3.0;
  ESS ≈ 3–4). Moderate: longer tanks mean more tets.
- **(c) Spoke taper.** Loft from a base ellipse or racetrack to a centre section around the bore [S49].
  Cheap in tets, moderate in CAD (OCC ThruSections). The spoke–shell junctions remain re-entrant edges
  for `round_reentrant_edges`.
- **(d) Bore.** Lower the bore bound to rp/Rt ≈ 0.06 [S37, S48, S50]. Moderate: near the floor.
- **Benefit:** the β ≈ 0.4–0.6 single, double and triple spoke class (PIP-II SSR2, ESS, MYRRHA, ANL),
  with realistic loading (D/λ 0.53–0.69 instead of ≈ 0.50).

**H4. DTL realism where the budget allows.**
- **(a) Half drift tubes on the end walls.** They replace our pillbox end cells; a flat 0-mode results.
  Cheap.
- **(b) A `dtl_cell` unit-cell family.** One cell, PEC end walls at the drift-tube mid-planes with half
  drift tubes, optional half stem.
  - For the 0-mode this is exact, since tangential E vanishes on both symmetry planes (inference from
    symmetry). It is the Superfish design object [S67] and can be compared with published cell tables.
  - Cheap: one gap, b1 = 0 without stems.
- **(c) Ranges.** Rd/Rt down to 0.15 (Linac4 0.173, cheap). Lc/Rt down to ≈ 0.3 in the unit cell only:
  narrow gaps cost tets, which the unit cell can afford and the 3-tube tank cannot.
- **Leave out:**
  - **Tanks beyond 4 cells:** infeasible. Two of the four 4-cell draws already exceed 30k tets [R1].
  - **Real bore sizes (rb ≈ 0.04 Rt):** below the fillet floor (0.05 Rt). Our 0.35–0.5 Rd could go down to
    ≈ 0.25 Rd at most (costly).
- **Sources:** S56, S59–S63, S67.

### Medium priority

**M1. HWR: low-β proportions and the beam-port shape. Moderate to costly.**
- Extend Ro/L down to ≈ 0.1 (IFMIF-like β ≈ 0.1 [S30]). Costly: the slender annulus raises tets in a family
  already at 9–27k.
- Add an optional racetrack (flattened) inner-conductor section around the beam port with a short blend
  [S30]: cheap in CAD, moderate in tets. Optionally a ring-shaped variant [S72].
- Optional mid-plane coupler stub and plunger tuner in the E region [S30]: plain cylinders, cheap.

**M2. Triple spoke and CH-like sections. Moderate to costly.**
- Three spokes (TSR-like, Lt/Rt 3.5–4 [S48]).
- 3–4 crossbars alternating by 90°, with short drift tubes around the bore (CH-like [S52, S53]).
- Covers multi-gap H-mode physics with few gaps; a full 7–19-gap CH is infeasible.

**M3. DTL details. Cheap.**
- A conical face angle in the revolved drift-tube profile [S62, S67].
- A cell-length ramp of 0–5 % per cell (inference from ESS).
- 0–2 slug tuners as short wall cylinders [S59, S61].

**M4. Coupler and pickup stubs on the outer walls of TEM cavities. Cheap.** Plain tubes as in the
elliptical family: mid-plane coupler on HWRs [S30], pickups on QWRs and spokes. They break the symmetries
and add realistic port perturbations.

**M5. Approximate real-design anchors. Cheap compute; needs the full texts.**
- Once the full texts are reachable, add perturbed anchors like the elliptical TESLA anchor.
- Candidates: TSR630 [S48], SSR1 / SSR2 [S37, S42], IFMIF HWR [S30], ISAC-II / SPIRAL2 QWRs [S20, S23],
  the Linac4 / Pramana DTL cell [S56, S63].
- Use them as validation geometries against published G and Epk/Eacc, matching conventions as in H1.

### Low priority / deliberately leave out

- **L1. Post couplers in 2–4-cell DTL tanks. Costly.** Thin rods with small tip gaps, in a family already
  at 15–47k tets. Their purpose, field stabilisation along long tanks [S56, S59, S61], does not apply to
  2–4 cells. They would add realistic extra modes near TM010. At most a single-post-coupler OOD set.
- **L2. IH-type structures. Moderate to costly.** Girders with alternating stems on 3–5 drift tubes [S54,
  S55]. Real L/D ≈ 11 is infeasible; a short section would be a synthetic diversity family.
- **L3. QWR steering corrections. Cheap, small effect, good as OOD.** Drift-tube face tilt 0.5–1.5° [S28];
  racetrack aperture [S26].
- **L4. RFQ. Leave out of training; at most an unmodulated 4-vane OOD section.** Costly: vane tips of a
  few % of the tank radius sit near the floor (inference). Its quadrupole mode has no on-axis
  accelerating field, so our R/Q, T and Eacc do not apply [S68, S69].
- **Leave out entirely:**
  - full-length DTL tanks (23–75 cells), CH tanks (7–19 gaps) and IH tanks (L/D ≈ 11): infeasible;
  - split-ring resonators: historic, and complex for little benefit.

## 6. Open questions and what I could not verify

1. **Access.** No full text was read (§0). All numbers are from search extracts. Some table columns may be
   misaligned in the extract: SPIRAL2 Table 2, the CH table, the old SSR2 Epk/Eacc 5.66, KOMAC's "45 mm
   inner nose radius" and "70° face angle".
2. **Not looked up** (search budget exhausted):
   - SARAF HWRs (176 MHz);
   - RAON QWR / HWR. An 81.25 MHz β 0.047 QWR with Ep/Ea 5.3, Bp/Ea 9.5 and R/Q 470 Ω appeared in one
     extract without a laboratory or a paper I could pin down, so it is left out of §3;
   - ALPI and HIE-ISOLDE low-β cavities;
   - the ATLAS 109 MHz β 0.15 QWR;
   - the Jülich triple spoke;
   - the MYRRHA CH injector;
   - FAIR p-linac CH and CCH;
   - GSI HELIAC;
   - Fermilab 201 MHz DTL dimensions.
3. **QWR geometry:** heights, inner-conductor diameters, tip-to-plate distances and drift-tube shapes. None
   came through. The H/D ≈ 2–5 and ri/Ro ≈ 0.37 statements are estimates (inference). The HIE-ISOLDE
   "300 mm / 950 mm" labels may be the diameter and height, but the extract does not say.
4. **HWR geometry:** cavity lengths (IFMIF, FRIB, PIP-II); FRIB HWR diameters; the Va of the β = 0.29
   cavity in the 2012 table. The derived 4.26 / 7.67 uses the SRF2011 design-for-manufacture numbers
   [S14], whose paper ID is unclear.
5. **Spoke geometry:**
   - base and centre widths and their axes;
   - end-wall nose depths;
   - SSR1 cavity length;
   - whether the ESS 994 mm is the interior or the flange-to-flange length.
   The full Hopper & Delayen PRST-AB paper [S49] (open access) has the parametrisation and is the first
   thing to read once access is restored.
6. **DTL cell data:**
   - g/βλ and face angles per cell (Linac4, ESS, J-PARC, SNS);
   - stem diameters;
   - post-coupler diameters and azimuths;
   - the ESS tank diameter.
   My expectation that real low-β g/L lies below our 0.4–0.65 is prior knowledge, not verified here.
7. **Conventions.** Several sources do not state the R/Q convention (V²/ωU or half of it). The China-ADS
   2013 and 2015 tables differ by up to 30 % for the same cavities [S32, S33]. FRIB QWR β is 0.041 or
   0.043 and 0.085 or 0.086, depending on the source [S6, S9].
8. **Check in our own labels:**
   - whether the hwr, spoke and dtl modes are filtered by `rq_floor` at β = 1 (inference in §2.8);
   - what β_opt our families actually span: compute V(β) on existing labels before changing ranges.
9. **Unattributed statements kept out of the numbers:**
   - a "160 MHz, β 0.11 cylindrical QWR vs cone HWR" comparison: Epk/Eacc 5.09 vs 4.85, Bpk/Eacc 11.41 vs
     7.25;
   - "spoke A/B ratio controls Epk/Eacc and C/D controls Bpk/Eacc".
   Both appeared in extracts without a source I could identify; they are consistent with the trends in §3
   but unverified.
