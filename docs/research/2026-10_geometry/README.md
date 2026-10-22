# Geometry literature review, 2026-10 — raw material of docs/33

Four research agents compared the generator's geometry families (`src/data_gen/cavity_shapes.py`, branch
`claude/field-model`) with published RF cavity designs. The synthesis, the proposals and the owner's open
decisions are in [docs/33_GEOMETRY_REVIEW.md](../../33_GEOMETRY_REVIEW.md). This folder keeps the raw reports and
the scripts behind the computed numbers, so that every number in docs/33 can be traced and re-run.

## Access limits (read before citing anything)

- The sandbox's network policy blocked every literature host: arxiv.org, jacow.org, proceedings.jacow.org,
  accelconf.web.cern.ch, epaper.kek.jp, osti.gov, cds.cern.ch, journals.aps.org, inspirehep.net and
  semanticscholar.org (proxy 403; WebFetch failed with DNS errors).
- No full text was read. Every literature number comes from a search-engine excerpt of the cited document, and
  each report tags its numbers by provenance:
  - excerpt;
  - abstract only;
  - open-source code;
  - derived;
  - estimate;
  - our repository.
- Check a number against the full text before using it as a target or a validation reference.
- Not affected by the access limits:
  - our own code and distributions;
  - the axisymmetric solves of report a3 (validated against the analytic pillbox, below).

## Reports

| file | area | families |
|---|---|---|
| `a1_elliptical.md` | elliptical SRF cavities (β 0.5–1), end groups, couplers | elliptical, elliptical_long, pillbox_pipes, pillbox_port |
| `a2_lowbeta.md` | low-β TEM cavities: QWR, HWR, spoke, CH / IH, DTL | hwr, spoke, dtl, coax_qw |
| `a3_nc_special.md` | normal-conducting cavities and structures, RF guns, crab cavities, HOM damping | reentrant, pillbox_pipes, ridged_box, composite, box, junction, pillbox_port |
| `a3_search_extracts.md` | raw search extracts behind a3 (provenance) | — |
| `a4_ml_methods.md` | ML surrogates, shape optimisation, required accuracy, fabrication errors, FoM conventions | all |

The reports are kept as written, in English. Paths in them that point to the session's scratch directory refer
to the scripts now in `scripts/`.

## Scripts

Run them from the repository root. They read the generator's code but change nothing. None of them writes into
the repository: `a3_family_stats.py` writes its JSON table to the current directory, and that output is data, so
do not commit it.

| script | what it computes | docs/33 | runtime |
|---|---|---|---|
| `chord_inflation.py` | stored acceleration length (axis chord, pipes included) / standard length, with the generator's own samplers | §2.2 | 7.5 min |
| `elliptical_dist.py` | elliptical mid-cell ratios, wall angle, anchor and low-β shares, rejection rate of random half-cells | §3.1 | 54 s |
| `deform_mag.py` | displacement of the smooth deformation map on a TESLA-size cell, for several `deform_max` | §5 | 10 s |
| `a1_halfcell.py` | standalone copy of the half-cell construction, plus the wall angle (imported by the other `a1_` scripts) | — | — |
| `a1_ratios.py` | published mid and end cells: ratios, wall angle, outcome of our validity test | §3.1 | 1 s |
| `a1_derived.py` | end/mid ratios, iris-tip radii, passband spacing, pipe cut-offs, field left at the caps, capped-coax estimate | §3.1, §3.9 | < 1 s |
| `a1_mc_generator.py` | acceptance and wall-angle distribution of the sampler and of the perturbed anchors | §3.1 | 2 min |
| `a1_notangent3.py`, `a1_solverfix.py` | the tangent-solver tolerance finding (multi-start check of the "no tangent" rejections) | §3.1 | 5 min each |
| `a1_widened.py` | acceptance and wall-angle spread: current vs widened ranges, current vs proposed rules | §3.1 | 2 min |
| `a3_axisym.py` | axisymmetric TM0 eigen-solver (NGSolve, H_φ formulation) for the generator's reentrant / pillbox_pipes profiles; run alone: the analytic pillbox check | §3.2, §3.3 | 1.3 s |
| `a3_family_stats.py` | R/Q, G, T and shape ratios of `n` draws of a family: `python …/a3_family_stats.py reentrant 400 0` | §3.2, §3.3 | ~0.2 s per draw |
| `a3_scan.py` | reentrant profile scan outside today's ranges (storage-ring region, capacity-loaded cells) | §3.3 | 8 s |
| `a3_slac.py` | SLAC 2π/3 cell, three periods between electric walls | §8 | 1.6 s |
| `a3_gun.py` | 1.6-cell BNL/SLAC/UCLA gun rebuilt from a table excerpt (iris thickness assumed) | §8 | 3 s |

The `a1_` scripts and `deform_mag.py` are standalone copies of the generator's code and import nothing from the
repository; the other scripts import `src.data_gen.cavity_shapes`. The `a1_` scripts were renamed from the names
the report uses (`halfcell.py` → `a1_halfcell.py`, and so on).

Check runs in this session, after the scripts were copied here:

| script | result | reference |
|---|---|---|
| `a3_axisym.py` | pillbox f +3.4·10⁻¹¹, R/Q +3.1·10⁻⁴, G −4.6·10⁻¹⁰ | analytic values |
| `a3_slac.py` | 2855.2 MHz at 2π/3 | published 2856 MHz |
| `a3_gun.py` | 2859.1 / 2879.0 MHz, separation 19.9 MHz | published 0-mode 2852.5, π-mode 2856.0 MHz (3.5 MHz) |
| `deform_mag.py` | median maximum displacement 8.42 / 0.84 / 0.17 mm for `deform_max` 0.5 / 0.05 / 0.01 | — |
| `a1_ratios.py`, `a1_derived.py` | the tables of a1 §3.1 and §4.1 | the report |
| `a1_solverfix.py` | 222 of 3000 random β = 1 draws (7.4 %) rejected with a valid wall; all 222 recovered by tight tolerances | the report |
| `a1_notangent3.py` | valid walls missed by the solver: 133 of 1500 perturbed-TESLA draws (8.9 %), 101 of 1500 perturbed-`ilc_ll` draws (6.7 %) | the report |
| `a1_mc_generator.py` | accepted: TESLA neighbourhood 58.1 %, `ilc_ll` 42.8 % (32.6 % rejected as re-entrant), random β = 1 50.1 % | the report |
| `a1_widened.py` | accepted: current ranges 50.1 % → 75.9 % with the proposed rules (5.4 % re-entrant); widened ranges 41.8 % → 65.3 % (13.9 % re-entrant) | the report |
| `elliptical_dist.py` | wall angle p1 / p10 / p50 / p90 / p99 = 61.5 / 68.7 / 78.9 / 87.1 / 89.7°; 50 % of random β = 1 half-cells accepted | docs/33 §3.1 |
| `chord_inflation.py` | the table of docs/33 §2.2 | — |

The solver-tolerance finding was also checked against the generator's own `half_cell_wall`, with only the
`least_squares` tolerances changed: 4000 draws, 7.4 % recovered (docs/33 §3.1).
