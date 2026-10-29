---
name: cst-compare
description: Compare our labels or model predictions with CST Microwave Studio eigenmode results - build the calibration set (STEP + reference labels), give the owner the CST settings and the run sheet, then tabulate frequencies and figures of merit (Q0, G, R/Q, T, Epk/Eacc, Bpk/Eacc) side by side with consistent conventions and explain every difference. Use when the owner brings CST numbers or wants to set up a CST comparison.
---
# Compare with CST

Background: `reports/CST eigenmode çözücüsü analizi.md` (how CST's hex FIT/PBA and tet FEM eigensolvers work,
their accuracy, and Appendix 3, the calibration runs) and the notes in `research_notes/CST eigenmode çözücüsü analizi/`.

## 1. Geometries and our reference values

- The calibration set: `python scripts/cst_calibration_set.py --out <dir> [--cases ...] [--max_ndof 600000]`.
  It writes `<case>.step` [mm] and `labels.h5` for four cases:
  - `calib_box`: 100 × 80 × 60 mm, analytic;
  - `calib_pillbox`: R 40, L 50 mm, analytic;
  - `pillbox_pipes`: test geometry 24;
  - `tesla_cell`: one TESLA mid-cell with pipes.
- Any generated geometry: the STEP files under `<TAG>/step/<family>/<family>_<id:08d>.step`. These exist for
  undeformed samples only; set deformation probability 0 for CST work. The matching labels are in the
  family's H5 shard, group `sample_<id>`.
- A UI prediction: the JSON export (modes, QoI) and the CSV of the mode table.
- Reference rules:
  - for the analytic cases, the analytic values are the reference;
  - for the box, our adaptive labels stop at 8 levels;
  - for the pillbox, the generator path falsely converged (docs/31 §6 item 14);
  - otherwise our label, with its adaptive stop reason and its last-level change as the uncertainty.

## 2. CST settings to give the owner (every run unless stated)

- Units mm and GHz. Background PEC, or all boundaries electric (Et = 0). Vacuum inside. No symmetry planes
  (full model).
- Eigenmode solver with a **tetrahedral** mesh:
  - solver order 3 (constant);
  - curved elements at the highest order available;
  - accuracy at the tightest setting (≤ 1e-9);
  - adaptive mesh refinement on, with ΔF ≤ 1e-6 and ≥ 3–4 passes.

  Record each pass: frequency and tets.
- 6 modes, frequencies above 0.
- Losses in post-processing only, with σ = 5.8e7 S/m (copper) and Rs at each mode's frequency.
- **R/Q:** Template Based Post Processing → 3D Eigenmode Result → R over Q.
  - Integrate along the beam axis over the cell only, excluding the pipes, with β = 1.
  - Ours is the **linac** convention, V²/(ωU) with the transit factor. CST's "circuit" convention is half of
    that.
- **Fields:** normalised to U = 1 J. Eacc = V / L_acc, using L_acc from the case table (cell length without
  pipes). Epk and Bpk are the maxima on the wall surface.
- Have the owner record:
  - the CST version;
  - the solver, mesh and adaptive history;
  - 6 frequencies;
  - for mode 0: Q0, R/Q, V, Epk, Hpk or Bpk;
  - the run time.

## 3. Compare

- Match the modes by field pattern and degeneracy, not by index alone. Degenerate pairs (TE111, TM110 in the
  pillbox) are compared as pairs; per-mode QoI of a pair depend on its rotation.
- Table per case: mode | f_ref | f_CST | Δf/f | (f_ours | Δf/f). Then mode 0's figures of merit:
  Q0, G, R/Q, T, Epk/Eacc, Bpk/Eacc, each with ours, CST, Δ%.
- **Expected agreement** (rough orders from the research report, not measured here yet):
  - tet FEM order 3 with curved elements: frequencies 1e-6–1e-8, Q0 / R/Q 1e-4–1e-3, peak fields ≈ 1e-2
    (peaks converge slowest);
  - hex FIT/PBA: frequencies ~1e-4 with a systematic sign, worse Q0 and peaks on curved walls.
- **Explain each larger difference** from this list before calling it a solver difference:
  - the R/Q convention;
  - the L_acc or integration line;
  - β;
  - the Rs frequency;
  - mm vs m;
  - pipes counted in L_acc;
  - symmetry planes;
  - an unconverged adaptive pass on either side;
  - sharp vs filleted edges;
  - mode mismatch.

## 4. Record

Write the tables and settings into a dated section of docs/27 (label quality) or a new numbered doc via
`docs-writer`, with the owner's CST version and run log, and add a docs/31 §3 measurement row. Calibration
results that change our conventions (e.g. a QoI rule) need the owner's decision first.
