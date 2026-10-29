---
name: new-family
description: Add a new cavity geometry family end to end - builder in cavity_shapes.py, generator registration, a fixed id block in families.tsv, train or OOD group, field-label settings (healing, fillets, element budget), tests, UI/TRUBA visibility and docs - with the owner's go-ahead because it extends the data generator. Use when a new cavity type (from docs/33 or the owner) should enter the datasets.
---
# Add a geometry family

Adding a family extends the data generator, so it needs the owner's go-ahead (AGENTS.md). Get it, with the
plan below, before writing code. The `label-pipeline` subagent can do the work.

## 1. Plan (for the owner's approval)

- **Physics:**
  - what the family represents (e.g. a QWR or a crab cavity), with a literature reference (docs/33 has the
    review);
  - the parameters with their ranges, and the units in m;
  - the topology: b1 and the number of boundary components. Handles are allowed in the E formulation.
  - the beam axis.
- **Group:** `train` or `ood` (never trained on). Train families use blocks 1–9 so far and OOD families 20–24.
  Take the next free block in `cluster/truba/families.tsv`, at most 31. A block is never renumbered or
  reused.
- **Counts:** n_total, and shard_size (a power of 2, fixed once shards exist).
- **Field-label specifics:**
  - re-entrant edges to round, and whether the fillet floor `--min_fillet` suits the smallest feature;
  - whether OCC healing is needed (`HEAL_FAMILIES`; it breaks some revolved solids);
  - the expected tets at the default mesh size against `model_max_elements`.

## 2. Implement

1. **Builder.** Add `build_<family>(occ, rng)` to `src/data_gen/cavity_shapes.py`.
   - Draw every parameter from `rng` in a fixed order.
   - Return `(shape_type, params)` with float params.
   - Set `_h_cap` when a small feature needs a finer mesh, and `_vol_div` for multi-cell shapes.
   - Build one closed solid in metres.
   - Accept `params=` for fixed designs, as `build_elliptical` does, when the family has a reference design
     (`ANCHORS`).
2. **Registry.** Register it in `BUILDERS` in `src/data_gen/dataset_generator_3d.py`. Add it to `FAMILIES`
   (train) or to `cs.OOD_FAMILIES`, and to `HEAL_FAMILIES` if needed.
3. **Ids.** Add the row to `cluster/truba/families.tsv`. The UI's Generate page and TRUBA page read the
   families from the API and from this file; check that both list the new family.
4. **Tests** in `tests/test_data_gen_3d.py` and `tests/test_field_labels.py`:
   - 3 ids build one valid, certified mesh (connected manifold, expected b1);
   - the same id reproduces the same `geom_params`;
   - one field-label sample at a coarse size passes `_check_reload`;
   - for STEP export, the volume matches the H5 to 1e-6.

## 3. Check the family

- Build ~20 ids (`--n_total 20 --n_workers 2 --threads 1`, field labels at the planned settings). Report:
  - the success rate and the re-draw reasons;
  - tets and label DOF;
  - time per sample;
  - the adaptive stop reasons;
  - the frequency range of mode 0;
  - the share of samples on the sharp-edge fallback.
- Look at 3–4 geometries: `scripts/show_geometries.py`, or open them in the web UI.
- Where an analytic or literature value exists for an anchor design, compare against it.
- Run `/verify` for `src/data_gen/`; the analytic pillbox validation must still PASS.

## 4. Record

- Add the family to docs/19 (families table) and docs/33 (if it came from the review), with its block and
  group.
- Add the cost row to docs/29 or docs/30.
- Add a docs/31 §1 note.
- Update the TRUBA README if `run.sh FAMILIES=train` now includes it.
