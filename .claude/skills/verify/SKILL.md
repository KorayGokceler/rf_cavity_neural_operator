---
name: verify
description: Run the checks that prove the current change works - tests chosen from the changed paths, the analytic pillbox validation for solver or label changes, the Ritz-bound checks for projection changes, the web build and a browser check for UI changes, dry-runs for TRUBA scripts, a clean paper build - and report the numbers. Use before claiming a task done or committing.
---
# Verify the current change

1. **See what changed:** `git status --short` and `git diff --stat <base>`, against the branch the owner works
   on.
2. **Choose the checks from the changed paths:**

   | changed | run |
   |---|---|
   | `src/models/`, `src/training/` | `python -m pytest -q tests/test_field_model.py tests/test_hcurl_projection.py tests/test_span_losses.py` |
   | `src/models/hcurl.py` (projection, Ritz, CG) | the above, plus `python scripts/paper/check_ritz_bound.py --data <LF H5> --threads 4`: the guarantee θ̃(1+η̄) ≥ λ_h holds in every row and there are no spurious modes with the projection |
   | `src/data_gen/` | `python -m pytest -q tests/test_field_labels.py tests/test_field_adaptive.py tests/test_data_gen_3d.py`, and `python scripts/validate_field_labels.py --max_ndof 300000 --threads 4`, which must print PASS with errors no worse than docs/31 §1 |
   | `src/data/` | `python -m pytest -q tests/test_field_model.py tests/test_mesh_features.py` |
   | `src/qoi/`, `src/viz/` | `python -m pytest -q tests/test_qoi.py tests/test_hcurl_display.py tests/test_viz_meshsample.py` |
   | `src/service/` | `python -m pytest -q tests/test_service_api.py tests/test_service_datasets.py tests/test_service_training.py tests/test_service_truba.py` |
   | `web/` | `cd web && npm run build`, `python -m pytest -q tests/test_service_api.py`, then a Playwright check against `python scripts/serve_web.py --data <root> --checkpoint_optional --no_token --port <p>`. Desktop and 390 px, light and dark, no console errors, no horizontal scroll at 390 px, timings before/after for touched interactions. Stop the server afterwards. |
   | `scripts/serve_web.py`, `scripts/colab_webui.py`, `Colab_*.ipynb` | `python -m pytest -q tests/test_serve_web.py tests/test_service_api.py`; for the notebook, check that the JSON parses and that the cell's flags exist in the script's `--help` |
   | `cluster/truba/` | `bash -n` on edited scripts, `bash cluster/truba/run.sh <dataset\|train\|all> LABELS=field ... --dry-run` (`LABELS=n0` must fail with a clear message), `python -m pytest -q tests/test_service_truba.py` |
   | `scripts/` (other) | the test that imports the script (`git grep -l <script name> tests/`), and one real run on a small input |
   | `paper/` | build in a scratch copy (`python merge_bib.py && latexmk -pdf main.tex`): no undefined references or citations, page count reported |
   | `docs/` only | links and section references resolve (`git grep` the cited sections); no test run needed |
   | anything else or unsure | `python -m pytest -q tests/` |

   The `numerics-check`, `ui` and `truba-ops` subagents run these in their own context.
3. **Get a review.** Changes to the solver, projection, Ritz layer, losses, figures of merit or the label
   generator get a `physics-review`, or a reviewer from another model, before committing.
4. **Report:**
   - each command, with its pass/fail counts;
   - the key numbers, with before/after where relevant;
   - every failure, with the end of its traceback.

   A failure is never called a flake without one re-run. Say explicitly which checks were skipped and why.
