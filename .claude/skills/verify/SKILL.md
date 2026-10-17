---
name: verify
description: Run the checks that prove the current change works - tests chosen from the changed paths, the analytic pillbox validation for solver or label changes, the web build and a browser check for UI changes - and report the numbers. Use before claiming a task done or committing.
---
# Verify the current change

1. `git status --short` and `git diff --stat` (against the base branch the owner works on) to see what changed.
2. Choose the checks from the changed paths:
   | changed | run |
   |---|---|
   | `src/models/`, `src/training/` | `python -m pytest -q tests/test_field_model.py tests/test_eigenspace_3d.py tests/test_span_losses.py` |
   | `src/data_gen/` | `python -m pytest -q tests/test_field_labels.py tests/test_field_adaptive.py tests/test_data_gen_3d.py` and `python scripts/validate_field_labels.py --max_ndof 300000 --threads 4` (must print PASS) |
   | `src/data/` | `python -m pytest -q tests/test_field_model.py tests/test_dataset_converter_3d.py` |
   | `src/qoi/` | `python -m pytest -q tests/test_qoi.py tests/test_qoi_data.py tests/test_qoi_torch.py` |
   | `src/service/` | `python -m pytest -q tests/test_service_api.py tests/test_service_datasets.py tests/test_service_training.py tests/test_service_truba.py` |
   | `web/` | `cd web && npm run build`, `python -m pytest -q tests/test_service_api.py`, then a Playwright check against `python scripts/serve_web.py --data <root> --checkpoint_optional` with screenshots and no console errors (stop the server afterwards) |
   | `cluster/truba/` | `bash cluster/truba/run.sh <cmd> ... --dry-run` and `python -m pytest -q tests/test_service_truba.py` |
   | anything else or unsure | `python -m pytest -q tests/` |
   The `numerics-check` and `ui` subagents run these in their own context when the owner asks for them.
3. For changes to the solver, projection, Ritz layer, losses or figures of merit, also have the diff
   reviewed (`physics-review` subagent, or a reviewer from another model).
4. Report: each command, pass/fail counts, the key numbers with before/after where relevant, and any failure
   with the end of its traceback. A failure is never called a flake without one re-run.
