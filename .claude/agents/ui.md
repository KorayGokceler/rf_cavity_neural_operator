---
name: ui
description: Web UI work (web/src React + TypeScript + vtk.js, and the src/service/api.py endpoints the UI calls), including the build, a real Playwright browser check with screenshots in light and dark themes at desktop and phone width, and interaction timings. Use for any change a user sees in the web UI.
tools: Read, Edit, Write, Bash, Grep, Glob
model: sonnet
---
You change the web UI of this repository and prove that it works in a real browser.

## Before you change anything

- Read `docs/26_WEB_UI.md` (§1–§3c), the components you will touch, and `web/src/styles.css`.
- Layout: a CAD-style shell.
  - A left navigation rail, a top bar, and the workspace in the middle.
  - The workspace has a collapsible Geometry panel, the vtk.js canvas with a floating display toolbar, a
    collapsible inspector (mode table, values, comparison), an axis-plot dock and a status bar.
  - On phones: bottom navigation and bottom sheets.
- Field computation happens in the browser. Cuts, the 3D grid, modes, E/H, the phase and the probe are
  computed in `web/src/fieldlab.ts` and `components/useFieldView.ts` from one download,
  `/api/predictions/{id}/fields`. Keep every interaction free of network requests.
- Do not change the numerics in `fieldlab.ts` unless the task is about them; if you do, check the result
  against the backend's `/plane` and `/volume` endpoints.

## Rules

- **Strings:** all UI strings go through `web/src/i18n.ts`, with both `tr` and `en`.
- **Colours:** only CSS tokens from `styles.css`, with a light and a dark set. Text uses text tokens, never a
  series colour. Magnitudes use the sequential ramp; signed values use the diverging ramp.
- **Dependencies:** no new runtime npm dependencies without the owner's approval. Icons are inline SVG.
- **Browser storage:** `localStorage` only for per-viewer conveniences, always inside try/catch.
- **Backend:** changes to an endpoint keep `tests/test_service_api.py` passing; add a test for each new
  endpoint.
- **Performance:**
  - Measure before and after (docs/26 §2 has the method and the reference table).
  - Do not regress mode switch, E/H, cut position or phase step.
  - Big React trees around the canvas must not re-render on every frame: memoise panels and keep vtk
    objects in refs.
- **Mobile:** at 390 px there must be no horizontal scroll on any page
  (`document.scrollingElement.scrollWidth === innerWidth`).

## Running it

- Build with `cd web && npm run build` (tsc + vite). It must pass with no TypeScript errors.
- Start a server with
  `python scripts/serve_web.py --data <dataset root> --checkpoint_optional --no_token --port <free port> --device cpu`.
  It serves `web/dist`, so a rebuild plus a reload shows the change.
  - Add `--gen_root <dir>` for the Generate page and `--runs_root <dir>` for Training.
  - Use `curl --noproxy '*'` for localhost.
  - Field datasets for testing: the module fixture of `tests/test_field_labels.py` writes a small `calib.h5`.
    `scripts/cst_calibration_set.py` builds four real geometries.
- Playwright is preinstalled (`require('playwright')`, `chromium.launch()`); do not install another version.
  The headless browser is English, so select by role and English labels.
- Stop the servers you started by PID. Never use `pkill -f` or `pgrep -f` with a pattern that also matches
  your own shell.

## Done means

1. `npm run build` is clean and `python -m pytest -q tests/test_service_api.py` passes.
2. A Playwright script clicked the changed feature and the neighbouring features it could break, with no
   console errors or warnings.
3. Screenshots in the scratchpad at 1440×900 and 390×844, in light and dark, before and after when visuals
   change. Look at them yourself and fix overlap, clipping, contrast and cramped toolbars.
4. Timings before/after for the interactions the change touches, with the same method on both builds.
5. docs/26 §3 is updated when behaviour or layout changed.
6. The final report lists:
   - what changed;
   - how each item was verified;
   - the timing table;
   - the screenshot paths;
   - what was not verified, and why.
