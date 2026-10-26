@AGENTS.md

## Claude Code

- Subagents in `.claude/agents/` (each file says when to use it): `explorer` (read-only search, cheap
  model), `numerics-check` (tests + analytic and Ritz-bound checks), `physics-review` (fresh-context
  review against the invariants above), `ui` (web UI + build + browser check), `label-pipeline`
  (geometry → mesh → labels → H5), `trainer` (training, evaluation, baselines), `truba-ops` (SLURM,
  dry-run only) and `docs-writer` (docs and paper). Use them for context-heavy work and independent
  checks; keep the main session on one model.
- Workflow commands in `.claude/skills/`: `/task` (plan with acceptance criteria before coding),
  `/verify` (checks chosen from the changed paths), `/handoff` (update docs/31 at the end of a session),
  `/experiment` (controlled training study), `/measure` (before/after timing and memory), `/cst-compare`
  (CST comparison), `/truba-run` (prepare a cluster run), `/new-family` (add a geometry family).
- Permissions for routine checks and the denied destructive commands are in `.claude/settings.json`;
  personal overrides go to `.claude/settings.local.json` (not committed).
- Keep this file and AGENTS.md short and stable: they are part of every request's cached prefix.
