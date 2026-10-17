@AGENTS.md

## Claude Code

- Subagents for this repo are in `.claude/agents/`: `explorer` (read-only search, cheap model),
  `numerics-check` (tests + analytic validation), `ui` (web UI + build + browser check) and
  `physics-review` (fresh-context review of a diff against the invariants above). Use them for
  context-heavy reading and independent checks; keep the main session on one model.
- Keep this file and AGENTS.md short and stable: they are part of every request's cached prefix.
