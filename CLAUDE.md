@AGENTS.md

## Claude Code

- Subagents for this repo are in `.claude/agents/`: `explorer` (read-only search, cheap model),
  `numerics-check` (tests + analytic validation), `ui` (web UI + build + browser check) and
  `physics-review` (fresh-context review of a diff against the invariants above). Use them for
  context-heavy reading and independent checks; keep the main session on one model.
- Workflow commands in `.claude/skills/`: `/task` (plan with acceptance criteria before coding),
  `/verify` (checks chosen from the changed paths), `/handoff` (update docs/31 at the end of a session).
- Permissions for routine checks and the denied destructive commands are in `.claude/settings.json`;
  personal overrides go to `.claude/settings.local.json` (not committed).
- Keep this file and AGENTS.md short and stable: they are part of every request's cached prefix.
