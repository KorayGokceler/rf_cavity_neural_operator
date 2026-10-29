---
name: task
description: Start a unit of work on this repository the agreed way - read the shared state, pick or restate the task, set up a branch, write a plan with acceptance criteria and wait for the owner's approval before changing code. Use at the start of any non-trivial change.
---
# Start a task

1. **Read the shared state.**
   - `AGENTS.md` and `docs/31_SESSION_HANDOFF.md` (state, numbers, open items §6).
   - The design doc of the area you will touch (`docs/README.md` lists them).
   - For a wide search, send the `explorer` subagent instead of reading everything yourself.
2. **Restate the task** in two or three sentences, in Turkish to the owner.
   - If it is an item of docs/31 §6, name it and claim it there (owner, branch).
   - If it is ambiguous, ask the one question whose answer changes the plan, with options and a
     recommendation.
3. **Branch.** Work on the branch the owner named. A large or separable change (a UI redesign, an
   experiment) goes on its own branch when the owner asks for that. With parallel agents on one machine, use a
   worktree: `bash scripts/agent_worktree.sh <short-name> [base]`.
4. **Measure first** when the task is about speed, memory or accuracy: record the baseline with `/measure`.
5. **Write the plan, then stop for approval:**
   - files to change and why;
   - what stays untouched, i.e. the AGENTS.md invariants that apply. "Do not change the data generator's
     behaviour without the owner's go-ahead" counts here.
   - acceptance criteria as checks with numbers: tests, analytic error bounds, timings, screenshots;
   - risks and how they will be detected;
   - who does what. A subagent's work goes in its own context:

     | area | subagent |
     |---|---|
     | `web/` | `ui` |
     | `src/data_gen/` | `label-pipeline` |
     | training and evaluation | `trainer` |
     | TRUBA | `truba-ops` |
     | docs and paper | `docs-writer` |
     | checks | `numerics-check` |
     | review | `physics-review` |
6. **After approval:**
   1. Implement.
   2. Run `/verify`.
   3. Get the review the change needs.
   4. Commit with a message that states the numbers, and push to the agreed branch.
   5. Finish with `/handoff` at the end of the session.
