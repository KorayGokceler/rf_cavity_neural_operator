---
name: task
description: Start a unit of work on this repository the agreed way - read the shared state, pick or restate the task, set up a branch, write a plan with acceptance criteria and wait for the owner's approval before changing code. Use at the start of any non-trivial change.
---
# Start a task

1. Read `AGENTS.md` and `docs/31_SESSION_HANDOFF.md` (state, numbers, open items §6). Read the design doc of
   the area you will touch (docs/README.md lists them).
2. Restate the task in two or three sentences. If it is an item of docs/31 §6, name it.
3. Branch: work on the branch the owner named. With parallel agents on one machine, use a worktree:
   `bash scripts/agent_worktree.sh <short-name>`.
4. Measure first when the task is about speed, memory or accuracy: record the baseline numbers.
5. Write the plan and stop for approval:
   - files to change and why;
   - what stays untouched (the invariants in AGENTS.md that apply);
   - acceptance criteria as checks with numbers (tests, analytic error bounds, timings, screenshots);
   - risks and how they will be detected.
6. After approval: implement, then run `/verify`, then commit with a message that states the numbers.
