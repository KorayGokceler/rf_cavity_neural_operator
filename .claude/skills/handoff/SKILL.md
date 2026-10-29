---
name: handoff
description: End-of-session handoff - update docs/31_SESSION_HANDOFF.md with what changed, the measured numbers and the open items, so the next agent (any model, any tool) starts from the repository. Use before ending a working session.
---
# Session handoff

Update `docs/31_SESSION_HANDOFF.md`. Keep it short, factual and in Turkish, like the rest of that file. The
`docs-writer` subagent can do this in its own context.

1. **§1 Durum:**
   - Change the rows this session affected.
   - Refresh "Son commitler" with this session's commit hashes and one-line summaries
     (`git log --oneline <start>..HEAD` on every branch the session pushed).
   - Name the branches that are not merged yet and what they hold. Example: "`claude/ui-redesign`: new UI,
     waiting for the owner".
2. **§3 Ölçümler:**
   - Add each new measurement as a compact table with its setup: sample ids, DOF, tets, threads, hardware,
     load. Then the numbers.
   - Replace numbers that a newer measurement supersedes, and `git grep` the old number in other docs.
3. **§4 Kararlar:** add decisions taken this session with their reason. Never delete an old decision; mark it
   superseded instead.
4. **§6 Açık işler:**
   - Tick off finished items.
   - Add new ones in priority order. Include things waiting for the owner, such as approvals, results from
     CST and data from Colab.
   - Keep the owner, branch and status columns current.
5. **Dates and docs.**
   - Update the date line at the top.
   - If behaviour changed, also update the matching `docs/NN_*.md`.
   - Check that every number you wrote has its source next to it.
6. **Commit and report.**
   - Commit only the docs (`docs: session handoff <date>`) and push to the branch the session worked on.
   - Report what you changed in two or three lines.
