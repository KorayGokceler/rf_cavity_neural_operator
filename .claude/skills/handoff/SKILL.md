---
name: handoff
description: End-of-session handoff - update docs/31_SESSION_HANDOFF.md with what changed, the measured numbers and the open items, so the next agent (any model, any tool) starts from the repository. Use before ending a working session.
---
# Session handoff

Update `docs/31_SESSION_HANDOFF.md`; keep it short, factual and in Turkish like the rest of that file:

1. **§1 Durum:** change the rows this session affected; refresh "Son commitler" with this session's
   commit hashes and one-line summaries (`git log --oneline` since the session started).
2. **§3 Ölçümler:** add each new measurement as a compact table with the setup (sample, DOF, threads,
   hardware) and the numbers; replace numbers that a newer measurement supersedes.
3. **§4 Kararlar:** add decisions taken this session with their reason; never delete an old decision,
   mark it superseded instead.
4. **§6 Açık işler:** tick off finished items, add new ones in priority order, keep the owner / branch /
   status columns current.
5. Update the date line at the top. If behaviour changed, also update the matching `docs/NN_*.md`.
6. Commit only the docs (`docs: session handoff <date>`), then report what you changed in two or three lines.
