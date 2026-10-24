# Editor notes — revision pass 1

The first full build compiles cleanly (no undefined citations or references) but is 31 pages. Target: about
17 pages of main text before the references. Material that is useful but too detailed may move to an
appendix file you own: `paper/sections/A{NN}_<name>.tex` starting with `\section{...}\label{sec:A{NN}-...}`
(the editor adds it after `\appendix`). Keep the rules of `paper/BRIEF.md` (§4); read BRIEF §1 again — the
thesis wording was corrected (see note 1).

## Page budgets (11pt, 1-inch margins; measured start pages in the first build in brackets)

| section | now | budget |
|---|---|---|
| 01 introduction | 2 | 1.5 |
| 02 related work | 1 | 1.5 (may grow slightly; it is the shortest) |
| 03 formulation | 3 | 2 |
| 04 method | 6 | 3.5 (+ appendix for proofs / implementation details) |
| 05 data | 4 | 2.5 (+ appendix: family table, generator speed-up table) |
| 06 multi-fidelity | 2 | 1.5 |
| 07 experiments | 4 | 3.5 |
| 08 discussion | 3 | 1.5 |

## Cross-section corrections (all sections must agree)

1. **Equal-cost result (docs/34 §8.2, corrected).** Each arm is a single training run (no seed repeats). LF 24
   (20 % lower label cost than HF 8) has mean frequency error 1.55 % vs 1.66 %; LF 40 (1.4× the cost) 1.37 %.
   Per geometry LF 40 beats HF 8 on 3 of 6 geometries; excluding the outlier geometry 27 the means are HF 8
   1.11 %, LF 24 1.14 %, LF 40 1.06 %. So the supported claim is: **LF training at equal or lower label cost is
   at least as accurate on held-out HF geometries as HF training, and more LF geometries help slightly.** Do
   not write "more accurate than HF" / "outperforms" without these qualifiers. The LF→HF gap (0.02 points) is
   the strong, clean result.
2. **The p2 "floor" — two different quantities.** (a) The discrete p2 eigenvalue on the model mesh vs the HF
   frequency: ≤ 0.033 % on the four geometries of docs/34 §4.1 (and the LF/HF label difference on identical
   meshes, +0.004 to +0.047 %, docs/34 §8.1 — LF *is* the p2 eigenvalue). This is the floor of the model space.
   (b) docs/30 §5: the Rayleigh quotients of the HF modes *projected onto* p2 exceed the p3 frequencies by
   0.04–0.9 % (two meshes of 11 300 and 13 052 tets). That is the frequency of the projected field *target*, an
   upper bound of (a), not the floor itself. Use (a) when you say "floor"; mention (b) only as the quality of
   the projected HF field target. Sections 05, 06, 07, 08 must use these words.
3. **Wall quantities LF vs HF** (Q0 −0.76 to −1.37 %, R/Q |Δ| ≤ 0.04 %, Epk/Eacc −28 to +6.2 %, Bpk/Eacc +5.4 to
   +8.1 %) come from docs/34 §3, measured on the earlier meshes where LF and HF meshes coincided only for one
   sample. Say so wherever they appear (06 owns the table; 08 refers to it).
4. **Bound wording.** Ritz values are upper bounds of the discrete eigenvalues of the space the model runs in
   (prop:upper-bound). Not of the exact eigenvalues.
5. **Overlaps to remove.** 08 "Why low-fidelity training works" repeats 06 — cut to one or two sentences with
   a \cref. The conclusion of 08 should not repeat the intro's contribution list verbatim. 05 and 06 both
   describe the start-mesh rule — 06 owns it; 05 gives one sentence and a \cref.
6. **References to fix or drop** (citation check follows in a later pass): `yaker2026neural` (byline not
   verified — drop unless verified), `kranjcevic2019constrained` (authors partly from memory — verify or
   drop), `edelen2018opportunities` (truncated authors — complete or drop), `hiptmair2002finite` (page range
   237–340 per Cambridge Core). Do not add new unverified references.
7. Section 05: the "wall-distance token features" must match what 04 describes (04 does describe the point
   features). Section 03: check the claim that NGSolve's `HCurl(order=p)` contains the gradients of H1 of order
   p+1 against src/data_gen/highorder.py and keep the p-indexing consistent with "p2 model, p3 labels".
8. Section 07 owns `tab:equal-cost`; 01/06/08 refer to it instead of repeating numbers beyond the headline.

## Deliverable

Revise only your own section file(s). Keep every `% src:` comment. Compile check in a private copy (several
agents work at once; never build inside `paper/` itself): `D=$(mktemp -d) && cp -r
/home/user/rf_cavity_neural_operator/paper/. $D && cd $D && python merge_bib.py && latexmk -pdf
-interaction=nonstopmode main.tex`. Make sure your section adds no errors, undefined references or overfull
boxes; if the build fails in someone else's section, ignore that. To list an appendix file of yours in that
private build, add `\appendix\input{sections/A..}` to the copy's main.tex only. Reply with what you cut or moved, the new length
estimate, and any remaining doubt.
