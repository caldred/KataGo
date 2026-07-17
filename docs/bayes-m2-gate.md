# M2 gate (pre-registered): the posterior must be honest before it steers anything

Written before the tests were implemented or run. The M2 state (commit
5897fbf3) is a passenger; this gate decides whether it graduates to M3.
Two parts: model-free correctness, then real-data calibration. Bands
fixed here; misses trigger attribution rounds (v6 precedent), never
silent tuning.

## Gate A — model-free (stub NN, debugSkipNeuralNet)

A1. **Posterior invariants** (every node with bayesState, after searches
at maxVisits in {20, 80, 300} on 5x5 and 7x7 from empty and from
mid-filled positions, several seeds): vPriv >= 0; vsOwn, vsKids >= 0;
kappa-derived betaKids finite; mu in [0.01, 0.99]; resolvable >= 0;
anchorFrozen implies the node has an NN output (or is root); every
evaluated non-terminal child of a bayes-computed parent is frozen by the
NEXT recompute of that parent. Any violation = FAIL.

A2. **Terminal-dominance minimax agreement**: positions where every
root child is terminal within <= 2 plies (nearly-full 5x5 boards,
constructed in-test), searched to exhaustion (visits >> tree size),
bayesDefaultSigma = 0.02 for the test. The Bayes root mu must agree
with the exact minimax over terminal values within 0.06 winrate
(tolerance = the documented terminal-conservatism: exact values enter
with the shared homogeneous v_u; 3 sigma of the residual shrinkage at
k <= 25). Report the actual gaps. Any gap > 0.06 = FAIL.

A3. **Max-vs-mean sanity** (reported, not gated): on the same searches,
the Bayes root mu vs the ordinary search winrate — direction and rough
magnitude of the difference (max backup should sit weakly above the
visit-weighted mean at max nodes with contested children).

## Gate B — real-data calibration probe (b18c384nbt, 9x9)

- Positions: 60 fresh self-play positions, generated exactly like M1
  phase A but seed base 500000 (disjoint from M1's games and from any
  gate ever run), moves 4-40, parent band |raw-0.5| <= 0.45.
- For each position: flag-on search (configs/bayes_m2_9x9.cfg + stock
  gtp base) at maxVisits in {30, 100} — visit currency ~= eval currency
  in tree mode (one NN eval per new node; the round-3 lesson says
  compare AT MATCHED EVALS, so both the claim and the band are per
  visit count). Reference: stock PUCT search, 1500 visits, same net.
- Readout: root bayes mu and claimed total sd via a debug GTP command
  added for the probe (kata-bayes-root).
- z = (mu - ref) / sd_claimed, sign in MOVER-of-root perspective... no:
  white perspective throughout (values are white-persp; no aggregation
  asymmetry — report raw).
- Bands (per visit count):
  - B=100: |mean z| <= 0.25; z std in [0.60, 1.35].
  - B=30:  |mean z| <= 0.30; z std in [0.50, 1.40].
  (Basis, recorded: the round-3 testbed stack showed -0.04..+0.07
  residual at B=100 with +0.13..+0.19 pockets at B=30 only at
  rho >= 0.6 (9x9 rho-hat = 0.26); z-std conservatism up to ~0.5 is the
  documented anchor-echo forfeit at strong anchors. The bands allow
  the documented conservatism and the measured small-budget optimism,
  nothing more.)
- Also reported, ungated: z by game phase; claimed sd vs |error|
  decile table; fraction of positions where the Bayes mu is closer to
  the reference than the ordinary search winrate is (curiosity metric,
  explicitly NOT a target).

## Failure protocol

Any Gate A failure: fix-and-rerun freely (correctness bugs, not
statistics). Any Gate B miss: attribution round first (which component,
which regime), documented in this file append-only; no coefficient may
move in response to a Gate B miss unless the move is derived from an M1
refit or a testbed gate, never from the probe numbers themselves.

## OUTCOME (2026-07-13, append-only)

**Gate A: PASS** (commit d57a1d5d). Invariants + freeze contract exact
over 30 stub searches; all 6 terminal-minimax positions within the 0.06
band (gaps 0.033-0.059 = the documented terminal conservatism); stub
stErr surprise (softplus floor 0.1733, not 0) and the genmove
root-promotion bayesState loss are documented in-code.

**Gate B: FAIL as scored — then fully attributed.** mean z = +0.382
(B=30, band 0.30) and +0.376 (B=100, band 0.25); z std PASSES both
budgets (1.13 / 1.20) and claimed sd tracks realized |error|
monotonically across deciles. In winrate units the miss is a ~+0.03
location drift with honest spread.

Attribution (side-split + matched-eval control, fresh 60 positions seed
base 500100, python/bayes_m2_control.py, bayes-data/m2-control.jsonl):

1. **Dominant component: shared shallow-vs-deep drift, NOT a Bayes
   defect.** The drift does not flip with side to move (B-to-move
   +0.18..+0.27 z, W-to-move +0.47..+0.54 — a max/min backup artifact
   would flip). A stock PUCT search at the SAME visit counts drifts
   white-high by MORE than the Bayes posterior (+0.040/+0.033 winrate
   at B=30/100 vs Bayes +0.030/+0.028): the b18 net's raw 9x9 opening
   evals sit white-optimistic relative to 1500-visit search, and every
   shallow consumer inherits it. Concentrated in the opening (z
   +0.70 at moves <= 10), where deep search corrects the net most.
2. **Bayes-specific residual: the known selection interaction, in its
   measured range.** Mover-perspective drift: Bayes +0.016/+0.022
   winrate (~ +0.14/+0.22 z) vs control ~0 (-0.010/+0.006). This is
   the adaptive-descent optional-stopping residual the round-3 testbed
   gate measured at +0.1..+0.19 z (absent under uniform descent) —
   tracked under roadmap item 2 (selection-aware accounting), not
   fixed here, per the working agreements.
3. **Band mis-design, owned:** this doc's own Gate B section cites the
   round-3 lesson ("compare AT MATCHED EVALS") and then set bands
   against the raw deep reference anyway. Scored against the matched
   control — the comparison round 3 actually prescribes — the Bayes
   location claim is at-or-better-than stock search at equal evals,
   with the selection residual as the only Bayes-attributable excess.

**Standing: FAIL-attributed (v5-gate-2 / round-3 precedent).** No
coefficient moved. M2 graduates to M3 with the selection residual
carried as a tracked known; the M3 decision-calibration gate must score
against matched-eval controls from the start, and the M3 match gate is
unaffected (Elo vs PUCT at equal visits is already a matched-eval
comparison by construction). Control-sample note: side-to-move split
43B/17W (vs probe 27B/33W) — conclusions rest on the within-side
decompositions, not the pooled mean.
