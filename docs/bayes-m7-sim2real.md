# M7 — sim-to-real bridge: why the stack that wins the testbed loses
# the engine match (pre-registered)

Written and committed BEFORE any M7 script or config exists. Standing
question (Cal, 2026-07-18): the frozen stack beats PUCT in 47/48
power-grid cells yet loses -366 Elo to PUCT at B = 64 on 9x9; M4 + M6
closed the choosing layer; either the testbed's generative model hides
an engine-critical crack, or the C++ implementation diverges from the
theory. M7 hunts both, in that order of prior probability.

## Measured cracks (from bayes-data/m5-tail.jsonl, BEFORE this doc;
## no match results consumed)

C1 **Same-net error correlation.** corr(first-eval error, prior-head
   error) per child = +0.38 pooled in the optimism convention (+0.53
   at parent extremity [0.3, 0.45), +0.14 at >= 0.45; n = 2,061).
   Policy and value come from one net: when it overrates a move, BOTH
   the d-mean prior and the eval overrate it. The posterior fuses them
   as independent; the testbed's dmean noise is independent of eval
   noise BY CONSTRUCTION, so no grid cell ever tested this. PUCT never
   fuses the prior as evidence — structurally immune.
C2 **Boundedness.** Fraction of deep child values within 0.05 of a
   value bound: 0.32 / 0.51 / 0.83 / 1.00 by parent-extremity bin —
   tri-modal bounded truth under Gaussian machinery.
C3 **Saturation.** Mean true gap vs the linear d-mean prediction by
   prior-gap bin: -0.024/-0.059, -0.086/-0.199, -0.122/-0.321,
   -0.179/-0.447, -0.245/-0.525 — the real curve runs at 40-55% of
   linear everywhere, and the engine consumes linear-with-floor.

## Phase A — paired engine replay (primary; no sim assumptions)

Protocol: the 421 M5 positions (deterministic sample, already fixed).
For each, two deterministic single-threaded searches from the SAME
position with the SAME net, maxVisits = 64, chosenMoveTemperature = 0,
no ponder, no resign:
- `bayes`: exactly the match bot-1 params (useBayesSearch +
  useBayesSelection + M2 coefficients, configs/bayes_m7_gtp.cfg);
- `puct`: stock search, same limits (configs/puct_m7_gtp.cfg).
Driven over GTP with kata-genmove_analyze; recorded per (position,
engine): chosen move, reported winrate, per-child visit counts (root
breadth = children with >= 1 visit), max PV length (depth proxy).
Each chosen move gets a neutral 500-visit label via the analysis
engine (reusing M5 child labels where the move coincides; append-only
new labels otherwise). Output: bayes-data/m7-replay.jsonl.

Readouts (registered):
- Paired per-position leak difference: mover-persp deep value of the
  bayes choice minus the puct choice; mean with cluster-bootstrap CI
  (by game). Baseline from the matches: -0.012/move excess leak.
- Deviation-from-top-prior rate per engine (baseline 37.5% for bayes
  in the lost games).
- Tree shape: root breadth and depth proxy per engine.

Predictions (scored honestly in the outcome):
- P-A1: the bayes choice leaks >= 0.008/position more than the puct
  choice (the match leak reproduces without opponent or compounding).
- P-A2: bayes root breadth >= 2x puct's at B = 64; bayes depth proxy
  materially lower. (A3 telemetry precedent: 20-50 root children at
  v = 300.)
- P-A3 (branch): if P-A1 FAILS (paired diff ~ 0), the per-decision
  layer is exonerated engine-side; attribution moves to trajectory
  effects (adversarial steering into belief-hostile regions), and
  Phase C runs FIRST to rule out implementation divergence before any
  such conclusion is recorded.

## Phase B — testbed ablation of the measured cracks (bmcts side)

Root-bandit sim at k = real (the 421 stored legal-policy vectors),
B = 64: truth and evals generated with four toggles — (i) saturating
gap curve (C3 table) vs linear; (ii) bounded tri-modal values (C2
rates) vs Gaussian; (iii) eval-prior error correlation +0.38 (C1) vs
0; (iv) claimed sigma as shipped vs realized. Algorithms: the ENGINE's
root fusion (d-mean with 5e-3 floor, [0.01, 0.99] clip, homoscedastic
sigma_r, argmax-mu) vs faithful PUCT (policy-gated allocation, choice
by visits). Readout: per-decision value-loss gap per toggle
combination (2^4, N = 421 positions x 40 noise draws, paired).
- P-B1: all toggles ON -> PUCT wins (engine reproduced); all OFF ->
  bayes wins (grid-v2 reproduced); the minimal flipping set includes
  toggle (iii) (same-net correlation).
Script: bmcts scripts/sim2real_m7.py; results append-only there.

## Phase C — one-eval-at-a-time implementation audit (conditional)

Trigger (registered): P-A1 fails; OR Phase A shows any theory-
inconsistent anomaly (e.g. breadth NOT elevated); OR Phase B's cracks
explain < half of the Phase-A paired gap. Method: config-gated
per-visit dump in bayessearch.cpp (selected path, eval received, root
sibling state after each recompute) on >= 20 deviated positions;
replay the identical eval sequence through the bmcts Python stack with
the engine coefficients; diff mu/sigma/anchor trajectories at fp
tolerance; first divergent step localizes the bug. The 596 goldens
cover the posterior FUNCTIONS; this audits the WIRING (set
construction, anchor freeze timing, KG routing, virtual children).

## What this is not

No engine constant changes, no head refits, no new chooser — M7 is
attribution only. Any fix that emerges gets its own registered gate
(M8+), with game-level scoring (deviation rate, excess leak, Elo) per
the M6 outcome's requirement.
