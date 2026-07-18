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

## Phase A outcome + Amendment A (2026-07-18; appended after the
## two-arm run, BEFORE the third arm runs)

Two-arm outcome (bayes-data/m7-replay.jsonl, n = 421): P-A1 CONFIRMED —
paired leak -0.0262 [-0.0341, -0.0187] mover-persp, -0.0573 on the 46%
disagreement positions; the match's per-decision defect reproduces with
no opponent and no compounding. P-A2 CONFIRMED beyond its band: root
breadth 41.8 vs 3.4, max PV depth 3.9 vs 10.7, chosen-move visits
median 5 vs 48, thin-pick rate 28.5% vs 0.0%; the leak concentrates in
the broad trees (-0.029 at breadth >= 20 vs -0.010 at breadth < 10).
Attribution moves to ALLOCATION: voi-KG buys root breadth at a budget
where Go's information lives at depth.

Amendment A (registered now): a third arm isolating allocation from
beliefs+chooser. Engine change: new SearchParams boolean
`useBayesChooseMu` (default false; diagnostic-only) that enables the
existing argmax-mu chooser in getChosenMoveLoc without
useBayesSelection, so a bot can run bayes posterior state as passenger
on a STOCK PUCT tree and still choose by posterior mean. No search
behavior changes for any existing config (both match flags on implies
the old path). Arm `hybrid` = bayes_m7_gtp.cfg minus useBayesSelection
plus useBayesChooseMu; same 421 positions, same labeling.
- P-A4: hybrid's paired leak vs puct is <= 1/3 of the bayes arm's
  (beliefs + chooser are approximately healthy when fed a deep tree;
  allocation carries the defect).
- If instead hybrid leaks comparably to bayes, the beliefs/chooser
  share the blame and Phase C (per-eval audit) runs on the hybrid
  configuration first (simpler tree dynamics to diff).

## Amendment A outcome (2026-07-18, append-only)

**P-A4 FAILS — the decomposition inverts the allocation story.** Hybrid
(bayes posterior as passenger on a stock PUCT tree, argmax-mu chooser;
tree: breadth 3.4, depth 10.8, chosen-visits median 42, thin 1.2%)
still leaks -0.0197 [-0.0272, -0.0128] vs puct — 75% of the full bayes
arm's -0.0262. Allocation explains only ~25% of the defect. Given the
SAME deep tree and the same evals, the posterior's mu ranking loses to
the plain visit-average ranking. This matches the M3 live observation
(F5: mu-best, search-stats-worst) and moves the primary attribution to
BELIEF INTEGRATION: the recursive posterior (max-backup E[max] offsets,
Stein shrinkage, per-level clip, heads at every level) degrades the
information it consumes relative to a plain average.

Localization (post-hoc splits, reported not gated): the hybrid leak
concentrates in NEAR-EVEN positions (-0.026 at extremity < 0.15,
-0.020 at [0.15, 0.3)) and vanishes at extremity >= 0.3 — NOT a
value-clip/bounds artifact. Color split: White-to-move leaks ~2x
Black-to-move (hybrid -0.0254 vs -0.0140; disagree 41.5% vs 31.1%),
echoing the A3 tests' uniformly negative white-persp mu-vs-searchWinrate
diffs — a possible max/min-node asymmetry, confounded at komi 7 by
position character. Phase C (per-eval audit vs the Python reference)
now runs per its registered trigger, targeted first at near-even
White-to-move positions from this replay set.

## Phase C protocol (Amendment B, registered before implementation)

Instrument: an audit dump in bayesRecomputeNodeStats, gated on the
environment variable KATAGO_BAYES_AUDIT (file path); when set, every
ROOT recompute appends one JSON line with the full set state — per arm:
move, prior, evaled/terminal/frozen, evalWinrate, evalStErr, stein mu,
posterior mu, vPriv, b, R, D, w, child visit count; set-level: n, k,
anchMu, anchVar, sigmaR, vU, varS, eD, vD, g, vX, kappaAlpha, dKids,
mKids, vKidsPriv, bOut, dBackup, and the node posterior written back
(mu, b, vPriv, resolvable). No behavior change whatsoever when the
variable is unset; diagnostic-only, like useBayesChooseMu.

Audit set: from the replay join, the 20 White-to-move and 10
Black-to-move positions with extremity < 0.15 where hybrid disagreed
with puct, ordered by |paired leak| descending (deterministic given
the existing data). Each position runs once under the hybrid config
and once under the bayes config with the dump enabled.

Layer 1 (wiring): for every dumped root recompute, feed the dumped
INPUTS (priors, evals, stErrs, anchor, frozen-child states, dumped
sigmaR/vU/varS/eD/vD/g) through the bmcts reference implementations
(posterior.py: extreme_moments_gaussian, shrink_siblings_stein, the
node posterior backup) with the engine coefficients, and diff against
the dumped OUTPUTS at 1e-9 relative tolerance (the golden-suite
standard). First divergent (position, recompute, field) localizes any
wiring bug. Also cross-check the engine's own derived stages (sigmaR
from stNode, vU/varS from evalStErrs, dMeans from priors) recomputed
in Python from primitives.
Layer 2 (theory): at each position's final recompute, decompose
mover-persp (mu_j - plain-visit-average_j) for the hybrid's chosen arm
vs puct's chosen arm into contributions: d-mean prior term, Stein
shrinkage pull toward the set, anchor offset (eD), and frozen-child
integration (cbs.mu vs the child's raw eval). The dominant term at the
misranked decisions is the finding.

Predictions: P-C1 layer 1 finds NO divergence (596 goldens + A-tests
make a raw math bug unlikely; the color asymmetry is more plausibly
model-level). P-C2 the dominant layer-2 term is the frozen-child
integration or the anchor offset, not the d-mean term (M5 already
exonerated the prior mean). Either prediction failing is itself the
lead.

## What this is not

No engine constant changes, no head refits, no new chooser — M7 is
attribution only. Any fix that emerges gets its own registered gate
(M8+), with game-level scoring (deviation rate, excess leak, Elo) per
the M6 outcome's requirement.
