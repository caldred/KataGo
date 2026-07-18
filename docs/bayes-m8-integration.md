# M8 — evidence-aware integration: from diagnosis to a shipping fix
# (pre-registered, phase 1)

Written and committed BEFORE any M8 script exists. Goal, per Cal
(2026-07-18): CBTS must WIN at KataGo, not merely document its failure;
the engine is the testing ground and re-derivation is on the table.

## Where M7 left the theory

No implementation bug (3,780/3,780 recomputes bit-perfect vs the
reference). The defect: mover-persp mu of the deepest-VERIFIED root arm
sits -0.089 below that arm's own subtree average, because the
per-level debiasing (E[max] anchor offset eD from the fresh-child
spread head; Stein shrinkage toward anchors frozen at first-eval
moments) treats every child as the M1 evidence class — a fresh noisy
eval — when a verified subtree's effective error is far smaller than
its claimed sigma. The correction is right for fresh children, wrong
for verified ones, and it compounds once per level of a deep line.

## Phase 1 — counterfactual attribution on full-tree dumps

Instrument extension: KATAGO_BAYES_AUDIT_ALL=1 extends the M7 dump to
EVERY node recompute (not just root), adding a node id and per-arm
child ids so the tree can be reconstructed. Diagnostic-only, unset =
no change.

Data: the M7 audit set (30 positions), hybrid config (PUCT tree =
deep verified lines; the regime that exposes the defect) and bayes
config (its own shallow-broad shape), one run each with the full dump.

Method: reconstruct each search's final tree from the last dump line
per node; re-integrate ALL node posteriors bottom-up in Python (the
bit-perfect reference, stable-sort convention) under counterfactual
rules, and score the ROOT ranking that results. Counterfactuals
(structural toggles, no fitted constants):

- C-A **anchor refresh**: each node's anchor tracks its current
  children-derived posterior instead of staying frozen at its
  first-eval moment (variance refreshed consistently).
- C-B **evidence-aware eD**: the E[max] premium subtracted from the
  anchor is computed only over the still-unverified arms of the set
  (verified arms contribute their posterior, not a fresh-draw slot).
- C-C **evidence-aware child sigma**: a frozen child's claimed
  residual spread contracts with its subtree evidence,
  sigma_eff = sigma_r / sqrt(1 + subtree evals) (the natural
  posterior-contraction scaling; structural, not tuned).
- C-D **no per-level clip** (isolation control for the [0.01, 0.99]
  clip's role).
- Combinations of the above as needed, smallest-set-first.

Readouts (registered):
1. Deficit closure: mover-persp (mu - own-subtree-avg) of the
   visit-picked arm at the root, per counterfactual; baseline -0.089.
   Success bar: some smallest set closes >= 2/3 of the deficit
   (to within -0.03) while moving the mu-picked arms' tracking
   (+0.010 baseline) by no more than +/-0.02.
2. Ranking repair: fraction of the 27 mu-vs-visit disagreements where
   the counterfactual root argmax-mu switches to the visit-picked arm
   (or to an arm with deeper verification than the baseline pick).
3. The per-level deficit profile along the principal line (depth curve
   of node mu - subtree avg), baseline vs counterfactual — the
   compounding claim made measurable.

Decision rule: the smallest closing set becomes the M8 mechanism and
proceeds to phase 2 (bmcts twin cell + engine implementation, each
with its own registered gate). If NO counterfactual closes the
deficit, the recursion is re-derived from the generative model with
subtree-verified evidence as a first-class evidence type (the
"rethink" branch, its own registration).

Honest limitation, recorded: counterfactual re-integration is STATIC —
it re-scores trees built by the baseline dynamics; allocation feedback
(a fixed integrator would build different trees) is not captured.
Phase-2 gates are the arbiter; phase 1 selects the mechanism.

## Phase-1 outcome (2026-07-18, append-only)

Base validation: the Python bottom-up re-integration reproduces the
engine's root arm mus to 1.9e-15 across all 30 hybrid trees — the
counterfactual machine is exact.

**No registered toggle meets the closure bar (2/3 of -0.089).**
  base -0.0889 | A -0.0948 | B -0.1106 | C -0.0582 | D -0.0949
  BC -0.0915 | AC -0.0479 | AB -0.1295 | ABC -0.1061 | ABCD -0.1119
- C (exclude verified arms' stale first evals from the Stein evidence)
  is the best single surgery: closes ~35%; best combo AC ~46%.
- B (evidence-aware eD) BACKFIRES (-0.111): the E[max] premium term is
  not the culprit in the naive direction — sign structure through the
  max/min alternation is more entangled than the toggle assumed.
- Anchor mean-refresh (A) alone does not help (variance refresh was
  explicitly out of scope pending derivation).

**Diagnostic limit E** (verified children carry their plain subtree
average with variance v_bar/visits — NOT a candidate, the bound):
switch rate 11/14 (root argmax-mu joins the visit ranking); deficit
trivially 0 at the root by construction. Together with the toggle
failures this settles the attribution: the defect is STRUCTURAL to the
recursion — no local term-surgery reaches it — and the corrected
recursion must asymptote to the subtree average as verification
accumulates.

**The registered re-derivation branch FIRES.** Phase 2 is therefore a
derivation task before any implementation: extend the generative model
with subtree-verified evidence as a first-class evidence type — a
child's delivered value must enter with (mean, variance) reflecting its
realized verification (variance contracting toward v_bar/visits-scale,
NOT re-shrunk against stale first-eval anchors), with the per-level
E[max]/Stein corrections derived to vanish in that limit. The testbed
never exposed this because its budget spread keeps subtree-verified
children rare; the bmcts twin cell (deep concentrated lines) is part of
phase 2 so the fix can be gated there first. Constants from derivation
or labeling data only, never match results.

## Phase 1b — repaired-recursion prototype (registered before running)

Derivation (2026-07-18, from the phase-1 attribution): the structural
defect is that VERIFIED-CHILD EVIDENCE NEVER FLOWS INTO THE SHARED
LEVEL C. The set's C-estimate comes only from the frozen anchor and
children's stale first evals, so unverified siblings' means stay
anchored to first-glance information while the verified line moves —
every E[max/min] then weighs an updated arm against stale phantoms.
(Toggle B failed because it REMOVED verified arms from the D-moments
rather than CONDITIONING on them.)

Toggle R (prototype recursion, per node, bottom-up):
1. C-posterior: precision-weighted fusion of (a) the anchor's
   C-estimate (mean anchMu - eD, variance c0 = A0 + vD - 2g — its
   fixed weight makes it wash out as evidence accumulates), (b) each
   verified child: mean mu_a - m_a, variance v_a + sigma_r^2 (v_a =
   the child's cf total claimed variance), (c) each fresh eval: mean
   e_a - m_a, variance sigma_a^2 + sigma_r^2.
2. Set assembly: verified arms (mu_a, v_a); fresh arms = product
   fusion of prior N(Chat + m_a, VarC + sigma_r^2) with N(e_a,
   sigma_a^2); unrevealed arms N(Chat + m_a, VarC + sigma_r^2).
3. Node value: E[max] via the existing Clark machinery; prototype
   carries all variance as private (b = 0) — the two-component
   (shared-s) split is deferred to the formal phase-2 derivation and
   noted as a prototype approximation.
Success bar unchanged (close >= 2/3 of -0.089; flip >= 2/3 of the
misrankings; mu-picked tracking within +/-0.02 of baseline).
No fitted constants: every quantity comes from existing heads or the
recursion itself.

## Phase-1b outcome + R2 registration (appended before R2 runs)

Toggle R closes only ~8% (-0.0815): phantom-riding repairs the
staleness but exposes the deeper defect — the E[max] premium is
POLICY-INCONSISTENT. Under (floored d-means, sigma_r = 0.119) each of
~60 phantom arms claims ~10% exceedance over the top arm where the
policy says ~0.1%; summed over the set this manufactures the per-level
premium. The premium is an order-statistics object and must be fed
order-statistics-consistent moments — exactly the role of the
P(best) inversion (M5: equivalent to the regression in-distribution,
divergent precisely in tail exceedance).

Toggle R2 = R with unverified-arm d-means from the policy inversion:
per node, solve the Gaussian-field means over the FULL legal prior
vector (no 5e-3 floor) such that P(arm a is max) = p_a with
s = sigma_r (M5 Amendment A solver, tau = 1). Verified arms unchanged
(they carry their posteriors). No fitted constants: s is the shipped
sigma_r; the inversion is a reparameterization of the policy itself.
Same success bar.

## Phase 2 (forward commitments, own docs before any run)

- bmcts twin: a deep-verified-line cell class (depth >= 8, allocation
  concentrated PUCT-style) where the baseline stack must REPRODUCE the
  under-crediting; the fix must close it there and stay non-inferior
  on every existing cell (M6-style two-tier scoring).
- Engine gate: match vs stock PUCT at B = 64, 9x9, game-level scoring:
  Elo + deviation rate (37.5% baseline) + per-move excess leak
  (-0.012 baseline) + the M7 replay readouts on fresh positions.
  No constant tuned on match results; anything fitted is fitted on
  labeling data with an M1-style registered protocol.

Scripts: python/bayes_m8_audit_all.py (dump collection),
bmcts scripts/m8_counterfactual.py (re-integration), written after
this doc's commit. Results append-only as always.
