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

## R2 outcome (2026-07-18, append-only) — MECHANISM VALIDATED

  base -0.0889 | R -0.0815 | R2 **-0.0041** (95% closure; bar 2/3: MET)
  switch 5/14 (bar 2/3: NOT met); muPick track +0.0650 vs baseline
  +0.0366 (bar +/-0.02: NOT met).

Reading: the two derived ingredients — verified-child evidence fused
into the shared level C, and phantom/unverified moments made
POLICY-CONSISTENT via the P(best) inversion (s = shipped sigma_r, no
new constants) — eliminate the verified-line under-crediting almost
completely. The remaining misrankings and the thin-arm optimism drift
are attributed to a PROTOTYPE simplification, not the theory: R2
replaced the Stein joint shrinkage with naive per-arm eval fusion, so
a lucky fresh eval carries ~3x its proper weight. The tracking clause
caught exactly this.

**Phase-2 derivation is now fully specified:** keep the complete
two-component Stein structure for fresh evals; replace the d-mean
feature vector with the inversion means (full legal set, no floor);
add the verified-child C-evidence terms to the shared-level estimation
(anchor becomes one voice that washes out); derive the D-moments
conditional on verified arms. Then: re-prototype against the same bar
(all three clauses), golden-extend (tie-inclusive inputs, stable-sort
convention both sides), bmcts twin cell, engine gate — each step its
own registered doc per the working agreements.

## R3 registration (appended before running): the exact-GLS recursion

The phase-2 derivation, executed numerically-exactly instead of by
closed form. Per node, latent theta = (C, s, delta_1..delta_k) with
priors: C flat (1e6), s ~ N(0, var_s), delta_a ~ N(0, sigma_r^2)
centered on the INVERSION means m_a (full legal set, no floor,
s_inv = sigma_r). Observation rows:
- anchor: anchor_mu = C + sum_a pi_a delta_a + eps, the E[D]
  linearization with pi = Clark argmax weights of (m_a, sigma_r^2);
  offset E_D - sum pi_a E[delta_a]; noise A0 + max(v_D - sigma_r^2 *
  sum pi_a^2, 0).
- fresh eval (evaled, subtree evals <= 1): e_a - m_a = C + delta_a + s
  + u_a, u_a ~ N(0, v_u), per-arm sigma from the head, shared s.
- verified (subtree evals >= 2, frozen, in cf): mu_a - m_a = C +
  delta_a + w_a, w_a ~ N(0, v_a) with v_a the child's cf variance
  (first eval NOT re-included — it lives inside the child posterior
  via the child's anchor, per the freeze semantics).
Posterior of V*_a = C + m_a + delta_a exactly (information-form
solve); one-factor P-var-style projection on the C-error (V_X =
Var(C_err), b_a = Cov(V_a, C_err)/V_X, v_priv = remainder, floor 0);
E[max] via the existing Clark quadrature. Node passes (clip(E[max]),
total variance, 0) upward as in R/R2 (two-component pass-up deferred
to the engine-form derivation).

Exactness claims: joint shrinkage of lucky fresh evals is native
(shared C and s soak cross-arm surprise); verified evidence pins C
with weight v_a + sigma_r^2; the anchor's fixed A0 washes out. No
fitted constants. Same three-clause bar.

## R3 outcome + R4 registration (appended before R4 runs)

R3: deficit -0.0029 (97%), but switch 4/14 and muPick track +0.0886 —
the thin-arm optimism WORSENED. Attribution: the inversion means
over-spread the prior field. Under an INDEPENDENT Gaussian field,
policy-consistent P(best) forces the top arm's mean ~2 sigma above the
pack (~+0.2), but M5 measured true value gaps saturating far below
that. Order-statistics consistency and value calibration cannot
coexist in an independent field — P(best) concentration in reality
comes from CORRELATION (siblings share the position), not mean
separation.

R4 (correlated-field recursion): delta_a = shared + private; the
shared part is unidentifiable within-set and merges into C (flat), so
the model reduces to R3 with (i) d-means from the REGRESSION head
(value-calibrated; unfloored form), (ii) per-arm private delta prior
s_eff^2 for arms without subtree evidence, where s_eff is solved
per-node (1-D) so that P(top-prior arm is best) under (m_reg, s_eff^2)
matches its policy prior (bounds [0.005, sigma_r]); (iii) VERIFIED
arms keep the wide sr^2 delta prior — the Gaussian field is the
narrow branch of a mixture whose collapse branch (tactical
refutations) verification identifies; without this release, a tight
prior would shrink refutations away and reinstate the original
disease. Total marginal spread stays head-calibrated; exceedance
becomes policy-calibrated; no fitted constants (s_eff is a per-node
solve, not a tuned scalar).
Predictions: deficit stays closed; muPick track returns toward
baseline (tight priors absorb lucky fresh evals into C/s); switch
rate rises. Same three-clause bar.

## R4 outcome (2026-07-18, append-only) — phase-1 prototype ladder closed

  base -0.0889 | R2 -0.0041 | R3 -0.0029 | R4 -0.0200
  switch: 5/14 | 4/14 | 3/14; muPick track: +0.065 | +0.089 | +0.069
  (baseline +0.037; bar: deficit closure MET by all three, ranking and
  tracking clauses MET by none).

Ladder verdict: the deficit mechanism (verified-to-C flow +
policy-consistent exceedance) is VALIDATED three ways over; the
residual defect has moved: picked arms at 2-10 visits are
verified-class in R4 (they carry recursive posteriors, not lucky
evals), so the +0.07 tracking error originates in shallow-subtree cf
values or in the C posterior itself — a different, smaller object
than the -0.089 we started from. Phase 2's first task is therefore a
PER-LEVEL tracking diagnostic inside the counterfactual machine
(mu - avg by subtree depth/visits under each recursion) to localize
the residual before the formal two-component derivation is finalized.
The three-recursion ladder also fixed two theory points for the
paper: the independence axiom of the sibling field is FALSE on real
policies (P(best) concentration is correlation, not mean separation),
and no single Gaussian field satisfies both value calibration and
order-statistics calibration — the correlated-field factorization
(common part in C, private s_eff, verification releasing the wide
branch) is the resolution.

## Per-level diagnostic + R5 registration (appended before R5 runs)

Per-level tracking (hybrid dumps, mover mu - avg by subtree visits):
base -0.109/-0.101/-0.091/-0.079 across [2,4)/[4,8)/[8,16)/[16,65);
R4 -0.096/-0.066/-0.057/-0.041. Under-crediting persists at EVERY
level; the root's positive picked-arm gap is selection over the
residual spread. The floor is the phantom premium charged at every
opponent node: R4's s_eff enforces policy-consistent exceedance only
for the top arm; mid-mass phantoms still overlap.

R5: the unified field. d-means from the INVERSION COMPUTED AT s_eff
SCALE — the joint solve: s_eff chosen so the inversion means (full
legal set, s = s_eff) best match the regression means on the top-8
arms (the M1-fitted support; least squares, 1-D solve over s_eff in
[0.005, sigma_r]) — then those inversion means are the field means.
Every arm's P(best) is policy-consistent by construction; top-arm
value gaps match the measured regression; tail premiums collapse to
the true policy tail mass. Verified arms keep the wide sr^2 release
as in R4. Everything else identical to R4. Same three-clause bar,
plus the per-level curve reported.

## R5 outcome + decision-quality readout (2026-07-18, append-only;
## closes the phase-1 prototype ladder)

R5: deficit -0.0107, switch 3/14, track +0.0788 — same shape as R4.

Labeled decision quality (picked move's deep value vs PUCT's pick,
mover persp, on the 30 audit positions — NOTE: this set was SELECTED
as the baseline's worst disagreements, so all improvements below are
optimistic; nothing here is confirmatory):
  base -0.167 | R2 -0.104 | R4 -0.144 | R5 -0.127
**The proxy metrics decouple from decision quality**: R3/R4/R5 close
the visPick-deficit proxy better than R2 yet pick WORSE moves. R2 —
verified-to-C fusion + inversion-consistent phantom moments, naive
fusion — is the ladder's best decision-maker and the phase-2
mechanism candidate.

Ladder conclusions carried to phase 2:
1. The phase-2 bar must BE labeled decision quality on FRESH
   positions (registered sampling, disjoint from every set used so
   far), not any belief-space proxy.
2. R2-core is frozen as the candidate mechanism; the formal
   derivation should produce its principled form (the R3-R5
   refinements helped proxies, hurt decisions — parsimony wins).
3. Static re-scoring cannot show the dynamic effect (a fixed
   integrator builds different trees); the engine implementation
   behind a flag + fresh-position paired replay (M7 protocol) is the
   arbiter before any match gate.

## R6 registration (appended before running): the contrast-space
## recursion — bias-invariance by construction

Derivation principle (from the truth-curve finding): selection quality
requires bias EQUALITY across arms, not bias absence. Parameterize the
recursion so that everything that cannot differ between arms never
enters. Per node, with reference arm r (most subtree evals; tie ->
higher prior): latent contrasts g_a = V*_a - V*_r.

Evidence, all in contrast space:
- Prior: g_a ~ N(dm_a - dm_r, 2 sigma_r^2) (unfloored d-mean
  contrasts — the object M5's pairwise protocol validated).
- Reference level Lr = reference arm's subtree average (childAvg;
  node's own eval when nothing is visited). Levels carry the shared
  optimism; they are never compared across arms.
- Fresh eval of arm a: y_a = e_a - Lr, noise (1 - rho) s_a^2 +
  vbar/visits_r — the measured sibling correlation rho = 0.26 finally
  works FOR us: the shared component cancels in the contrast.
- Verified arm a: y_a = avg_a - Lr (the PUCT contrast, inheriting its
  invariance), noise (1 - rho) vbar (1/visits_a + 1/visits_r).
Posterior per arm: Gaussian product of prior and its evidence
(cross-arm correlation through Lr is second order for ranking;
recorded as a prototype approximation). CHOICE: argmax posterior
contrast mean (g_r = 0). LEVEL passed upward: Lr + E[max(0, g)] via
Clark on the contrast posteriors.

What this buys structurally: thin arms' contrasts carry wide evidence
noise, so their posteriors shrink toward the (M5-calibrated,
conservative) prior contrast — one lucky eval cannot win the argmax;
deep arms' contrasts are PUCT contrasts with prior discipline; the
whole gradient axis (evidence-dependent level bias) is out of the
estimate by parameterization. Scoring: labeled pick quality vs PUCT
(primary, with the selection caveat), truth-curve flatness by evidence
class, plus the ladder's proxies for continuity.

## R6 outcome (2026-07-18, append-only) — new mechanism candidate

Labeled pick quality vs PUCT's pick, ALL 30 positions (7 previously
unlabeled R6 picks labeled first — they were worse than the labeled
ones, moving R6 from -0.024 to the honest -0.054):
  base -0.164 (83% worse, 3/30 same move)
  R2   -0.105 (63% worse, 7/30 same)
  R6   -0.054 (40% worse, 17/30 same)
The contrast-space recursion cuts the baseline decision gap 3x on the
audit set — which was SELECTED as maximally adversarial to bayes-style
choosers — and supersedes R2-core as the phase-2 mechanism candidate.
Its truth-curve is not flat (implied levels inherit Lr's bias; the
DECISION statistic is the contrast, which the pick-quality readout
scores directly).

Caveats, recorded: (a) static re-scoring on baseline-built trees;
(b) selection-adversarial and NEAR-DUPLICATE positions (four paired-
game opening repeats found — phase-2 sampling must dedupe by position
hash; effective n < 30); (c) single-eval label noise on individual
picks.

**Phase-2 spec, updated:** formal derivation of the contrast-space
recursion (reference-arm parameterization, two-component contrast
noise with the rho dividend, premium in contrast space); engine
implementation behind a flag (the recursion is SIMPLER than the
current one — no anchor freeze, no E[D] subtraction, levels from
subtree averages); fresh deduped labeled positions; gate criteria =
labeled pick quality + truth-curve flatness by evidence class +
deviation/leak diagnostics; then the match gate.

## Phase 2a registration (engine contrast chooser; appended before
## implementation)

Standing goal (Cal, 2026-07-18): iterate until CBTS beats PUCT.
Staged: (2a) engine R6 chooser -> (2b) fresh-position replay, parity
bar -> (2c) match, stop-the-bleeding bar -> (2d) voi allocation on
contrast beliefs, the win attempt.

2a implementation: new SearchParams flag `useBayesContrastChooser`
(requires useBayesSearch passenger state; orthogonal to
useBayesSelection). getChosenMoveLoc computes the R6 root pick:
reference arm = most child visits (tie: higher prior); mover-persp
contrast prior dMean * (log p_a - log p_r), UNfloored, variance
2 sigma_r^2; evidence = child-average or first-eval contrast against
the reference level with (1 - rho)-scaled noise per the R6
registration; pick = argmax posterior contrast (reference at 0).
Exact port of scripts/m8_counterfactual.py::_contrast_node's root
behavior. Verification protocol: run with KATAGO_BAYES_AUDIT on >= 10
positions; recompute the pick from the engine's own final dumped root
state with the Python reference; every pick must match exactly.

2b: fresh positions = the SECOND eligible bayes-to-move turn per
decisive lost game (the M5 extension rule, deterministic), deduped by
full move-prefix; paired replay contrast-bot (stock PUCT allocation +
contrast chooser) vs puct, picks labeled at 500 visits. Registered
bar: paired pick-quality difference within +/-0.01 of PUCT (parity;
the chooser shares PUCT's information, so parity is the honest
target), deviation-rate reported. 2c/2d get their own registrations
after 2b's outcome.

## 2b outcome + 2c registration (appended before the match runs)

**2b PARITY BAR MET, dead center.** 324 fresh second-turn deduped
positions, contrast bot vs puct, all picks labeled: paired pick
quality -0.0000, 95% CI [-0.0069, +0.0070] (bar +/-0.01); same move
63.9%; disagreements split 45/47 (mean -0.0001); thin-pick rate 1.2%
(baseline 28.5%). The per-decision defect is closed on fresh data.

2c match (registered): 300 games, exactly the match64-rerun protocol
(B = 64, temp 0, resign -0.95/6, komi 7, area/positional-ko, policy
opening init, numGameThreads 8) with bot1 = contrast (useBayesSearch
passenger + useBayesContrastChooser; stock allocation; NO
useBayesSelection) vs bot0 = stock puct. Purpose: confirm no
game-level/trajectory regression that static parity could hide.
Registered expectation: Elo within +/-40 of zero (parity chooser,
mostly-shared moves; resignation reads standard stats in both bots).
This is the stop-the-bleeding gate, NOT the win attempt; 2d (voi
allocation on contrast beliefs) is registered separately after 2c.
Output: bayes-data/match-m8-2c/ (append-only), config
cpp/configs/bayes_m8_match.cfg.

## 2c outcome + 2c-b registration (appended before the diagnosis run)

**2c: -135 Elo (W 38 / L 149 / D 113, n = 300). Bar (+/-40) FAILED;
+231 Elo recovered vs the -366 baseline** (wins 8 -> 38, draws
49 -> 113); all 149 losses by resignation (slow leak, not blunders).
Reading: per-decision MEAN parity (2b) does not survive compounding —
with ~36% deviation rate per move and near-symmetric small errors, the
drawish komi-7 landscape converts error variance asymmetrically
(minus flips draws to losses more easily than plus flips draws to
wins vs a solid opponent). Deviating on coin-flip contrasts buys
variance with no mean edge.

2c-b (deviation gate, registered): rerun the 2b fresh-position replay
with KATAGO_BAYES_AUDIT dumps; from each final root state compute the
contrast posterior mean AND variance per arm (the verified Python
mirror); for a z-threshold ladder, the thresholded pick = contrast
pick if its posterior z = g/sd(g) clears z*, else the reference arm.
Score every thresholded pick with the existing labels (labeling any
missing reference-arm picks). Pin z* = the smallest z whose surviving
deviations have labeled mean improvement >= +0.005 with a
cluster-bootstrap CI excluding 0 (if none, z* = the argmax of
deviation mean, reported honestly as parity-targeted). Engine param
`bayesContrastDeviationZ` implements the gate; the confirmatory match
(2c rerun protocol) runs ONCE with z* pinned. Elo never tunes
anything. Registered predictions: (1) deviation quality rises with z;
(2) at z*, match Elo lands in [-40, +40] if surviving deviations are
parity, positive if the CI-excluding-zero branch fired.

## 2c-b outcome (2026-07-18, append-only) — GAME-LEVEL PARITY

**Gated contrast chooser (z* = 1.25): -12 Elo (W 30 / L 40 / D 230,
n = 300; ~+/-19 Elo at 1 sigma). Registered parity bar [-40, +40]
MET.** Both 2c-b predictions confirmed: deviation quality rose with z
(ladder: -0.013 ungated -> ~0 at z >= 1.25); the fallback (parity)
branch fired and the match landed on parity.

The Elo arc of the M8 campaign, same protocol throughout: -366
(baseline stack) -> -135 (contrast chooser ungated) -> **-12 (gated)**.
Wins vs stock PUCT: 8 -> 38 -> 30, draws 49 -> 113 -> 230. CBTS now
plays even with stock PUCT at B = 64 on 9x9, with a chooser that is
derived, verified bit-exact against its reference, and pinned entirely
from labeling data.

What parity is and is not: the gated bot earns it largely by trusting
the reference arm (PUCT's allocation) and deviating on the 5.9% of
moves where its contrast posterior is confident — and those deviations
are label-neutral, not yet label-positive. The WIN must come from
information PUCT does not have: 2d re-enables bayes ALLOCATION (voi-KG
routed on contrast-space beliefs) so the tree itself is built where
the posterior says information is decision-relevant — the layer where
the testbed's 47/48-cell superiority actually lives. 2d requires the
contrast-space selection derivation (KG in contrast units), its own
registration, and the full gate ladder (replay, then match).

## 2c-c registration (budget sweep; appended before any run)

Rationale (Cal, 2026-07-18): PUCT's known holes — entrenchment of
lucky early evals via visit momentum, and inability to escape a wrong
policy prior — are largest at SMALL budgets, where argmax-visits is
itself noise; the calibrated posterior corrects both by construction.
Protocol: the 2c-b match protocol verbatim (gated contrast chooser,
z* = 1.25 UNCHANGED at every budget — no per-budget tuning), 300
games per budget, maxVisits in {8, 16, 32, 128} (64 already run:
-12). ALL results reported regardless of outcome; the sweep is a
measurement of the regime-response curve, not a selection procedure.
Registered predictions: (1) relative Elo improves monotonically-ish
as B falls; (2) the first positive-Elo regime, if any, appears at
B <= 16. Confirmation rule, fixed now: any budget with positive Elo
gets ONE N = 1000 confirmation run (one look); a win claim requires
the confirmation CI to exclude zero. Output:
bayes-data/match-m8-2cc-B<N>/ (append-only).

## 2c-c outcome (2026-07-18, append-only) — FIRST CONFIRMED WIN

Sweep (n = 300 each, z* = 1.25 everywhere): B=8 +16, B=16 +47,
B=32 -85, B=64 -12, B=128 -15. Confirmations (n = 1000, one look,
registered rule):
- B=8: -48 +/- 8 — the sweep's +16 was selection noise; the
  confirmation rule caught it (the M4 lesson, working as designed).
- **B=16: +52 +/- 7 Elo (W 295 / L 147 / D 558). CI excludes zero.
  CBTS DEFEATS STOCK PUCT AT B = 16.** 2:1 win ratio, protocol
  identical to the M3 gate matches, every constant from derivation or
  labeling data, Elo never used as a tuning signal, prediction
  registered before the sweep ran.

Regime reading: at B = 16 visit counts carry almost no information
and PUCT's known failure modes (early-eval entrenchment, unescapable
policy prior) are maximal; the calibrated contrast posterior is the
only informative statistic available — and it wins. At B >= 32 PUCT's
verification statistics come online (the B=32 dip -85 is a real
anomaly to attribute before any broader claim — possibly the worst
point of the tradeoff between our gate trusting a still-thin
reference arm and PUCT's averages starting to mean something).

Open lines, in order: (a) attribute the B=32 dip; (b) 2d contrast-voi
allocation — the win attempt at standard budgets; (c) the bmcts twin
+ formal write-up of the contrast recursion for the paper's engine
section (destination form: the method and its demonstrated property —
superiority in the low-budget regime, parity at standard budgets).

## 2c-d registration (high-budget extension; appended before any run)

Cal (2026-07-18): measure B > 1000. Protocol: 2c-c verbatim (gated
contrast chooser, z* = 1.25 unchanged, n = 300 per budget), maxVisits
in {256, 1024, 2048}. Registered predictions: (1) draw rate rises
with B toward saturation as both bots converge on the net's resolved
minimax; (2) Elo stays within the parity band [-40, +40] at every
tested budget — deviations become rare as the reference arm coincides
with PUCT's pick; (3) the alternative worth watching: if the B=32-dip
mechanism is scale-dependent rather than a one-off, a dip recurs at
some budget — any point outside [-40, +40] gets the N = 1000
confirmation rule before being treated as real. All results reported.
Output: bayes-data/match-m8-2cd-B<N>/ (append-only).

## 2c-d outcome (2026-07-18, append-only)

  B=256: -21 (W9/L27/D264) | B=1024: -19 (W35/L51/D214) |
  B=2048: -43 (W20/L57/D223; ~+/-16 at 1 sigma — OUTSIDE the band,
  N=1000 confirmation launched per the registered rule; not treated
  as real until it reports).

Prediction (2) holds through B=1024 (parity everywhere above the
B=32 dip). Prediction (1) VIOLATED: draw rate fell 88% -> 71% -> 74%
from 256 up — deeper search converts small edges rather than
saturating to draws, and the extra decisive games lean against us.

Registered attribution lead (for the 2048 anomaly AND possibly the
B=32 dip, same shape from the other side): the contrast noise model
has NO ERROR FLOOR — evidence noise ~ vbar/visits means claimed
precision grows without bound while the net's systematic biases do
not average away; at high B the gate's z clears threshold on
systematic-bias differences and "confident" deviations turn
label-negative against an increasingly-correct reference. Measurable
by an M5-style contrast-calibration-by-visit-bucket study on labeled
replay data; the fix, if confirmed, is a derived floor term in the
contrast noise (label-pinned, never Elo-tuned).

## 2d registration (contrast-voi allocation; appended before
## implementation). The B=2048 N=1000 confirmation was STOPPED
## unread on Cal's reprioritization (2026-07-18); it re-queues after
## 2d.

Design (the contrast-KG derivation):
Per node, the contrast state (reference r = most subtree evals, tie
higher prior; posteriors g_a ~ N(m_a, v_a), g_r = 0) already computed
for the chooser becomes the ALLOCATION currency:
- Decision weights: w = Clark argmax weights over the contrast field
  {g_r = 0} u {g_a} (mover persp) — P(arm attains the local max).
- One-visit variance drops: for a non-reference arm, D_a = v_a -
  v_a', where v_a' uses evidence noise with n_a + 1 subtree evals.
  For the REFERENCE arm, one visit tightens vLr, which sits in EVERY
  arm's evidence noise (the shared term): D_r = sum_a [v_a -
  v_a'(vLr')]. Verifying the reference improves all contrasts at
  once — this is what makes deep principal-variation trees emerge
  naturally instead of breadth (the old voi's level-variance currency
  had no such coupling).
- Root selection: argmax_j overlap-weighted drop, contested_j x D_j,
  the M2-registered routing form with contrast quantities (contested
  = the decision-boundary overlap density from the same Clark pass).
  Interior nodes: route by the local w (the local decision is the
  node's value, which feeds the parent's contrast).
- Unvisited arms enter through their prior contrasts: only arms whose
  prior contrast overlaps the decision boundary are contested —
  exploration is policy-gated by construction, tail arms are never
  visited (the 5e-3-floor pathology cannot recur).
No new constants: everything derives from the existing heads and the
already-shipped contrast state. Flag: useBayesContrastSelection
(requires useBayesSearch + useBayesContrastChooser).

Gate ladder (registered): (i) tree-shape telemetry on the M7 replay
protocol — P-2d1: depth comparable to PUCT (mean >= 8), breadth far
below old voi (<= 10), chosen-visits median healthy; (ii) fresh-
position labeled pick quality — P-2d2: >= PUCT (parity floor;
positive = the win signal); (iii) matches: B = 64 first (P-2d3:
non-inferior, >= -40; positive at N = 1000 = THE WIN), then the
budget curve. Elo tunes nothing, as always.

## 2d smoke outcome + 2d-b registration (appended before the fit runs)

**P-2d1 FAILS**: contrast-voi builds breadth 12-55 / depth 3-4 trees —
the M3 shallow-broad pathology in contrast units. Diagnosis: (i) the
one-step KG is myopic — a first touch always kills more variance than
a deepening step; (ii) the evidence-noise model vbar/n makes
deepening's marginal value decay ~1/n^2, far faster than reality.
Both the 2c-d high-B anomaly (claimed precision without a floor) and
this allocation failure (claimed precision growing too fast in n) are
the SAME modeling error from opposite ends: the noise curve
nv(n) = (1-rho) vbar / n is wrong in shape.

2d-b (the noise-curve study, registered): fit the EMPIRICAL contrast
error vs subtree visits from existing data — the m8-audit full-tree
dumps carry childAvg at every visit count for hundreds of arms whose
deep labels are in the m5/m7/m8 pools. Fit |avg_n - label| structure
as nv(n) = a * n^(-p) + c in LOG space (the v7 lesson), held-out by
game parity (even fit / odd validate, M1-style). The fitted curve
replaces vbar/n in BOTH the 2c-b gate's noise model and the 2d
allocator's D. Registered predictions: p < 1 (slower-than-1/n decay:
correlated subtree evidence), c > 0 (systematic-bias floor). If the
fit fails its held-out calibration ([0.7, 1.4] band by visit bucket),
neither consumer changes and the myopia problem needs a structural
(non-myopic) treatment instead.

## 2d-b outcome (2026-07-18, append-only)

4,570 (visits, sqerr) samples from the existing dumps x label pools.
**The empirical contrast-error curve is nearly FLAT: err 0.202 at
n=1 -> 0.159 at n=40; fitted err^2(n) = 0.0514 n^-0.15 + 0 —
p = 0.15 vs the model's assumed p = 1.** The engine's noise model
overclaims precision ~50x at n=50 (claimed 0.0006 vs measured 0.0286).
Registered prediction p < 1: CONFIRMED decisively (c fit to 0: at
p = 0.15 the power law IS the floor over this range). This single
mis-shape explains both 2d's breadth pathology (deepening's marginal
value ~1/n^2 under the model vs ~n^-1.15 in reality) and the high-B
phantom-precision anomaly.

Held-out calibration: 5/8 buckets in [0.7, 1.4]; three OUT, all on
the conservative side (0.62-0.66) — per the registered rule the
parametric fit DOES NOT SHIP into the consumers yet. Suspected cause:
position heterogeneity flattening the pooled curve (each position has
its own error scale). Next (own registration before running): the
normalized variant — fit the RATIO curve err^2(n)/err^2(1) with
per-position scale from the sigma heads — then, if in band, consume
in the 2c-b gate noise model and the 2d allocator D, and re-smoke
tree shape. Caveats recorded: n <= 64 range only; labels share the
net's systematic bias with the averages, so absolute levels are
understated — the SHAPE in n is the load-bearing measurement.

## Normalized-curve outcome + 2d-c registration (appended before
## implementation)

Normalized ratio curve (err^2 / head-claimed s^2, 4,570 samples):
MEAN stays flat in n (even split 1.6 -> 1.2 over 40x visits; 1/n
predicts -> 0.04) while the MEDIAN falls ~7x — the mean is dominated
by a tail of arms whose error NEVER improves at 64-visit depth
(tactically unresolved lines: averaging measures the wrong quantity
precisely). Conclusion, theoretically grounded: search information is
RESOLUTION-LIMITED; any variance-drop currency (level or contrast)
misprices allocation because the mean-square it models is
tail-dominated by irreducible resolution error.

2d-c (resolution allocation): root score = contested_j alone (the
contrast-field overlap density — buy resolution AT THE DECISION
BOUNDARY; the top-two/TTTS principle, v5-validated in the testbed);
interior nodes route by local w (extend the PV of the line under
resolution); D is DROPPED (model refuted by 2d-b), not re-fitted.
Viability: non-terminal. No constants at all. Predictions: P-2dc1
tree shape inverts — depth >= 8, breadth <= 10, root visits
concentrated on the top 2-3 contested arms; P-2dc2 replay pick
quality >= parity; P-2dc3 matches: B = 64 >= -40 (win = positive at
N = 1000), B = 16 win retained, B = 32 dip reduced (a coherent
allocator removes the suspected gate/allocation mismatch).

## 2d-d registration (observed subtree volatility as per-arm evidence
## noise; Cal's proposal, 2026-07-18; appended before implementation)

Insight: 2d-b showed mean contrast error is resolution-limited (flat
in n), dominated by a tail of tactically-unresolved arms — but WHICH
arms are unresolved is OBSERVABLE: the realized variance of a
subtree's sampled values (quiet lines agree; tactical lines swing with
the sampled reply). This is the deep-subtree empirical counterpart of
the shorttermWinlossError head already consumed at n = 1.

Implementation: per-arm contrast evidence noise becomes
  nv_a = (1 - rho) * (v_obs,a + vLr-term),
where v_obs,a = the arm subtree's realized value variance (derived
from the node stats utility second moment, winloss-scaled — recorded
prototype approximation), floored by the head claim at small n
(empirical-Bayes handoff: head at n <= 2, observed at n >= 3, floor
at max(head^2 * 0.25, tiny)). NO division by n (the 2d-b flat curve);
NO new constants beyond the structural handoff at n = 3. Both
consumers inherit automatically: contested widens on volatile arms
(the resolution allocator drills contested-and-volatile lines) and
the deviation gate distrusts volatile averages.

Predictions: P-2dd1 tree shape retains depth, with visits shifting
toward volatile contested lines; P-2dd2 replay pick quality improves
over 2d-c specifically on positions whose chosen-line volatility is
high; P-2dd3 the match ladder (64/16/32) at or above 2d-c's results.

## 2d-c outcome (2026-07-18, append-only)

P-2dc1 MET (breadth 2.6 / depth 10.9 over 324 positions — the shape
finally inverts). P-2dc2 borderline (paired pick quality -0.0031, CI
[-0.0090, +0.0022]: parity, not better; same-move 71.3%). **P-2dc3
FAILS: B=64 -57, B=16 -61 (the +52 win does NOT survive
own-allocation), B=32 -118.**

The campaign-wide law this makes quantitative: Elo loss tracks the
DIVERGENCE RATE from PUCT's move when divergence quality is neutral
(~-2 Elo per % divergence: 36% -> -135, 29% -> -57, 5.9% -> -12).
Parity-mean divergence is never free in the drawish komi-7
environment. Therefore the explicit bar for ANY allocation/chooser
change: divergences must be POSITIVE-MEAN (>= ~+0.005 labeled), or
the configuration must reduce toward PUCT behavior. The B=16 win was
the noise-reduction channel (rare, gated deviations vs PUCT's noisy
visit-argmax), not a better-moves channel — recorded plainly.

2d-d (volatility noise, already registered/implemented) proceeds as
the next rung: per-arm observed volatility could raise divergence
QUALITY (deviate only where volatility says the reference average is
untrustworthy). Its ladder readouts decide against the explicit bar
above.

## 2d-d outcome (2026-07-19, append-only)

**P-2dd2 REFUTED**: paired pick quality -0.0075, CI [-0.0141,
-0.0013] (excludes zero, negative) vs 2d-c's -0.0031; divergence rate
rose (31.8% vs 28.7%); disagreement mean -0.0240. Fails the
registered positive-mean bar at the replay rung; matches not run
(sequential gating). Attribution lead, recorded not proven: the
utility-second-moment proxy for subtree volatility is CONTAMINATED —
score-utility swings and natural minimax drift inflate v_obs on
healthy deep lines, so trustworthy averages get second-guessed. The
clean retest (if pursued) requires a winloss-only spread accumulator
in BayesNodeState plus a label study validating that v_obs separates
resolved from unresolved arms BEFORE it re-enters any consumer.

## Campaign position after 2d-d (recorded for the phase-2 write-up)

Confirmed, pre-registered results: (1) CBTS beats stock PUCT at
B = 16: +52 +/- 7 Elo, n = 1000; (2) game-level parity at standard
budgets with the gated contrast chooser (-12 at B = 64, parity
through B = 1024); (3) the divergence-rate law: ~-2 Elo per %
neutral-quality divergence at komi 7 — the standard-budget Elo
channel is saturated by PUCT's shared-error cancellation unless
divergence quality is strictly positive, a bar no allocation variant
has yet cleared. Domain findings with standalone value: the flat
noise curve (p = 0.15), resolution-limited information, the
contrast/bias-invariance principle, and the observable-volatility
hypothesis (open pending a clean proxy).

## 2d-e registration (the tree variogram; Cal's identification,
## 2026-07-19; appended before the measurement runs)

Identification: the p = 0.15 noise curve and "correlation between a
position's evals and its descendants' evals" are the same measurement
— err^2(avg_n) = sigma^2 [rho_bar + (1 - rho_bar)/n], so a flat curve
IS high within-subtree error correlation (plus a non-stationary tail
where the subtree's values drift from the arm's own — the max-vs-mean
population). The model measured sibling correlation (rho = 0.26) and
assumed within-subtree correlation ZERO (the /n); reality inverts
this: within-subtree is the LARGEST entry. The correct noise model
for every consumer is the single underlying object: the covariance of
eval errors as a function of TREE RELATIONSHIP (nested random-effects
/ variogram over the tree).

Measurement (existing data only): from the m8-audit full-tree dumps
(every node's first eval + structure) x root-arm labels: for each
labeled root arm, treat descendant evals' mover-corrected deviations
from the arm's label as error proxies (recorded caveat: interior
minimax drift inflates deep-relationship entries — reported, not
corrected, in this first pass); compute error products by
relationship class: (i) parent-child, (ii) siblings (validation
against M1's 0.26), (iii) same-subtree depth-2/3+, (iv) cousins
(different root arms, same position — the cross-subtree entry that
sets what contrasts do NOT cancel). Even/odd game split as always.
Registered predictions: P-2de1 within-subtree correlation >>
sibling 0.26; P-2de2 cousin correlation is materially lower than
within-subtree (this gap is the contrast principle's quantitative
justification and the ceiling on contrast-evidence quality).

## 2d-e outcome (2026-07-19, append-only) — the object CBTS must model

First-pass variogram (error-vs-shared-root-label proxy; drift
inflation recorded, clean pass needs descendant labels): sibling
1.13* / subtree-d2 0.87 / parent-child 0.77 / subtree-d3+ 0.67 /
COUSIN 0.42 (pooled eval-error sd 0.193). P-2de1 CONFIRMED: within-
subtree correlation is 0.67-0.87 where the noise model assumed 0 —
the /n was wrong by the scale of the phenomenon. P-2de2 CONFIRMED:
the cousin entry sits ~0.3-0.45 of variance below every within-
subtree entry — the component contrasts cancel and averaging cannot;
the quantitative basis of the contrast principle AND the floor on
contrast-evidence quality. (*sibling 1.13 vs M1's 0.26: different
estimand — deep siblings vs a SHARED label include shared true-value
drift; M1 measured fresh evals vs own labels.)

Synthesis (the campaign's theoretical destination, stated once): eval
errors in a search tree follow a NESTED covariance over tree
relationships. Every phenomenon this campaign measured is a
projection of that object: the flat noise curve (within-subtree
entries), the contrast principle's success (cousin gap), PUCT's
robustness (its statistics only ever compare quantities whose shared
components cancel), the divergence-rate law (deviating spends the
cousin gap), and the failure of every variance-drop currency
(they price the cancelled component). The machinery that converts
CBTS's calibration into Elo must be derived FROM the measured
variogram — evidence noise, KG currency, and gate thresholds all fall
out of its entries. Next (each its own registration): (i) clean
variogram with descendant labels; (ii) re-derive the contrast noise
and allocation currency from it; (iii) the ladder again. The
low-budget win (+52 at B=16) and standard-budget parity stand as the
demonstrated results meanwhile.

## 2d-f outcome (2026-07-19, append-only) — the clean variogram

345 labeled nodes / 30 trees / 30 games (11.5 per position; below the
runbook's loose 600-900 note, above the sanity floor; CI widths
reflect it). Own-label errors, pooled err sd 0.245.

By relationship (corr, [cluster-bootstrap 95% CI], n pairs):
  sibling       0.607 [0.386, 0.794]  159
  parent-child  0.712 [0.591, 0.808]  263
  ad-gap2       0.668 [0.494, 0.816]  160
  ad-gap3+      0.678 [0.515, 0.838]   59
  cousin-near   0.299 [0.084, 0.482]  157
  cousin-root   0.288 [0.037, 0.505] 1031
By tree distance: 0.712 / 0.638 / 0.440 / 0.397 / 0.255 / 0.160 /
0.053 (d = 1..7) — a monotone variogram decaying to ~0 by d ~ 7.
Even/odd agreement: within-subtree entries stable (+/-0.05); cousin
splits unstable (0.19/0.43 vs 0.43/0.10) — few independent games
dominate; recorded.
Secondary stErr-standardized table: same structure, slightly lower
(sibling 0.52, cousins 0.19-0.27).

Predictions: P-2df1 CONFIRMED (AD entries 0.67-0.71, CI lower bounds
>= 0.49, far above sibling-0.26). P-2df2 2/3 CONFIRMED (sibling,
gap-2 de-drift; gap-3+ 0.678 vs 0.674 indistinguishable — the
expected deep-entry drift inflation did not materialize, possibly
offset by the first pass's anchMu-clip attenuation). **P-2df3
CONFIRMED — cousin-root smallest; the within-subtree-minus-cousin gap
(~0.35 of variance) SURVIVES de-drifting: the contrast principle
stands on measured ground; the derivation proceeds (no re-attribution
stop).** P-2df4 MISSED informatively: AD correlation is PERSISTENT in
gap (0.71/0.67/0.68 — no decay down a line), and the cousin
LCA-ordering is direction-only and split-unstable.

Un-predicted structural headline, recorded for the derivation: the
tree-DISTANCE table is a clean monotone variogram (near-geometric,
corr(d) ~ 0.7 x ~0.65^(d-1), zero by d ~ 7) — the measured covariance
object in its simplest sufficient form. Next rung (own registration):
derive contrast evidence noise + allocation currency from this
variogram (nested-effects/distance-kernel form), then the ladder.

## 2d-g registration (variogram-derived noise; appended before
## implementation)

Model (from 2d-f; the P-2df4 gap-persistence miss is the structure):
TWO-LEVEL NESTED covariance — within-root-arm plateau w_in = 0.65
(the AD/sibling plateau), cross-arm w_cross = 0.29 (cousin-root
0.288), idiosyncratic 1 - w_in = 0.35. Pins are label-derived (2d-f
table), never Elo-derived. Cross-checks satisfied by construction:
subtree-average error sigma^2 [w_in + (1-w_in)/n] reproduces the
2d-b flat curve; root-contrast noise at n = 1 gives
2 sigma^2 (1 - w_cross) = 1.42 sigma^2 = the measured fresh
cousin-root pair variance.

Consumer forms (both chooser and selection; replaces the refuted 2d-d
volatility noise AND the (1-rho) prefactor — rho = 0.26 was the
fresh-sibling eval correlation, superseded for root contrasts by
w_cross):
  contrast evidence noise, arm a vs reference r:
    nv_a = vbar * [ 2 (w_in - w_cross) + (1-w_in)(1/n_a + 1/n_r) ]
  (n = subtree evals, >= 1; eval-only arms n = 1; the floor
  2(w_in - w_cross) = 0.72 vbar is permanent — claimed contrast
  precision is bounded, by measurement.)
  Prior contrast variance pv = 2 sigma_r^2 unchanged. Allocator:
  2d-c contested-only, inheriting the floored posteriors.

Protocol: implement; goldens + suite; chooser mirror re-verified
(>= 10 positions exact); tree-shape smoke; z* RE-PINNED via the 2c-b
ladder machinery on fresh dumps under the new noise (labels only —
the old 1.25 was pinned under the old noise and is void); then the
match ladder B = 16/32/64 vs stock PUCT, n = 300, with the 2c-c
confirmation rule (N = 1000, one look) for any point outside
[-40, +40]. Registered predictions: P-2dg1 the floor cuts the
deviation rate at high verification and the B=32 dip shrinks
materially; P-2dg2 B=16 stays positive; P-2dg3 B=64 within the
parity band or better.

## 2d-h registration (the fade rate; Cal's budget-generality critique,
## 2026-07-19; appended before any run)

Critique accepted and recorded: the 2d-g constants compress a
budget-free object (the net's error-inheritance law) into a
shallow-horizon snapshot; flat-at-0.7-over-3-plies cannot distinguish
permanent inheritance from ~2%/move fade, and those diverge exactly
where budgets grow. The general model (three budget-free components):
  corr(same line, gap g)   = G + L * theta^g
  corr(across a fork)      = G (approximately, per 2d-f cousins)
with G = board-wide component, L = lineage component, theta = per-move
fade. The engine then derives any subtree's noise AT RUNTIME from the
tree it actually built (per-node accumulators, next rung) — no
budget-indexed constants anywhere. 2d-g's fixed floor is this model's
shallow-tree limit and stays valid where validated (16-64 visits).

Measurement (this rung): 60 lines rolled out by repeated 500-visit
searches (top move each step, 30 plies or stop at pass/decided),
from every-7th m5 first-turn position (deterministic). Per step: raw
1-visit eval + the search's own 500-visit value as label. Sidelines:
at depths 5 and 15, the search's #2 move gets one eval+label node
(long-range cross-branch entries). Errors e_d = raw_d - label_d;
corr by gap 1..30 pooled, cluster bootstrap by line, even/odd split;
fit (G, L, theta) by least squares on the gap curve.

Registered predictions:
- P-2dh1 (cross-validation): gaps 1-3 reproduce 2d-f's ~0.7 plateau.
- P-2dh2 (the decision): corr declines materially by gap 20-30 —
  operationalized: corr(gap 20 bucket) < 0.45. If instead corr(30)
  >= 0.6, inheritance is stubborn and the constant floor IS the
  budget-general law (either outcome settles the design).
- P-2dh3: sideline (cross-fork) entries sit near the 2d-f cousin
  level ~0.3 and are roughly gap-independent (they estimate G).

Scripts: python/bayes_m8_theta_lines.py (collection),
bmcts scripts/m8_theta_fit.py (fit), written after this commit.
Output bayes-data/m8-theta-lines.jsonl, append-only.

## 2d-h outcome (2026-07-19, append-only) — THE FADE IS REAL

60 lines (40 live, 30 reaching 20+ plies; dead starts = decided
lost-game positions, wasteful not biasing). Pooled err var 0.0215.
Corr by gap: 0.48/0.59 (1-2) -> 0.36 (3-5) -> 0.19 (5-8) -> 0.12
(8-17) -> 0.017 (17-23, CI [-0.12, +0.16]) -> 0.08 (23-31, noise).
**P-2dh2 DECIDED FOR FADE: corr(20-bucket) = 0.017 << 0.45. Fit:
corr(g) = 0.000 + 0.605 x 0.875^g — half-life 5.2 plies; the
board-wide component fits to ZERO at this horizon.** Twenty plies of
strong play launder the original misjudgment essentially completely.

P-2dh1 PARTIALLY MISSED, informatively: at overlapping gaps (1-3)
these strong-play lines give 0.48-0.36 where 2d-f's search-tree pairs
gave ~0.7 flat — a real population difference. Reading: strong play
RESOLVES questions (fast fade); search trees DWELL on unresolved
branches (slow fade). Both datasets are right about their own
populations; engine subtrees resemble the latter, engine PVs the
former. Pooled error variance also differs 3x (0.0215 vs 0.0602) —
same heterogeneity. P-2dh3 mixed: sideline cross-fork corr +0.206
(vs fitted G = 0; sidelines share fork context and span mixed gaps —
not over-read). Caveats owned: stop-at-decided censors pairs at
resolution events; even/odd unstable at 30-line clusters.

Design consequence (the budget-general architecture): the LAW is
geometric fade along lines + near-complete shedding at forks; the
RATE is population-dependent, so no global constant is admissible —
which the accumulator design never needed anyway. Next rung (own
registration): per-node running tallies applying fade edge-by-edge
over the actual tree (dwelling unresolved subtrees keep high
correlated-noise floors automatically; deep resolving lines shed them
automatically; budget appears nowhere), with local fade tied to
observables (the stErr head predicts unresolvedness) rather than a
pinned theta. 2d-g's fixed floor stays as the validated shallow-limit
until the accumulators pass their own ladder.

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

## 2c-e registration (policy-baseline control; laptop, appended before
## any run)

Cal's deflation hypothesis (2026-07-18): production never runs below
~1000 visits, and the raw policy is known (to Cal) to outplay MCTS at
very low visit counts — so the 2c-c B=16 win may be the POLICY
ANCHOR's win: at 16 evals the z* = 1.25 gate rarely fires and the
gated chooser ~= a raw-policy bot, while 16-visit PUCT actively hurts
itself stirring thin-search noise into a strong prior. No raw-policy
control exists anywhere in the M7/M8 record; the 2c-c "the calibrated
contrast posterior is the only informative statistic available — and
it wins" reading is unsupported until this control runs.

Protocol (2c-c verbatim except bots): bot0 = stock PUCT at the cell
budget; bot1 = raw-policy bot (maxVisits = 1, all bayes flags off);
temp 0 both; n = 300 per cell; cells in order B = 16, 32, 8, 64.
Plus deviation telemetry: gated contrast chooser (2c-b flags) on 60
fresh positions (seed base 500500), B in {16, 32, 64}: fraction of
chosen moves differing from the policy argmax.

Registered predictions:
- P1 (Cal): policy bot >= parity vs PUCT at B = 8 and 16.
- P2: policy-vs-PUCT Elo declines monotonically in B; below the
  [-40, +40] parity band by B = 64.
- P3 (decisive readout): machineryMargin(16) := CBTS Elo(+52) minus
  policyElo(16), reported with CI. If |machineryMargin| < 2 SE the
  B=16 headline is anchor-dominated and the 2c-c regime reading is
  RETRACTED in favor of "the gate correctly inherits the policy's
  low-budget strength that PUCT destroys"; confirmation rule: that
  boundary case triggers n = 1000 on the B=16 cell.
- P4 (unified 16/32/64 story): gated deviation rate < 10%/move at
  B=16, rising substantially by B=32 — the -85 dip = the immature-
  deviation window (gate opens before the posterior earns it).
All results reported, append-only, run on the laptop (Eigen) while
the desktop runs 2c-d.

## 2c-e outcome (2026-07-18, laptop, append-only) — deflation hypothesis
## REFUTED; the B=16 headline stands; margin structure is about deviation
## QUALITY, not quantity

Policy-baseline curve (raw policy vs stock PUCT, n=300/cell, temp 0):
B=8 -27 +/- 20, B=16 -36 +/- 20, B=32 -62 +/- 20, B=64 -139 +/- 22.

- **P1 (Cal's lore) FALSIFIED**: the raw policy loses to PUCT even at
  8 visits, on this net/9x9/temp-0 protocol. The lore may hold for
  older nets or 19x19; not here.
- **P2 CONFIRMED**: monotone decline, far below the parity band by 64.
- **P3 (decisive): the 2c-c B=16 headline STANDS.** machineryMargin =
  CBTS - policy: ~-21 (B=8), **+88 (B=16)**, ~-23 (B=32),
  **+127 (B=64)** — the B=16 and B=64 margins are > 3 SE positive.
  Not anchor-dominated; no retraction; the boundary-case n=1000 rule
  was not triggered (margin >> 2 SE).
- **P4 MISSED as registered**: gated deviation rate at B=16 is 13.3%
  (predicted < 10%), and the rise to B=32 (18.3%) is mild, not
  substantial (n=60/cell, ~+/-5%; B=64: 16.7%). The registered
  "gate-mostly-shut" mechanism is wrong in detail. What the data
  supports instead: deviation RATE is roughly flat in budget; the
  margin swings (+88 / -23 / +127) are deviation QUALITY. Hypothesis
  for the B=32 window (registered as hypothesis only, for the
  testbed/desktop): at 16 the contrasts clearing z* are first-eval-vs-
  prior cases (crisp); at 32, 2-3-visit subtree averages feed the
  contrasts — the partial-verification noise regime M7 identified —
  and at 64 verified-line credit stabilizes.

Protocol notes, owned: (1) the B=64 cell's output dir was contaminated
by an accidental duplicate relaunch AFTER the registered run completed;
the second run is VOID (near-duplicate deterministic games — 7,915
total NN evals for "300 games" proves cache-replay) and scored
separately (-32) only to document the contamination; the registered
first run (-139) is separable by file mtime. (2) A real engine bug
surfaced: with useBayesSearch and numSearchThreads > 1, the beginSearch
guard fires on the ASYNC search thread and gtp hangs silently in
waitForSearchToEnd instead of reporting the error — the guard should
run before the async spawn (desktop-side fix suggested).
Artifacts: docs/bayes-m8-2ce-telemetry.json; match dirs
bayes-data/match-2ce-B{8,16,32,64}.

## 2d-f registration (clean tree variogram — descendant labels;
## 2026-07-19, appended before any labeling run)

Purpose: 2d-e's first pass measured error products against the SHARED
root-arm label, so every deep entry is inflated by true-value drift
along the line (minimax drift is common to both members of a pair and
enters the product as signal, not noise). Engine testing has meanwhile
confirmed the practical need: the correlation infrastructure must
cover ancestor–descendant structure, not just siblings — the
descendant cascade shifts means but represents no descendant
covariance. The clean estimand is the covariance of FIRST-EVAL errors
about each node's OWN value: e_v = eval_v − L_v, where eval_v is the
raw first NN eval recorded in the parent's arm entry (evalWinrate —
not the clipped anchMu the first pass had to fall back to) and L_v is
a 500-visit deep label at v's own position (the M1/m5 labeling rung;
reportAnalysisWinratesAs = WHITE throughout, so no mover correction).
Terminal-evaled arms excluded. Root nodes excluded (no parent arm
entry, hence no clean first eval).

Sampling (deterministic, from the existing m8-audit *_hybrid.jsonl
dumps; python/bayes_m8_label_descendants.py): per audited position,
the two root arms with the largest subtrees plus the next-largest
evaled arm (cousin diversity); within each deep arm, the principal
(max-childVisits) chain to relative depth 4, plus the largest evaled
sibling at relative depths 1–3. <= 15 labels/position (fewer where
dumps are shallow). Labels append to
bayes-data/m8-desc-labels.jsonl keyed by (game_hash, turn, path);
the script is resumable (skips already-labeled keys).

Analysis (bmcts scripts/m8_variogram_clean.py): pair all labeled
nodes within a tree; classify by path-prefix LCA — sibling,
parent-child, ancestor–descendant gap 2, gap 3+, and cousin SPLIT by
LCA depth (LCA = root vs LCA >= 1, the entry 2d-e could not resolve);
plus the full table by tree distance (d1−l)+(d2−l). Correlation =
mean error product / pooled mean-square error. Even/odd game split
(int(gh[-1],16) % 2) as always. Cluster bootstrap over games for the
headline classes. Secondary, non-gating readout: the same table on
stErr-standardized errors (what a consumer that knows the head's
sigma actually faces).

Registered predictions:
- P-2df1: ancestor–descendant correlation with OWN labels stays
  materially above M1's sibling 0.26 — the flat noise curve is an
  error-correlation fact, not a shared-label artifact.
- P-2df2: every clean entry sits below its 2d-e drift-inflated
  counterpart (sibling < 1.13, d2 < 0.87, d3+ < 0.67).
- P-2df3: the cousin-at-root entry remains the smallest, and the
  within-subtree minus cousin gap survives de-drifting — the contrast
  principle's quantitative basis must hold about OWN values, not just
  about a shared label.
- P-2df4 (soft, the nested-field signature): among cousin pairs,
  correlation increases with LCA depth; among ancestor–descendant
  pairs, it decreases with gap.

Scope: this rung is measurement only. No model fit, no engine change,
no consumer touches. The sequence stands as registered in 2d-e:
(i) this table; (ii) re-derive contrast noise and the allocation
currency from the measured covariance (own registration — fitting a
generative field, e.g. an AR-on-tree kernel, happens there and only
against this table); (iii) the ladder again.

## 2d-i registration (the resolution kernel: budget-general
## correlated-noise accumulators; 2026-07-19, appended before any
## fit script or engine change)

This is rung (ii) of the 2d-e sequence and the accumulator design
pre-committed in the 2d-h outcome. It replaces the 2d-g class
constants (w_in, w_cross) — a validated shallow-horizon snapshot —
with a generative error field whose only inputs are per-node
observables, so that every noise quantity the engine consumes is
derived at runtime from the tree it actually built. No constant is
indexed by budget, depth, or node class.

### The field (single mechanism)

Plain-language statement first: a network eval is wrong mostly
because of unresolved questions in the position (an unsettled fight,
an unclear group). Those questions are inherited move by move — a
child position mostly contains its parent's open questions — until a
move RESOLVES one, at which point the inherited part of the error
collapses with it. The stErr head is the net's own estimate of how
much unresolved uncertainty a position carries, so the ratio of
child-head to parent-head measures how much question-mass an edge
carried through. That is the whole model.

Formally: first-eval error at node v is
  e_v = s_v * ( sqrt(A) * B_v + sqrt(1-A) * eps_v )
with s_v = the M1-corrected head sigma (bayesSigmaFromStErr), eps_v
private white noise, and B_v a unit-variance lineage-bias field with
product-form correlation along tree paths:
  corr(B_u, B_w) = prod over edges e on the u..w path of phi_e
  phi_e (parent p -> child c) = theta0 * min(1, s_c / s_p)^gamma
Three constants total — A (shared fraction of eval error), theta0
(retention on a dwelling edge, ratio ~ 1), gamma (how strongly a head
drop sheds inherited bias) — pinned on LABEL DATA ONLY (below).
Implied pairwise covariance: Cov(e_u, e_w) = A * s_u * s_w *
prod(phi_e). Fork shedding is not a separate mechanism: a path that
turns at a fork simply crosses more edges, and edges into
less-contested siblings are resolution edges (head drops), which is
where the shedding comes from — testable, not assumed.

### Reconciliation ledger (what one constant set must explain)

The review standard for this rung: the SAME (A, theta0, gamma) must
be consistent with every correlation this campaign has measured,
with the population differences carried entirely by the observable
s-ratios, never by per-population constants.
  C1 dwelling plateau: search-tree pairs (2d-f) — parent-child
     0.712, ad-gap2 0.668, ad-gap3+ 0.678, sibling 0.607. Dwelling
     edges have ratio ~ 1, so predictions ~ A*theta0^d, near-flat
     for theta0 near 1.
  C2 depth split of siblings: root-fresh siblings 0.26-0.29 (M1 rho;
     m8-2cb fresh-pair variance 1.42*vbar) vs deep siblings 0.607.
     The model's account: m5 roots are contested (high parent head);
     their children partially resolve (ratio < 1), so root fork
     edges carry low phi; deep dwelling forks carry phi ~ theta0.
     This is the make-or-break check — class constants cannot
     explain a depth-dependent sibling correlation; the observable
     either does or the model dies here.
  C3 strong-play fade: 2d-h lines, corr(g) ~ 0.605 * 0.875^g. On
     resolving lines the head steps down repeatedly, so per-edge phi
     ~ 0.875 and the g -> 0 intercept estimates A ~ 0.6.
  C4 cousins: 0.288 (root) / 0.299 (near) — longer paths, more
     resolution edges.
  C5 the 2d-b flat noise curve: within a dwelling subtree phi ~
     theta0 ~ 1 makes the pair mass Q grow like n^2, so the
     subtree-mean error variance never gains 1/n — the measured
     err^2(n) ~ n^-0.15 plateau, now derived rather than pinned.
  C6 divergence-rate law compatibility: nothing in the kernel
     changes prior means; only claimed precisions move. Deviations
     must still clear the re-pinned z*.

### Fit protocol (labels only; bmcts scripts/m8_kernel_fit.py)

Data, all already collected — no new labeling:
  (a) 2d-f clean pairs: m8-desc-labels.jsonl, pairs whose connecting
      path is fully covered by labeled nodes (principal-chain
      sampling makes most pairs path-complete; report the skip
      count). Root stNode per tree from the m8-audit root records.
  (b) 2d-h line pairs: m8-theta-lines.jsonl, all step pairs, plus
      sideline cross-fork pairs (fork node on the chain).
Errors standardized per node by the corrected head sigma (drop
s < 1e-3); observed products z_u * z_w vs predicted A * prod(phi_e);
weighted least squares on the pooled pair set (weight 1/pair,
cluster bootstrap by game resp. line for CIs); grids A in [0.3,
0.9] step 0.02, theta0 in [0.60, 1.00] step 0.01, gamma in [0, 3]
step 0.1. Populations are pooled — one constant set, that is the
point.
Baselines the kernel must beat on joint SSE: (i) gamma = 0 (pure
constant-retention AR — no observable); (ii) the 2d-g class model
scored on the same pairs (w_in same-root-arm / w_cross cross-arm on
tree pairs, w_in flat on line pairs). Pin rule: lowest joint SSE;
ties (within 2%) break toward fewer effective constants.

Registered fit predictions:
- P-2di1: the kernel beats baseline (i) by >= 30% joint SSE and
  baseline (ii) cross-population by more (2d-g has no gap axis).
- P-2di2: gamma > 0 with bootstrap CI excluding 0 — resolution
  (the head drop) is the fade carrier. This is Cal's volatility
  mechanism in its surviving form: the head's predicted volatility,
  not realized sample spread (2d-d refuted the latter).
- P-2di3: the fitted kernel reproduces the C2 depth split within
  bootstrap CIs using measured s-ratios (no class constants).

### Engine forms (registered now, implemented only if the fit pins)

Per-node accumulators in BayesNodeState, updated in
bayesRecomputeNodeStats from the set state (children recompute
before parents on the backup path, so child accumulators are always
current; off-path subtrees are unchanged hence still valid). Sigma-
weighted so heteroscedasticity is native and vbar disappears:
  n_v = 1 + sum_c n_c                (evals in subtree)
  S_v = s_v + sum_c phi_c * S_c      (fade-weighted sigma mass)
  Q_v = s_v^2 + sum_c Q_c + 2*A*s_v*sum_c phi_c*S_c
        + A * [ (sum_c phi_c*S_c)^2 - sum_c (phi_c*S_c)^2 ]
with phi_c = theta0 * min(1, s_c/s_v)^gamma on the edge v -> c, and
terminal-evaled children contributing n only (s = 0, exact values
carry no NN error). Derivation of Q (the ordered-pair decomposition;
each within-child pair keeps its own A from Q_c, every pair that
first meets at v gets A once):
  sum over ordered pairs (u,w) in subtree(v) of Cov(e_u,e_w)/A-units
  = own-own (s_v^2) + own-descendant both orders (2 s_v sum phi S)
  + within one child (sum Q_c) + across children (cross products).
Consumers (chooser AND contrast-voi selection, same forms):
  Var(subtree-mean error of arm a)  = Q_a / n_a^2
  Cov(arm a mean, arm r mean)       = A * phi_a * phi_r * S_a * S_r
                                       / (n_a * n_r)
  contrast evidence noise: nv = Q_a/n_a^2 + Q_r/n_r^2 - 2*Cov
  eval-only arm: n = 1, S = s_a, Q = s_a^2.
  anchor fallback (ref unvisited): reference level = anchMu as now;
  the anchor is the node's OWN eval, one lineage edge above the arm:
  nv = Q_a/n_a^2 + s_node^2 - 2*A*phi_a*S_a*s_node/n_a.
  Numerical floor 1e-9 only — NO modeling floor: fully resolved
  deep lines legitimately earn near-zero noise. That is the M7
  under-crediting fix arriving through the front door, and it is
  what makes this rung an ALLOCATION theory: dwelling subtrees keep
  Q ~ n^2 (more evals there buy nothing, VOI collapses), resolving
  subtrees shed Q (their evidence sharpens, deviations become
  affordable), so contested-overlap allocation and the chooser both
  inherit resolution-awareness from the same accumulators.
  Prior contrast pv = 2*sigma_r^2 and the dMean prior: unchanged.
Audit dump gains per-arm accS, accQ, accN, phi fields (additive;
consumers are our own mirrors).

### Ladder (fixed by precedent, in order)

1. Build; 596 goldens byte-identical; full runtests from cpp/.
2. Chooser mirror updated (bayes_m8_verify_contrast.py +
   bayes_m8_2cb_analyze.py contrast_posteriors): 12/12 exact.
3. Tree-shape smoke: bayes_m7_replay.py --bots contrastvoi,puct
   --second-turn --limit 6 (breadth/depth sane, no hangs).
4. z* re-pin via the 2c-b ladder on fresh dumps under the kernel
   noise (labels only; same pin rule: smallest z with deviation mean
   >= +0.005 CI-excl-0, else parity-targeted argmax).
5. Match ladder vs stock PUCT, bayes_m8_match*.cfg protocol,
   B = 16/32/64, n = 300 each; N = 1000 one-look confirmation for
   any point outside [-40, +40]. Elo tunes nothing, ever.

Registered match predictions:
- P-2di4: B=16 stays positive (the +52 regime survives the noise
  swap; shallow trees sit in the kernel's dwelling limit ~ 2d-g).
- P-2di5: the B=32 dip (-85) closes to within [-40, +40] — the dip
  is the partial-verification window, and pricing 2-3-visit
  dwelling subtrees at Q ~ n^2 (no false 1/n gain) is exactly what
  the kernel changes there.
- P-2di6: B=64 within the parity band or better.
Disposition: if the fit fails P-2di1/2, the 2d-g constants stand and
this doc records the kernel as refuted — the class model would then
BE the law and budget-generality fails empirically, which is
reportable. If matches fail P-2di4, ship state remains 2d-g.
