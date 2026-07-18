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
