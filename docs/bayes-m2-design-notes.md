# M2 design notes — sequential-reveal shrinkage (needs testbed validation)

The one genuine mechanism change (not translation) the port requires,
recorded before implementation. Per the working agreements this design
must be prototyped and gated in the bmcts testbed before the KataGo
implementation of it is trusted; the C++ side may scaffold it behind the
flag in the meantime.

## The gap

bmcts `shrink_siblings` assumes expansion reveals ALL k sibling evals at
once; its joint posterior (anchor + k evals) and the exact one-shared-
variable-X collapse depend on every child loading identically on the
anchor blend. KataGo reveals NO child evals at expansion — a child's
first eval arrives at its own first visit, one at a time, in
search-chosen order (which is itself selection-biased: good-prior
children get evaluated first).

## Proposed generalization: partial-observation shrinkage

State per sibling set: the anchor measurement (m_A, V_A) of C (offset by
c_k*sigma_d as in bmcts), the set's shared eval draw s ~ N(0, var_s),
and the subset E of children with evals {e_a}. Recompute the set's joint
posterior from scratch whenever E grows (same recompute-from-state shape
as the rest of the stack; no incremental drift):

- C | anchor, {e_a, a in E}: precision p_A = 1/(V_A + var_s) as before,
  p_e = |E| / (sigma_d^2 + v_u) with e_bar over E only.
- Evaluated children: mu_a = C_hat + w*(e_a - m_t) as before (w, m_t
  from the |E|-child versions of the bmcts formulas).
- Unevaluated children: mu_a = C_hat (+ a policy-prior mean offset, see
  below), v_priv = sigma_d^2 + (C-posterior variance apportioned via the
  same X bookkeeping), b = 1 on the set's X.
- The X decomposition (kappa_alpha etc.) needs re-deriving for |E| < k;
  the BLUP orthogonality argument in the shrink_siblings docstring
  should go through with k -> |E| for the eval terms, but this is
  exactly what the testbed prototype must check (MC vs generative
  process, the v6 gate-1 style).

Selection-ordered reveal is the second thing the testbed must check:
children are evaluated in search-preference order, so the evals fused
early are a biased subsample of the set (the ts+inf selection-bias
lesson from v6 says this can matter; voi's determinism does not remove
the ordering bias). Gate: calibration of unevaluated-sibling posteriors
under sequential reveal with selection-ordered vs random-ordered
fusion.

## Policy-prior mean offset (new head, fittable from M1 data)

bmcts had no per-child mean signal at expansion; KataGo has the policy
prior. A d-mean head E[V*_a - C | prior_a] would give unevaluated
children distinct means instead of all sitting at C_hat. The M1 dataset
already records (prior, deep value) per child, so this head is fittable
without new data. NOT required for M2 correctness (mu_a = C_hat is the
prior-free special case); record held-out value before adopting.

## Testbed verdict (bmcts, 2026-07-12) — gate FAILED, requirements below

The prototype + pre-registered gate ran in the testbed
(bmcts: docs/seqreveal-m2-gate.md, results/seqreveal-m2*.log,
shrink_siblings_partial + Bayes(seq_reveal=True)). Answers to this
note's open questions, and the conditions the C++ implementation must
now satisfy:

1. **The BLUP orthogonality argument DOES survive k -> |E|** (the
   identity only needs w = sigma_d^2/v_n) — the evaluated block keeps
   the all-at-once structure exactly (GLS-verified). **But the exact
   one-shared-X collapse does NOT survive |E| < k**: the posterior is
   two-block exchangeable (eval-eval / unev-unev / cross) and a
   one-factor state can honor only two of the three blocks. Use the
   variance-matching projection (b_unev = sqrt(Var(A)/V_XE), private =
   sigma_d^2 exactly): the BLUP-projection alternative biases the
   parent Clark max up to z-mean +0.48 at k=20/low |E|; P-var passed
   every well-specified max band.
2. **The policy-prior d-mean offset head is REQUIRED, not optional.**
   Value-ordered reveal (the realistic policy-ordered case) makes E a
   biased-high subsample: prior-free unevaluated-sibling posteriors hit
   z-mean +1.9. This head must ship with M2, fit from the M1 data.
3. **The dropped Cov(alpha, d_a) anchor echo is exposed at low |E|**
   (claimed max-level variance up to ~2x true at rho=0/strong anchors;
   conservative-only). All-at-once bmcts masks it; sequential reveal
   lives exactly where it bites. Un-dropping it (spread_table already
   computes covDd) is a NEW mechanism change: derive + gate in the
   testbed before porting.
4. **Adaptive descent adds a seq-specific ~+0.1..+0.19 root optimism**
   vs matched all-at-once eval counts (optional stopping within sets;
   absent under uniform descent, invariant to projection) — the
   selection-aware-accounting roadmap item now has a measured M2 form.
5. **The voi routing statistic does not survive partial reveal as-is**
   (voi-seq regret WORSE than uniform-seq at eval budgets while ts-seq
   is healthy). Re-derive contested x resolvable under partial
   observation before trusting voi in the engine.
6. What held: unevaluated-sibling posteriors (the object descent
   consumes) calibrated in-search at every rho under every policy; the
   recompute-from-scratch-per-reveal shape is sound.

## Testbed round 3 (bmcts, 2026-07-12, same day) — preconditions LANDED

Rounds 3 resolved requirements 1–5 above (bmcts:
docs/seqreveal-m2-round3-gate.md; results/seqreveal-m2-round3.{json,log}).
**The sequential-reveal stack to port is frozen:** Stein-corrected
d-mean-aware posterior (`shrink_siblings_stein`) + prior d-mean head +
KG routing, P-var projection — in bmcts flags:
`seq_reveal, use_stein, use_dmean, policy="voi", seq_kg`.

1. **Stein anchor**: one closed form covers the un-dropped Cov(alpha,
   d_a) AND per-child prior means (GLS-validated at every |E|). With it,
   child-level and pairwise claims are EXACT under the generative
   process (z std [0.97, 1.03]). New PROOF, exportable: the anchor echo
   is only half covariance-fixable — the exact n=0 posterior of the
   set's max given the anchor has variance anchor_var exactly (the
   anchor measured C+D and the target IS C+D; the cancellation is
   max-functional tail dependence). Stein removes exactly 2g of the
   2 v_D claimed excess; the remainder is a conservative-only max-level
   residual at strong anchors. It ships as documented conservatism —
   do NOT attempt to fix it with any covariance-level model in C++.
2. **d-mean head**: REQUIRED and a large regret win (roughly halves
   testbed regret at 100 evals vs the best headless variant; best root
   calibration on the board). Key theory point for the port: reveal-
   order bias is a MEASURABILITY question — any order the engine can
   realize is a function of the policy prior the posterior already
   conditions on, hence bias-free (prior-ordered reveal calibrates to
   |z mean| 0.033). The scary true-value-ordered drift is an
   unrealizable oracle stress. Fit the head on M1 data; residual in
   LOG space per the v7 lesson; sigma_r misreport degrades kappa-like
   gracefully (2x conservative, 0.5x mildly optimistic).
3. **Routing**: under the per-eval cost model the currency is the
   predicted ONE-REVEAL variance drop D (data-independent, two dummy
   posterior calls; backup D = max_i w_i D_i — the old G-backup shape
   with honest units), NOT total resolvable variance R. voi-KG beats
   voi-R everywhere (p <= 0.005), beats uniform at B=100 (p <= 0.028;
   voi-R LOST to uniform), ties Thompson. R keeps only the
   exhaustion-gating role.
4. **Selection residual**: largely dissolved by the round-3 stack at
   B=100 (-0.04..+0.07 vs matched all-at-once); +0.13..+0.19 pockets at
   B=30, rho >= 0.6 remain — tracked under selection-aware accounting,
   not a port blocker.
5. Still open before the FULL recommended stack exists in seq mode:
   use_infer (regional-bias inference) is not re-derived for growing E
   — port order should be seq stack first, inference second.

## What M2 ships regardless of the above

- The exact math library (bayesposterior.{h,cpp}): Clark pair/fused
  max+weights, overlap density, std-max moments, 9-pt GH
  node_posterior_from_children, all-at-once shrink_siblings — golden-
  tested against the Python reference.
- SearchParams plumbing (useBayesSearch + head-coefficient params),
  consumed by nothing until the state/backup branch lands.
- The backup identity is unchanged by sequential reveal: once children
  have posterior state, recomputeNodeStats' Bayes branch is exactly
  node_posterior_from_children + the beta_kids relabelling (bmcts
  _recompute, use_shrink path), with R = sum w_i R_i.
