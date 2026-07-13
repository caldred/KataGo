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
