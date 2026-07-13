# Bayesian search port plan (bmcts → KataGo)

Port of the validated bmcts stack (github.com/caldred — `~/Code/bmcts`,
v3–v7: two-component error filter, hierarchical shrinkage, decision-relevant
VOI descent, regional-bias inference; no statistically significant loss to
per-config-tuned PUCT anywhere in its test grid) into KataGo's search,
behind a config switch, gated the same way the testbed was: calibration
before strength, pre-registered predictions, no constant tuned on match
results.

## What the testbed validated (frozen stack to port)

- Per-node Gaussian posterior with a two-component error decomposition
  (private + sibling-shared), moment-matched max backup (Clark + 9-pt
  quadrature over the shared draw).
- Fresh children hierarchically shrunk toward the parent anchor.
- Descent: deterministic decision-relevant VOI — contested × resolvable
  (root: tie-boundary density = dE[max]/dv_i; interior: dVar(max)/dv_i ≈
  p_argmax) — with resolvable variance `R` backed up as Σ w_i R_i.
- Regional-bias inference: one Kalman step per sibling set per first
  expansion, from the residual between surface opinion and children-derived
  verdict (h = b_pre − b_post; R from the shrinkage decomposition).
- Heads: fit calibration models on data, in LOG space for scale variables
  (the v7 lesson: a linear spread head was catastrophically overconfident).

## What the map found (cpp/, v1.16.3)

- **Selection fork point**: `selectBestChildToDescend`
  (search/searchexplorehelpers.cpp:324); the human-SL branch there is the
  pattern for a policy switch. PUCT itself: `getExploreSelectionValue`
  (:38). FPU: `getFpuValueForChildrenAssumeVisited` (:265).
- **Backup hook**: `recomputeNodeStats` (search/searchupdatehelpers.cpp:151)
  — already a full recompute-from-children per backup, structurally the
  same shape as bmcts `_recompute`. Leaf insertion: `addLeafValue` (:12).
- **A σ head already exists**: `NNOutput.shorttermWinlossError` /
  `shorttermScoreError` (neuralnet/nninputs.h:129) — per-position predicted
  eval error. Currently consumed as a sample WEIGHT
  (`computeWeightFromNNOutput`, searchupdatehelpers.cpp:98); the port
  consumes it as posterior variance instead (and must not double-count by
  leaving uncertainty-weighting on).
- **A regional-bias mechanism already exists**: `subtreeValueBiasTable`
  (keyed on last-two-moves + local 5×5 pattern; correction =
  running mean of children-minus-own-eval deltas, factor 0.45). Our v6
  inference is the principled per-sibling-set cousin; the bias-table hook
  sites (addLeafValue:27, recomputeNodeStats:255) are where it plugs in.
  Comparing/ablating against subtreeValueBias is a pre-registered study.
- **Hazards, in order**:
  1. **No per-child evals at expansion.** KataGo's net returns policy
     priors + ONE parent value; children get values only when individually
     visited (FPU proxies them until then). The testbed's "expansion
     reveals all k child evals" cost model does not hold. Fresh-child
     priors must come from the anchor + policy prior (mean) and a fitted
     σ_d head (spread) — the shrinkage anchor machinery becomes MORE
     load-bearing, not less. Budget currency = NN evals (unchanged in
     spirit: one eval per new node).
  2. **Graph search (DAG)**: nodes have multiple parents; sibling-set
     exclusivity breaks. Milestone 1 forces `useGraphSearch=false`
     (supported config) and revisits DAG-compatibility later (edge-visit
     weighting is the existing precedent).
  3. **Virtual loss + multithreading**: VOI is a deterministic argmax; N
     threads would all descend the same path. v1: single-threaded
     correctness; then reinterpret virtual loss as posterior thinning
     (a pending eval reduces the child's claimed resolvable variance —
     principled, no coin flips).
  4. **Win-prob space**: utilities are bounded; Gaussians are approximate
     there. The P-game result (two-moment machinery survives non-Gaussian
     values) is the reason to expect this to be tolerable; calibration
     gates will say.

## Milestones (each gated, committed at boundaries)

- **M0 (done)**: fork `caldred/KataGo` branch `bayes-mcts`; Eigen build
  green; `runtests` passes; this plan.
- **M1 — heads offline**: dump (position, parent eval, children-after-
  visit evals, shorttermError) tuples from ordinary KataGo searches on
  self-play openings; fit ρ (sibling residual correlation) and σ_d
  (log-space regression, features: policy prior entropy/mass, parent
  shorttermError) exactly as bmcts heads.py does. Gate: held-out
  calibration of the three heads (the testbed's head-quality report).
- **M2 — posterior state + backup**: side-table (or node fields) carrying
  (mu, v_priv, b) per node + (vx, R) per node; Clark/quadrature backup in a
  `recomputeNodeStats` branch; fresh-child shrinkage from anchor + M1
  heads; `useBayesSearch` SearchParams switch plumbed via setup.cpp.
  Gate: extended `runoutputtests` (no-NN stub) — posterior invariants,
  exhaustive small-board minimax agreement; and a calibration probe:
  claimed root posterior vs deep-search reference values over a position
  suite (z bands, the bmcts gate).
  **Preconditions from the seq-reveal testbed gate (bmcts
  docs/seqreveal-m2-gate.md, FAILED-and-attributed 2026-07-12; details in
  docs/bayes-m2-design-notes.md)**: sequential reveal must use the P-var
  projection (never BLUP); the policy-prior d-mean offset head is
  REQUIRED (fit added to the M1 gate as Amendment A); the dropped
  Cov(alpha, d_a) anchor echo is conservative-only but up to 2x at low
  |E| — un-dropping it (Stein anchor) is a new mechanism change needing
  its own testbed gate before porting; calibration probes compare in
  EVAL currency, not expansion currency (the ±0.15 z band at 30 evals is
  mis-scaled even for the all-at-once stack).
- **M3 — VOI descent**: fork the argmax in `selectBestChildToDescend`;
  single-thread first. Gate: decision calibration (claimed P(best) vs
  empirical vs the deep-search reference) + fixed-visits match vs PUCT
  KataGo at low visits (32/64/128 — where the testbed says the edge
  lives), pre-registered: no significant Elo loss at any tested visit
  count; report where it wins.
  **Precondition (same testbed gate)**: the contested x resolvable
  routing statistic does NOT survive partial reveal as-is (voi-seq
  regret was worse than uniform-seq at eval budgets while ts-seq was
  healthy) — re-derive and gate it in the testbed before forking the
  argmax here. Also carried forward: adaptive descent adds a measured
  ~+0.1..+0.19 seq-specific root optimism (optional stopping within
  sets) — the selection-aware-accounting roadmap item now has a
  concrete M2 form.
- **M4 — regional-bias inference**: the v6 Kalman step at sibling-set
  granularity; ablate against `subtreeValueBias` (off/on/both). Gate:
  calibration unchanged + match.
- **M5 — threading**: virtual-loss-as-posterior-thinning; batched
  speculative descent. Gate: strength scaling with threads.

## Working agreements (inherited from bmcts)

Calibration gates before strength gates; never tune a constant on match
results; falsifications are findings — write them down; results
append-only; the bmcts testbed remains the cheap falsification ground —
any mechanism change discovered here goes back through it first.
