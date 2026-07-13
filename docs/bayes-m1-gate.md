# M1 — offline head fitting: protocol and gate (pre-registered)

Written BEFORE any fitting run, per the working agreements. Data protocol
frozen here; sample-size numbers were filled in from the speed benchmark
before the first extraction run; nothing below was altered after seeing
fit results. Findings will be appended in a separate section, append-only.

## Setup

- Net: `kata1-b18c384nbt-s9996604416-d4316597426.bin.gz` (katagotraining.org,
  model version >= 9, has shortterm-error heads). Eigen CPU backend, v1.16.3,
  branch `bayes-mcts`.
- Board: **9x9 only** (Cal's directive 2026-07-12; 19x19 out of scope for
  now). Komi 7.0, Tromp-Taylor rules. Transfer caveat recorded: all M1
  numbers are 9x9 numbers.
- Extraction path: the stock **analysis engine** (no C++ changes). rootInfo
  exposes `rawWinrate` (raw NN value, winrate scale), `rawStWrError`
  (shorttermWinlossError x 0.5, i.e. sd in winrate scale), `winrate`
  (search value), and `policy`. All values requested white-perspective.
  This replaces the planned in-search `bayesDataDump` hook: same tuples,
  zero risk to search code. What it does NOT capture (recorded): sibling
  sets deeper inside a search tree — our sets are all "root of a queried
  position". In Go, unlike the testbed, depth-in-tree is not a real
  feature (position phase is), so phase comes from move number instead.

## Position sampling

- Self-play with the same net: moves sampled from the raw policy
  (maxVisits=1 queries), temperature 1.0 for moves 1-6, then 0.5;
  game ends at pass-pass, 70 moves, or when abs(rawWinrate - 0.5) > 0.48.
- Parent positions: ~3 per game at move numbers spaced >= 6 apart,
  starting at a per-game random offset in [4, 12] (moves 4-40 overall).
  Game id recorded for cluster-aware splits and bootstraps.
- Parent filter (expansion-time information only): abs(rawWinrate - 0.5)
  <= 0.45 and >= 4 legal candidate children after the prior floor.

## Per-record protocol

Per parent (sibling set): parent raw eval + rawStWrError + full policy +
parent deep value (see below). Per child = top-k of the policy with
k = 8 and prior floor 0.01 (record priors; k < 8 allowed, >= 4 required):
play the move, then

- child first eval: maxVisits=1 query -> rawWinrate, rawStWrError
  (this is exactly what the Bayes search will see at that child's first
  visit — KataGo reveals no per-child values at expansion, hazard #1);
- child deep label: maxVisits=500 query -> search winrate.

Top-k-by-policy truncation is a recorded bias: sigma_d and rho are
measured over the children the search would actually consider, which is
also how the fitted heads will be consumed. Child prior is a candidate
feature, so the truncation is conditioned on observables.

Label semantics (recorded before fitting): the "truth" is an 800-visit
search value, so residuals measure the error an 800-visit search can
resolve. Shared net bias that survives 800 visits cancels out of both
first eval and label, so rho-hat estimates the *resolvable* shared
fraction — which is the operationally correct quantity for a search whose
corrections come from its own deeper evaluations, and an underestimate of
total shared bias. sigma-hat is inflated by label noise (deep value has
its own error); direction noted, no correction applied in the primary
numbers.

## Fits (numpy, replicating bmcts heads.py where applicable)

Held-out split: by game id, 80/20. All spread/variance fits in LOG space
(the v7 lesson — linear sigma_d head was catastrophically overconfident).

1. **sigma check**: is rawStWrError calibrated as the sd of
   (first eval - deep value)? Report ratio RMS(residual)/RMS(reported sd)
   globally, by phase (moves <=10 / 11-25 / >25), and by reported-sd
   decile. If off, fit log-space correction log sigma =
   a + b log sigma_reported and report held-out ratio after correction.
2. **rho**: pooled within-set residual correlation, heads.py estimator
   (cross-moments over k(k-1), normalized by mean square residual),
   sample-size weighted across sets; by phase; cluster bootstrap (by
   game) CI.
3. **sigma_d**: within-set variance (ddof=1) of deep child values;
   LOG-LOG regression with lognormal mean correction at prediction time.
   Candidate features (ALL available at expansion time in a real search —
   deviation from heads.py recorded: KataGo reveals no sibling evals at
   expansion, so the observed-eval-spread feature does not exist here):
   log parent rawStWrError^2, policy entropy of the renormalized top-k,
   top-k policy mass, log #legal moves, move number. Selection among
   these on held-out calibration only.

## Gate bands (pass/fail, pre-registered)

- sigma: held-out ratio (after correction if one is fitted) in [0.9, 1.1];
  raw ratio reported either way. Per-phase ratios in [0.75, 1.35].
- rho: cluster-bootstrap 95% CI half-width <= 0.10. No band on the value
  itself — it is a measurement, not a target. Decision rule recorded NOW:
  rho-hat < 0.10 -> the two-component filter is not load-bearing on 9x9;
  M2 still carries (mu, v_priv, b) state (cheap, validated) but the M2/M3
  gates must not lean on rho benefits, and the plan doc gets amended
  honestly. rho-hat >= 0.30 -> machinery load-bearing, proceed as planned.
- sigma_d: held-out R^2 of the log-log model > 0 (vs constant-only, by
  paired bootstrap); held-out calibration ratio
  mean(realized spread^2 / predicted sigma_d^2) in [0.7, 1.4]; and the
  constant-only fallback's held-out log-residual sd reported, so M2 knows
  the cost of dropping features.
- Sanity floor for all three: n_sets(held-out) >= 50, else extend the run
  before reading results (extension is more games, never re-splitting).

## Amendment A (2026-07-12, pre-registered BEFORE any fitting; extraction
## was ~50/310 sets in, no fit had been run on any real-data record)

The bmcts sequential-reveal gate (bmcts docs/seqreveal-m2-gate.md, run
today) FAILED in its prior-free form and promoted the policy-prior
d-mean offset head from optional to REQUIRED for the port: under
policy-ordered reveal, evaluated children are a biased-high subsample
and prior-free unevaluated-sibling posteriors drifted to z-mean +1.9.
The M1 dataset already records (prior, deep value) per child, so the
head is fitted from the same extraction run. Registered now, before
results:

4. **d-mean head**: within-set regression of centered deep values on
   centered log-priors, in MOVER perspective (white-perspective values
   sign-flipped for Black-to-move sets so "higher prior -> higher
   value" is the expected direction):
     y_a = deep_a_mover - mean_set(deep_mover)
     x_a = log prior_a - mean_set(log prior)
     y_a ~ c * x_a  (per phase; pooled slope sample-size weighted)
   Only within-set contrasts are identified (C is unobserved); the
   anchor-offset consequence (E[max d_a] with per-child means replacing
   c_k*sigma_d) is an M2-integration concern, recorded there.
5. **sigma_d residual variant**: the sigma_d fit of item 3 re-run on
   within-set variance of the d-mean residuals (y_a - c*x_a). Both
   variants reported; which one the engine consumes is decided at M2
   integration on the testbed's terms, not on these numbers.

Bands (pre-registered): d-mean held-out within-set R^2 > 0 by paired
bootstrap (>= 95% of resamples), pooled slope POSITIVE in mover
perspective (a negative or ~0 slope means the policy prior carries no
mean signal on 9x9 and the requirement escalates back to the testbed —
the port cannot satisfy the seq-reveal precondition prior-free).
sigma_d-residual variant: same calibration band as item 3 ([0.7, 1.4]
held-out ratio).

## Sample sizes (filled from benchmark before the run; frozen at run start)

Benchmark (Eigen, b18c384nbt, 9x9): ~22 visits/s single search, ~88
visits/s aggregate with 8 concurrent single-threaded searches (~67 NN
evals/s, batch size 1 — Eigen parallelism is across threads only).

- Deep label visits: **500** (revised from the 800 sketch on benchmark
  arithmetic BEFORE the first run; 800 would cost ~9h wall. A 500-visit
  9x9 search remains a far deeper reference than the 1-visit eval whose
  error it labels; the label-noise direction is already recorded above.)
- Target: 110 games -> ~320 sibling sets, k <= 8 children each
  (~2,900 deep queries ~= 1.4M visits ~= 3-4.5h wall with cache overlap).
  Held-out floor check: 0.2 x 280+ >= 56 > 50. Extension if needed = more
  games (never re-splitting), per the sanity floor above.
