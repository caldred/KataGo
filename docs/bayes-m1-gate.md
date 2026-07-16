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

## Amendment B (2026-07-13, registered AFTER the item-4 findings below —
## a model-family comparison on the same data, decided before running it)

Cal's specification question: the d-mean offset should plausibly live in
LOGIT space (winrate is bounded; the same policy preference should move
an even position more than a decided one; sigmoid keeps beliefs in
[0,1]). Registered comparison, same data, no new extraction:

- Logit variant: deep values clipped to [0.01, 0.99], logit, mover
  perspective; within-set centered regression on centered log-priors;
  predictions mapped back through the sigmoid at the set's mean-logit
  operating point and re-centered, so both models are scored in VALUE
  space (the space the engine consumes).
- Decision metric (registered now): held-out value-space SSE, paired
  bootstrap by set — adopt logit iff it beats the winrate-space fit in
  >= 80% of resamples; otherwise keep winrate-space (simpler, already
  passed its band). Tie-breaker diagnostics reported either way: local
  slope by |set mean - 0.5| tercile (logit predicts the value-space
  slope shrinks near extremes), and the residual-sigma_d calibration
  refit under the logit head.
- Honesty note: this follows the item-4 findings (slope/R^2 seen), so
  it is a registered COMPARISON, not a pre-registration; the decision
  rule above was fixed before the comparison ran.

**Amendment B outcome (run same day): KEEP WINRATE-SPACE.** The logit
variant is worse on every registered metric: held-out value-space R^2
0.218 vs 0.260, beats-winrate in only 3.2% of paired resamples (needed
>= 80%), and its residual-sigma_d calibration falls OUT of band
(0.63-0.65 vs [0.7, 1.4]; log-residual sd 1.67 vs 1.03-1.11). The
tercile diagnostic explains why: the empirical value-space slope is
FLAT in |set mean - 0.5| (0.061 / 0.071 / 0.060 across terciles) where
a constant logit slope implies 0.093 / 0.079 / 0.042 — on 9x9, the
policy-preference effect on winrate does not attenuate near decided
positions within the sampled range. The specification concern was
reasonable; the data answers it. One consumption-time guard survives
the concern: the engine should clip shrunken child means into
[0.01, 0.99] (implementation guard, not a head change). Caveat
recorded: parents were filtered to |raw - 0.5| <= 0.45, so very
near-decided sets are underrepresented; if M3+ calibration probes show
drift specifically at extreme positions, revisit with data sampled
there. Full numbers: docs/bayes-m1-fit-report.json (v2 keys
d_mean_logit_*, sigma_d_resid_logit_*).

## FINDINGS (2026-07-13, append-only; full report docs/bayes-m1-fit-report.json)

Run: 278 sibling sets (310 sampled, 32 symHash-dedup/skip), 109 games,
1,519 children, held-out 55 sets (floor 50: OK). Actual wall ~7h
(eval-bound at ~67 NN evals/s; the 3-4.5h estimate assumed more cache
help than materialized).

**sigma — PASS with correction.** rawStWrError UNDERSTATES realized
error: raw held-out ratio 1.266 (train 1.303; phases 1.22/1.31/1.27,
inside the [0.75,1.35] per-phase band; deciles 1.12-1.59, worst at the
smallest reported errors). The registered log-space correction fits
a=-0.9584, b=0.9826 — slope ~= 1, i.e. an almost purely multiplicative
~1.2x miscalibration — and the corrected held-out ratio is **1.047**,
inside [0.9,1.1]. Engine form (log sigma = A + B log stErr, chi^2_1
offset folded in): **A = 0.1560, B = 0.9826**.

**rho — measurement PASSES stability; neither decision branch fires.**
Pooled rho-hat = **0.263**, cluster-bootstrap 95% CI [0.184, 0.334]
(half-width 0.075 <= 0.10). Above 0.10, below 0.30: sibling eval-error
correlation on 9x9 is real but moderate — the two-component machinery
is justified, with expected benefits closer to the testbed's rho=0.3
cells than its rho=0.6/0.9 stress rows. Phase structure: opening 0.08,
mid 0.28, late 0.25 (regional bias appears once fighting starts).
Reminder (registered): labels are 500-visit values, so this is the
RESOLVABLE shared fraction — a lower bound on total shared bias.

**d-mean head (Amendment A) — PASS, strongly.** Pooled mover-persp
slope +0.0638 on centered log prior (phases +0.044/+0.068/+0.075, all
positive); held-out within-set R^2 = 0.260; beats-zero in 100% of 2,000
paired bootstrap resamples (band: >= 95%). The head the seq-reveal
testbed gate made REQUIRED is comfortably fittable from this data.

**sigma_d, raw spread (item 3) — FAIL.** Every candidate incl.
constant-only lands at held-out calibration 0.22-0.41, far below the
[0.7,1.4] band; full-model beats-const bootstrap 0.938 also misses 0.95.
Attribution (checked directly): raw within-set log spread^2 is NOT
lognormal — sd 2.34 with skew -1.67, a long left tail of "all top-k
moves equivalent" sets down to spread^2 ~ e^-14 — so the lognormal mean
correction (exp(s2/2) ~= 12x at s2=4.97) systematically overpredicts.
The raw spread is dominated by prior STRUCTURE, not symmetric d_a
scatter: the same v7 lesson in a new coat — the model family must be
well-specified for the variable, and on real 9x9 data the prior-free
spread isn't the lognormal object the testbed's was.

**sigma_d on d-mean residuals (item 5) — PASS.** Removing the
prior-explained mean drops held-out log-residual sd from ~2.7 to ~1.06
(near-Gaussian) and calibration lands in-band: const-only 0.779,
parent-stErr feature 0.859 (selected on calibration per the registered
rule; its incremental value over const is NOT bootstrap-significant —
full model 0.897 < 0.95 — so const-only is the honest fallback). Engine
form: **log sigma_d^2 = -1.9881 + 0.5022 log stErr_parent^2** (+0.526
lognormal correction), or const **log sigma_d^2 = -4.2554** (+0.633).

**Consequence for M2 (matches the testbed verdict independently):** the
engine must consume d-mean + residual-sigma_d together; the prior-free
sigma_d path is NOT calibrated on 9x9 and must not ship as the default.
A bayesDMean param (slope on centered log prior) joins the param set at
M2 integration. rho: set bayesRho = 0.26 on 9x9.
