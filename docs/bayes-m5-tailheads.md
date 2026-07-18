# M5 — tail-data head comparison: protocol (pre-registered)

Written BEFORE any extraction or scoring run, per the working agreements
(doc committed before scripts). Motivation: the M3 Gate-2 failure and the
bmcts M4 recommendation gate (docs/rec-winnerscurse-m4.md there) together
moved attribution AWAY from the choosing layer and onto the beliefs
themselves — specifically per-arm prior-head residuals in the regime the
M1 data never sampled: low-prior moves (M1 truncated at prior floor 0.01)
and sharp positions (M1 parent filter |raw - 0.5| <= 0.45). This
experiment labels exactly that regime, from the positions where the
engine actually lost, and scores FROZEN head candidates there. Nothing
scored here is fitted here (one registered exception: the heteroscedastic
sigma_r head, which gets its own split). Findings will be appended in a
separate section, append-only.

## Setup

- Net: `kata1-b18c384nbt-s9996604416-d4316597426.bin.gz` (same as M1-M3).
- Engine: branch `bayes-mcts` @ the commit this doc lands on, CUDA backend
  (Windows desktop, RTX 3090), v1.16.3. Backend change from M1's Eigen is
  recorded; the 596-golden suite passes on this build, and NN outputs are
  the same net evaluated by a different backend (FP nondeterminism moves
  labels by amounts far below label noise; not a registered concern).
- Board 9x9, komi 7.0, tromp-taylor (identical to the match rules
  koPOSITIONAL/scoreAREA/taxNONE/sui1).
- Extraction path: stock analysis engine (as in M1), config
  `cpp/configs/bayes_m5_analysis.cfg` (GPU thread/batch settings; same
  reporting semantics as M1: white-perspective, rawStWrError in winrate
  units).

## Position sampling (deterministic; no RNG anywhere in the protocol)

- Source: the two failed Gate-2 matches, `bayes-data/match64/` (temp
  0.60/0.20) and `bayes-data/match64-rerun/` (temp 0), 300 games each.
  Eligible games: bayes played and LOST decisively (RE winner is the
  opponent's color; draws excluded). Count at registration: 421
  (178 + 243).
- From each eligible game, ONE position: among turns t (0-indexed, the
  position BEFORE the t-th recorded move) with max(4, startTurnIdx) <= t
  <= min(50, L-1) where L = recorded game length and startTurnIdx comes
  from the SGF comment (policy-initialized opening moves are excluded at
  the low end), keep turns where bayes is to move; from that eligible
  list T pick T[H mod |T|], H = int of the last 4 hex digits of the
  game's gameHash. Games with empty T are skipped (count reported).
- NO parent filter of any kind: no extremity band, no minimum-children
  requirement beyond >= 2 legal moves. This is the point of the run.
- The position's game continuation (the move bayes actually played, and
  the final result) is recorded but is NOT visible to any candidate.

## Per-position protocol

Parent queries: 1-visit query with includePolicy (raw eval, rawStWrError,
full legal-move policy — expansion-time information, exactly what the
engine sees), and a 500-visit query (deep reference value, label only).

Child selection (deterministic, from the parent's raw policy only):

1. `ref` — the highest-prior legal move.
2. `game` — the move bayes actually played at this position, if
   different from `ref`.
3. `longshot[t]` for each target prior t in {0.03, 0.01, 0.003, 0.001,
   0.0003}: the not-yet-selected legal move minimizing |log p - log t|,
   subject to p >= 1e-4; the target is skipped if the nearest such move
   has p outside [t/3, 3t] (no move lives near that decile). Pass counts
   as a legal move throughout (it has a prior like any other).

Up to 7 children per position. Per selected child: play the move, then a
1-visit query (first eval + rawStWrError — what the search sees at first
visit) and a 500-visit query (deep label). 500 matches the M1 label
depth, so residual scales are directly comparable to the M1 fit; the M4
handoff sketch said 300, revised to 500 on GPU affordability BEFORE any
run (recorded here).

Label semantics: identical to M1 — labels are 500-visit search values,
residuals measure resolvable error, label noise inflates all sd
estimates equally across candidates and buckets of equal visit depth.

## Candidates (all coefficients FROZEN before this run)

All scoring is on pairwise contrasts against the `ref` child, in MOVER
perspective, which cancels the set-centering constant and the anchor:

    y_a  = mover_sign * (deep_a - deep_ref)
    r_a  = y_a - yhat_a          (residual; yhat from each candidate)

C1 **shipped-floored** (the engine exactly as it played the matches,
   incl. the 5e-3 prior floor in bayessearch.cpp):
   yhat_a = 0.0638 * (log max(p_a, 5e-3) - log max(p_ref, 5e-3)).
C2 **shipped-unfloored** (the M1 regression without the consumption
   guard): yhat_a = 0.0638 * (log p_a - log p_ref).
C3 **logit link** (M1 Amendment B's loser, re-scored in the regime it
   was rejected outside of): yhat_a = sigmoid(l0 + 0.4235 * (log p_a -
   log p_ref)) - sigmoid(l0), l0 = logit(clip(mover-persp parent RAW
   eval, 0.01, 0.99)). Slope 0.4235 = the Amendment B train fit
   (m1-fit-report-v2.json d_mean_logit_slope_train).
C4 **P(best) inversion** (Cal's proposal, laptop-validated as exactly
   equivalent to C2 in-distribution): treat the full legal-move policy
   (pass included) as arm-best probabilities q ∝ p^(1/tau); solve for
   means m_a such that independent N(m_a, s^2) arms satisfy
   P(arm a is max) = q_a; yhat_a = m_a - m_ref. Primary s = 0.119 (the
   shipped constant residual sigma_r, exp(-4.2554/2)), tau = 1.0;
   secondary variant tau = 0.7 (the laptop's marginal-improvement value)
   reported alongside. Numerics (registered): P computed by 1-D
   quadrature of phi((x-m_a)/s)/s * prod_b Phi((x-m_b)/s) on 2001 points
   spanning [min m - 6s, max m + 6s]; damped log-space fixed point
   m_a <- m_a + 0.5 s (log q_a - log P_a), stop at max |log q - log P|
   < 1e-6 or 500 iters; non-converged positions excluded and counted.

Qualitative divergence (recorded so the readout is falsifiable, not
post-hoc): below the 5e-3 floor C1 goes FLAT (overrates longshots if
value keeps falling in log p -> negative tail bias); C2 stays log-linear;
C4's Gaussian-max link implies a sub-log-linear tail (spacing grows like
s*sqrt(2 log(1/q)), i.e. FLATTER than C2 for tiny p — if C2 is right, C4
overshoots optimistic; if C4 is right, C2 predicts longshots too bad ->
positive tail bias). C3 attenuates near decided positions where the M1
mid-band data said the slope is flat. These four cannot all be unbiased
in the tail; that separation is the experiment.

## Sigma_r part (the component M4 proved the chooser cannot fix)

S1 **Descriptive (no fitting)**: per bucket (below), (a) sd of C2
   residuals r_a vs the shipped homoscedastic pairwise claim
   sqrt(2)*0.119 = 0.168; (b) first-eval check: sd(mover_sign*(first_a -
   deep_a)) vs the M1-corrected claim exp(0.1560)*st_a^0.9826, ratio per
   bucket (the M1 sigma gate re-run in the unsampled regime).
S2 **Fitted heteroscedastic head** (the ONE fitted object; own split,
   registered): fit on positions whose game has even parity (int of last
   hex digit of gameHash mod 2 == 0), validate on odd. Form (log space,
   v7 lesson): log sigma_r^2 = a + b*log max(p_a, 1e-4) + c*|parent_raw
   - 0.5|, by least squares on log r_a^2 with the lognormal mean
   correction applied at prediction. Report held-out per-bucket
   calibration ratio mean(r^2/sigma_hat^2).

## Buckets (registered)

- Longshot prior: p_a in [1e-4, 3e-4), [3e-4, 1e-3), [1e-3, 3e-3),
  [3e-3, 1e-2), [1e-2, 3e-2), [3e-2, 1]. "Tail" = the first three
  (p < 3e-3), all below the engine's 5e-3 floor.
- Parent extremity: |mover parent raw - 0.5| in [0, 0.15), [0.15, 0.30),
  [0.30, 0.45), [0.45, 0.5] (the last is the band M1 never sampled).
- Secondary metric: per-position max |z| (z = r_a / candidate's claimed
  sd) across its children, per candidate; reported against the Gaussian
  order-statistic expectation for that position's child count.

All CIs: cluster bootstrap by game (2000 resamples). Bias = mean r_a per
bucket with CI; scale = sd ratio per bucket with CI.

## Decision rules (registered)

- R1 (mean head): candidate X "loses the tail" if some tail bucket has
  |bias| >= 0.03 with 95% CI excluding 0. Candidate Y "wins the tail
  over X" if Y has no such bucket AND Y's tail-pooled squared-residual
  sum beats X's in >= 80% of paired cluster-bootstrap resamples. A
  winner over the shipped head (C1) becomes the integration candidate
  for the next engine milestone — subject to its own pre-registered gate
  there; nothing ships from this doc.
- R2 (sigma_r): heteroscedasticity is established if any tail or
  extremity bucket has S1(a) sd ratio >= 1.5 with CI excluding 1.0.
  Then a heteroscedastic sigma_r head is REQUIRED for the next
  integration, and S2 is its candidate iff its held-out per-bucket
  calibration lands in [0.7, 1.4] (the M1 band) for every bucket with
  >= 30 held-out children.
- R3 (null): if no candidate separates under R1 and R2 finds nothing,
  the prior-head/tail hypothesis is DOWN-WEIGHTED and attribution moves
  to the remaining M3 suspects (adversarial shift; non-prior belief
  features). Recorded either way.
- Sanity floor: >= 150 labeled positions and >= 300 tail children
  (p < 3e-3) pooled; if unmet, EXTEND (second position per game at the
  next eligible turn index after the first pick, then more matches),
  never re-split, never re-look.
- One look: the analysis script runs once on the completed dataset. A
  <= 10-position smoke run to debug the pipeline writes to a separate
  file (`bayes-data/m5-smoke.jsonl`) and is excluded from analysis.

## Amendment A (2026-07-17, numerics only; registered BEFORE any analysis
## of the real dataset — found during the doc-sanctioned smoke-file
## pipeline debug, extraction complete but unanalyzed)

The registered C4 fixed-point iteration fails on real match policies:
(1) linear-P underflow at extreme spreads (a large early step pushes an
arm's mean low enough that its quadrature P underflows to 0; the log
error saturates at ~690 and the update runs away) — 6/10 smoke
positions at tau=1.0, 10/10 at tau=0.7; (2) where it does not run away,
convergence for tiny-q arms is geometric-slow (uniform step underweights
arms far below the leader, whose sensitivity dlogP/dm scales with
distance/s^2). Replacement, changing the ITERATION ONLY, not the
equation, tolerance (1e-6), grid (2001), or exclusion rule: log-space
quadrature via logsumexp (no underflow) and per-arm Newton
preconditioning denom = max(1/s, (x* - m)/s^2) with damping 0.7 and step
clip +/- 2s; iteration cap 2000. On the smoke records where the
registered scheme converged, the replacement reaches the same fixed
point to 3e-8 (after removing the additive-constant gauge); on all 10
smoke records x both taus it converges in <= 20 iterations. No scoring
rule, coefficient, bucket, or decision rule is touched.

## Cost arithmetic (frozen at registration)

~421 positions x (1 + 500 parent + up to 7 x 501 child visits) ~= 1.7M
visits. GPU benchmark (19x19, this build): 274 visits/s single-stream,
~1400 nnEvals/s batched; with 16 concurrent analysis queries expect
20-40 min wall. Output: `bayes-data/m5-tail.jsonl` (append-only), one
line per position; analysis report `docs/bayes-m5-report.json`.
