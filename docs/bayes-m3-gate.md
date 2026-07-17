# M3 gate (pre-registered): voi-KG selection — calibration, then strength

Written before the selection fork was implemented. M3 ports the round-3
frozen routing (bmcts `policy="voi", seq_kg`): score_j = contested_j x
D_j, contested = overlap density dE[max]/dv_i at the root (min-root:
negate means) and the p_argmax weights w_j at interior nodes; D_j = the
predicted one-reveal variance drop for unevaluated children (two dummy
Stein calls at |E| and |E|+1), and the child's backed-up D = max_i w_i
D_i for evaluated ones; deterministic argmax, exact ties toward the
better mover mean; viability filter per bmcts _expandable (terminal
children with revealed values are never selected; exhausted subtrees
D ~ 0 fall back to best-mu). No use_share / use_infer (not re-derived
for sequential reveal; port order is fixed in the plan doc).

Engine switch: new param `useBayesSelection` (requires useBayesSearch),
DEFAULT FALSE so every committed M2 behavior stays reproducible; gate
and match configs set it true explicitly. PUCT branch untouched when
off. Single-threaded, tree mode (M2 guards inherited).

## Gate 1 — decision calibration (before any match game)

- 60 fresh positions, generation protocol as M2 Gate B, seed base
  500200 (disjoint from every prior batch). Full stack (selection ON),
  budgets B in {30, 100} visits; reference = stock PUCT 1500 visits
  (analysis engine, white perspective).
- Claimed-vs-empirical P(best): claimed = the root w (p_argmax weight)
  of the recommended move at budget exhaustion; empirical = fraction of
  positions where the recommendation equals the reference best move.
  Band: claimed - empirical in [-0.20, +0.12] per budget (conservative
  claims allowed, v5-gate-2 precedent; +0.12 ~= 2 binomial SE at n=60).
- Location re-check, scored the round-3 way THIS time (the M2 Gate B
  lesson): Bayes-specific mover-perspective drift = mean_mover(mu - ref)
  minus the SAME quantity from a matched-eval stock-PUCT control batch
  (fresh positions, seed base 500300, analysis engine at B visits).
  Band: Bayes-specific drift <= +0.030 winrate at each B (the tracked
  selection residual's measured envelope, +0.14..+0.22 z at sd ~0.10);
  the shared shallow-vs-deep component is reported, not gated (net
  property, established at M2).

## Gate 2 — fixed-visit match vs stock PUCT

- `katago match`: bot0 = stock PUCT defaults (useBayesSearch=false),
  bot1 = full Bayes stack (search+selection, M1 coefficients from
  configs/bayes_m2_9x9.cfg). Identical everything else: same net, 9x9,
  komi 7, tromp-taylor, numSearchThreads=1 both, alternating colors,
  policy-initialized openings for variety (match defaults), maxVisits
  per cell in {32, 64, 128}.
- N = 300 games per visit count, fixed in advance, ONE look per cell.
- Pre-registered predictions, scored honestly:
  1. No statistically significant Elo loss at any tested visit count
     (two-sided p < 0.05 on the win rate per cell). This is the pass
     band — the testbed's standing result transferred.
  2. Where the edge shows, it shows at LOW visits (32/64) — the
     testbed says smarter allocation matters most when evals are
     scarce. Report per-cell Elo with CI either way.
- Failure protocol: a significant loss in any cell = FAIL for that
  cell; attribution round (selection telemetry: which nodes eat budget,
  D-vs-R routing share, depth profiles vs PUCT) BEFORE any code or
  constant changes; mechanism changes go back through the bmcts
  testbed per the working agreements. No constant may be tuned on
  match results, ever (standing rule).

## Order of operations (fixed)

1. Implement + model-free checks (invariants still green with
   selection ON; stub-NN match smoke completes without error).
2. Gate 1. If FAIL: attribute; do not proceed to matches until the
   calibration story is understood and written down.
3. Gate 2 cells in order 64, 32, 128 (64 first: the testbed's
   strongest-expected cell; if something is catastrophically wrong it
   shows fastest where the edge should be).
