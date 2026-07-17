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

## Gate 1 OUTCOME (2026-07-13, append-only)

**B=100: PASS both bands** (pBest-empirical -0.192; Bayes-specific
mover drift -0.001). **B=30: FAIL both as scored — attributed:**

1. **Mover drift +0.039 vs band 0.030: the unpaired estimator's batch
   noise.** The registered protocol subtracted means across two
   independent 60-position batches (~±0.018 SE on the difference — a
   design flaw owned here). Paired attribution run (seed 500400, same
   positions for stack and control, bayes-data/m3-gate1-paired.jsonl):
   **+0.008 +- 0.018 SE** — inside the band, indistinguishable from
   zero. The underlying quantity is in band; the registered estimator
   was too noisy for the band's width.
2. **pBest -0.30 vs band floor -0.20: conservative-only, two stacked
   causes.** (a) Structural conservatism at tiny reveal counts (~15 of
   ~82 root moves revealed at B=30: unrevealed-mass keeps P(best)
   spread wide; claims rise 0.38 -> 0.46 by B=100). (b) Metric
   inflation: "empirical" = agreement with the reference engine's top
   move, and on near-ties — exactly where low claims are honest — both
   engines share the same policy prior and agree anyway (claims <= 0.21
   still "hit" 40%). The testbed's exact-truth version of this metric
   doesn't have (b), which is why its conservatism band was tighter.
   Reliability is intact and monotone: top-tercile claims hit 0.85
   (B=30) / 0.95 (B=100); misses carry half the claimed pBest of hits.
   Replicated on the paired batch (-0.32).

No optimism anywhere; no coefficient moved. Calibration story written
= Gate 2 may proceed per the order of operations below.

## Gate 2, cell B=64, FIRST RUN (2026-07-17): FAIL — attributed to an
## unported piece of the stack, protocol amended below (append-only)

Result as scored: 295 completed games, bayes 3 wins / 175 losses / 117
draws = 20.9% +- 2.4% score, **-232 +- 25 Elo, p ~ 0. FAIL.** SGFs +
log: bayes-data/match64/ (first run).

Attribution (in order run):
1. Suspect "final move = visit-argmax is wrong under KG": KILLED for
   the deterministic case — on 60 stored positions at B=64, the
   visit-based choice matched the 1500-visit reference 73% vs
   argmax-posterior-mu 65% (n.s. difference, 72% mutual agreement).
2. Games are mechanically sound (colors 149/148, ~54 moves, resignation
   correct — bayes evaluates its lost endgames at 0.01 and resigns).
   Bayes simply falls behind in the early/middle game.
3. **Confirmed cause: the visit-TEMPERATURE sampler x KG's flat visit
   distribution.** Measured at the empty board, B=64: bayes root visits
   are near-uniform (top share ~0.14 across 9 arms; PUCT concentrates
   0.4-0.7). The search's own values are right (it scores the good
   quartet 0.47 vs 0.41 for the rest) — but match play sampled from
   VISITS with chosenMoveTemperatureEarly = 0.60, i.e. roughly a
   quarter of early moves were exploration arms the search itself
   rated ~6 winrate points worse. KG spends visits by information, not
   preference; every consumer of "visits ~= preference" breaks. The
   final-move rule of the validated stack (bmcts recommend():
   deterministic argmax posterior mean, mover perspective) was never
   ported — an implementation omission of the same kind as Gate A's
   fix-and-rerun class, not a tuning response to match results.

**Amended protocol for the rerun (registered BEFORE the rerun):**
- The bayes bot's chosen move = argmax mover-perspective posterior mu
  over the root set (the bmcts recommend() rule), deterministic
  (implemented in getChosenMoveLoc behind useBayesSelection).
- Both bots run chosenMoveTemperature = 0 and
  chosenMoveTemperatureEarly = 0 (PUCT at temp 0 = its standard
  strongest play, so this amendment cannot flatter the bayes side).
  Opening diversity comes from the policy-initialized openings alone.
- Everything else identical; N = 300, one look, same decision rule.
  The first-run result stands in the record as the outcome of the
  unamended protocol.

## Order of operations (fixed)

1. Implement + model-free checks (invariants still green with
   selection ON; stub-NN match smoke completes without error).
2. Gate 1. If FAIL: attribute; do not proceed to matches until the
   calibration story is understood and written down.
3. Gate 2 cells in order 64, 32, 128 (64 first: the testbed's
   strongest-expected cell; if something is catastrophically wrong it
   shows fastest where the edge should be).
