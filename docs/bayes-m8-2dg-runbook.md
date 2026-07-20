# M8 2d-g desktop runbook — accurate kernel table (referee + repeat labels)

Registration: docs/bayes-m8-integration.md "2d-g registration";
design + moment system: bmcts docs/measurement-2dg.md.
Sim gate: PASSED (bmcts results/measurement-sim-2dg-n{80,300}.json;
estimator + fixture validation recorded in measurement-2dg.md §6).
Scope: measurement only. The registered 2d-f run and its file are
untouched; run this AFTER 2d-f labeling is complete (it labels the
same node set, discovered from the same dumps).

## Step 0 — choose the two referee nets (record the choice HERE)

Requirements, in priority order:
1. Trained in a DIFFERENT run from the b18 workhorse (independence of
   regional biases is the entire point; strength is secondary).
2. The two referees also as independent of each other as practical
   (different sizes/runs — e.g. a b6c96 and an older different-run
   b18/b10; NOT two checkpoints of the same run).
3. Strong enough that 500-visit labels are sane (any published kata1
   net qualifies).

Referee B = ______________________  (fill in exact file)
Referee C = ______________________

Independence is then MEASURED, not assumed: the same-position label
gaps (A-B, A-C, B-C msq) and the T-BB/T-CC vs T-ref comparison are
the overidentification readouts. Note: in correlation units T-BB ≈
T-ref by construction (numerator and diagonal inflate together —
fixture-validated); the sharper readout is the DIAGONAL comparison:
diag_BB − diag_ref ≈ a_B²·Vρ_B + σ_νB². If the B-C same-position gap
is materially SMALLER than predicted from those diagonals, the
referees share bias — record it and treat T-ref as a lower bound.

## Step 1 — label (KataGo/python/)

```
python3 bayes_m8_label_2dg.py \
    --katago ../cpp/build-cuda/Release/katago.exe \
    --model-a ../models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz \
    --model-b ../models/<referee-B>.bin.gz \
    --model-c ../models/<referee-C>.bin.gz \
    --data ../bayes-data
```

- Appends to `bayes-data/m8-desc-labels-2dg.jsonl`; resumable by
  (game_hash, turn, path, net, rep). Engines are launched lazily; if
  VRAM is tight run twice: once with `--repeat-mod 0` (referees only,
  2 engines), once more with the default to add the A-net repeats.
- Volume: 2 referee labels per 2d-f node (~1200-1800 queries) plus
  ~2/5 of a node-count of repeat labels. Roughly 2-3x the 2d-f
  labeling time on CUDA.

## Step 2 — the accurate table (bmcts/)

```
python scripts/m8_variogram_v2.py --data ../KataGo/bayes-data
```

Prints T-same (continuity), **T-ref (primary)** with cluster CIs and
even/odd split, T-BB/T-CC, per-class attenuation (1-a)^2 and
retention a-hat, repeat-label noise, same-position gaps, and scores
P-2dg1..4 automatically. Writes results/variogram-2dg.json.

## Step 3 — score + append outcome

Evaluate P-2dg1..4 (bmcts docs/measurement-2dg.md §4) and append a
"2d-g outcome" section to docs/bayes-m8-integration.md: both tables,
CIs, attenuation/retention, referee-independence readouts, verdicts.
The kernel family decision (one-chain vs line-persistent, theory doc
§3 decision rule) is made on T-ref ONLY.

Stopping rule: cluster-CI half-width <= 0.05 on sibling,
parent-child, cousin-root in T-ref. If not met: the extension path is
more audited positions (bayes_m8_audit_all.py on fresh positions ->
2d-f labeling -> this runbook again); double until met or plateau.

## What comes after (own registrations)

With T-ref in hand: (ii) of the 2d-e sequence — fit kernel families
(a)-(d) to T-ref by held-out even/odd split, re-derive contrast noise
and the allocation currency from the fitted kernel; retention a-hat
retroactively corrects M1's rho and every same-net-labeled constant.
The bmcts testbed twin (nested error field) is being built against
the same kernel families in parallel.
