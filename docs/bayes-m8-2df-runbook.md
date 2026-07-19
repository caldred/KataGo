# M8 2d-f desktop runbook — clean tree variogram (descendant labels)

Registration: docs/bayes-m8-integration.md, "2d-f registration"
(commit ca118346). Scope: MEASUREMENT ONLY. No model fit, no engine
change, no consumer touches on this rung.

## Prerequisites (all already on the desktop)

- m8-audit dumps: `bayes-data/m8-audit/*_hybrid.jsonl` (from
  bayes_m8_audit_all.py). Only the `_hybrid` dumps are read.
- The m5/m7/m8 position files (`m5-tail.jsonl` etc.) — used to
  reconstruct board positions by (game_hash, turn).
- CUDA binary + b18 model (same pair used for all prior labeling).

Pull both repos first: this branch (bayes-mcts @ ca118346+) and
bmcts main (@ fb94314+).

## Step 1 — label descendants (KataGo/python/)

```
python3 bayes_m8_label_descendants.py \
    --katago ../cpp/build-cuda/Release/katago.exe \
    --model ../models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz \
    --data ../bayes-data
```

- Appends to `bayes-data/m8-desc-labels.jsonl`; RESUMABLE — safe to
  re-run after an interruption, it skips already-labeled
  (game_hash, turn, path) keys. Do not delete the file to "restart".
- Expected volume: <= 15 labels/position (fewer where dumps are
  shallow); ~600-900 total at 500 visits. Under an hour on CUDA.
- Sanity while running: stderr prints one line per position with the
  running label total. If the total is far below ~6/position, check
  chain depth in the dumps before proceeding (shallow B=64 trees
  shrink the yield; the registration anticipates this, but a
  near-empty file means a bug, not shallow trees).

## Step 2 — the variogram (bmcts/)

```
python scripts/m8_variogram_clean.py --data ../KataGo/bayes-data
```

Prints: relationship table (sibling / parent-child / ad-gap2 /
ad-gap3+ / cousin-near / cousin-root), tree-distance table, cluster
bootstrap CIs, even/odd split, and a secondary stErr-standardized
table (non-gating).

## Step 3 — score the registered predictions, append the outcome

Evaluate against the registration (no post-hoc additions):

- P-2df1: ancestor-descendant corr materially above M1's sibling 0.26.
- P-2df2: every clean entry below its 2d-e drift-inflated counterpart
  (sibling < 1.13, d2 < 0.87, d3+ < 0.67).
- P-2df3: cousin-at-root smallest; within-subtree minus cousin gap
  survives de-drifting.
- P-2df4 (soft): cousin corr increases with LCA depth;
  ancestor-descendant corr decreases with gap.

Append a "2d-f outcome" section to docs/bayes-m8-integration.md
(append-only, as always): the two tables, CIs, even/odd agreement,
each prediction CONFIRMED/REFUTED/MISSED, and any caveats owned.
Commit dumps of the numbers, not screenshots.

## What comes after (each its OWN registration, not this run)

(ii) re-derive contrast noise + allocation currency from the measured
covariance; (iii) the ladder again. If P-2df3 FAILS (cousin gap was a
shared-label artifact), stop and re-open the contrast principle's
attribution before any derivation work — that failure mode changes
the sequence, not just the numbers.
