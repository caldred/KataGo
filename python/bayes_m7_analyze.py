#!/usr/bin/env python3
"""M7 Phase A analysis (docs/bayes-m7-sim2real.md): paired leak,
deviation rates, tree shape. Vectorized cluster bootstrap by game.

Usage: python3 bayes_m7_analyze.py --data ../bayes-data/m7-replay.jsonl
"""
import argparse
import json

import numpy as np

N_BOOT = 4000
SEED = 20260718


def boot_mean(vals, games, seed=SEED):
    """Cluster bootstrap of the mean, vectorized: per-game sums."""
    ug = sorted(set(games))
    gi = {g: i for i, g in enumerate(ug)}
    s = np.zeros(len(ug))
    n = np.zeros(len(ug))
    for v, g in zip(vals, games):
        s[gi[g]] += v
        n[gi[g]] += 1
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)),
                        size=N_BOOT).astype(float)
    means = (W @ s) / np.maximum(W @ n, 1)
    return (float(np.mean(vals)),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.data) if l.strip()]
    print(f"n = {len(recs)} paired positions")

    mv = np.array([1.0 if r["pla"] == "W" else -1.0 for r in recs])
    db = np.array([r["bot_bayes"]["deep"] for r in recs])
    dp = np.array([r["bot_puct"]["deep"] for r in recs])
    games = [r["game_hash"] for r in recs]
    diff = mv * (db - dp)  # >0: bayes chose better

    m, lo, hi = boot_mean(diff, games)
    print(f"\nP-A1 paired leak (bayes - puct, mover persp): "
          f"{m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  frac bayes strictly worse: {(diff < -1e-6).mean():.3f}  "
          f"strictly better: {(diff > 1e-6).mean():.3f}  "
          f"same move: {np.mean([r['bot_bayes']['move'] == r['bot_puct']['move'] for r in recs]):.3f}")

    for bot in ("bot_bayes", "bot_puct"):
        br = np.array([r[bot]["breadth"] for r in recs])
        dep = np.array([r[bot]["maxdepth"] for r in recs])
        cv = np.array([r[bot]["chosen_visits"] for r in recs])
        print(f"\n{bot}: breadth mean {br.mean():.1f} (p90 {np.percentile(br,90):.0f})"
              f"  maxdepth mean {dep.mean():.1f} (p10 {np.percentile(dep,10):.0f})"
              f"  chosen visits mean {cv.mean():.1f} median {np.median(cv):.0f}"
              f"  thin (<=2): {(cv <= 2).mean():.3f}")

    # breadth/leak interaction: does the leak concentrate where bayes
    # went shallow-broad?
    brb = np.array([r["bot_bayes"]["breadth"] for r in recs])
    for lo_, hi_ in ((0, 10), (10, 20), (20, 85)):
        msk = (brb >= lo_) & (brb < hi_)
        if msk.sum() > 20:
            sub = diff[msk]
            print(f"  bayes breadth [{lo_},{hi_}): n={msk.sum()} "
                  f"paired leak {sub.mean():+.4f}")

    # disagreement-only conditional
    dis = np.array([r["bot_bayes"]["move"] != r["bot_puct"]["move"]
                    for r in recs])
    if dis.sum() > 10:
        m2, lo2, hi2 = boot_mean(diff[dis], [g for g, d in zip(games, dis)
                                             if d])
        print(f"\ndisagreement positions only (n={dis.sum()}): "
              f"paired leak {m2:+.4f} CI [{lo2:+.4f}, {hi2:+.4f}]")


if __name__ == "__main__":
    main()
