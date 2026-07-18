#!/usr/bin/env python3
"""M7 Amendment A readout: join the hybrid arm with the two-arm replay
and score P-A4 (hybrid leak <= 1/3 of the bayes arm's).

Usage: python3 bayes_m7_hybrid_join.py --data ../bayes-data
"""
import argparse
import json
from pathlib import Path

import numpy as np

N_BOOT = 4000
SEED = 20260718


def boot_mean(vals, games, seed=SEED):
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
    return (float(np.mean(vals)), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    data = Path(args.data)

    two = {(r["game_hash"], r["turn"]): r
           for r in map(json.loads, open(data / "m7-replay.jsonl"))}
    hyb = {(r["game_hash"], r["turn"]): r
           for r in map(json.loads, open(data / "m7-replay-hybrid.jsonl"))}
    keys = sorted(set(two) & set(hyb))
    print(f"joined positions: {len(keys)}")

    rows = []
    for k in keys:
        r2, rh = two[k], hyb[k]
        mv = 1.0 if r2["pla"] == "W" else -1.0
        rows.append({
            "game": r2["game_hash"],
            "b_vs_p": mv * (r2["bot_bayes"]["deep"] - r2["bot_puct"]["deep"]),
            "h_vs_p": mv * (rh["bot_hybrid"]["deep"] - r2["bot_puct"]["deep"]),
            "h": rh["bot_hybrid"],
        })
    games = [r["game"] for r in rows]
    for name, key in (("bayes  vs puct", "b_vs_p"),
                      ("hybrid vs puct", "h_vs_p")):
        vals = np.array([r[key] for r in rows])
        m, lo, hi = boot_mean(vals, games)
        print(f"{name}: {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"worse {np.mean(vals < -1e-6):.3f} better "
              f"{np.mean(vals > 1e-6):.3f}")
    br = np.array([r["h"]["breadth"] for r in rows])
    dep = np.array([r["h"]["maxdepth"] for r in rows])
    cv = np.array([r["h"]["chosen_visits"] for r in rows])
    print(f"hybrid tree: breadth {br.mean():.1f}  maxdepth {dep.mean():.1f}"
          f"  chosen visits median {np.median(cv):.0f} thin "
          f"{(cv <= 2).mean():.3f}")
    b = np.array([r["b_vs_p"] for r in rows]).mean()
    h = np.array([r["h_vs_p"] for r in rows]).mean()
    print(f"\nP-A4 (hybrid leak <= 1/3 of bayes leak): "
          f"{h:+.4f} vs 1/3 * {b:+.4f} = {b/3:+.4f}  -> "
          f"{'PASS' if h >= b / 3 else 'FAIL'}")


if __name__ == "__main__":
    main()
