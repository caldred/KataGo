#!/usr/bin/env python3
"""M2 Gate B attribution control (docs/bayes-m2-gate.md failure protocol).

Question: does an ORDINARY search at the probe's budgets show the same
shallow-vs-deep drift against the 1500-visit reference that the Bayes
posterior showed? Fresh positions (seed base 500100), same generation
protocol/filters as the probe; all values via the analysis engine, white
perspective. Outputs mean(value_B - ref) decompositions comparable to the
probe's mean(mu - ref).
"""
import argparse
import json
import statistics as st
import sys

sys.path.insert(0, ".")
from bayes_m1_extract import Engine, sample_move, BOARD, RESIGN_BAND, PARENT_BAND
import random

SEED_BASE = 500100


def gen_positions(eng, games, max_positions):
    out = []
    for g in range(games):
        rng = random.Random(SEED_BASE * 1000 + g)
        moves = []
        passes = 0
        next_parent = rng.randint(4, 12)
        for move_num in range(70):
            resp = eng.position_query(moves, visits=1, include_policy=True)
            raw = resp["rootInfo"]["rawWinrate"]
            policy = resp["policy"]
            if abs(raw - 0.5) > RESIGN_BAND:
                break
            if move_num >= next_parent and move_num <= 40 and abs(raw - 0.5) <= PARENT_BAND:
                out.append({"game_id": g, "move_num": move_num,
                            "moves": list(moves),
                            "pla": "B" if len(moves) % 2 == 0 else "W"})
                next_parent = move_num + 6
                if len(out) >= max_positions:
                    return out
            pla = "B" if len(moves) % 2 == 0 else "W"
            temp = 1.0 if move_num < 6 else 0.5
            mv = sample_move(policy, rng, temp)
            from bayes_m1_extract import idx_to_gtp
            if mv == BOARD * BOARD:
                passes += 1
                if passes >= 2:
                    break
            else:
                passes = 0
            moves.append([pla, idx_to_gtp(mv)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--max-positions", type=int, default=60)
    args = ap.parse_args()

    eng = Engine(args.katago, args.config, args.model)
    try:
        positions = gen_positions(eng, args.games, args.max_positions)
        print(f"{len(positions)} positions", file=sys.stderr)
        rows = []
        with open(args.out, "a", buffering=1) as f:
            for p in positions:
                r = {"game_id": p["game_id"], "move_num": p["move_num"],
                     "pla": p["pla"]}
                for tag, v in (("wr30", 30), ("wr100", 100), ("ref", 1500)):
                    resp = eng.position_query(p["moves"], visits=v)
                    r[tag] = resp["rootInfo"]["winrate"]
                rows.append(r)
                f.write(json.dumps(r) + "\n")

        summary = {}
        for tag, B in (("wr30", 30), ("wr100", 100)):
            d_white = [r[tag] - r["ref"] for r in rows]
            d_mover = [(r[tag] - r["ref"]) * (1 if r["pla"] == "W" else -1)
                       for r in rows]
            by_pla = {}
            for pla in ("B", "W"):
                sel = [r[tag] - r["ref"] for r in rows if r["pla"] == pla]
                by_pla[pla] = {"n": len(sel), "mean": st.mean(sel),
                               "sd": st.stdev(sel)}
            summary[f"B{B}"] = {
                "mean_drift_white": st.mean(d_white),
                "mean_drift_mover": st.mean(d_mover),
                "by_root_pla": by_pla,
            }
        print(json.dumps(summary, indent=1))
        with open(args.out, "a") as f:
            f.write(json.dumps({"summary": summary}) + "\n")
    finally:
        eng.close()


if __name__ == "__main__":
    main()
