#!/usr/bin/env python3
"""M8: label specific (position, move) pairs missing from the pool
(R6 pick completion). Appends to bayes-data/m8-extra-labels.jsonl.

Usage:
  python3 bayes_m8_label_extra.py --katago ... --model ... --data ... \
      --pairs pairs.json    # [[game_hash, turn, MOVE], ...]
"""
import argparse
import json
import sys
from pathlib import Path

from bayes_m5_extract import Engine, parse_games, idx_to_gtp

CFG = Path(__file__).resolve().parent.parent / "cpp" / "configs" / \
    "bayes_m5_analysis.cfg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pairs", required=True)
    args = ap.parse_args()
    pairs = json.load(open(args.pairs))
    positions, _ = parse_games(args.data)
    by_key = {(p["game_hash"], p["turn"]): p for p in positions}
    from bayes_m8_2cb_run import fresh_positions
    for p in fresh_positions(args.data):
        by_key.setdefault((p["game_hash"], p["turn"]), p)
    eng = Engine(args.katago, str(CFG), args.model)
    out = open(Path(args.data) / "m8-extra-labels.jsonl", "a", buffering=1)
    try:
        for gh, turn, move in pairs:
            pos = by_key[(gh, turn)]
            pla = "B" if len(pos["moves"]) % 2 == 0 else "W"
            moves = [[c, idx_to_gtp(x)] for c, x in pos["moves"]]
            resp = eng.position_query(moves + [[pla, move]], visits=500)
            rec = {"game_hash": gh, "turn": turn, "move": move,
                   "deep": resp["rootInfo"]["winrate"], "visits": 500}
            out.write(json.dumps(rec) + "\n")
            print(f"  {gh[:8]} t={turn} {move} -> {rec['deep']:.4f}",
                  file=sys.stderr)
    finally:
        out.close()
        eng.close()


if __name__ == "__main__":
    main()
