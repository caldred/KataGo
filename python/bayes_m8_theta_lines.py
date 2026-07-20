#!/usr/bin/env python3
"""M8 2d-h collection (docs/bayes-m8-integration.md 2d-h registration):
roll out 60 lines by repeated 500-visit searches (top move each step,
up to 30 plies, stop at pass/decided); per step record the raw 1-visit
eval + the search's 500-visit value (the label). Sidelines at depths 5
and 15 (the search's #2 move, one eval+label node each).

Output: bayes-data/m8-theta-lines.jsonl (one record per line, resumable
by game_hash+turn key).

Usage:
  python3 bayes_m8_theta_lines.py --katago ../cpp/build-cuda/Release/katago.exe \
    --model ../models/kata1-....bin.gz --data ../bayes-data
"""
import argparse
import json
import sys
from pathlib import Path

from bayes_m5_extract import Engine, parse_games, idx_to_gtp

CFG = Path(__file__).resolve().parent.parent / "cpp" / "configs" / \
    "bayes_m5_analysis.cfg"
MAX_PLIES = 30
SIDELINE_DEPTHS = (5, 15)
DECIDED_BAND = 0.48


def top_moves(resp, n=2):
    infos = sorted(resp.get("moveInfos", []), key=lambda m: m["order"])
    return [m["move"] for m in infos[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    data = Path(args.data)
    out_path = data / "m8-theta-lines.jsonl"
    done = set()
    if out_path.exists():
        for r in map(json.loads, out_path.open()):
            done.add((r["game_hash"], r["turn"]))

    positions, _ = parse_games(args.data)
    picked = positions[::7][:60]
    eng = Engine(args.katago, str(CFG), args.model)
    out = open(out_path, "a", buffering=1)
    try:
        for i, pos in enumerate(picked):
            key = (pos["game_hash"], pos["turn"])
            if key in done:
                continue
            moves = [[c, idx_to_gtp(x)] for c, x in pos["moves"]]
            pla = "B" if len(pos["moves"]) % 2 == 0 else "W"
            steps = []
            sidelines = []
            cur = list(moves)
            cur_pla = pla
            stop = "max"
            for d in range(MAX_PLIES + 1):
                raw = eng.position_query(cur, visits=1)
                deep = eng.position_query(cur, visits=500)
                steps.append({
                    "d": d,
                    "raw": raw["rootInfo"]["rawWinrate"],
                    "stErr": raw["rootInfo"]["rawStWrError"],
                    "label": deep["rootInfo"]["winrate"],
                })
                if abs(deep["rootInfo"]["winrate"] - 0.5) > DECIDED_BAND:
                    stop = "decided"
                    break
                tops = top_moves(deep, 2)
                if not tops or tops[0] == "pass":
                    stop = "pass"
                    break
                if d in SIDELINE_DEPTHS and len(tops) > 1 \
                        and tops[1] != "pass":
                    smoves = cur + [[cur_pla, tops[1]]]
                    sraw = eng.position_query(smoves, visits=1)
                    sdeep = eng.position_query(smoves, visits=500)
                    sidelines.append({
                        "fork_d": d,
                        "raw": sraw["rootInfo"]["rawWinrate"],
                        "stErr": sraw["rootInfo"]["rawStWrError"],
                        "label": sdeep["rootInfo"]["winrate"],
                    })
                cur = cur + [[cur_pla, tops[0]]]
                cur_pla = "W" if cur_pla == "B" else "B"
            out.write(json.dumps({
                "game_hash": pos["game_hash"], "turn": pos["turn"],
                "stop": stop, "steps": steps, "sidelines": sidelines,
            }) + "\n")
            print(f"  {i+1}/{len(picked)} {pos['game_hash'][:8]} "
                  f"len={len(steps)} stop={stop}", file=sys.stderr)
    finally:
        out.close()
        eng.close()
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
