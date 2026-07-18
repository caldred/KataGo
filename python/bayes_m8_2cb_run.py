#!/usr/bin/env python3
"""M8 2c-b diagnosis collection (docs/bayes-m8-integration.md): the 2b
fresh-position set, contrast bot, one engine process per position with
KATAGO_BAYES_AUDIT so the final root state is dumped. Records the
engine's move per position. Output: bayes-data/m8-2cb/<gh>_<turn>.jsonl
dumps + bayes-data/m8-2cb-moves.jsonl.

Usage:
  python3 bayes_m8_2cb_run.py --katago ... --model ... --data ...
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from bayes_m5_extract import (MOVE_RE, sgf_to_idx, idx_to_gtp,
                              TURN_MIN, TURN_MAX)

CFG_DIR = Path(__file__).resolve().parent.parent / "cpp" / "configs"


def fresh_positions(data):
    """Identical to bayes_m7_replay --second-turn sampling."""
    fresh = []
    seen = set()
    for sub in ("match64", "match64-rerun"):
        for path in sorted((Path(data) / sub).glob("*.sgfs")):
            for lineno, line in enumerate(path.open()):
                line = line.strip()
                m = re.search(r"RE\[([BW])\+", line)
                if m is None or not line.startswith("(;"):
                    continue
                if "PB[bayes]" in line:
                    bayes = "B"
                elif "PW[bayes]" in line:
                    bayes = "W"
                else:
                    continue
                if m.group(1) == bayes:
                    continue
                gh = re.search(r"gameHash=([0-9A-F]+)", line)
                sti = re.search(r"startTurnIdx=(\d+)", line)
                if gh is None or sti is None:
                    continue
                moves = [(c, sgf_to_idx(xy))
                         for c, xy in MOVE_RE.findall(line)]
                lo = max(TURN_MIN, int(sti.group(1)))
                hi = min(TURN_MAX, len(moves) - 1)
                elig = [t for t in range(lo, hi + 1)
                        if ("B" if t % 2 == 0 else "W") == bayes]
                if not elig:
                    continue
                h = int(gh.group(1)[-4:], 16)
                t1 = elig[h % len(elig)]
                after = [t for t in elig if t > t1]
                if not after:
                    continue
                t2 = after[0]
                prefix = tuple(moves[:t2])
                if prefix in seen:
                    continue
                seen.add(prefix)
                fresh.append({"game_hash": gh.group(1), "turn": t2,
                              "moves": moves[:t2]})
    return fresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    data = Path(args.data)
    outdir = data / "m8-2cb"
    outdir.mkdir(exist_ok=True)
    positions = fresh_positions(data)
    print(f"positions: {len(positions)}", file=sys.stderr)
    out = open(data / "m8-2cb-moves.jsonl", "a", buffering=1)
    try:
        for i, pos in enumerate(positions):
            pla = "B" if len(pos["moves"]) % 2 == 0 else "W"
            dump = outdir / f"{pos['game_hash']}_{pos['turn']}.jsonl"
            if dump.exists():
                dump.unlink()
            env = dict(os.environ)
            env["KATAGO_BAYES_AUDIT"] = str(dump)
            cmds = ["boardsize 9", "komi 7"]
            cmds += [f"play {c} {idx_to_gtp(x)}" for c, x in pos["moves"]]
            cmds += [f"genmove {pla}", "quit"]
            r = subprocess.run(
                [args.katago, "gtp", "-config",
                 str(CFG_DIR / "contrast_m8_gtp.cfg"),
                 "-model", args.model],
                input="\n".join(cmds) + "\n", capture_output=True,
                text=True, env=env, timeout=300)
            move = None
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.startswith("= ") and line[2:].strip():
                    move = line[2:].strip()
            out.write(json.dumps({"game_hash": pos["game_hash"],
                                  "turn": pos["turn"], "pla": pla,
                                  "move": move}) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(positions)}", file=sys.stderr)
    finally:
        out.close()
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
