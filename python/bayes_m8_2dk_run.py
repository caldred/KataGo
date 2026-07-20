#!/usr/bin/env python3
"""M8 2d-k Step 1 collection (docs/bayes-m8-integration.md 2d-k
registration): ~40 audited contrastvoi searches at B=64 with
KATAGO_BAYES_AUDIT_ALL, every-8th m5 first-turn position. Output:
bayes-data/m8-2dk/<gh>_<turn>.jsonl full-tree trajectory dumps.

Usage:
  python3 bayes_m8_2dk_run.py --katago ... --model ... --data ...
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from bayes_m5_extract import parse_games, idx_to_gtp

CFG_DIR = Path(__file__).resolve().parent.parent / "cpp" / "configs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    data = Path(args.data)
    outdir = data / "m8-2dk"
    outdir.mkdir(exist_ok=True)
    positions, _ = parse_games(args.data)
    picked = positions[::8][:args.n]
    for i, pos in enumerate(picked):
        pla = "B" if len(pos["moves"]) % 2 == 0 else "W"
        dump = outdir / f"{pos['game_hash']}_{pos['turn']}.jsonl"
        if dump.exists():
            continue  # resumable
        env = dict(os.environ)
        env["KATAGO_BAYES_AUDIT"] = str(dump)
        env["KATAGO_BAYES_AUDIT_ALL"] = "1"
        cmds = ["boardsize 9", "komi 7"]
        cmds += [f"play {c} {idx_to_gtp(x)}" for c, x in pos["moves"]]
        cmds += [f"genmove {pla}", "quit"]
        subprocess.run(
            [args.katago, "gtp", "-config",
             str(CFG_DIR / "contrastvoi_m8_gtp.cfg"),
             "-model", args.model],
            input="\n".join(cmds) + "\n", capture_output=True,
            text=True, env=env, timeout=300)
        print(f"  {i+1}/{len(picked)} {pos['game_hash'][:8]}",
              file=sys.stderr)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
