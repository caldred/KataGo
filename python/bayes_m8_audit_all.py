#!/usr/bin/env python3
"""M8 phase 1 dump collection (docs/bayes-m8-integration.md): the M7
audit set, hybrid + bayes configs, with KATAGO_BAYES_AUDIT_ALL so every
node recompute is dumped (tree-reconstructible via nid/childNid).

Usage:
  python3 bayes_m8_audit_all.py --katago ../cpp/build-cuda/Release/katago.exe \
    --model ../models/kata1-....bin.gz --data ../bayes-data
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from bayes_m5_extract import parse_games, idx_to_gtp
from bayes_m7_audit_run import select_audit_set, CFG_DIR, CFGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    data = Path(args.data)
    outdir = data / "m8-audit"
    outdir.mkdir(exist_ok=True)

    picked = select_audit_set(data)
    print(f"audit set: {len(picked)} positions", file=sys.stderr)
    positions, _ = parse_games(data)
    by_key = {(p["game_hash"], p["turn"]): p for p in positions}

    for i, key in enumerate(picked):
        pos = by_key[key]
        pla = "B" if len(pos["moves"]) % 2 == 0 else "W"
        gtp_moves = [(c, idx_to_gtp(x)) for c, x in pos["moves"]]
        for cfg_name, cfg_file in CFGS.items():
            dump = outdir / f"{key[0]}_{key[1]}_{cfg_name}.jsonl"
            if dump.exists():
                dump.unlink()
            env = dict(os.environ)
            env["KATAGO_BAYES_AUDIT"] = str(dump)
            env["KATAGO_BAYES_AUDIT_ALL"] = "1"
            cmds = ["boardsize 9", "komi 7"]
            cmds += [f"play {c} {m}" for c, m in gtp_moves]
            cmds += [f"genmove {pla}", "quit"]
            r = subprocess.run(
                [args.katago, "gtp", "-config", str(CFG_DIR / cfg_file),
                 "-model", args.model],
                input="\n".join(cmds) + "\n", capture_output=True,
                text=True, env=env, timeout=300)
            n_lines = sum(1 for _ in dump.open()) if dump.exists() else 0
            if n_lines == 0:
                print(f"WARNING: empty dump {dump.name} rc={r.returncode}",
                      file=sys.stderr)
        print(f"  {i+1}/{len(picked)}", file=sys.stderr)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
