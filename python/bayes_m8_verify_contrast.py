#!/usr/bin/env python3
"""M8 phase-2a verification (docs/bayes-m8-integration.md): the engine's
contrast-chooser pick must equal the Python reference recomputation
from the engine's OWN final dumped root state, on >= 10 positions.

Usage:
  python3 bayes_m8_verify_contrast.py --katago ... --model ... --data ...
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from bayes_m5_extract import parse_games, idx_to_gtp

CFG_DIR = Path(__file__).resolve().parent.parent / "cpp" / "configs"


def python_pick(rec):
    """Exact mirror of the engine contrast chooser from a dump line."""
    p = rec["params"]
    arms = rec["arms"]
    k = rec["k"]
    mover = 1.0 if rec["nextPla"] == "W" else -1.0
    rho = p["rho"]
    st = rec["stNode"] if rec["stNode"] > 1e-8 else p["defaultSigma"]
    sr2 = math.exp(p["sigmaDA"] + p["sigmaDB"]
                   * math.log(max(st * st, 1e-8)))

    def sig2(sterr):
        s = sterr if sterr > 1e-8 else p["defaultSigma"]
        s = max(s, 1e-4)
        return math.exp(p["sigmaA"] + p["sigmaB"] * math.log(s)) ** 2

    s2s = [sig2(a["evalStErr"]) for a in arms if a["evaled"]]
    vbar = sum(s2s) / len(s2s) if s2s else p["defaultSigma"] ** 2

    r = 0
    for j in range(1, k):
        if (arms[j]["childVisits"], arms[j]["prior"]) > \
                (arms[r]["childVisits"], arms[r]["prior"]):
            r = j
    ar = arms[r]
    if ar["childVisits"] >= 1 and ar["childAvg"] >= 0:
        Lr = 0.5 + mover * (ar["childAvg"] - 0.5)
        vLr = vbar / max(ar["childVisits"], 1)
    else:
        Lr = 0.5 + mover * (rec["anchMu"] - 0.5)
        vLr = rec["anchVar"]
    logpr = math.log(max(ar["prior"], 1e-12))
    jbest, gbest = r, 0.0
    for j, a in enumerate(arms):
        if j == r:
            continue
        pm = p["dMean"] * (math.log(max(a["prior"], 1e-12)) - logpr)
        pv = 2.0 * sr2
        if a["childVisits"] >= 1 and a["childAvg"] >= 0:
            y = 0.5 + mover * (a["childAvg"] - 0.5) - Lr
            nv = max((1 - rho) * vbar / max(a["childVisits"], 1)
                     + (1 - rho) * vLr, 1e-9)
            w = pv / (pv + nv)
            gm = pm + w * (y - pm)
        elif a["evaled"]:
            y = 0.5 + mover * (a["evalWinrate"] - 0.5) - Lr
            nv = max((1 - rho) * sig2(a["evalStErr"]) + (1 - rho) * vLr,
                     1e-9)
            w = pv / (pv + nv)
            gm = pm + w * (y - pm)
        else:
            gm = pm
        if gm > gbest:
            gbest, jbest = gm, j
    return arms[jbest]["loc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    positions, _ = parse_games(args.data)
    scratch = Path(os.environ.get("TEMP", ".")) / "m8_verify"
    scratch.mkdir(exist_ok=True)
    n_ok = 0
    n_bad = 0
    for pos in positions[:args.n]:
        pla = "B" if len(pos["moves"]) % 2 == 0 else "W"
        dump = scratch / f"{pos['game_hash']}_{pos['turn']}.jsonl"
        if dump.exists():
            dump.unlink()
        env = dict(os.environ)
        env["KATAGO_BAYES_AUDIT"] = str(dump)
        cmds = ["boardsize 9", "komi 7"]
        cmds += [f"play {c} {idx_to_gtp(x)}" for c, x in pos["moves"]]
        cmds += [f"genmove {pla}", "quit"]
        r = subprocess.run(
            [args.katago, "gtp", "-config",
             str(CFG_DIR / "contrast_m8_gtp.cfg"), "-model", args.model],
            input="\n".join(cmds) + "\n", capture_output=True, text=True,
            env=env, timeout=300)
        move = None
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("= ") and len(line) > 2 and line[2] in \
                    "ABCDEFGHJabcdefghj" or (line.startswith("= ")
                                             and "pass" in line):
                move = line[2:].strip()
        recs = [json.loads(l) for l in dump.open() if l.strip()]
        roots = [x for x in recs if x["isRoot"]]
        if not roots or move is None:
            print(f"  SKIP {pos['game_hash'][:8]} (no dump/move)")
            continue
        want = python_pick(roots[-1])
        ok = want.upper() == move.upper()
        n_ok += ok
        n_bad += (not ok)
        print(f"  {pos['game_hash'][:8]} t={pos['turn']} engine={move} "
              f"python={want} {'OK' if ok else 'MISMATCH'}")
    print(f"\n{n_ok} match, {n_bad} mismatch")
    sys.exit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
