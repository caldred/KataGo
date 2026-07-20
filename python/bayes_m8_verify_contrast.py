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


# M8 2d-i resolution-kernel constants (bayesnodestate.h BayesKernel;
# docs/bayes-m8-integration.md 2d-i amendment 1, m8_kernel_fit --pin-tree)
KERNEL_A = 0.923
KERNEL_THETA0 = 0.85
KERNEL_GAMMA = 1.5


def python_pick(rec):
    """Exact mirror of the engine contrast chooser from a dump line."""
    p = rec["params"]
    arms = rec["arms"]
    k = rec["k"]
    mover = 1.0 if rec["nextPla"] == "W" else -1.0
    st = rec["stNode"] if rec["stNode"] > 1e-8 else p["defaultSigma"]
    sr2 = math.exp(p["sigmaDA"] + p["sigmaDB"]
                   * math.log(max(st * st, 1e-8)))

    def sig(sterr):
        s = sterr if sterr > 1e-8 else p["defaultSigma"]
        s = max(s, 1e-4)
        return math.exp(p["sigmaA"] + p["sigmaB"] * math.log(s))

    s_node = rec["nodeSigma0"]

    def phi(s_child):
        return KERNEL_THETA0 * min(1.0, s_child / max(s_node, 1e-12)) \
            ** KERNEL_GAMMA

    def arm_acc(a):
        """-> (n, S, Q, phi) per the engine armAcc lambda."""
        if a["frozen"] and not a["terminal"]:
            return (a["accN"], a["accS"], a["accQ"],
                    phi(a["childSigma0"]))
        s = sig(a["evalStErr"])
        return 1, s, s * s, phi(s)

    r = 0
    for j in range(1, k):
        if (arms[j]["childVisits"], arms[j]["prior"]) > \
                (arms[r]["childVisits"], arms[r]["prior"]):
            r = j
    ar = arms[r]
    ref_is_anchor = not (ar["childVisits"] >= 1 and ar["childAvg"] >= 0)
    if not ref_is_anchor:
        Lr = 0.5 + mover * (ar["childAvg"] - 0.5)
        n_r, s_r, q_r, phi_r = arm_acc(ar)
    else:
        Lr = 0.5 + mover * (rec["anchMu"] - 0.5)

    def contrast_noise(n_a, s_a, q_a, phi_a):
        var_a = q_a / (n_a * n_a)
        if ref_is_anchor:
            return var_a + s_node * s_node \
                - 2.0 * KERNEL_A * phi_a * s_a * s_node / n_a
        var_r = q_r / (n_r * n_r)
        cov = KERNEL_A * phi_a * phi_r * s_a * s_r / (n_a * n_r)
        return var_a + var_r - 2.0 * cov

    logpr = math.log(max(ar["prior"], 1e-12))
    jbest, gbest = r, 0.0
    for j, a in enumerate(arms):
        if j == r:
            continue
        pm = p["dMean"] * (math.log(max(a["prior"], 1e-12)) - logpr)
        pv = 2.0 * sr2
        if a["childVisits"] >= 1 and a["childAvg"] >= 0:
            y = 0.5 + mover * (a["childAvg"] - 0.5) - Lr
            nv = max(contrast_noise(*arm_acc(a)), 1e-9)
            w = pv / (pv + nv)
            gm = pm + w * (y - pm)
        elif a["evaled"]:
            y = 0.5 + mover * (a["evalWinrate"] - 0.5) - Lr
            s = sig(a["evalStErr"])
            nv = max(contrast_noise(1, s, s * s, phi(s)), 1e-9)
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
