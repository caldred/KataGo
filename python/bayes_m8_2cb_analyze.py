#!/usr/bin/env python3
"""M8 2c-b analysis: deviation quality vs contrast-posterior confidence.
For each dumped root state, compute per-arm contrast posterior (mean,
var) with the verified Python mirror; for each z threshold, pick =
contrast top if z clears, else reference arm; score picks with labels
vs PUCT's 2b pick. Prints the ladder and the registered z* pin.

Usage: python3 bayes_m8_2cb_analyze.py --data ../bayes-data
       [--emit-missing pairs.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

Z_LADDER = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


# M8 2d-i resolution-kernel constants (bayesnodestate.h BayesKernel;
# docs/bayes-m8-integration.md 2d-i amendment 1)
KERNEL_A = 0.923
KERNEL_THETA0 = 0.85
KERNEL_GAMMA = 1.5


def contrast_posteriors(rec):
    """(ref_idx, per-arm g_mean, g_var, locs) — mirror of the engine."""
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
    gm = np.zeros(k)
    gv = np.full(k, 1e-12)
    for j, a in enumerate(arms):
        if j == r:
            continue
        pm = p["dMean"] * (math.log(max(a["prior"], 1e-12)) - logpr)
        pv = 2.0 * sr2
        if a["childVisits"] >= 1 and a["childAvg"] >= 0:
            y = 0.5 + mover * (a["childAvg"] - 0.5) - Lr
            nv = max(contrast_noise(*arm_acc(a)), 1e-9)
        elif a["evaled"]:
            y = 0.5 + mover * (a["evalWinrate"] - 0.5) - Lr
            s = sig(a["evalStErr"])
            nv = max(contrast_noise(1, s, s * s, phi(s)), 1e-9)
        else:
            gm[j] = pm
            gv[j] = pv
            continue
        w = pv / (pv + nv)
        gm[j] = pm + w * (y - pm)
        gv[j] = pv * nv / (pv + nv)
    return r, gm, gv, [a["loc"] for a in arms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--emit-missing", default=None)
    args = ap.parse_args()
    data = Path(args.data)

    labels = {}
    for r in map(json.loads, open(data / "m5-tail.jsonl")):
        for k in r["children"]:
            labels[(r["game_hash"], r["turn"], k["move"].upper())] = \
                k["deep"]
    for fn, keys in (("m7-replay.jsonl", ("bot_bayes", "bot_puct")),
                     ("m7-replay-hybrid.jsonl", ("bot_hybrid",)),
                     ("m8-2b-replay.jsonl", ("bot_contrast", "bot_puct"))):
        f = data / fn
        if f.exists():
            for r in map(json.loads, open(f)):
                for kk in keys:
                    if kk in r:
                        labels[(r["game_hash"], r["turn"],
                                r[kk]["move"].upper())] = r[kk]["deep"]
    extra = data / "m8-extra-labels.jsonl"
    if extra.exists():
        for r in map(json.loads, open(extra)):
            labels[(r["game_hash"], r["turn"], r["move"].upper())] = \
                r["deep"]

    puct = {(r["game_hash"], r["turn"]): r["bot_puct"]
            for r in map(json.loads, open(data / "m8-2b-replay.jsonl"))}

    rows = []
    missing = set()
    for f in sorted((data / "m8-2cb").glob("*.jsonl")):
        gh = f.stem.rsplit("_", 1)[0]
        turn = int(f.stem.rsplit("_", 1)[1])
        if (gh, turn) not in puct:
            continue
        recs = [json.loads(l) for l in f.open() if l.strip()]
        roots = [x for x in recs if x["isRoot"]]
        if not roots:
            continue
        rec = roots[-1]
        mover = 1.0 if rec["nextPla"] == "W" else -1.0
        r_idx, gm, gv, locs = contrast_posteriors(rec)
        jc = int(np.argmax(gm))
        z = gm[jc] / math.sqrt(max(gv[jc], 1e-12))
        pmove = puct[(gh, turn)]["move"].upper()
        pdeep = labels.get((gh, turn, pmove))
        for mv in (locs[jc].upper(), locs[r_idx].upper()):
            if (gh, turn, mv) not in labels:
                missing.add((gh, turn, mv))
        rows.append({"gh": gh, "turn": turn, "mover": mover,
                     "ref": locs[r_idx].upper(), "dev": locs[jc].upper(),
                     "gm": float(gm[jc]), "z": float(z),
                     "pmove": pmove, "pdeep": pdeep})

    if missing:
        print(f"{len(missing)} (position, move) pairs lack labels",
              file=sys.stderr)
        if args.emit_missing:
            json.dump([list(x) for x in sorted(missing)],
                      open(args.emit_missing, "w"))
            print(f"wrote {args.emit_missing}; label then rerun",
                  file=sys.stderr)
            return

    print(f"n = {len(rows)} positions")
    print(f"{'z*':>5} {'dev rate':>9} {'dev mean':>9} {'CI':>18} "
          f"{'overall vs puct':>15}")
    for zt in Z_LADDER:
        devs, overall, games = [], [], []
        for row in rows:
            if row["pdeep"] is None:
                continue
            deviate = row["gm"] > 0 and row["z"] >= zt
            mv = row["dev"] if deviate else row["ref"]
            d = labels.get((row["gh"], row["turn"], mv))
            if d is None:
                continue
            val = row["mover"] * (d - row["pdeep"])
            overall.append(val)
            games.append(row["gh"])
            if deviate and mv != row["pmove"]:
                devs.append(val)
        overall = np.array(overall)
        devs = np.array(devs)
        # cluster bootstrap on deviations
        lo = hi = float("nan")
        if len(devs) >= 5:
            rng = np.random.default_rng(20260718)
            bs = [np.mean(rng.choice(devs, len(devs))) for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{zt:>5.2f} {len(devs)/max(len(overall),1):>9.3f} "
              f"{(devs.mean() if len(devs) else float('nan')):>+9.4f} "
              f"[{lo:+.4f},{hi:+.4f}] {overall.mean():>+15.4f}")


if __name__ == "__main__":
    main()
