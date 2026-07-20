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


def contrast_posteriors(rec):
    """(ref_idx, per-arm g_mean, g_var, locs) — mirror of the engine."""
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
    W_IN, W_CROSS = 0.65, 0.29  # 2d-g label-pinned kernel constants
    r = 0
    for j in range(1, k):
        if (arms[j]["childVisits"], arms[j]["prior"]) > \
                (arms[r]["childVisits"], arms[r]["prior"]):
            r = j
    ar = arms[r]
    if ar["childVisits"] >= 1 and ar["childAvg"] >= 0:
        Lr = 0.5 + mover * (ar["childAvg"] - 0.5)
        n_ref = max(ar["childVisits"], 1)
    else:
        Lr = 0.5 + mover * (rec["anchMu"] - 0.5)
        n_ref = 1

    def contrast_noise(n_a):
        return vbar * (2.0 * (W_IN - W_CROSS)
                       + (1.0 - W_IN) * (1.0 / n_a + 1.0 / n_ref))

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
            nv = max(contrast_noise(max(a["childVisits"], 1)), 1e-9)
        elif a["evaled"]:
            y = 0.5 + mover * (a["evalWinrate"] - 0.5) - Lr
            nv = max(contrast_noise(1), 1e-9)
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
