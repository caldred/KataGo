#!/usr/bin/env python3
"""M8 2c-e deviation telemetry (docs/bayes-m8-integration.md, 2c-e
registration): gated contrast chooser's deviation rate from the policy
argmax at B in {16, 32, 64}, on 60 fresh positions (seed base 500500).

P4: rate < 10%/move at B=16, rising substantially by B=32.
"""
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "python"))
from bayes_m2_probe import (AnalysisEngine, GtpEngine, sample_move,
                            idx_to_gtp)

EXCLUDE_KEYS = {"botName"}  # match-only keys; passing them hangs gtp


def gen_positions(eng, seed_base=500500, games=30, maxpos=60):
    out = []
    for g in range(games):
        rng = random.Random(seed_base * 1000 + g)
        moves = []
        passes = 0
        next_parent = rng.randint(4, 12)
        for mn in range(70):
            r = eng.position_query(moves, visits=1, include_policy=True)
            raw = r["rootInfo"]["rawWinrate"]
            pol = r["policy"]
            if abs(raw - 0.5) > 0.48:
                break
            if mn >= next_parent and mn <= 40 and abs(raw - 0.5) <= 0.45:
                top = max((p, i) for i, p in enumerate(pol) if p > 0)[1]
                out.append({"moves": list(moves),
                            "pla": "B" if len(moves) % 2 == 0 else "W",
                            "policyTop": idx_to_gtp(top)})
                next_parent = mn + 6
                if len(out) >= maxpos:
                    return out
            pla = "B" if len(moves) % 2 == 0 else "W"
            mv = sample_move(pol, rng, 1.0 if mn < 6 else 0.5)
            if mv == 81:
                passes += 1
                if passes >= 2:
                    break
            else:
                passes = 0
            moves.append([pla, idx_to_gtp(mv)])
    return out


def main():
    katago = "cpp/build/katago"
    model = "models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
    eng = AnalysisEngine(katago, "cpp/configs/bayes_m1_analysis.cfg", model)
    try:
        positions = gen_positions(eng)
    finally:
        eng.close()
    print(f"{len(positions)} positions", file=sys.stderr)

    ov_all = {}
    for line in open("cpp/configs/bayes_m8_match.cfg"):
        line = line.split("#")[0].strip()
        if "=" in line:
            k, v = [x.strip() for x in line.split("=", 1)]
            if k.endswith("1") and k[:-1] not in EXCLUDE_KEYS:
                ov_all[k[:-1]] = v
    res = {}
    for b in (16, 32, 64):
        ov = dict(ov_all)
        ov.update({"maxVisits": str(b), "ponderingEnabled": "false",
                   "numSearchThreads": "1",  # global key, not bot-suffixed;
                   # without it the useBayesSearch guard fires on the async
                   # search thread and gtp hangs silently (engine bug, noted
                   # in the 2c-e outcome)
                   "chosenMoveTemperature": "0",
                   "chosenMoveTemperatureEarly": "0"})
        gtp = GtpEngine(katago, "cpp/configs/gtp_example.cfg", model, ov)
        dev = 0
        try:
            gtp.cmd("boardsize 9")
            gtp.cmd("komi 7")
            for p in positions:
                gtp.cmd("clear_board")
                for pla, mv in p["moves"]:
                    gtp.cmd(f"play {pla} {mv}")
                mv = gtp.cmd(f"genmove {p['pla']}")
                if mv.upper() != p["policyTop"].upper():
                    dev += 1
        finally:
            gtp.close()
        res[f"B{b}"] = {"n": len(positions), "deviations": dev,
                        "rate": dev / len(positions)}
        print(f"B={b}: deviation rate {dev}/{len(positions)} = "
              f"{dev/len(positions):.3f}", file=sys.stderr)
    print(json.dumps(res))
    with open("bayes-data/m8-2ce-deviation-telemetry.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
