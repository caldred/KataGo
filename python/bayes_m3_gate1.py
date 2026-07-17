#!/usr/bin/env python3
"""M3 Gate 1 driver (docs/bayes-m3-gate.md): decision calibration for the
full Bayes stack (search + voi-KG selection).

Phase 1: 60 fresh positions (seed base 500200), reference value + best
move from 1500-visit stock searches (analysis engine, WHITE persp).
Phase 2: full-stack GTP measurements at B in {30, 100}: kata-search,
then kata-bayes-root (mu, sd, move, pBest).
Phase 3 control: fresh 60 positions (seed base 500300), stock PUCT
analysis values at B and 1500 -> mover-perspective drift baseline.

Metrics per B:
  - mean claimed pBest vs empirical P(recommendation == ref best move);
    band: diff in [-0.20, +0.12].
  - Bayes-specific mover drift = mean_mover(mu - ref) [bayes batch]
    - mean_mover(wr_B - ref) [control batch]; band <= +0.030.
"""
import argparse
import json
import random
import statistics as st
import sys

sys.path.insert(0, ".")
from bayes_m2_probe import (AnalysisEngine, GtpEngine, parse_bayes_config,
                            parse_bayes_root, sample_move, idx_to_gtp)

RESIGN_BAND = 0.48
PARENT_BAND = 0.45
BOARD = 9


def gen_positions(eng, seed_base, games, max_positions, want_ref_move):
    out = []
    for g in range(games):
        rng = random.Random(seed_base * 1000 + g)
        moves = []
        passes = 0
        next_parent = rng.randint(4, 12)
        for move_num in range(70):
            resp = eng.position_query(moves, visits=1, include_policy=True)
            raw = resp["rootInfo"]["rawWinrate"]
            policy = resp["policy"]
            if abs(raw - 0.5) > RESIGN_BAND:
                break
            if (move_num >= next_parent and move_num <= 40
                    and abs(raw - 0.5) <= PARENT_BAND):
                out.append({"game_id": g, "move_num": move_num,
                            "moves": list(moves),
                            "pla": "B" if len(moves) % 2 == 0 else "W"})
                next_parent = move_num + 6
                if len(out) >= max_positions:
                    return out
            pla = "B" if len(moves) % 2 == 0 else "W"
            mv = sample_move(policy, rng, 1.0 if move_num < 6 else 0.5)
            if mv == BOARD * BOARD:
                passes += 1
                if passes >= 2:
                    break
            else:
                passes = 0
            moves.append([pla, idx_to_gtp(mv)])
    return out


def add_refs(eng, positions, budgets, ref_visits, want_budget_values):
    from concurrent.futures import ThreadPoolExecutor

    def one(p):
        resp = eng.query({"moves": p["moves"], "rules": "tromp-taylor",
                          "komi": 7.0, "boardXSize": 9, "boardYSize": 9,
                          "maxVisits": ref_visits})
        p["ref"] = resp["rootInfo"]["winrate"]
        infos = resp.get("moveInfos", [])
        p["ref_best"] = infos[0]["move"] if infos else None
        if want_budget_values:
            for b in budgets:
                r2 = eng.query({"moves": p["moves"], "rules": "tromp-taylor",
                                "komi": 7.0, "boardXSize": 9, "boardYSize": 9,
                                "maxVisits": b})
                p[f"wr{b}"] = r2["rootInfo"]["winrate"]

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(one, positions))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--analysis-config", required=True)
    ap.add_argument("--gtp-config", required=True)
    ap.add_argument("--bayes-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--max-positions", type=int, default=60)
    ap.add_argument("--budgets", default="30,100")
    ap.add_argument("--paired-seed", type=int, default=None,
                    help="attribution mode: ONE batch at this seed base, "
                         "control values on the SAME positions (paired)")
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]

    out = open(args.out, "a", buffering=1)
    eng = AnalysisEngine(args.katago, args.analysis_config, args.model)
    try:
        if args.paired_seed is not None:
            print(f"PAIRED attribution mode, seed {args.paired_seed}",
                  file=sys.stderr)
            bpos = gen_positions(eng, args.paired_seed, args.games,
                                 args.max_positions, True)
            add_refs(eng, bpos, budgets, 1500, want_budget_values=True)
            cpos = bpos
            print(f"  {len(bpos)} paired positions", file=sys.stderr)
        else:
            print("phase 1: bayes-batch positions + refs", file=sys.stderr)
            bpos = gen_positions(eng, 500200, args.games, args.max_positions, True)
            add_refs(eng, bpos, budgets, 1500, want_budget_values=False)
            print(f"  {len(bpos)} positions", file=sys.stderr)
            print("phase 3 prep: control batch", file=sys.stderr)
            cpos = gen_positions(eng, 500300, args.games, args.max_positions, False)
            add_refs(eng, cpos, budgets, 1500, want_budget_values=True)
            print(f"  {len(cpos)} control positions", file=sys.stderr)
    finally:
        eng.close()

    overrides = parse_bayes_config(args.bayes_config)
    overrides.update({"useBayesSelection": "true", "ponderingEnabled": "false"})
    results = {b: [] for b in budgets}
    for b in budgets:
        ov = dict(overrides)
        ov["maxVisits"] = str(b)
        gtp = GtpEngine(args.katago, args.gtp_config, args.model, ov)
        try:
            gtp.cmd("boardsize 9")
            gtp.cmd("komi 7")
            for p in bpos:
                gtp.cmd("clear_board")
                for pla, mv in p["moves"]:
                    gtp.cmd(f"play {pla} {mv}")
                gtp.cmd(f"kata-search {p['pla']}")
                ro = parse_bayes_root(gtp.cmd("kata-bayes-root"))
                rec = {**{k: p[k] for k in ("game_id", "move_num", "pla",
                                            "ref", "ref_best")},
                       "B": b, **ro}
                if args.paired_seed is not None:
                    rec["wr_control"] = p[f"wr{b}"]
                    rec["moves"] = p["moves"]
                results[b].append(rec)
                out.write(json.dumps(rec) + "\n")
        finally:
            gtp.close()
        print(f"  measured B={b}", file=sys.stderr)

    summary = {}
    for b in budgets:
        rows = results[b]
        claimed = [r["pBest"] for r in rows if r.get("pBest") is not None]
        hit = [1.0 if r.get("move") == r["ref_best"] else 0.0 for r in rows]
        dm_b = st.mean([(r["mu"] - r["ref"]) * (1 if r["pla"] == "W" else -1)
                        for r in rows])
        dm_c = st.mean([(p[f"wr{b}"] - p["ref"]) * (1 if p["pla"] == "W" else -1)
                        for p in cpos])
        dw_b = st.mean([r["mu"] - r["ref"] for r in rows])
        dw_c = st.mean([p[f"wr{b}"] - p["ref"] for p in cpos])
        paired_stats = None
        if args.paired_seed is not None:
            pd = [((r["mu"] - r["ref"]) - (r["wr_control"] - r["ref"]))
                  * (1 if r["pla"] == "W" else -1) for r in rows]
            paired_stats = {"paired_mover_diff_mean": st.mean(pd),
                            "paired_mover_diff_se":
                                st.stdev(pd) / (len(pd) ** 0.5)}
        summary[f"B{b}"] = {
            **(paired_stats or {}),
            "n": len(rows),
            "mean_claimed_pBest": st.mean(claimed) if claimed else None,
            "empirical_match_rate": st.mean(hit),
            "pBest_minus_empirical": (st.mean(claimed) - st.mean(hit))
                                     if claimed else None,
            "mover_drift_bayes": dm_b,
            "mover_drift_control": dm_c,
            "bayes_specific_mover_drift": dm_b - dm_c,
            "white_drift_bayes": dw_b,
            "white_drift_control": dw_c,
        }
    print(json.dumps(summary, indent=1))
    out.write(json.dumps({"summary": summary}) + "\n")
    out.close()


if __name__ == "__main__":
    main()
