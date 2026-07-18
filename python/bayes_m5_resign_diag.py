#!/usr/bin/env python3
"""EXPLORATORY diagnostic (not a registered gate): neutral 500-visit
re-evaluation of the final positions of the Gate-2 games bayes resigned.

Question: at the moment of resignation, what was bayes's mover-persp
winrate according to a neutral PUCT search? If often >> 0.025 (the
resign threshold on its own reported values), premature resignation
driven by pessimistic bayesMu is a real loss mechanism.

Output: bayes-data/m5-resign-diag.jsonl (one line per resigned game).
"""
import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bayes_m5_extract import Engine, MOVE_RE, sgf_to_idx, idx_to_gtp


def parse_resigned(data_dir):
    games = []
    for sub in ("match64", "match64-rerun"):
        for path in sorted((Path(data_dir) / sub).glob("*.sgfs")):
            for lineno, line in enumerate(path.open()):
                line = line.strip()
                m = re.search(r"RE\[([BW])\+R\]", line)
                if m is None:
                    continue
                winner = m.group(1)
                if "PB[bayes]" in line:
                    bayes = "B"
                elif "PW[bayes]" in line:
                    bayes = "W"
                else:
                    continue
                if winner == bayes:
                    continue  # opponent resigned
                gh = re.search(r"gameHash=([0-9A-F]+)", line)
                moves = [(c, sgf_to_idx(xy)) for c, xy in MOVE_RE.findall(line)]
                games.append({
                    "match": sub, "file": path.name, "line": lineno,
                    "game_hash": gh.group(1) if gh else None,
                    "bayes": bayes, "n_moves": len(moves),
                    "moves": moves,
                })
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--visits", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    games = parse_resigned(args.data)
    print(f"resigned games: {len(games)}", file=sys.stderr)
    eng = Engine(args.katago, args.config, args.model)
    out = open(args.out, "a", buffering=1)
    lock = threading.Lock()
    done = 0

    try:
        def run_one(g):
            nonlocal done
            try:
                mvs = [[c, idx_to_gtp(i)] for c, i in g["moves"]]
                pla = "B" if len(g["moves"]) % 2 == 0 else "W"
                resp = eng.position_query(mvs, visits=args.visits)
                wr = resp["rootInfo"]["winrate"]  # white persp
                mover = 1.0 if pla == "W" else -1.0
                rec = {k: g[k] for k in ("match", "file", "line",
                                         "game_hash", "bayes", "n_moves")}
                rec["final_pla"] = pla
                rec["pla_is_bayes"] = (pla == g["bayes"])
                rec["winrate_white"] = wr
                bayes_sign = 1.0 if g["bayes"] == "W" else -1.0
                rec["bayes_winrate"] = 0.5 + bayes_sign * (wr - 0.5)
                rec["visits"] = args.visits
            except Exception as e:
                print(f"SKIP {g['game_hash']}: {e}", file=sys.stderr)
                rec = None
            with lock:
                done += 1
                if rec is not None:
                    out.write(json.dumps(rec) + "\n")
                if done % 50 == 0:
                    print(f"  {done}/{len(games)}", file=sys.stderr)
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(run_one, games))
    finally:
        out.close()
        eng.close()


if __name__ == "__main__":
    main()
