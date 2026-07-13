#!/usr/bin/env python3
"""M1 head-fitting data extraction (docs/bayes-m1-gate.md).

Drives the stock analysis engine (no C++ changes): self-play position
generation from the raw policy, then per sibling set records the parent's
raw eval / shortterm error / policy / deep value and, for each top-k child,
its move, prior, first raw eval + shortterm error, and a deep reference
search value. Output: JSONL, one line per sibling set, append-only.

All values are white-perspective (reportAnalysisWinratesAs = WHITE in the
config). rawStWrError is already in winrate units (engine reports
shorttermWinlossError * 0.5).

Usage:
  python3 bayes_m1_extract.py --katago ../cpp/build/katago \
    --model ../models/kata1-....bin.gz --config ../cpp/configs/bayes_m1_analysis.cfg \
    --out ../bayes-data/m1-9x9.jsonl --games 135 --deep-visits 800
"""
import argparse
import json
import math
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

BOARD = 9
KOMI = 7.0
RULES = "tromp-taylor"
MAX_GAME_MOVES = 70
RESIGN_BAND = 0.48       # stop game generation when |raw-0.5| exceeds this
PARENT_BAND = 0.45       # parent filter (expansion-time info only)
TOPK = 8
PRIOR_FLOOR = 0.01
MIN_CHILDREN = 4
PARENTS_PER_GAME = 3
PARENT_SPACING = 6
COLS = "ABCDEFGHJ"       # GTP columns, no I


def idx_to_gtp(idx):
    if idx == BOARD * BOARD:
        return "pass"
    y, x = divmod(idx, BOARD)
    return f"{COLS[x]}{BOARD - y}"


class Engine:
    def __init__(self, katago, config, model):
        self.proc = subprocess.Popen(
            [katago, "analysis", "-config", config, "-model", model],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True)
        self.lock = threading.Lock()
        self.cond = threading.Condition()
        self.responses = {}
        self.counter = 0
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue
            with self.cond:
                self.responses[resp.get("id")] = resp
                self.cond.notify_all()

    def query(self, obj):
        with self.lock:
            self.counter += 1
            qid = f"q{self.counter}"
        obj = {**obj, "id": qid}
        with self.lock:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()
        with self.cond:
            while qid not in self.responses:
                self.cond.wait(timeout=600)
                if self.proc.poll() is not None and qid not in self.responses:
                    raise RuntimeError("engine died")
            resp = self.responses.pop(qid)
        if "error" in resp:
            raise RuntimeError(f"engine error: {resp}")
        return resp

    def position_query(self, moves, visits, include_policy=False):
        return self.query({
            "moves": moves, "rules": RULES, "komi": KOMI,
            "boardXSize": BOARD, "boardYSize": BOARD,
            "maxVisits": visits, "includePolicy": include_policy,
        })

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def sample_move(policy, rng, temp):
    """Sample a legal move index from policy^(1/temp)."""
    entries = [(i, p) for i, p in enumerate(policy) if p > 0]
    if not entries:
        return BOARD * BOARD  # pass
    logits = [math.log(p) / temp for _, p in entries]
    mx = max(logits)
    ws = [math.exp(l - mx) for l in logits]
    tot = sum(ws)
    r = rng.random() * tot
    for (i, _), w in zip(entries, ws):
        r -= w
        if r <= 0:
            return i
    return entries[-1][0]


def gen_game(eng, game_id, rng):
    """Self-play one game from the raw policy; return parent snapshots."""
    moves = []
    parents = []
    next_parent = rng.randint(4, 12)
    passes = 0
    for move_num in range(MAX_GAME_MOVES):
        resp = eng.position_query(moves, visits=1, include_policy=True)
        root = resp["rootInfo"]
        policy = resp["policy"]
        raw = root["rawWinrate"]
        if abs(raw - 0.5) > RESIGN_BAND:
            break
        if (move_num >= next_parent and len(parents) < PARENTS_PER_GAME
                and move_num <= 40):
            cands = sorted(
                ((p, i) for i, p in enumerate(policy) if p >= PRIOR_FLOOR),
                reverse=True)[:TOPK]
            if abs(raw - 0.5) <= PARENT_BAND and len(cands) >= MIN_CHILDREN:
                parents.append({
                    "game_id": game_id,
                    "move_num": move_num,
                    "moves": list(moves),
                    "raw": raw,
                    "st_err": root["rawStWrError"],
                    "this_hash": root.get("thisHash"),
                    "sym_hash": root.get("symHash"),
                    "n_legal": sum(1 for p in policy if p >= 0),
                    "children": [{"idx": i, "prior": p} for p, i in cands],
                })
                next_parent = move_num + PARENT_SPACING
            else:
                next_parent = move_num + 2   # retry a bit later
        pla = "B" if len(moves) % 2 == 0 else "W"
        temp = 1.0 if move_num < 6 else 0.5
        mv = sample_move(policy, rng, temp)
        if mv == BOARD * BOARD:
            passes += 1
            if passes >= 2:
                break
        else:
            passes = 0
        moves.append([pla, idx_to_gtp(mv)])
    return parents


def label_parent(eng, parent, deep_visits, seen_hashes, seen_lock):
    """Phase B: deep parent value + per-child first eval and deep label."""
    with seen_lock:
        h = parent.get("sym_hash")
        if h is not None and h in seen_hashes:
            return None
        if h is not None:
            seen_hashes.add(h)
    moves = parent["moves"]
    pla = "B" if len(moves) % 2 == 0 else "W"
    deep = eng.position_query(moves, visits=deep_visits)
    rec = {k: parent[k] for k in
           ("game_id", "move_num", "raw", "st_err", "this_hash", "sym_hash",
            "n_legal")}
    rec["pla"] = pla
    rec["deep"] = deep["rootInfo"]["winrate"]
    rec["deep_visits"] = deep_visits
    kids = []
    for ch in parent["children"]:
        gtp = idx_to_gtp(ch["idx"])
        cmoves = moves + [[pla, gtp]]
        first = eng.position_query(cmoves, visits=1)
        cdeep = eng.position_query(cmoves, visits=deep_visits)
        kids.append({
            "move": gtp, "prior": ch["prior"],
            "raw": first["rootInfo"]["rawWinrate"],
            "st_err": first["rootInfo"]["rawStWrError"],
            "deep": cdeep["rootInfo"]["winrate"],
        })
    rec["children"] = kids
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=135)
    ap.add_argument("--deep-visits", type=int, default=800)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--game-workers", type=int, default=8)
    ap.add_argument("--label-workers", type=int, default=16)
    args = ap.parse_args()

    eng = Engine(args.katago, args.config, args.model)
    out = open(args.out, "a", buffering=1)
    out_lock = threading.Lock()
    seen_hashes = set()
    seen_lock = threading.Lock()
    n_written = 0

    try:
        print(f"phase A: generating {args.games} games", file=sys.stderr)
        parents = []
        with ThreadPoolExecutor(args.game_workers) as pool:
            futs = [pool.submit(gen_game, eng, g,
                                random.Random(args.seed * 100000 + g))
                    for g in range(args.games)]
            for f in futs:
                parents.extend(f.result())
        print(f"phase A done: {len(parents)} parent positions",
              file=sys.stderr)

        print("phase B: labeling", file=sys.stderr)
        done = 0
        def run_one(p):
            nonlocal done, n_written
            try:
                rec = label_parent(eng, p, args.deep_visits,
                                   seen_hashes, seen_lock)
            except Exception as e:
                print(f"SKIP set (game {p['game_id']} mv {p['move_num']}): {e}",
                      file=sys.stderr)
                rec = None
            with out_lock:
                done += 1
                if rec is not None:
                    out.write(json.dumps(rec) + "\n")
                    n_written += 1
                if done % 10 == 0:
                    print(f"  labeled {done}/{len(parents)} "
                          f"(written {n_written})", file=sys.stderr)
        with ThreadPoolExecutor(args.label_workers) as pool:
            list(pool.map(run_one, parents))
        print(f"done: {n_written} sibling sets -> {args.out}",
              file=sys.stderr)
    finally:
        out.close()
        eng.close()


if __name__ == "__main__":
    main()
