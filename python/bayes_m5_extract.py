#!/usr/bin/env python3
"""M5 tail-data extraction (docs/bayes-m5-tailheads.md).

Deterministic position sampling from the Gate-2 lost games
(bayes-data/match64*/), then per position labels the ref / game-move /
longshot children with 1-visit first evals and 500-visit deep values via
the stock analysis engine. Output: JSONL, one line per position,
append-only. No RNG anywhere.

Usage:
  python3 bayes_m5_extract.py --katago ../cpp/build-cuda/Release/katago.exe \
    --model ../models/kata1-....bin.gz --config ../cpp/configs/bayes_m5_analysis.cfg \
    --data ../bayes-data --out ../bayes-data/m5-tail.jsonl [--limit 10]
"""
import argparse
import json
import math
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BOARD = 9
KOMI = 7.0
RULES = "tromp-taylor"
DEEP_VISITS = 500
PRIOR_MIN = 1e-4
LONGSHOT_TARGETS = [0.03, 0.01, 0.003, 0.001, 0.0003]
TURN_MIN = 4
TURN_MAX = 50
COLS = "ABCDEFGHJ"
PASS_IDX = BOARD * BOARD

MOVE_RE = re.compile(r";([BW])\[([a-z]{0,2})\]")


def idx_to_gtp(idx):
    if idx == PASS_IDX:
        return "pass"
    y, x = divmod(idx, BOARD)
    return f"{COLS[x]}{BOARD - y}"


def sgf_to_idx(coord):
    if coord == "" or coord == "tt":
        return PASS_IDX
    x = ord(coord[0]) - ord("a")
    y = ord(coord[1]) - ord("a")
    return y * BOARD + x


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


def parse_games(data_dir):
    """Yield eligible (bayes lost decisively) games with sampled position."""
    positions = []
    skipped_empty = 0
    for sub in ("match64", "match64-rerun"):
        for path in sorted((Path(data_dir) / sub).glob("*.sgfs")):
            for lineno, line in enumerate(path.open()):
                line = line.strip()
                if not line.startswith("(;"):
                    continue
                m = re.search(r"RE\[([BW])\+", line)
                if m is None:
                    continue  # draw or no result
                winner = m.group(1)
                if "PB[bayes]" in line:
                    bayes = "B"
                elif "PW[bayes]" in line:
                    bayes = "W"
                else:
                    continue
                if winner == bayes:
                    continue  # bayes won
                gh = re.search(r"gameHash=([0-9A-F]+)", line)
                sti = re.search(r"startTurnIdx=(\d+)", line)
                if gh is None or sti is None:
                    continue
                game_hash = gh.group(1)
                start_turn = int(sti.group(1))
                # Moves: everything after the header node's closing.
                moves = [(c, sgf_to_idx(xy)) for c, xy in MOVE_RE.findall(line)]
                lo = max(TURN_MIN, start_turn)
                hi = min(TURN_MAX, len(moves) - 1)
                elig = [t for t in range(lo, hi + 1)
                        if ("B" if t % 2 == 0 else "W") == bayes]
                if not elig:
                    skipped_empty += 1
                    continue
                h = int(game_hash[-4:], 16)
                t = elig[h % len(elig)]
                positions.append({
                    "match": sub, "file": path.name, "line": lineno,
                    "game_hash": game_hash, "result": m.group(0)[3:],
                    "bayes": bayes, "turn": t,
                    "moves": moves[:t],
                    "game_move_idx": moves[t][1],
                })
    return positions, skipped_empty


def select_children(policy, game_move_idx):
    """Registered child selection from the parent's raw policy."""
    legal = {i: p for i, p in enumerate(policy) if p >= 0}
    if len(legal) < 2:
        return None
    ref = max(legal, key=lambda i: legal[i])
    chosen = [("ref", ref)]
    if game_move_idx != ref and game_move_idx in legal:
        chosen.append(("game", game_move_idx))
    taken = {i for _, i in chosen}
    for t in LONGSHOT_TARGETS:
        cands = [(abs(math.log(p) - math.log(t)), i)
                 for i, p in legal.items()
                 if p >= PRIOR_MIN and i not in taken]
        if not cands:
            continue
        d, i = min(cands)
        if not (t / 3.0 <= legal[i] <= t * 3.0):
            continue
        chosen.append((f"ls{t:g}", i))
        taken.add(i)
    return chosen


def to_move_list(moves):
    return [[c, idx_to_gtp(i)] for c, i in moves]


def label_position(eng, pos):
    moves = to_move_list(pos["moves"])
    pla = "B" if len(pos["moves"]) % 2 == 0 else "W"
    assert pla == pos["bayes"]
    shallow = eng.position_query(moves, visits=1, include_policy=True)
    root = shallow["rootInfo"]
    policy = shallow["policy"]
    sel = select_children(policy, pos["game_move_idx"])
    if sel is None:
        return None
    deep = eng.position_query(moves, visits=DEEP_VISITS)
    rec = {k: pos[k] for k in ("match", "file", "line", "game_hash",
                               "result", "bayes", "turn")}
    rec["pla"] = pla
    rec["parent_raw"] = root["rawWinrate"]
    rec["parent_st_err"] = root["rawStWrError"]
    rec["parent_deep"] = deep["rootInfo"]["winrate"]
    rec["deep_visits"] = DEEP_VISITS
    rec["n_legal"] = sum(1 for p in policy if p >= 0)
    rec["legal_policy"] = [[i, p] for i, p in enumerate(policy) if p >= 0]
    kids = []
    for tag, idx in sel:
        gtp = idx_to_gtp(idx)
        cmoves = moves + [[pla, gtp]]
        first = eng.position_query(cmoves, visits=1)
        cdeep = eng.position_query(cmoves, visits=DEEP_VISITS)
        kids.append({
            "tag": tag, "move": gtp, "prior": policy[idx],
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
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke-run cap on positions (0 = all)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    positions, skipped = parse_games(args.data)
    print(f"eligible positions: {len(positions)} "
          f"(games skipped for empty turn range: {skipped})", file=sys.stderr)
    if args.limit:
        positions = positions[:args.limit]
        print(f"smoke run: limited to {len(positions)}", file=sys.stderr)

    eng = Engine(args.katago, args.config, args.model)
    out = open(args.out, "a", buffering=1)
    out_lock = threading.Lock()
    done = 0
    n_written = 0

    try:
        def run_one(p):
            nonlocal done, n_written
            try:
                rec = label_position(eng, p)
            except Exception as e:
                print(f"SKIP {p['game_hash']} t={p['turn']}: {e}",
                      file=sys.stderr)
                rec = None
            with out_lock:
                done += 1
                if rec is not None:
                    out.write(json.dumps(rec) + "\n")
                    n_written += 1
                if done % 20 == 0:
                    print(f"  labeled {done}/{len(positions)} "
                          f"(written {n_written})", file=sys.stderr)
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(run_one, positions))
        print(f"done: {n_written} positions -> {args.out}", file=sys.stderr)
    finally:
        out.close()
        eng.close()


if __name__ == "__main__":
    main()
