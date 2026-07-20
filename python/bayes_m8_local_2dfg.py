#!/usr/bin/env python3
"""2d-fg-local pipeline (docs/bayes-m8-integration.md "2d-fg-local
registration"): regenerate audit dumps from the local match-2ce-B64
archive, then run the 2d-f + 2d-g labeling protocols and both
variogram analyses, entirely on the laptop (Eigen).

Stages (each resumable; state = files on disk):
  1 positions : sample one mid-game position per game (seeded)
  2 dumps     : hybrid_m7_gtp.cfg + KATAGO_BAYES_AUDIT_ALL, B=64
  3 referees  : g170-b6c96 (net B) + g170e-b10c128 (net C), 500v
  4 labels-2df: b18 own labels, 500v (the slow stage, ~17.5 s/label)
  5 repeats   : b18 x2 extra seeds on a 1-in-5 node subset
  6 analysis  : m8_variogram_clean.py + m8_variogram_v2.py (bmcts)

Usage:
  python3 bayes_m8_local_2dfg.py --stage all [--games 60] [--visits 500]
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from bayes_m5_extract import Engine
from bayes_m8_label_descendants import load_tree, pick_nodes
from bayes_m8_label_2dg import SeededEngine, repeat_pick

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "bayes-data-local"
CFG_DIR = ROOT / "cpp" / "configs"
KATAGO = ROOT / "cpp" / "build" / "katago"
MODEL_A = ROOT / "models" / "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
MODEL_B = ROOT / "cpp" / "tests" / "models" / \
    "g170-b6c96-s175395328-d26788732.bin.gz"
MODEL_C = ROOT / "cpp" / "tests" / "models" / \
    "g170e-b10c128-s1141046784-d204142634.bin.gz"
ANALYSIS_CFG = CFG_DIR / "bayes_m5_analysis.cfg"
SRC = ROOT / "bayes-data" / "match-2ce-B64"

GTP_COLS = "ABCDEFGHJ"


def sgf_to_gtp(mv: str) -> str:
    if mv == "" or mv == "tt":
        return "pass"
    c, r = ord(mv[0]) - 97, ord(mv[1]) - 97
    return f"{GTP_COLS[c]}{9 - r}"


def stage_positions(n_games: int):
    out = DATA / "positions.jsonl"
    if out.exists():
        n_have = sum(1 for _ in out.open())
        if n_have >= n_games:
            print(f"[positions] exists ({n_have})")
            return
        # top-up: sorted-hash order + per-game seeding make the first
        # n_have rows identical; regenerate the superset
        print(f"[positions] extending {n_have} -> {n_games}")
        out.unlink()
    games = []
    for path in sorted(SRC.glob("*.sgfs")):
        for line in path.open():
            line = line.strip()
            if not line.startswith("(;"):
                continue
            gh = re.search(r"gameHash=([0-9A-F]+)", line)
            if gh is None:
                continue
            moves = re.findall(r";([BW])\[([a-z]{0,2})\]", line)
            if len(moves) < 14:
                continue
            games.append((gh.group(1), moves))
    games = {gh: mv for gh, mv in games}
    picked = sorted(games)[:n_games]
    with out.open("w") as f:
        for gh in picked:
            moves = games[gh]
            rng = np.random.default_rng(int(gh[:8], 16))
            t = int(rng.integers(10, min(28, len(moves) - 1)))
            f.write(json.dumps({
                "game_hash": gh, "turn": t,
                "moves": [[c, sgf_to_gtp(m)] for c, m in moves[:t]],
                "next": "B" if t % 2 == 0 else "W"}) + "\n")
    print(f"[positions] wrote {len(picked)}")


def positions():
    return [json.loads(x) for x in (DATA / "positions.jsonl").open()]


def stage_dumps():
    outdir = DATA / "m8-audit"
    outdir.mkdir(exist_ok=True)
    todo = [p for p in positions()
            if not (outdir / f"{p['game_hash']}_{p['turn']}_hybrid.jsonl"
                    ).exists()]
    print(f"[dumps] {len(todo)} to generate")
    for i, p in enumerate(todo):
        dump = outdir / f"{p['game_hash']}_{p['turn']}_hybrid.jsonl"
        env = dict(os.environ)
        env["KATAGO_BAYES_AUDIT"] = str(dump)
        env["KATAGO_BAYES_AUDIT_ALL"] = "1"
        cmds = ["boardsize 9", "komi 7"]
        cmds += [f"play {c} {m}" for c, m in p["moves"]]
        cmds += [f"genmove {p['next']}", "quit"]
        try:
            r = subprocess.run(
                [str(KATAGO), "gtp", "-config",
                 str(CFG_DIR / "hybrid_m7_gtp.cfg"), "-model", str(MODEL_A)],
                input="\n".join(cmds) + "\n", capture_output=True, text=True,
                env=env, timeout=600)
        except subprocess.TimeoutExpired:
            # known branch bug: gtp can hang silently (2c-e protocol note).
            # Skip the position; the stage is resumable.
            print(f"  TIMEOUT, skipping {dump.name}", file=sys.stderr)
            if dump.exists():
                dump.unlink()
            continue
        n = sum(1 for _ in dump.open()) if dump.exists() else 0
        if n == 0:
            print(f"  WARNING empty dump {dump.name} rc={r.returncode}",
                  file=sys.stderr)
        print(f"  [dumps] {i + 1}/{len(todo)} {p['game_hash'][:8]} "
              f"t={p['turn']} ({n} recs)", file=sys.stderr)


def picks_for(p):
    dump = DATA / "m8-audit" / f"{p['game_hash']}_{p['turn']}_hybrid.jsonl"
    if not dump.exists():
        return []
    nodes, root_nid = load_tree(dump)
    if root_nid is None:
        return []
    return pick_nodes(nodes, root_nid)


def run_labels(tag, out_name, jobs_fn, engines):
    out_path = DATA / out_name
    done = set()
    if out_path.exists():
        for r in map(json.loads, open(out_path)):
            done.add((r["game_hash"], r["turn"],
                      tuple(m for _, m in r["path"]),
                      r.get("net", "A"), r.get("rep", 0)))
    out = open(out_path, "a", buffering=1)
    n_new = 0
    try:
        for pi, p in enumerate(positions()):
            base = p["moves"]
            for path, ev, st, nid in picks_for(p):
                pkey = tuple(m for _, m in path)
                for net, rep in jobs_fn(p, path):
                    key = (p["game_hash"], p["turn"], pkey, net, rep)
                    if key in done:
                        continue
                    resp = engines[(net, rep)].position_query(
                        base + path, visits=ARGS.visits)
                    rec = {"game_hash": p["game_hash"], "turn": p["turn"],
                           "cfg": "hybrid", "path": path,
                           "depth": len(path), "nid": nid,
                           "eval": ev, "stErr": st,
                           "deep": resp["rootInfo"]["winrate"],
                           "visits": ARGS.visits}
                    if not (net == "A" and rep == 0):
                        rec["net"], rec["rep"] = net, rep
                        rec["model"] = {"A": MODEL_A, "B": MODEL_B,
                                        "C": MODEL_C}[net].name
                    out.write(json.dumps(rec) + "\n")
                    done.add(key)
                    n_new += 1
            print(f"  [{tag}] {pi + 1}/{len(positions())} "
                  f"(total {n_new})", file=sys.stderr)
    finally:
        out.close()
    print(f"[{tag}] {n_new} new labels")


def stage_referees():
    engines = {("B", 0): SeededEngine(str(KATAGO), str(ANALYSIS_CFG),
                                      str(MODEL_B), "B0"),
               ("C", 0): SeededEngine(str(KATAGO), str(ANALYSIS_CFG),
                                      str(MODEL_C), "C0")}
    try:
        run_labels("referees", "m8-desc-labels-2dg.jsonl",
                   lambda p, path: [("B", 0), ("C", 0)], engines)
    finally:
        for e in engines.values():
            e.close()


def stage_labels_2df():
    engines = {("A", 0): Engine(str(KATAGO), str(ANALYSIS_CFG),
                                str(MODEL_A))}
    try:
        run_labels("2df", "m8-desc-labels.jsonl",
                   lambda p, path: [("A", 0)], engines)
    finally:
        for e in engines.values():
            e.close()


def stage_repeats():
    engines = {("A", 1): SeededEngine(str(KATAGO), str(ANALYSIS_CFG),
                                      str(MODEL_A), "A1"),
               ("A", 2): SeededEngine(str(KATAGO), str(ANALYSIS_CFG),
                                      str(MODEL_A), "A2")}
    try:
        run_labels("repeats", "m8-desc-labels-2dg.jsonl",
                   lambda p, path: ([("A", 1), ("A", 2)]
                                    if repeat_pick(p["game_hash"],
                                                   p["turn"], path, 5)
                                    else []), engines)
    finally:
        for e in engines.values():
            e.close()


def stage_analysis():
    bmcts = Path.home() / "Code" / "bmcts"
    for script in ("m8_variogram_clean.py", "m8_variogram_v2.py"):
        print(f"\n===== {script} =====")
        r = subprocess.run(
            [str(bmcts / ".venv" / "bin" / "python"),
             str(bmcts / "scripts" / script), "--data", str(DATA)],
            capture_output=True, text=True)
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr[-2000:], file=sys.stderr)


STAGES = {"positions": lambda: stage_positions(ARGS.games),
          "dumps": stage_dumps, "referees": stage_referees,
          "labels2df": stage_labels_2df, "repeats": stage_repeats,
          "analysis": stage_analysis}


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=list(STAGES) + ["all"])
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--visits", type=int, default=500)
    ARGS = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    order = list(STAGES) if ARGS.stage == "all" else [ARGS.stage]
    for s in order:
        print(f"\n========== stage {s} ==========")
        STAGES[s]()


if __name__ == "__main__":
    main()
