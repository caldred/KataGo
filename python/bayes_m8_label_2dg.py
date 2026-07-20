#!/usr/bin/env python3
"""M8 2d-g: referee + repeat labels for the accurate kernel measurement
(bmcts docs/measurement-2dg.md; registration in
docs/bayes-m8-integration.md "2d-g registration").

Labels the SAME node set as the registered 2d-f pass (identical
deterministic picker, imported from bayes_m8_label_descendants) with:

  - referee net B, one 500-visit label per node   (net="B", rep=0)
  - referee net C, one 500-visit label per node   (net="C", rep=0)
  - workhorse net A, TWO extra labels on a deterministic ~1-in-5 node
    subset, each from its own engine process with its own
    searchRandSeed override (net="A", rep=1,2) — the repeat-label
    instrument for label idiosyncratic noise.

Appends to bayes-data/m8-desc-labels-2dg.jsonl; resumable by
(game_hash, turn, path, net, rep). NEVER touches the registered 2d-f
file m8-desc-labels.jsonl (whose entries are net="A", rep=0 by
convention).

Referee choice (see runbook docs/bayes-m8-2dg-runbook.md): two nets
from training runs as independent of the b18 workhorse (and of each
other) as practical. Record the exact model files in the runbook.

Usage:
  python3 bayes_m8_label_2dg.py \
      --katago ../cpp/build-cuda/Release/katago.exe \
      --model-a  ../models/<b18 workhorse>.bin.gz \
      --model-b  ../models/<referee 1>.bin.gz \
      --model-c  ../models/<referee 2>.bin.gz \
      --data ../bayes-data [--visits 500] [--repeat-mod 5]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from bayes_m5_extract import Engine, parse_games, idx_to_gtp
from bayes_m8_label_descendants import CFG, load_tree, pick_nodes


class SeededEngine(Engine):
    """Engine with a searchRandSeed override so repeat labels are
    independent draws of the search (and separate processes keep NN
    caches disjoint)."""

    def __init__(self, katago, config, model, seed_tag):
        import subprocess
        import threading
        self.proc = subprocess.Popen(
            [katago, "analysis", "-config", config, "-model", model,
             "-override-config", f"searchRandSeed=2dg-{seed_tag}"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True)
        self.lock = threading.Lock()
        self.cond = threading.Condition()
        self.responses = {}
        self.counter = 0
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()


def repeat_pick(gh, turn, path, mod):
    if mod <= 0:
        return False
    key = f"{gh}|{turn}|{'/'.join(m for _, m in path)}".encode()
    return int(hashlib.sha1(key).hexdigest(), 16) % mod == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--model-c", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--visits", type=int, default=500)
    ap.add_argument("--repeat-mod", type=int, default=5,
                    help="1-in-N nodes get the A-net repeat labels")
    ap.add_argument("--max-positions", type=int, default=0)
    args = ap.parse_args()
    data = Path(args.data)

    positions, _ = parse_games(args.data)
    by_key = {(p["game_hash"], p["turn"]): p for p in positions}
    from bayes_m8_2cb_run import fresh_positions
    for p in fresh_positions(args.data):
        by_key.setdefault((p["game_hash"], p["turn"]), p)

    out_path = data / "m8-desc-labels-2dg.jsonl"
    done = set()
    if out_path.exists():
        for r in map(json.loads, open(out_path)):
            done.add((r["game_hash"], r["turn"],
                      tuple(m for _, m in r["path"]), r["net"], r["rep"]))

    files = sorted((data / "m8-audit").glob("*_hybrid.jsonl"))
    if args.max_positions:
        files = files[:args.max_positions]

    model_of = {"A": args.model_a, "B": args.model_b, "C": args.model_c}
    models = {k: Path(v).name for k, v in model_of.items()}
    engines = {}

    def get_engine(net, rep):
        key = (net, rep)
        if key not in engines:
            engines[key] = SeededEngine(args.katago, str(CFG),
                                        model_of[net], f"{net}{rep}")
        return engines[key]

    out = open(out_path, "a", buffering=1)
    n_new = 0
    try:
        for fi, f in enumerate(files):
            gh = f.stem.split("_")[0]
            turn = int(f.stem.split("_")[-2])
            nodes, root_nid = load_tree(f)
            if root_nid is None or (gh, turn) not in by_key:
                continue
            pos = by_key[(gh, turn)]
            base = [[c, idx_to_gtp(x)] for c, x in pos["moves"]]
            for path, ev, st, nid in pick_nodes(nodes, root_nid):
                pkey = tuple(m for _, m in path)
                jobs = [("B", 0), ("C", 0)]
                if repeat_pick(gh, turn, path, args.repeat_mod):
                    jobs += [("A", 1), ("A", 2)]
                for net, rep in jobs:
                    if (gh, turn, pkey, net, rep) in done:
                        continue
                    resp = get_engine(net, rep).position_query(
                        base + path, visits=args.visits)
                    rec = {"game_hash": gh, "turn": turn, "cfg": "hybrid",
                           "path": path, "depth": len(path), "nid": nid,
                           "eval": ev, "stErr": st, "net": net, "rep": rep,
                           "model": models[net],
                           "deep": resp["rootInfo"]["winrate"],
                           "visits": args.visits}
                    out.write(json.dumps(rec) + "\n")
                    done.add((gh, turn, pkey, net, rep))
                    n_new += 1
            print(f"  {fi+1}/{len(files)} {gh[:8]} t={turn} "
                  f"(total 2dg labels {n_new})", file=sys.stderr)
    finally:
        out.close()
        for e in engines.values():
            e.close()
    print(f"done: {n_new} new labels -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
