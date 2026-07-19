#!/usr/bin/env python3
"""M8 2d-f: descendant labels for the clean tree variogram
(docs/bayes-m8-integration.md 2d-f registration).

From each m8-audit *_hybrid.jsonl dump: the two root arms with the
largest subtrees plus the next-largest evaled arm; within each deep
arm the principal (max-childVisits) chain to relative depth 4 plus
the largest evaled sibling at relative depths 1-3. Each picked node
gets a 500-visit deep label at its OWN position (path from root).
Appends to bayes-data/m8-desc-labels.jsonl; resumable.

Usage:
  python3 bayes_m8_label_descendants.py --katago ... --model ... \
      --data ../bayes-data [--visits 500] [--max-positions N]
"""
import argparse
import json
import sys
from pathlib import Path

from bayes_m5_extract import Engine, parse_games, idx_to_gtp

CFG = Path(__file__).resolve().parent.parent / "cpp" / "configs" / \
    "bayes_m5_analysis.cfg"

CHAIN_DEPTH = 4      # principal chain to this relative depth
SIB_DEPTHS = (1, 2, 3)  # add one sibling at these relative depths
N_DEEP_ARMS = 2
N_EXTRA_ARMS = 1


def load_tree(f):
    """Final state per nid; returns (nodes, root_nid)."""
    nodes = {}
    root_nid = None
    for line in f.open():
        rec = json.loads(line)
        nodes[rec["nid"]] = rec
        if rec["isRoot"]:
            root_nid = rec["nid"]
    return nodes, root_nid


def evaled_arms(rec):
    return [a for a in rec["arms"] if a["evaled"] and not a["terminal"]]


def pick_nodes(nodes, root_nid):
    """Returns [(path, eval, stErr, nid)]; path = [[color, loc], ...]."""
    root = nodes[root_nid]
    arms = evaled_arms(root)
    deep = sorted((a for a in arms if a["childNid"] in nodes),
                  key=lambda a: -a["childVisits"])
    chosen = deep[:N_DEEP_ARMS]
    chosen_locs = {a["loc"] for a in chosen}
    rest = sorted((a for a in arms if a["loc"] not in chosen_locs),
                  key=lambda a: -a["childVisits"])
    picks = {}

    def add(path, a):
        key = tuple(m for _, m in path)
        if key not in picks:
            picks[key] = (path, a["evalWinrate"], a["evalStErr"],
                          a["childNid"])

    for a in rest[:N_EXTRA_ARMS]:
        add([[root["nextPla"], a["loc"]]], a)

    for arm in chosen:
        parent = root
        a = arm
        path = []
        depth = 1
        while True:
            path_here = path + [[parent["nextPla"], a["loc"]]]
            add(path_here, a)
            if depth in SIB_DEPTHS:
                sibs = sorted(
                    (s for s in evaled_arms(parent)
                     if s["loc"] != a["loc"]),
                    key=lambda s: -s["childVisits"])
                if sibs:
                    add(path + [[parent["nextPla"], sibs[0]["loc"]]],
                        sibs[0])
            if depth >= CHAIN_DEPTH:
                break
            child = nodes.get(a["childNid"])
            if child is None:
                break
            nxt = sorted(evaled_arms(child),
                         key=lambda s: -s["childVisits"])
            # prefer a continuation that itself has a dumped child
            deeper = [s for s in nxt if s["childNid"] in nodes]
            if depth + 1 < CHAIN_DEPTH and deeper:
                nxt = deeper
            if not nxt:
                break
            parent, a = child, nxt[0]
            path = path_here
            depth += 1
    return list(picks.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--visits", type=int, default=500)
    ap.add_argument("--max-positions", type=int, default=0)
    args = ap.parse_args()
    data = Path(args.data)

    positions, _ = parse_games(args.data)
    by_key = {(p["game_hash"], p["turn"]): p for p in positions}
    from bayes_m8_2cb_run import fresh_positions
    for p in fresh_positions(args.data):
        by_key.setdefault((p["game_hash"], p["turn"]), p)

    out_path = data / "m8-desc-labels.jsonl"
    done = set()
    if out_path.exists():
        for r in map(json.loads, open(out_path)):
            done.add((r["game_hash"], r["turn"],
                      tuple(m for _, m in r["path"])))

    files = sorted((data / "m8-audit").glob("*_hybrid.jsonl"))
    if args.max_positions:
        files = files[:args.max_positions]

    eng = Engine(args.katago, str(CFG), args.model)
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
                key = (gh, turn, tuple(m for _, m in path))
                if key in done:
                    continue
                resp = eng.position_query(base + path, visits=args.visits)
                rec = {"game_hash": gh, "turn": turn, "cfg": "hybrid",
                       "path": path, "depth": len(path), "nid": nid,
                       "eval": ev, "stErr": st,
                       "deep": resp["rootInfo"]["winrate"],
                       "visits": args.visits}
                out.write(json.dumps(rec) + "\n")
                done.add(key)
                n_new += 1
            print(f"  {fi+1}/{len(files)} {gh[:8]} t={turn} "
                  f"(total labels {n_new})", file=sys.stderr)
    finally:
        out.close()
        eng.close()
    print(f"done: {n_new} new labels -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
