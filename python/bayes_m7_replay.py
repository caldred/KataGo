#!/usr/bin/env python3
"""M7 Phase A: paired engine replay (docs/bayes-m7-sim2real.md).

For each M5 position, run the match's bayes bot and stock PUCT (both
GTP, 64 visits, temp 0, single-threaded, same net) and record chosen
move + tree telemetry via kata-genmove_analyze. Then label each chosen
move with a neutral 500-visit analysis query (reusing M5 labels where
the move coincides). Output: bayes-data/m7-replay.jsonl, append-only.

Usage:
  python3 bayes_m7_replay.py --katago ../cpp/build-cuda/Release/katago.exe \
    --model ../models/kata1-....bin.gz --data ../bayes-data \
    --out ../bayes-data/m7-replay.jsonl [--limit 10]
"""
import argparse
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

from bayes_m5_extract import (Engine, MOVE_RE, sgf_to_idx, idx_to_gtp,
                              parse_games)

CFG_DIR = Path(__file__).resolve().parent.parent / "cpp" / "configs"


class GtpBot:
    """Minimal GTP driver around a katago gtp process."""

    def __init__(self, katago, config, model, name):
        self.name = name
        self.proc = subprocess.Popen(
            [katago, "gtp", "-config", str(config), "-model", model],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("boardsize 9")
        self.send("komi 7")

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(f"{self.name} gtp died on: {cmd}")
            line = line.rstrip("\n")
            if line == "" and lines:
                break
            if line:
                lines.append(line)
        return lines

    def genmove_analyze(self, color):
        """kata-genmove_analyze: returns (move, info_lines)."""
        self.proc.stdin.write(f"kata-genmove_analyze {color}\n")
        self.proc.stdin.flush()
        infos, move = [], None
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(f"{self.name} died in genmove_analyze")
            line = line.rstrip("\n")
            if line.startswith("info "):
                infos.append(line)
            elif line.startswith("play "):
                move = line.split()[1]
            elif line == "" and move is not None:
                break
        return move, infos

    def reset(self):
        self.send("clear_board")
        self.send("komi 7")

    def play_moves(self, moves):
        for c, idx in moves:
            col = "B" if c == "B" else "W"
            self.send(f"play {col} {idx_to_gtp(idx)}")

    def close(self):
        try:
            self.send("quit")
        except Exception:
            pass
        self.proc.kill()


def parse_infos(info_lines):
    """Extract per-move (visits, winrate, pv depth) from
    kata-genmove_analyze output. All moves arrive on one line as
    repeated 'info move ...' segments; later reports of a move
    overwrite earlier ones."""
    per_move = {}
    for line in info_lines:
        for seg in line.split("info move ")[1:]:
            toks = seg.split()
            if not toks:
                continue
            mv = toks[0]
            vis = re.search(r"\bvisits (\d+)", seg)
            wr = re.search(r"\bwinrate ([\d.eE+-]+)", seg)
            pv = re.search(r"\bpv ((?:\S+ ?)+?)(?:$| info | pvVisits)", seg)
            depth = len(pv.group(1).split()) if pv else 0
            per_move[mv] = (int(vis.group(1)) if vis else 0,
                            float(wr.group(1)) if wr else None, depth)
    breadth = sum(1 for v, _, _ in per_move.values() if v >= 1)
    maxdepth = max((d for _, _, d in per_move.values()), default=0)
    return breadth, maxdepth, per_move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    positions, _ = parse_games(args.data)
    if args.limit:
        positions = positions[:args.limit]
    print(f"positions: {len(positions)}", file=sys.stderr)

    # Existing M5 labels: (game_hash, turn, move) -> deep value
    known = {}
    m5 = Path(args.data) / "m5-tail.jsonl"
    for line in m5.open():
        r = json.loads(line)
        for k in r["children"]:
            known[(r["game_hash"], r["turn"], k["move"])] = k["deep"]

    bots = {
        "bayes": GtpBot(args.katago, CFG_DIR / "bayes_m7_gtp.cfg",
                        args.model, "bayes"),
        "puct": GtpBot(args.katago, CFG_DIR / "puct_m7_gtp.cfg",
                       args.model, "puct"),
    }
    labeler = Engine(args.katago,
                     str(CFG_DIR / "bayes_m5_analysis.cfg"), args.model)
    out = open(args.out, "a", buffering=1)

    try:
        for i, pos in enumerate(positions):
            pla = "B" if len(pos["moves"]) % 2 == 0 else "W"
            rec = {k: pos[k] for k in ("match", "file", "line", "game_hash",
                                       "turn")}
            rec["bayes_color"] = pos["bayes"]
            rec["pla"] = pla
            gmoves = [[c, idx_to_gtp(x)] for c, x in pos["moves"]]
            ok = True
            for bname, bot in bots.items():
                try:
                    bot.reset()
                    bot.play_moves(pos["moves"])
                    move, infos = bot.genmove_analyze(pla)
                    breadth, maxdepth, per_move = parse_infos(infos)
                    key = (pos["game_hash"], pos["turn"],
                           move.upper() if move != "pass" else "pass")
                    if key in known:
                        deep = known[key]
                    else:
                        resp = labeler.position_query(
                            gmoves + [[pla, move]], visits=500)
                        deep = resp["rootInfo"]["winrate"]
                    ch = per_move.get(move, (0, None, 0))
                    rec["bot_" + bname] = {
                        "move": move, "deep": deep,
                        "chosen_visits": ch[0], "chosen_winrate": ch[1],
                        "breadth": breadth, "maxdepth": maxdepth,
                        "top_visits": dict(sorted(
                            ((m, v) for m, (v, _, _) in per_move.items()),
                            key=lambda kv: -kv[1])[:5])}
                except Exception as e:
                    print(f"SKIP {pos['game_hash']} {bname}: {e}",
                          file=sys.stderr)
                    ok = False
                    break
            if ok:
                out.write(json.dumps(rec) + "\n")
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(positions)}", file=sys.stderr)
    finally:
        out.close()
        for bot in bots.values():
            bot.close()
        labeler.close()


if __name__ == "__main__":
    main()
