#!/usr/bin/env python3
"""M2 Gate B calibration probe driver (docs/bayes-m2-gate.md, Gate B).

Two phases:
  A. Position generation + reference values, via the stock analysis engine,
     exactly like the M1 phase-A extraction (python/bayes_m1_extract.py) but
     with seed base 500000 (disjoint from M1 and every gate ever run):
     25 self-play games from the raw policy (temp 1.0 for the first 6 moves,
     then 0.5), up to 3 positions per game at moves 4-40 spaced >= 6 apart,
     parent filter |rawWinrate - 0.5| <= 0.45, stop at 60 positions.
     Reference per position: maxVisits=1500 stock query -> rootInfo winrate
     (white perspective; reportAnalysisWinratesAs = WHITE in the config).
  B. Flag-on measurement, via a GTP engine started with the stock gtp config
     plus every parameter of cpp/configs/bayes_m2_9x9.cfg overridden in.
     For each position: clear_board, replay the moves, run the flag-on
     search with `kata-search <side>`, then read the root posterior with
     `kata-bayes-root`. The whole sweep runs once per visit budget B in
     {30, 100} (separate gtp processes).

NOTE on kata-search vs genmove: the gate readout must come from the SEARCHED
root. `genmove` plays the move, and Search::makeMove promotes the played
child to a fresh root COPY whose bayesState is NULL (SearchNode copy
constructor), so `kata-bayes-root` after genmove answers "none" — and even if
the state were copied it would describe the position AFTER the move.
`kata-search` runs the identical genmove search without playing the move,
leaving the searched root (and its bayesState) intact. Positions are replayed
from clear_board each time regardless.

Output: JSONL, one record per (position, B):
  {game_id, move_num, pla, ref, mu, sd, visits, B}
plus a final {"summary": ...} record; the summary is also printed to stdout.
z = (mu - ref) / sd, white perspective throughout.

The curiosity metric fraction(|mu-ref| < |searchWinrate-ref|) is NOT
COLLECTED by this driver (it is explicitly ungated; the ordinary root
winrate is not captured to keep the readout single-command).

Usage (full pre-registered run, operator-only):
  python3 bayes_m2_probe.py \
    --model ../models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz \
    --out ../bayes-data/m2-probe.jsonl

Smoke test of the plumbing (3 positions, B=30 only, 200-visit refs):
  python3 bayes_m2_probe.py --model ... --out ... --smoke
"""
import argparse
import json
import math
import os
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
PARENT_BAND = 0.45       # position filter (expansion-time info only)
POSITIONS_PER_GAME = 3
POSITION_SPACING = 6
MIN_MOVE = 4
MAX_MOVE = 40
SEED_BASE = 500000
COLS = "ABCDEFGHJ"       # GTP columns, no I

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def idx_to_gtp(idx):
    if idx == BOARD * BOARD:
        return "pass"
    y, x = divmod(idx, BOARD)
    return f"{COLS[x]}{BOARD - y}"


# ---------------------------------------------------------------------------
# Analysis engine (phase A) — same pattern as bayes_m1_extract.py
# ---------------------------------------------------------------------------

class AnalysisEngine:
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
                self.cond.wait(timeout=1200)
                if self.proc.poll() is not None and qid not in self.responses:
                    raise RuntimeError("analysis engine died")
            resp = self.responses.pop(qid)
        if "error" in resp:
            raise RuntimeError(f"analysis engine error: {resp}")
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
    """Self-play one game from the raw policy; return position snapshots."""
    moves = []
    positions = []
    next_pos = rng.randint(4, 12)
    passes = 0
    for move_num in range(MAX_GAME_MOVES):
        resp = eng.position_query(moves, visits=1, include_policy=True)
        root = resp["rootInfo"]
        policy = resp["policy"]
        raw = root["rawWinrate"]
        if abs(raw - 0.5) > RESIGN_BAND:
            break
        if (move_num >= max(next_pos, MIN_MOVE)
                and len(positions) < POSITIONS_PER_GAME
                and move_num <= MAX_MOVE):
            if abs(raw - 0.5) <= PARENT_BAND:
                positions.append({
                    "game_id": game_id,
                    "move_num": move_num,
                    "moves": list(moves),
                    "pla": "B" if len(moves) % 2 == 0 else "W",
                    "raw": raw,
                })
                next_pos = move_num + POSITION_SPACING
            else:
                next_pos = move_num + 2   # retry a bit later
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
    return positions


# ---------------------------------------------------------------------------
# GTP engine (phase B)
# ---------------------------------------------------------------------------

class GtpEngine:
    def __init__(self, katago, config, model, overrides):
        override_str = ",".join(f"{k}={v}" for k, v in overrides.items())
        self.proc = subprocess.Popen(
            [katago, "gtp", "-model", model, "-config", config,
             "-override-config", override_str],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True)

    def cmd(self, command):
        if self.proc.poll() is not None:
            raise RuntimeError(f"gtp engine died before command: {command!r}")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(f"gtp engine died during command: {command!r}")
            line = line.rstrip("\r\n")
            if line == "":
                if lines:
                    break
                continue
            lines.append(line)
        resp = "\n".join(lines)
        if resp.startswith("?"):
            raise RuntimeError(f"gtp error for {command!r}: {resp}")
        if resp.startswith("= "):
            return resp[2:]
        if resp == "=":
            return ""
        raise RuntimeError(f"unparseable gtp response for {command!r}: {resp}")

    def close(self):
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def parse_bayes_config(path):
    """key=value pairs from the bayes cfg, comments stripped."""
    overrides = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            overrides[k.strip()] = v.strip()
    if "useBayesSearch" not in overrides:
        raise RuntimeError(f"{path} does not look like the bayes config")
    return overrides


def parse_bayes_root(resp):
    """'mu <f> sd <f> vsKids <f> resolvable <f> visits <i>' -> dict."""
    if resp.strip() == "none":
        raise RuntimeError("kata-bayes-root answered 'none' (no bayes root state)")
    toks = resp.split()
    if len(toks) % 2 != 0:
        raise RuntimeError(f"unparseable kata-bayes-root response: {resp!r}")
    d = dict(zip(toks[0::2], toks[1::2]))
    for key in ("mu", "sd", "vsKids", "resolvable", "visits"):
        if key not in d:
            raise RuntimeError(f"missing {key!r} in kata-bayes-root response: {resp!r}")
    out = {
        "mu": float(d["mu"]),
        "sd": float(d["sd"]),
        "vsKids": float(d["vsKids"]),
        "resolvable": float(d["resolvable"]),
        "visits": int(d["visits"]),
    }
    #M3 extension (docs/bayes-m3-gate.md Gate 1): chosen move + claimed P(best)
    if "move" in d:
        out["move"] = d["move"]
    if "pBest" in d:
        out["pBest"] = float(d["pBest"])
    return out


def measure_positions(katago, gtp_config, model, bayes_overrides, positions, b_visits):
    """Run the flag-on sweep at one visit budget; yield per-position readouts."""
    overrides = dict(bayes_overrides)
    overrides.update({
        "maxVisits": str(b_visits),
        "ponderingEnabled": "false",
        "allowResignation": "false",
        "logAllGTPCommunication": "false",
        "logSearchInfo": "false",
        "logToStderr": "false",
    })
    eng = GtpEngine(katago, gtp_config, model, overrides)
    results = []
    try:
        eng.cmd(f"boardsize {BOARD}")
        eng.cmd(f"komi {KOMI:g}")
        eng.cmd(f"kata-set-rules {RULES}")
        for i, pos in enumerate(positions):
            eng.cmd("clear_board")
            for pla, mv in pos["moves"]:
                eng.cmd(f"play {pla} {mv}")
            eng.cmd(f"kata-search {pos['pla']}")
            readout = parse_bayes_root(eng.cmd("kata-bayes-root"))
            results.append(readout)
            print(f"  B={b_visits} position {i+1}/{len(positions)} "
                  f"(game {pos['game_id']} mv {pos['move_num']}): "
                  f"mu={readout['mu']:.4f} sd={readout['sd']:.4f} "
                  f"ref={pos['ref']:.4f} visits={readout['visits']}",
                  file=sys.stderr)
    finally:
        eng.close()
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def mean_std(xs):
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    if n < 2:
        return m, None
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def phase_of(move_num):
    if move_num <= 10:
        return "moves<=10"
    if move_num <= 25:
        return "moves11-25"
    return "moves>25"


def summarize_b(records):
    zs = []
    z_by_phase = {"moves<=10": [], "moves11-25": [], "moves>25": []}
    n_zero_sd = 0
    for r in records:
        if r["sd"] <= 0:
            n_zero_sd += 1
            continue
        z = (r["mu"] - r["ref"]) / r["sd"]
        zs.append(z)
        z_by_phase[phase_of(r["move_num"])].append(z)
    mean_z, std_z = mean_std(zs)
    phases = {}
    for name, pz in z_by_phase.items():
        m, s = mean_std(pz)
        phases[name] = {"n": len(pz), "mean_z": m, "std_z": s}

    by_sd = sorted(records, key=lambda r: r["sd"])
    n = len(by_sd)
    deciles = []
    for d in range(10):
        lo = (d * n) // 10
        hi = ((d + 1) * n) // 10
        chunk = by_sd[lo:hi]
        if not chunk:
            continue
        m_sd, _ = mean_std([r["sd"] for r in chunk])
        m_err, _ = mean_std([abs(r["mu"] - r["ref"]) for r in chunk])
        deciles.append({"decile": d + 1, "n": len(chunk),
                        "mean_claimed_sd": m_sd, "mean_abs_error": m_err})

    return {
        "n": len(zs),
        "n_zero_sd_skipped": n_zero_sd,
        "mean_z": mean_z,
        "std_z": std_z,
        "z_by_phase": phases,
        "claimed_sd_vs_abs_error_deciles": deciles,
        "fraction_mu_closer_than_search_winrate":
            "NOT COLLECTED (ungated curiosity metric; ordinary root winrate "
            "not captured by this driver)",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", default=os.path.join(REPO_ROOT, "cpp/build/katago"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--analysis-config",
                    default=os.path.join(REPO_ROOT, "cpp/configs/bayes_m1_analysis.cfg"))
    ap.add_argument("--gtp-config",
                    default=os.path.join(REPO_ROOT, "cpp/configs/gtp_example.cfg"))
    ap.add_argument("--bayes-config",
                    default=os.path.join(REPO_ROOT, "cpp/configs/bayes_m2_9x9.cfg"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--max-positions", type=int, default=60)
    ap.add_argument("--ref-visits", type=int, default=1500)
    ap.add_argument("--budgets", default="30,100",
                    help="comma-separated maxVisits budgets for the flag-on sweep")
    ap.add_argument("--game-workers", type=int, default=8)
    ap.add_argument("--ref-workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing test: 3 positions, B=30 only, 200-visit refs")
    args = ap.parse_args()

    n_games = args.games
    max_positions = args.max_positions
    ref_visits = args.ref_visits
    budgets = [int(b) for b in args.budgets.split(",")]
    if args.smoke:
        max_positions = 3
        ref_visits = 200
        budgets = [30]

    bayes_overrides = parse_bayes_config(args.bayes_config)

    # ---- Phase A: positions + refs (analysis engine) ----
    eng = AnalysisEngine(args.katago, args.analysis_config, args.model)
    positions = []
    try:
        if args.smoke:
            print("phase A (smoke): generating games until 3 positions",
                  file=sys.stderr)
            g = 0
            while len(positions) < max_positions and g < n_games:
                positions.extend(gen_game(eng, g, random.Random(SEED_BASE + g)))
                g += 1
        else:
            print(f"phase A: generating {n_games} games (seed base {SEED_BASE})",
                  file=sys.stderr)
            with ThreadPoolExecutor(args.game_workers) as pool:
                futs = [pool.submit(gen_game, eng, g, random.Random(SEED_BASE + g))
                        for g in range(n_games)]
                for f in futs:
                    positions.extend(f.result())
        positions.sort(key=lambda p: (p["game_id"], p["move_num"]))
        positions = positions[:max_positions]
        print(f"phase A: {len(positions)} positions; "
              f"labeling refs at {ref_visits} visits", file=sys.stderr)

        def label_ref(pos):
            deep = eng.position_query(pos["moves"], visits=ref_visits)
            pos["ref"] = deep["rootInfo"]["winrate"]
        with ThreadPoolExecutor(args.ref_workers) as pool:
            list(pool.map(label_ref, positions))
    finally:
        eng.close()

    if len(positions) == 0:
        print("no positions generated; aborting", file=sys.stderr)
        sys.exit(1)

    # ---- Phase B: flag-on measurement per budget ----
    out = open(args.out, "a", buffering=1)
    summary = {}
    try:
        for b_visits in budgets:
            print(f"phase B: flag-on sweep at maxVisits={b_visits}", file=sys.stderr)
            readouts = measure_positions(
                args.katago, args.gtp_config, args.model, bayes_overrides,
                positions, b_visits)
            records = []
            for pos, ro in zip(positions, readouts):
                rec = {
                    "game_id": pos["game_id"],
                    "move_num": pos["move_num"],
                    "pla": pos["pla"],
                    "ref": pos["ref"],
                    "mu": ro["mu"],
                    "sd": ro["sd"],
                    "visits": ro["visits"],
                    "B": b_visits,
                }
                records.append(rec)
                out.write(json.dumps(rec) + "\n")
            summary[f"B{b_visits}"] = summarize_b(records)

        summary_obj = {
            "n_positions": len(positions),
            "seed_base": SEED_BASE,
            "ref_visits": ref_visits,
            "smoke": bool(args.smoke),
            "per_budget": summary,
        }
        out.write(json.dumps({"summary": summary_obj}) + "\n")
        print(json.dumps(summary_obj, indent=2))
    finally:
        out.close()


if __name__ == "__main__":
    main()
