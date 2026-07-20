"""Count distinct games (by move sequence) per match dir, and score on
distinct games only (one vote per distinct game+color assignment)."""
import math
import re
import sys
from collections import Counter
from pathlib import Path

data = Path(r"C:\Users\calal\code\KataGo\bayes-data")
MOVES = re.compile(r";[BW]\[([a-z]{0,2})\]")

for d in sys.argv[1:]:
    games = {}
    dup = 0
    for f in (data / d).glob("*.sgfs"):
        for line in f.open(encoding="utf-8", errors="replace"):
            if not line.startswith("(;"):
                continue
            m = re.search(r"RE\[([BW0])", line)
            if m is None:
                continue
            if "PB[puct]" in line:
                bot = "W"
            elif "PW[puct]" in line:
                bot = "B"
            else:
                continue
            key = (bot, tuple(MOVES.findall(line)))
            if key in games:
                dup += 1
                continue
            r = m.group(1)
            games[key] = ("D" if r == "0" else "W" if r == bot else "L")
    c = Counter(games.values())
    w, l, dr = c["W"], c["L"], c["D"]
    n = w + l + dr
    if n == 0:
        print(f"{d}: no games")
        continue
    p = (w + 0.5 * dr) / n
    p = min(max(p, 1e-9), 1 - 1e-9)
    elo = 400.0 * math.log10(p / (1 - p))
    var = ((w * (1 - p) ** 2 + dr * (0.5 - p) ** 2 + l * p ** 2) / n) / n
    sd = 400.0 / math.log(10) / (p * (1 - p)) * math.sqrt(max(var, 1e-12))
    print(f"{d}: total={n + dup} distinct={n} (dup {dup}) "
          f"W{w}/L{l}/D{dr}  Elo {elo:+.0f} +/- {sd:.0f}")
