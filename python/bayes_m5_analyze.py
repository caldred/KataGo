#!/usr/bin/env python3
"""M5 tail-data head comparison: registered analysis
(docs/bayes-m5-tailheads.md). Runs ONCE on the completed dataset.

Scores the frozen candidates C1-C4 on pairwise mover-perspective
contrasts vs the ref child, buckets by longshot prior and parent
extremity, S1 descriptive sigma checks, S2 heteroscedastic fit
(even-parity fit / odd-parity validation), decision rules R1/R2.

Usage:
  python3 bayes_m5_analyze.py --data ../bayes-data/m5-tail.jsonl \
    --report ../docs/bayes-m5-report.json
"""
import argparse
import json
import math
import sys

import numpy as np
from scipy.special import ndtr, log_ndtr, logsumexp  # Phi, log Phi

DMEAN = 0.0638
FLOOR = 5e-3
LOGIT_SLOPE = 0.4235
SIGMA_R = 0.119                    # shipped const residual sigma_r
PAIR_CLAIM = math.sqrt(2.0) * SIGMA_R
SIGMA_A, SIGMA_B = 0.1560, 0.9826  # first-eval corrected sigma claim
INV_S = 0.119
CHI2_LOG_OFFSET = 1.2704           # -E[log chi^2_1]
PRIOR_BINS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1.0]
PRIOR_LABELS = ["[1e-4,3e-4)", "[3e-4,1e-3)", "[1e-3,3e-3)",
                "[3e-3,1e-2)", "[1e-2,3e-2)", "[3e-2,1]"]
TAIL_BINS = {0, 1, 2}
EXT_BINS = [0.0, 0.15, 0.30, 0.45, 0.51]
EXT_LABELS = ["[0,0.15)", "[0.15,0.30)", "[0.30,0.45)", "[0.45,0.5]"]
N_BOOT = 2000
BOOT_SEED = 20260717


def logit(v):
    v = min(max(v, 0.01), 0.99)
    return math.log(v / (1.0 - v))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def pbest_invert(priors, s, tau):
    """Solve for Gaussian means matching argmax probabilities.

    priors: array of legal-move priors (>0). Returns means (identified up
    to an additive constant) or None if not converged.

    Numerics per doc Amendment A: log-space quadrature (no underflow) and
    a per-arm Newton preconditioner with step clipping; same fixed point
    as the originally registered scheme (verified to 3e-8 on the smoke
    records where that scheme converged), same tolerance.
    """
    q = np.maximum(np.asarray(priors, dtype=float), 1e-12) ** (1.0 / tau)
    q = q / q.sum()
    logq = np.log(q)
    m = s * (logq - logq.mean())  # rough init, correct ordering
    for _ in range(2000):
        lo = m.min() - 6.0 * s
        hi = m.max() + 6.0 * s
        x = np.linspace(lo, hi, 2001)
        dx = x[1] - x[0]
        z = (x[None, :] - m[:, None]) / s          # k x G
        logphi = -0.5 * z * z - math.log(math.sqrt(2 * math.pi) * s)
        logPhi = log_ndtr(z)
        S = logPhi.sum(axis=0)                     # G
        logint = logphi + (S[None, :] - logPhi)    # k x G
        logP = logsumexp(logint, axis=1) + math.log(dx)
        logP = logP - logsumexp(logP)
        err = logq - logP
        if np.abs(err).max() < 1e-6:
            return m
        # dlogP_a/dm_a ~ (x*_a - m_a)/s^2 for arms far below the leader,
        # floored at 1/s for self-dominated arms; x*_a = integrand argmax.
        xstar = x[np.argmax(logint, axis=1)]
        denom = np.maximum(1.0 / s, (xstar - m) / (s * s))
        m = m + 0.7 * np.clip(err / denom, -2.0 * s, 2.0 * s)
    return None


def load(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    return recs


def build_rows(recs):
    """One row per scored (non-ref) child; C4 solved per position."""
    rows = []
    c4_fail = 0
    no_scored = 0
    for rec in recs:
        mover = 1.0 if rec["pla"] == "W" else -1.0
        kids = rec["children"]
        ref = next(k for k in kids if k["tag"] == "ref")
        others = [k for k in kids if k["tag"] != "ref"]
        if not others:
            no_scored += 1
            continue
        ext = abs(rec["parent_raw"] - 0.5)
        l0 = logit(0.5 + mover * (rec["parent_raw"] - 0.5))
        lp = {i: p for i, p in rec["legal_policy"]}
        idxs = sorted(lp)
        pri = [lp[i] for i in idxs]
        m1 = pbest_invert(pri, INV_S, 1.0)
        m2 = pbest_invert(pri, INV_S, 0.7)
        if m1 is None or m2 is None:
            c4_fail += 1
            continue
        gtp_to_pos = {}
        for k in kids:
            gtp_to_pos[k["move"]] = None
        # map moves back to policy indices via prior match
        def move_mean(kid, mvec):
            # find index whose prior matches this child's recorded prior
            cands = [j for j, i in enumerate(idxs)
                     if abs(lp[i] - kid["prior"]) < 1e-12]
            return mvec[cands[0]] if cands else None
        mref1 = move_mean(ref, m1)
        mref2 = move_mean(ref, m2)
        for k in others:
            y = mover * (k["deep"] - ref["deep"])
            pa, pr = k["prior"], ref["prior"]
            c1 = DMEAN * (math.log(max(pa, FLOOR)) - math.log(max(pr, FLOOR)))
            c2 = DMEAN * (math.log(pa) - math.log(pr))
            c3 = sigmoid(l0 + LOGIT_SLOPE * (math.log(pa) - math.log(pr))) \
                - sigmoid(l0)
            ma1 = move_mean(k, m1)
            ma2 = move_mean(k, m2)
            if ma1 is None or mref1 is None:
                continue
            c4 = ma1 - mref1
            c4t = ma2 - mref2
            pbin = np.searchsorted(PRIOR_BINS, pa, side="right") - 1
            pbin = min(max(pbin, 0), len(PRIOR_LABELS) - 1)
            ebin = np.searchsorted(EXT_BINS, ext, side="right") - 1
            ebin = min(max(ebin, 0), len(EXT_LABELS) - 1)
            first_resid = mover * (k["raw"] - k["deep"])
            first_claim = math.exp(SIGMA_A) * (k["st_err"] ** SIGMA_B)
            rows.append({
                "game": rec["game_hash"], "tag": k["tag"], "prior": pa,
                "pbin": int(pbin), "ebin": int(ebin), "ext": ext,
                "parity": int(rec["game_hash"][-1], 16) % 2,
                "y": y, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c4t": c4t,
                "first_resid": first_resid, "first_claim": first_claim,
                "n_kids": len(others),
            })
    return rows, c4_fail, no_scored


def cluster_boot(rows, stat, n=N_BOOT, seed=BOOT_SEED):
    """Cluster bootstrap by game; stat maps row-list -> float or None."""
    games = sorted({r["game"] for r in rows})
    by_game = {g: [] for g in games}
    for r in rows:
        by_game[r["game"]].append(r)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(games), len(games), replace=True)
        sample = [r for i in pick for r in by_game[games[i]]]
        v = stat(sample)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def resid(rows, cand):
    return np.array([r["y"] - r[cand] for r in rows])


def bucket_stats(rows, cands):
    out = {}
    for axis, nb, labels in (("pbin", len(PRIOR_LABELS), PRIOR_LABELS),
                             ("ebin", len(EXT_LABELS), EXT_LABELS)):
        out[axis] = []
        for b in range(nb):
            sub = [r for r in rows if r[axis] == b]
            entry = {"bucket": labels[b], "n": len(sub)}
            if len(sub) >= 5:
                for c in cands:
                    rr = resid(sub, c)
                    lo, hi = cluster_boot(
                        sub, lambda s, c=c: float(np.mean(resid(s, c)))
                        if s else None)
                    entry[c] = {
                        "bias": float(rr.mean()), "bias_ci": [lo, hi],
                        "sd": float(rr.std(ddof=1)) if len(rr) > 1 else None,
                    }
                # S1a: C2 residual sd ratio vs homoscedastic pairwise claim
                lo, hi = cluster_boot(
                    sub, lambda s: float(np.std(resid(s, "c2"), ddof=1)
                                         / PAIR_CLAIM)
                    if len(s) > 1 else None)
                entry["s1a_ratio"] = {
                    "ratio": float(resid(sub, "c2").std(ddof=1) / PAIR_CLAIM),
                    "ci": [lo, hi]}
                # S1b: first-eval ratio RMS(resid)/RMS(claim)
                fr = np.array([r["first_resid"] for r in sub])
                fc = np.array([r["first_claim"] for r in sub])
                lo, hi = cluster_boot(
                    sub, lambda s: float(
                        np.sqrt(np.mean([r["first_resid"] ** 2 for r in s]))
                        / np.sqrt(np.mean([r["first_claim"] ** 2 for r in s])))
                    if s else None)
                entry["s1b_ratio"] = {
                    "ratio": float(np.sqrt((fr ** 2).mean())
                                   / np.sqrt((fc ** 2).mean())),
                    "ci": [lo, hi]}
            out[axis].append(entry)
    return out


def tail_sse_pairs(rows, cands):
    """Paired cluster-bootstrap win fractions on tail-pooled SSE."""
    tail = [r for r in rows if r["pbin"] in TAIL_BINS]
    games = sorted({r["game"] for r in tail})
    by_game = {g: [] for g in games}
    for r in tail:
        by_game[r["game"]].append(r)
    rng = np.random.default_rng(BOOT_SEED + 1)
    wins = {(a, b): 0 for a in cands for b in cands if a != b}
    for _ in range(N_BOOT):
        pick = rng.choice(len(games), len(games), replace=True)
        sample = [r for i in pick for r in by_game[games[i]]]
        sse = {c: float(np.sum(resid(sample, c) ** 2)) for c in cands}
        for a in cands:
            for b in cands:
                if a != b and sse[a] < sse[b]:
                    wins[(a, b)] += 1
    point = {c: float(np.sum(resid(tail, c) ** 2)) for c in cands}
    return ({f"{a}<{b}": w / N_BOOT for (a, b), w in wins.items()},
            point, len(tail))


def max_z(recs_rows, cands):
    """Per-position max |z| vs Gaussian order-stat expectation."""
    by_pos = {}
    for r in recs_rows:
        by_pos.setdefault((r["game"], r["n_kids"]), []).append(r)
    out = {}
    for c in cands:
        obs = []
        exp = []
        for (_, nk), rs in by_pos.items():
            zs = np.abs(resid(rs, c)) / PAIR_CLAIM
            obs.append(float(zs.max()))
            n = len(rs)
            # E[max of n |N(0,1)|] by quadrature
            x = np.linspace(0, 8, 4001)
            cdf = (2 * ndtr(x) - 1) ** n
            exp.append(float(np.trapezoid(1 - cdf, x)))
        out[c] = {"mean_obs": float(np.mean(obs)),
                  "mean_gauss_exp": float(np.mean(exp))}
    return out


def s2_fit(rows):
    """Heteroscedastic sigma_r: fit even-parity, validate odd."""
    fit = [r for r in rows if r["parity"] == 0]
    val = [r for r in rows if r["parity"] == 1]
    if len(fit) < 30 or len(val) < 30:
        return {"error": "insufficient data", "n_fit": len(fit),
                "n_val": len(val)}
    X = np.array([[1.0, math.log(max(r["prior"], 1e-4)), r["ext"]]
                  for r in fit])
    rr = resid(fit, "c2")
    yy = np.log(np.maximum(rr ** 2, 1e-12))
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)

    def sig2(r):
        return math.exp(coef[0] + coef[1] * math.log(max(r["prior"], 1e-4))
                        + coef[2] * r["ext"] + CHI2_LOG_OFFSET)
    cal = {}
    for axis, nb, labels in (("pbin", len(PRIOR_LABELS), PRIOR_LABELS),
                             ("ebin", len(EXT_LABELS), EXT_LABELS)):
        cal[axis] = []
        for b in range(nb):
            sub = [r for r in val if r[axis] == b]
            if len(sub) < 30:
                cal[axis].append({"bucket": labels[b], "n": len(sub),
                                  "ratio": None})
                continue
            rv = resid(sub, "c2")
            ratio = float(np.mean(rv ** 2
                                  / np.array([sig2(r) for r in sub])))
            cal[axis].append({"bucket": labels[b], "n": len(sub),
                              "ratio": ratio,
                              "in_band": bool(0.7 <= ratio <= 1.4)})
    return {"coef": [float(c) for c in coef],
            "n_fit": len(fit), "n_val": len(val), "heldout_calibration": cal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    recs = load(args.data)
    rows, c4_fail, no_scored = build_rows(recs)
    cands = ["c1", "c2", "c3", "c4", "c4t"]
    n_tail = sum(1 for r in rows if r["pbin"] in TAIL_BINS)
    n_games = len({r["game"] for r in rows})

    report = {
        "n_positions": len(recs), "n_games_scored": n_games,
        "n_scored_children": len(rows), "n_tail_children": n_tail,
        "c4_nonconverged_positions": c4_fail,
        "positions_without_scored_children": no_scored,
        "sanity_floor": {"positions_ok": len(recs) >= 150,
                         "tail_ok": n_tail >= 300},
        "buckets": bucket_stats(rows, cands),
    }
    wins, sse_point, n_tail2 = tail_sse_pairs(rows, cands)
    report["tail_sse"] = {"point": sse_point, "win_frac": wins,
                          "n": n_tail2}
    report["max_z"] = max_z(rows, cands)
    report["s2"] = s2_fit(rows)

    with open(args.report, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {args.report}", file=sys.stderr)

    # Console summary
    print(f"\npositions={len(recs)} scored_children={len(rows)} "
          f"tail={n_tail} c4fail={c4_fail}")
    for axis in ("pbin", "ebin"):
        print(f"\n== bias by {axis} ==")
        hdr = f"{'bucket':>12} {'n':>5}" + "".join(
            f" {c:>16}" for c in cands)
        print(hdr)
        for e in report["buckets"][axis]:
            if "c1" not in e:
                print(f"{e['bucket']:>12} {e['n']:>5}  (too few)")
                continue
            line = f"{e['bucket']:>12} {e['n']:>5}"
            for c in cands:
                b = e[c]["bias"]
                lo, hi = e[c]["bias_ci"]
                sig = "*" if lo is not None and (lo > 0 or hi < 0) else " "
                line += f" {b:+.3f}[{lo:+.2f},{hi:+.2f}]{sig}"
            print(line)
        print(f"{'':>12} S1a sd-ratio / S1b first-eval ratio:")
        for e in report["buckets"][axis]:
            if "s1a_ratio" not in e:
                continue
            a = e["s1a_ratio"]
            b = e["s1b_ratio"]
            print(f"{e['bucket']:>12} {e['n']:>5} "
                  f"s1a={a['ratio']:.2f}[{a['ci'][0]:.2f},{a['ci'][1]:.2f}] "
                  f"s1b={b['ratio']:.2f}[{b['ci'][0]:.2f},{b['ci'][1]:.2f}]")
    print("\n== tail-pooled SSE ==")
    for c in cands:
        print(f"  {c}: {sse_point[c]:.3f}")
    print("win fractions (row beats col):")
    for k, v in sorted(wins.items()):
        print(f"  {k}: {v:.3f}")
    print("\n== max|z| ==")
    for c, v in report["max_z"].items():
        print(f"  {c}: obs {v['mean_obs']:.2f} vs gauss "
              f"{v['mean_gauss_exp']:.2f}")
    print("\n== S2 hetero fit ==")
    print(json.dumps(report["s2"], indent=1)[:2000])


if __name__ == "__main__":
    main()
