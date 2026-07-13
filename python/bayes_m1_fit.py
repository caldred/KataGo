#!/usr/bin/env python3
"""M1 head fitting + gate report (docs/bayes-m1-gate.md; recipe follows
bmcts src/bmcts/heads.py, adapted where pre-registered).

Input: JSONL from bayes_m1_extract.py. Residual per child:
r = first raw eval - deep reference value (white perspective, winrate units).

Fits:
  sigma  — is rawStWrError calibrated as sd of r? Ratios global/by-phase/
           by-decile; log-space correction log r^2 ~ a + b log st_err^2
           (Gaussian-consistent: predicted log sigma^2 adds +1.27036,
           -E[log chi^2_1], to the OLS intercept).
  rho    — pooled within-set residual correlation, heads.py cross-moment
           estimator, pair-count weighting; cluster bootstrap by game.
  sigma_d— LOG-LOG regression of within-set deep-value variance on
           expansion-time features only; lognormal mean correction at
           prediction; model chosen on held-out calibration only.

Held-out split: by game_id, 80/20 (game_id % 5 == 4 held out).
"""
import argparse
import json
import math
import sys

import numpy as np

LOG_CHI2_1_MEAN = -1.27036  # E[log chi^2_1]


def load(path):
    sets_ = []
    with open(path) as f:
        for line in f:
            if line.strip():
                sets_.append(json.loads(line))
    return sets_


def phase_of(move_num):
    return 0 if move_num <= 10 else (1 if move_num <= 25 else 2)


def features(rec):
    """Expansion-time features for sigma_d (pre-registered candidate set)."""
    pri = np.array([c["prior"] for c in rec["children"]])
    topk_mass = float(pri.sum())
    q = pri / pri.sum()
    ent = float(-(q * np.log(q)).sum())
    return {
        "log_st2": math.log(max(rec["st_err"], 1e-6) ** 2),
        "entropy": ent,
        "topk_mass": topk_mass,
        "log_nlegal": math.log(max(rec["n_legal"], 1)),
        "move_num": float(rec["move_num"]),
    }


FEATURE_ORDER = ["log_st2", "entropy", "topk_mass", "log_nlegal", "move_num"]


def build(sets_):
    rows = []
    for rec in sets_:
        r = np.array([c["raw"] - c["deep"] for c in rec["children"]])
        st = np.array([max(c["st_err"], 1e-6) for c in rec["children"]])
        deeps = np.array([c["deep"] for c in rec["children"]])
        rows.append({
            "game": rec["game_id"],
            "phase": phase_of(rec["move_num"]),
            "held": rec["game_id"] % 5 == 4,
            "r": r, "st": st,
            "spread2": float(np.var(deeps, ddof=1)),
            "feat": features(rec),
            "k": len(r),
        })
    return rows


# ---------- sigma ----------

def sigma_report(rows, out):
    def ratio(sel):
        if not sel:
            return float("nan")
        rr = np.concatenate([w["r"] for w in sel])
        ss = np.concatenate([w["st"] for w in sel])
        return math.sqrt(float(np.mean(rr * rr)) / float(np.mean(ss * ss)))

    train = [w for w in rows if not w["held"]]
    held = [w for w in rows if w["held"]]
    out["sigma_ratio_train"] = ratio(train)
    out["sigma_ratio_held"] = ratio(held)
    out["sigma_ratio_by_phase"] = {
        ph: ratio([w for w in rows if w["phase"] == ph])
        for ph in (0, 1, 2) if any(w["phase"] == ph for w in rows)}

    rr = np.concatenate([w["r"] for w in rows])
    ss = np.concatenate([w["st"] for w in rows])
    dec = np.quantile(ss, np.linspace(0, 1, 11))
    by_dec = []
    for i in range(10):
        m = (ss >= dec[i]) & (ss <= dec[i + 1] if i == 9 else ss < dec[i + 1])
        if m.sum() > 5:
            by_dec.append(round(math.sqrt(
                float(np.mean(rr[m] ** 2)) / float(np.mean(ss[m] ** 2))), 3))
    out["sigma_ratio_by_decile"] = by_dec

    # log-space correction fitted on train children
    rt = np.concatenate([w["r"] for w in train])
    st = np.concatenate([w["st"] for w in train])
    y = np.log(np.maximum(rt * rt, 1e-12))
    x = np.log(st * st)
    b = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    a = float(y.mean() - b * x.mean())
    out["sigma_corr_coef"] = {"a": a, "b": b}

    def corrected_ratio(sel):
        if not sel:
            return float("nan")
        rr = np.concatenate([w["r"] for w in sel])
        ss = np.concatenate([w["st"] for w in sel])
        pred2 = np.exp(a + b * np.log(ss * ss) - LOG_CHI2_1_MEAN)
        return math.sqrt(float(np.mean(rr * rr)) / float(np.mean(pred2)))

    out["sigma_ratio_held_corrected"] = corrected_ratio(held)


# ---------- rho ----------

def rho_of(sel):
    num = den = 0.0
    for w in sel:
        r, k = w["r"], w["k"]
        s = r.sum()
        cross = (s * s - (r * r).sum())        # sum over ordered pairs
        num += cross
        den += (k - 1) * (r * r).sum()          # pair-count-consistent norm
    return num / max(den, 1e-18)


def rho_report(rows, out, n_boot=1000, seed=7):
    out["rho_all"] = rho_of(rows)
    out["rho_by_phase"] = {
        ph: rho_of([w for w in rows if w["phase"] == ph])
        for ph in (0, 1, 2) if any(w["phase"] == ph for w in rows)}
    games = sorted({w["game"] for w in rows})
    by_game = {g: [w for w in rows if w["game"] == g] for g in games}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(games, size=len(games), replace=True)
        sel = [w for g in pick for w in by_game[g]]
        boots.append(rho_of(sel))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    out["rho_ci95"] = [float(lo), float(hi)]
    out["rho_ci_halfwidth"] = float((hi - lo) / 2)


# ---------- sigma_d ----------

def _design(rows, names):
    X = np.array([[w["feat"][n] for n in names] for w in rows])
    return np.column_stack([np.ones(len(rows)), X]) if names else \
        np.ones((len(rows), 1))


def sigma_d_report(rows, out, seed=11):
    train = [w for w in rows if not w["held"]]
    held = [w for w in rows if w["held"]]
    if not train or not held:
        out["sigma_d_models"] = "SKIPPED: empty train or held split"
        return
    ytr = np.log(np.maximum([w["spread2"] for w in train], 1e-12))
    yhe = np.log(np.maximum([w["spread2"] for w in held], 1e-12))

    candidates = [[], *([n] for n in FEATURE_ORDER), FEATURE_ORDER]
    results = {}
    for names in candidates:
        Xtr, Xhe = _design(train, names), _design(held, names)
        beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        res_tr = ytr - Xtr @ beta
        s2 = float(np.mean(res_tr ** 2))
        pred_he = Xhe @ beta
        res_he = yhe - pred_he
        # held-out R^2 in log space vs constant-only
        ss0 = float(np.sum((yhe - ytr.mean()) ** 2))
        r2 = 1 - float(np.sum(res_he ** 2)) / max(ss0, 1e-18)
        # calibration: realized spread2 / predicted E[spread2]
        pred2 = np.exp(pred_he + 0.5 * s2)
        calib = float(np.mean(np.exp(yhe) / pred2))
        results["+".join(names) if names else "const"] = {
            "beta": [round(float(x), 4) for x in beta], "s2_res": round(s2, 4),
            "heldout_logR2": round(r2, 4), "heldout_calib": round(calib, 3),
            "heldout_log_resid_sd": round(float(np.std(res_he)), 3),
        }
    out["sigma_d_models"] = results

    # paired bootstrap: full model heldout R2 > 0?
    names = FEATURE_ORDER
    Xtr, Xhe = _design(train, names), _design(held, names)
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    res_full = (yhe - Xhe @ beta) ** 2
    res_const = (yhe - ytr.mean()) ** 2
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(yhe)
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        diffs.append(float(np.mean(res_const[idx]) - np.mean(res_full[idx])))
    out["sigma_d_full_beats_const_frac"] = float(np.mean(np.array(diffs) > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    rows = build(load(args.data))
    held_n = sum(1 for w in rows if w["held"])
    out = {"n_sets": len(rows), "n_sets_held": held_n,
           "n_games": len({w["game"] for w in rows}),
           "n_children": int(sum(w["k"] for w in rows))}
    if held_n < 50:
        print(f"WARNING: held-out sets {held_n} < 50 "
              f"(gate sanity floor — extend the run)", file=sys.stderr)
    sigma_report(rows, out)
    rho_report(rows, out)
    sigma_d_report(rows, out)

    print(json.dumps(out, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
