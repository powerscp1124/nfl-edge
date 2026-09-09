#!/usr/bin/env python3
"""Fit and validate a probability calibration from saved backtest candidates.

    python scripts/backtest.py --season 2025 --weeks 1-22 --save-bets bets.csv
    python scripts/calibrate_probs.py --bets bets.csv --train 1-11 --test 12-22

The split is *temporal*, never random. Shuffling props and holding out a
fraction lets week 14 inform a fit that is then scored on week 3, which is the
same leak ``walk_forward_splits`` exists to prevent -- and it would flatter the
calibration exactly the way it flatters a model.

The fit is done on every candidate the model evaluated, not only the ones the
thresholds backed. Fitting on selected bets alone conditions on the model's own
confidence and hides the overconfidence being corrected.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.calibration import (  # noqa: E402
    ProbabilityCalibrator,
    brier,
    logloss,
)
from app.core.edge import BetThresholds  # noqa: E402
from app.core.odds import american_to_decimal  # noqa: E402


def parse_weeks(spec: str) -> set[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return set(range(int(lo), int(hi) + 1))
    return {int(w) for w in spec.split(",")}


def week_of(game_id: str) -> int | None:
    parts = game_id.split("_")
    try:
        return int(parts[1])
    except (IndexError, ValueError):
        return None


def load(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if r["result"] not in ("win", "loss"):
            continue          # pushes carry no information about a probability
        week = week_of(r["game_id"])
        if week is None:
            continue
        out.append({
            "week": week,
            "prob": float(r["model_prob"]),
            "market_prob": float(r["market_prob"]),
            "american": float(r["american"]),
            "confidence": float(r["confidence"]),
            "won": 1.0 if r["result"] == "win" else 0.0,
            "selected": r["selected"].lower() == "true",
        })
    return out


def reliability(probs, outcomes, edges=(0.4, 0.5, 0.6, 0.7, 0.8, 1.01)):
    probs, outcomes = np.asarray(probs), np.asarray(outcomes)
    rows = []
    lo = 0.0
    for hi in edges:
        mask = (probs >= lo) & (probs < hi)
        if mask.sum():
            rows.append((lo, hi, int(mask.sum()),
                         float(probs[mask].mean()),
                         float(outcomes[mask].mean())))
        lo = hi
    return rows


def wagers(rows, probs, thresholds: BetThresholds, min_confidence: float):
    """Re-select under a given set of probabilities.

    Approximates the recommendation tiers with their edge and confidence
    floors rather than re-running ``evaluate_prop``; the review ceiling is
    applied the same way, so an implausible edge is still routed away.
    """
    picked = []
    for row, p in zip(rows, probs):
        edge = p - row["market_prob"]
        if edge >= thresholds.review_edge:
            continue
        if row["confidence"] < min_confidence:
            continue
        if edge < thresholds.lean_edge:
            continue
        profit = (american_to_decimal(row["american"]) - 1.0)
        picked.append({"edge": edge,
                       "pnl": profit if row["won"] else -1.0,
                       "won": row["won"]})
    return picked


def report(name, picked):
    if not picked:
        print(f"  {name:24} no qualifying bets")
        return
    pnl = sum(p["pnl"] for p in picked)
    print(f"  {name:24} n={len(picked):5d}  roi={pnl / len(picked):+.4f}  "
          f"win={np.mean([p['won'] for p in picked]):.4f}  "
          f"units={pnl:+.2f}  avg_edge={np.mean([p['edge'] for p in picked]):+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bets", required=True)
    ap.add_argument("--train", default="1-11")
    ap.add_argument("--test", default="12-22")
    ap.add_argument("--min-confidence", type=float, default=60.0)
    args = ap.parse_args()

    rows = load(args.bets)
    train_weeks, test_weeks = parse_weeks(args.train), parse_weeks(args.test)
    train = [r for r in rows if r["week"] in train_weeks]
    test = [r for r in rows if r["week"] in test_weeks]
    print(f"{len(rows)} graded candidates: "
          f"{len(train)} train (weeks {args.train}), "
          f"{len(test)} test (weeks {args.test})")
    if not train or not test:
        print("Not enough data on one side of the split.")
        return 1

    cal = ProbabilityCalibrator.fit([r["prob"] for r in train],
                                    [r["won"] for r in train])
    print(f"\nFitted on train: a={cal.a:.4f}  b={cal.b:+.4f}  "
          f"n={cal.n_samples}")
    print(f"  A stated 65% becomes {cal.apply(0.65):.1%} "
          f"({cal.confidence_retained:.0%} of the claimed deviation kept)")

    raw = np.array([r["prob"] for r in test])
    y = np.array([r["won"] for r in test])
    adj = cal.apply(raw)
    coin = np.full(y.size, 0.5)

    print("\n=== Held-out scores (lower is better) ===")
    print(f"  {'':14}{'brier':>10}{'log loss':>11}")
    for label, p in (("raw", raw), ("calibrated", adj), ("always 50%", coin)):
        print(f"  {label:14}{brier(p, y):10.5f}{logloss(p, y):11.5f}")

    print("\n=== Held-out reliability ===")
    # Bin by the RAW probability throughout, and report what the calibrated
    # map says about the same rows. Binning each series by its own values
    # would compare different populations, and zipping the two would silently
    # drop every bin the calibrated (much narrower) series does not reach.
    print(f"  {'raw bin':>10} {'n':>6} {'raw pred':>9} {'cal pred':>9} "
          f"{'actual':>8}")
    lo = 0.0
    for hi in (0.4, 0.5, 0.6, 0.7, 0.8, 1.01):
        mask = (raw >= lo) & (raw < hi)
        if mask.sum():
            print(f"  {lo:.0%}-{hi:.0%} {int(mask.sum()):6d} "
                  f"{raw[mask].mean():9.3f} {adj[mask].mean():9.3f} "
                  f"{y[mask].mean():8.3f}")
        lo = hi

    print("\n=== What the thresholds would back on the test weeks ===")
    thresholds = BetThresholds()
    report("raw probabilities", wagers(test, raw, thresholds,
                                       args.min_confidence))
    report("calibrated", wagers(test, adj, thresholds, args.min_confidence))

    print("\nProjections are probabilistic estimates with real uncertainty. "
          "No wager is guaranteed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
