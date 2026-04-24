#!/usr/bin/env python3
"""Analyze cross_matrix.json: print MCC matrix per model + summary stats.

Shows:
  - Per-model 3x3 MCC table (train × test), diagonals marked
  - Drop delta (diag - avg off-diag) — negative = corpus-specific overfit
  - Best / worst transfers per model
  - Model ranking by mean diag MCC and mean off-diag MCC
  - NaN-clamped cells flagged

Usage:
    python3 scripts/thesis_followups/cross_dataset_matrix/analyze.py \\
        [--json results/cross_matrix/summary/cross_matrix.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CORPORA = ["human", "non_human", "all"]


def _get_mcc(cell: dict) -> tuple[float, float, int, int]:
    """Return (mean_mcc, std_mcc, n_seeds, total_nan). Zero if empty."""
    if not isinstance(cell, dict):
        return 0.0, 0.0, 0, 0
    agg = cell.get("aggregate", {}).get("mcc", {})
    n = cell.get("n_seeds", 0)
    per = cell.get("per_seed", {})
    total_nan = 0
    if isinstance(per, dict):
        for m in per.values():
            if isinstance(m, dict):
                total_nan += int(m.get("n_nan_clamped", 0))
    return float(agg.get("mean", 0.0)), float(agg.get("std", 0.0)), int(n), total_nan


def _print_matrix(model: str, trains: dict) -> dict:
    """Print 3x3 matrix for one model, return summary metrics."""
    print(f"\n{'=' * 68}")
    print(f"  {model.upper():^64}")
    print(f"{'=' * 68}")
    header = f"{'train \\ test':<14}" + "".join(f"{c:>16s}" for c in CORPORA)
    print(header)
    print("-" * 68)
    diag_vals, off_vals = [], []
    nan_flag = False
    for train in CORPORA:
        row = f"{train:<14}"
        tests_dict = trains.get(train, {})
        for test in CORPORA:
            cell = tests_dict.get(test, {})
            mean, std, n, nan = _get_mcc(cell)
            if n == 0:
                row += f"{'—':>16s}"
                continue
            marker = "*" if train == test else " "
            warn = "!" if nan > 0 else " "
            row += f"{marker}{mean:.3f}±{std:.3f}{warn}{'':>5s}"
            if train == test:
                diag_vals.append(mean)
            else:
                off_vals.append(mean)
            if nan > 0:
                nan_flag = True
        print(row)
    print("-" * 68)
    diag_mean = sum(diag_vals) / max(len(diag_vals), 1)
    off_mean = sum(off_vals) / max(len(off_vals), 1)
    drop = diag_mean - off_mean
    print(f"  diagonal mean: {diag_mean:.4f} (n={len(diag_vals)})")
    print(f"  off-diag mean: {off_mean:.4f} (n={len(off_vals)})")
    print(f"  drop (diag − off): {drop:+.4f}  "
          f"{'(strong corpus specificity)' if drop > 0.2 else '(transfers well)' if drop < 0.05 else ''}")
    if nan_flag:
        print("  ! = some seeds had NaN predictions clamped")
    print("  * = diagonal (train == test)")
    return {
        "diag_mean": diag_mean,
        "off_mean": off_mean,
        "drop": drop,
        "n_diag": len(diag_vals),
        "n_off": len(off_vals),
    }


def _best_worst(data: dict) -> None:
    """Report best / worst transfer per model."""
    print(f"\n{'=' * 68}")
    print(f"  {'BEST / WORST TRANSFERS (off-diagonal only)':^64}")
    print(f"{'=' * 68}")
    for model, trains in data.items():
        if model.startswith("_") or not isinstance(trains, dict):
            continue
        transfers = []
        for train in CORPORA:
            for test in CORPORA:
                if train == test:
                    continue
                cell = trains.get(train, {}).get(test, {})
                mean, std, n, _ = _get_mcc(cell)
                if n > 0:
                    transfers.append((train, test, mean, std, n))
        if not transfers:
            continue
        transfers.sort(key=lambda r: r[2], reverse=True)
        best = transfers[0]
        worst = transfers[-1]
        print(f"\n  {model:12s}")
        print(f"    best:  {best[0]:>10s} → {best[1]:<10s}  MCC={best[2]:.4f}±{best[3]:.4f}")
        print(f"    worst: {worst[0]:>10s} → {worst[1]:<10s}  MCC={worst[2]:.4f}±{worst[3]:.4f}")


def _ranking(summaries: dict) -> None:
    """Rank models by diag and off-diag mean MCC."""
    print(f"\n{'=' * 68}")
    print(f"  {'MODEL RANKING':^64}")
    print(f"{'=' * 68}")
    print(f"\n  By diagonal MCC (within-corpus performance):")
    ordered = sorted(summaries.items(), key=lambda kv: kv[1]["diag_mean"], reverse=True)
    for i, (model, s) in enumerate(ordered, 1):
        print(f"    {i}. {model:12s}  diag_MCC={s['diag_mean']:.4f}  (n={s['n_diag']} diag cells)")
    print(f"\n  By off-diagonal MCC (cross-corpus transfer):")
    ordered = sorted(summaries.items(), key=lambda kv: kv[1]["off_mean"], reverse=True)
    for i, (model, s) in enumerate(ordered, 1):
        print(f"    {i}. {model:12s}  off_MCC={s['off_mean']:.4f}  (n={s['n_off']} off-diag cells)")
    print(f"\n  By drop (small drop = good transfer):")
    ordered = sorted(summaries.items(), key=lambda kv: kv[1]["drop"])
    for i, (model, s) in enumerate(ordered, 1):
        print(f"    {i}. {model:12s}  drop={s['drop']:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json",
        default="results/cross_matrix/summary/cross_matrix.json",
    )
    args = ap.parse_args()
    p = Path(args.json)
    if not p.exists():
        print(f"[fatal] {p} not found. Run aggregate.py first.", file=sys.stderr)
        return 2
    with p.open() as f:
        data = json.load(f)
    summaries = {}
    for model, trains in data.items():
        if model.startswith("_") or not isinstance(trains, dict):
            continue
        summaries[model] = _print_matrix(model, trains)
    _best_worst(data)
    _ranking(summaries)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
