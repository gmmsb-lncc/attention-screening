#!/usr/bin/env python3
"""Audit NaN-clamped cells in cross_matrix aggregate output.

Reads cross_matrix.json (written by aggregate.py) and prints every
(model, train, test) cell where at least one seed had its predictions
clamped from NaN to 0.5. Identifies which baselines broke under which
cross-dataset transfer directions.

Usage:
    python3 scripts/thesis_followups/cross_dataset_matrix/nan_audit.py \\
        [--json results/cross_matrix/summary/cross_matrix.json]

Output format:
    <model>  <train> -> <test>  nan_seeds=<bad>/<total>  avg_nan=<float>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json",
        default="results/cross_matrix/summary/cross_matrix.json",
        help="Path to cross_matrix.json (default: results/cross_matrix/summary/cross_matrix.json)",
    )
    args = ap.parse_args()

    p = Path(args.json)
    if not p.exists():
        print(f"[fatal] {p} not found. Run aggregate.py first.", file=sys.stderr)
        return 2

    with p.open() as f:
        data = json.load(f)

    rows = []
    for model, trains in data.items():
        # Skip top-level non-model entries (e.g. "_leakage") and
        # malformed branches.
        if model.startswith("_") or not isinstance(trains, dict):
            continue
        for train, tests in trains.items():
            if not isinstance(tests, dict):
                continue
            for test, cell in tests.items():
                if not isinstance(cell, dict):
                    continue
                per = cell.get("per_seed", {})
                if not isinstance(per, dict):
                    continue
                bad = sum(
                    1 for _, m in per.items()
                    if isinstance(m, dict) and m.get("n_nan_clamped", 0) > 0
                )
                if not bad:
                    continue
                total = len(per)
                total_nan = sum(
                    m.get("n_nan_clamped", 0)
                    for m in per.values() if isinstance(m, dict)
                )
                avg = total_nan / max(total, 1)
                rows.append((model, train, test, bad, total, avg))

    if not rows:
        print("[ok] no NaN-clamped cells found")
        return 0

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    for model, train, test, bad, total, avg in rows:
        print(
            f"{model:10s}  {train:>10s} -> {test:<10s}  "
            f"nan_seeds={bad}/{total}  avg_nan={avg:.0f}"
        )
    print(f"\n[summary] {len(rows)} cells with NaN predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
