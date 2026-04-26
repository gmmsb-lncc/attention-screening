"""Aggregate A+D grid sweep results into a CSV + markdown table.

Walks results/grid_AD/<lrX_LY>/test/benchmark_comparison.json files and
prints a summary sorted by test_mcc descending.

Usage:
  python3 scripts/v8/aggregate_AD_grid.py --root results/grid_AD
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CELL_PAT = re.compile(r"lr(?P<lr>[\d.]+)_L(?P<L>\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Optional output CSV path (default stdout only).")
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for cell_dir in sorted(args.root.iterdir()):
        if not cell_dir.is_dir():
            continue
        m = CELL_PAT.match(cell_dir.name)
        if not m:
            continue
        bc = cell_dir / "test" / "benchmark_comparison.json"
        if not bc.exists():
            continue
        with open(bc) as f:
            data = json.load(f)
        res = data["results"].get("level4_cnn_mlp", {})
        rows.append({
            "lr_mult_lig": float(m.group("lr")),
            "layers_lig": int(m.group("L")),
            "mcc": float(res.get("mcc", 0)),
            "mcc_std": float(res.get("mcc_std", 0)),
            "f1": float(res.get("f1", 0)),
            "f1_std": float(res.get("f1_std", 0)),
            "auc": float(res.get("auc", 0)),
            "auc_std": float(res.get("auc_std", 0)),
            "precision": float(res.get("precision", 0)),
            "recall": float(res.get("recall", 0)),
            "n_seeds": len(data.get("metadata", {}).get("seeds", [])),
            "elapsed_s": float(data.get("metadata", {}).get("elapsed_seconds", 0)),
        })

    if not rows:
        print(f"[aggregate] no results under {args.root}")
        return

    rows.sort(key=lambda r: r["mcc"], reverse=True)

    header = ("lr_mult_lig", "layers_lig", "mcc", "mcc_std",
              "f1", "auc", "precision", "recall", "n_seeds")
    fmt = "{:>11} {:>10} {:>7.4f} {:>7.4f}  {:>6.4f}  {:>6.4f}  {:>9.4f}  {:>6.4f}  {:>7}"
    print()
    print(f"{'lr_lig':>11} {'layers':>10} {'mcc':>7} {'mcc_std':>7}  {'f1':>6}  {'auc':>6}  {'precision':>9}  {'recall':>6}  {'n_seeds':>7}")
    print("-" * 90)
    for r in rows:
        print(fmt.format(
            r["lr_mult_lig"], r["layers_lig"], r["mcc"], r["mcc_std"],
            r["f1"], r["auc"], r["precision"], r["recall"], r["n_seeds"],
        ))
    print()

    # Highlight Pareto front: cells not dominated on (mcc, -mcc_std).
    pareto = []
    for r in rows:
        dominated = False
        for s in rows:
            if (s["mcc"] >= r["mcc"]) and (s["mcc_std"] <= r["mcc_std"]) and \
               (s["mcc"] > r["mcc"] or s["mcc_std"] < r["mcc_std"]):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    print("Pareto front (max mcc, min std):")
    for r in pareto:
        print(f"  lr={r['lr_mult_lig']:>4} L={r['layers_lig']}  mcc={r['mcc']:.4f} ± {r['mcc_std']:.4f}")
    print()

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[aggregate] CSV → {args.out}")


if __name__ == "__main__":
    main()
