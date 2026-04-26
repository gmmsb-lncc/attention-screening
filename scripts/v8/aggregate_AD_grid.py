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
    p.add_argument("--root", action="append", required=True, type=Path,
                   help="Grid root dir. Pass multiple times to aggregate "
                        "across corpora (e.g. --root results/grid_AD_human "
                        "--root results/grid_AD_non_human).")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional output CSV path (default stdout only).")
    return p.parse_args()


def _corpus_from_root(root: Path) -> str:
    """Infer corpus name from path like .../grid_AD_<corpus>."""
    name = root.name
    if name.startswith("grid_AD_"):
        return name.removeprefix("grid_AD_")
    return name


def collect_rows(root: Path) -> list[dict]:
    corpus = _corpus_from_root(root)
    rows = []
    for cell_dir in sorted(root.iterdir()):
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
            "corpus": corpus,
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
    return rows


def print_corpus_table(corpus: str, rows: list[dict]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["mcc"], reverse=True)
    print()
    print(f"=== corpus: {corpus} (n_cells={len(rows)}) ===")
    print(f"{'lr_lig':>7} {'layers':>7} {'mcc':>7} {'mcc_std':>7}  {'f1':>6}  {'auc':>6}  {'prec':>6}  {'rec':>6}  {'seeds':>5}")
    print("-" * 76)
    fmt = "{:>7} {:>7} {:>7.4f} {:>7.4f}  {:>6.4f}  {:>6.4f}  {:>6.4f}  {:>6.4f}  {:>5}"
    for r in rows:
        print(fmt.format(
            r["lr_mult_lig"], r["layers_lig"], r["mcc"], r["mcc_std"],
            r["f1"], r["auc"], r["precision"], r["recall"], r["n_seeds"],
        ))
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
    print(f"  Pareto (max mcc, min std):", end=" ")
    print(", ".join(f"lr={p['lr_mult_lig']}/L={p['layers_lig']} mcc={p['mcc']:.4f}±{p['mcc_std']:.4f}"
                    for p in pareto))


def main():
    args = parse_args()
    all_rows: list[dict] = []
    for root in args.root:
        rows = collect_rows(root)
        if not rows:
            print(f"[aggregate] no results under {root}")
            continue
        corpus = _corpus_from_root(root)
        print_corpus_table(corpus, rows)
        all_rows.extend(rows)

    if all_rows and len(args.root) > 1:
        # Cross-corpus mean MCC per cell (lr, L) across corpora that have it.
        from collections import defaultdict
        by_cell: dict[tuple, list[float]] = defaultdict(list)
        by_cell_std: dict[tuple, list[float]] = defaultdict(list)
        for r in all_rows:
            by_cell[(r["lr_mult_lig"], r["layers_lig"])].append(r["mcc"])
            by_cell_std[(r["lr_mult_lig"], r["layers_lig"])].append(r["mcc_std"])
        print()
        print(f"=== cross-corpus mean per cell ===")
        print(f"{'lr_lig':>7} {'layers':>7} {'mean_mcc':>9} {'mean_std':>9}  {'n_corpora':>9}")
        print("-" * 50)
        agg = sorted(by_cell.items(), key=lambda kv: sum(kv[1])/len(kv[1]), reverse=True)
        for (lr, L), mccs in agg:
            stds = by_cell_std[(lr, L)]
            print(f"{lr:>7} {L:>7} {sum(mccs)/len(mccs):>9.4f} {sum(stds)/len(stds):>9.4f}  {len(mccs):>9}")

    if args.out and all_rows:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n[aggregate] CSV → {args.out}")


if __name__ == "__main__":
    main()
