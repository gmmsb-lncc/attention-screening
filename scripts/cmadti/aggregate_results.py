#!/usr/bin/env python3
"""Aggregate canonical CMA-DTI metrics across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ("mcc", "auroc", "auprc", "f1", "precision", "recall", "accuracy", "loss")


def aggregate_runs(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("no CMA-DTI runs to aggregate")
    corpus = runs[0]["corpus"]
    if any(run["corpus"] != corpus for run in runs):
        raise ValueError("mixed corpora")
    aggregate = {}
    for split in ("train", "validation", "test"):
        aggregate[split] = {}
        for metric in METRICS:
            values = np.asarray([run[split][metric] for run in runs], dtype=float)
            aggregate[split][metric] = {
                "mean": float(values.mean()), "std": float(values.std(ddof=0))
            }
        thresholds = np.asarray([run[split]["threshold"] for run in runs], dtype=float)
        aggregate[split]["threshold"] = {
            "mean": float(thresholds.mean()), "std": float(thresholds.std(ddof=0))
        }
    return {
        "model": "CMA-DTI", "corpus": corpus, "split": "universal_scaffold",
        "seeds": [int(run["seed"]) for run in runs], "n_seeds": len(runs),
        "methodology": {
            "model_selection": "validation AUROC (upstream CMA-DTI criterion)",
            "threshold_optimization": "validation MCC-optimal (no test leakage)",
            "dispersion": "population standard deviation across canonical seeds",
        },
        "per_seed": runs, "aggregate": aggregate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("human", "non_human", "all"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    parser.add_argument("--results-root", type=Path, default=Path("results/cmadti"))
    args = parser.parse_args()
    paths = [args.results_root / f"cmadti_{args.corpus}_seed{seed}" / "cmadti_results.json"
             for seed in args.seeds]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing result(s):\n  " + "\n  ".join(map(str, missing)))
    result = aggregate_runs([json.loads(path.read_text()) for path in paths])
    output = args.results_root / f"cmadti_{args.corpus}_aggregate.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
