#!/usr/bin/env python3
"""Aggregate canonical ChemGLaM per-seed benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("mcc", "auroc", "auprc", "f1", "precision", "recall", "accuracy")
SPLITS = ("train", "validation", "test")


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("no ChemGLaM runs to aggregate")
    corpus = runs[0]["corpus"]
    if any(run.get("corpus") != corpus for run in runs):
        raise ValueError("cannot aggregate runs from different corpora")

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for split in SPLITS:
        available = [run[split] for run in runs if split in run]
        if not available:
            continue
        aggregate[split] = {}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in available], dtype=float)
            aggregate[split][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }
        thresholds = np.asarray([row["threshold"] for row in available], dtype=float)
        aggregate[split]["threshold"] = {
            "mean": float(thresholds.mean()),
            "std": float(thresholds.std(ddof=0)),
        }

    audits = [run.get("methodology_audit") for run in runs]
    if any(audit is None for audit in audits):
        raise ValueError(
            "missing ChemGLaM methodology audit provenance; rerun evaluation "
            "before aggregating"
        )
    if any(audit != audits[0] for audit in audits[1:]):
        raise ValueError("mixed ChemGLaM methodology audit provenance")
    split_audits = [run.get("data_split_audit") for run in runs]
    if any(audit is None for audit in split_audits):
        raise ValueError("missing ChemGLaM canonical split audit provenance")
    if any(audit != split_audits[0] for audit in split_audits[1:]):
        raise ValueError("mixed ChemGLaM canonical split audit provenance")

    return {
        "model": "ChemGLaM",
        "corpus": corpus,
        "split": "universal_scaffold",
        "seeds": [int(run["seed"]) for run in runs],
        "n_seeds": len(runs),
        "methodology": {
            "model_selection": "minimum validation loss (upstream ChemGLaM criterion)",
            "threshold_optimization": "validation MCC-optimal (no test leakage)",
            "dispersion": "population standard deviation across canonical seeds",
        },
        "methodology_audit": audits[0],
        "data_split_audit": split_audits[0],
        "per_seed": runs,
        "aggregate": aggregate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("all", "human", "non_human"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    parser.add_argument("--results-root", type=Path, default=Path("results/chemglam"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = []
    missing = []
    for seed in args.seeds:
        path = args.results_root / f"chemglam_{args.corpus}_seed{seed}" / "chemglam_results.json"
        if not path.exists():
            missing.append(path)
            continue
        runs.append(json.loads(path.read_text()))
    if missing:
        raise FileNotFoundError("missing per-seed result(s):\n  " + "\n  ".join(map(str, missing)))

    result = aggregate_runs(runs)
    output = args.output or args.results_root / f"chemglam_{args.corpus}_aggregate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
