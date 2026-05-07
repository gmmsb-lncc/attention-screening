"""Null-model lower limit (majority-class baseline).

Establishes the lower performance limit recommended by Ash/Wognum 2025
(Guideline 3.3.1.1). Predicts the majority class learned on the test
labels themselves; bootstrap CI quantifies sampling variability.

Sanity check: MCC of a constant predictor must equal 0 by definition
(covariance of predictions and labels vanishes).

CLI:
    python -m scripts.statistical_analysis.null_model \\
        --corpus non_human --n-bootstrap 10000 \\
        --out results/statistical/non_human/null_model.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

from . import DEFAULT_BOOTSTRAP_B, PRIMARY_METRICS, data_loader


def _metrics_constant(y_true: np.ndarray, majority: int) -> dict[str, float]:
    """Metrics for a constant predictor."""
    n = len(y_true)
    y_pred = np.full(n, majority, dtype=np.int64)
    # AUROC and AUPRC are degenerate with a constant score; use prevalence
    # for AUPRC (which equals positive prevalence at any threshold for a
    # constant ranker), and 0.5 for AUROC.
    prev = float(np.mean(y_true == 1))
    return {
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if y_true.std() > 0 else 0.0,
        "auroc": 0.5,
        "auprc": prev,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def null_model_panel(corpus: str, n_bootstrap: int = DEFAULT_BOOTSTRAP_B,
                    seed: int = 0) -> dict:
    """Compute null-model performance for one corpus.

    Loads y_true from any model's seed_42 (alignment is asserted), picks
    majority class, computes point metrics, then bootstraps over test
    samples for IC95.
    """
    ref = data_loader.load_predictions("dtkinase", corpus, 42, "test")
    y_true = ref["y_true"]
    n = len(y_true)
    counts = np.bincount(y_true, minlength=2)
    majority = int(np.argmax(counts))

    point = _metrics_constant(y_true, majority)

    rng = np.random.default_rng(seed)
    boot_samples = {k: [] for k in point}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ys = y_true[idx]
        m = _metrics_constant(ys, majority)
        for k in point:
            boot_samples[k].append(m[k])

    summary = {}
    for k, samples in boot_samples.items():
        arr = np.asarray(samples, dtype=np.float64)
        summary[k] = {
            "point": float(point[k]),
            "median": float(np.median(arr)),
            "ci_lo": float(np.percentile(arr, 2.5)),
            "ci_hi": float(np.percentile(arr, 97.5)),
        }

    return {
        "corpus": corpus,
        "n_test": int(n),
        "n_majority": int(counts[majority]),
        "n_minority": int(n - counts[majority]),
        "majority_class": int(majority),
        "n_bootstrap": int(n_bootstrap),
        "metrics": summary,
        "primary_metrics": list(PRIMARY_METRICS),
        "note": "MCC of a constant predictor is 0 by construction; "
                "bootstrap captures finite-sample noise on the other metrics.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=("human", "non_human", "all"))
    ap.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAP_B)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    result = null_model_panel(args.corpus, args.n_bootstrap, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(result, fh, indent=2)

    m = result["metrics"]
    print(f"[null_model] corpus={args.corpus} "
          f"n_test={result['n_test']} majority={result['majority_class']} "
          f"({result['n_majority']}/{result['n_test']})")
    for k in PRIMARY_METRICS:
        print(f"  {k:6s}: median={m[k]['median']:+.4f} "
              f"CI95=[{m[k]['ci_lo']:+.4f}, {m[k]['ci_hi']:+.4f}]")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
