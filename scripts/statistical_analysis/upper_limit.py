"""Experimental-variability upper limit (Brown 2009 / Kramer 2012).

Estimates the maximum achievable classification performance given the
known noise in IC50 / Ki / Kd assays. The procedure perturbs the
per-sample pChEMBL values with Gaussian noise of magnitude
sigma_log10 = log10(2) approximately 0.301 (the "2-fold" assay noise
benchmark documented in Brown, Muchmore, Hajduk 2009 and re-quantified
in Kramer, Kalliokoski, Gedeck, Vulpetti 2012), re-thresholds at
pChEMBL >= 6, and computes the agreement (MCC) with the original
labels. The median of 10^4 trials gives the empirical performance
ceiling that any classifier could reach without violating assay
reproducibility.

CLI:
    python -m scripts.statistical_analysis.upper_limit \\
        --corpus non_human --sigma-log10 0.301 \\
        --threshold-pchembl 6.0 --n-trials 10000 \\
        --out results/statistical/non_human/upper_limit.json
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

from . import PRIMARY_METRICS, data_loader

DEFAULT_SIGMA_LOG10 = 0.301  # = log10(2), 2-fold IC50 assay noise (Brown 2009).
DEFAULT_THRESHOLD = 6.0      # pChEMBL >= 6 corresponds to IC50 <= 1 microM.
DEFAULT_N_TRIALS = 10_000


def _scoring_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    score: np.ndarray) -> dict[str, float]:
    if y_true.std() == 0 or y_pred.std() == 0:
        mcc = 0.0
    else:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    if len(set(y_true)) == 2:
        auroc = float(roc_auc_score(y_true, score))
        auprc = float(average_precision_score(y_true, score))
    else:
        auroc = 0.5
        auprc = float(np.mean(y_true))
    return {
        "mcc": mcc,
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def upper_limit_panel(corpus: str,
                     sigma_log10: float = DEFAULT_SIGMA_LOG10,
                     threshold_pchembl: float = DEFAULT_THRESHOLD,
                     n_trials: int = DEFAULT_N_TRIALS,
                     seed: int = 0) -> dict:
    """Simulate the assay-noise upper limit for one corpus.

    For each of n_trials draws p_t = p_real + N(0, sigma_log10), we
    re-threshold and measure agreement with the original labels.

    The "score" for AUROC/AUPRC purposes is the perturbed pChEMBL value
    itself (more pChEMBL means stronger predicted activity), which is
    the natural ranker the noisy assay would produce.
    """
    p_real = data_loader.load_test_pchembl(corpus)
    y_true = (p_real >= threshold_pchembl).astype(np.int64)
    n = len(p_real)

    rng = np.random.default_rng(seed)
    samples = {k: [] for k in PRIMARY_METRICS + ("accuracy", "precision", "recall")}

    for _ in range(n_trials):
        noise = rng.normal(0.0, sigma_log10, size=n)
        p_noisy = p_real + noise
        y_pred = (p_noisy >= threshold_pchembl).astype(np.int64)
        m = _scoring_metrics(y_true, y_pred, p_noisy)
        for k in samples:
            samples[k].append(m[k])

    summary = {}
    for k, vals in samples.items():
        arr = np.asarray(vals, dtype=np.float64)
        summary[k] = {
            "median": float(np.median(arr)),
            "ci_lo": float(np.percentile(arr, 2.5)),
            "ci_hi": float(np.percentile(arr, 97.5)),
            "mean": float(np.mean(arr)),
        }

    pos_rate = float(np.mean(y_true))
    return {
        "corpus": corpus,
        "n_test": int(n),
        "sigma_log10": float(sigma_log10),
        "threshold_pchembl": float(threshold_pchembl),
        "n_trials": int(n_trials),
        "positive_rate": pos_rate,
        "metrics": summary,
        "primary_metrics": list(PRIMARY_METRICS),
        "method": ("Brown 2009 / Kramer 2012: 2-fold IC50 noise injected "
                   "as N(0, log10(2)) on pChEMBL, re-thresholded at >=6."),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=("human", "non_human", "all"))
    ap.add_argument("--sigma-log10", type=float, default=DEFAULT_SIGMA_LOG10)
    ap.add_argument("--threshold-pchembl", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    result = upper_limit_panel(
        args.corpus, args.sigma_log10, args.threshold_pchembl,
        args.n_trials, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(result, fh, indent=2)

    m = result["metrics"]
    print(f"[upper_limit] corpus={args.corpus} n_test={result['n_test']} "
          f"sigma_log10={result['sigma_log10']:.3f} "
          f"pos_rate={result['positive_rate']:.3f}")
    for k in PRIMARY_METRICS:
        print(f"  {k:6s}: median={m[k]['median']:.4f} "
              f"CI95=[{m[k]['ci_lo']:.4f}, {m[k]['ci_hi']:.4f}]")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
