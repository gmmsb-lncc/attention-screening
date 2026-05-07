"""Simultaneous CI plot (Tukey HSD), per (corpus, metric).

Replaces the bar chart with mean +/- 1 sigma from the thesis figures
(slides 22-24 / Chapter 5 figures) with the simultaneous confidence
interval plot recommended by Ash/Wognum 2025 G4. Non-overlapping
intervals indicate FWER-controlled significance under Tukey HSD.

CLI:
    python -m scripts.statistical_analysis.plot_simultaneous_ci \\
        --corpus non_human --metric mcc \\
        --out results/statistical/non_human/figures/sim_ci_mcc.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from . import MODELS, PRIMARY_METRICS, SEEDS, data_loader

_MODEL_LABELS = {
    "dtkinase": "DT-Kinase",
    "drugban":  "DrugBAN",
    "graphban": "GraphBAN",
    "conplex":  "ConPLex",
}
_METRIC_LABELS = {
    "mcc": "MCC", "auroc": "AUROC", "auprc": "AUPRC", "f1": "F1",
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
}


def _per_seed_metric(seed_data: dict, metric: str) -> float:
    y = seed_data["y_true"]
    p = seed_data["y_prob"]
    pred = (p >= seed_data["threshold"]).astype(np.int64)
    if metric == "mcc":
        return float(matthews_corrcoef(y, pred)) if y.std() > 0 else 0.0
    if metric == "auroc":
        return float(roc_auc_score(y, p)) if len(set(y)) == 2 else 0.5
    if metric == "auprc":
        return float(average_precision_score(y, p)) if len(set(y)) == 2 else float(np.mean(y))
    if metric == "f1":
        return float(f1_score(y, pred, zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    if metric == "precision":
        return float(precision_score(y, pred, zero_division=0))
    if metric == "recall":
        return float(recall_score(y, pred, zero_division=0))
    raise ValueError(metric)


def render(corpus: str, metric: str, out_path: Path,
          comparison_name: str = "dtkinase") -> None:
    panel = data_loader.load_panel(corpus, MODELS, SEEDS)
    values, labels = [], []
    for model in MODELS:
        for sd in panel[model]:
            values.append(_per_seed_metric(sd, metric))
            labels.append(model)
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)

    tukey = pairwise_tukeyhsd(endog=values, groups=labels, alpha=0.05)
    fig = tukey.plot_simultaneous(comparison_name=comparison_name,
                                  ylabel="Model",
                                  xlabel=_METRIC_LABELS.get(metric, metric))
    ax = fig.axes[0]

    # Apply human-readable labels.
    yticklabels = ax.get_yticklabels()
    new_labels = []
    for t in yticklabels:
        raw = t.get_text()
        new_labels.append(_MODEL_LABELS.get(raw, raw))
    ax.set_yticklabels(new_labels)
    ax.set_title(
        f"Simultaneous CI (Tukey HSD, alpha=0.05)\n"
        f"corpus={corpus} metric={_METRIC_LABELS.get(metric, metric)}; "
        f"n_seeds={len(SEEDS)}; comparison: {_MODEL_LABELS.get(comparison_name, comparison_name)}"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), format="png",
               bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=("human", "non_human", "all"))
    ap.add_argument("--metric", default="mcc", choices=PRIMARY_METRICS)
    ap.add_argument("--comparison-name", default="dtkinase",
                    help="Model used as the comparison anchor (default: dtkinase).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    render(args.corpus, args.metric, args.out, args.comparison_name)
    print(f"[plot_simultaneous_ci] {args.out}")


if __name__ == "__main__":
    main()
