"""Simultaneous CI plot (Tukey HSD) extended with committee PoE as 5th point.

Extends plot_simultaneous_ci.py to include the canonical PoE committee
(committee_per_seed_poe/{corpus}/per_seed_metrics.csv) alongside the
four base models. The committee is treated as a fifth "system" in the
Tukey HSD comparison.

CLI:
    python -m scripts.statistical_analysis.plot_simultaneous_ci_with_committee \\
        --corpus all --metric mcc \\
        --committee-csv results/committee_per_seed_poe/all/per_seed_metrics.csv \\
        --out results/statistical/all/figures/sim_ci_mcc_with_committee.pdf
"""
from __future__ import annotations

import argparse
import csv
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
    "committee_poe": "Comite PoE",
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


def _load_committee_per_seed(csv_path: Path, metric: str) -> list[float]:
    """Load committee_poe per-seed values for the given metric from CSV."""
    values = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["system"] == "committee_poe":
                values.append(float(row[metric]))
    if not values:
        raise ValueError(f"No committee_poe rows found in {csv_path}")
    return values


def render(corpus: str, metric: str, committee_csv: Path, out_path: Path,
           comparison_name: str = "dtkinase") -> None:
    # Load base 4 models via data_loader
    panel = data_loader.load_panel(corpus, MODELS, SEEDS)
    values, labels = [], []
    for model in MODELS:
        for sd in panel[model]:
            values.append(_per_seed_metric(sd, metric))
            labels.append(model)

    # Append committee PoE as 5th system
    committee_values = _load_committee_per_seed(committee_csv, metric)
    for v in committee_values:
        values.append(v)
        labels.append("committee_poe")

    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)

    tukey = pairwise_tukeyhsd(endog=values, groups=labels, alpha=0.05)
    fig = tukey.plot_simultaneous(comparison_name=comparison_name,
                                  ylabel="Sistema",
                                  xlabel=_METRIC_LABELS.get(metric, metric))
    ax = fig.axes[0]

    # Apply human-readable labels and remember tick positions per raw label
    yticklabels = ax.get_yticklabels()
    raw_to_ypos = {}
    new_labels = []
    for t in yticklabels:
        raw = t.get_text()
        new_labels.append(_MODEL_LABELS.get(raw, raw))
        raw_to_ypos[raw] = t.get_position()[1]
    ax.set_yticklabels(new_labels)

    # Highlight committee_poe in distinct green color (poolcol #1B813E from
    # CLAUDE.md TikZ palette, matching tab:comite-completo-3corpora).
    # Approach: overlay green CI segment + marker on top of statsmodels' red
    # default (since committee is significantly different from DT-Kinase
    # comparison_name, it would otherwise render as red like other "outside-CI"
    # groups; we want it visually distinct as the thesis result).
    committee_y = raw_to_ypos.get("committee_poe")
    committee_color = "#1B813E"
    if committee_y is not None:
        import numpy as _np
        committee_mean = float(_np.mean(committee_values))
        # Find CI bar segment for committee in LineCollections (segs at y=committee_y)
        for coll in ax.collections:
            for seg in coll.get_segments():
                if _np.isclose(seg[0, 1], committee_y) and _np.isclose(seg[1, 1], committee_y):
                    ax.plot(seg[:, 0], seg[:, 1], color=committee_color,
                            linewidth=2.5, solid_capstyle="round", zorder=10)
                    break
        # Overlay green marker at committee mean (high zorder)
        ax.plot([committee_mean], [committee_y], marker="o",
                markersize=9, markerfacecolor=committee_color,
                markeredgecolor=committee_color, zorder=11)
        # Vertical reference line at committee mean
        ax.axvline(committee_mean, linestyle=":", color=committee_color,
                   alpha=0.7, linewidth=1.2)
        # Recolor the y-tick label for committee row
        for t in ax.get_yticklabels():
            if t.get_text() == _MODEL_LABELS["committee_poe"]:
                t.set_color(committee_color)
                t.set_fontweight("bold")
    ax.set_title(
        f"Simultaneous CI (Tukey HSD, alpha=0.05) + Comite PoE\n"
        f"corpus={corpus} metric={_METRIC_LABELS.get(metric, metric)}; "
        f"n_seeds={len(SEEDS)}; comparison: {_MODEL_LABELS.get(comparison_name, comparison_name)}"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), format="png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[plot_simultaneous_ci_with_committee] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=("human", "non_human", "all"))
    ap.add_argument("--metric", default="mcc", choices=tuple(_METRIC_LABELS))
    ap.add_argument("--committee-csv", required=True, type=Path,
                    help="CSV with committee_poe per-seed metrics")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--comparison-name", default="dtkinase")
    args = ap.parse_args()
    render(args.corpus, args.metric, args.committee_csv, args.out, args.comparison_name)


if __name__ == "__main__":
    main()
