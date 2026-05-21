"""Pairwise mcsim heatmap (Hedges' g + Tukey HSD stars) extended with
committee PoE as 5th system.

5x5 matrix where:
- Cell color encodes paired Hedges' g (J(4)) with diverging colormap centered at 0.
- Cell annotation: numeric g_paired with significance stars from Tukey HSD p_adj:
    "***" p_adj < 0.001  ** p_adj < 0.01  * p_adj < 0.05  blank otherwise.
- Diagonal blanked. Committee row/column label rendered bold green.

g and Tukey HSD computed on-the-fly from per-seed MCC of 4 base models (via
data_loader) + committee_poe (via CSV), to ensure consistency across all 5
systems (avoids reusing precomputed 4x4 effect_size.json / anova_tukey.json
which exclude committee).

CLI:
    python -m scripts.statistical_analysis.plot_mcsim_heatmap_with_committee \\
        --corpus all --metric mcc \\
        --committee-csv results/committee_per_seed_poe/all/per_seed_metrics.csv \\
        --out figures/statistical/mcsim_mcc_all_with_committee.pdf
"""
from __future__ import annotations

import argparse
import csv
import math
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
_COMMITTEE_COLOR = "#1B813E"  # poolcol from CLAUDE.md TikZ palette


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
    values = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["system"] == "committee_poe":
                values.append(float(row[metric]))
    if not values:
        raise ValueError(f"No committee_poe rows found in {csv_path}")
    return values


def _hedges_g_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Hedges' g with J(n-1) small-sample correction.

    g = (mean(a-b) / sd(a-b)) * J(df), df = n-1.
    """
    n = len(a)
    if n != len(b) or n < 2:
        return float("nan")
    diffs = a - b
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    if sd_diff == 0:
        return 0.0 if mean_diff == 0 else float("inf") * math.copysign(1, mean_diff)
    d = mean_diff / sd_diff
    # Hedges' J correction approximation (cap3 eq:hedges-correction)
    df = n - 1
    J = 1.0 - 3.0 / (4.0 * df - 1.0)
    return d * J


def _stars(p_adj: float) -> str:
    if np.isnan(p_adj):
        return ""
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    if p_adj < 0.05:
        return "*"
    return ""


def render(corpus: str, metric: str, committee_csv: Path, out_path: Path) -> None:
    # Load per-seed metric for 4 base models + committee
    panel = data_loader.load_panel(corpus, MODELS, SEEDS)
    per_seed: dict[str, list[float]] = {}
    for m in MODELS:
        per_seed[m] = [_per_seed_metric(sd, metric) for sd in panel[m]]
    per_seed["committee_poe"] = _load_committee_per_seed(committee_csv, metric)

    systems = list(MODELS) + ["committee_poe"]
    n = len(systems)
    pretty = [_MODEL_LABELS[s] for s in systems]

    # Build 5x5 paired Hedges' g matrix
    G = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(systems):
        for j, b in enumerate(systems):
            if i == j:
                G[i, j] = 0.0
            else:
                G[i, j] = _hedges_g_paired(
                    np.asarray(per_seed[a]),
                    np.asarray(per_seed[b]),
                )

    # Build 5x5 Tukey HSD p_adj matrix
    values = []
    labels = []
    for s in systems:
        for v in per_seed[s]:
            values.append(v)
            labels.append(s)
    tukey = pairwise_tukeyhsd(np.asarray(values), np.asarray(labels), alpha=0.05)
    P = np.full((n, n), np.nan, dtype=np.float64)
    idx = {s: i for i, s in enumerate(systems)}
    for r in tukey._results_table.data[1:]:
        # statsmodels row: [group1, group2, meandiff, p_adj, lower, upper, reject]
        g1, g2, _, p_adj = r[0], r[1], r[2], r[3]
        i, j = idx[g1], idx[g2]
        P[i, j] = float(p_adj)
        P[j, i] = float(p_adj)

    # Plot heatmap
    vmax = float(np.nanmax(np.abs(G)))
    vmax = max(vmax, 0.1)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(G, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center",
                        color="black", fontsize=10)
                continue
            stars = _stars(P[i, j])
            txt = f"{G[i, j]:+.2f}\n{stars}" if stars else f"{G[i, j]:+.2f}"
            color = "white" if abs(G[i, j]) > 0.6 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(pretty, rotation=30, ha="right")
    ax.set_yticklabels(pretty)

    # Highlight committee row/column label in green + bold
    committee_idx = systems.index("committee_poe")
    for t in ax.get_xticklabels():
        if t.get_text() == _MODEL_LABELS["committee_poe"]:
            t.set_color(_COMMITTEE_COLOR)
            t.set_fontweight("bold")
    for t in ax.get_yticklabels():
        if t.get_text() == _MODEL_LABELS["committee_poe"]:
            t.set_color(_COMMITTEE_COLOR)
            t.set_fontweight("bold")
    # Bold green border on committee row + column for visual highlight
    for k in range(n):
        # Cells (committee_idx, k) and (k, committee_idx)
        for ci, cj in [(committee_idx, k), (k, committee_idx)]:
            if ci == cj:
                continue
            ax.add_patch(plt.Rectangle((cj - 0.5, ci - 0.5), 1, 1,
                                       fill=False, edgecolor=_COMMITTEE_COLOR,
                                       linewidth=1.5, zorder=3))

    ax.set_xlabel("Sistema B (coluna)")
    ax.set_ylabel("Sistema A (linha)")
    ax.set_title(
        f"Heatmap pareado de Hedges' g (J(4)), {_METRIC_LABELS.get(metric, metric)}\n"
        f"corpus={corpus}; comite PoE = 5o sistema; "
        f"* p_adj<0.05  ** <0.01  *** <0.001 (Tukey HSD)"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Hedges' g pareado (J(4))")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), format="png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[plot_mcsim_heatmap_with_committee] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=("human", "non_human", "all"))
    ap.add_argument("--metric", default="mcc", choices=PRIMARY_METRICS)
    ap.add_argument("--committee-csv", required=True, type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    render(args.corpus, args.metric, args.committee_csv, args.out)


if __name__ == "__main__":
    main()
