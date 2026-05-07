"""Pairwise MCSim heatmap (effect size + significance), per (corpus, metric).

A 4 x 4 matrix where:
- Cell color encodes the paired Hedges' g (J(4) primary), with diverging
  colormap centered on g = 0.
- Cell annotation: numeric g_paired with significance stars from the
  Tukey HSD adjusted p-value:
    "***" p_adj < 0.001
    "**"  p_adj < 0.01
    "*"   p_adj < 0.05
    blank otherwise.
- Diagonal blanked (model versus itself).

Inputs: outputs of effect_size.py and anova_tukey.py for the same
(corpus, metric).

CLI:
    python -m scripts.statistical_analysis.plot_mcsim_heatmap \\
        --corpus non_human --metric mcc \\
        --effect-source results/statistical/non_human/effect_size.json \\
        --pvalue-source results/statistical/non_human/anova_tukey.json \\
        --out results/statistical/non_human/figures/mcsim_mcc.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import MODELS, PRIMARY_METRICS

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


def _stars(p_adj: float) -> str:
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    if p_adj < 0.05:
        return "*"
    return ""


def _load_g_matrix(effect_path: Path, metric: str) -> np.ndarray:
    with effect_path.open() as fh:
        eff = json.load(fh)
    pairs = eff["by_metric"][metric]["pairs"]
    n = len(MODELS)
    G = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(MODELS):
        for j, b in enumerate(MODELS):
            if i == j:
                G[i, j] = 0.0
            else:
                key = f"{a}__vs__{b}"
                G[i, j] = pairs[key]["paired"]["g_paired"]
    return G


def _load_padj_matrix(pvalue_path: Path, metric: str) -> np.ndarray:
    with pvalue_path.open() as fh:
        av = json.load(fh)
    rows = av["by_metric"][metric]["tukey"]
    n = len(MODELS)
    P = np.full((n, n), np.nan, dtype=np.float64)
    idx = {m: i for i, m in enumerate(MODELS)}
    for r in rows:
        i, j = idx[r["group1"]], idx[r["group2"]]
        P[i, j] = r["p_adj"]
        P[j, i] = r["p_adj"]
    return P


def render(corpus: str, metric: str, effect_path: Path,
          pvalue_path: Path, out_path: Path) -> None:
    G = _load_g_matrix(effect_path, metric)
    P = _load_padj_matrix(pvalue_path, metric)
    n = len(MODELS)
    pretty = [_MODEL_LABELS[m] for m in MODELS]

    vmax = float(np.nanmax(np.abs(G)))
    vmax = max(vmax, 0.1)  # ensure non-zero scale
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(G, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center",
                        color="black", fontsize=10)
                continue
            stars = _stars(P[i, j]) if not np.isnan(P[i, j]) else ""
            txt = f"{G[i, j]:+.2f}\n{stars}" if stars else f"{G[i, j]:+.2f}"
            color = "white" if abs(G[i, j]) > 0.6 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(pretty, rotation=30, ha="right")
    ax.set_yticklabels(pretty)
    ax.set_xlabel("Model B (column)")
    ax.set_ylabel("Model A (row)")
    ax.set_title(
        f"MCSim heatmap: paired Hedges' g (A vs B), {_METRIC_LABELS.get(metric, metric)}\n"
        f"corpus={corpus}; * p_adj<0.05  ** p_adj<0.01  *** p_adj<0.001 (Tukey HSD)"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Hedges' g (paired, J(4))")
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
    ap.add_argument("--effect-source", type=Path, required=True)
    ap.add_argument("--pvalue-source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    render(args.corpus, args.metric, args.effect_source,
          args.pvalue_source, args.out)
    print(f"[plot_mcsim_heatmap] {args.out}")


if __name__ == "__main__":
    main()
