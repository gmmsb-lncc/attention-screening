"""Statistical comparison toolkit (Ash/Wognum 2025-aligned).

This package implements the statistical protocol declared in
`docs/01-methodology/statistical_protocol.md`. Components:

- `data_loader`        — shared I/O over raw_predictions.npz + TSVs.
- `null_model`         — majority-class lower limit per corpus.
- `upper_limit`        — assay-variability upper limit (Brown 2009 / Kramer 2012).
- `effect_size`        — Hedges' g paired (J(4)) and unpaired (J(8)).
- `posthoc_classification` — precision@recall, recall@precision, TNR@recall.
- `tost_sensitivity`   — TOST equivalence over multiple bands.
- `anova_tukey`        — RM-ANOVA + Bonferroni inter-metric + Tukey HSD.
- `plot_simultaneous_ci` — figures replacing bar charts.
- `plot_mcsim_heatmap` — pairwise effect-size heatmap with significance stars.
- `aggregate_panel`    — JSON + LaTeX + reporting checklist compliance.

Three deviations from Ash/Wognum 2025 are preserved (see protocol §2):
D1 single scaffold split + 5 seeds, D2 paired bootstrap as primary inference,
D3 TOST band 0.05 MCC SESOI-anchored.
"""
from __future__ import annotations

MODELS = ("dtkinase", "drugban", "graphban", "conplex")
CORPORA = ("human", "non_human", "all")
SEEDS = (42, 123, 456, 789, 1024)
PRIMARY_METRICS = ("mcc", "auroc", "f1", "auprc")
DEFAULT_BOOTSTRAP_B = 10_000
PRIMARY_TOST_BAND = 0.05  # MCC, SESOI operational anchor (D3 primary)
HEDGES_NU_PAIRED = 4      # n - 1 for n=5 paired design (J(4) primary)
HEDGES_NU_UNPAIRED = 8    # 2(n - 1) for n=5 unpaired (J(8) cross-check)


def hedges_J(nu: int) -> float:
    """Small-sample correction factor for Cohen's d.

    Hedges (1981), J(nu) = 1 - 3 / (4*nu - 1). For paired design with
    n seeds, nu = n - 1 (Cohen's d_z). For unpaired comparison of two
    independent groups of size n each, nu = 2(n - 1).
    """
    if nu <= 0:
        raise ValueError(f"nu must be positive, got {nu}")
    return 1.0 - 3.0 / (4.0 * nu - 1.0)
