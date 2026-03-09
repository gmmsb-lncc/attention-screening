"""Benchmark visualization: grouped bar, radar, heatmap, MCC ranking, strip.

All plot functions follow the same signature and return the saved file path
(or ``None`` on failure).  ``generate_all`` orchestrates them.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from benchmark.config import (
    LEVEL_COLORS,
    LEVEL_LABELS,
    METRICS_ORDER,
    BenchmarkConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_ORDER = [
    "level1a_fp_knn",
    "level1a_fp_mlp",
    "level1b_ligmean_knn",
    "level1b_ligmean_mlp",
    "level1c_ligattn_knn",
    "level1c_ligattn_mlp",
    "level2_meanpool_knn",
    "level2_meanpool_mlp",
    "level3_attnpool_knn",
    "level3_attnpool_mlp",
    "level4_crossatt_knn",
    "level4_crossatt_mlp",
    "level5_da_knn",
    "level5_da_mlp",
    "level5b_da_knn",
    "level5b_da_mlp",
]


def _available_models(aggregated: Dict) -> List[str]:
    """Return model keys that have at least one non-None metric."""
    return [
        k
        for k in _MODEL_ORDER
        if k in aggregated and any(aggregated[k].get(m) is not None for m in METRICS_ORDER)
    ]


def _save_and_close(fig: plt.Figure, path: str) -> str:
    """Save a figure, close it, and return the path."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Plot: grouped bar chart
# ---------------------------------------------------------------------------

def plot_grouped_bar_chart(
    aggregated: Dict[str, Dict],
    config: BenchmarkConfig,
) -> Optional[str]:
    """Grouped bar chart: metrics on x-axis, bars grouped by model."""
    models = _available_models(aggregated)
    if not models:
        return None

    n_metrics = len(METRICS_ORDER)
    n_models = len(models)
    bar_w = 0.8 / n_models
    x_pos = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(12, 6))
    all_vals: list[float] = []

    for i, mk in enumerate(models):
        row = aggregated[mk]
        vals = [row.get(m) if row.get(m) is not None else np.nan for m in METRICS_ORDER]
        stds = [row.get(f"{m}_std") or 0.0 for m in METRICS_ORDER]
        all_vals.extend(v for v in vals if not np.isnan(v))
        offset = (i - n_models / 2 + 0.5) * bar_w

        bars = ax.bar(
            x_pos + offset,
            [v if not np.isnan(v) else 0 for v in vals],
            bar_w * 0.9,
            yerr=stds if any(s > 0 for s in stds) else None,
            capsize=3,
            label=LEVEL_LABELS[mk],
            color=LEVEL_COLORS[mk],
            edgecolor="white",
            linewidth=0.5,
        )

        for bar, val in zip(bars, vals):
            if not np.isnan(val) and val != 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    max(bar.get_height(), 0) + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.upper() for m in METRICS_ORDER], fontsize=11)
    y_min = min(all_vals) if all_vals else 0
    ax.set_ylim(min(0, y_min - 0.05), 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Model Comparison — {config.dataset} / ESM-2 {config.embedding} / Scaffold Split",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return _save_and_close(fig, os.path.join(config.resolved_output_dir, "benchmark_grouped_bar.png"))


# ---------------------------------------------------------------------------
# Plot: radar chart
# ---------------------------------------------------------------------------

def plot_radar_chart(
    aggregated: Dict[str, Dict],
    config: BenchmarkConfig,
) -> Optional[str]:
    """Radar (spider) chart comparing all models across metrics."""
    models = _available_models(aggregated)
    if not models:
        return None

    n_metrics = len(METRICS_ORDER)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    all_radar: list[float] = []

    for mk in models:
        row = aggregated[mk]
        vals = [row.get(m) if row.get(m) is not None else 0.0 for m in METRICS_ORDER]
        all_radar.extend(vals)
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=LEVEL_LABELS[mk], color=LEVEL_COLORS[mk], markersize=5)
        ax.fill(angles, vals, alpha=0.08, color=LEVEL_COLORS[mk])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in METRICS_ORDER], fontsize=11)
    r_min = min(all_radar) if all_radar else 0
    ax.set_ylim(min(0, r_min - 0.05), 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, alpha=0.6)
    ax.set_title(
        f"Radar — {config.dataset} / ESM-2 {config.embedding} / Scaffold Split",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9, framealpha=0.9)
    fig.tight_layout()

    return _save_and_close(fig, os.path.join(config.resolved_output_dir, "benchmark_radar.png"))


# ---------------------------------------------------------------------------
# Plot: heatmap
# ---------------------------------------------------------------------------

def plot_metric_heatmap(
    aggregated: Dict[str, Dict],
    config: BenchmarkConfig,
) -> Optional[str]:
    """Heatmap: models (rows) vs metrics (columns)."""
    models = _available_models(aggregated)
    if not models:
        return None

    n_models = len(models)
    n_metrics = len(METRICS_ORDER)
    matrix = np.full((n_models, n_metrics), np.nan)

    for i, mk in enumerate(models):
        row = aggregated[mk]
        for j, m in enumerate(METRICS_ORDER):
            val = row.get(m)
            if val is not None:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(10, max(3, n_models * 0.8 + 1.5)))
    im = ax.imshow(matrix, cmap=plt.cm.RdYlGn, aspect="auto", vmin=0, vmax=1)

    for i in range(n_models):
        for j in range(n_metrics):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=11, color="gray")
            else:
                std = aggregated[models[i]].get(f"{METRICS_ORDER[j]}_std")
                txt = f"{val:.3f}"
                if std and std > 0:
                    txt += f"\n\u00b1{std:.3f}"
                ax.text(
                    j, i, txt,
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if val < 0.4 else "black",
                )

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels([m.upper() for m in METRICS_ORDER], fontsize=11)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([LEVEL_LABELS[mk] for mk in models], fontsize=11)
    ax.set_title(
        f"Performance Heatmap — {config.dataset} / ESM-2 {config.embedding} / Scaffold Split",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Score", fontsize=11)
    fig.tight_layout()

    return _save_and_close(fig, os.path.join(config.resolved_output_dir, "benchmark_heatmap.png"))


# ---------------------------------------------------------------------------
# Plot: MCC ranking
# ---------------------------------------------------------------------------

def plot_mcc_ranking(
    aggregated: Dict[str, Dict],
    config: BenchmarkConfig,
) -> Optional[str]:
    """Horizontal bar chart ranking models by MCC."""
    models = _available_models(aggregated)
    if not models:
        return None

    items = [
        (mk, aggregated[mk].get("mcc", 0.0), aggregated[mk].get("mcc_std", 0.0) or 0.0)
        for mk in models
        if aggregated[mk].get("mcc") is not None
    ]
    if not items:
        return None

    items.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(9, max(3, len(items) * 0.8 + 1)))
    y = np.arange(len(items))
    mccs = [x[1] for x in items]
    stds = [x[2] for x in items]
    colors = [LEVEL_COLORS[x[0]] for x in items]
    labels = [LEVEL_LABELS[x[0]] for x in items]

    bars = ax.barh(
        y, mccs,
        xerr=stds if any(s > 0 for s in stds) else None,
        capsize=4, color=colors, edgecolor="white", linewidth=0.5, height=0.6,
    )

    for bar, mcc_val, std_val in zip(bars, mccs, stds):
        txt = f"{mcc_val:.3f}"
        if std_val > 0:
            txt += f" \u00b1 {std_val:.3f}"
        ax.text(
            max(bar.get_width(), 0) + 0.01,
            bar.get_y() + bar.get_height() / 2,
            txt,
            ha="left", va="center", fontsize=10, fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("MCC", fontsize=12)
    x_min, x_max = min(mccs), max(mccs)
    ax.set_xlim(min(0, x_min - 0.05), max(x_max * 1.25, 0.1))
    ax.set_title(
        f"MCC Ranking — {config.dataset} / ESM-2 {config.embedding} / Scaffold Split",
        fontsize=13,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()

    return _save_and_close(fig, os.path.join(config.resolved_output_dir, "benchmark_mcc_ranking.png"))


# ---------------------------------------------------------------------------
# Plot: per-metric strip
# ---------------------------------------------------------------------------

def plot_level_comparison_strip(
    aggregated: Dict[str, Dict],
    config: BenchmarkConfig,
) -> Optional[str]:
    """Strip chart with one panel per metric."""
    models = _available_models(aggregated)
    if not models or not METRICS_ORDER:
        return None

    n_metrics = len(METRICS_ORDER)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(3 * n_metrics, max(3, len(models) * 0.7 + 1)),
        sharey=True,
    )
    if n_metrics == 1:
        axes = [axes]

    y = np.arange(len(models))

    for j, metric in enumerate(METRICS_ORDER):
        ax = axes[j]
        vals = []
        stds = []
        colors = []
        for mk in models:
            v = aggregated[mk].get(metric)
            s = aggregated[mk].get(f"{metric}_std")
            vals.append(v if v is not None else np.nan)
            stds.append(s if s is not None else 0.0)
            colors.append(LEVEL_COLORS[mk])

        plot_vals = [v if not np.isnan(v) else 0.0 for v in vals]
        ax.barh(
            y, plot_vals,
            xerr=stds if any(s > 0 for s in stds) else None,
            capsize=3, color=colors, edgecolor="white", linewidth=0.5, height=0.55,
        )

        for i, (v, _s) in enumerate(zip(vals, stds)):
            if not np.isnan(v) and v != 0:
                ax.text(max(v, 0) + 0.005, i, f"{v:.3f}", ha="left", va="center", fontsize=8)

        v_min = min((v for v in vals if not np.isnan(v)), default=0)
        ax.set_xlim(min(0, v_min - 0.05), 1.05)
        ax.set_title(metric.upper(), fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LEVEL_LABELS[mk] for mk in models], fontsize=10)
    fig.suptitle(
        f"Per-Metric Comparison — {config.dataset} / ESM-2 {config.embedding} / Scaffold Split",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    return _save_and_close(
        fig,
        os.path.join(config.resolved_output_dir, "benchmark_per_metric.png"),
    )


# ---------------------------------------------------------------------------
# Aggregate entry point
# ---------------------------------------------------------------------------

def generate_all(
    aggregated: Dict[str, Dict],
    config: BenchmarkConfig,
) -> List[str]:
    """Generate all benchmark visualizations. Returns list of saved paths."""
    print("\n[Visualizations]")
    paths: list[str] = []

    for plot_fn in (
        plot_grouped_bar_chart,
        plot_radar_chart,
        plot_metric_heatmap,
        plot_mcc_ranking,
        plot_level_comparison_strip,
    ):
        try:
            path = plot_fn(aggregated, config)
            if path:
                paths.append(path)
        except Exception as exc:
            print(f"  WARNING: {plot_fn.__name__} failed: {exc}")

    return paths
