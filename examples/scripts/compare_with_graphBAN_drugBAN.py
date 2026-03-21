#!/usr/bin/env python3
"""Generate GraphBAN vs DrugBAN vs L3 comparison plots from four JSON files.

Expected inputs:
- GraphBAN results JSON (with aggregate train/test blocks)
- DrugBAN results JSON (with aggregated test block)
- Benchmark train JSON (for L3 AttnPool+KNN train metrics)
- Benchmark test JSON (for L3 AttnPool+KNN test metrics)

Outputs:
- 4 PNG figures
- 2 CSV summary tables
- 1 README markdown with quick notes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METRICS = ["accuracy", "f1", "precision", "recall", "mcc", "auroc"]
L3_KEY = "level3_attnpool_knn"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_l3_metrics(split_json: dict) -> Dict[str, Tuple[float, float]]:
    if "results" not in split_json or L3_KEY not in split_json["results"]:
        raise KeyError(f"Cannot find L3 key '{L3_KEY}' in benchmark JSON")

    r = split_json["results"][L3_KEY]
    return {
        "accuracy": (r["accuracy"], r["accuracy_std"]),
        "f1": (r["f1"], r["f1_std"]),
        "precision": (r["precision"], r["precision_std"]),
        "recall": (r["recall"], r["recall_std"]),
        "mcc": (r["mcc"], r["mcc_std"]),
        "auroc": (r["auc"], r["auc_std"]),
    }


def _build_rows(
    graph_train: Dict[str, Tuple[float, float]],
    graph_test: Dict[str, Tuple[float, float]],
    drug_test: Dict[str, Tuple[float, float]],
    l3_train: Dict[str, Tuple[float, float]],
    l3_test: Dict[str, Tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_test = []
    rows_train = []

    for metric in METRICS:
        rows_test.extend(
            [
                {
                    "Model": "GraphBAN",
                    "Metric": metric.upper(),
                    "Mean": graph_test[metric][0],
                    "Std": graph_test[metric][1],
                },
                {
                    "Model": "DrugBAN",
                    "Metric": metric.upper(),
                    "Mean": drug_test[metric][0],
                    "Std": drug_test[metric][1],
                },
                {
                    "Model": "L3 (AttnPool+KNN)",
                    "Metric": metric.upper(),
                    "Mean": l3_test[metric][0],
                    "Std": l3_test[metric][1],
                },
            ]
        )

        rows_train.extend(
            [
                {
                    "Model": "GraphBAN",
                    "Metric": metric.upper(),
                    "Mean": graph_train[metric][0],
                    "Std": graph_train[metric][1],
                },
                {
                    "Model": "L3 (AttnPool+KNN)",
                    "Metric": metric.upper(),
                    "Mean": l3_train[metric][0],
                    "Std": l3_train[metric][1],
                },
            ]
        )

    return pd.DataFrame(rows_test), pd.DataFrame(rows_train)


def _save_test_figure(df_test: pd.DataFrame, out_dir: Path, colors: dict) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=df_test, x="Metric", y="Mean", hue="Model", palette=colors, ax=ax)

    for i, metric in enumerate(df_test["Metric"].unique()):
        sub = df_test[df_test["Metric"] == metric].reset_index(drop=True)
        for j, row in sub.iterrows():
            x = i + (-0.27 + j * 0.27)
            ax.errorbar(
                x=x,
                y=row["Mean"],
                yerr=row["Std"],
                fmt="none",
                ecolor="black",
                capsize=3,
                lw=1,
            )

    ax.set_ylim(0.0, 0.9)
    ax.set_title("Test Metrics Comparison (Non-Human, Scaffold Split)")
    ax.set_ylabel("Score")
    plt.tight_layout()
    plt.savefig(out_dir / "11_test_metrics_3models_mean_std.png", dpi=300)
    plt.close(fig)


def _save_train_figure(df_train: pd.DataFrame, out_dir: Path, colors: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=df_train, x="Metric", y="Mean", hue="Model", palette=colors, ax=ax)

    for i, metric in enumerate(df_train["Metric"].unique()):
        sub = df_train[df_train["Metric"] == metric].reset_index(drop=True)
        for j, row in sub.iterrows():
            x = i + (-0.18 + j * 0.36)
            ax.errorbar(
                x=x,
                y=row["Mean"],
                yerr=row["Std"],
                fmt="none",
                ecolor="black",
                capsize=3,
                lw=1,
            )

    ax.set_ylim(0.0, 1.0)
    ax.set_title("Train Metrics Comparison (GraphBAN vs L3)")
    ax.set_ylabel("Score")
    plt.tight_layout()
    plt.savefig(out_dir / "12_train_metrics_graphban_vs_l3.png", dpi=300)
    plt.close(fig)


def _save_gap_figure(
    graph_train: Dict[str, Tuple[float, float]],
    graph_test: Dict[str, Tuple[float, float]],
    l3_train: Dict[str, Tuple[float, float]],
    l3_test: Dict[str, Tuple[float, float]],
    out_dir: Path,
    colors: dict,
) -> None:
    rows_gap = []
    for model, tr, te in [
        ("GraphBAN", graph_train, graph_test),
        ("L3 (AttnPool+KNN)", l3_train, l3_test),
    ]:
        for metric in ["mcc", "auroc", "accuracy"]:
            rows_gap.append(
                {
                    "Model": model,
                    "Metric": metric.upper(),
                    "Train": tr[metric][0],
                    "Test": te[metric][0],
                    "Gap (Train-Test)": tr[metric][0] - te[metric][0],
                }
            )

    gap_df = pd.DataFrame(rows_gap)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(
        data=gap_df,
        x="Metric",
        y="Gap (Train-Test)",
        hue="Model",
        palette=colors,
        ax=axes[0],
    )
    axes[0].set_title("Generalization Gap")
    axes[0].axhline(0, color="black", lw=1)

    long_tt = gap_df.melt(
        id_vars=["Model", "Metric"],
        value_vars=["Train", "Test"],
        var_name="Split",
        value_name="Score",
    )
    sns.barplot(
        data=long_tt,
        x="Metric",
        y="Score",
        hue="Split",
        ax=axes[1],
        palette=["#4daf4a", "#e41a1c"],
    )
    axes[1].set_title("Train vs Test Means (GraphBAN + L3)")

    plt.tight_layout()
    plt.savefig(out_dir / "13_generalization_gap_train_vs_test.png", dpi=300)
    plt.close(fig)


def _save_heatmap(
    graph_test: Dict[str, Tuple[float, float]],
    drug_test: Dict[str, Tuple[float, float]],
    l3_test: Dict[str, Tuple[float, float]],
    out_dir: Path,
) -> None:
    rank_rows = []
    for model, test_map in [
        ("GraphBAN", graph_test),
        ("DrugBAN", drug_test),
        ("L3 (AttnPool+KNN)", l3_test),
    ]:
        rank_rows.append(
            {
                "Model": model,
                "MCC": test_map["mcc"][0],
                "AUROC": test_map["auroc"][0],
                "F1": test_map["f1"][0],
            }
        )

    rank_df = pd.DataFrame(rank_rows).set_index("Model")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.heatmap(
        rank_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Score"},
        ax=ax,
    )
    ax.set_title("Test Performance Heatmap (Higher is Better)")
    plt.tight_layout()
    plt.savefig(out_dir / "14_test_heatmap_mcc_auroc_f1.png", dpi=300)
    plt.close(fig)


def _save_csv_and_readme(
    graph_train: Dict[str, Tuple[float, float]],
    graph_test: Dict[str, Tuple[float, float]],
    drug_test: Dict[str, Tuple[float, float]],
    l3_train: Dict[str, Tuple[float, float]],
    l3_test: Dict[str, Tuple[float, float]],
    out_dir: Path,
) -> None:
    summary_test = []
    for metric in METRICS:
        summary_test.append(
            {
                "metric": metric,
                "graphban_mean": graph_test[metric][0],
                "graphban_std": graph_test[metric][1],
                "drugban_mean": drug_test[metric][0],
                "drugban_std": drug_test[metric][1],
                "l3_knn_mean": l3_test[metric][0],
                "l3_knn_std": l3_test[metric][1],
                "delta_graph_minus_drug": graph_test[metric][0] - drug_test[metric][0],
                "delta_graph_minus_l3": graph_test[metric][0] - l3_test[metric][0],
                "delta_l3_minus_drug": l3_test[metric][0] - drug_test[metric][0],
            }
        )

    summary_train = []
    for metric in METRICS:
        summary_train.append(
            {
                "metric": metric,
                "graphban_mean": graph_train[metric][0],
                "graphban_std": graph_train[metric][1],
                "l3_knn_mean": l3_train[metric][0],
                "l3_knn_std": l3_train[metric][1],
                "delta_graph_minus_l3": graph_train[metric][0] - l3_train[metric][0],
            }
        )

    pd.DataFrame(summary_test).to_csv(out_dir / "comparison_summary_test_3models.csv", index=False)
    pd.DataFrame(summary_train).to_csv(
        out_dir / "comparison_summary_train_graphban_vs_l3.csv", index=False
    )

    values = {
        "GraphBAN": graph_test["mcc"][0],
        "DrugBAN": drug_test["mcc"][0],
        "L3 (AttnPool+KNN)": l3_test["mcc"][0],
    }

    lines = [
        "# Comparison: GraphBAN vs DrugBAN vs L3 (AttnPool+KNN)",
        "",
        "## Data Sources",
        "- GraphBAN: fair protocol JSON (train/val/test aggregates)",
        "- DrugBAN: test-only aggregate in attached JSON",
        "- L3: benchmark train/test aggregate from attached JSON",
        "",
        "## Best Test MCC",
    ]

    for model, mcc in sorted(values.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {model}: {mcc:.6f}")

    lines.extend(
        [
            "",
            "## Note",
            "- DrugBAN input does not include aggregated train metrics; train plots include GraphBAN and L3 only.",
        ]
    )

    (out_dir / "README_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GraphBAN, DrugBAN, and L3 (AttnPool+KNN) using four JSON inputs."
    )
    parser.add_argument(
        "--graphban-json",
        required=True,
        type=Path,
        help="Path to GraphBAN results JSON",
    )
    parser.add_argument(
        "--drugban-json",
        required=True,
        type=Path,
        help="Path to DrugBAN results JSON",
    )
    parser.add_argument(
        "--l3-train-json",
        required=True,
        type=Path,
        help="Path to benchmark train JSON containing level3_attnpool_knn",
    )
    parser.add_argument(
        "--l3-test-json",
        required=True,
        type=Path,
        help="Path to benchmark test JSON containing level3_attnpool_knn",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/comparisons/graphban_drugban_l3_non_human"),
        help="Directory to save generated plots and tables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sns.set_theme(style="whitegrid")

    graph = _load_json(args.graphban_json)
    drug = _load_json(args.drugban_json)
    l3_train_json = _load_json(args.l3_train_json)
    l3_test_json = _load_json(args.l3_test_json)

    graph_train = {
        m: (graph["aggregate"]["train"][m]["mean"], graph["aggregate"]["train"][m]["std"])
        for m in METRICS
    }
    graph_test = {
        m: (graph["aggregate"]["test"][m]["mean"], graph["aggregate"]["test"][m]["std"])
        for m in METRICS
    }
    drug_test = {
        m: (drug["aggregated"][m]["mean"], drug["aggregated"][m]["std"]) for m in METRICS
    }
    l3_train = _extract_l3_metrics(l3_train_json)
    l3_test = _extract_l3_metrics(l3_test_json)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "GraphBAN": "#1f77b4",
        "DrugBAN": "#ff7f0e",
        "L3 (AttnPool+KNN)": "#2ca02c",
    }

    df_test, df_train = _build_rows(graph_train, graph_test, drug_test, l3_train, l3_test)

    _save_test_figure(df_test, out_dir, colors)
    _save_train_figure(df_train, out_dir, colors)
    _save_gap_figure(graph_train, graph_test, l3_train, l3_test, out_dir, colors)
    _save_heatmap(graph_test, drug_test, l3_test, out_dir)
    _save_csv_and_readme(graph_train, graph_test, drug_test, l3_train, l3_test, out_dir)

    print(f"[OK] Comparison files generated at: {out_dir}")


if __name__ == "__main__":
    main()
