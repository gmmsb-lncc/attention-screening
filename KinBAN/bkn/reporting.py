"""Result aggregation and formatted summary reporting."""
from __future__ import annotations

import numpy as np


def aggregate_results(all_metrics: list[dict]) -> dict:
    """Compute mean ± std across seeds for train/val/test splits."""
    metric_names = ["accuracy", "f1", "precision", "recall", "mcc", "auroc"]
    agg: dict = {}

    for split in ["train", "val", "test"]:
        agg[split] = {}
        for m in metric_names:
            values = [r[split][m] for r in all_metrics]
            agg[split][m] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

    threshold_values = [r["val_threshold"] for r in all_metrics]
    agg["val_threshold"] = {
        "mean": float(np.mean(threshold_values)),
        "std": float(np.std(threshold_values)),
        "values": threshold_values,
    }
    agg["training_time_s"] = {
        "mean": float(np.mean([r["training_time_s"] for r in all_metrics])),
    }

    native_mcc_vals = [
        r["graphban_native"]["mcc"] for r in all_metrics if "graphban_native" in r
    ]
    if native_mcc_vals:
        agg["graphban_native_mcc"] = {
            "mean": float(np.mean(native_mcc_vals)),
            "std": float(np.std(native_mcc_vals)),
            "values": native_mcc_vals,
        }

    return agg


def print_summary_table(agg: dict, dataset: str, seeds: list[int]) -> None:
    """Print a formatted train/val/test summary table across all seeds."""
    print(f"\n{'='*72}")
    print(f"  BKN (ESM-2 + MoLFormer + AttnPool) — {dataset} ({len(seeds)} seeds)")
    print(f"{'='*72}")
    print(
        f"  {'Metric':<12} {'Train Mean':>11} {'± Std':>7}  "
        f"{'Val Mean':>9} {'± Std':>7}  {'Test Mean':>10} {'± Std':>7}"
    )
    print(f"  {'─'*68}")

    for m in ["mcc", "auroc", "f1", "accuracy", "precision", "recall"]:
        tr = agg["train"][m]
        vl = agg["val"][m]
        te = agg["test"][m]
        print(
            f"  {m.upper():<12} {tr['mean']:>11.4f} {tr['std']:>7.4f}  "
            f"{vl['mean']:>9.4f} {vl['std']:>7.4f}  "
            f"{te['mean']:>10.4f} {te['std']:>7.4f}"
        )

    print(f"  {'─'*68}")
    if "graphban_native_mcc" in agg:
        native = agg["graphban_native_mcc"]
        print(
            f"  {'MCC (native)':<12} {'N/A':>11} {'':>7}  {'N/A':>9} {'':>7}  "
            f"{native['mean']:>10.4f} {native['std']:>7.4f}  "
            f"(test-set threshold, not used)"
        )
    print(f"  Avg training time: {agg['training_time_s']['mean']:.1f}s per seed")
