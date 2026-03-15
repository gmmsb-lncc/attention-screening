"""Reporting utilities: terminal table and JSON export.

Separated from visualization to keep rendering logic focused.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np

from benchmark.config import (
    LEVEL_LABELS,
    METRICS_ORDER,
    SUPPORTED_EMBEDDINGS,
    BenchmarkConfig,
)


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

MODEL_DISPLAY_ORDER = [
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
    "level3a_attnpool_mlp",
    "level4_crossatt_knn",
    "level4_crossatt_mlp",
    "level5_da_knn",
    "level5_da_mlp",
    "level5b_da_knn",
    "level5b_da_mlp",
    "level6a_ban_knn",
    "level6a_ban_mlp",
    "level6b_ban_knn",
    "level6b_ban_mlp",
]


def print_comparison_table(
    aggregated: Dict[str, Dict],
    config: BenchmarkConfig,
) -> None:
    """Print a formatted comparison table to the terminal."""
    print("\n" + "=" * 90)
    print(
        f"BENCHMARK COMPARISON: {config.dataset} / ESM-2 {config.embedding} / Scaffold Split"
    )
    print("=" * 90)

    header = f"{'Model':<22s}"
    for metric in METRICS_ORDER:
        header += f"  {metric.upper():>10s}"
    print(header)
    print("-" * 90)

    for model_key in MODEL_DISPLAY_ORDER:
        if model_key not in aggregated:
            continue
        row = aggregated[model_key]
        label = LEVEL_LABELS[model_key]
        line = f"{label:<22s}"
        for metric in METRICS_ORDER:
            val = row.get(metric)
            std = row.get(f"{metric}_std")
            if val is None:
                line += f"  {'N/A':>10s}"
            elif std and std > 0:
                cell = f"{val:.3f}\u00b1{std:.3f}"
                line += f"  {cell:>10s}"
            else:
                line += f"  {val:>10.4f}"
        print(line)

    print("=" * 90)


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def save_benchmark_json(
    aggregated: Dict[str, Dict],
    config: BenchmarkConfig,
    elapsed_seconds: float,
) -> str:
    """Save ``benchmark_comparison.json`` and return its path."""
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": config.dataset,
            "embedding": config.embedding,
            "embedding_full": SUPPORTED_EMBEDDINGS.get(config.embedding, config.embedding),
            "split": "scaffold",
            "levels_executed": config.levels,
            "seeds": config.resolved_seeds,
            "elapsed_seconds": round(elapsed_seconds, 1),
            "rigor": {
                "strict_level_completeness": os.getenv("BENCHMARK_STRICT_LEVEL_COMPLETENESS", "1"),
                "require_train_selection_for_test": os.getenv("BENCHMARK_REQUIRE_TRAIN_SELECTION", "1"),
                "early_stopping_patience_rule": "10% of epochs",
            },
            "mlp_runtime_settings": {
                "BENCHMARK_MLP_USE_CV": os.getenv("BENCHMARK_MLP_USE_CV", "1"),
                "BENCHMARK_MLP_FOLDS": os.getenv("BENCHMARK_MLP_FOLDS", "3"),
                "BENCHMARK_MLP_CAL_RESTARTS": os.getenv("BENCHMARK_MLP_CAL_RESTARTS", "3"),
                "BENCHMARK_MLP_ENSEMBLE": os.getenv("BENCHMARK_MLP_ENSEMBLE", "5"),
                "BENCHMARK_MLP_OVERSAMPLE": os.getenv("BENCHMARK_MLP_OVERSAMPLE", "1"),
                "BENCHMARK_LEVEL3_SELECTION_METRIC": os.getenv("BENCHMARK_LEVEL3_SELECTION_METRIC", ""),
                "BENCHMARK_LEVEL3_DOWNSTREAM_EVAL_EVERY": os.getenv("BENCHMARK_LEVEL3_DOWNSTREAM_EVAL_EVERY", ""),
                "BENCHMARK_LEVEL3_USE_AUX_CHANNEL": os.getenv("BENCHMARK_LEVEL3_USE_AUX_CHANNEL", ""),
            },
        },
        "results": {},
    }

    for model_key, row in aggregated.items():
        label = LEVEL_LABELS.get(model_key, model_key)
        entry = {"label": label}
        for metric in METRICS_ORDER:
            val = row.get(metric)
            entry[metric] = round(val, 6) if val is not None else None
            std = row.get(f"{metric}_std")
            entry[f"{metric}_std"] = round(std, 6) if std is not None else None

        seed_results = row.get("seed_results")
        if isinstance(seed_results, dict):
            mcc_values = []
            for seed_payload in seed_results.values():
                if isinstance(seed_payload, dict):
                    mcc = seed_payload.get("mcc")
                    if isinstance(mcc, (int, float)):
                        mcc_values.append(float(mcc))
            if len(mcc_values) >= 2:
                low, high = _bootstrap_ci95(mcc_values)
                entry["mcc_ci95_low"] = round(low, 6)
                entry["mcc_ci95_high"] = round(high, 6)
                entry["mcc_n_seeds"] = len(mcc_values)
        output["results"][model_key] = entry

    output_dir = config.resolved_output_dir
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "benchmark_comparison.json")

    with open(path, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"\nBenchmark JSON saved: {path}")
    return path


def _bootstrap_ci95(values: List[float], n_boot: int = 2000) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        v = float(arr.mean()) if arr.size == 1 else 0.0
        return v, v

    rng = np.random.default_rng(42)
    means = np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = float(sample.mean())
    low = float(np.percentile(means, 2.5))
    high = float(np.percentile(means, 97.5))
    return low, high
