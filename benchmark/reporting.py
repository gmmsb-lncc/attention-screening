"""Reporting utilities: terminal table and JSON export.

Separated from visualization to keep rendering logic focused.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List

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
    "level1_fp_knn",
    "level1_fp_mlp",
    "level2_emb_knn",
    "level2_emb_mlp",
    "level3_mat_knn",
    "level3_mat_mlp",
    "level4_crossatt_knn",
    "level4_crossatt_mlp",
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
        output["results"][model_key] = entry

    output_dir = config.resolved_output_dir
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "benchmark_comparison.json")

    with open(path, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"\nBenchmark JSON saved: {path}")
    return path
