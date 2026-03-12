"""Reporting utilities: terminal table and JSON export.

Separated from visualization to keep rendering logic focused.
"""

from __future__ import annotations

import json
import os
import glob
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from benchmark.config import (
    LEVEL_LABELS,
    METRICS_ORDER,
    PCHEMBL_ACTIVITY_THRESHOLD,
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

def _load_split_records(config: BenchmarkConfig) -> Dict[str, object]:
    """Load full scaffold split records (train/val/test) for JSON export."""
    scaffold_dir = config.scaffold_split_dir
    split_paths = {
        "train": os.path.join(scaffold_dir, "scenarios/Sc", "universal_train.tsv"),
        "val": os.path.join(scaffold_dir, "scenarios/Sc", "universal_val.tsv"),
        "test": os.path.join(scaffold_dir, "universal_test.tsv"),
    }

    source_filter = config.dataset_source_filter
    records: Dict[str, object] = {}

    for split_name, split_path in split_paths.items():
        if not os.path.exists(split_path):
            records[split_name] = {"path": split_path, "rows": 0, "records": []}
            continue

        df = pd.read_csv(split_path, sep="\t")
        if source_filter is not None and "dataset_source" in df.columns:
            df = df[df["dataset_source"] == source_filter].reset_index(drop=True)

        if "label" not in df.columns and "pchembl_value" in df.columns:
            df["label"] = (df["pchembl_value"] >= PCHEMBL_ACTIVITY_THRESHOLD).astype(int)

        records[split_name] = {
            "path": split_path,
            "rows": int(len(df)),
            "records": df.to_dict(orient="records"),
        }

    return records


def _load_seed_outputs(config: BenchmarkConfig) -> Dict[str, object]:
    """Load all per-seed level JSON outputs into the final benchmark JSON."""
    pattern = os.path.join(config.resolved_output_dir, "**", "seed_*", "*_knn_mlp_results.json")
    paths = sorted(glob.glob(pattern, recursive=True))

    outputs: Dict[str, object] = {}
    for p in paths:
        rel = os.path.relpath(p, config.resolved_output_dir)
        try:
            with open(p) as fh:
                outputs[rel] = json.load(fh)
        except (OSError, json.JSONDecodeError):
            outputs[rel] = {"error": "failed_to_load"}

    return outputs


def _build_seed_outputs_summary(seed_outputs: Dict[str, object]) -> Dict[str, object]:
    """Build compact per-seed summary for dashboard consumption."""
    summary: Dict[str, object] = {}

    for rel_path, payload in seed_outputs.items():
        if not isinstance(payload, dict):
            summary[rel_path] = {"error": "invalid_payload"}
            continue

        scaffold_block = payload.get("Split by Scaffold")
        if not isinstance(scaffold_block, dict):
            summary[rel_path] = {"error": "missing_scaffold_block"}
            continue

        seed_summary: Dict[str, object] = {}
        for model_name in ("KNN", "MLP"):
            model_block = scaffold_block.get(model_name)
            if not isinstance(model_block, dict):
                continue

            details = model_block.get("details") if isinstance(model_block.get("details"), dict) else {}
            eval_block = details.get("evaluation") if isinstance(details.get("evaluation"), dict) else {}
            cal_block = details.get("calibration") if isinstance(details.get("calibration"), dict) else {}

            seed_summary[model_name] = {
                "threshold": model_block.get("threshold"),
                "accuracy": model_block.get("accuracy"),
                "mcc": model_block.get("mcc"),
                "f1": model_block.get("f1"),
                "auc": model_block.get("auc"),
                "calibration_rows": cal_block.get("n_rows"),
                "evaluation_rows": eval_block.get("n_rows"),
            }

        summary[rel_path] = seed_summary

    return summary


def _build_seed_outputs_aggregate(seed_outputs_summary: Dict[str, object]) -> Dict[str, object]:
    """Aggregate threshold and key metrics across seeds by level/model."""
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for rel_path, payload in seed_outputs_summary.items():
        if not isinstance(payload, dict):
            continue

        level_dir = rel_path.split(os.sep)[0] if rel_path else "unknown_level"
        level_bucket = grouped.setdefault(level_dir, {})

        for model_name in ("KNN", "MLP"):
            model_payload = payload.get(model_name)
            if not isinstance(model_payload, dict):
                continue

            metric_bucket = level_bucket.setdefault(model_name, {})
            for metric_key in ("threshold", "accuracy", "mcc", "f1", "auc"):
                value = model_payload.get(metric_key)
                if isinstance(value, (int, float)):
                    metric_bucket.setdefault(metric_key, []).append(float(value))

    aggregate: Dict[str, object] = {}
    for level_dir, model_map in grouped.items():
        level_summary: Dict[str, object] = {}
        for model_name, metrics_map in model_map.items():
            metric_summary: Dict[str, float] = {}
            for metric_key, values in metrics_map.items():
                arr = np.asarray(values, dtype=np.float64)
                if arr.size == 0:
                    continue
                metric_summary[f"{metric_key}_mean"] = float(np.mean(arr))
                metric_summary[f"{metric_key}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                metric_summary[f"{metric_key}_n"] = int(arr.size)
            level_summary[model_name] = metric_summary
        aggregate[level_dir] = level_summary

    return aggregate


def save_benchmark_json(
    aggregated: Dict[str, Dict],
    config: BenchmarkConfig,
    elapsed_seconds: float,
) -> str:
    """Save ``benchmark_comparison.json`` and return its path."""
    seed_outputs = _load_seed_outputs(config)
    seed_outputs_summary = _build_seed_outputs_summary(seed_outputs)

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
            "data_splits": _load_split_records(config),
            "seed_outputs": seed_outputs,
            "seed_outputs_summary": seed_outputs_summary,
            "seed_outputs_aggregate": _build_seed_outputs_aggregate(seed_outputs_summary),
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
