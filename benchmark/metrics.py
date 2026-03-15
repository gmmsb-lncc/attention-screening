"""Metric aggregation across benchmark levels.

Collects per-level results and unifies them into a single dict
suitable for reporting and visualization.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np

from benchmark.config import METRICS_ORDER


def extract_metric(
    results_dict: Dict,
    model_key: str,
    metric: str,
) -> Optional[float]:
    """Safely extract a metric value from a model results block."""
    if not results_dict or model_key not in results_dict:
        return None
    val = results_dict[model_key].get(metric)
    if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
        return float(val)
    return None


def extract_metric_std(
    results_dict: Dict,
    model_key: str,
    metric: str,
) -> Optional[float]:
    """Extract standard deviation for a metric from multi-seed results."""
    if not results_dict or model_key not in results_dict:
        return None
    val = results_dict[model_key].get(f"{metric}_std")
    if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
        return float(val)
    return None


def find_scaffold_key(results: Dict) -> Optional[str]:
    """Find the scaffold scenario key in a results dict.

    Different modules may name the key slightly differently, so we
    search for ``"scaffold"`` in a case-insensitive manner.
    """
    if not results:
        return None

    for key in results:
        normalised = key.replace("\n", " ").lower()
        if "scaffold" in normalised:
            return key

    if results:
        first_key = next(iter(results))
        warnings.warn(
            f"No 'scaffold' scenario found; falling back to '{first_key}'"
        )
        return first_key

    return None


def aggregate_benchmark_metrics(
    *,
    level1a_results: Optional[Dict] = None,
    level1b_results: Optional[Dict] = None,
    level1c_results: Optional[Dict] = None,
    level2_results: Optional[Dict] = None,
    level3_results: Optional[Dict] = None,
    level3a_results: Optional[Dict] = None,
    level4_results: Optional[Dict] = None,
    level5_results: Optional[Dict] = None,
    level5b_results: Optional[Dict] = None,
    level6a_results: Optional[Dict] = None,
    level6b_results: Optional[Dict] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Aggregate metrics from all levels into a unified dict.

    Returns::

        {
            "level1a_fp_knn": {"accuracy": 0.85, "accuracy_std": 0.02, ...},
            "level1a_fp_mlp": {...},
            ...
        }
    """
    aggregated: Dict[str, Dict[str, Optional[float]]] = {}

    _level_mapping = [
        (level1a_results, [("KNN", "level1a_fp_knn"), ("MLP", "level1a_fp_mlp")]),
        (level1b_results, [("KNN", "level1b_ligmean_knn"), ("MLP", "level1b_ligmean_mlp")]),
        (level1c_results, [("KNN", "level1c_ligattn_knn"), ("MLP", "level1c_ligattn_mlp")]),
        (level2_results, [("KNN", "level2_meanpool_knn"), ("MLP", "level2_meanpool_mlp")]),
        (level3_results, [("KNN", "level3_attnpool_knn"), ("MLP", "level3_attnpool_mlp")]),
        (level3a_results, [("MLP", "level3a_attnpool_mlp")]),
        (level4_results, [("KNN", "level4_crossatt_knn"), ("MLP", "level4_crossatt_mlp")]),
        (level5_results, [("KNN", "level5_da_knn"), ("MLP", "level5_da_mlp")]),
        (level5b_results, [("KNN", "level5b_da_knn"), ("MLP", "level5b_da_mlp")]),
        (level6a_results, [("KNN", "level6a_ban_knn"), ("MLP", "level6a_ban_mlp")]),
        (level6b_results, [("KNN", "level6b_ban_knn"), ("MLP", "level6b_ban_mlp")]),
    ]

    for results, model_pairs in _level_mapping:
        if not results:
            continue
        sc_key = find_scaffold_key(results)
        if not sc_key or sc_key not in results:
            continue
        sc = results[sc_key]
        for model_key, label_key in model_pairs:
            row: Dict[str, Optional[float]] = {}
            for metric in METRICS_ORDER:
                row[metric] = extract_metric(sc, model_key, metric)
                row[f"{metric}_std"] = extract_metric_std(sc, model_key, metric)
            aggregated[label_key] = row

    return aggregated
