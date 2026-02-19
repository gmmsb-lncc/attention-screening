"""Output writers for scaffold split artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


def ensure_output_dir(output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_dataset_splits(
    dataset_name: str,
    splits: Dict[str, pd.DataFrame],
    output_dir: str,
) -> Dict[str, str]:
    out = ensure_output_dir(output_dir)
    paths: Dict[str, str] = {}

    for split_name in ("train", "val", "test"):
        filename = f"{dataset_name}_{split_name}.tsv"
        path = out / filename
        splits[split_name].to_csv(path, sep="\t", index=False)
        paths[split_name] = str(path)

    return paths


def write_universal_scaffolds(scaffolds_payload: Any, output_dir: str) -> str:
    out = ensure_output_dir(output_dir)
    path = out / "test_scaffolds_universal.json"
    if isinstance(scaffolds_payload, list):
        payload = {
            "mode": "shared_scaffold",
            "n_scaffolds": len(scaffolds_payload),
            "shared_scaffolds": scaffolds_payload,
        }
    elif isinstance(scaffolds_payload, dict):
        payload = scaffolds_payload
    else:
        raise TypeError("scaffolds_payload must be a list or dict")

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(path)


def write_combined_test(
    human_test_df: pd.DataFrame,
    non_human_test_df: pd.DataFrame,
    output_dir: str,
) -> str:
    out = ensure_output_dir(output_dir)
    path = out / "universal_test.tsv"

    h = human_test_df.copy()
    h["dataset_source"] = "human"

    n = non_human_test_df.copy()
    n["dataset_source"] = "non_human"

    combined = pd.concat([h, n], axis=0, ignore_index=True)
    combined.to_csv(path, sep="\t", index=False)
    return str(path)


def write_manifest(manifest: Dict, output_dir: str) -> str:
    out = ensure_output_dir(output_dir)
    path = out / "manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return str(path)


def write_scenario_splits(
    scenario_code: str,
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str,
    dropped_df: pd.DataFrame | None = None,
) -> Dict[str, str]:
    scenario_dir = ensure_output_dir(Path(output_dir) / "scenarios" / scenario_code)
    paths: Dict[str, str] = {}

    train_path = scenario_dir / f"{dataset_name}_train.tsv"
    val_path = scenario_dir / f"{dataset_name}_val.tsv"
    train_df.to_csv(train_path, sep="\t", index=False)
    val_df.to_csv(val_path, sep="\t", index=False)
    paths["train"] = str(train_path)
    paths["val"] = str(val_path)

    if dropped_df is not None and len(dropped_df) > 0:
        dropped_path = scenario_dir / f"{dataset_name}_dropped.tsv"
        dropped_df.to_csv(dropped_path, sep="\t", index=False)
        paths["dropped"] = str(dropped_path)

    return paths


def write_distribution_summary(summary_rows: Iterable[Dict[str, Any]], output_dir: str) -> str:
    out = ensure_output_dir(output_dir)
    path = out / "split_class_distribution_summary.tsv"
    df = pd.DataFrame(list(summary_rows))
    df.to_csv(path, sep="\t", index=False)
    return str(path)
