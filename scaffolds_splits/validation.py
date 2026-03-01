"""Validation helpers for scaffold-based split outputs."""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Set

import pandas as pd


def _set_overlap_size(a: Set[str], b: Set[str]) -> int:
    return len(a & b)


def validate_dataset_split(dataset_name: str, splits: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    required = {"train", "val", "test"}
    if set(splits.keys()) != required:
        raise ValueError(f"{dataset_name}: splits keys must be exactly {required}")

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    for split_name, split_df in splits.items():
        if split_df.empty:
            raise ValueError(f"{dataset_name}: split '{split_name}' is empty")

    # Compound disjointness
    train_comp = set(train_df["chembl_id"].astype(str).unique())
    val_comp = set(val_df["chembl_id"].astype(str).unique())
    test_comp = set(test_df["chembl_id"].astype(str).unique())

    if _set_overlap_size(train_comp, val_comp) > 0:
        raise ValueError(f"{dataset_name}: train/val compound overlap detected")
    if _set_overlap_size(train_comp, test_comp) > 0:
        raise ValueError(f"{dataset_name}: train/test compound overlap detected")
    if _set_overlap_size(val_comp, test_comp) > 0:
        raise ValueError(f"{dataset_name}: val/test compound overlap detected")

    # Scaffold disjointness
    train_scaf = set(train_df["scaffold"].astype(str).unique())
    val_scaf = set(val_df["scaffold"].astype(str).unique())
    test_scaf = set(test_df["scaffold"].astype(str).unique())

    if _set_overlap_size(train_scaf, val_scaf) > 0:
        raise ValueError(f"{dataset_name}: train/val scaffold overlap detected")
    if _set_overlap_size(train_scaf, test_scaf) > 0:
        raise ValueError(f"{dataset_name}: train/test scaffold overlap detected")
    if _set_overlap_size(val_scaf, test_scaf) > 0:
        raise ValueError(f"{dataset_name}: val/test scaffold overlap detected")

    # Test label presence (hard requirement).
    test_labels = set(test_df["label"].astype(int).unique().tolist())
    if not {0, 1}.issubset(test_labels):
        raise ValueError(
            f"{dataset_name}: test split must contain labels 0 and 1; got labels={sorted(test_labels)}"
        )

    summary = {}
    for split_name, split_df in splits.items():
        summary[split_name] = {
            "rows": int(len(split_df)),
            "unique_compounds": int(split_df["chembl_id"].nunique()),
            "unique_scaffolds": int(split_df["scaffold"].nunique()),
            "pos_rows": int((split_df["label"] == 1).sum()),
            "neg_rows": int((split_df["label"] == 0).sum()),
        }

    return {
        "dataset": dataset_name,
        "summary": summary,
        "test_scaffolds": sorted(test_scaf),
    }


def warn_split_quality(
    dataset_name: str,
    splits: Dict[str, pd.DataFrame],
    min_scaffolds: int = 10,
    max_class_rate_divergence: float = 0.05,
    min_rows_for_monotonic_warning: int = 10,
) -> List[str]:
    """Emit warnings for quality issues in splits. Returns list of warning messages."""
    msgs: List[str] = []
    train_df = splits.get("train", pd.DataFrame())
    val_df = splits.get("val", pd.DataFrame())
    test_df = splits.get("test", pd.DataFrame())

    # 1. Low scaffold count in val.
    if not val_df.empty and "scaffold" in val_df.columns:
        n_val_scaffolds = val_df["scaffold"].nunique()
        if n_val_scaffolds < min_scaffolds:
            msg = (
                f"[{dataset_name}] WARNING: val has only {n_val_scaffolds} scaffolds "
                f"(recommended >= {min_scaffolds})"
            )
            warnings.warn(msg, stacklevel=2)
            msgs.append(msg)

    # 2. Monotonic scaffolds in val/test with many rows.
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        if split_df.empty or "scaffold" not in split_df.columns:
            continue
        scaff_rates = split_df.groupby("scaffold")["label"].agg(["mean", "size"])
        mono = scaff_rates[
            ((scaff_rates["mean"] == 0.0) | (scaff_rates["mean"] == 1.0))
            & (scaff_rates["size"] >= min_rows_for_monotonic_warning)
        ]
        if len(mono) > 0:
            total_mono_rows = int(mono["size"].sum())
            msg = (
                f"[{dataset_name}] WARNING: {split_name} has {len(mono)} monotonic scaffold(s) "
                f"(>= {min_rows_for_monotonic_warning} rows each, {total_mono_rows} rows total)"
            )
            warnings.warn(msg, stacklevel=2)
            msgs.append(msg)

    # 3. Class rate divergence between train and test.
    if not train_df.empty and not test_df.empty and "label" in train_df.columns:
        train_rate = train_df["label"].mean()
        test_rate = test_df["label"].mean()
        divergence = abs(train_rate - test_rate)
        if divergence > max_class_rate_divergence:
            msg = (
                f"[{dataset_name}] WARNING: class rate divergence train vs test = "
                f"{divergence:.3f} ({train_rate:.3f} vs {test_rate:.3f}), "
                f"exceeds threshold {max_class_rate_divergence:.3f}"
            )
            warnings.warn(msg, stacklevel=2)
            msgs.append(msg)

    return msgs


def validate_universal_test_scaffolds(
    human_test_scaffolds: Iterable[str],
    non_human_test_scaffolds: Iterable[str],
) -> None:
    hs = set(human_test_scaffolds)
    ns = set(non_human_test_scaffolds)
    if hs != ns:
        extra_h = sorted(hs - ns)[:10]
        extra_n = sorted(ns - hs)[:10]
        raise ValueError(
            "Universal test scaffold mismatch between datasets. "
            f"Only in human: {extra_h}; only in non_human: {extra_n}"
        )
