"""Validation helpers for scaffold-based split outputs."""

from __future__ import annotations

from typing import Dict, Iterable, Set

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
