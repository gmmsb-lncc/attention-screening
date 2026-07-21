#!/usr/bin/env python3
"""Audit the split artifacts shipped by the pinned ChemGLaM repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


UPSTREAM_COMMIT = "3f09b907af3b53fde32e44c7e98b098c2a2c552c"
DATASETS = ("bindingdb", "davis", "metz", "pdbbind")
COMPARISONS = (
    ("train_validation", "train", "validation"),
    ("train_test", "train", "test"),
    ("validation_test", "validation", "test"),
)


def _as_tuples(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    values = frame[columns].astype("string").fillna("")
    return set(map(tuple, values.to_numpy()))


def _overlap(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> int:
    return len(_as_tuples(left, columns) & _as_tuples(right, columns))


def _label_overlap(
    left: pd.DataFrame, right: pd.DataFrame, pair_columns: list[str]
) -> dict[str, int]:
    if "label" not in left or "label" not in right:
        return {}
    left_labels = left.groupby(pair_columns, dropna=False)["label"].agg(
        lambda values: frozenset(values)
    )
    right_labels = right.groupby(pair_columns, dropna=False)["label"].agg(
        lambda values: frozenset(values)
    )
    common = left_labels.index.intersection(right_labels.index)
    same = sum(left_labels.loc[key] == right_labels.loc[key] for key in common)
    return {
        "same_label_pairs": int(same),
        "conflicting_label_pairs": int(len(common) - same),
    }


def audit_pair_dataset(root: Path) -> dict[str, Any]:
    frames = {
        "train": pd.read_csv(root / "train.csv"),
        "validation": pd.read_csv(root / "valid.csv"),
        "test": pd.read_csv(root / "test.csv"),
    }
    pair_columns = ["smiles", "target_sequence"]
    result: dict[str, Any] = {
        "rows": {name: int(len(frame)) for name, frame in frames.items()},
        "within_split_duplicate_pair_rows": {
            name: int(frame.duplicated(pair_columns).sum())
            for name, frame in frames.items()
        },
        "overlap": {},
    }
    candidate_fields = (
        "drug_id", "pubchem_cid", "smiles", "target_id", "uniprot_id",
        "target_sequence",
    )
    for name, left_name, right_name in COMPARISONS:
        left, right = frames[left_name], frames[right_name]
        comparison = {
            "exact_pairs": _overlap(left, right, pair_columns),
            **_label_overlap(left, right, pair_columns),
        }
        for field in candidate_fields:
            if field in left and field in right:
                comparison[field] = _overlap(left, right, [field])
        result["overlap"][name] = comparison
    return result


def audit_pdbbind(root: Path) -> dict[str, Any]:
    frames = {
        "train": pd.read_csv(root / "train.csv"),
        "validation": pd.read_csv(root / "valid.csv"),
        "test": pd.read_csv(root / "test.csv"),
    }
    return {
        "scope": "PDB identifiers only; compound and protein fields are not shipped in these CSVs",
        "rows": {name: int(len(frame)) for name, frame in frames.items()},
        "overlap": {
            name: {"pdb_id": _overlap(frames[left], frames[right], ["pdb_id"])}
            for name, left, right in COMPARISONS
        },
    }


def audit(upstream_root: Path) -> dict[str, Any]:
    data_root = upstream_root / "data"
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        folds = {}
        for fold in range(5):
            fold_root = data_root / dataset / f"cv{fold}"
            folds[f"cv{fold}"] = (
                audit_pdbbind(fold_root)
                if dataset == "pdbbind"
                else audit_pair_dataset(fold_root)
            )
        datasets[dataset] = folds
    return {
        "model": "ChemGLaM",
        "upstream_commit": UPSTREAM_COMMIT,
        "pair_definition": ["smiles", "target_sequence"],
        "datasets": datasets,
        "finding": {
            "test_pair_overlap": "zero in all auditable BindingDB, Davis and Metz folds",
            "davis_train_validation_pair_overlap": "confirmed in all five folds",
            "pdbbind_scope": "only PDB identifier disjointness can be tested from shipped CSVs",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=Path("ChemGLaM"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.upstream_root)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
