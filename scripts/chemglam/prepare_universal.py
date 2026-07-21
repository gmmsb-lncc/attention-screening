#!/usr/bin/env python3
"""Convert canonical scaffold splits to ChemGLaM's CSV contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED = {"canonical_smiles", "seq", "seq_id", "label"}
SPLIT_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))


def _keys(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    values = frame[columns].astype("string").fillna("")
    return set(map(tuple, values.to_numpy()))


def audit_source_splits(frames: dict[str, pd.DataFrame]) -> dict:
    """Fail closed if the canonical scaffold boundary is not disjoint."""
    comparisons = {}
    for left_name, right_name in SPLIT_PAIRS:
        left, right = frames[left_name], frames[right_name]
        row = {
            "exact_pairs": len(
                _keys(left, ["canonical_smiles", "seq"])
                & _keys(right, ["canonical_smiles", "seq"])
            ),
            "compounds": len(
                _keys(left, ["canonical_smiles"])
                & _keys(right, ["canonical_smiles"])
            ),
            "targets": len(_keys(left, ["seq_id"]) & _keys(right, ["seq_id"])),
        }
        if "scaffold" in left and "scaffold" in right:
            row["scaffolds"] = len(
                _keys(left, ["scaffold"]) & _keys(right, ["scaffold"])
            )
        comparisons[f"{left_name}_{right_name}"] = row

    leakage = {
        name: {field: value for field, value in row.items() if field != "targets" and value}
        for name, row in comparisons.items()
    }
    leakage = {name: row for name, row in leakage.items() if row}
    if leakage:
        raise ValueError(f"canonical split leakage detected: {leakage}")
    return {
        "status": "passed",
        "protocol": "universal_bemis_murcko_scaffold",
        "required_zero_overlap": ["exact_pairs", "compounds", "scaffolds"],
        "target_overlap_allowed": True,
        "comparisons": comparisons,
    }


def convert(path: Path, split: str, corpus: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if "dataset_source" not in frame:
        if corpus == "all":
            raise ValueError(f"{path}: all corpus requires dataset_source")
        frame["dataset_source"] = corpus
    renamed = frame.rename(
        columns={
            "canonical_smiles": "smiles",
            "seq": "target_sequence",
            "seq_id": "target_id",
        }
    )
    columns = ["smiles", "target_sequence", "target_id", "label", "dataset_source"]
    columns.extend(column for column in ("chembl_id", "scaffold") if column in renamed)
    result = renamed[columns].copy()
    result.insert(0, "source_row", frame.index.astype(int))
    result.insert(1, "split", split)
    result["label"] = result["label"].astype(int)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, default=Path("scaffolds_splits/output"))
    parser.add_argument("--corpus", choices=("all", "human", "non_human"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or Path("data/chemglam") / args.corpus
    prefix = "universal" if args.corpus == "all" else args.corpus

    paths = {
        split: args.split_root / f"{prefix}_{split}.tsv"
        for split in ("train", "val", "test")
    }
    source_frames = {split: pd.read_csv(path, sep="\t") for split, path in paths.items()}
    split_audit = audit_source_splits(source_frames)
    frames = {
        split: convert(path, split, args.corpus) for split, path in paths.items()
    }
    output.mkdir(parents=True, exist_ok=True)

    train_valid = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    n_train = len(frames["train"])
    split_indices = {
        "train": list(range(n_train)),
        "valid": list(range(n_train, len(train_valid))),
    }
    train_valid.to_csv(output / "train_valid.csv", index=False)
    frames["train"].to_csv(output / "train.csv", index=False)
    frames["val"].to_csv(output / "val.csv", index=False)
    frames["test"].to_csv(output / "test.csv", index=False)
    (output / "train_valid_split.json").write_text(
        json.dumps(split_indices, indent=2) + "\n"
    )

    manifest = {
        split: {
            "rows": len(frame),
            "positive": int(frame["label"].sum()),
            "negative": int((frame["label"] == 0).sum()),
            "unique_targets": int(frame["target_id"].nunique()),
            "human_rows": int((frame["dataset_source"] == "human").sum()),
            "non_human_rows": int((frame["dataset_source"] == "non_human").sum()),
        }
        for split, frame in frames.items()
    }
    manifest["split_audit"] = split_audit
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
