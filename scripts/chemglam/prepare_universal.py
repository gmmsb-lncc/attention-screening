#!/usr/bin/env python3
"""Convert the canonical scaffold splits to ChemGLaM's CSV contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED = {"canonical_smiles", "seq", "seq_id", "label", "dataset_source"}


def convert(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    result = frame.rename(
        columns={
            "canonical_smiles": "smiles",
            "seq": "target_sequence",
            "seq_id": "target_id",
        }
    )[["smiles", "target_sequence", "target_id", "label", "dataset_source"]]
    result.insert(0, "source_row", frame.index.astype(int))
    result.insert(1, "split", split)
    result["label"] = result["label"].astype(int)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, default=Path("scaffolds_splits/output"))
    parser.add_argument("--output", type=Path, default=Path("data/chemglam/universal"))
    args = parser.parse_args()

    frames = {
        split: convert(args.split_root / f"universal_{split}.tsv", split)
        for split in ("train", "val", "test")
    }
    args.output.mkdir(parents=True, exist_ok=True)

    train_valid = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    n_train = len(frames["train"])
    split_indices = {
        "train": list(range(n_train)),
        "valid": list(range(n_train, len(train_valid))),
    }
    train_valid.to_csv(args.output / "train_valid.csv", index=False)
    frames["train"].to_csv(args.output / "train.csv", index=False)
    frames["val"].to_csv(args.output / "val.csv", index=False)
    frames["test"].to_csv(args.output / "test.csv", index=False)
    (args.output / "train_valid_split.json").write_text(
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
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
