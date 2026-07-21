#!/usr/bin/env python3
"""Convert canonical scaffold splits to CMA-DTI's tabular contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

REQUIRED = {"canonical_smiles", "seq", "seq_id", "chembl_id", "label"}
MAX_DRUG_NODES = 310


def convert(path: Path, split: str, corpus: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if "dataset_source" not in frame:
        frame["dataset_source"] = corpus
    result = frame.rename(columns={
        "canonical_smiles": "SMILES", "seq": "Protein",
        "seq_id": "target_id", "label": "Y",
    })[["SMILES", "Protein", "target_id", "chembl_id", "Y", "dataset_source"]]
    result.insert(0, "source_row", frame.index.astype(int))
    result.insert(1, "split", split)
    result["Y"] = result["Y"].astype(int)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=Path("scaffolds_splits/output"))
    parser.add_argument("--corpus", choices=("all", "human", "non_human"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("data/cmadti") / args.corpus
    prefix = "universal" if args.corpus == "all" else args.corpus
    frames = {
        split: convert(args.split_root / f"{prefix}_{split}.tsv", split, args.corpus)
        for split in ("train", "val", "test")
    }
    output.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        frame.to_csv(output / f"{split}.csv", index=False)

    unique_smiles = pd.concat(
        [frame["SMILES"] for frame in frames.values()], ignore_index=True
    ).drop_duplicates()
    atom_counts = []
    invalid = []
    for smiles in unique_smiles:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            invalid.append(str(smiles))
        else:
            atom_counts.append(mol.GetNumAtoms())
    if invalid:
        raise ValueError(f"invalid canonical SMILES encountered: {invalid[:5]}")
    max_atoms = max(atom_counts, default=0)
    if max_atoms > MAX_DRUG_NODES:
        raise ValueError(
            f"max molecule has {max_atoms} atoms, exceeds MAX_DRUG_NODES={MAX_DRUG_NODES}"
        )

    manifest = {
        "corpus": args.corpus,
        "max_drug_nodes": MAX_DRUG_NODES,
        "max_observed_atoms": max_atoms,
        "unique_smiles": int(len(unique_smiles)),
        "splits": {
            split: {
                "rows": int(len(frame)),
                "positive": int(frame["Y"].sum()),
                "negative": int((frame["Y"] == 0).sum()),
                "unique_targets": int(frame["target_id"].nunique()),
                "human_rows": int((frame["dataset_source"] == "human").sum()),
                "non_human_rows": int((frame["dataset_source"] == "non_human").sum()),
            }
            for split, frame in frames.items()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
