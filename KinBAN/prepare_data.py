"""Convert thesis scaffold splits to GraphBAN CSV format.

GraphBAN expects CSV files with columns: SMILES, Protein, Y
located at: datasets/{dataset}/scaffold/{train,val,test}.csv

Usage:
    python prepare_data.py --dataset non_human
    python prepare_data.py --dataset human
    python prepare_data.py --dataset all  # merges human + non_human
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Paths relative to this script's directory
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SCAFFOLD_DIR = REPO_ROOT / "scaffolds_splits" / "output"

# Column mapping: thesis → GraphBAN
COLUMN_MAP = {
    "canonical_smiles": "SMILES",
    "seq": "Protein",
    "label": "Y",
}

# GraphBAN truncates proteins to 1022 internally (ESM-1b limit)
MAX_PROTEIN_LEN = 1022


def load_split(dataset: str, split: str) -> pd.DataFrame:
    """Load a single split TSV from the scaffold output directory."""
    if split == "test":
        # Test files are at top level
        path = SCAFFOLD_DIR / f"{dataset}_test.tsv.gz"
        if not path.exists():
            path = SCAFFOLD_DIR / f"{dataset}_test.tsv"
    else:
        # Train/val files are under scenarios/Sc/
        path = SCAFFOLD_DIR / "scenarios" / "Sc" / f"{dataset}_{split}.tsv.gz"
        if not path.exists():
            path = SCAFFOLD_DIR / "scenarios" / "Sc" / f"{dataset}_{split}.tsv"
        if not path.exists():
            # Fallback to top level (some datasets use flat layout)
            path = SCAFFOLD_DIR / f"{dataset}_{split}.tsv.gz"
            if not path.exists():
                path = SCAFFOLD_DIR / f"{dataset}_{split}.tsv"

    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    df = pd.read_csv(path, sep="\t")
    return df


def convert_to_graphban(df: pd.DataFrame) -> pd.DataFrame:
    """Convert thesis dataframe to GraphBAN format."""
    required = list(COLUMN_MAP.keys())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df[required].rename(columns=COLUMN_MAP).copy()

    # Validate: drop rows with missing SMILES or Protein sequences
    before = len(out)
    out = out.dropna(subset=["SMILES", "Protein"])
    out = out[out["SMILES"].str.strip().astype(bool)]
    out = out[out["Protein"].str.strip().astype(bool)]
    dropped = before - len(out)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing SMILES/Protein")

    # Ensure Y is integer 0/1
    out["Y"] = out["Y"].astype(int)

    # Warn about long protein sequences
    long_seqs = (out["Protein"].str.len() > MAX_PROTEIN_LEN).sum()
    if long_seqs > 0:
        max_len = out["Protein"].str.len().max()
        print(
            f"  Warning: {long_seqs} sequences exceed {MAX_PROTEIN_LEN} residues "
            f"(max={max_len}). GraphBAN will truncate them internally."
        )

    return out.reset_index(drop=True)


def prepare_dataset(dataset: str, output_dir: Path) -> None:
    """Prepare all splits for a dataset."""
    print(f"\n{'='*60}")
    print(f"Preparing dataset: {dataset}")
    print(f"{'='*60}")

    out_path = output_dir / "datasets" / "kinase" / dataset / "scaffold"
    out_path.mkdir(parents=True, exist_ok=True)

    stats = {}
    for split in ["train", "val", "test"]:
        print(f"\n  [{split}]")
        if dataset == "all":
            # Merge human + non_human for "all" dataset
            dfs = []
            for sub in ["human", "non_human"]:
                try:
                    df_sub = load_split(sub, split)
                    dfs.append(df_sub)
                    print(f"    Loaded {sub}: {len(df_sub)} rows")
                except FileNotFoundError as e:
                    print(f"    Skipped {sub}: {e}")
            if not dfs:
                raise RuntimeError(f"No data found for 'all' dataset, split={split}")
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = load_split(dataset, split)
            print(f"    Loaded: {len(df)} rows")

        graphban_df = convert_to_graphban(df)

        csv_path = out_path / f"{split}.csv"
        graphban_df.to_csv(csv_path, index=False)

        pos = int((graphban_df["Y"] == 1).sum())
        neg = int((graphban_df["Y"] == 0).sum())
        stats[split] = {"total": len(graphban_df), "pos": pos, "neg": neg}
        print(f"    Saved: {csv_path} ({len(graphban_df)} rows, pos={pos}, neg={neg})")

    # Summary
    print(f"\n  Summary for '{dataset}':")
    for split, s in stats.items():
        ratio = s["pos"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"    {split:5s}: {s['total']:>8,} rows  (pos={s['pos']:>6,}, neg={s['neg']:>6,}, ratio={ratio:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert scaffold splits to GraphBAN format")
    parser.add_argument(
        "--dataset",
        choices=["non_human", "human", "all"],
        default="non_human",
        help="Dataset to prepare (default: non_human)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Output directory (default: GraphBAN/)",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in ["non_human", "human", "all"]:
            prepare_dataset(ds, args.output_dir)
    else:
        prepare_dataset(args.dataset, args.output_dir)

    print("\nDone! Data ready for GraphBAN training.")


if __name__ == "__main__":
    main()
