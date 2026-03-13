"""Convert scaffold splits to DrugBAN CSV format.

DrugBAN data format expected by most training scripts:
- Columns: SMILES, Protein, Y
- Files: datasets/kinase/{dataset}/scaffold/{train,val,test}.csv

Usage:
    python prepare_data.py --dataset non_human
    python prepare_data.py --dataset human
    python prepare_data.py --dataset all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SCAFFOLD_DIR = REPO_ROOT / "scaffolds_splits" / "output"

COLUMN_MAP = {
    "canonical_smiles": "SMILES",
    "seq": "Protein",
    "label": "Y",
}

MAX_PROTEIN_LEN = 1022


def load_split(dataset: str, split: str) -> pd.DataFrame:
    """Load a split TSV from scaffold output."""
    if split == "test":
        path = SCAFFOLD_DIR / f"{dataset}_test.tsv.gz"
        if not path.exists():
            path = SCAFFOLD_DIR / f"{dataset}_test.tsv"
    else:
        path = SCAFFOLD_DIR / "scenarios" / "Sc" / f"{dataset}_{split}.tsv.gz"
        if not path.exists():
            path = SCAFFOLD_DIR / "scenarios" / "Sc" / f"{dataset}_{split}.tsv"
        if not path.exists():
            path = SCAFFOLD_DIR / f"{dataset}_{split}.tsv.gz"
            if not path.exists():
                path = SCAFFOLD_DIR / f"{dataset}_{split}.tsv"

    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    return pd.read_csv(path, sep="\t")


def convert_to_drugban(df: pd.DataFrame) -> pd.DataFrame:
    """Convert thesis dataframe to DrugBAN format."""
    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out: pd.DataFrame = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP).copy()

    before = len(out)
    out = out.dropna(subset=["SMILES", "Protein"])
    out = out[out["SMILES"].astype(str).str.strip().astype(bool)]
    out = out[out["Protein"].astype(str).str.strip().astype(bool)]
    dropped = before - len(out)
    if dropped > 0:
        print(f"  Dropped {dropped} invalid rows")

    out["Y"] = out["Y"].astype(int)

    long_count = int((out["Protein"].str.len() > MAX_PROTEIN_LEN).sum())
    if long_count > 0:
        max_len = int(out["Protein"].str.len().max())
        print(
            f"  Warning: {long_count} proteins exceed {MAX_PROTEIN_LEN} residues "
            f"(max={max_len})."
        )

    return out.reset_index(drop=True)


def prepare_dataset(dataset: str, output_dir: Path) -> None:
    out_path = output_dir / "datasets" / "kinase" / dataset / "scaffold"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\nPreparing dataset: {dataset}")
    stats: dict[str, dict[str, int]] = {}

    for split in ["train", "val", "test"]:
        if dataset == "all":
            frames: list[pd.DataFrame] = []
            for sub in ["human", "non_human"]:
                try:
                    s = load_split(sub, split)
                    frames.append(s)
                    print(f"  [{split}] loaded {sub}: {len(s)} rows")
                except FileNotFoundError as e:
                    print(f"  [{split}] skipped {sub}: {e}")
            if not frames:
                raise RuntimeError(f"No data found for dataset={dataset}, split={split}")
            df = pd.concat(frames, ignore_index=True)
        else:
            df = load_split(dataset, split)
            print(f"  [{split}] loaded: {len(df)} rows")

        out_df = convert_to_drugban(df)
        out_file = out_path / f"{split}.csv"
        out_df.to_csv(out_file, index=False)

        pos = int((out_df["Y"] == 1).sum())
        neg = int((out_df["Y"] == 0).sum())
        stats[split] = {"total": len(out_df), "pos": pos, "neg": neg}
        print(f"  [{split}] saved: {out_file} ({len(out_df)} rows, pos={pos}, neg={neg})")

    print("  Summary:")
    for split in ["train", "val", "test"]:
        s = stats[split]
        ratio = (s["pos"] / s["total"] * 100) if s["total"] else 0.0
        print(
            f"    {split:5s}: {s['total']:>8,} rows "
            f"(pos={s['pos']:>6,}, neg={s['neg']:>6,}, ratio={ratio:.1f}%)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert scaffold splits to DrugBAN format")
    parser.add_argument(
        "--dataset",
        choices=["non_human", "human", "all"],
        default="non_human",
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Output directory (default: DrugBAN/)",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in ["non_human", "human", "all"]:
            prepare_dataset(ds, args.output_dir)
    else:
        prepare_dataset(args.dataset, args.output_dir)

    print("\nDone! Data ready for DrugBAN.")


if __name__ == "__main__":
    main()
