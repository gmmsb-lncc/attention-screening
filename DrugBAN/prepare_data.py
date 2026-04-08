"""Convert universal scaffold splits to DrugBAN CSV format.

All three datasets use the UNIVERSAL scaffold split
(universal_train/val/test.tsv) to ensure no test scaffold leaks across
any corpus.  For 'human' and 'non_human' we filter by the
``dataset_source`` column.  For 'all' we use the full universal file.

DrugBAN data format:
  - Columns: SMILES, Protein, Y
  - Files: datasets/kinase/{dataset}/scaffold/{train,val,test}.csv

Usage:
    python prepare_data.py --dataset non_human
    python prepare_data.py --dataset human
    python prepare_data.py --dataset all
    python prepare_data.py --all   # prepare all three at once
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

# Mapping from dataset name to the value of the ``dataset_source`` column.
# None means "no filter" (use all rows from the universal file).
CORPUS_FILTER: dict[str, str | None] = {
    "human": "human",
    "non_human": "non_human",
    "all": None,
}


def _universal_path(split: str) -> Path:
    """Return the path to the universal split file, preferring .tsv over .tsv.gz."""
    stem = f"universal_{split}"
    gz = SCAFFOLD_DIR / f"{stem}.tsv.gz"
    plain = SCAFFOLD_DIR / f"{stem}.tsv"
    if gz.exists():
        return gz
    if plain.exists():
        return plain
    raise FileNotFoundError(
        f"Universal split file not found for split='{split}'. "
        f"Expected one of:\n  {gz}\n  {plain}"
    )


def load_universal_split(dataset: str, split: str) -> pd.DataFrame:
    """Load a split from the universal scaffold files, filtering by corpus if needed.

    Parameters
    ----------
    dataset:
        One of 'human', 'non_human', 'all'.
    split:
        One of 'train', 'val', 'test'.
    """
    path = _universal_path(split)
    df = pd.read_csv(path, sep="\t")

    corpus_value = CORPUS_FILTER[dataset]
    if corpus_value is not None:
        if "dataset_source" not in df.columns:
            raise ValueError(
                f"Column 'dataset_source' not found in {path}. "
                f"Cannot filter for dataset='{dataset}'."
            )
        before = len(df)
        df = df[df["dataset_source"] == corpus_value].copy()
        print(
            f"  [{split}] filtered '{corpus_value}': "
            f"{before} → {len(df)} rows"
        )
    else:
        print(f"  [{split}] universal (all corpora): {len(df)} rows")

    return df


def convert_to_drugban(df: pd.DataFrame) -> pd.DataFrame:
    """Convert thesis dataframe to DrugBAN format (SMILES, Protein, Y)."""
    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out: pd.DataFrame = (
        df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP).copy()
    )

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
            f"  Warning: {long_count} proteins exceed {MAX_PROTEIN_LEN} "
            f"residues (max={max_len})."
        )

    return out.reset_index(drop=True)


def prepare_dataset(dataset: str, output_dir: Path) -> None:
    """Prepare one dataset (human / non_human / all) using the universal split."""
    if dataset not in CORPUS_FILTER:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from: {list(CORPUS_FILTER)}"
        )

    out_path = output_dir / "datasets" / "kinase" / dataset / "scaffold"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\nPreparing dataset: {dataset}  [universal scaffold split]")
    stats: dict[str, dict[str, int]] = {}

    for split in ["train", "val", "test"]:
        df_raw = load_universal_split(dataset, split)
        out_df = convert_to_drugban(df_raw)
        out_file = out_path / f"{split}.csv"
        out_df.to_csv(out_file, index=False)

        pos = int((out_df["Y"] == 1).sum())
        neg = int((out_df["Y"] == 0).sum())
        stats[split] = {"total": len(out_df), "pos": pos, "neg": neg}
        print(
            f"  [{split}] saved: {out_file} "
            f"({len(out_df)} rows, pos={pos}, neg={neg})"
        )

    print("  Summary:")
    for split in ["train", "val", "test"]:
        s = stats[split]
        ratio = (s["pos"] / s["total"] * 100) if s["total"] else 0.0
        print(
            f"    {split:5s}: {s['total']:>8,} rows "
            f"(pos={s['pos']:>6,}, neg={s['neg']:>6,}, ratio={ratio:.1f}%)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert universal scaffold splits to DrugBAN format"
    )
    parser.add_argument(
        "--dataset",
        choices=["non_human", "human", "all"],
        default=None,
        help="Dataset to prepare (default: all three)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="prepare_all",
        help="Prepare all three datasets (non_human, human, all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Output directory (default: DrugBAN/)",
    )
    args = parser.parse_args()

    datasets_to_run: list[str]
    if args.prepare_all or args.dataset is None:
        datasets_to_run = ["non_human", "human", "all"]
    else:
        datasets_to_run = [args.dataset]

    for ds in datasets_to_run:
        prepare_dataset(ds, args.output_dir)

    print("\nDone! Data ready for DrugBAN (universal scaffold split).")


if __name__ == "__main__":
    main()
