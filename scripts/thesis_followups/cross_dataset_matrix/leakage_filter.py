#!/usr/bin/env python3
"""Post-hoc leakage filter for cross-dataset evaluation.

Produces a filtered test TSV keyed on (seq_hash, canonical_smiles),
dropping any test row already present in the training corpus train+val.

Corpus file conventions (scaffolds_splits/output/):
    human_{train,val,test}.tsv       — 13 cols, Human only
    non_human_{train,val,test}.tsv   — 13 cols, Non-Human only
    universal_{train,val,test}.tsv   — 14 cols (adds dataset_source), H ∪ NH

The "all" corpus maps to universal_*.tsv. Test TSVs keep their native
schema; downstream inference scripts consume them as-is.

Usage:
    python3 scripts/thesis_followups/cross_dataset_matrix/leakage_filter.py \\
        --train-corpus all --test-corpus human \\
        --out-dir results/cross_matrix/filters/all_to_human
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SPLITS_DIR = REPO / "scaffolds_splits" / "output"

_CORPUS_FILE = {
    "human": "human",
    "non_human": "non_human",
    "all": "universal",
}

_SMILES_COL = "canonical_smiles"
_SEQ_COL = "seq"
_LABEL_COL = "label"


def _seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.strip().encode("utf-8")).hexdigest()[:12]


_CANON_CACHE: dict[str, str] = {}


def _canon(smiles: str) -> str:
    """Canonicalize SMILES via RDKit when available, else fall back to trim.

    The scaffolds_splits pipeline already writes canonical SMILES, so a
    trim fallback is safe for leakage detection across sibling TSVs.
    Cached per-SMILES to amortize RDKit parsing over duplicated rows.
    """
    cached = _CANON_CACHE.get(smiles)
    if cached is not None:
        return cached
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        _CANON_CACHE[smiles] = smiles.strip()
        return _CANON_CACHE[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        _CANON_CACHE[smiles] = smiles.strip()
    else:
        _CANON_CACHE[smiles] = Chem.MolToSmiles(mol, canonical=True)
    return _CANON_CACHE[smiles]


_SEQ_HASH_CACHE: dict[str, str] = {}


def _seq_hash_cached(seq: str) -> str:
    h = _SEQ_HASH_CACHE.get(seq)
    if h is not None:
        return h
    h = _seq_hash(seq)
    _SEQ_HASH_CACHE[seq] = h
    return h


def _read_tsv(corpus: str, split: str) -> pd.DataFrame:
    stem = _CORPUS_FILE[corpus]
    path = SPLITS_DIR / f"{stem}_{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def build_leakage_index(train_corpus: str) -> set[tuple[str, str]]:
    """Build the (seq_hash, canonical_smiles) set from train+val of `train_corpus`."""
    pairs: set[tuple[str, str]] = set()
    for split in ("train", "val"):
        df = _read_tsv(train_corpus, split)
        for seq, smi in zip(df[_SEQ_COL].astype(str), df[_SMILES_COL].astype(str)):
            pairs.add((_seq_hash_cached(seq), _canon(smi)))
    return pairs


def filter_test_tsv(
    test_corpus: str,
    leakage_index: set[tuple[str, str]],
    out_dir: Path,
) -> dict:
    df = _read_tsv(test_corpus, "test")
    n_original = len(df)

    seq_hashes = df[_SEQ_COL].astype(str).map(_seq_hash_cached)
    smi_canon = df[_SMILES_COL].astype(str).map(_canon)
    keys = list(zip(seq_hashes, smi_canon))
    keep_mask = [key not in leakage_index for key in keys]

    clean = df[keep_mask].reset_index(drop=True)
    n_clean = len(clean)
    n_removed = n_original - n_clean

    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = out_dir / "test_clean.tsv"
    clean.to_csv(out_tsv, sep="\t", index=False)

    report = {
        "train_corpus": None,
        "test_corpus": test_corpus,
        "n_original": int(n_original),
        "n_clean": int(n_clean),
        "n_removed": int(n_removed),
        "frac_leaked": float(n_removed) / float(n_original) if n_original else 0.0,
        "pos_rate_original": float(df[_LABEL_COL].astype(int).mean()) if n_original else 0.0,
        "pos_rate_clean": float(clean[_LABEL_COL].astype(int).mean()) if n_clean else 0.0,
        "out_tsv": str(out_tsv),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-corpus", required=True, choices=list(_CORPUS_FILE))
    ap.add_argument("--test-corpus", required=True, choices=list(_CORPUS_FILE))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    index = build_leakage_index(args.train_corpus)
    report = filter_test_tsv(args.test_corpus, index, out_dir)
    report["train_corpus"] = args.train_corpus
    report["n_train_val_pairs"] = len(index)

    (out_dir / "leakage_report.json").write_text(json.dumps(report, indent=2))

    print(
        f"[{args.train_corpus} -> {args.test_corpus}] "
        f"n_original={report['n_original']} "
        f"n_clean={report['n_clean']} "
        f"frac_leaked={report['frac_leaked']:.4f} "
        f"pos_rate: {report['pos_rate_original']:.3f} -> {report['pos_rate_clean']:.3f}"
    )


if __name__ == "__main__":
    main()
