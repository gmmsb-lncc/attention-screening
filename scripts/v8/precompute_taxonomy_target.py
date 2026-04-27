#!/usr/bin/env python3
"""Pre-compute NCBI Taxonomy lineage as one-hot vector per kinase target.

Lineage levels: kingdom, phylum, class, order, family (5 OHE slots,
concatenated). Each slot has its own vocabulary.

Requires local NCBI taxdump:
    wget https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz
    tar xzf taxdump.tar.gz -C data/ncbi_taxonomy/

Output:
    data/embeddings/v8/taxonomy_{corpus}/{seq_id}.npy  (shape (D,), float32)
    data/embeddings/v8/taxonomy_{corpus}.vocab.json

Usage:
    python3 scripts/v8/precompute_taxonomy_target.py --corpus non_human \
        --taxdump data/ncbi_taxonomy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

_CORPUS_TSV = {
    "human":     REPO / "scaffolds_splits/output/human",
    "non_human": REPO / "scaffolds_splits/output/non_human",
    "all":       REPO / "scaffolds_splits/output/universal",
}

LEVELS = ("kingdom", "phylum", "class", "order", "family")


def _load_targets(corpus: str) -> pd.DataFrame:
    stem = _CORPUS_TSV[corpus]
    frames = []
    for split in ("train", "val", "test"):
        path = Path(f"{stem}_{split}.tsv")
        if not path.exists():
            continue
        frames.append(pd.read_csv(path, sep="\t", usecols=["seq_id", "organism"]))
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["seq_id"]).reset_index(drop=True)


def _load_taxdump(taxdump_dir: Path):
    """Load names.dmp + nodes.dmp into minimal lookup dicts."""
    name_to_taxid: dict[str, int] = {}
    names_path = taxdump_dir / "names.dmp"
    if not names_path.exists():
        print(f"[fatal] taxdump not found at {taxdump_dir} — extrai taxdump.tar.gz lá",
              file=sys.stderr)
        sys.exit(1)
    with open(names_path) as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip("|\n\t ").split("|")]
            if len(parts) < 4:
                continue
            taxid, name, _, ntype = parts[0], parts[1], parts[2], parts[3]
            if ntype == "scientific name":
                name_to_taxid.setdefault(name, int(taxid))
    parent: dict[int, int] = {}
    rank: dict[int, str] = {}
    with open(taxdump_dir / "nodes.dmp") as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip("|\n\t ").split("|")]
            if len(parts) < 3:
                continue
            taxid, par, rk = int(parts[0]), int(parts[1]), parts[2]
            parent[taxid] = par
            rank[taxid] = rk
    id_to_name: dict[int, str] = {v: k for k, v in name_to_taxid.items()}
    return name_to_taxid, parent, rank, id_to_name


def _lineage(taxid: int, parent: dict[int, int], rank: dict[int, str],
             id_to_name: dict[int, str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {lvl: None for lvl in LEVELS}
    seen = set()
    cur = taxid
    while cur and cur not in seen and cur != 1:
        seen.add(cur)
        r = rank.get(cur)
        if r in out:
            out[r] = id_to_name.get(cur)
        cur = parent.get(cur, 0)
    # kingdom normalization (NCBI uses "superkingdom" for Archaea/Bacteria/Eukaryota)
    if out["kingdom"] is None:
        cur = taxid
        while cur and cur != 1:
            if rank.get(cur) == "superkingdom":
                out["kingdom"] = id_to_name.get(cur); break
            cur = parent.get(cur, 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(_CORPUS_TSV))
    ap.add_argument("--taxdump", required=True, help="Dir with names.dmp + nodes.dmp")
    ap.add_argument("--output-root", default="data/embeddings/v8")
    args = ap.parse_args()

    out_dir = REPO / args.output_root / f"taxonomy_{args.corpus}"
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir.parent / f"taxonomy_{args.corpus}.vocab.json"

    df = _load_targets(args.corpus)
    print(f"[taxonomy/{args.corpus}] {len(df)} targets únicos")

    name_to_taxid, parent, rank, id_to_name = _load_taxdump(Path(args.taxdump))
    print(f"  taxdump loaded: {len(name_to_taxid)} scientific names, {len(parent)} nodes")

    # Pass 1: collect all unique level values
    lineages: dict[str, dict[str, str | None]] = {}
    for _, row in df.iterrows():
        tid = name_to_taxid.get(row.organism)
        if tid is None:
            lineages[row.seq_id] = {lvl: None for lvl in LEVELS}
        else:
            lineages[row.seq_id] = _lineage(tid, parent, rank, id_to_name)

    # Build vocab per-level
    vocab: dict[str, list[str]] = {lvl: sorted({v for lin in lineages.values()
                                                for v in [lin.get(lvl)] if v}) for lvl in LEVELS}
    vocab_path.write_text(json.dumps(vocab))
    total = sum(len(v) for v in vocab.values())
    print(f"  vocab sizes: {[f'{k}:{len(v)}' for k, v in vocab.items()]} total={total}")

    # Build index
    idx: dict[str, dict[str, int]] = {}
    offset = 0
    for lvl in LEVELS:
        idx[lvl] = {v: offset + i for i, v in enumerate(vocab[lvl])}
        offset += len(vocab[lvl])

    # Encode
    for seq_id, lin in lineages.items():
        vec = np.zeros(total, dtype=np.float32)
        for lvl in LEVELS:
            v = lin.get(lvl)
            if v and v in idx[lvl]:
                vec[idx[lvl][v]] = 1.0
        np.save(out_dir / f"{seq_id}.npy", vec)

    print(f"[done] cache → {out_dir}")


if __name__ == "__main__":
    main()
