#!/usr/bin/env python3
"""Pre-compute Pfam domain profile (top-K kinome-relevant OHE) per kinase.

Uses pyhmmer (native-parallel) to scan each sequence against Pfam-A.hmm
and builds a one-hot vector over the K most frequent domains observed.

Output:
    data/embeddings/v8/pfam_{corpus}/{seq_id}.npy  (shape (K,), float32)
    data/embeddings/v8/pfam_{corpus}.vocab.json    (K domain accessions)

Requires a local Pfam-A.hmm database. Download via:
    wget https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
    gunzip Pfam-A.hmm.gz

Usage:
    python3 scripts/v8/precompute_pfam_target.py --corpus non_human \
        --hmm /path/to/Pfam-A.hmm --top-k 50
"""
from __future__ import annotations

import argparse
import json
import os
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


def _load_targets(corpus: str) -> pd.DataFrame:
    stem = _CORPUS_TSV[corpus]
    frames = []
    for split in ("train", "val", "test"):
        path = Path(f"{stem}_{split}.tsv")
        if not path.exists():
            continue
        frames.append(pd.read_csv(path, sep="\t", usecols=["seq_id", "seq"]))
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["seq_id"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(_CORPUS_TSV))
    ap.add_argument("--hmm", required=True, help="Path to Pfam-A.hmm (uncompressed)")
    ap.add_argument("--output-root", default="data/embeddings/v8")
    ap.add_argument("--threads", type=int, default=0, help="0 = all available")
    ap.add_argument("--top-k", type=int, default=50, help="top-K domains em OHE")
    ap.add_argument("--evalue", type=float, default=1e-5)
    args = ap.parse_args()

    try:
        import pyhmmer  # type: ignore
    except ImportError:
        print("[fatal] pyhmmer not installed (pip install pyhmmer)", file=sys.stderr)
        sys.exit(1)

    out_dir = REPO / args.output_root / f"pfam_{args.corpus}"
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir.parent / f"pfam_{args.corpus}.vocab.json"

    df = _load_targets(args.corpus)
    print(f"[pfam/{args.corpus}] {len(df)} sequências únicas")

    alphabet = pyhmmer.easel.Alphabet.amino()
    # Build Easel digital sequences
    sequences = [
        pyhmmer.easel.TextSequence(name=str(row.seq_id).encode(), sequence=row.seq).digitize(alphabet)
        for _, row in df.iterrows()
    ]

    print(f"  loading Pfam-A.hmm from {args.hmm} ...")
    with pyhmmer.plan7.HMMFile(args.hmm) as hmm_file:
        hmms = list(hmm_file)
    print(f"  {len(hmms)} HMMs loaded")

    cpus = args.threads or (os.cpu_count() or 1)
    print(f"  running hmmscan on {cpus} threads ...")
    # Scan: for each sequence, which HMMs hit it?
    hits_by_seq: dict[str, set[str]] = {str(s.name.decode()): set() for s in sequences}
    # pyhmmer: hmmscan(queries=hmms, sequences=seqs) returns TopHits per sequence
    for top_hits in pyhmmer.hmmer.hmmscan(sequences, hmms, cpus=cpus, E=args.evalue):
        qname = top_hits.query_name.decode() if top_hits.query_name else None
        if qname is None:
            continue
        for hit in top_hits:
            acc = hit.accession.decode() if hit.accession else hit.name.decode()
            hits_by_seq.setdefault(qname, set()).add(acc.split(".")[0])  # drop version

    # Build vocabulary = top-K most frequent domains
    from collections import Counter
    counter: Counter[str] = Counter()
    for accs in hits_by_seq.values():
        counter.update(accs)
    vocab = [acc for acc, _ in counter.most_common(args.top_k)]
    vocab_path.write_text(json.dumps(vocab))
    print(f"  vocab size: {len(vocab)}  → {vocab_path}")

    idx = {acc: i for i, acc in enumerate(vocab)}
    for seq_id, accs in hits_by_seq.items():
        vec = np.zeros(len(vocab), dtype=np.float32)
        for acc in accs:
            if acc in idx:
                vec[idx[acc]] = 1.0
        np.save(out_dir / f"{seq_id}.npy", vec)

    # Zero vector for sequences without hits (probably had errors)
    for _, row in df.iterrows():
        if not (out_dir / f"{row.seq_id}.npy").exists():
            np.save(out_dir / f"{row.seq_id}.npy", np.zeros(len(vocab), dtype=np.float32))

    print(f"[done] cache → {out_dir}")


if __name__ == "__main__":
    main()
