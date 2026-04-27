#!/usr/bin/env python3
"""Pre-compute BioBERT per-token embeddings for every unique kinase target.

Input:  data/embeddings/v8/uniprot_{corpus}/{seq_id}.json
Output: data/embeddings/v8/biobert_{corpus}/{seq_id}.npy
        shape (L, 768), float32, UNPOOLED last_hidden_state.

The text fed to BioBERT concatenates:
  <protein_name> . <organism> . function: <function text>. keywords: <KW...>.
  GO: <top-K GO terms>. KEGG: <KEGG ids>.
Truncated to 512 tokens (BioBERT max position).

Attention pooling happens inside the v8 model (see AttentionPool1D).

Usage:
    python3 scripts/v8/precompute_biobert_target.py --corpus non_human
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel  # type: ignore

REPO = Path(__file__).resolve().parents[2]
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
MAX_LEN = 512

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
        frames.append(pd.read_csv(path, sep="\t",
                                   usecols=["seq_id", "target_kinase", "organism"]))
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["seq_id"]).reset_index(drop=True)


def _compose_text(name: str, organism: str, ann: dict) -> str:
    parts = [name.strip(), organism.strip()]
    if ann.get("function"):
        parts.append("function: " + " ".join(ann["function"]))
    if ann.get("keywords"):
        parts.append("keywords: " + ", ".join(ann["keywords"]))
    if ann.get("go"):
        parts.append("GO: " + "; ".join(ann["go"][:20]))
    if ann.get("kegg"):
        parts.append("KEGG: " + ", ".join(ann["kegg"]))
    return " . ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(_CORPUS_TSV))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output-root", default="data/embeddings/v8")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    uniprot_dir = REPO / args.output_root / f"uniprot_{args.corpus}"
    out_dir = REPO / args.output_root / f"biobert_{args.corpus}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not uniprot_dir.exists():
        print(f"[fatal] {uniprot_dir} not found — rode fetch_uniprot_annotations.py primeiro",
              file=sys.stderr)
        sys.exit(1)

    df = _load_targets(args.corpus)
    print(f"[biobert/{args.corpus}] {len(df)} targets únicos")

    rows = []
    for _, row in df.iterrows():
        if (out_dir / f"{row.seq_id}.npy").exists():
            continue
        json_path = uniprot_dir / f"{row.seq_id}.json"
        ann = json.loads(json_path.read_text()) if json_path.exists() else {}
        text = _compose_text(row.target_kinase, row.organism, ann)
        rows.append((row.seq_id, text))
    print(f"  {len(rows)} faltam (skip {len(df) - len(rows)})")
    if not rows:
        print("[done]"); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and not args.no_amp
    print(f"  device={device} amp={use_amp}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(rows), args.batch_size), desc="biobert"):
            chunk = rows[i:i + args.batch_size]
            ids = [s for s, _ in chunk]
            texts = [t for _, t in chunk]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_LEN)
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                res = model(**enc)
            hs = res.last_hidden_state.float().cpu().numpy()
            attn = enc["attention_mask"].cpu().numpy()
            for j, sid in enumerate(ids):
                L = int(attn[j].sum())
                np.save(out_dir / f"{sid}.npy", hs[j, :L].astype(np.float32))
    print(f"[done] cache → {out_dir}")


if __name__ == "__main__":
    main()
