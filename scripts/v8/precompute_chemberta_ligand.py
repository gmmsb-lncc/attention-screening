#!/usr/bin/env python3
"""Pre-compute ChemBERTa-77M-MTR per-token embeddings for every unique
compound in a kinase corpus.

Output layout:
    data/embeddings/v8/chemberta_{corpus}/{chembl_id}.npy
Each .npy is a (L, 384) float32 array of last_hidden_state, UNPOOLED.
Attention pooling is applied inside the v8 model at training time
(see AttentionPool1D in benchmark/levels/level4_cnn_v8.py).

Optimizations:
  - Batch forward on GPU with AMP (fp16 autocast)
  - tokenizers parallelism env var enabled
  - Idempotent: skips {chembl_id}.npy that already exist
  - Parallel save via multiprocessing.Pool
  - Unique SMILES (by chembl_id) only

Usage:
    python3 scripts/v8/precompute_chemberta_ligand.py --corpus non_human
    python3 scripts/v8/precompute_chemberta_ligand.py --corpus human --batch-size 256
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Thread-parallel tokenizer BEFORE import transformers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaModel  # type: ignore

REPO = Path(__file__).resolve().parents[2]

_CORPUS_TSV = {
    "human":     REPO / "scaffolds_splits/output/human",
    "non_human": REPO / "scaffolds_splits/output/non_human",
    "all":       REPO / "scaffolds_splits/output/universal",
}

MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
MAX_LEN = 290                 # match GraphBAN upstream usage


def _load_unique(corpus: str) -> pd.DataFrame:
    stem = _CORPUS_TSV[corpus]
    frames = []
    for split in ("train", "val", "test"):
        path = Path(f"{stem}_{split}.tsv")
        if not path.exists():
            continue
        frames.append(pd.read_csv(path, sep="\t",
                                   usecols=["chembl_id", "canonical_smiles"]))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["chembl_id"])
    return df.reset_index(drop=True)


def _save_one(args: tuple[str, np.ndarray, Path]) -> None:
    chembl_id, arr, out_dir = args
    np.save(out_dir / f"{chembl_id}.npy", arr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(_CORPUS_TSV))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--output-root", default="data/embeddings/v8")
    ap.add_argument("--num-save-workers", type=int, default=8)
    ap.add_argument("--no-amp", action="store_true", help="disable FP16 autocast")
    args = ap.parse_args()

    out_dir = REPO / args.output_root / f"chemberta_{args.corpus}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_unique(args.corpus)
    print(f"[chemberta/{args.corpus}] {len(df)} compostos únicos")

    todo = [(row.chembl_id, row.canonical_smiles) for _, row in df.iterrows()
            if not (out_dir / f"{row.chembl_id}.npy").exists()]
    print(f"  {len(todo)} faltam (skip {len(df) - len(todo)} já existentes)")
    if not todo:
        print("[done] nada a fazer"); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")
    use_amp = (device.type == "cuda") and not args.no_amp

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(
        MODEL_NAME, add_pooling_layer=False, use_safetensors=True,
    ).to(device).eval()

    save_pool: ProcessPoolExecutor | None = None
    if args.num_save_workers > 0:
        save_pool = ProcessPoolExecutor(max_workers=args.num_save_workers)
    save_futures: list = []

    try:
        with torch.no_grad():
            for i in tqdm(range(0, len(todo), args.batch_size), desc="chemberta"):
                chunk = todo[i:i + args.batch_size]
                chids = [cid for cid, _ in chunk]
                smis = [smi for _, smi in chunk]
                enc = tokenizer(smis, return_tensors="pt", padding=True,
                                truncation=True, max_length=MAX_LEN)
                enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    res = model(**enc)
                hs = res.last_hidden_state.float().cpu().numpy()  # (B, L, 384)
                attn = enc["attention_mask"].cpu().numpy()
                for j, cid in enumerate(chids):
                    L = int(attn[j].sum())
                    arr = hs[j, :L].astype(np.float32)
                    if save_pool is not None:
                        save_futures.append(save_pool.submit(_save_one, (cid, arr, out_dir)))
                        if len(save_futures) > 1024:
                            for f in save_futures:
                                f.result()
                            save_futures.clear()
                    else:
                        np.save(out_dir / f"{cid}.npy", arr)
    finally:
        if save_pool is not None:
            for f in save_futures:
                f.result()
            save_pool.shutdown(wait=True)

    print(f"[done] cache → {out_dir}")


if __name__ == "__main__":
    main()
