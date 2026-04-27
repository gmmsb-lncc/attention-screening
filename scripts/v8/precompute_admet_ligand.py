#!/usr/bin/env python3
"""Pre-compute ADMET-AI 41-dim predictions per compound.

Output layout:
    data/embeddings/v8/admet_{corpus}/{chembl_id}.npy  (shape (41,), float32)

Uses the `admet_ai` package (https://github.com/swansonk14/admet_ai)
which wraps Chemprop + TDC models. Runs on GPU when available.

Usage:
    python3 scripts/v8/precompute_admet_ligand.py --corpus non_human
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# ADMET-AI via Lightning hits CUDA init even when we don't need the GPU.
# If the installed torch is compiled against a newer CUDA than the driver
# supports, the Lightning trainer crashes during Accelerator setup.
# Force CPU unless the user explicitly opts in via ADMET_USE_CUDA=1.
if os.environ.get("ADMET_USE_CUDA", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]

_CORPUS_TSV = {
    "human":     REPO / "scaffolds_splits/output/human",
    "non_human": REPO / "scaffolds_splits/output/non_human",
    "all":       REPO / "scaffolds_splits/output/universal",
}

# 41 ADMET-AI default endpoints (actual list queried from package at runtime)
# to validate output dimensionality.


def _load_unique(corpus: str) -> pd.DataFrame:
    stem = _CORPUS_TSV[corpus]
    frames = []
    for split in ("train", "val", "test"):
        path = Path(f"{stem}_{split}.tsv")
        if not path.exists():
            continue
        frames.append(pd.read_csv(path, sep="\t",
                                   usecols=["chembl_id", "canonical_smiles"]))
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["chembl_id"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(_CORPUS_TSV))
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--output-root", default="data/embeddings/v8")
    args = ap.parse_args()

    out_dir = REPO / args.output_root / f"admet_{args.corpus}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_unique(args.corpus)
    print(f"[admet/{args.corpus}] {len(df)} compostos únicos")

    todo = df[~df["chembl_id"].apply(lambda c: (out_dir / f"{c}.npy").exists())].reset_index(drop=True)
    print(f"  {len(todo)} faltam (skip {len(df) - len(todo)} já existentes)")
    if len(todo) == 0:
        print("[done] nada a fazer"); return

    # Lazy import to let script fail with clear message if package missing
    try:
        from admet_ai import ADMETModel  # type: ignore
    except ImportError:
        print("[fatal] admet_ai não instalado. Instale via: pip install admet_ai", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device={device}  (set ADMET_USE_CUDA=1 to force GPU if driver supports it)")
    model = ADMETModel()                     # loads default 41-endpoint bundle

    # Batch predict across all todo SMILES; ADMETModel.predict returns DataFrame.
    smiles = todo["canonical_smiles"].tolist()
    chembl_ids = todo["chembl_id"].tolist()

    for i in tqdm(range(0, len(smiles), args.batch_size), desc="admet"):
        chunk_smi = smiles[i:i + args.batch_size]
        chunk_cid = chembl_ids[i:i + args.batch_size]
        preds = model.predict(smiles=chunk_smi)              # DataFrame indexed by SMILES
        # preds may be DataFrame or dict — normalize
        if isinstance(preds, pd.DataFrame):
            arr = preds.to_numpy().astype(np.float32)
        else:
            arr = np.asarray(preds, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        for j, cid in enumerate(chunk_cid):
            np.save(out_dir / f"{cid}.npy", arr[j])

    sample = np.load(out_dir / f"{chembl_ids[0]}.npy")
    print(f"[done] cache → {out_dir} (sample shape {sample.shape})")


if __name__ == "__main__":
    main()
