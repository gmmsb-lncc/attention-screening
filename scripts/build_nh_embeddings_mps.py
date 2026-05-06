#!/usr/bin/env python3
"""Build ESM-2 8M + MoLFormer per-token matrices for NH split on MPS.

Outputs:
  results/protein_model_benchmark_non_human_v2/esm2_t6_8M_UR50D/build/protein_matrices/{seq_id}_matrix.npy
  results/protein_model_benchmark_non_human_v2/esm2_t6_8M_UR50D/build/molformer_matrix/{chembl_id}_molformer_matrix.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, EsmModel, EsmTokenizer

REPO = Path(__file__).resolve().parent.parent
SPLIT_DIR = REPO / "scaffolds_splits" / "output"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pairs(corpus: str) -> pd.DataFrame:
    parts = []
    for split in ("train", "val", "test"):
        f = SPLIT_DIR / f"{corpus}_{split}.tsv"
        parts.append(pd.read_csv(f, sep="\t"))
    df = pd.concat(parts, ignore_index=True)
    return df


def build_proteins(df: pd.DataFrame, out_dir: Path, device: torch.device, batch_size: int = 4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    uniq = df.drop_duplicates("seq_id")[["seq_id", "seq"]]
    todo = [(sid, seq) for sid, seq in zip(uniq["seq_id"], uniq["seq"])
            if not (out_dir / f"{sid}_matrix.npy").exists()]
    print(f"  Proteins: {len(uniq)} unique, {len(todo)} to compute")
    if not todo:
        return
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").eval().to(device)
    with torch.inference_mode():
        for i in tqdm(range(0, len(todo), batch_size), desc="ESM-2 8M", unit="batch"):
            chunk = todo[i:i + batch_size]
            seqs = [s for _, s in chunk]
            enc = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=1022)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn)
            hidden = out.last_hidden_state
            for j, (sid, seq) in enumerate(chunk):
                mask_j = attn[j].bool().cpu()
                seq_len = max(min(len(seq), int(mask_j.sum().item()) - 2), 1)
                mat = hidden[j, 1: seq_len + 1, :].float().cpu().numpy()
                np.save(out_dir / f"{sid}_matrix.npy", mat)
    del model, tokenizer


def build_ligands(df: pd.DataFrame, out_dir: Path, device: torch.device, batch_size: int = 32) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    uniq = df.drop_duplicates("chembl_id")[["chembl_id", "canonical_smiles"]]
    todo = [(cid, smi) for cid, smi in zip(uniq["chembl_id"], uniq["canonical_smiles"])
            if not (out_dir / f"{cid}_molformer_matrix.npy").exists()]
    print(f"  Ligands: {len(uniq)} unique, {len(todo)} to compute")
    if not todo:
        return
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True).eval().to(device)
    with torch.inference_mode():
        for i in tqdm(range(0, len(todo), batch_size), desc="MoLFormer", unit="batch"):
            chunk = todo[i:i + batch_size]
            smiles = [s for _, s in chunk]
            enc = tokenizer(smiles, padding=True, truncation=True, max_length=512, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn)
            hidden = out.last_hidden_state
            for j, (cid, _) in enumerate(chunk):
                mask = attn[j].bool().cpu()
                mat = hidden[j][mask].float().cpu().numpy()
                np.save(out_dir / f"{cid}_molformer_matrix.npy", mat)
    del model, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="non_human", choices=["non_human", "human"])
    ap.add_argument("--prot-batch", type=int, default=4)
    ap.add_argument("--lig-batch", type=int, default=32)
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    base = REPO / "results" / f"protein_model_benchmark_{args.corpus}_v2" / "esm2_t6_8M_UR50D" / "build"
    df = load_pairs(args.corpus)
    print(f"Corpus={args.corpus} pairs={len(df)} prots={df['seq_id'].nunique()} ligs={df['chembl_id'].nunique()}")

    build_proteins(df, base / "protein_matrices", device, batch_size=args.prot_batch)
    build_ligands(df, base / "molformer_matrix", device, batch_size=args.lig_batch)
    print("Done.")


if __name__ == "__main__":
    main()
