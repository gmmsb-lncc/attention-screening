"""Re-cache per-token MoLFormer embeddings using a LoRA fine-tuned model.

Loads MoLFormer + LoRA delta from `lora_finetune_molformer.py` output,
runs inference over every unique SMILES across train/val/test splits of
the chosen corpus, and writes per-token .npy matrices to the output
directory in the standard `<chembl_id>_matrix.npy` format.

Usage:
  python3 scripts/v8/recache_molformer_lora.py \
      --lora-dir results/lora/molformer_kinase_v1 \
      --splits-dir scaffolds_splits/output \
      --corpus non_human \
      --out-dir results/lora/molformer_cache_v1/non_human

The output dir is then passed via env to the benchmark:
  BENCHMARK_LEVEL4CNN_LIGAND_CACHE_OVERRIDE=results/lora/molformer_cache_v1/non_human
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Reuse LoRA wrapper from the fine-tune script.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_finetune_molformer import LoRALinear, attach_lora  # noqa: E402


def load_model_with_lora(lora_dir: Path, device: torch.device):
    meta = json.loads((lora_dir / "meta.json").read_text())
    print(f"[recache] meta: {meta}")
    tokenizer = AutoTokenizer.from_pretrained(meta["model_path"], trust_remote_code=True)
    model = AutoModel.from_pretrained(meta["model_path"], trust_remote_code=True)
    model.to(device).eval()
    target_layers = list(range(-meta["top_layers"], 0))
    n_wrapped = attach_lora(
        model,
        target_layer_indices=target_layers,
        target_module_names=meta["target_modules"],
        rank=meta["rank"],
        alpha=meta["alpha"],
    )
    state = torch.load(lora_dir / "lora_state.pt", map_location=device)
    own = dict(model.named_parameters())
    loaded = 0
    for k, v in state.items():
        if k in own:
            with torch.no_grad():
                own[k].copy_(v)
            loaded += 1
    print(f"[recache] loaded LoRA params: {loaded}/{len(state)} (wrapped {n_wrapped} modules)")
    return model, tokenizer, meta


def collect_unique_smiles(splits_dir: Path, corpus: str) -> dict[str, str]:
    """Return {chembl_id: smiles} for all unique molecules across train/val/test."""
    out: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for prefix in (corpus, "universal"):
            tsv = splits_dir / f"{prefix}_{split}.tsv"
            if not tsv.exists():
                continue
            df = pd.read_csv(tsv, sep="\t")
            smiles_col = "smiles" if "smiles" in df.columns else "canonical_smiles"
            id_col = "chembl_id"
            df = df[[id_col, smiles_col]].drop_duplicates()
            for cid, sm in zip(df[id_col], df[smiles_col]):
                out.setdefault(cid, sm)
    return out


@torch.inference_mode()
def encode_one(model, tokenizer, smiles: str, max_length: int, device) -> np.ndarray:
    enc = tokenizer(
        smiles, max_length=max_length, truncation=True,
        padding=False, return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attn)
    # MoLFormer returns (last_hidden_state, ...) — take per-token states.
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return h.squeeze(0).float().cpu().numpy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lora-dir", required=True, type=Path)
    p.add_argument("--splits-dir", required=True, type=Path)
    p.add_argument("--corpus", required=True, choices=["human", "non_human", "all"])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--max-length", type=int, default=202)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, meta = load_model_with_lora(args.lora_dir, device)

    smiles_map = collect_unique_smiles(args.splits_dir, args.corpus)
    print(f"[recache] unique molecules: {len(smiles_map)}")

    n_skip, n_done, n_err = 0, 0, 0
    for cid, sm in tqdm(smiles_map.items(), desc="encoding"):
        out_path = args.out_dir / f"{cid}_matrix.npy"
        if args.skip_existing and out_path.exists():
            n_skip += 1
            continue
        try:
            h = encode_one(model, tokenizer, sm, args.max_length, device)
            np.save(out_path, h.astype(np.float32))
            n_done += 1
        except Exception as e:
            print(f"[recache] FAIL {cid}: {e}")
            n_err += 1

    print(f"[recache] done: {n_done} written, {n_skip} skipped, {n_err} failed")
    summary = {
        "lora_dir": str(args.lora_dir),
        "corpus": args.corpus,
        "out_dir": str(args.out_dir),
        "n_molecules": len(smiles_map),
        "n_done": n_done,
        "n_skipped": n_skip,
        "n_failed": n_err,
        "max_length": args.max_length,
        "lora_meta": meta,
    }
    with open(args.out_dir / "_recache_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[recache] summary: {args.out_dir / '_recache_summary.json'}")


if __name__ == "__main__":
    main()
