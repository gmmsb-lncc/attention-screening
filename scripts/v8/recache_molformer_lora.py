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
    model.to(device)
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
def encode_batch(
    model, tokenizer, smiles_list: list[str], max_length: int, device,
    amp_dtype=torch.bfloat16, use_cuda: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a batch. Returns (per_token_hidden, attention_mask).

    Per-token hidden trimmed to true sequence length per item later.
    Output dtype: float32 to preserve numerics in cache files.
    """
    enc = tokenizer(
        smiles_list, max_length=max_length, truncation=True,
        padding=True, return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device, non_blocking=True)
    attn = enc["attention_mask"].to(device, non_blocking=True)
    if use_cuda and amp_dtype != torch.float32:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            out = model(input_ids=input_ids, attention_mask=attn)
    else:
        out = model(input_ids=input_ids, attention_mask=attn)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return h.float().cpu().numpy(), attn.cpu().numpy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lora-dir", required=True, type=Path)
    p.add_argument("--splits-dir", required=True, type=Path)
    p.add_argument("--corpus", required=True, choices=["human", "non_human", "all"])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--max-length", type=int, default=202)
    p.add_argument("--batch-size", type=int, default=64,
                   help="GPU batch size for inference. RTX 4090 24GB handles 64 easily.")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                     "fp32": torch.float32}[args.precision]
        print(f"[recache] device={device}, precision={args.precision}")
    else:
        amp_dtype = torch.float32
        print(f"[recache] device=CPU (precision forced to fp32)")
    model, tokenizer, meta = load_model_with_lora(args.lora_dir, device)

    smiles_map = collect_unique_smiles(args.splits_dir, args.corpus)
    print(f"[recache] unique molecules: {len(smiles_map)}")

    items = list(smiles_map.items())
    if args.skip_existing:
        before = len(items)
        items = [(cid, sm) for cid, sm in items
                 if not (args.out_dir / f"{cid}_matrix.npy").exists()]
        print(f"[recache] skipping {before - len(items)} already cached")
    n_skip = len(smiles_map) - len(items)

    n_done, n_err = 0, 0
    bar = tqdm(range(0, len(items), args.batch_size), desc="encoding (batched)")
    for start in bar:
        chunk = items[start:start + args.batch_size]
        cids = [cid for cid, _ in chunk]
        smis = [sm for _, sm in chunk]
        try:
            hidden, attn = encode_batch(
                model, tokenizer, smis, args.max_length, device,
                amp_dtype=amp_dtype, use_cuda=use_cuda,
            )
            for i, cid in enumerate(cids):
                # Trim to true seq length per item (drop pad rows)
                seq_len = int(attn[i].sum())
                mat = hidden[i, :seq_len, :].astype(np.float32)
                np.save(args.out_dir / f"{cid}_matrix.npy", mat)
                n_done += 1
        except Exception as e:
            print(f"[recache] FAIL batch starting {cids[0]}: {e}")
            n_err += len(chunk)
        bar.set_postfix({"done": n_done, "fail": n_err})

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
