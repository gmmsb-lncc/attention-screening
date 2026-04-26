"""LoRA fine-tuning of MoLFormer top-K layers via continued MLM pretraining.

Trains a small low-rank adaptation (Hu et al. 2022) on top of the frozen
MoLFormer-XL-both-10pct backbone. Only matrices A,B (rank=8 default) are
trainable; original weights preserved exactly. After training, save:
  - lora_state.pt : dict {layer_idx: {"A": Tensor, "B": Tensor}}
  - meta.json     : config (rank, alpha, target_layers, target_modules)

Usage:
  python3 scripts/v8/lora_finetune_molformer.py \
      --train-tsv scaffolds_splits/output/non_human_train.tsv \
      --output-dir results/lora/molformer_kinase_v1 \
      --rank 8 --alpha 16 --top-layers 2 --epochs 10 --batch-size 32

Notes:
- Self-supervised MLM only (no labels) → no leakage.
- Train SMILES only (test SMILES never seen by encoder during FT).
- Target layers: top-K of MoLFormer encoder; modules: q,k,v,o projections.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


# ---------------------------------------------------------------------------
# LoRA module
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps an nn.Linear with a low-rank delta: y = Wx + α/r · (BA)x.

    Original weight stays frozen. Only A,B trainable. B init zero so
    output equals base layer at t=0.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_f = base.in_features
        out_f = base.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora_out = torch.nn.functional.linear(x, self.lora_A)
        lora_out = torch.nn.functional.linear(lora_out, self.lora_B)
        return out + self.scaling * lora_out


def attach_lora(
    model: nn.Module,
    target_layer_indices: list[int],
    target_module_names: list[str],
    rank: int = 8,
    alpha: int = 16,
) -> int:
    """Attach LoRA to specified Linear modules in specified encoder layers.

    Returns count of wrapped modules. Modifies model in place.
    """
    n_wrapped = 0
    # Find encoder layers — robust across MoLFormer/RoBERTa-like topologies.
    layers = None
    for path in (
        ("molformer", "encoder", "layer"),
        ("roberta", "encoder", "layer"),
        ("encoder", "layer"),
        ("base_model", "encoder", "layer"),
    ):
        cur = model
        for p in path:
            cur = getattr(cur, p, None)
            if cur is None:
                break
        if cur is not None:
            layers = cur
            break
    if layers is None:
        raise RuntimeError(
            "Could not locate encoder.layer in model. Inspect model.children().")

    n_total = len(layers)
    for idx in target_layer_indices:
        if idx < 0:
            idx = n_total + idx
        layer = layers[idx]
        for module_name in target_module_names:
            module = layer
            parts = module_name.split(".")
            for p in parts[:-1]:
                module = getattr(module, p)
            leaf = getattr(module, parts[-1])
            if not isinstance(leaf, nn.Linear):
                continue
            wrapped = LoRALinear(leaf, rank=rank, alpha=alpha)
            setattr(module, parts[-1], wrapped)
            n_wrapped += 1
    return n_wrapped


def collect_lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract A,B tensors from all LoRALinear modules, keyed by full param name."""
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if ("lora_A" in name or "lora_B" in name) and param.requires_grad
    }


# ---------------------------------------------------------------------------
# Dataset (MLM)
# ---------------------------------------------------------------------------

class SMILESMLMDataset(Dataset):
    def __init__(
        self, smiles: list[str], tokenizer, max_length: int = 202,
    ):
        self.smiles = smiles
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tok(
            self.smiles[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def mlm_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    mask_prob: float = 0.15,
):
    """Apply BERT-style MLM masking. Returns (masked_ids, labels)."""
    labels = input_ids.clone()
    rand = torch.rand_like(input_ids, dtype=torch.float)
    special_ids = set(tokenizer.all_special_ids)
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        special_mask |= input_ids == sid
    mask = (rand < mask_prob) & ~special_mask & (attention_mask == 1)
    labels[~mask] = -100
    masked = input_ids.clone()
    rand2 = torch.rand_like(input_ids, dtype=torch.float)
    replace_mask = mask & (rand2 < 0.8)
    random_mask = mask & (rand2 >= 0.8) & (rand2 < 0.9)
    if tokenizer.mask_token_id is not None:
        masked[replace_mask] = tokenizer.mask_token_id
    rand_tok = torch.randint(
        low=5, high=tokenizer.vocab_size, size=input_ids.shape, dtype=input_ids.dtype,
    )
    masked[random_mask] = rand_tok[random_mask]
    return masked, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-tsv", required=True, type=Path)
    p.add_argument("--val-tsv", type=Path, default=None)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model-path", default="ibm/MoLFormer-XL-both-10pct")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--top-layers", type=int, default=2,
                   help="Apply LoRA to the top-K encoder layers.")
    p.add_argument("--target-modules", default="attention.self.query,attention.self.key,attention.self.value,attention.output.dense",
                   help="Comma-separated dot-paths within each layer.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max-length", type=int, default=202)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lora-ft] device={device}")

    print(f"[lora-ft] loading {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_path, trust_remote_code=True,
    )
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    target_layer_indices = list(range(-args.top_layers, 0))
    target_modules = args.target_modules.split(",")
    n_wrapped = attach_lora(
        model,
        target_layer_indices=target_layer_indices,
        target_module_names=target_modules,
        rank=args.rank,
        alpha=args.alpha,
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[lora-ft] wrapped {n_wrapped} Linear modules with LoRA(rank={args.rank}, α={args.alpha})")
    print(f"[lora-ft] trainable params: {n_train:,} / {n_total:,} ({100*n_train/n_total:.3f}%)")

    df = pd.read_csv(
        args.train_tsv, sep="\t",
        compression="gzip" if str(args.train_tsv).endswith(".gz") else None,
    )
    smiles_col = "smiles" if "smiles" in df.columns else "canonical_smiles"
    train_smiles = df[smiles_col].drop_duplicates().tolist()
    print(f"[lora-ft] train SMILES (unique): {len(train_smiles)}")

    train_ds = SMILESMLMDataset(train_smiles, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    n_steps = args.epochs * len(train_loader)
    n_warmup = int(args.warmup_ratio * n_steps)

    def lr_lambda(step: int) -> float:
        if step < n_warmup:
            return step / max(1, n_warmup)
        progress = (step - n_warmup) / max(1, n_steps - n_warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        running, count = 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in bar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            masked, labels = mlm_mask(input_ids, attn, tokenizer, args.mask_prob)
            labels = labels.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            ctx = (
                torch.amp.autocast("cuda", dtype=torch.float16)
                if scaler is not None else torch.amp.autocast("cpu", enabled=False)
            )
            with ctx:
                out = model(input_ids=masked, attention_mask=attn, labels=labels)
                loss = out.loss
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            scheduler.step()
            global_step += 1
            running += float(loss.item())
            count += 1
            bar.set_postfix({"loss": f"{running/max(count,1):.4f}"})
        print(f"[lora-ft] epoch {epoch+1}: avg_loss={running/max(count,1):.4f}")

    state = collect_lora_state(model)
    out_state = args.output_dir / "lora_state.pt"
    torch.save(state, out_state)
    meta = {
        "model_path": args.model_path,
        "rank": args.rank,
        "alpha": args.alpha,
        "top_layers": args.top_layers,
        "target_modules": target_modules,
        "n_wrapped": n_wrapped,
        "n_trainable": n_train,
        "n_total_model": n_total,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "mask_prob": args.mask_prob,
        "n_train_smiles": len(train_smiles),
        "train_tsv": str(args.train_tsv),
        "seed": args.seed,
    }
    with open(args.output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[lora-ft] saved {out_state} ({sum(v.numel() for v in state.values()):,} params)")
    print(f"[lora-ft] saved {args.output_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
