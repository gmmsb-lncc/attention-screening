#!/usr/bin/env python3
"""Count trainable vs frozen params for the DT-Kinase v7 reference model.

Run from the repo root with the project env activated:

    source env/bin/activate
    python3 scripts/count_v7_params.py                  # ESM-2 8M (default)
    python3 scripts/count_v7_params.py --esm 150M
    python3 scripts/count_v7_params.py --esm 650M

Loads the official v7 config (configs/v7.yaml) and instantiates the
InteractionMapCNN used by the benchmark. Prints a markdown-ready table with
trainable/frozen breakdown for the narrative fix of Section 5 (replace
"81x smaller encoder" framing with an honest trainable-params comparison).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmark.levels.level4_cnn import InteractionMapCNN  # noqa: E402

ESM_TOTAL = {"8M": 7_840_000, "150M": 148_800_000, "650M": 652_400_000}
ESM_DIM = {"8M": 320, "150M": 640, "650M": 1280}
MOLFORMER_TOTAL = 46_800_000
MOLFORMER_DIM = 768


def count(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_v7(protein_dim: int) -> InteractionMapCNN:
    """Exactly mirrors configs/v7.yaml."""
    return InteractionMapCNN(
        protein_dim=protein_dim,
        ligand_dim=MOLFORMER_DIM,
        num_heads=8,
        head_dim=32,
        cnn_channels=64,
        dropout=0.35,
        variant="v7",
        mlp_head=False,
        use_adapter=True,
        adapter_bottleneck_prot=256,
        adapter_bottleneck_lig=512,
        adapter_layers=1,
        adapter_self_attn=True,
        contrastive_dim=0,
        cosine_feat=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", choices=list(ESM_DIM), default="8M")
    args = ap.parse_args()

    protein_dim = ESM_DIM[args.esm]
    model = build_v7(protein_dim)

    modules = {
        "prot_adapter": model.prot_adapter,
        "lig_adapter": model.lig_adapter,
        "prot_heads": model.prot_heads,
        "lig_heads": model.lig_heads,
        "cnn": model.cnn,
        "pool": model.pool,
        "classifier": model.classifier,
    }
    row_total = 0
    rows = []
    for name, sub in modules.items():
        n = count(sub)
        row_total += n
        rows.append((name, n))

    trainable_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_backbone = ESM_TOTAL[args.esm] + MOLFORMER_TOTAL

    print(f"# DT-Kinase v7 parameter breakdown (ESM-2 {args.esm})")
    print()
    print(f"Protein dim: {protein_dim}, Ligand dim: {MOLFORMER_DIM}")
    print(f"Config: configs/v7.yaml (num_heads=8, head_dim=32, cnn_channels=64,")
    print(f"        adapter=1/1/self_attn, prot_bottleneck=256, lig_bottleneck=512)")
    print()
    print("| Componente                | Parametros | Status      |")
    print("|---------------------------|-----------:|-------------|")
    print(f"| ESM-2 {args.esm:<18}  | {ESM_TOTAL[args.esm]:>10,} | Congelado   |")
    print(f"| MoLFormer                 | {MOLFORMER_TOTAL:>10,} | Congelado   |")
    for name, n in rows:
        print(f"| {name:<25} | {n:>10,} | Treinavel   |")
    print(f"| **TOTAL TREINAVEL**       | **{trainable_all:>10,}** | Treinavel   |")
    print(f"| **TOTAL CONGELADO**       | **{frozen_backbone:>10,}** | Congelado   |")
    print()

    # Consistency check
    sanity = sum(n for _, n in rows)
    if sanity != trainable_all:
        print(f"WARNING: submodule sum {sanity:,} != model trainable {trainable_all:,}")


if __name__ == "__main__":
    main()
