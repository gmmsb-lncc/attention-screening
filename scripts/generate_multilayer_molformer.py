#!/usr/bin/env python3
"""
Generate multi-layer MoLFormer matrices (all hidden states, compressed).

For each molecule, saves a compressed .npz file containing ALL hidden
layer outputs from MoLFormer.  This enables multi-layer feature extraction
in the benchmark pipeline (e.g., concatenating representations from
different layers for richer ligand descriptors).

Output format per molecule:
    {chembl_id}_multilayer.npz with keys:
        "layer_0" : [seq_len, 768]  (embedding layer)
        "layer_1" : [seq_len, 768]  (transformer block 1)
        ...
        "layer_6" : [seq_len, 768]  (transformer block 6 — last hidden)

Usage:
    python scripts/generate_multilayer_molformer.py \\
        --dataset tests/datasets/kinase_non_human_compounds.tsv \\
        --output_base results/protein_model_benchmark_non_human_v2 \\
        --models esm2_t6_8M_UR50D

    # Or for all models:
    python scripts/generate_multilayer_molformer.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_molformer(device: torch.device):
    """Load MoLFormer model and tokenizer from local cache or HuggingFace."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    # Try local cache first.
    script_dir = Path(__file__).resolve().parent.parent
    cache_dir = script_dir / "llm" / "models_cache" / "molformer"
    tokenizer_path = cache_dir / "tokenizer"
    model_path = cache_dir / "model"

    if tokenizer_path.exists() and model_path.exists():
        tqdm.write(f"  Loading MoLFormer from cache: {cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), trust_remote_code=True
        )
        model = AutoModelForMaskedLM.from_pretrained(
            str(model_path), trust_remote_code=True
        )
    else:
        hub_name = "DeepChem/MoLFormer-c3-1.1B"
        tqdm.write(f"  Downloading MoLFormer from HuggingFace: {hub_name}")
        tokenizer = AutoTokenizer.from_pretrained(hub_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(hub_name, trust_remote_code=True)

    model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    tqdm.write(
        f"  MoLFormer loaded: {n_layers} transformer layers, "
        f"hidden_dim={model.config.hidden_size}"
    )
    return tokenizer, model, n_layers


def generate_multilayer_matrices(
    dataset_path: str,
    output_base: str,
    protein_models: list[str],
    *,
    force: bool = False,
    max_length: int = 512,
) -> None:
    """Generate compressed multi-layer MoLFormer matrices for all molecules."""

    # --- Load dataset ---
    tqdm.write(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, sep="\t")
    smiles_map = dict(zip(df["chembl_id"], df["canonical_smiles"]))
    tqdm.write(f"  {len(smiles_map)} unique molecules")

    # --- Load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, n_layers = load_molformer(device)

    # --- Process each protein model directory ---
    for prot_model in protein_models:
        model_dir = Path(output_base) / prot_model
        if not model_dir.exists():
            tqdm.write(f"  WARNING: Directory not found: {model_dir}")
            continue

        # Output directory for multi-layer matrices
        multilayer_dir = model_dir / "build" / "molformer_multilayer"
        multilayer_dir.mkdir(parents=True, exist_ok=True)

        # Determine which molecules to process from existing embeddings.
        # Use the existing molformer_matrix directory to find IDs.
        molformer_dir = model_dir / "build" / "molformer_matrix"
        if not molformer_dir.exists():
            tqdm.write(f"  WARNING: No molformer_matrix dir: {molformer_dir}")
            # Fall back to all molecules in smiles_map
            chembl_ids = list(smiles_map.keys())
        else:
            chembl_ids = []
            for f in sorted(molformer_dir.glob("*.npy")):
                cid = f.stem.replace("_molformer_matrix", "").replace("_matrix", "")
                if cid in smiles_map:
                    chembl_ids.append(cid)

        tqdm.write(
            f"\n  Processing {prot_model}: "
            f"{len(chembl_ids)} molecules → {multilayer_dir}"
        )

        processed = skipped = errors = 0
        pbar = tqdm(chembl_ids, desc=f"    {prot_model}", unit="mol", leave=False)

        for chembl_id in pbar:
            out_path = multilayer_dir / f"{chembl_id}_multilayer.npz"

            if out_path.exists() and not force:
                skipped += 1
                continue

            smiles = smiles_map[chembl_id]
            try:
                inputs = tokenizer(
                    smiles,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_length,
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)

                # hidden_states is a tuple of (n_layers + 1) tensors:
                #   [0] = embedding layer output
                #   [1..n_layers] = transformer block outputs
                hidden_states = outputs.hidden_states

                # Save all layers compressed (removing batch dim, special tokens).
                layer_dict = {}
                for layer_idx, hs in enumerate(hidden_states):
                    # Remove batch dim, strip [CLS] (first) and [SEP] (last) tokens.
                    tokens = hs[0, 1:-1, :].cpu().numpy().astype(np.float16)
                    layer_dict[f"layer_{layer_idx}"] = tokens

                np.savez_compressed(out_path, **layer_dict)
                processed += 1

            except Exception as exc:
                tqdm.write(f"    ERROR {chembl_id}: {exc}")
                errors += 1

            pbar.set_postfix(ok=processed, skip=skipped, err=errors)

        tqdm.write(
            f"  Done ({prot_model}): {processed} generated, "
            f"{skipped} skipped, {errors} errors"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-layer MoLFormer matrices (compressed)"
    )
    parser.add_argument(
        "--dataset",
        default="tests/datasets/kinase_non_human_compounds.tsv",
        help="Path to dataset TSV with chembl_id and canonical_smiles columns",
    )
    parser.add_argument(
        "--output_base",
        default="results/protein_model_benchmark_non_human_v2",
        help="Base directory containing protein model subdirectories",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "esm2_t6_8M_UR50D",
            "esm2_t30_150M_UR50D",
            "esm2_t33_650M_UR50D",
        ],
        help="Protein model names to process",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate existing files"
    )
    args = parser.parse_args()

    generate_multilayer_matrices(
        dataset_path=args.dataset,
        output_base=args.output_base,
        protein_models=args.models,
        force=args.force,
    )


if __name__ == "__main__":
    # Auto-detect project root
    script_dir = Path(__file__).resolve().parent.parent
    os.chdir(script_dir)
    main()
