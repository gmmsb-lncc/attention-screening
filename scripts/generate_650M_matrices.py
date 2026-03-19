#!/usr/bin/env python3
"""
Generate ESM-2 650M protein matrices AND MoLFormer ligand matrices
in the exact directory layout expected by the benchmark pipeline.

Target structure:
  results/protein_model_benchmark_{dataset}_v2/esm2_t33_650M_UR50D/build/
    ├── protein_matrices/   ← {seq_id}_matrix.npy   (L, 1280)
    └── molformer_matrix/   ← {chembl_id}_molformer_matrix.npy (T, 768)

Usage (on remote GPU server):
  # Non-human dataset
  python scripts/generate_650M_matrices.py --dataset non_human

  # Human dataset
  python scripts/generate_650M_matrices.py --dataset human

  # Both datasets
  python scripts/generate_650M_matrices.py --dataset all

  # Skip ligand matrices (already generated)
  python scripts/generate_650M_matrices.py --dataset non_human --skip-ligands

  # Skip protein matrices (already generated)
  python scripts/generate_650M_matrices.py --dataset non_human --skip-proteins

  # Custom batch size (reduce if OOM on your GPU)
  python scripts/generate_650M_matrices.py --dataset non_human --batch-size 1

Author: DockTKinase Team
Date: March 2026
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ESM_MODEL_NAME = "esm2_t33_650M_UR50D"
ESM_DIM = 1280
MOLFORMER_DIM = 768

DATASET_FILES = {
    "non_human": "tests/datasets/kinase_non_human_compounds.tsv",
    "human": "tests/datasets/kinase_human_compounds.tsv",
}

BASE_RESULTS_DIR = "results/protein_model_benchmark_{dataset}_v2"


# ---------------------------------------------------------------------------
# Protein (ESM-2 650M) matrix generation
# ---------------------------------------------------------------------------

def generate_protein_matrices(
    df: pd.DataFrame,
    output_dir: Path,
    batch_size: int = 4,
    device: str = "auto",
) -> int:
    """Generate per-residue embedding matrices using ESM-2 650M.

    Each protein sequence produces a matrix of shape (L, 1280) where L
    is the sequence length. Saved as ``{seq_id}_matrix.npy``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``seq_id`` and ``sequence`` columns.
    output_dir : Path
        Directory to save protein matrices.
    batch_size : int
        Number of sequences per batch (reduce if GPU OOM).
    device : str
        ``'auto'``, ``'cuda'``, or ``'cpu'``.

    Returns
    -------
    int
        Number of matrices generated.
    """
    import esm

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    print(f"  Device: {dev}")

    # Load model
    print(f"  Loading ESM-2 model: {ESM_MODEL_NAME} ...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
    model = model.eval().to(dev)
    batch_converter = alphabet.get_batch_converter()
    print(f"  Model loaded successfully on {dev}")

    # Deduplicate proteins
    unique = df[["seq_id", "sequence"]].drop_duplicates("seq_id")

    # Skip already-generated
    existing = {f.stem.replace("_matrix", "") for f in output_dir.glob("*_matrix.npy")}
    to_generate = unique[~unique["seq_id"].isin(existing)]
    print(f"  Unique proteins: {len(unique)} | Already exist: {len(existing)} | To generate: {len(to_generate)}")

    if len(to_generate) == 0:
        print("  ✅ All protein matrices already exist — skipping.")
        return 0

    generated = 0
    total = len(to_generate)
    records = list(to_generate.itertuples(index=False))

    for start in range(0, total, batch_size):
        batch_records = records[start : start + batch_size]
        batch_data = [(r.seq_id, r.sequence) for r in batch_records]

        try:
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(dev)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)

            # Extract per-residue representations (layer 33 = last layer for 650M)
            representations = results["representations"][33]  # (B, L+2, 1280)

            for i, (seq_id, seq) in enumerate(batch_data):
                # Remove BOS and EOS tokens: keep positions 1..(L)
                seq_len = len(seq)
                matrix = representations[i, 1 : seq_len + 1, :].cpu().numpy()
                assert matrix.shape == (seq_len, ESM_DIM), (
                    f"Shape mismatch for {seq_id}: {matrix.shape} vs ({seq_len}, {ESM_DIM})"
                )
                np.save(output_dir / f"{seq_id}_matrix.npy", matrix.astype(np.float32))
                generated += 1

            if generated % 50 == 0 or generated == total:
                print(f"  Progress: {generated}/{total} ({100*generated/total:.1f}%)")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"  ⚠️  OOM on batch starting at {start} — processing one-by-one...")
                for r in batch_records:
                    try:
                        _, _, tokens = batch_converter([(r.seq_id, r.sequence)])
                        tokens = tokens.to(dev)
                        with torch.no_grad():
                            res = model(tokens, repr_layers=[33], return_contacts=False)
                        mat = res["representations"][33][0, 1 : len(r.sequence) + 1, :].cpu().numpy()
                        np.save(output_dir / f"{r.seq_id}_matrix.npy", mat.astype(np.float32))
                        generated += 1
                    except Exception as inner_e:
                        print(f"  ❌ Failed on {r.seq_id} (len={len(r.sequence)}): {inner_e}")
                        torch.cuda.empty_cache()
            else:
                raise

    print(f"  ✅ Generated {generated} protein matrices in {output_dir}")
    return generated


# ---------------------------------------------------------------------------
# Ligand (MoLFormer) matrix generation
# ---------------------------------------------------------------------------

def generate_ligand_matrices(
    df: pd.DataFrame,
    output_dir: Path,
    batch_size: int = 32,
    device: str = "auto",
) -> int:
    """Generate per-token MoLFormer embedding matrices for ligands.

    Each SMILES produces a matrix of shape (T, 768) where T is the
    token count. Saved as ``{chembl_id}_molformer_matrix.npy``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``chembl_id`` and ``canonical_smiles`` columns.
    output_dir : Path
        Directory to save ligand matrices.
    batch_size : int
        Number of SMILES per batch.
    device : str
        ``'auto'``, ``'cuda'``, or ``'cpu'``.

    Returns
    -------
    int
        Number of matrices generated.
    """
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    print(f"  Device: {dev}")

    # Try local cache first, then HuggingFace
    model_paths = [
        ("llm/models_cache/molformer/model", "llm/models_cache/molformer/tokenizer"),
        ("ibm/MoLFormer-XL-both-10pct", "ibm/MoLFormer-XL-both-10pct"),
    ]

    tokenizer = None
    mf_model = None
    for model_path, tok_path in model_paths:
        try:
            print(f"  Loading MoLFormer from: {tok_path}")
            tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            mf_model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
            mf_model = mf_model.eval().to(dev)
            print(f"  ✅ MoLFormer loaded")
            break
        except Exception as e:
            print(f"  ⚠️  Failed from {model_path}: {e}")
            continue

    if mf_model is None:
        print("  ❌ Could not load MoLFormer from any source")
        return 0

    # Deduplicate ligands
    unique = df[["chembl_id", "canonical_smiles"]].drop_duplicates("chembl_id")

    # Skip already-generated
    existing = set()
    for f in output_dir.glob("*_matrix.npy"):
        stem = f.stem.replace("_molformer_matrix", "_matrix").replace("_matrix", "")
        existing.add(stem)
    for f in output_dir.glob("*_molformer_matrix.npy"):
        stem = f.stem.replace("_molformer_matrix", "")
        existing.add(stem)

    to_generate = unique[~unique["chembl_id"].isin(existing)]
    print(f"  Unique ligands: {len(unique)} | Already exist: {len(existing)} | To generate: {len(to_generate)}")

    if len(to_generate) == 0:
        print("  ✅ All ligand matrices already exist — skipping.")
        return 0

    generated = 0
    total = len(to_generate)
    records = list(to_generate.itertuples(index=False))

    for start in range(0, total, batch_size):
        batch_records = records[start : start + batch_size]

        for r in batch_records:
            try:
                inputs = tokenizer(
                    r.canonical_smiles,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=512,
                ).to(dev)

                with torch.no_grad():
                    outputs = mf_model(**inputs, output_hidden_states=True)

                matrix = outputs.hidden_states[-1][0].cpu().numpy()  # (T, 768)
                np.save(
                    output_dir / f"{r.chembl_id}_molformer_matrix.npy",
                    matrix.astype(np.float32),
                )
                generated += 1

            except Exception as e:
                print(f"  ❌ Failed on {r.chembl_id}: {e}")

        if generated % 100 == 0 or start + batch_size >= total:
            print(f"  Progress: {generated}/{total} ({100*generated/total:.1f}%)")

    print(f"  ✅ Generated {generated} ligand matrices in {output_dir}")
    return generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate ESM-2 650M protein + MoLFormer ligand matrices for the benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["non_human", "human", "all"],
        default="non_human",
        help="Which dataset(s) to process (default: non_human)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Protein batch size (default: 4; reduce to 1 if GPU OOM)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for inference (default: auto)",
    )
    parser.add_argument(
        "--skip-proteins",
        action="store_true",
        help="Skip protein matrix generation (only generate ligand matrices)",
    )
    parser.add_argument(
        "--skip-ligands",
        action="store_true",
        help="Skip ligand matrix generation (only generate protein matrices)",
    )
    args = parser.parse_args()

    datasets = ["non_human", "human"] if args.dataset == "all" else [args.dataset]

    print("=" * 70)
    print("  ESM-2 650M + MoLFormer Matrix Generator")
    print("=" * 70)
    print(f"  Model:    {ESM_MODEL_NAME} (protein_dim={ESM_DIM})")
    print(f"  Datasets: {datasets}")
    print(f"  Batch:    {args.batch_size}")
    print(f"  Device:   {args.device}")
    print(f"  Proteins: {'SKIP' if args.skip_proteins else 'generate'}")
    print(f"  Ligands:  {'SKIP' if args.skip_ligands else 'generate'}")
    print("=" * 70)

    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"  Processing dataset: {dataset}")
        print(f"{'='*50}")

        # Load data
        data_file = DATASET_FILES.get(dataset)
        if not data_file or not Path(data_file).exists():
            print(f"  ❌ Dataset file not found: {data_file}")
            continue

        df = pd.read_csv(data_file, sep="\t")
        print(f"  Loaded {len(df)} rows from {data_file}")

        # Build output paths
        base = Path(BASE_RESULTS_DIR.format(dataset=dataset))
        build_dir = base / ESM_MODEL_NAME / "build"

        protein_out = build_dir / "protein_matrices"
        ligand_out = build_dir / "molformer_matrix"

        print(f"  Protein output: {protein_out}")
        print(f"  Ligand output:  {ligand_out}")

        # Generate protein matrices
        if not args.skip_proteins:
            print(f"\n  --- Generating protein matrices (ESM-2 650M) ---")
            n_prot = generate_protein_matrices(
                df, protein_out, batch_size=args.batch_size, device=args.device
            )
        else:
            print(f"\n  --- Skipping protein matrices ---")

        # Generate ligand matrices
        if not args.skip_ligands:
            print(f"\n  --- Generating ligand matrices (MoLFormer) ---")
            n_lig = generate_ligand_matrices(
                df, ligand_out, batch_size=32, device=args.device
            )
        else:
            print(f"\n  --- Skipping ligand matrices ---")

    print("\n" + "=" * 70)
    print("  ✅ All done!")
    print(f"  Output structure created at: results/protein_model_benchmark_*_v2/{ESM_MODEL_NAME}/build/")
    print("=" * 70)
    print("\n  To run the benchmark with 650M embeddings:")
    print(f'  EMBEDDING=650M bash run_benchmark.sh')


if __name__ == "__main__":
    main()
