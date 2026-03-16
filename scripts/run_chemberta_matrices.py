#!/usr/bin/env python3
"""
Generate ChemBERTa-77M-MTR per-token ligand embedding matrices.

Uses the same ChemBERTa model as GraphBAN (DeepChem/ChemBERTa-77M-MTR, 384-d).
Produces per-token matrices stored as {chembl_id}_chemberta_matrix.npy for use
with the CrossAttention pipeline.

Usage:
    python scripts/run_chemberta_matrices.py --dataset non_human --embedding 650M
    python scripts/run_chemberta_matrices.py --dataset human --embedding 8M 150M 650M
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaModel


CHEMBERTA_MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
CHEMBERTA_MAX_LEN = 290

DATASET_PATHS = {
    "human": "tests/datasets/kinase_human_compounds.tsv",
    "non_human": "tests/datasets/kinase_non_human_compounds.tsv",
    "all": "tests/datasets/kinase_all_compounds.tsv",
}

EMBEDDING_BASE = "results/protein_model_benchmark_{dataset}_v2"

SUPPORTED_EMBEDDINGS = {
    "8M": "esm2_t6_8M_UR50D",
    "150M": "esm2_t30_150M_UR50D",
    "650M": "esm2_t33_650M_UR50D",
}


def generate_chemberta_matrices(
    dataset: str,
    embedding_names: list[str],
    device: str = "cuda",
    force: bool = False,
) -> None:
    """Generate ChemBERTa per-token matrices for all unique SMILES in a dataset."""

    dataset_path = Path(DATASET_PATHS[dataset])
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path, sep="\t")
    unique_smiles = df.drop_duplicates(subset="chembl_id")[["chembl_id", "canonical_smiles"]].copy()
    print(f"Dataset: {dataset} — {len(unique_smiles)} unique compounds")

    # Load model once
    print(f"\nLoading ChemBERTa ({CHEMBERTA_MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_NAME)
    model = RobertaModel.from_pretrained(
        CHEMBERTA_MODEL_NAME, add_pooling_layer=False, use_safetensors=True
    )
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    print(f"  Device: {dev}")

    for emb_short in embedding_names:
        emb_full = SUPPORTED_EMBEDDINGS.get(emb_short, emb_short)
        base_dir = Path(EMBEDDING_BASE.format(dataset=dataset)) / emb_full / "build"
        output_dir = base_dir / "chemberta_matrix"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Embedding: {emb_full}  |  Output: {output_dir}")
        print(f"{'='*60}")

        skipped = 0
        generated = 0

        for _, row in tqdm(unique_smiles.iterrows(), total=len(unique_smiles), desc="ChemBERTa"):
            chembl_id = row["chembl_id"]
            smiles = row["canonical_smiles"]
            out_file = output_dir / f"{chembl_id}_chemberta_matrix.npy"

            if out_file.exists() and not force:
                skipped += 1
                continue

            inputs = tokenizer(
                smiles,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=CHEMBERTA_MAX_LEN,
            )
            inputs = {k: v.to(dev) for k, v in inputs.items()}

            with torch.no_grad():
                output = model(**inputs)

            # last_hidden_state: [1, seq_len, 384]
            matrix = output.last_hidden_state[0].cpu().numpy().astype(np.float32)
            np.save(out_file, matrix)
            generated += 1

        print(f"  Generated: {generated}, Skipped (exists): {skipped}")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ChemBERTa-77M-MTR per-token ligand matrices"
    )
    parser.add_argument(
        "--dataset", "-d", required=True, choices=list(DATASET_PATHS.keys()),
        help="Dataset to process"
    )
    parser.add_argument(
        "--embedding", "-e", nargs="+", default=["650M"],
        help="Protein embedding shorthand(s) whose build/ dir to use (default: 650M)"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing matrices"
    )
    args = parser.parse_args()

    generate_chemberta_matrices(
        dataset=args.dataset,
        embedding_names=args.embedding,
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
