#!/usr/bin/env python3
"""
Extract Attention Matrices from ESM-2 Models
=============================================

This script extracts self-attention matrices from ESM-2 models for all proteins
in the dataset. The attention matrices capture how each residue "attends" to
other residues in the sequence.

Output format:
    - Shape: [seq_len, seq_len] (aggregated over heads and using last layer)
    - Files: {seq_id}_attention.npy

Usage:
    python extract_attention_matrices.py --model 8M --dataset non_human
    python extract_attention_matrices.py --model 150M --dataset non_human
    python extract_attention_matrices.py --model 3B --dataset non_human
    python extract_attention_matrices.py --run_all --dataset non_human

Author: DockTKinase Team
Date: January 2025
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import gc

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.build.embeddings.strategies.esm2_strategy import ESM2Strategy


# Model configurations
MODELS = {
    '8M': 'esm2_t6_8M_UR50D',
    '150M': 'esm2_t30_150M_UR50D',
    '3B': 'esm2_t36_3B_UR50D'
}

# Dataset paths
DATASET_PATHS = {
    'human': 'tests/datasets/kinase_human_compounds.tsv',
    'non_human': 'tests/datasets/kinase_non_human_compounds.tsv',
    'all': 'tests/datasets/kinase_all_compounds.tsv'
}

# Results base path
RESULTS_BASE = 'results/protein_model_benchmark_{dataset_type}_v2'


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_sequences(dataset_path: str) -> Dict[str, str]:
    """Load unique protein sequences from dataset."""
    df = pd.read_csv(dataset_path, sep='\t')

    # Get unique seq_id -> sequence mapping
    sequences = {}
    for _, row in df[['seq_id', 'target_sequence']].drop_duplicates().iterrows():
        seq_id = str(row['seq_id'])
        sequences[seq_id] = row['target_sequence']

    return sequences


def extract_attention_matrices(
    model_short: str,
    dataset_type: str,
    aggregate_heads: str = 'mean',
    aggregate_layers: str = 'last',
    force: bool = False
) -> None:
    """
    Extract attention matrices for all proteins using specified ESM-2 model.

    Args:
        model_short: Short model name ('8M', '150M', '3B')
        dataset_type: Dataset type ('human', 'non_human', 'all')
        aggregate_heads: How to aggregate heads ('mean', 'max')
        aggregate_layers: How to aggregate layers ('last', 'mean', 'all')
        force: Force re-extraction even if files exist
    """
    model_name = MODELS[model_short]
    dataset_path = project_root / DATASET_PATHS[dataset_type]
    output_base = project_root / RESULTS_BASE.format(dataset_type=dataset_type)
    output_dir = output_base / model_name / 'build' / 'attention_matrices'

    print(f"\n{'='*60}")
    print(f"EXTRACTING ATTENTION MATRICES")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_type}")
    print(f"Output: {output_dir}")
    print(f"Aggregate heads: {aggregate_heads}")
    print(f"Aggregate layers: {aggregate_layers}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences
    print(f"\nLoading sequences from {dataset_path}...")
    sequences = load_sequences(str(dataset_path))
    print(f"Found {len(sequences)} unique proteins")

    # Check existing files
    if not force:
        existing = set()
        for f in output_dir.glob('*_attention.npy'):
            seq_id = f.stem.replace('_attention', '')
            existing.add(seq_id)

        to_process = {k: v for k, v in sequences.items() if k not in existing}
        print(f"Already extracted: {len(existing)}")
        print(f"To process: {len(to_process)}")

        if len(to_process) == 0:
            print("All attention matrices already extracted. Use --force to re-extract.")
            return
    else:
        to_process = sequences

    # Get device
    device = get_device()
    print(f"Device: {device}")

    # Load ESM-2 model
    print(f"\nLoading ESM-2 model: {model_name}...")
    strategy = ESM2Strategy()
    model, alphabet = strategy.load(model_name, device)
    print("Model loaded successfully")

    # Extract attention matrices
    print(f"\nExtracting attention matrices...")
    errors = []

    for seq_id, sequence in tqdm(to_process.items(), desc="Extracting"):
        output_file = output_dir / f"{seq_id}_attention.npy"

        try:
            # Generate attention matrix
            attention_matrix = strategy.generate_attention_matrix(
                model=model,
                auxiliary_objects=alphabet,
                sequence=sequence,
                device=device,
                aggregate_heads=aggregate_heads,
                aggregate_layers=aggregate_layers
            )

            # Save
            np.save(output_file, attention_matrix.astype(np.float32))

        except Exception as e:
            errors.append((seq_id, str(e)))
            continue

    # Cleanup
    del model, alphabet
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully extracted: {len(to_process) - len(errors)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nErrors:")
        for seq_id, error in errors[:10]:
            print(f"  {seq_id}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Verify output
    sample_files = list(output_dir.glob('*_attention.npy'))[:3]
    if sample_files:
        print(f"\nSample output shapes:")
        for f in sample_files:
            arr = np.load(f)
            print(f"  {f.name}: {arr.shape}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract attention matrices from ESM-2 models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model', type=str, default='150M',
        choices=['8M', '150M', '3B'],
        help='ESM-2 model size (default: 150M)'
    )

    parser.add_argument(
        '--dataset', type=str, default='non_human',
        choices=['human', 'non_human', 'all'],
        help='Dataset type (default: non_human)'
    )

    parser.add_argument(
        '--run_all', action='store_true',
        help='Run extraction for all models (8M, 150M, 3B)'
    )

    parser.add_argument(
        '--aggregate_heads', type=str, default='mean',
        choices=['mean', 'max'],
        help='How to aggregate attention heads (default: mean)'
    )

    parser.add_argument(
        '--aggregate_layers', type=str, default='last',
        choices=['last', 'mean', 'all'],
        help='How to aggregate layers (default: last)'
    )

    parser.add_argument(
        '--force', action='store_true',
        help='Force re-extraction even if files exist'
    )

    args = parser.parse_args()

    if args.run_all:
        for model_short in ['8M', '150M', '3B']:
            extract_attention_matrices(
                model_short=model_short,
                dataset_type=args.dataset,
                aggregate_heads=args.aggregate_heads,
                aggregate_layers=args.aggregate_layers,
                force=args.force
            )
    else:
        extract_attention_matrices(
            model_short=args.model,
            dataset_type=args.dataset,
            aggregate_heads=args.aggregate_heads,
            aggregate_layers=args.aggregate_layers,
            force=args.force
        )


if __name__ == '__main__':
    main()
