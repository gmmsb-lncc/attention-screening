#!/usr/bin/env python3
"""
Generate One-Hot Encoding for proteins based on protein index.

This script generates one-hot encoded vectors for proteins where each unique
protein is represented by its index position. For N unique proteins, each
protein gets a vector of N dimensions with 1 at its index position and 0s elsewhere.

Example:
- 299 unique proteins → 299 dimensions per protein
- Protein at index 0: [1, 0, 0, ..., 0]
- Protein at index 1: [0, 1, 0, ..., 0]
- Protein at index 298: [0, 0, 0, ..., 1]
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List


class ProteinOneHotEncoder:
    """Calculate Amino Acid Composition (AAC) features."""

class ProteinOneHotEncoder:
    """One-hot encoder for proteins based on their index."""

    def __init__(self, n_proteins: int):
        """
        Initialize one-hot encoder.

        Args:
            n_proteins: Total number of unique proteins
        """
        self.n_proteins = n_proteins

    def encode(self, protein_index: int) -> np.ndarray:
        """
        Create one-hot encoding for a single protein.

        Args:
            protein_index: Index of the protein (0 to n_proteins-1)

        Returns:
            One-hot vector of length n_proteins
        """
        encoding = np.zeros(self.n_proteins, dtype=np.float32)
        encoding[protein_index] = 1.0
        return encoding

    def encode_batch(self, protein_indices: List[int]) -> np.ndarray:
        """
        Create one-hot encodings for a batch of proteins.

        Args:
            protein_indices: List of protein indices

        Returns:
            Array of shape (n_proteins, n_proteins) with one-hot encodings
        """
        encodings = np.zeros((len(protein_indices), self.n_proteins), dtype=np.float32)
        for i, idx in enumerate(protein_indices):
            encodings[i, idx] = 1.0
        return encodings


class ProteinRepresentationPipeline:
    """Pipeline for generating protein one-hot representations."""

    def __init__(self, input_path: Path, output_path: Path):
        """
        Initialize pipeline.

        Args:
            input_path: Path to proteins CSV file
            output_path: Path to save encodings NPY file
        """
        self.input_path = input_path
        self.output_path = output_path

    def load_proteins(self) -> pd.DataFrame:
        """Load proteins from CSV."""
        print(f"Loading proteins from {self.input_path}")
        df = pd.read_csv(self.input_path)
        print(f"Loaded {len(df)} protein entries")
        print(f"Unique sequences: {df['seq'].nunique()}")
        print(f"Duplicate sequences: {len(df) - df['seq'].nunique()}")
        return df

    def generate_encodings(self, df: pd.DataFrame) -> np.ndarray:
        """Generate one-hot encodings for unique protein sequences."""
        # Get unique sequences with their first occurrence index
        unique_seqs = df['seq'].unique()
        n_unique = len(unique_seqs)
        
        print(f"\nGenerating one-hot encodings based on {n_unique} unique sequences")
        print(f"One-hot dimension: {n_unique}")
        
        # Create mapping from sequence to one-hot index
        seq_to_idx = {seq: idx for idx, seq in enumerate(unique_seqs)}
        
        # Create encoder with number of unique sequences
        encoder = ProteinOneHotEncoder(n_unique)
        
        # Encode each protein by mapping its sequence to the unique index
        protein_indices = [seq_to_idx[seq] for seq in df['seq']]
        encodings = encoder.encode_batch(protein_indices)

        print(f"Generated {len(encodings)} one-hot encodings of shape {encodings.shape}")
        return encodings

    def save_encodings(self, encodings: np.ndarray) -> None:
        """Save encodings to NPY file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.output_path, encodings)
        print(f"Saved encodings to {self.output_path}")
        print(f"Shape: {encodings.shape}, dtype: {encodings.dtype}")

    def run(self) -> None:
        """Execute complete pipeline."""
        df = self.load_proteins()
        encodings = self.generate_encodings(df)
        self.save_encodings(encodings)
        
        print("\n=== Summary ===")
        print(f"Total protein entries: {len(df)}")
        print(f"Unique sequences: {encodings.shape[1]}")
        print(f"Encodings generated: {len(encodings)}")
        print(f"Encoding dimension: {encodings.shape[1]} (one-hot based on unique sequences)")
        print(f"Example: First unique sequence = [1, 0, 0, ..., 0]")
        print(f"Example: Last unique sequence = [0, 0, 0, ..., 1]")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate protein one-hot encodings')
    parser.add_argument('--tsv-path', type=str, help='TSV path (unused)')
    parser.add_argument('--results-suffix', type=str, default='results_non_human',
                       help='Results directory suffix')
    parser.add_argument('--embeddings-dir', type=str, help='Embeddings directory (unused)')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / 'data' / args.results_suffix / 'processed' / 'proteins.csv'
    output_path = base_dir / 'data' / args.results_suffix / 'embeddings' / 'protein_onehot.npy'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Proteins file not found: {input_path}")

    print(f"Dataset: {args.results_suffix}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}\n")

    pipeline = ProteinRepresentationPipeline(input_path, output_path)
    pipeline.run()

    print("\n✓ Protein one-hot encoding generation completed successfully")


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
