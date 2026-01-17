#!/usr/bin/env python3
"""
Generate Morgan fingerprints for ligands.

This script generates 2048-bit Morgan fingerprints (radius=2) from SMILES strings
using RDKit, following the same approach as SMI-TED but with simpler features.

SOLID Principles:
- Single Responsibility: Each class handles one aspect (loading/generating/saving)
- Dependency Inversion: Depends on abstractions (Path, DataFrame)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem


class MorganFingerprintGenerator:
    """Generate Morgan fingerprints from SMILES strings."""

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        use_features: bool = False
    ):
        """
        Initialize Morgan fingerprint generator.

        Args:
            radius: Radius for Morgan algorithm (default: 2)
            n_bits: Number of bits in fingerprint (default: 2048)
            use_features: Use feature-based fingerprints (default: False)
        """
        self.radius = radius
        self.n_bits = n_bits
        self.use_features = use_features

    def smiles_to_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """
        Convert SMILES to Morgan fingerprint.

        Args:
            smiles: SMILES string

        Returns:
            Fingerprint as numpy array or None if invalid SMILES
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.radius,
                nBits=self.n_bits,
                useFeatures=self.use_features
            )
            return np.array(fp, dtype=np.float32)

        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {e}")
            return None

    def generate_batch(self, smiles_list: list) -> np.ndarray:
        """
        Generate fingerprints for a batch of SMILES.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Array of shape (n_samples, n_bits)
        """
        fingerprints = []
        valid_indices = []

        for idx, smiles in enumerate(smiles_list):
            fp = self.smiles_to_fingerprint(smiles)
            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(idx)
            else:
                print(f"Warning: Invalid SMILES at index {idx}: '{smiles}'")

        if not fingerprints:
            raise ValueError("No valid SMILES found in batch")

        return np.vstack(fingerprints), valid_indices


class LigandRepresentationPipeline:
    """Pipeline for generating ligand representations."""

    def __init__(self, input_path: Path, output_path: Path):
        """
        Initialize pipeline.

        Args:
            input_path: Path to ligands CSV file
            output_path: Path to save fingerprints NPY file
        """
        self.input_path = input_path
        self.output_path = output_path
        self.generator = MorganFingerprintGenerator(radius=2, n_bits=2048)

    def load_ligands(self) -> pd.DataFrame:
        """Load ligands from CSV."""
        print(f"Loading ligands from {self.input_path}")
        df = pd.read_csv(self.input_path)
        print(f"Loaded {len(df)} ligands")
        return df

    def generate_fingerprints(self, df: pd.DataFrame) -> np.ndarray:
        """Generate fingerprints for all ligands."""
        print(f"Generating Morgan fingerprints (radius={self.generator.radius}, bits={self.generator.n_bits})")
        
        smiles_list = df['canonical_smiles'].tolist()
        fingerprints, valid_indices = self.generator.generate_batch(smiles_list)

        if len(valid_indices) != len(smiles_list):
            invalid_count = len(smiles_list) - len(valid_indices)
            print(f"Warning: {invalid_count} invalid SMILES were skipped")

        print(f"Generated {len(fingerprints)} fingerprints of shape {fingerprints.shape}")
        return fingerprints

    def save_fingerprints(self, fingerprints: np.ndarray) -> None:
        """Save fingerprints to NPY file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.output_path, fingerprints)
        print(f"Saved fingerprints to {self.output_path}")
        print(f"Shape: {fingerprints.shape}, dtype: {fingerprints.dtype}")

    def run(self) -> None:
        """Execute complete pipeline."""
        df = self.load_ligands()
        fingerprints = self.generate_fingerprints(df)
        self.save_fingerprints(fingerprints)
        
        print("\n=== Summary ===")
        print(f"Total ligands: {len(df)}")
        print(f"Fingerprints generated: {len(fingerprints)}")
        print(f"Fingerprint dimension: {fingerprints.shape[1]}")
        print(f"Success rate: {len(fingerprints)/len(df)*100:.2f}%")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Morgan fingerprints')
    parser.add_argument('--tsv-path', type=str, help='TSV path (unused)')
    parser.add_argument('--results-suffix', type=str, default='results_non_human',
                       help='Results directory suffix')
    parser.add_argument('--embeddings-dir', type=str, help='Embeddings directory (unused)')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / 'data' / args.results_suffix / 'processed' / 'ligands.csv'
    output_path = base_dir / 'data' / args.results_suffix / 'embeddings' / 'morgan_fp.npy'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Ligands file not found: {input_path}")

    print(f"Dataset: {args.results_suffix}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}\n")

    pipeline = LigandRepresentationPipeline(input_path, output_path)
    pipeline.run()

    print("\n✓ Morgan fingerprint generation completed successfully")


if __name__ == '__main__':
    main()
