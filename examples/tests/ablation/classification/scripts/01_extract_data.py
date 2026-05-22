#!/usr/bin/env python3
"""
Extract and organize data from TSV file for ablation study.

This script extracts unique proteins and ligands from the kinase dataset,
creating separate CSV files and index mappings for efficient processing.

SOLID Principles:
- Single Responsibility: Each class/function has one clear purpose
- Open/Closed: Extensible through configuration, closed for modification
- Liskov Substitution: Not applicable (no inheritance)
- Interface Segregation: Not applicable (no interfaces)
- Dependency Inversion: Depends on abstractions (pathlib, pandas)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


class DataExtractor:
    """Extract and organize protein-ligand interaction data."""

    def __init__(self, tsv_path: Path, output_dir: Path):
        """
        Initialize data extractor.

        Args:
            tsv_path: Path to input TSV file
            output_dir: Directory for output files
        """
        self.tsv_path = tsv_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load TSV file into DataFrame."""
        print(f"Loading data from {self.tsv_path}")
        df = pd.read_csv(self.tsv_path, sep='\t')
        print(f"Loaded {len(df)} interactions")
        return df

    def extract_proteins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique proteins with their sequences."""
        proteins = df[['seq_id', 'seq']].drop_duplicates('seq_id')
        proteins = proteins.reset_index(drop=True)
        print(f"Extracted {len(proteins)} unique proteins")
        return proteins

    def extract_ligands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique ligands with their SMILES."""
        ligands = df[['chembl_id', 'canonical_smiles']].drop_duplicates('chembl_id')
        ligands = ligands.reset_index(drop=True)
        print(f"Extracted {len(ligands)} unique ligands")
        return ligands

    def extract_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract interaction data with labels."""
        interactions = df[['chembl_id', 'seq_id', 'pchembl_value']].copy()
        # Binary label: 1 if pchembl_value >= 6.5 (active), 0 otherwise (inactive)
        interactions['label'] = (interactions['pchembl_value'] >= 6.5).astype(int)
        interactions = interactions.reset_index(drop=True)
        print(f"Extracted {len(interactions)} interactions")
        return interactions

    def create_index_mappings(
        self,
        proteins: pd.DataFrame,
        ligands: pd.DataFrame
    ) -> Dict[str, Dict[str, int]]:
        """
        Create bidirectional index mappings for proteins and ligands.

        Args:
            proteins: DataFrame with protein data
            ligands: DataFrame with ligand data

        Returns:
            Dictionary with protein and ligand mappings
        """
        mappings = {
            'protein_to_idx': {
                str(seq_id): idx for idx, seq_id in enumerate(proteins['seq_id'])
            },
            'idx_to_protein': {
                str(idx): str(seq_id) for idx, seq_id in enumerate(proteins['seq_id'])
            },
            'ligand_to_idx': {
                str(chembl_id): idx for idx, chembl_id in enumerate(ligands['chembl_id'])
            },
            'idx_to_ligand': {
                str(idx): str(chembl_id) for idx, chembl_id in enumerate(ligands['chembl_id'])
            }
        }
        return mappings

    def save_data(
        self,
        proteins: pd.DataFrame,
        ligands: pd.DataFrame,
        interactions: pd.DataFrame,
        mappings: Dict[str, Dict[str, int]]
    ) -> None:
        """Save extracted data to CSV and JSON files."""
        # Save CSV files
        proteins_path = self.output_dir / 'proteins.csv'
        ligands_path = self.output_dir / 'ligands.csv'
        interactions_path = self.output_dir / 'interactions.csv'

        proteins.to_csv(proteins_path, index=False)
        ligands.to_csv(ligands_path, index=False)
        interactions.to_csv(interactions_path, index=False)

        print(f"\nSaved files:")
        print(f"  - {proteins_path} ({len(proteins)} proteins)")
        print(f"  - {ligands_path} ({len(ligands)} ligands)")
        print(f"  - {interactions_path} ({len(interactions)} interactions)")

        # Save index mappings
        mappings_path = self.output_dir / 'index_mapping.json'
        with open(mappings_path, 'w') as f:
            json.dump(mappings, f, indent=2)
        print(f"  - {mappings_path}")

    def extract_and_save(self) -> None:
        """Execute complete extraction pipeline."""
        df = self.load_data()
        
        proteins = self.extract_proteins(df)
        ligands = self.extract_ligands(df)
        interactions = self.extract_interactions(df)
        mappings = self.create_index_mappings(proteins, ligands)
        
        self.save_data(proteins, ligands, interactions, mappings)
        
        # Print summary statistics
        print("\n=== Summary ===")
        print(f"Total proteins: {len(proteins)}")
        print(f"Total ligands: {len(ligands)}")
        print(f"Total interactions: {len(interactions)}")
        print(f"Active interactions: {interactions['label'].sum()}")
        print(f"Inactive interactions: {len(interactions) - interactions['label'].sum()}")
        print(f"Activity ratio: {interactions['label'].mean():.3f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Extract data for ablation study')
    parser.add_argument('--tsv-path', type=str, 
                       default='${PROJECT_ROOT}/tests/datasets/kinase_non_human_compounds.tsv',
                       help='Path to input TSV file')
    parser.add_argument('--results-suffix', type=str, default='results_non_human',
                       help='Suffix for results directory (e.g., results_non_human, results_human)')
    parser.add_argument('--embeddings-dir', type=str, help='Path to embeddings directory (unused here)')
    
    args = parser.parse_args()
    
    # Paths
    tsv_path = Path(args.tsv_path)
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'data' / args.results_suffix / 'processed'

    # Validate input
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    print(f"Dataset: {args.results_suffix}")
    print(f"TSV: {tsv_path}")
    print(f"Output: {output_dir}\n")

    # Extract data
    extractor = DataExtractor(tsv_path, output_dir)
    extractor.extract_and_save()
    
    print("\n✓ Data extraction completed successfully")


if __name__ == '__main__':
    main()
