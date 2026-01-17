#!/usr/bin/env python3
"""
01_extract_data_regression.py - Extract data for regression ablation study.

This script extracts and organizes data for pChEMBL value prediction (regression).
Uses the same data as classification but keeps continuous pchembl_value as target.

Author: DockTKinase Team
Date: January 2026
"""

import json
import shutil
from pathlib import Path

import pandas as pd
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Source data from classification (already extracted)
CLASSIFICATION_DATA_DIR = BASE_DIR.parent / "classification" / "data" / "processed"

# Original TSV file
TSV_PATH = Path("/media/leon/ssd2tb/docktkinase/tests/datasets/kinase_non_human_compounds.tsv")


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def extract_data():
    """Extract and organize data for regression task."""
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("📊 EXTRACTING DATA FOR REGRESSION ABLATION STUDY")
    print("=" * 70)
    print()
    
    # Check if classification data exists
    if CLASSIFICATION_DATA_DIR.exists():
        print("✅ Found existing classification data, reusing...")
        
        # Copy proteins and ligands files (same as classification)
        for file in ['proteins.csv', 'ligands.csv', 'index_mapping.json']:
            src = CLASSIFICATION_DATA_DIR / file
            dst = PROCESSED_DIR / file
            if src.exists():
                shutil.copy(src, dst)
                print(f"   Copied: {file}")
        
        # Load interactions with pchembl_value (regression target)
        interactions_src = CLASSIFICATION_DATA_DIR / 'interactions.csv'
        if interactions_src.exists():
            interactions = pd.read_csv(interactions_src)
            # Keep pchembl_value as target (drop binary label)
            interactions_reg = interactions[['chembl_id', 'seq_id', 'pchembl_value']].copy()
            interactions_reg.to_csv(PROCESSED_DIR / 'interactions_regression.csv', index=False)
            print(f"   Created: interactions_regression.csv")
    else:
        print("⚠️ Classification data not found, extracting from TSV...")
        
        # Load original TSV
        df = pd.read_csv(TSV_PATH, sep='\t')
        print(f"   Loaded {len(df)} interactions from TSV")
        
        # Extract proteins
        proteins = df[['seq_id', 'seq']].drop_duplicates('seq_id').reset_index(drop=True)
        proteins.to_csv(PROCESSED_DIR / 'proteins.csv', index=False)
        print(f"   Extracted {len(proteins)} unique proteins")
        
        # Extract ligands
        ligands = df[['chembl_id', 'canonical_smiles']].drop_duplicates('chembl_id').reset_index(drop=True)
        ligands.to_csv(PROCESSED_DIR / 'ligands.csv', index=False)
        print(f"   Extracted {len(ligands)} unique ligands")
        
        # Extract interactions with pchembl_value
        interactions = df[['chembl_id', 'seq_id', 'pchembl_value']].copy()
        interactions.to_csv(PROCESSED_DIR / 'interactions_regression.csv', index=False)
        print(f"   Extracted {len(interactions)} interactions for regression")
        
        # Create index mappings
        mappings = {
            'protein_to_idx': {str(seq_id): idx for idx, seq_id in enumerate(proteins['seq_id'])},
            'idx_to_protein': {str(idx): str(seq_id) for idx, seq_id in enumerate(proteins['seq_id'])},
            'ligand_to_idx': {str(chembl_id): idx for idx, chembl_id in enumerate(ligands['chembl_id'])},
            'idx_to_ligand': {str(idx): str(chembl_id) for idx, chembl_id in enumerate(ligands['chembl_id'])}
        }
        with open(PROCESSED_DIR / 'index_mapping.json', 'w') as f:
            json.dump(mappings, f, indent=2)
    
    # Load and print statistics
    interactions = pd.read_csv(PROCESSED_DIR / 'interactions_regression.csv')
    proteins = pd.read_csv(PROCESSED_DIR / 'proteins.csv')
    ligands = pd.read_csv(PROCESSED_DIR / 'ligands.csv')
    
    print()
    print("=" * 70)
    print("📊 DATA STATISTICS")
    print("=" * 70)
    print(f"   Proteins:     {len(proteins):,}")
    print(f"   Ligands:      {len(ligands):,}")
    print(f"   Interactions: {len(interactions):,}")
    print()
    print("   pChEMBL Value Distribution:")
    print(f"      Min:    {interactions['pchembl_value'].min():.2f}")
    print(f"      Max:    {interactions['pchembl_value'].max():.2f}")
    print(f"      Mean:   {interactions['pchembl_value'].mean():.2f}")
    print(f"      Median: {interactions['pchembl_value'].median():.2f}")
    print(f"      Std:    {interactions['pchembl_value'].std():.2f}")
    print()
    print(f"✅ Data saved to: {PROCESSED_DIR}")
    
    return interactions, proteins, ligands


if __name__ == '__main__':
    extract_data()
