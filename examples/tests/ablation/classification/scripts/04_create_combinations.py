#!/usr/bin/env python3
"""
Generate all 4 representation combinations for ablation study.

This script creates combinations by loading existing ESM-2/SMI-TED embeddings
and newly generated Morgan FP and Protein One-Hot representations, then combines them
for each interaction.

Combinations:
- C1: ESM-2 + SMI-TED (most complex, learned representations)
- C2: ESM-2 + Morgan FP (mixed: learned protein + handcrafted ligand)
- C3: Protein One-Hot + SMI-TED (mixed: simple protein + learned ligand)
- C4: Protein One-Hot + Morgan FP (simplest, all handcrafted)

SOLID Principles:
- Single Responsibility: Each function handles one loading type
- Dependency Inversion: Depends on abstractions (Path, Dict)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


def load_interactions(path: Path) -> pd.DataFrame:
    """Load interactions CSV."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} interactions")
    return df


def load_index_mapping(path: Path) -> Dict:
    """Load index mappings JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def load_esm_proteins(model_path: Path) -> Dict[str, np.ndarray]:
    """Load ESM-2 protein embeddings from proteins directory."""
    proteins_dir = model_path / 'build' / 'proteins'
    print(f"  Loading ESM-2 proteins from {proteins_dir.name}")
    
    protein_embeddings = {}
    for prot_file in proteins_dir.glob('*_embedding.npy'):
        seq_id = prot_file.stem.replace('_embedding', '')
        protein_embeddings[seq_id] = np.load(prot_file)
    
    print(f"    {len(protein_embeddings)} proteins")
    return protein_embeddings


def load_smited_ligands(model_path: Path) -> Dict[str, np.ndarray]:
    """Load SMI-TED ligand embeddings from ligand_matrices directory."""
    ligands_dir = model_path / 'build' / 'ligand_matrices'
    print(f"  Loading SMI-TED ligands from {ligands_dir.name}")
    
    ligand_embeddings = {}
    for lig_file in ligands_dir.glob('*_matrix.npy'):
        chembl_id = lig_file.stem.replace('_matrix', '')
        emb = np.load(lig_file)
        ligand_embeddings[chembl_id] = emb.flatten()
    
    print(f"    {len(ligand_embeddings)} ligands")
    return ligand_embeddings


def load_onehot_proteins(onehot_path: Path, proteins_csv: Path) -> Dict[str, np.ndarray]:
    """Load one-hot protein encodings."""
    print(f"  Loading one-hot proteins from {onehot_path.name}")
    onehot_array = np.load(onehot_path)
    proteins_df = pd.read_csv(proteins_csv)
    
    protein_embeddings = {}
    for idx, row in proteins_df.iterrows():
        protein_embeddings[str(row['seq_id'])] = onehot_array[idx]
    
    print(f"    {len(protein_embeddings)} proteins")
    return protein_embeddings


def load_morgan_ligands(morgan_path: Path, ligands_csv: Path) -> Dict[str, np.ndarray]:
    """Load Morgan fingerprints."""
    print(f"  Loading Morgan FP ligands from {morgan_path.name}")
    morgan_array = np.load(morgan_path)
    ligands_df = pd.read_csv(ligands_csv)
    
    ligand_embeddings = {}
    for idx, row in ligands_df.iterrows():
        ligand_embeddings[str(row['chembl_id'])] = morgan_array[idx]
    
    print(f"    {len(ligand_embeddings)} ligands")
    return ligand_embeddings


def combine_and_save(
    df: pd.DataFrame,
    protein_repr: Dict[str, np.ndarray],
    ligand_repr: Dict[str, np.ndarray],
    output_path: Path,
    combination_name: str
) -> None:
    """
    Combine protein and ligand representations for each interaction and save.
    
    Args:
        df: Interactions DataFrame
        protein_repr: Dict of protein embeddings {seq_id: embedding}
        ligand_repr: Dict of ligand embeddings {chembl_id: embedding}
        output_path: Output directory
        combination_name: Name for output files
    """
    combined_features = []
    labels = []
    skipped = 0
    
    for _, row in df.iterrows():
        protein_id = str(row['seq_id'])
        ligand_id = str(row['chembl_id'])
        
        # Skip if missing
        if protein_id not in protein_repr or ligand_id not in ligand_repr:
            skipped += 1
            continue
        
        protein_emb = protein_repr[protein_id]
        ligand_emb = ligand_repr[ligand_id]
        
        combined = np.concatenate([protein_emb, ligand_emb])
        combined_features.append(combined)
        labels.append(row['label'])
    
    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} interactions with missing embeddings")
    
    features = np.vstack(combined_features)
    labels = np.array(labels)
    
    # Save
    features_path = output_path / f'{combination_name}_features.npy'
    labels_path = output_path / f'{combination_name}_labels.npy'
    
    np.save(features_path, features)
    np.save(labels_path, labels)
    
    # Print summary
    first_protein = next(iter(protein_repr.values()))
    first_ligand = next(iter(ligand_repr.values()))
    
    print(f"\n  ✓ Saved {combination_name}")
    print(f"    Protein: {first_protein.shape[0]}D, Ligand: {first_ligand.shape[0]}D → Combined: {features.shape[1]}D")
    print(f"    Samples: {len(features)}, Positive: {labels.sum()} ({labels.mean()*100:.1f}%)")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create representation combinations')
    parser.add_argument('--tsv-path', type=str, help='TSV path (unused)')
    parser.add_argument('--results-suffix', type=str, default='results_non_human',
                       help='Results directory suffix')
    parser.add_argument('--embeddings-dir', type=str,
                       default='${PROJECT_ROOT}/results/protein_model_benchmark_non_human_v2',
                       help='Path to embeddings directory')
    args = parser.parse_args()
    
    # Paths
    ablation_dir = Path(__file__).parent.parent
    embeddings_base = Path(args.embeddings_dir)
    
    interactions_path = ablation_dir / 'data' / args.results_suffix / 'processed' / 'interactions.csv'
    proteins_csv = ablation_dir / 'data' / args.results_suffix / 'processed' / 'proteins.csv'
    ligands_csv = ablation_dir / 'data' / args.results_suffix / 'processed' / 'ligands.csv'
    morgan_path = ablation_dir / 'data' / args.results_suffix / 'embeddings' / 'morgan_fp.npy'
    onehot_path = ablation_dir / 'data' / args.results_suffix / 'embeddings' / 'protein_onehot.npy'
    output_dir = ablation_dir / 'data' / args.results_suffix / 'combinations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load interactions
    print("="*60)
    print("ABLATION STUDY: REPRESENTATION COMBINATIONS")
    print("="*60)
    print(f"Dataset: {args.results_suffix}")
    print(f"Embeddings: {embeddings_base}")
    print(f"Output: {output_dir}\n")
    
    df = load_interactions(interactions_path)
    
    # ESM-2 models
    esm_models = [
        'esm2_t6_8M_UR50D',
        'esm2_t30_150M_UR50D',
        'esm2_t36_3B_UR50D'
    ]
    
    print(f"\nProcessing {len(esm_models)} ESM-2 models × 3 combinations + 1 OneHot×Morgan = 10 total\n")
    
    # Process each ESM-2 model
    for model_name in esm_models:
        print(f"{'#'*60}")
        print(f"# Model: {model_name}")
        print(f"{'#'*60}\n")
        
        model_path = embeddings_base / model_name
        
        # Load ESM-2 and SMI-TED
        esm_proteins = load_esm_proteins(model_path)
        smited_ligands = load_smited_ligands(model_path)

        # C1: ESM-2 + SMI-TED
        print(f"C1: ESM-2 + SMI-TED")

        # Check if we have enough data to proceed
        if len(esm_proteins) == 0 or len(smited_ligands) == 0:
            print(f"  ⚠️ Skipping C1 for {model_name} - insufficient data")
        else:
            combine_and_save(
                df, esm_proteins, smited_ligands, output_dir,
                f'C1_ESM_{model_name}_SMITED'
            )
        
        # C2: ESM-2 + Morgan FP
        print(f"\nC2: ESM-2 + Morgan FP")
        morgan_ligands = load_morgan_ligands(morgan_path, ligands_csv)

        # Check if we have enough data to proceed
        if len(esm_proteins) == 0 or len(morgan_ligands) == 0:
            print(f"  ⚠️ Skipping C2 for {model_name} - insufficient data")
        else:
            combine_and_save(
                df, esm_proteins, morgan_ligands, output_dir,
                f'C2_ESM_{model_name}_Morgan'
            )
        
        # C3: One-Hot + SMI-TED
        print(f"\nC3: One-Hot + SMI-TED")
        onehot_proteins = load_onehot_proteins(onehot_path, proteins_csv)

        # Check if we have enough data to proceed
        if len(onehot_proteins) == 0 or len(smited_ligands) == 0:
            print(f"  ⚠️ Skipping C3 for {model_name} - insufficient data")
        else:
            combine_and_save(
                df, onehot_proteins, smited_ligands, output_dir,
                f'C3_OneHot_SMITED_{model_name}'
            )
        
        print()
    
    # C4: One-Hot + Morgan FP (independent of ESM-2)
    print(f"{'#'*60}")
    print(f"# Combination C4 (independent)")
    print(f"{'#'*60}\n")
    print(f"C4: One-Hot + Morgan FP")
    onehot_proteins = load_onehot_proteins(onehot_path, proteins_csv)
    morgan_ligands = load_morgan_ligands(morgan_path, ligands_csv)

    # Check if we have enough data to proceed
    if len(onehot_proteins) == 0 or len(morgan_ligands) == 0:
        print(f"  ⚠️ Skipping C4 - insufficient data")
    else:
        combine_and_save(
            df, onehot_proteins, morgan_ligands, output_dir,
            'C4_OneHot_Morgan'
        )
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n✓ All combinations generated successfully!")
    print(f"\nCombinations created:")
    print(f"  C1 (ESM+SMITED):     {len(esm_models)} variants")
    print(f"  C2 (ESM+Morgan):     {len(esm_models)} variants")
    print(f"  C3 (OneHot+SMITED):  {len(esm_models)} variants")
    print(f"  C4 (OneHot+Morgan):  1 variant")
    print(f"\n  Total: {len(esm_models) * 3 + 1} combinations")
    print(f"\nOutput: {output_dir}")
    print("\nNext: Run classification with KNN and MLP classifiers")


if __name__ == '__main__':
    main()
