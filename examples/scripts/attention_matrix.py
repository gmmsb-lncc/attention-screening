#!/usr/bin/env python3
"""
DockTKinase - Attention Matrix CLI
==================================

Cross-Attention model for protein-ligand binding affinity prediction.
Supports both vector embeddings (mean-pooled) and matrix embeddings (per-residue/per-token).

SOLID Principles Applied:
    - Single Responsibility: Each function has one job
    - Open/Closed: Easy to extend without modifying
    - Dependency Inversion: Depends on abstractions (modules)

Usage:
    # With pre-computed matrices:
    python attention_matrix.py --attention-matrix on --input data.tsv --build results/

    # Generate matrices and train:
    python attention_matrix.py --attention-matrix on --input data.tsv --build results/ --generate-matrices

    # Use vector mode (mean-pooled embeddings):
    python attention_matrix.py --attention-matrix on --input data.tsv --build results/ --mode vector
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from attention_matrix import (
    AttentionMatrixConfig,
    EmbeddingDataLoader,
    AttentionMatrixPipeline,
)


# =============================================================================
# Configuration (Single Responsibility: CLI argument handling)
# =============================================================================

@dataclass
class CLIArgs:
    """Immutable container for CLI arguments."""
    attention_matrix: str
    input_file: Optional[str]
    build_dir: Optional[str]  # Directory with build results (embeddings)
    output_dir: Optional[str]
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    threshold: float
    device: str
    seed: int
    # New arguments for matrix generation
    generate_matrices: bool
    mode: str  # 'vector' or 'matrix'
    esm_model: str
    force_regenerate: bool


def parse_arguments() -> CLIArgs:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description='DockTKinase - Attention Matrix Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Core arguments
    parser.add_argument('--attention-matrix', type=str, choices=['on', 'off'], default='on')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Input TSV file with compound data')
    parser.add_argument('--build', '-b', type=str, default=None,
                        help='Build output directory from run_complete_pipeline (contains embeddings)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for attention matrix results')
    
    # Matrix generation options
    parser.add_argument('--generate-matrices', action='store_true',
                        help='Generate per-residue/per-token matrices if they do not exist')
    parser.add_argument('--mode', type=str, choices=['vector', 'matrix'], default='matrix',
                        help='Embedding mode: vector (mean-pooled) or matrix (per-residue/per-token)')
    parser.add_argument('--esm-model', type=str, default='esm2_t6_8M_UR50D',
                        help='ESM model for protein embeddings')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='Force regeneration of matrices even if they exist')
    
    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    
    # Other
    parser.add_argument('--threshold', type=float, default=7.0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    return CLIArgs(
        attention_matrix=args.attention_matrix,
        input_file=args.input,
        build_dir=args.build,
        output_dir=args.output,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        threshold=args.threshold,
        device=args.device,
        seed=args.seed,
        generate_matrices=args.generate_matrices,
        mode=args.mode,
        esm_model=args.esm_model,
        force_regenerate=args.force_regenerate
    )


# =============================================================================
# Validation (Single Responsibility: Input validation)
# =============================================================================

def validate_inputs(args: CLIArgs) -> Optional[str]:
    """
    Validate CLI inputs. Returns error message or None if valid.
    
    Follows fail-fast principle: validate early, fail clearly.
    """
    if not args.input_file or not args.build_dir:
        return "Error: --input and --build are required when --attention-matrix is on"
    
    if not Path(args.input_file).exists():
        return f"Error: Input file not found: {args.input_file}"
    
    # Build directory doesn't need to exist if we're generating matrices
    if not args.generate_matrices and not Path(args.build_dir).exists():
        return f"Error: Build directory not found: {args.build_dir}"
    
    return None


# =============================================================================
# Matrix Generation (Single Responsibility: Generate embedding matrices)
# =============================================================================

def detect_device(device_str: str) -> str:
    """Auto-detect best available device."""
    if device_str != 'auto':
        return device_str
    
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def generate_protein_matrices(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    device: str,
    force: bool = False
) -> int:
    """Generate protein embedding matrices (per-residue)."""
    from build.embeddings.protein_embedding import ProteinEmbedding
    from build.core.config import BuildConfig
    
    # Create output directory
    matrix_dir = output_dir / 'protein_matrices'
    matrix_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique sequences
    unique_seqs = df.groupby('seq_id')['seq'].first()
    
    # Check which already exist
    existing = set()
    if not force:
        for f in matrix_dir.glob('*_matrix.npy'):
            existing.add(f.stem.replace('_matrix', ''))
    
    to_process = [(sid, seq) for sid, seq in unique_seqs.items() if sid not in existing]
    
    if not to_process:
        print(f"      ✅ All {len(unique_seqs)} protein matrices already exist")
        return len(unique_seqs)
    
    print(f"      📊 Generating {len(to_process)} protein matrices (skipping {len(existing)} existing)")
    
    # Initialize generator
    config = BuildConfig(
        input_tsv='',
        output_dir=str(output_dir),
        esm_model=model_name,
        device=device
    )
    
    generator = ProteinEmbedding(config=config, model_name=model_name)
    generator.initialize()
    
    # Generate matrices
    success = 0
    for i, (seq_id, sequence) in enumerate(to_process):
        try:
            # Generate per-residue matrix
            matrix = generator.generate_embedding_matrix(sequence)
            
            if matrix is not None:
                output_file = matrix_dir / f"{seq_id}_matrix.npy"
                np.save(output_file, matrix)
                success += 1
                
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"         [{i+1}/{len(to_process)}] {seq_id}: shape {matrix.shape}")
        except Exception as e:
            print(f"         ❌ Error processing {seq_id}: {e}")
    
    print(f"      ✅ Generated {success}/{len(to_process)} protein matrices")
    return success + len(existing)


def generate_ligand_matrices(
    df: pd.DataFrame,
    output_dir: Path,
    device: str,
    force: bool = False
) -> int:
    """Generate ligand embedding matrices (per-token)."""
    from build.embeddings.ligand_embedding import LigandEmbedding
    from build.core.config import BuildConfig
    
    # Create output directory
    matrix_dir = output_dir / 'ligand_matrices'
    matrix_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique SMILES
    ligand_col = 'molecule_chembl_id' if 'molecule_chembl_id' in df.columns else 'chembl_id'
    smiles_col = 'canonical_smiles' if 'canonical_smiles' in df.columns else 'smiles'
    
    unique_ligands = df.groupby(ligand_col)[smiles_col].first()
    
    # Check which already exist
    existing = set()
    if not force:
        for f in matrix_dir.glob('*_matrix.npy'):
            existing.add(f.stem.replace('_matrix', ''))
    
    to_process = [(lid, smi) for lid, smi in unique_ligands.items() if lid not in existing]
    
    if not to_process:
        print(f"      ✅ All {len(unique_ligands)} ligand matrices already exist")
        return len(unique_ligands)
    
    print(f"      📊 Generating {len(to_process)} ligand matrices (skipping {len(existing)} existing)")
    
    # Initialize generator
    config = BuildConfig(
        input_tsv='',
        output_dir=str(output_dir),
        ligand_model='SMI-TED',
        device=device
    )
    
    generator = LigandEmbedding(config=config, model_name='SMI-TED')
    generator.initialize()
    
    # Generate matrices
    success = 0
    for i, (lig_id, smiles) in enumerate(to_process):
        try:
            # Generate per-token matrix
            matrix = generator.generate_embedding_matrix(smiles)
            
            if matrix is not None:
                output_file = matrix_dir / f"{lig_id}_matrix.npy"
                np.save(output_file, matrix)
                success += 1
                
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"         [{i+1}/{len(to_process)}] {lig_id}: shape {matrix.shape}")
            else:
                # SMI-TED doesn't support per-token, use vector as fallback
                vector = generator.generate_embedding(smiles)
                if vector is not None:
                    # Reshape to matrix format [1, dim]
                    matrix = vector.reshape(1, -1)
                    output_file = matrix_dir / f"{lig_id}_matrix.npy"
                    np.save(output_file, matrix)
                    success += 1
        except Exception as e:
            print(f"         ❌ Error processing {lig_id}: {e}")
    
    print(f"      ✅ Generated {success}/{len(to_process)} ligand matrices")
    return success + len(existing)


def check_and_generate_matrices(args: CLIArgs, df: pd.DataFrame) -> bool:
    """
    Check if matrices exist and generate if needed.
    
    Returns:
        True if matrices are ready, False otherwise
    """
    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    
    protein_dir = build_dir / 'protein_matrices'
    ligand_dir = build_dir / 'ligand_matrices'
    
    # Count existing matrices
    n_protein_matrices = len(list(protein_dir.glob('*_matrix.npy'))) if protein_dir.exists() else 0
    n_ligand_matrices = len(list(ligand_dir.glob('*_matrix.npy'))) if ligand_dir.exists() else 0
    
    # Get expected counts
    n_unique_proteins = df['seq_id'].nunique()
    ligand_col = 'molecule_chembl_id' if 'molecule_chembl_id' in df.columns else 'chembl_id'
    n_unique_ligands = df[ligand_col].nunique()
    
    # Check if matrices are complete
    matrices_complete = (
        n_protein_matrices >= n_unique_proteins and 
        n_ligand_matrices >= n_unique_ligands
    )
    
    if matrices_complete and not args.force_regenerate:
        print(f"      ✅ Matrices already exist ({n_protein_matrices} proteins, {n_ligand_matrices} ligands)")
        return True
    
    if not args.generate_matrices and not matrices_complete:
        print(f"      ⚠️  Missing matrices: {n_unique_proteins - n_protein_matrices} proteins, "
              f"{n_unique_ligands - n_ligand_matrices} ligands")
        print(f"      Use --generate-matrices to create them")
        return False
    
    # Generate matrices
    print(f"\n[1.5/3] Generating embedding matrices...")
    device = detect_device(args.device)
    print(f"        Device: {device}")
    
    generate_protein_matrices(df, build_dir, args.esm_model, device, args.force_regenerate)
    generate_ligand_matrices(df, build_dir, device, args.force_regenerate)
    
    return True


# =============================================================================
# Configuration Factory (Single Responsibility: Create config from args)
# =============================================================================

def create_config(args: CLIArgs, data_summary: dict) -> AttentionMatrixConfig:
    """Create pipeline configuration from CLI args and data summary."""
    return AttentionMatrixConfig(
        protein_dim=data_summary['protein_dim'],
        ligand_dim=data_summary['ligand_dim'],
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        activity_threshold=args.threshold,
        device=args.device,
        random_state=args.seed
    )


# =============================================================================
# Output (Single Responsibility: Display results)
# =============================================================================

def print_banner():
    """Print application banner."""
    print("\n" + "=" * 60)
    print("  DockTKinase - Attention Matrix")
    print("=" * 60)


def print_data_summary(summary: dict):
    """Print data loading summary."""
    print(f"      Samples: {summary['n_samples']} | "
          f"Protein: {summary['protein_dim']}d | "
          f"Ligand: {summary['ligand_dim']}d")
    print(f"      Split: {summary['n_train']}/{summary['n_val']}/{summary['n_test']} "
          f"(train/val/test)")


def print_results(metrics: dict, output_dir: str):
    """Print training results."""
    cls = metrics['classification']
    reg = metrics['regression']
    
    print(f"\n      Classification:")
    print(f"        Accuracy:  {cls['accuracy']:.4f}")
    print(f"        ROC-AUC:   {cls['roc_auc']:.4f}")
    print(f"        F1 Score:  {cls['f1']:.4f}")
    print(f"        MCC:       {cls['mcc']:.4f}")
    
    print(f"\n      Regression (pChEMBL):")
    print(f"        R²:         {reg['r2']:.4f}")
    print(f"        Pearson r:  {reg['pearson_r']:.4f}")
    print(f"        Spearman ρ: {reg['spearman_r']:.4f}")
    print(f"        Kendall τ:  {reg.get('kendall_tau', 0):.4f}")
    print(f"        CCC (Lin):  {reg.get('ccc', 0):.4f}")
    print(f"        RMSE:       {reg['rmse']:.4f}")
    print(f"        MAE:        {reg['mae']:.4f}")
    print(f"        Expl. Var:  {reg.get('explained_variance', 0):.4f}")
    
    print(f"\n  Output: {output_dir}")
    print("=" * 60 + "\n")


def print_disabled_message():
    """Print message when attention matrix is disabled."""
    print("\n  Attention matrix is OFF.")
    print("  Use --attention-matrix on to enable training.\n")


def print_usage_example():
    """Print usage example on validation error."""
    print("\nUsage:")
    print("  # Train with pre-computed matrices:")
    print("  python attention_matrix.py --attention-matrix on \\")
    print("      --input tests/datasets/kinase_test_small.tsv \\")
    print("      --build results/test_pipeline_check/build")
    print("")
    print("  # Generate matrices and train:")
    print("  python attention_matrix.py --attention-matrix on \\")
    print("      --input tests/datasets/kinase_test_small.tsv \\")
    print("      --build results/test_pipeline_check/build \\")
    print("      --generate-matrices")
    print("")
    print("  # Use vector mode (mean-pooled embeddings):")
    print("  python attention_matrix.py --attention-matrix on \\")
    print("      --input tests/datasets/kinase_test_small.tsv \\")
    print("      --build results/test_pipeline_check/build \\")
    print("      --mode vector\n")


# =============================================================================
# Main Orchestrator (Single Responsibility: Coordinate workflow)
# =============================================================================

def run_pipeline(args: CLIArgs) -> int:
    """
    Execute the attention matrix training pipeline.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Step 1: Load dataset
    print(f"\n[1/4] Loading dataset...")
    df = pd.read_csv(args.input_file, sep='\t')
    print(f"      Samples: {len(df)}")
    print(f"      Mode: {args.mode}")
    
    # Step 1.5: Check/Generate matrices (for matrix mode)
    if args.mode == 'matrix':
        if not check_and_generate_matrices(args, df):
            return 1
    
    # Step 2: Load embeddings
    print(f"\n[2/4] Loading embeddings...")
    loader = EmbeddingDataLoader(
        results_dir=args.build_dir,
        data_file=args.input_file,
        activity_threshold=args.threshold,
        embedding_mode=args.mode  # Pass mode to loader
    )
    loader.load_dataset()
    loader.load_embeddings_from_files()
    
    # Step 2.5: Load or create splits
    print(f"\n[2.5/4] Loading splits...")
    loader.load_splits()
    
    summary = loader.get_data_summary()
    print(f"      Valid samples: {summary['n_samples']}")
    print(f"      Protein dim: {summary['protein_dim']}d | Ligand dim: {summary['ligand_dim']}d")
    print(f"      Split: {summary['n_train']}/{summary['n_val']}/{summary['n_test']} (train/val/test)")
    
    # Step 3: Train model
    print(f"\n[3/4] Training model...")
    output_dir = args.output_dir or str(Path(args.build_dir) / 'attention_matrix')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    config = create_config(args, summary)
    pipeline = AttentionMatrixPipeline(config=config, output_dir=output_dir)
    
    results = pipeline.run_with_precomputed_embeddings(
        protein_embeddings=loader.protein_embeddings,
        ligand_embeddings=loader.ligand_embeddings,
        binary_labels=loader.binary_labels,
        regression_targets=loader.regression_targets,
        train_idx=loader.train_idx,
        val_idx=loader.val_idx,
        test_idx=loader.test_idx
    )
    
    # Step 4: Show results
    print(f"\n[4/4] Results:")
    print_results(results['metrics'], output_dir)
    
    return 0


def main() -> int:
    """
    Application entry point.
    
    Follows the principle of least surprise: clear flow, explicit returns.
    """
    args = parse_arguments()
    print_banner()
    
    # Handle OFF mode (early return pattern)
    if args.attention_matrix == 'off':
        print_disabled_message()
        return 0
    
    # Validate inputs (fail-fast pattern)
    error = validate_inputs(args)
    if error:
        print(f"\n{error}")
        print_usage_example()
        return 1
    
    # Execute pipeline
    return run_pipeline(args)


if __name__ == '__main__':
    sys.exit(main())
