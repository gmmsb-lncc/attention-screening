#!/usr/bin/env python3
"""
DockTKinase - Attention Matrix CLI
==================================

Cross-Attention model for protein-ligand binding affinity prediction.

SOLID Principles Applied:
    - Single Responsibility: Each function has one job
    - Open/Closed: Easy to extend without modifying
    - Dependency Inversion: Depends on abstractions (modules)

Usage:
    python attention_matrix.py --attention-matrix on --input data.tsv --embeddings results/
    python attention_matrix.py --attention-matrix off
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

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
    embeddings_dir: Optional[str]
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


def parse_arguments() -> CLIArgs:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description='DockTKinase - Attention Matrix Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Core arguments
    parser.add_argument('--attention-matrix', type=str, choices=['on', 'off'], default='on')
    parser.add_argument('--input', '-i', type=str, default=None)
    parser.add_argument('--embeddings', '-e', type=str, default=None)
    parser.add_argument('--output', '-o', type=str, default=None)
    
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
        embeddings_dir=args.embeddings,
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
        seed=args.seed
    )


# =============================================================================
# Validation (Single Responsibility: Input validation)
# =============================================================================

def validate_inputs(args: CLIArgs) -> Optional[str]:
    """
    Validate CLI inputs. Returns error message or None if valid.
    
    Follows fail-fast principle: validate early, fail clearly.
    """
    if not args.input_file or not args.embeddings_dir:
        return "Error: --input and --embeddings are required when --attention-matrix is on"
    
    if not Path(args.input_file).exists():
        return f"Error: Input file not found: {args.input_file}"
    
    if not Path(args.embeddings_dir).exists():
        return f"Error: Embeddings directory not found: {args.embeddings_dir}"
    
    return None


# =============================================================================
# Data Loading (Single Responsibility: Load and summarize data)
# =============================================================================

def load_data(args: CLIArgs) -> EmbeddingDataLoader:
    """Load embeddings and splits from disk."""
    loader = EmbeddingDataLoader(
        results_dir=args.embeddings_dir,
        data_file=args.input_file,
        activity_threshold=args.threshold
    )
    loader.load_dataset()
    loader.load_embeddings_from_files()
    loader.load_splits()
    return loader


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
    print(f"        R²:        {reg['r2']:.4f}")
    print(f"        Pearson r: {reg['pearson_r']:.4f}")
    print(f"        Spearman:  {reg['spearman_r']:.4f}")
    print(f"        RMSE:      {reg['rmse']:.4f}")
    print(f"        MAE:       {reg['mae']:.4f}")
    
    print(f"\n  Output: {output_dir}")
    print("=" * 60 + "\n")


def print_disabled_message():
    """Print message when attention matrix is disabled."""
    print("\n  Attention matrix is OFF.")
    print("  Use --attention-matrix on to enable training.\n")


def print_usage_example():
    """Print usage example on validation error."""
    print("\nUsage:")
    print("  python attention_matrix.py --attention-matrix on \\")
    print("      --input tests/datasets/kinase_non_human_compounds.tsv \\")
    print("      --embeddings results/kinase_non_human_full\n")


# =============================================================================
# Main Orchestrator (Single Responsibility: Coordinate workflow)
# =============================================================================

def run_pipeline(args: CLIArgs) -> int:
    """
    Execute the attention matrix training pipeline.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Step 1: Load data
    print(f"\n[1/3] Loading data...")
    loader = load_data(args)
    summary = loader.get_data_summary()
    print_data_summary(summary)
    
    # Step 2: Train model
    print(f"\n[2/3] Training model...")
    output_dir = args.output_dir or str(Path(args.embeddings_dir) / 'attention_matrix')
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
    
    # Step 3: Show results
    print(f"\n[3/3] Results:")
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
