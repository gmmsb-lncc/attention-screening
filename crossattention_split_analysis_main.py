#!/usr/bin/env python3
"""
CrossAttention Split Analysis: Main Entry Point
===============================================

This script serves as the main entry point for the CrossAttention split analysis module.
It provides a command-line interface to run the analysis with different configurations.

For detailed usage and configuration options, please refer to the README.md file
in the crossattention_split_analysis package directory.
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from crossattention_split_analysis import run_crossattention_analysis, TrainingConfig
from crossattention_split_analysis.experiment import run_single_analysis
from crossattention_split_analysis.config import (
    SUPPORTED_EMBEDDINGS,
    DEFAULT_SEEDS
)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_comparative_summary(all_embedding_results: dict):
    """Print comparative summary across all embeddings."""
    if not all_embedding_results:
        return

    print("\n" + "=" * 80)
    print("COMPARATIVE SUMMARY: ALL EMBEDDINGS")
    print("=" * 80)

    # Header
    print(f"\n{'Scenario':<30} ", end="")
    for emb in all_embedding_results.keys():
        print(f"{emb:>15}", end="")
    print()
    print("-" * (30 + 15 * len(all_embedding_results)))

    # Get all scenarios from first result
    first_result = list(all_embedding_results.values())[0]
    if not first_result:
        return

    scenarios = list(first_result.keys())

    for scenario in scenarios:
        scenario_clean = scenario.replace('\n', ' ')
        print(f"{scenario_clean:<30} ", end="")

        for emb, results in all_embedding_results.items():
            if results and scenario in results:
                metrics = results[scenario].get('CNN+CrossAttn', {})
                mcc = metrics.get('mcc', 0)
                mcc_std = metrics.get('mcc_std', 0)
                if mcc_std > 0:
                    print(f"{mcc:.3f}±{mcc_std:.3f}".rjust(15), end="")
                else:
                    print(f"{mcc:.3f}".rjust(15), end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()

    print("=" * 80)
    print("(Values shown: MCC ± std)")


def main():
    parser = argparse.ArgumentParser(
        description="CrossAttention Split Analysis for Protein-Ligand Affinity Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single embedding with default settings
  %(prog)s --embedding 150M --dataset non_human

  # Run all embeddings (8M, 150M, 650M)
  %(prog)s --run_all --dataset non_human

  # Run specific scenarios only
  %(prog)s --embedding 150M --dataset non_human --scenarios scaffold

  # Custom training parameters
  %(prog)s --embedding 150M --dataset non_human --epochs 300 --patience 20 --batch_size 64

  # Force recalculation with custom seeds
  %(prog)s --embedding 150M --dataset non_human --force --seeds 42 123 456 789 1024

  # Debug mode for detailed error traces
  %(prog)s --embedding 150M --dataset non_human --debug

Available scenarios:
  - scaffold                : Split by scaffold (fixed precomputed Sc split)
        """
    )

    # Required arguments
    parser.add_argument(
        '--dataset', '-d',
        choices=['human', 'non_human', 'all'],
        required=True,
        help='Dataset type to use'
    )

    # Embedding selection
    parser.add_argument(
        '--embedding', '-e',
        choices=['8M', '150M', '650M'],
        help='ESM-2 embedding model size (8M, 150M, 650M)'
    )

    parser.add_argument(
        '--run_all',
        action='store_true',
        help='Run analysis for all supported embedding sizes (8M, 150M, 650M)'
    )

    # Scenario selection
    parser.add_argument(
        '--scenarios', '-s',
        type=str,
        default=None,
        help='Comma-separated scenario list (only scaffold is supported). '
             'Use "scaffold" or "all" (maps to scaffold).'
    )

    parser.add_argument(
        '--scaffold_split_dir',
        type=str,
        default='scaffolds_splits/output',
        help='Directory with precomputed scaffold splits from scaffold_split.py '
             '(default: scaffolds_splits/output)'
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        help='Maximum number of training epochs (default: 500)'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=30,
        help='Early stopping patience in epochs (default: 30)'
    )

    parser.add_argument(
        '--no-early-stopping',
        action='store_true',
        help='Disable early stopping (train for full --epochs)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Training batch size (default: 32)'
    )

    parser.add_argument(
        '--learning_rate', '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )

    # Output and reproducibility
    parser.add_argument(
        '--output_dir', '-o',
        default='./results/crossattention_analysis',
        help='Output directory for results (default: ./results/crossattention_analysis)'
    )

    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=DEFAULT_SEEDS,
        help=f'Random seeds for reproducibility (default: {DEFAULT_SEEDS})'
    )

    # Flags
    parser.add_argument(
        '--use_attention',
        action='store_true',
        help='Use attention matrices instead of embeddings for protein'
    )

    parser.add_argument(
        '--molformer_ligand',
        action='store_true',
        help='Use MoLFormer matrices instead of SMI-TED for ligand representation'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recalculation even if results exist'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with full error tracebacks'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.run_all and args.embedding is None:
        parser.error("--embedding is required unless --run_all is specified")

    # Parse scenarios
    if args.scenarios:
        if args.scenarios.strip().lower() == 'all':
            scenarios = ['scaffold']
        else:
            scenarios = [s.strip() for s in args.scenarios.split(',') if s.strip()]
        normalized = {s.lower() for s in scenarios}
        if normalized not in ({'scaffold'}, {'sc'}, {'split_by_scaffold'}):
            parser.error(
                "This pipeline supports only scaffold scenario. "
                "Use --scenarios scaffold (or --scenarios all)."
            )
        scenarios = ['scaffold']
    else:
        scenarios = ['scaffold']

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine embeddings to run
    if args.run_all:
        embeddings_to_run = ['8M', '150M', '650M']
    else:
        embeddings_to_run = [args.embedding]

    # Print configuration
    print("=" * 70)
    print("CROSSATTENTION SPLIT ANALYSIS")
    print("=" * 70)
    print(f"Dataset:          {args.dataset}")
    print(f"Embeddings:       {embeddings_to_run}")
    print(f"Scenarios:        {scenarios}")
    print(f"Output directory: {args.output_dir}")
    print(f"Scaffold split dir:{args.scaffold_split_dir}")
    print(f"Protein input:    {'Attention matrices' if args.use_attention else 'Per-token embeddings'}")
    print(f"Ligand input:     {'Per-token embeddings (MoLFormer)' if args.molformer_ligand else 'Per-token embeddings (SMI-TED)'}")
    print(f"Seeds:            {args.seeds} (n={len(args.seeds)})")
    print("-" * 70)
    print("Training Parameters:")
    print(f"  Epochs:         {args.epochs}")
    if args.no_early_stopping:
        print(f"  Early stopping: DISABLED")
    else:
        print(f"  Patience:       {args.patience}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.learning_rate}")
    print("-" * 70)
    print(f"Debug mode:       {args.debug}")
    print(f"Force recalc:     {args.force}")
    print("=" * 70)

    # Track results for comparative summary
    all_embedding_results = {}
    all_embedding_times = {}

    total_start_time = time.time()

    # Run analysis for each embedding
    for i, embedding in enumerate(embeddings_to_run, 1):
        print(f"\n{'#' * 70}")
        print(f"# [{i}/{len(embeddings_to_run)}] Running analysis for {embedding} embedding...")
        print(f"{'#' * 70}")

        embedding_start_time = time.time()

        try:
            # Set patience to None if early stopping is disabled
            patience = None if args.no_early_stopping else args.patience

            result = run_single_analysis(
                embedding_name=embedding,
                dataset_type=args.dataset,
                output_dir=args.output_dir,
                seeds=args.seeds,
                force=args.force,
                use_attention=args.use_attention,
                scenarios=scenarios,
                num_epochs=args.epochs,
                patience=patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                use_molformer_ligand=args.molformer_ligand,
                scaffold_split_dir=args.scaffold_split_dir
            )

            embedding_time = time.time() - embedding_start_time
            all_embedding_times[embedding] = embedding_time

            if result is None:
                print(f"  Skipped {embedding} (results already exist or data not found)")
                all_embedding_results[embedding] = None
            else:
                print(f"  ✓ Completed {embedding} analysis in {format_time(embedding_time)}")
                all_embedding_results[embedding] = result

        except Exception as e:
            embedding_time = time.time() - embedding_start_time
            all_embedding_times[embedding] = embedding_time
            all_embedding_results[embedding] = None

            print(f"  ✗ Error running {embedding} analysis: {e}")
            if args.debug:
                print("\n" + "=" * 50)
                print("DEBUG TRACEBACK:")
                print("=" * 50)
                traceback.print_exc()
                print("=" * 50 + "\n")
            continue

    total_time = time.time() - total_start_time

    # Print timing summary
    print("\n" + "=" * 70)
    print("EXECUTION TIME SUMMARY")
    print("=" * 70)
    for emb, emb_time in all_embedding_times.items():
        status = "✓" if all_embedding_results.get(emb) else "✗"
        print(f"  {status} {emb}: {format_time(emb_time)}")
    print("-" * 70)
    print(f"  TOTAL: {format_time(total_time)}")
    print("=" * 70)

    # Print comparative summary if multiple embeddings
    if len(embeddings_to_run) > 1:
        valid_results = {k: v for k, v in all_embedding_results.items() if v is not None}
        if valid_results:
            print_comparative_summary(valid_results)

    print("\n✓ Analysis complete!")
    print(f"  Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
