#!/usr/bin/env python3
"""
Encoder Architecture Comparison - Main Entry Point.

This script compares different encoder architectures for protein-ligand
affinity prediction tasks using three evaluation scenarios.

See encoder_compare/README.md for detailed documentation.
"""

import argparse
import sys

from encoder_compare.config import (
    SUPPORTED_EMBEDDINGS,
    PROTEIN_DIMS,
    LIGAND_DIM,
    ENCODER_TYPES,
    TrainingConfig
)
from encoder_compare.experiment import run_encoder_comparison
from encoder_compare.visualization import plot_comparison, print_summary, save_results


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare encoder architectures for protein-ligand affinity prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single embedding with single seed (fastest)
  python compare_encoder_architectures.py --embedding 150M --dataset non_human

  # Multiple seeds for robust statistics
  python compare_encoder_architectures.py --embedding 150M --dataset non_human --seeds 42 123 456

  # All embeddings
  python compare_encoder_architectures.py --run_all --dataset non_human

  # Resume from checkpoint (default)
  python compare_encoder_architectures.py --embedding 150M --dataset non_human

  # Start fresh (ignore checkpoint)
  python compare_encoder_architectures.py --embedding 150M --dataset non_human --no-resume
        """
    )

    parser.add_argument(
        '--embedding',
        type=str,
        default='150M',
        choices=['8M', '150M', '650M'],
        help='Embedding model size (default: 150M)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='non_human',
        choices=['human', 'non_human'],
        help='Dataset type (default: non_human)'
    )
    parser.add_argument(
        '--run_all',
        action='store_true',
        help='Run all embeddings (8M, 150M, 650M)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs (default: 200)'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42],
        help='Random seeds for multiple runs (default: 42)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from checkpoint (start fresh)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/media/storage/leon/semantic-screening/encoder_comparison_results',
        help='Output directory for results'
    )

    return parser.parse_args()


def print_experiment_info(args: argparse.Namespace) -> None:
    """Print experiment configuration."""
    print("\n" + "=" * 70)
    print("ENCODER ARCHITECTURE COMPARISON")
    print("=" * 70)
    print(f"Encoders: {', '.join(e.upper() for e in ENCODER_TYPES)}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Seeds: {args.seeds}")
    print(f"Resume: {not args.no_resume}")


def get_embeddings_to_run(args: argparse.Namespace) -> list:
    """Get list of embeddings to run."""
    if args.run_all:
        return list(SUPPORTED_EMBEDDINGS.values())
    else:
        return [SUPPORTED_EMBEDDINGS.get(args.embedding, args.embedding)]


def run_single_embedding(
    embedding_name: str,
    args: argparse.Namespace
) -> None:
    """Run comparison for a single embedding."""
    print(f"\n{'#' * 70}")
    print(f"EMBEDDING: {embedding_name}")
    print(f"{'#' * 70}")

    # Create config
    protein_dim = PROTEIN_DIMS.get(embedding_name, 640)
    config = TrainingConfig(
        protein_dim=protein_dim,
        ligand_dim=LIGAND_DIM,
        num_epochs=args.epochs
    )

    # Run experiments
    results = run_encoder_comparison(
        embedding_name=embedding_name,
        dataset_type=args.dataset,
        output_dir=args.output_dir,
        config=config,
        seeds=args.seeds,
        resume=not args.no_resume
    )

    # Save and visualize results
    if results:
        save_results(results, embedding_name, args.dataset, args.output_dir)
        plot_comparison(results, embedding_name, args.dataset, args.output_dir)
        print_summary(results)


def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    print_experiment_info(args)

    embeddings_to_run = get_embeddings_to_run(args)

    for embedding_name in embeddings_to_run:
        run_single_embedding(embedding_name, args)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user. Progress saved in checkpoint.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
