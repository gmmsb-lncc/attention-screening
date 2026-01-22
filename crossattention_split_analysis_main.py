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
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from crossattention_split_analysis import run_crossattention_analysis, TrainingConfig
from crossattention_split_analysis.experiment import run_single_analysis
from crossattention_split_analysis.config import SUPPORTED_EMBEDDINGS, DEFAULT_SEEDS


def main():
    parser = argparse.ArgumentParser(
        description="CrossAttention Split Analysis for Protein-Ligand Affinity Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --embedding 150M --dataset non_human
  %(prog)s --run_all --dataset non_human
  %(prog)s --embedding 150M --dataset non_human --use-attention
  %(prog)s --embedding 150M --dataset non_human --force
        """
    )
    
    parser.add_argument(
        '--embedding', '-e',
        choices=['8M', '150M', '650M'],
        help='ESM-2 embedding model size (8M, 150M, 650M)'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        choices=['human', 'non_human', 'all'],
        required=True,
        help='Dataset type to use'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        default='./results/crossattention_analysis',
        help='Output directory for results (default: ./results/crossattention_analysis)'
    )
    
    parser.add_argument(
        '--run_all',
        action='store_true',
        help='Run analysis for all supported embedding sizes'
    )
    
    parser.add_argument(
        '--use_attention',
        action='store_true',
        help='Use attention matrices instead of embeddings'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recalculation even if results exist'
    )
    
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=DEFAULT_SEEDS,
        help='Random seeds for reproducibility (default: %(default)s)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.run_all and args.embedding is None:
        parser.error("--embedding is required unless --run_all is specified")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine embeddings to run
    embeddings_to_run = []
    if args.run_all:
        embeddings_to_run = ['8M', '150M', '650M']
    else:
        embeddings_to_run = [args.embedding]
    
    print("=" * 70)
    print("CROSSATTENTION SPLIT ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Embeddings: {embeddings_to_run}")
    print(f"Output directory: {args.output_dir}")
    print(f"Use attention: {args.use_attention}")
    print(f"Seeds: {args.seeds}")
    print(f"Force recalculation: {args.force}")
    print("=" * 70)
    
    # Run analysis for each embedding
    for embedding in embeddings_to_run:
        print(f"\nRunning analysis for {embedding} embedding...")
        
        try:
            result = run_single_analysis(
                embedding_name=embedding,
                dataset_type=args.dataset,
                output_dir=args.output_dir,
                seeds=args.seeds,
                force=args.force,
                use_attention=args.use_attention
            )
            
            if result is None:
                print(f"  Skipped {embedding} (results already exist or not found)")
            else:
                print(f"  Completed {embedding} analysis")
                
        except Exception as e:
            print(f"  Error running {embedding} analysis: {e}")
            continue
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()