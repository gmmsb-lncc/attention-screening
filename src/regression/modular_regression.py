#!/usr/bin/env python3
"""
Modular Regression CLI Interface - DockTKinase
==============================================

Command line interface 100% compatible with the original
pipeline, but using the modular implementation.

Usage:
    python modular_regression.py embeddings.npy targets.npy
    python modular_regression.py embeddings.npy targets.npy --models RandomForest XGBoost
    python modular_regression.py embeddings.npy targets.npy --output results/my_test
"""

import argparse
import sys
from pathlib import Path

# Adicionar src ao path se necessário
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from regression.modular_pipeline import RegressionPipeline, run_regression_pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Modular Regression Pipeline - DockTKinase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:

  # Train all models:
  python modular_regression.py embeddings.npy targets.npy
  
  # Train specific models:
  python modular_regression.py embeddings.npy targets.npy \\
      --models RandomForest GradientBoosting XGBoost
  
  # Specify output directory:
  python modular_regression.py embeddings.npy targets.npy \\
      --output results/my_experiment
  
  # Configure splits:
  python modular_regression.py embeddings.npy targets.npy \\
      --test-size 0.15 --val-size 0.15
  
  # Custom seed:
  python modular_regression.py embeddings.npy targets.npy \\
      --random-state 123
  
Available models:
  - RandomForest
  - GradientBoosting
  - XGBoost (if installed)
  - Ridge
  - Lasso
  - ElasticNet
  - SVR
  - KNN
  - DecisionTree
  - MLP
  - LightGBM (if installed)
  - CatBoost (if installed)
        """
    )
    
    # Positional arguments
    parser.add_argument(
        'embeddings',
        type=str,
        help='Path to embeddings file (.npy or .npz)'
    )
    
    parser.add_argument(
        'targets',
        type=str,
        help='Path to targets file (.npy)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='List of models to train (default: all)'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='results/regression',
        help='Output directory for results (default: results/regression)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Validation set proportion (default: 0.1)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Seed para reprodutibilidade (padrão: 42)'
    )
    
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Modo silencioso (sem output verboso)'
    )
    
    return parser.parse_args()


def validate_args(args):
    """Validate arguments."""
    # Validate input files
    embeddings_path = Path(args.embeddings)
    if not embeddings_path.exists():
        print(f"❌ Error: Embeddings file not found: {embeddings_path}")
        sys.exit(1)
    
    targets_path = Path(args.targets)
    if not targets_path.exists():
        print(f"❌ Error: Targets file not found: {targets_path}")
        sys.exit(1)
    
    # Validate splits
    if not 0 < args.test_size < 1:
        print(f"❌ Error: test-size must be between 0 and 1")
        sys.exit(1)
    
    if not 0 < args.val_size < 1:
        print(f"❌ Error: val-size must be between 0 and 1")
        sys.exit(1)
    
    if args.test_size + args.val_size >= 1:
        print(f"❌ Error: test-size + val-size must be less than 1")
        sys.exit(1)


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Validate
    validate_args(args)
    
    # Configure verbosity
    verbose = not args.quiet
    
    if verbose:
        print("🚀 Modular Regression Pipeline - DockTKinase")
        print("=" * 70)
        print(f"   Embeddings: {args.embeddings}")
        print(f"   Targets: {args.targets}")
        print(f"   Output: {args.output}")
        if args.models:
            print(f"   Models: {', '.join(args.models)}")
        else:
            print(f"   Models: All available")
        print(f"   Test size: {args.test_size}")
        print(f"   Val size: {args.val_size}")
        print(f"   Random state: {args.random_state}")
        print("=" * 70)
        print()
    
    # Criar e executar pipeline
    try:
        pipeline = RegressionPipeline(
            embeddings_path=args.embeddings,
            targets_path=args.targets,
            output_dir=args.output,
            models_to_train=args.models,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            verbose=verbose
        )
        
        results = pipeline.run()
        
        if verbose:
            print(f"\n✅ Pipeline executed successfully!")
            print(f"   Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error executing pipeline: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
