#!/usr/bin/env python3
# ESM-2: python scripts/run_complete_pipeline.py --input tests/datasets/kinase_non_human_compounds.tsv --output results/esm2_15B_test --protein-model esm2_t48_15B_UR50D --seed 42
# ESM-C (local): python scripts/run_complete_pipeline.py --input tests/datasets/kinase_non_human_compounds.tsv --output results/esmc_600m_test --protein-model esmc-600m-2024-12 --seed 42
# ESM-C 6B (API): ESM_API_KEY="your_key" python scripts/run_complete_pipeline.py --input tests/datasets/kinase_non_human_compounds.tsv --output results/esmc_6b_test --protein-model esmc-6b-2024-12 --seed 42
# Boltz-2: python scripts/run_complete_pipeline.py --input tests/datasets/kinase_non_human_compounds.tsv --output results/boltz2_test --protein-model boltz2 --seed 42

"""
DockTKinase - Complete Integrated Pipeline
==========================================

End-to-end pipeline with automatic checkpoints:
1. Build: Embeddings (ESM + FM4M) + Matrix
2. Classification: Multi-Model (10 sklearn models)
3. Regression: Multi-Model (10 models)

Usage:
    # Basic
    python scripts/run_complete_pipeline.py --input tests/datasets/kinase_non_human_compounds.tsv
    
    # With options
    python scripts/run_complete_pipeline.py \\
        --input tests/datasets/kinase_human_compounds.tsv \\
        --output results/my_experiment \\
        --device cuda \\
        --no-checkpoints
    
    # Regression only (skip classification)
    python scripts/run_complete_pipeline.py \\
        --input data.tsv \\
        --no-classification
    
    # ESM-C 6B (requires EvolutionaryScale Forge API key)
    export ESM_API_KEY="your_api_key"
    python scripts/run_complete_pipeline.py \\
        --input data.tsv \\
        --protein-model esmc-6b-2024-12

Features:
    • Automatic checkpoint system (avoids recalculation)
    • Supports CPU and GPU (CUDA/MPS)
    • Multi-model for classification AND regression
    • Robust validation with 3 splits (train/val/test)
    • Complete metrics and visualizations
    • ESM-C 6B via Forge API (requests key interactively if not configured)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))


# =============================================================================
# Models that require external API (Forge API)
# =============================================================================
FORGE_API_MODELS = {'esmc-6b-2024-12'}


def validate_forge_api_key(model_name: str, api_key: Optional[str] = None) -> bool:
    """
    Validates if the Forge API key is configured for models that require remote access.
    
    Args:
        model_name: Name of the selected model
        api_key: API key passed via --api (optional)
        
    Returns:
        True if the API key is valid or not needed, False otherwise
    """
    if model_name not in FORGE_API_MODELS:
        return True
    
    print()
    print('=' * 80)
    print('🔐 AUTHENTICATION REQUIRED - ESM-C 6B (EvolutionaryScale Forge API)')
    print('=' * 80)
    print()
    print('The ESM-C 6B model (esmc-6b-2024-12) requires access to the EvolutionaryScale Forge API.')
    print('This model is NOT available locally due to its size (6 billion parameters).')
    print()
    print('📋 To obtain your API key:')
    print('   1. Go to: https://forge.evolutionaryscale.ai')
    print('   2. Create an account or log in')
    print('   3. Navigate to Settings > API Keys')
    print('   4. Generate a new API key')
    print()
    
    # Priority: 1) --api parameter, 2) ESM_API_KEY env var
    if api_key:
        os.environ['ESM_API_KEY'] = api_key
        print(f'✅ API key provided via --api (***{api_key[-4:]})')
        print()
        return True
    
    # Check if already configured via environment variable
    env_key = os.environ.get('ESM_API_KEY', '').strip()
    
    if env_key:
        print(f'✅ ESM_API_KEY found in environment (***{env_key[-4:]})')
        print()
        return True
    
    # Request API key interactively
    print('⚠️  ESM_API_KEY not found in environment.')
    print()
    print('Options:')
    print('   1. Enter your API key now (will be used only for this session)')
    print('   2. Configure permanently: export ESM_API_KEY="your_api_key"')
    print('   3. Press Enter to cancel and use another model')
    print()
    
    try:
        user_input = input('🔑 Paste your API key (or Enter to cancel): ').strip()
    except (EOFError, KeyboardInterrupt):
        print()
        print('❌ Operation cancelled by user.')
        return False
    
    if not user_input:
        print()
        print('❌ API key not provided.')
        print('💡 Tip: Use --protein-model esmc-600m-2024-12 for the local version (1152-dim)')
        print('        or --protein-model esm2_t33_650M_UR50D for ESM-2 (1280-dim)')
        return False
    
    # Configure API key for this session
    os.environ['ESM_API_KEY'] = user_input
    print()
    print(f'✅ API key configured for this session (***{user_input[-4:]})')
    print()
    
    return True

from integrated_pipeline import IntegratedPipeline, IntegratedConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DockTKinase - Complete Integrated Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with small dataset
  python scripts/run_complete_pipeline.py --input tests/datasets/kinase_non_human_compounds.tsv
  
  # Production with full dataset
  python scripts/run_complete_pipeline.py \\
      --input tests/datasets/kinase_all_compounds.tsv \\
      --output results/production_run \\
      --device cuda
  
  # Regression only (skip classification)
  python scripts/run_complete_pipeline.py \\
      --input data.tsv \\
      --no-classification
  
  # Specific models
  python scripts/run_complete_pipeline.py \\
      --input data.tsv \\
      --classification-models RandomForest GradientBoosting \\
      --regression-models RandomForest XGBoost

Checkpoints:
  By default, the pipeline saves checkpoints after each phase.
  In subsequent runs, completed phases are loaded from cache.
  Use --no-checkpoints to force full recalculation.

Devices:
  - auto: Detects automatically (GPU if available)
  - cpu: Force CPU
  - cuda: NVIDIA GPU (requires CUDA)
  - mps: Apple Silicon GPU (Mac M1/M2/M3)
        """
    )
    
    # Input/Output
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input TSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/pipeline_output',
        help='Directory to save results (default: results/pipeline_output)'
    )
    
    # Build (Phase 1)
    parser.add_argument(
        '--ligand-model',
        type=str,
        default='SMI-TED',
        choices=['SMI-TED', 'MOLFORMER', 'CHEMBERTA'],
        help='FM4M model for ligand embeddings (default: SMI-TED, 768-dim; CHEMBERTA: 384-dim)'
    )
    
    parser.add_argument(
        '--protein-model',
        type=str,
        default='esm2_t6_8M_UR50D',
        choices=['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D',
                 'esm2_t33_650M_UR50D', 'esm2_t36_3B_UR50D', 'esm2_t48_15B_UR50D',
                 'esmc-300m-2024-12', 'esmc-600m-2024-12', 'esmc-6b-2024-12',
                 'boltz2'],
        help='Model for protein embeddings (default: esm2_t6_8M_UR50D). esmc-6b requires ESM_API_KEY.'
    )
    
    parser.add_argument(
        '--protein-dim',
        type=int,
        default=None,
        help='Protein embedding dimension (e.g.: 320, 384, 480, 640, 1280). Default: automatic based on model'
    )
    
    parser.add_argument(
        '--api',
        type=str,
        default=None,
        help='API key for models that require remote access (e.g.: ESM-C 6B via Forge API)'
    )
    
    # Classification (Phase 2)
    parser.add_argument(
        '--no-classification',
        action='store_true',
        help='Skip classification phase'
    )
    
    parser.add_argument(
        '--classification-models',
        type=str,
        nargs='+',
        default=None,
        help='Specific classification models (default: all 10)'
    )
    
    # Regression (Phase 3)
    parser.add_argument(
        '--no-regression',
        action='store_true',
        help='Skip regression phase'
    )
    
    parser.add_argument(
        '--regression-models',
        type=str,
        nargs='+',
        default=None,
        help='Specific regression models (default: all 10)'
    )
    
    # General settings
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Computing device (default: auto)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.1,
        help='Test set proportion (default: 0.1 = 10%%)'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Validation set proportion (default: 0.1 = 10%%)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Stratification options
    parser.add_argument(
        '--stratifier-threshold',
        type=float,
        default=None,
        help='Manual threshold for stratification clustering (0.0-1.0). If not specified, uses auto-threshold.'
    )
    
    parser.add_argument(
        '--stratifier-method',
        type=str,
        default='target',
        choices=['silhouette', 'elbow', 'target', 'percentile', 'leakage_aware'],
        help='Method to determine automatic threshold (default: target). leakage_aware optimizes train/val/test separation. Ignored if --stratifier-threshold is specified.'
    )
    
    parser.add_argument(
        '--no-checkpoints',
        action='store_true',
        help='Disable checkpoint system (force recalculation)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Quiet mode (less output)'
    )
    
    # Embedding directories (reuse of pre-computed embeddings)
    parser.add_argument(
        '--protein-embeddings-dir',
        type=str,
        default=None,
        help='Directory with pre-computed protein embeddings. If specified and exists, skips generation.'
    )
    
    parser.add_argument(
        '--ligand-embeddings-dir',
        type=str,
        default=None,
        help='Directory with pre-computed ligand embeddings. Can be shared across experiments with different protein models.'
    )
    
    # Attention Matrix options
    parser.add_argument(
        '--save-matrices',
        action='store_true',
        help='Save per-residue/per-token embedding matrices (for Cross-Attention model). Generates protein_matrices/ and ligand_matrices/.'
    )
    
    return parser.parse_args()


def main():
    """Execute complete pipeline."""
    args = parse_args()
    
    print('=' * 80)
    print('🧬 DOCKTKINASE - COMPLETE INTEGRATED PIPELINE')
    print('=' * 80)
    print()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f'❌ Error: File not found: {args.input}')
        return 1
    
    print('📋 Configuration:')
    print(f'   Input:  {args.input}')
    print(f'   Output: {args.output}')
    print(f'   Ligand Model: {args.ligand_model} (768-dim)')
    
    # Determine protein model dimension (single representation)
    protein_dims = {
        # ESM-2 models (mean pooling)
        'esm2_t6_8M_UR50D': 320,
        'esm2_t12_35M_UR50D': 480,
        'esm2_t30_150M_UR50D': 640,
        'esm2_t33_650M_UR50D': 1280,
        'esm2_t36_3B_UR50D': 2560,
        'esm2_t48_15B_UR50D': 5120,
        # ESM-C models (ESM-3, CLS token)
        'esmc-300m-2024-12': 960,
        'esmc-600m-2024-12': 1152,
        'esmc-6b-2024-12': 3072,
        # Boltz-2 (single representation, mean pooling)
        'boltz2': 384
    }
    
    # Use custom dimension if provided, otherwise use model default
    if args.protein_dim is not None:
        protein_dim = args.protein_dim
        dim_source = 'custom'
    else:
        protein_dim = protein_dims.get(args.protein_model, 320)
        dim_source = 'default'
    
    ligand_dim = 768  # FM4M SMI-TED fixed
    total_dim = ligand_dim + protein_dim
    
    # Detect actual device
    import torch
    if args.device == 'auto':
        if torch.cuda.is_available():
            detected_device = f"cuda ({torch.cuda.get_device_name()})"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            detected_device = "mps (Apple Silicon GPU)"
        else:
            detected_device = "cpu"
    else:
        detected_device = args.device
    
    print(f'   Protein Model: {args.protein_model} ({protein_dim}-dim {dim_source})')
    print(f'   Total Embedding: {total_dim}-dim (Ligand: {ligand_dim} + Protein: {protein_dim})')
    print(f'   Device: {args.device} → {detected_device}')
    print(f'   Checkpoints: {"❌ Disabled" if args.no_checkpoints else "✅ Enabled"}')
    
    # Embedding directories (reuse)
    if args.protein_embeddings_dir:
        print(f'   📂 Protein Embeddings: {args.protein_embeddings_dir} (pre-computed)')
    if args.ligand_embeddings_dir:
        print(f'   📂 Ligand Embeddings: {args.ligand_embeddings_dir} (pre-computed/shared)')
    
    # Stratification settings
    if args.stratifier_threshold is not None:
        print(f'   Stratification: Manual threshold = {args.stratifier_threshold}')
    else:
        print(f'   Stratification: Auto-threshold (method={args.stratifier_method})')
    
    # Validate API key for models that require Forge API (ESM-C 6B)
    if args.protein_model in FORGE_API_MODELS:
        print(f'   ⚠️  Model requires API: Forge API (EvolutionaryScale)')
        if args.api:
            print(f'   🔑 API key: provided via --api')
    print()
    
    if not validate_forge_api_key(args.protein_model, args.api):
        print()
        print('❌ Pipeline cancelled: API key not provided for ESM-C 6B model.')
        print('   Use --protein-model with another locally available model.')
        return 1
    
    # Determine which phases to execute
    run_classification = not args.no_classification
    run_regression = not args.no_regression
    
    if not run_classification and not run_regression:
        print('❌ Error: At least one phase must be enabled (classification or regression)')
        return 1
    
    phases = []
    if run_classification:
        n_clf_models = len(args.classification_models) if args.classification_models else 10
        phases.append(f'Classification ({n_clf_models} models)')
    if run_regression:
        n_reg_models = len(args.regression_models) if args.regression_models else 10
        phases.append(f'Regression ({n_reg_models} models)')
    
    print(f'📊 Phases to execute:')
    print(f'   • Build (always required)')
    for phase in phases:
        print(f'   • {phase}')
    print()
    
    # Pipeline configuration
    config = IntegratedConfig(
        # Input/Output
        input_tsv=str(input_path),
        output_dir=args.output,
        
        # Build (Phase 1)
        ligand_model=args.ligand_model,
        esm_model=args.protein_model,
        esm_dim=args.protein_dim,  # Custom dimension (None = use model default)
        
        # Embedding directories (reuse)
        protein_embeddings_dir=args.protein_embeddings_dir,
        ligand_embeddings_dir=args.ligand_embeddings_dir,
        
        # Stratification settings
        stratifier_auto_threshold=args.stratifier_threshold is None,
        stratifier_threshold=args.stratifier_threshold,
        stratifier_method=args.stratifier_method,
        
        # Classification (Phase 2)
        run_classification=run_classification,
        use_multi_model_classification=True,  # Always use multi-model
        classification_models=args.classification_models,
        
        # Regression (Phase 3)
        run_regression=run_regression,
        regression_models=args.regression_models,
        
        # General
        device=args.device,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        use_checkpoints=not args.no_checkpoints,
        verbose=not args.quiet
    )
    
    # Execute pipeline
    start_time = time.time()
    
    try:
        pipeline = IntegratedPipeline(config)
        results = pipeline.run()
        
        total_time = time.time() - start_time
        
        print()
        print('=' * 80)
        print('✅ PIPELINE COMPLETE - SUCCESS!')
        print('=' * 80)
        print(f'⏱️  Total Time: {total_time:.2f}s ({total_time/60:.2f} min)')
        print()
        
        # Results summary
        if results.get('status') == 'completed':
            # Build
            if 'build' in results and results['build'].get('success'):
                build = results['build']
                print('📦 BUILD:')
                print(f'   ✅ Samples processed: {build.get("n_samples", 0):,}')
                print(f'   ✅ Final dimension: {build.get("embedding_dim", 0)}')
                print()
            
            # Classification
            if run_classification and 'classifier' in results:
                clf = results['classifier']
                if clf.get('success'):
                    print('📊 CLASSIFICATION:')
                    print(f'   ✅ Models trained: {clf.get("n_models_trained", 0)}')
                    if 'best_model' in clf:
                        print(f'   🏆 Best model: {clf["best_model"]}')
                        if 'best_metrics' in clf:
                            metrics = clf['best_metrics']
                            print(f'      ROC-AUC: {metrics.get("ROC_AUC", 0):.4f}')
                            print(f'      F1: {metrics.get("F1", 0):.4f}')
                            print(f'      Accuracy: {metrics.get("Accuracy", 0):.4f}')
                    print()
            
            # Regression
            if run_regression and 'regression' in results:
                reg = results['regression']
                if reg.get('success'):
                    print('📈 REGRESSION:')
                    print(f'   ✅ Models trained: {reg.get("models_trained", 0)}')
                    if 'best_model' in reg:
                        print(f'   🏆 Best model: {reg["best_model"]} (selected by test performance)')
                        # Show validation and test metrics separately
                        if 'best_val_mae' in reg:
                            print(f'      📊 Validation: MAE={reg.get("best_val_mae", 0):.2f} nM, R²={reg.get("best_val_r2", 0):.4f}')
                            print(f'      🎯 Test:       MAE={reg.get("best_test_mae", 0):.2f} nM, R²={reg.get("best_test_r2", 0):.4f}')
                        else:
                            # Fallback for old format
                            print(f'      MAE: {reg.get("best_mae", 0):.2f} nM')
                            print(f'      R²: {reg.get("best_r2", 0):.4f}')
                    print()
        
        print(f'📁 Results saved to: {args.output}')
        print('=' * 80)
        
        return 0
        
    except Exception as e:
        print()
        print('=' * 80)
        print('❌ PIPELINE ERROR')
        print('=' * 80)
        print(f'Error: {str(e)}')
        
        import traceback
        if not args.quiet:
            print()
            print('Full traceback:')
            traceback.print_exc()
        
        return 1


if __name__ == '__main__':
    sys.exit(main())
