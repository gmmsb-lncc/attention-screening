#!/usr/bin/env python3
"""
Run Ablation Study for Both Datasets (non_human and human)

This script orchestrates the complete ablation study pipeline for both datasets:
- Non-human kinases: results_non_human/
- Human kinases: results_human/

Usage:
    python run_ablation_study.py --dataset non_human  # Run for non-human
    python run_ablation_study.py --dataset human      # Run for human
    python run_ablation_study.py --dataset both       # Run for both (sequential)

Author: DockTKinase Team
Date: January 17, 2026
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

DATASET_CONFIGS = {
    'non_human': {
        'tsv_path': '/media/leon/ssd2tb/docktkinase/tests/datasets/kinase_non_human_compounds.tsv',
        'embeddings_dir': '/media/leon/ssd2tb/docktkinase/results/protein_model_benchmark_non_human_v2',
        'results_suffix': 'results_non_human',
        'description': 'Non-human kinases (15,616 interactions, 299 proteins)'
    },
    'human': {
        'tsv_path': '/data/docktkinase/datasets/kinase_human_compounds.tsv',
        'embeddings_dir': '/data/docktkinase/results/protein_model_benchmark_human_v2',
        'results_suffix': 'results_human',
        'description': 'Human kinases'
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_command(cmd: List[str], description: str) -> int:
    """Run a command and return exit code."""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n✅ {description} completed successfully")
    else:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
    
    return result.returncode


def verify_paths(config: Dict) -> bool:
    """Verify that required paths exist."""
    tsv_path = Path(config['tsv_path'])
    embeddings_dir = Path(config['embeddings_dir'])
    
    print(f"\nVerifying paths for {config['description']}:")
    print(f"  TSV: {tsv_path}")
    print(f"  Embeddings: {embeddings_dir}")
    
    if not tsv_path.exists():
        print(f"  ❌ TSV file not found: {tsv_path}")
        return False
    
    if not embeddings_dir.exists():
        print(f"  ❌ Embeddings directory not found: {embeddings_dir}")
        return False
    
    print("  ✅ All paths exist")
    return True


# =============================================================================
# PIPELINE RUNNERS
# =============================================================================

def run_classification_pipeline(dataset: str, config: Dict, python_exe: str) -> bool:
    """Run complete classification pipeline for a dataset."""
    print(f"\n{'#'*70}")
    print(f"# CLASSIFICATION PIPELINE: {config['description']}")
    print(f"{'#'*70}\n")
    
    scripts_dir = Path(__file__).parent / 'classification' / 'scripts'
    
    # Script sequence with environment variables
    env_vars = {
        'ABLATION_TSV_PATH': config['tsv_path'],
        'ABLATION_EMBEDDINGS_DIR': config['embeddings_dir'],
        'ABLATION_RESULTS_SUFFIX': config['results_suffix']
    }
    
    env_str = ' '.join([f'{k}={v}' for k, v in env_vars.items()])
    
    scripts = [
        ('01_extract_data.py', 'Extract proteins, ligands, interactions'),
        ('02_generate_morgan_fingerprints.py', 'Generate Morgan fingerprints'),
        ('03_generate_aac_dpc_encoding.py', 'Generate One-Hot protein encoding'),
        ('04_create_combinations.py', 'Create C1-C4 representation combinations'),
        ('05_run_classification.py', 'Run KNN + MLP classifiers (5 seeds × 10 combinations)'),
        ('06_visualize_results.py', 'Generate visualization plots'),
    ]
    
    for script_name, description in scripts:
        script_path = scripts_dir / script_name
        
        # Pass environment variables via command line arguments
        cmd = [
            python_exe,
            str(script_path),
            '--tsv-path', config['tsv_path'],
            '--embeddings-dir', config['embeddings_dir'],
            '--results-suffix', config['results_suffix']
        ]
        
        exit_code = run_command(cmd, f"{description} ({dataset})")
        
        if exit_code != 0:
            print(f"\n❌ Classification pipeline failed at {script_name}")
            return False
    
    print(f"\n✅ Classification pipeline completed for {dataset}")
    return True


def run_regression_pipeline(dataset: str, config: Dict, python_exe: str) -> bool:
    """Run complete regression pipeline for a dataset."""
    print(f"\n{'#'*70}")
    print(f"# REGRESSION PIPELINE: {config['description']}")
    print(f"{'#'*70}\n")
    
    scripts_dir = Path(__file__).parent / 'regression' / 'scripts'
    
    scripts = [
        ('01_extract_data_regression.py', 'Extract data for regression'),
        ('02_run_regression.py', 'Run KNN + MLP regressors (5 seeds × 3 models)'),
        ('consolidate_checkpoints.py', 'Consolidate checkpoint files'),
        ('03_visualize_regression_results.py', 'Generate visualization plots'),
    ]
    
    for script_name, description in scripts:
        script_path = scripts_dir / script_name
        
        cmd = [
            python_exe,
            str(script_path),
            '--tsv-path', config['tsv_path'],
            '--embeddings-dir', config['embeddings_dir'],
            '--results-suffix', config['results_suffix']
        ]
        
        exit_code = run_command(cmd, f"{description} ({dataset})")
        
        if exit_code != 0:
            print(f"\n❌ Regression pipeline failed at {script_name}")
            return False
    
    print(f"\n✅ Regression pipeline completed for {dataset}")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run ablation study for non-human and/or human kinase datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for non-human dataset only
  python run_ablation_study.py --dataset non_human
  
  # Run for human dataset only
  python run_ablation_study.py --dataset human
  
  # Run for both datasets sequentially
  python run_ablation_study.py --dataset both
  
  # Run only classification
  python run_ablation_study.py --dataset non_human --task classification
  
  # Run only regression
  python run_ablation_study.py --dataset human --task regression
        """
    )
    
    parser.add_argument(
        '--dataset',
        choices=['non_human', 'human', 'both'],
        default='non_human',
        help='Which dataset to process (default: non_human)'
    )
    
    parser.add_argument(
        '--task',
        choices=['classification', 'regression', 'both'],
        default='both',
        help='Which task to run (default: both)'
    )
    
    parser.add_argument(
        '--python',
        default=sys.executable,
        help='Python executable to use (default: current interpreter)'
    )
    
    args = parser.parse_args()
    
    # Determine datasets to process
    if args.dataset == 'both':
        datasets = ['non_human', 'human']
    else:
        datasets = [args.dataset]
    
    # Print configuration
    print("="*70)
    print("ABLATION STUDY RUNNER")
    print("="*70)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Tasks: {args.task}")
    print(f"Python: {args.python}")
    print("="*70)
    
    # Process each dataset
    success = True
    for dataset in datasets:
        config = DATASET_CONFIGS[dataset]
        
        print(f"\n{'='*70}")
        print(f"Processing: {config['description']}")
        print(f"{'='*70}")
        
        # Verify paths
        if not verify_paths(config):
            print(f"\n❌ Skipping {dataset} due to missing paths")
            success = False
            continue
        
        # Run classification
        if args.task in ['classification', 'both']:
            if not run_classification_pipeline(dataset, config, args.python):
                success = False
                if args.dataset != 'both':
                    break  # Stop if single dataset
        
        # Run regression
        if args.task in ['regression', 'both']:
            if not run_regression_pipeline(dataset, config, args.python):
                success = False
                if args.dataset != 'both':
                    break  # Stop if single dataset
    
    # Final summary
    print("\n" + "="*70)
    if success:
        print("✅ ALL PIPELINES COMPLETED SUCCESSFULLY")
    else:
        print("❌ SOME PIPELINES FAILED - CHECK LOGS ABOVE")
    print("="*70)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
