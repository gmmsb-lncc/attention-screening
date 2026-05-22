#!/usr/bin/env python3
"""
Simple ablation study runner with automatic dataset detection.

Usage:
    cd ${PROJECT_ROOT}/results/protein_model_benchmark_non_human_v2
    python /path/to/ablation/ablation.py
    
    OR
    
    cd /data/docktkinase/results/protein_model_benchmark_human_v2
    python /path/to/ablation/ablation.py

The script automatically detects the dataset based on current directory.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

DATASET_CONFIGS = {
    'non_human': {
        'embeddings_dir': '${PROJECT_ROOT}/results/protein_model_benchmark_non_human_v2',
        'tsv_path': '${PROJECT_ROOT}/tests/datasets/kinase_non_human_compounds.tsv',
        'results_suffix': 'results_non_human',
        'name': 'Non-Human Kinases'
    },
    'human': {
        'embeddings_dir': '/data/docktkinase/results/protein_model_benchmark_human_v2',
        'tsv_path': '/data/docktkinase/tests/datasets/kinase_human_compounds.tsv',
        'results_suffix': 'results_human',
        'name': 'Human Kinases'
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_dataset_from_cwd() -> Optional[str]:
    """Detect dataset based on current working directory."""
    cwd = os.getcwd()
    
    # Check if we're in a known embeddings directory
    for dataset_key, config in DATASET_CONFIGS.items():
        if cwd.startswith(config['embeddings_dir']):
            return dataset_key
    
    # Check if directory name contains hints
    if 'non_human' in cwd.lower():
        return 'non_human'
    elif 'human' in cwd.lower() and 'non_human' not in cwd.lower():
        return 'human'
    
    return None


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_step(step_num: int, total: int, description: str):
    """Print step information."""
    print(f"\n[{step_num}/{total}] {description}")
    print("-" * 70)


def run_command(cmd: list, description: str, cwd: Optional[Path] = None) -> bool:
    """Run a command and return success status."""
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with exit code {e.returncode}")
        if e.stdout:
            print(f"stdout: {e.stdout[:500]}")
        if e.stderr:
            print(f"stderr: {e.stderr[:500]}")
        return False


# =============================================================================
# PIPELINE RUNNERS
# =============================================================================

def run_classification_pipeline(config: Dict, python_exe: str, base_dir: Path) -> bool:
    """Run classification pipeline."""
    print_header(f"CLASSIFICATION: {config['name']}")
    
    scripts_dir = base_dir / 'classification' / 'scripts'
    results_suffix = config['results_suffix']
    
    steps = [
        ("Extract data", '01_extract_data.py'),
        ("Generate Morgan fingerprints", '02_generate_morgan_fingerprints.py'),
        ("Generate protein One-Hot encoding", '03_generate_aac_dpc_encoding.py'),
        ("Create combinations C1-C4", '04_create_combinations.py'),
        ("Run classifiers (KNN + MLP)", '05_run_classification.py'),
        ("Generate visualizations", '06_visualize_results.py'),
    ]
    
    total_steps = len(steps)
    
    for i, (description, script) in enumerate(steps, 1):
        print_step(i, total_steps, description)
        
        cmd = [
            python_exe,
            str(scripts_dir / script),
            '--tsv-path', config['tsv_path'],
            '--embeddings-dir', config['embeddings_dir'],
            '--results-suffix', results_suffix
        ]
        
        if not run_command(cmd, description):
            return False
    
    print_header(f"✓ Classification Complete: {config['name']}")
    return True


def run_regression_pipeline(config: Dict, python_exe: str, base_dir: Path) -> bool:
    """Run regression pipeline."""
    print_header(f"REGRESSION: {config['name']}")
    
    scripts_dir = base_dir / 'regression' / 'scripts'
    results_suffix = config['results_suffix']
    
    steps = [
        ("Extract data for regression", '01_extract_data_regression.py'),
        ("Run regressors (KNN + MLP)", '02_run_regression.py'),
        ("Consolidate checkpoints", 'consolidate_checkpoints.py'),
        ("Generate visualizations", '03_visualize_regression_results.py'),
    ]
    
    total_steps = len(steps)
    
    for i, (description, script) in enumerate(steps, 1):
        print_step(i, total_steps, description)
        
        cmd = [
            python_exe,
            str(scripts_dir / script),
            '--tsv-path', config['tsv_path'],
            '--embeddings-dir', config['embeddings_dir'],
            '--results-suffix', results_suffix
        ]
        
        # Special handling for long-running regression experiments
        if script == '02_run_regression.py':
            print("⚠️  This step may take 3-6 hours. Running in foreground...")
            print("    Press Ctrl+C to cancel and run manually with nohup")
        
        if not run_command(cmd, description):
            return False
    
    print_header(f"✓ Regression Complete: {config['name']}")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Simple ablation study runner with auto-detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect from current directory
  cd ${PROJECT_ROOT}/results/protein_model_benchmark_non_human_v2
  python ablation.py
  
  # Explicit dataset selection
  python ablation.py --dataset human
  
  # Run only classification
  python ablation.py --task classification
  
  # Skip already-completed tasks
  python ablation.py --skip classification
        """
    )
    
    parser.add_argument(
        '--dataset',
        choices=['non_human', 'human', 'auto'],
        default='auto',
        help='Dataset to process (default: auto-detect from cwd)'
    )
    
    parser.add_argument(
        '--task',
        choices=['classification', 'regression', 'both'],
        default='both',
        help='Which task to run (default: both)'
    )
    
    parser.add_argument(
        '--skip',
        choices=['classification', 'regression', 'none'],
        default='none',
        help='Skip a specific task (useful if already completed)'
    )
    
    parser.add_argument(
        '--python',
        default=sys.executable,
        help='Python executable (default: current interpreter)'
    )
    
    args = parser.parse_args()
    
    # Determine base directory (ablation/)
    script_dir = Path(__file__).parent
    base_dir = script_dir
    
    # Auto-detect or use specified dataset
    if args.dataset == 'auto':
        detected = detect_dataset_from_cwd()
        if detected:
            dataset = detected
            print(f"✓ Auto-detected dataset: {dataset}")
        else:
            print("✗ Could not auto-detect dataset from current directory")
            print("  Please specify with --dataset non_human or --dataset human")
            print(f"  Current directory: {os.getcwd()}")
            return 1
    else:
        dataset = args.dataset
    
    config = DATASET_CONFIGS[dataset]
    
    # Print configuration
    print_header("ABLATION STUDY CONFIGURATION")
    print(f"Dataset: {config['name']}")
    print(f"TSV: {config['tsv_path']}")
    print(f"Embeddings: {config['embeddings_dir']}")
    print(f"Results: {config['results_suffix']}")
    print(f"Task: {args.task}")
    print(f"Python: {args.python}")
    
    # Verify paths exist
    if not Path(config['tsv_path']).exists():
        print(f"\n✗ TSV file not found: {config['tsv_path']}")
        return 1
    
    if not Path(config['embeddings_dir']).exists():
        print(f"\n✗ Embeddings directory not found: {config['embeddings_dir']}")
        return 1
    
    print("\n✓ All paths verified")
    
    # Run tasks
    success = True
    
    if args.task in ['classification', 'both'] and args.skip != 'classification':
        if not run_classification_pipeline(config, args.python, base_dir):
            success = False
    
    if args.task in ['regression', 'both'] and args.skip != 'regression':
        if not run_regression_pipeline(config, args.python, base_dir):
            success = False
    
    # Final summary
    print_header("SUMMARY")
    
    if success:
        print("✓ All tasks completed successfully!")
        print(f"\nResults location:")
        print(f"  Classification: {base_dir}/classification/{config['results_suffix']}/")
        print(f"  Regression: {base_dir}/regression/{config['results_suffix']}/")
        print(f"\nFigures:")
        print(f"  {base_dir}/classification/{config['results_suffix']}/figures/")
        print(f"  {base_dir}/regression/{config['results_suffix']}/figures/")
        return 0
    else:
        print("✗ Some tasks failed. Check output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
