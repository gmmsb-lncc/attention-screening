"""Summary and result saving utilities."""

import os
import sys
import json
import platform
import subprocess
import hashlib
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import scipy
from scipy import stats as scipy_stats

from ..utils.device import get_reproducibility_info


def _get_primary_model(scenario_block: Dict) -> tuple[str, Dict]:
    """Return model name and metrics from a scenario block."""
    if not scenario_block:
        return "UnknownModel", {}
    if 'CNN+CrossAttn' in scenario_block:
        return 'CNN+CrossAttn', scenario_block['CNN+CrossAttn']
    model_name, metrics = next(iter(scenario_block.items()))
    return model_name, metrics


# =============================================================================
# ENVIRONMENT LOGGING
# =============================================================================

def get_environment_info() -> Dict:
    """
    Collect comprehensive environment information for reproducibility.

    Returns:
        Dictionary with environment details
    """
    import torch
    import pandas as pd
    import sklearn

    env_info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
        },
        'packages': {
            'torch': torch.__version__,
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'pandas': pd.__version__,
            'sklearn': sklearn.__version__,
        },
        'pytorch': get_reproducibility_info(),
        'git': _get_git_info()
    }

    return env_info


def _get_git_info() -> Dict:
    """Get git repository information if available."""
    git_info = {'available': False}

    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        git_info['commit_hash'] = commit_hash
        git_info['commit_short'] = commit_hash[:8]

        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        git_info['branch'] = branch

        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        git_info['has_uncommitted_changes'] = len(status) > 0
        git_info['available'] = True

    except (subprocess.CalledProcessError, FileNotFoundError):
        git_info['note'] = 'Git information not available'

    return git_info


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def calculate_statistics(seed_results: Dict) -> Dict:
    """
    Calculate mean and sample std across seeds with 95% CI.

    Uses ddof=1 for sample standard deviation (Bessel's correction).
    """
    n_seeds = len(seed_results)

    metrics_arrays = {
        'accuracy': [],
        'mcc': [],
        'auc': [],
        'f1': [],
        'loss': []
    }

    for seed_data in seed_results.values():
        for metric in metrics_arrays:
            if metric in seed_data:
                metrics_arrays[metric].append(seed_data[metric])

    ddof = 1 if n_seeds > 1 else 0

    stats = {
        'n_seeds': n_seeds,
        'mean': {},
        'std': {},
    }

    for metric, values in metrics_arrays.items():
        if values:
            arr = np.array(values)
            stats['mean'][metric] = float(np.mean(arr))
            stats['std'][metric] = float(np.std(arr, ddof=ddof))

    # Add 95% CI if n >= 2
    if n_seeds >= 2:
        t_critical = scipy_stats.t.ppf(0.975, df=n_seeds - 1)
        stats['ci_95'] = {}
        for metric, values in metrics_arrays.items():
            if values:
                arr = np.array(values)
                sem = np.std(arr, ddof=1) / np.sqrt(n_seeds)
                stats['ci_95'][metric] = float(t_critical * sem)

    return stats


# =============================================================================
# RESULT SAVING
# =============================================================================

def save_results(
    all_results: Dict,
    split_stats: Dict,
    embedding_name: str,
    dataset_type: str,
    output_dir: str,
    prefix: str,
    config: Optional[Dict] = None
) -> str:
    """
    Save results to JSON file with environment info.

    Args:
        all_results: Results dictionary by scenario
        split_stats: Split statistics by scenario
        embedding_name: Embedding model name
        dataset_type: Dataset type
        output_dir: Output directory
        prefix: Filename prefix
        config: Training configuration dictionary

    Returns:
        Path to saved file
    """
    env_info = get_environment_info()
    model_name = "UnknownModel"
    if all_results:
        first_scenario_key = next(iter(all_results))
        model_name, _ = _get_primary_model(all_results[first_scenario_key])

    results_dict = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'model': model_name,
            'embedding': embedding_name,
            'dataset_type': dataset_type,
            'best_model_selection': 'best_val_mcc'
        },
        'environment': env_info,
        'config': config,
        'split_statistics': {
            k.replace('\n', ' '): v for k, v in split_stats.items()
        },
        'model_results': {
            scenario.replace('\n', ' '): metrics
            for scenario, models in all_results.items()
            for model_name, metrics in models.items()
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    filename = f'{prefix}crossattention_analysis_results.json'
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"Results saved: {filepath}")

    # Print environment summary
    git_info = env_info.get('git', {})
    if git_info.get('available'):
        print(f"  Git: {git_info.get('branch', 'unknown')}@{git_info.get('commit_short', 'unknown')}")

    return filepath


def print_summary(all_results: Dict) -> None:
    """
    Print summary of results.

    Args:
        all_results: Results dictionary by scenario
    """
    print("\n" + "-" * 50)
    print("SUMMARY")
    print("-" * 50)

    scenarios = list(all_results.keys())

    for scenario in scenarios:
        _, metrics = _get_primary_model(all_results[scenario])
        print(f"  {scenario.replace(chr(10), ' ')}: "
              f"Acc={metrics['accuracy']:.4f}, MCC={metrics['mcc']:.4f}")

    # Calculate MCC drop
    if len(scenarios) >= 2:
        _, first_metrics = _get_primary_model(all_results[scenarios[0]])
        _, last_metrics = _get_primary_model(all_results[scenarios[-1]])
        first_mcc = first_metrics.get('mcc', 0.0)
        last_mcc = last_metrics.get('mcc', 0.0)
        if first_mcc > 0:
            drop_pct = 100 * (first_mcc - last_mcc) / first_mcc
            print(f"\n  MCC drop: {first_mcc:.3f} -> {last_mcc:.3f} ({drop_pct:.0f}% drop)")
