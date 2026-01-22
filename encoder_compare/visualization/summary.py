"""Summary and result saving utilities."""

import os
import sys
import json
import platform
import subprocess
import hashlib
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import scipy
from scipy import stats as scipy_stats
import warnings

from ..config import ENCODER_TYPES
from ..utils.device import get_reproducibility_info


# =============================================================================
# ENVIRONMENT LOGGING
# =============================================================================

def get_environment_info() -> Dict:
    """
    Collect comprehensive environment information for reproducibility.

    Returns:
        Dictionary with environment details including:
        - Python version
        - Package versions (torch, numpy, scipy, pandas, sklearn)
        - CUDA/GPU information
        - Git commit hash (if in a git repo)
        - Platform information
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
            'processor': platform.processor(),
        },
        'packages': {
            'torch': torch.__version__,
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'pandas': pd.__version__,
            'sklearn': sklearn.__version__,
        },
    }

    # Add PyTorch/CUDA specific info
    env_info['pytorch'] = get_reproducibility_info()

    # Try to get git information
    env_info['git'] = _get_git_info()

    return env_info


def _get_git_info() -> Dict:
    """Get git repository information if available."""
    git_info = {'available': False}

    try:
        # Get current commit hash
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        git_info['commit_hash'] = commit_hash
        git_info['commit_short'] = commit_hash[:8]

        # Get current branch
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        git_info['branch'] = branch

        # Check for uncommitted changes
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        git_info['has_uncommitted_changes'] = len(status) > 0
        git_info['available'] = True

        if git_info['has_uncommitted_changes']:
            git_info['warning'] = (
                'Uncommitted changes detected. Results may not be '
                'exactly reproducible from this commit.'
            )

    except (subprocess.CalledProcessError, FileNotFoundError):
        git_info['available'] = False
        git_info['note'] = 'Git information not available'

    return git_info


def _get_code_hash(files: List[str] = None) -> str:
    """
    Compute hash of key source files for change detection.

    Args:
        files: List of file paths to hash. If None, uses default list.

    Returns:
        MD5 hash of concatenated file contents
    """
    if files is None:
        # Default to key experiment files
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        files = [
            os.path.join(base_dir, 'experiment.py'),
            os.path.join(base_dir, 'config.py'),
            os.path.join(base_dir, 'training', 'stable_trainer.py'),
            os.path.join(base_dir, 'training', 'evaluator.py'),
            os.path.join(base_dir, 'data', 'splits.py'),
        ]

    hasher = hashlib.md5()
    for filepath in files:
        try:
            with open(filepath, 'rb') as f:
                hasher.update(f.read())
        except FileNotFoundError:
            hasher.update(f"MISSING:{filepath}".encode())

    return hasher.hexdigest()


# =============================================================================
# STATISTICAL TESTING
# =============================================================================

def perform_pairwise_tests(
    results: Dict,
    scenario: str = 'New Comp + New Kinase',
    metric: str = 'mcc',
    alpha: float = 0.05
) -> Dict:
    """
    Perform pairwise statistical tests between all encoder pairs.

    Uses Wilcoxon signed-rank test (non-parametric, paired) which is
    appropriate for small sample sizes and doesn't assume normality.

    Args:
        results: Results dictionary with seed_results
        scenario: Scenario to compare
        metric: Metric to compare (mcc, auc, f1, accuracy)
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with test results for each pair
    """
    encoders = list(results.keys())

    # Extract seed values for each encoder
    encoder_scores = {}
    for encoder in encoders:
        seed_results = results[encoder][scenario]['seed_results']
        seeds = sorted(seed_results.keys())
        encoder_scores[encoder] = [seed_results[s][metric] for s in seeds]

    # Check all encoders have same seeds
    n_seeds = len(encoder_scores[encoders[0]])
    for encoder in encoders:
        if len(encoder_scores[encoder]) != n_seeds:
            raise ValueError(f"Encoder {encoder} has different number of seeds")

    # Perform pairwise tests
    pairwise_results = {}
    comparisons = []

    for i, enc_a in enumerate(encoders):
        for enc_b in encoders[i+1:]:
            scores_a = np.array(encoder_scores[enc_a])
            scores_b = np.array(encoder_scores[enc_b])

            # Wilcoxon signed-rank test (paired, non-parametric)
            # Requires at least 6 samples for meaningful results
            if n_seeds >= 6:
                try:
                    stat, p_value = scipy_stats.wilcoxon(scores_a, scores_b)
                    test_used = 'wilcoxon'
                except ValueError as e:
                    # All differences are zero
                    stat, p_value = 0.0, 1.0
                    test_used = 'wilcoxon (identical)'
            elif n_seeds >= 3:
                # For smaller samples, use paired t-test with warning
                stat, p_value = scipy_stats.ttest_rel(scores_a, scores_b)
                test_used = 'paired_t (n<6)'
            else:
                stat, p_value = np.nan, np.nan
                test_used = 'insufficient_samples'

            # Calculate effect size (Cohen's d for paired samples)
            diff = scores_a - scores_b
            if np.std(diff, ddof=1) > 0:
                cohens_d = np.mean(diff) / np.std(diff, ddof=1)
            else:
                cohens_d = 0.0

            comparison = {
                'encoder_a': enc_a,
                'encoder_b': enc_b,
                'mean_a': float(np.mean(scores_a)),
                'mean_b': float(np.mean(scores_b)),
                'diff': float(np.mean(diff)),
                'statistic': float(stat) if not np.isnan(stat) else None,
                'p_value': float(p_value) if not np.isnan(p_value) else None,
                'test_used': test_used,
                'cohens_d': float(cohens_d),
                'significant': p_value < alpha if not np.isnan(p_value) else None,
                'winner': enc_a if np.mean(diff) > 0 else enc_b if np.mean(diff) < 0 else 'tie'
            }

            pairwise_results[f"{enc_a}_vs_{enc_b}"] = comparison
            comparisons.append(comparison)

    # Apply Holm-Bonferroni correction for multiple comparisons
    if comparisons:
        p_values = [c['p_value'] for c in comparisons if c['p_value'] is not None]
        if p_values:
            corrected = _holm_bonferroni_correction(p_values, alpha)
            idx = 0
            for comp in comparisons:
                if comp['p_value'] is not None:
                    comp['p_value_corrected'] = corrected['adjusted_p_values'][idx]
                    comp['significant_corrected'] = corrected['reject'][idx]
                    idx += 1

    return {
        'scenario': scenario,
        'metric': metric,
        'alpha': alpha,
        'n_seeds': n_seeds,
        'pairwise': pairwise_results,
        'correction_method': 'holm-bonferroni'
    }


def _holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Dict:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    More powerful than Bonferroni while still controlling family-wise error rate.

    Args:
        p_values: List of p-values
        alpha: Significance level

    Returns:
        Dictionary with adjusted p-values and rejection decisions
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted_p = np.zeros(n)
    reject = np.zeros(n, dtype=bool)

    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        adjusted_alpha = alpha / (n - i)
        adjusted_p[idx] = min(p * (n - i), 1.0)
        reject[idx] = p < adjusted_alpha

    return {
        'adjusted_p_values': adjusted_p.tolist(),
        'reject': reject.tolist(),
        'adjusted_alpha': [alpha / (n - i) for i in range(n)]
    }


def print_statistical_comparison(results: Dict, scenario: str = 'New Comp + New Kinase') -> None:
    """
    Print statistical comparison between encoders.

    Args:
        results: Results dictionary
        scenario: Scenario to compare
    """
    test_results = perform_pairwise_tests(results, scenario, metric='mcc')

    print("\n" + "=" * 80)
    print(f"STATISTICAL COMPARISON - {scenario} (MCC)")
    print("=" * 80)
    print(f"Seeds: {test_results['n_seeds']}, Correction: {test_results['correction_method']}")
    print("-" * 80)
    print(f"{'Comparison':<25} {'Diff':<10} {'p-value':<12} {'p-adj':<12} {'Effect':<10} {'Sig?'}")
    print("-" * 80)

    for name, comp in test_results['pairwise'].items():
        diff = comp['diff']
        p_val = comp['p_value']
        p_adj = comp.get('p_value_corrected', p_val)
        effect = comp['cohens_d']
        sig = comp.get('significant_corrected', comp['significant'])

        # Format values
        p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
        p_adj_str = f"{p_adj:.4f}" if p_adj is not None else "N/A"
        sig_str = "YES *" if sig else "no"
        winner_indicator = f"({comp['winner']} better)" if diff != 0 else "(tie)"

        print(f"{name:<25} {diff:+.4f}    {p_str:<12} {p_adj_str:<12} {effect:+.2f}      {sig_str}")

    print("-" * 80)
    print("Effect size interpretation: |d| < 0.2 = small, 0.2-0.8 = medium, > 0.8 = large")
    print("* Significant after Holm-Bonferroni correction")
    print("=" * 80)


def save_results(
    results: Dict,
    embedding_name: str,
    dataset_type: str,
    output_dir: str,
    config: Dict = None
) -> str:
    """
    Save results to JSON file with statistical tests and environment info.

    Includes comprehensive metadata for reproducibility:
    - Environment information (Python, packages, CUDA)
    - Git commit hash
    - Code hash for change detection
    - Statistical tests with corrections

    Args:
        results: Results dictionary
        embedding_name: Embedding model name
        dataset_type: Dataset type
        output_dir: Directory to save results
        config: Training configuration dictionary (optional)

    Returns:
        Path to saved file
    """
    # Collect environment information
    env_info = get_environment_info()
    code_hash = _get_code_hash()

    # Perform statistical tests for all scenarios
    statistical_tests = {}
    scenarios = list(list(results.values())[0].keys())

    for scenario in scenarios:
        try:
            statistical_tests[scenario] = perform_pairwise_tests(
                results, scenario, metric='mcc'
            )
        except Exception as e:
            statistical_tests[scenario] = {'error': str(e)}

    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'embedding': embedding_name,
            'dataset': dataset_type,
            'encoders': ENCODER_TYPES,
            'code_hash': code_hash,
            'statistical_note': (
                'p_value_corrected uses Holm-Bonferroni correction for multiple comparisons. '
                'Use significant_corrected for determining statistical significance.'
            )
        },
        'environment': env_info,
        'config': config,
        'results': results,
        'statistical_tests': statistical_tests
    }

    os.makedirs(output_dir, exist_ok=True)
    short_name = embedding_name.replace('esm2_', '').replace('_UR50D', '')
    filepath = os.path.join(output_dir, f'{dataset_type}_{short_name}_encoder_comparison.json')

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved: {filepath}")

    # Print environment summary
    git_info = env_info.get('git', {})
    if git_info.get('available'):
        print(f"  Git: {git_info.get('branch', 'unknown')}@{git_info.get('commit_short', 'unknown')}")
        if git_info.get('has_uncommitted_changes'):
            print(f"  ⚠️  Warning: Uncommitted changes detected")
    print(f"  Code hash: {code_hash[:12]}...")

    return filepath


def print_summary(results: Dict) -> None:
    """
    Print summary table with mean ± std and statistical comparisons.

    Args:
        results: Results dictionary
    """
    scenarios = list(list(results.values())[0].keys())
    encoders = list(results.keys())

    _print_header()
    _print_table_header(scenarios)
    _print_table_data(results, encoders, scenarios)
    _print_best_encoders(results, encoders)

    # Print statistical comparison for the hardest scenario
    try:
        print_statistical_comparison(results, 'New Comp + New Kinase')
    except Exception as e:
        print(f"\nNote: Could not perform statistical comparison: {e}")


def _print_header() -> None:
    """Print table header."""
    print("\n" + "=" * 100)
    print("SUMMARY - MCC (PRIMARY METRIC) - mean ± std across seeds")
    print("=" * 100)


def _print_table_header(scenarios: list) -> None:
    """Print table column headers."""
    print(f"\n{'Encoder':<15}", end='')
    for s in scenarios:
        print(f"{s:<28}", end='')
    print("MCC Drop%")
    print("-" * 100)


def _print_table_data(results: Dict, encoders: list, scenarios: list) -> None:
    """Print table data rows."""
    for encoder in encoders:
        print(f"{encoder.upper():<15}", end='')

        # Print MCC for each scenario
        for s in scenarios:
            mcc_mean = results[encoder][s]['mean']['mcc']
            mcc_std = results[encoder][s]['std']['mcc']
            print(f"{mcc_mean:.4f}±{mcc_std:.4f}{'':<14}", end='')

        # Print MCC drop
        drop, drop_std = _calculate_mcc_drop(results, encoder)
        print(f"{drop:.1f}±{drop_std:.1f}%")


def _print_best_encoders(results: Dict, encoders: list) -> None:
    """Print best encoders summary."""
    print("\n" + "-" * 100)
    print("BEST ENCODERS:")

    # Best on true generalization
    best_hard_mcc = max(
        encoders,
        key=lambda e: results[e]['New Comp + New Kinase']['mean']['mcc']
    )
    hard_mcc_mean = results[best_hard_mcc]['New Comp + New Kinase']['mean']['mcc']
    hard_mcc_std = results[best_hard_mcc]['New Comp + New Kinase']['std']['mcc']

    print(f"  🏆 Best on True Generalization (New Comp + New Kinase): {best_hard_mcc.upper()}")
    print(f"      MCC = {hard_mcc_mean:.4f} ± {hard_mcc_std:.4f}")

    # Smallest drop (best robustness)
    best_generalization = min(
        encoders,
        key=lambda e: _calculate_mcc_drop(results, e)[0]
    )
    print(f"  📉 Smallest MCC Drop (best robustness): {best_generalization.upper()}")


def _calculate_mcc_drop(results: Dict, encoder: str) -> tuple:
    """
    Calculate MCC drop and uncertainty for an encoder.

    Args:
        results: Results dictionary
        encoder: Encoder name

    Returns:
        Tuple of (drop_percentage, drop_std)
    """
    random_mcc = results[encoder]['Random Split']['mean']['mcc']
    random_std = results[encoder]['Random Split']['std']['mcc']
    hard_mcc = results[encoder]['New Comp + New Kinase']['mean']['mcc']
    hard_std = results[encoder]['New Comp + New Kinase']['std']['mcc']

    if random_mcc > 0:
        drop = 100 * (random_mcc - hard_mcc) / random_mcc
        drop_std = 100 * np.sqrt(
            (hard_std / random_mcc) ** 2 +
            ((random_mcc - hard_mcc) * random_std / random_mcc ** 2) ** 2
        )
    else:
        drop = 0
        drop_std = 0

    return drop, drop_std
