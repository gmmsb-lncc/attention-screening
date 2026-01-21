"""Summary and result saving utilities."""

import os
import json
from typing import Dict
from datetime import datetime
import numpy as np

from ..config import ENCODER_TYPES


def save_results(
    results: Dict,
    embedding_name: str,
    dataset_type: str,
    output_dir: str
) -> str:
    """
    Save results to JSON file.

    Args:
        results: Results dictionary
        embedding_name: Embedding model name
        dataset_type: Dataset type
        output_dir: Directory to save results

    Returns:
        Path to saved file
    """
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'embedding': embedding_name,
            'dataset': dataset_type,
            'encoders': ENCODER_TYPES
        },
        'results': results
    }

    os.makedirs(output_dir, exist_ok=True)
    short_name = embedding_name.replace('esm2_', '').replace('_UR50D', '')
    filepath = os.path.join(output_dir, f'{dataset_type}_{short_name}_encoder_comparison.json')

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved: {filepath}")
    return filepath


def print_summary(results: Dict) -> None:
    """
    Print summary table with mean ± std.

    Args:
        results: Results dictionary
    """
    scenarios = list(list(results.values())[0].keys())
    encoders = list(results.keys())

    _print_header()
    _print_table_header(scenarios)
    _print_table_data(results, encoders, scenarios)
    _print_best_encoders(results, encoders)


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
