"""Plotting utilities for CrossAttention analysis results."""

import os
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12


def plot_results(
    all_results: Dict,
    split_stats: Dict,
    embedding_name: str,
    output_dir: str,
    prefix: str
) -> str:
    """
    Generate comparison plots for CrossAttention results.

    Creates a figure with:
    1. Accuracy comparison across scenarios
    2. MCC comparison with drop percentage annotation

    Args:
        all_results: Results dictionary by scenario
        split_stats: Split statistics by scenario
        embedding_name: Name of embedding model
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        Path to saved plot
    """
    scenarios = list(all_results.keys())

    # Extract metrics
    accs = [all_results[s]['CNN+CrossAttn']['accuracy'] for s in scenarios]
    mccs = [all_results[s]['CNN+CrossAttn']['mcc'] for s in scenarios]
    test_sizes = [split_stats[s]['test_size'] for s in scenarios]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(scenarios))
    width = 0.6

    # Plot 1: Accuracy
    ax1 = axes[0]
    bars1 = ax1.bar(x, accs, width, color='#2ecc71', edgecolor='black', linewidth=1.5)

    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'CNN+CrossAttention Accuracy\n({embedding_name})',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=10)
    ax1.set_ylim(0, 1.1)

    # Annotate accuracy bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(
            f'{height:.3f}',
            xy=(bar.get_x() + bar.get_width()/2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )

    # Annotate test sizes
    for i, ts in enumerate(test_sizes):
        ax1.annotate(
            f'n={ts}',
            xy=(i, 0),
            xytext=(0, -20),
            textcoords="offset points",
            ha='center', va='top',
            fontsize=9, color='gray'
        )

    # Plot 2: MCC
    ax2 = axes[1]
    bars2 = ax2.bar(x, mccs, width, color='#9b59b6', edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('MCC', fontsize=12)
    ax2.set_title(f'CNN+CrossAttention MCC\n({embedding_name})',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=10)
    ax2.set_ylim(0, 1.0)

    # Annotate MCC bars
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(
            f'{height:.3f}',
            xy=(bar.get_x() + bar.get_width()/2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )

    # Add MCC drop annotation
    if len(mccs) >= 2 and mccs[0] > 0:
        drop_pct = 100 * (mccs[0] - mccs[-1]) / mccs[0]
        ax2.annotate(
            f'↓ {drop_pct:.0f}%',
            xy=(len(scenarios)-1, mccs[-1]),
            xytext=(30, 50),
            textcoords="offset points",
            ha='center',
            fontsize=14, fontweight='bold', color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2)
        )

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{prefix}crossattention_split_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved: {filepath}")
    return filepath
