"""Plotting utilities for encoder comparison results."""

import os
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(
    results: Dict,
    embedding_name: str,
    dataset_type: str,
    output_dir: str
) -> None:
    """
    Create comparison plots for encoder results.

    Generates two plots:
    1. MCC comparison across scenarios (with error bars)
    2. MCC degradation from easy to hard scenario

    Args:
        results: Results dictionary
        embedding_name: Embedding model name
        dataset_type: Dataset type
        output_dir: Directory to save plot
    """
    scenarios = list(list(results.values())[0].keys())
    encoders = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    _plot_mcc_comparison(axes[0], results, scenarios, encoders, embedding_name)
    _plot_mcc_degradation(axes[1], results, encoders)

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    short_name = embedding_name.replace('esm2_', '').replace('_UR50D', '')
    filepath = os.path.join(output_dir, f'{dataset_type}_{short_name}_encoder_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved: {filepath}")


def _plot_mcc_comparison(
    ax: plt.Axes,
    results: Dict,
    scenarios: list,
    encoders: list,
    embedding_name: str
) -> None:
    """Plot MCC comparison across scenarios with error bars."""
    x = np.arange(len(scenarios))
    width = 0.25

    for i, encoder in enumerate(encoders):
        mccs = [results[encoder][s]['mean']['mcc'] for s in scenarios]
        stds = [results[encoder][s]['std']['mcc'] for s in scenarios]

        bars = ax.bar(
            x + i * width, mccs, width,
            yerr=stds, capsize=5,
            label=encoder.upper(),
            error_kw={'elinewidth': 1, 'capthick': 1}
        )

        _annotate_bars(ax, bars, mccs, stds)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('MCC')
    ax.set_title(f'MCC Comparison - {embedding_name}')
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.replace(' ', '\n') for s in scenarios], fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1)


def _plot_mcc_degradation(ax: plt.Axes, results: Dict, encoders: list) -> None:
    """Plot MCC degradation from easy to hard scenario."""
    drops, drop_stds = _calculate_mcc_drops(results, encoders)

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    bar_colors = [colors[i % len(colors)] for i in range(len(encoders))]

    bars = ax.bar(
        encoders, drops,
        yerr=drop_stds, capsize=5,
        color=bar_colors,
        error_kw={'elinewidth': 1, 'capthick': 1}
    )

    _annotate_bars_percent(ax, bars, drops, drop_stds)

    ax.set_xlabel('Encoder Type')
    ax.set_ylabel('MCC Drop (%)')
    ax.set_title('MCC Degradation (Easy→Hard: Random Split → True Generalization)')
    ax.set_xticklabels([e.upper() for e in encoders])


def _calculate_mcc_drops(results: Dict, encoders: list) -> tuple:
    """Calculate MCC drops and uncertainties."""
    drops = []
    drop_stds = []

    for encoder in encoders:
        random_mcc = results[encoder]['Random Split']['mean']['mcc']
        random_std = results[encoder]['Random Split']['std']['mcc']
        hard_mcc = results[encoder]['New Comp + New Kinase']['mean']['mcc']
        hard_std = results[encoder]['New Comp + New Kinase']['std']['mcc']

        drop = 100 * (random_mcc - hard_mcc) / random_mcc if random_mcc > 0 else 0

        # Error propagation
        if random_mcc > 0:
            drop_std = 100 * np.sqrt(
                (hard_std / random_mcc) ** 2 +
                ((random_mcc - hard_mcc) * random_std / random_mcc ** 2) ** 2
            )
        else:
            drop_std = 0

        drops.append(drop)
        drop_stds.append(drop_std)

    return drops, drop_stds


def _annotate_bars(ax: plt.Axes, bars, values: list, stds: list) -> None:
    """Annotate bars with value ± std."""
    for bar, val, std in zip(bars, values, stds):
        height = bar.get_height()
        ax.annotate(
            f'{val:.3f}±{std:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            fontsize=7
        )


def _annotate_bars_percent(ax: plt.Axes, bars, values: list, stds: list) -> None:
    """Annotate bars with percentage value ± std."""
    for bar, val, std in zip(bars, values, stds):
        height = bar.get_height()
        ax.annotate(
            f'{val:.1f}±{std:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            fontsize=9,
            fontweight='bold'
        )
