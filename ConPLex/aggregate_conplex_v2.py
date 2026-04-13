#!/usr/bin/env python3
"""
aggregate_conplex_v2.py — Aggregate ConPLex v2 results across repetitions.

Computes mean ± std for all metrics at both τ=0.5 (v1) and τ* (v2),
and prints a comparison table ready for thesis integration.

Usage:
    python aggregate_conplex_v2.py --results-dir results_v2

Output: prints summary table and saves aggregate_v2.json
"""

import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser


def aggregate_dataset(results_dir: Path, dataset: str) -> dict:
    """Aggregate results across reps for a given dataset."""
    reps = []
    for rep in range(10):  # support up to 10 reps
        result_file = results_dir / f"conplex_v2_{dataset}_rep{rep}" / "results_v2.json"
        if result_file.exists():
            with open(result_file) as f:
                reps.append(json.load(f))
    
    if not reps:
        return None
    
    metrics = ['MCC', 'AUROC', 'F1', 'Accuracy', 'Precision', 'Recall']
    
    # Collect metrics
    v1_metrics = {m: [] for m in metrics}
    v2_metrics = {m: [] for m in metrics}
    thresholds = []
    
    for r in reps:
        for m in metrics:
            v1_metrics[m].append(r['test']['at_fixed_0.5'][m])
            v2_metrics[m].append(r['test']['at_calibrated'][m])
        thresholds.append(r['optimal_threshold'])
    
    # Compute stats
    result = {
        'dataset': dataset,
        'n_reps': len(reps),
        'optimal_thresholds': thresholds,
        'threshold_mean': round(np.mean(thresholds), 4),
        'threshold_std': round(np.std(thresholds), 4),
        'v1_fixed_0.5': {},
        'v2_calibrated': {},
        'improvement': {},
    }
    
    for m in metrics:
        v1_arr = np.array(v1_metrics[m])
        v2_arr = np.array(v2_metrics[m])
        
        result['v1_fixed_0.5'][m] = {
            'mean': round(float(np.mean(v1_arr)), 4),
            'std': round(float(np.std(v1_arr)), 4),
        }
        result['v2_calibrated'][m] = {
            'mean': round(float(np.mean(v2_arr)), 4),
            'std': round(float(np.std(v2_arr)), 4),
        }
        result['improvement'][m] = {
            'delta': round(float(np.mean(v2_arr) - np.mean(v1_arr)), 4),
            'relative_pct': round(float(
                (np.mean(v2_arr) - np.mean(v1_arr)) / max(abs(np.mean(v1_arr)), 1e-8) * 100
            ), 1),
        }
    
    return result


def print_comparison_table(agg: dict):
    """Print a formatted comparison table."""
    ds = agg['dataset'].replace('_', ' ').title()
    
    print(f"\n{'='*75}")
    print(f" ConPLex: {ds} — v1 (τ=0.5) vs v2 (τ* from val)")
    print(f" {agg['n_reps']} reps | τ* = {agg['threshold_mean']:.4f} ± {agg['threshold_std']:.4f}")
    print(f"{'='*75}")
    print(f"  {'Metric':<12} {'v1 (τ=0.5)':>18} {'v2 (τ* val)':>18} {'Δ':>10}")
    print(f"  {'-'*12} {'-'*18} {'-'*18} {'-'*10}")
    
    metrics = ['MCC', 'AUROC', 'F1', 'Accuracy', 'Precision', 'Recall']
    for m in metrics:
        v1 = agg['v1_fixed_0.5'][m]
        v2 = agg['v2_calibrated'][m]
        delta = agg['improvement'][m]['delta']
        
        v1_str = f"{v1['mean']:.3f} ± {v1['std']:.3f}"
        v2_str = f"{v2['mean']:.3f} ± {v2['std']:.3f}"
        
        print(f"  {m:<12} {v1_str:>18} {v2_str:>18} {delta:>+10.3f}")
    
    print(f"{'='*75}")
    
    # LaTeX-ready line for thesis
    print(f"\n  LaTeX (for thesis table — v2 calibrated):")
    for m in metrics:
        v2 = agg['v2_calibrated'][m]
        mean_str = f"{v2['mean']:.3f}".replace('.', '{,}')
        std_str = f"{v2['std']:.3f}".replace('.', '{,}')
        print(f"    {m:<12} & ${mean_str} \\pm {std_str}$")


def main():
    parser = ArgumentParser()
    parser.add_argument("--results-dir", default="./results_v2")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    all_results = {}
    
    for dataset in ['non_human', 'human', 'all']:
        agg = aggregate_dataset(results_dir, dataset)
        if agg:
            all_results[dataset] = agg
            print_comparison_table(agg)
    
    # Save aggregate
    output_file = results_dir / 'aggregate_v2.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAggregate saved to: {output_file}")


if __name__ == "__main__":
    main()
