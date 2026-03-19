#!/usr/bin/env python3
"""
Consolidate checkpoint CSVs into a single regression_summary.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Consolidate checkpoint files')
    parser.add_argument('--tsv-path', type=str, help='TSV path (unused)')
    parser.add_argument('--results-suffix', type=str, default='results_non_human',
                       help='Results directory suffix')
    parser.add_argument('--embeddings-dir', type=str, help='Embeddings directory (unused)')
    return parser.parse_args()


args = parse_args()
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / args.results_suffix

print(f"Dataset: {args.results_suffix}")
print(f"Results: {RESULTS_DIR}\n")

def consolidate_checkpoints():
    """Merge all checkpoint CSVs into regression_summary.csv"""
    
    checkpoint_files = sorted(RESULTS_DIR.glob("regression_summary_*.csv"))
    
    if not checkpoint_files:
        print("❌ No checkpoint files found!")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files:")
    for f in checkpoint_files:
        print(f"  - {f.name}")
    
    # Load and concatenate
    dfs = []
    for csv_file in checkpoint_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by model, seed, regressor
    combined_df = combined_df.sort_values(['model', 'seed', 'regressor'])
    
    # Save consolidated file
    output_file = RESULTS_DIR / "regression_summary.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Consolidated {len(combined_df)} results into: {output_file}")
    print(f"\nModels: {combined_df['model'].unique().tolist()}")
    print(f"Seeds: {sorted(combined_df['seed'].unique().tolist())}")
    print(f"Regressors: {combined_df['regressor'].unique().tolist()}")
    
    return combined_df

if __name__ == '__main__':
    consolidate_checkpoints()
