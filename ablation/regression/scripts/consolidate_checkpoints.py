#!/usr/bin/env python3
"""
Consolidate checkpoint CSVs into a single regression_summary.csv
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

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
