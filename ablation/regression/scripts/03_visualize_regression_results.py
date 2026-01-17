#!/usr/bin/env python3
"""
03_visualize_regression_results.py - Visualize regression ablation study results.

Creates comprehensive visualizations for regression experiments:
1. Comparison plot (2x3 grid): R², RMSE, MAE, Pearson, Spearman, MSE
2. Scatter plots: Predicted vs Actual values
3. Error distribution plots
4. Model comparison heatmap

Author: DockTKinase Team
Date: January 2026
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'KNN': '#3498db',      # Blue
    'MLP': '#e74c3c',      # Red
    '8M': '#2ecc71',       # Green
    '150M': '#9b59b6',     # Purple
    '3B': '#f39c12',       # Orange
}

MODEL_LABELS = {
    'esm2_t6_8M_UR50D': 'ESM-2 8M',
    'esm2_t30_150M_UR50D': 'ESM-2 150M',
    'esm2_t36_3B_UR50D': 'ESM-2 3B',
}


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def load_results() -> pd.DataFrame:
    """Load regression results from CSV."""
    csv_path = RESULTS_DIR / 'regression_summary.csv'
    return pd.read_csv(csv_path)


def plot_metrics_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Create 2x3 grid comparing all metrics across models and regressors.
    
    Layout:
    - Row 1: R², RMSE, MAE
    - Row 2: Pearson r, Spearman r, CCC (MCC equivalent)
    """
    
    metrics = ['test_r2', 'test_rmse', 'test_mae', 'test_pearson_r', 'test_spearman_r', 'test_ccc']
    metric_labels = ['R²', 'RMSE', 'MAE', 'Pearson r', 'Spearman r', 'CCC']
    
    models = ['esm2_t6_8M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t36_3B_UR50D']
    model_labels_list = ['ESM-2 8M', 'ESM-2 150M', 'ESM-2 3B']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Regression Ablation Study: ESM-2 Models Comparison\n(Non-Human Kinases, Random 80/10/10 Split, 5 Seeds)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        x = np.arange(len(models))
        width = 0.35
        
        # Calculate mean and std for each model and regressor
        knn_means = []
        knn_stds = []
        mlp_means = []
        mlp_stds = []
        
        for model in models:
            knn_data = df[(df['model'] == model) & (df['regressor'] == 'KNN')][metric]
            mlp_data = df[(df['model'] == model) & (df['regressor'] == 'MLP')][metric]
            
            knn_means.append(knn_data.mean())
            knn_stds.append(knn_data.std())
            mlp_means.append(mlp_data.mean())
            mlp_stds.append(mlp_data.std())
        
        # Plot bars with error bars
        bars1 = ax.bar(x - width/2, knn_means, width, label='KNN', 
                       color=COLORS['KNN'], alpha=0.8, yerr=knn_stds, capsize=3)
        bars2 = ax.bar(x + width/2, mlp_means, width, label='MLP', 
                       color=COLORS['MLP'], alpha=0.8, yerr=mlp_stds, capsize=3)
        
        # Add value labels on bars
        for bars, means in [(bars1, knn_means), (bars2, mlp_means)]:
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.annotate(f'{mean:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_ylabel(label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels_list)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Adjust y-axis: 0-1 for normalized metrics, auto for error metrics
        if metric in ['test_r2', 'test_pearson_r', 'test_spearman_r', 'test_ccc']:
            ax.set_ylim(0.0, 1.0)
        else:
            # For RMSE and MAE, let it auto-scale but start from 0
            current_ylim = ax.get_ylim()
            ax.set_ylim(0.0, current_ylim[1])
    
    # Single legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98),
               fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_path = output_dir / "regression_metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_r2_focus(df: pd.DataFrame, output_dir: Path):
    """Create focused R² comparison plot."""
    
    models = ['esm2_t6_8M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t36_3B_UR50D']
    model_labels_list = ['ESM-2 8M', 'ESM-2 150M', 'ESM-2 3B']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    knn_means = []
    knn_stds = []
    mlp_means = []
    mlp_stds = []
    
    for model in models:
        knn_data = df[(df['model'] == model) & (df['regressor'] == 'KNN')]['test_r2']
        mlp_data = df[(df['model'] == model) & (df['regressor'] == 'MLP')]['test_r2']
        
        knn_means.append(knn_data.mean())
        knn_stds.append(knn_data.std())
        mlp_means.append(mlp_data.mean())
        mlp_stds.append(mlp_data.std())
    
    bars1 = ax.bar(x - width/2, knn_means, width, label='KNN Regressor', 
                   color=COLORS['KNN'], alpha=0.8, yerr=knn_stds, capsize=5)
    bars2 = ax.bar(x + width/2, mlp_means, width, label='MLP Regressor', 
                   color=COLORS['MLP'], alpha=0.8, yerr=mlp_stds, capsize=5)
    
    # Add value labels
    for bars, means, stds in [(bars1, knn_means, knn_stds), (bars2, mlp_means, mlp_stds)]:
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.annotate(f'{mean:.4f}\n±{std:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('ESM-2 Model', fontsize=12, fontweight='bold')
    ax.set_title('pChEMBL Value Prediction: R² Score by Model and Regressor\n(Random 80/10/10 Split, Mean ± Std over 5 Seeds)', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels_list, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_path = output_dir / "regression_r2_focus.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_correlation_comparison(df: pd.DataFrame, output_dir: Path):
    """Create correlation metrics comparison (Pearson vs Spearman)."""
    
    models = ['esm2_t6_8M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t36_3B_UR50D']
    model_labels_list = ['ESM-2 8M', 'ESM-2 150M', 'ESM-2 3B']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (metric, title) in enumerate([('test_pearson_r', 'Pearson Correlation'), 
                                            ('test_spearman_r', 'Spearman Correlation')]):
        ax = axes[idx]
        x = np.arange(len(models))
        width = 0.35
        
        knn_means = []
        knn_stds = []
        mlp_means = []
        mlp_stds = []
        
        for model in models:
            knn_data = df[(df['model'] == model) & (df['regressor'] == 'KNN')][metric]
            mlp_data = df[(df['model'] == model) & (df['regressor'] == 'MLP')][metric]
            
            knn_means.append(knn_data.mean())
            knn_stds.append(knn_data.std())
            mlp_means.append(mlp_data.mean())
            mlp_stds.append(mlp_data.std())
        
        bars1 = ax.bar(x - width/2, knn_means, width, label='KNN', 
                       color=COLORS['KNN'], alpha=0.8, yerr=knn_stds, capsize=4)
        bars2 = ax.bar(x + width/2, mlp_means, width, label='MLP', 
                       color=COLORS['MLP'], alpha=0.8, yerr=mlp_stds, capsize=4)
        
        # Add value labels
        for bars, means in [(bars1, knn_means), (bars2, mlp_means)]:
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.annotate(f'{mean:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels_list)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle('Correlation Metrics: Predicted vs Actual pChEMBL Values', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    output_path = output_dir / "regression_correlation_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_error_metrics(df: pd.DataFrame, output_dir: Path):
    """Create error metrics comparison (RMSE vs MAE)."""
    
    models = ['esm2_t6_8M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t36_3B_UR50D']
    model_labels_list = ['ESM-2 8M', 'ESM-2 150M', 'ESM-2 3B']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (metric, title, ylabel) in enumerate([
        ('test_rmse', 'Root Mean Squared Error (RMSE)', 'RMSE (pChEMBL units)'),
        ('test_mae', 'Mean Absolute Error (MAE)', 'MAE (pChEMBL units)')
    ]):
        ax = axes[idx]
        x = np.arange(len(models))
        width = 0.35
        
        knn_means = []
        knn_stds = []
        mlp_means = []
        mlp_stds = []
        
        for model in models:
            knn_data = df[(df['model'] == model) & (df['regressor'] == 'KNN')][metric]
            mlp_data = df[(df['model'] == model) & (df['regressor'] == 'MLP')][metric]
            
            knn_means.append(knn_data.mean())
            knn_stds.append(knn_data.std())
            mlp_means.append(mlp_data.mean())
            mlp_stds.append(mlp_data.std())
        
        bars1 = ax.bar(x - width/2, knn_means, width, label='KNN', 
                       color=COLORS['KNN'], alpha=0.8, yerr=knn_stds, capsize=4)
        bars2 = ax.bar(x + width/2, mlp_means, width, label='MLP', 
                       color=COLORS['MLP'], alpha=0.8, yerr=mlp_stds, capsize=4)
        
        # Add value labels
        for bars, means in [(bars1, knn_means), (bars2, mlp_means)]:
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.annotate(f'{mean:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels_list)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle('Error Metrics: pChEMBL Value Prediction Errors', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    output_path = output_dir / "regression_error_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_heatmap_summary(df: pd.DataFrame, output_dir: Path):
    """Create heatmap summarizing all results."""
    
    models = ['esm2_t6_8M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t36_3B_UR50D']
    model_labels_list = ['ESM-2 8M', 'ESM-2 150M', 'ESM-2 3B']
    metrics = ['test_r2', 'test_pearson_r', 'test_spearman_r', 'test_ccc']
    metric_labels = ['R²', 'Pearson r', 'Spearman r', 'CCC']
    
    # Create data for heatmap
    data = []
    row_labels = []
    
    for model, model_label in zip(models, model_labels_list):
        for regressor in ['KNN', 'MLP']:
            row_data = []
            for metric in metrics:
                mean_val = df[(df['model'] == model) & (df['regressor'] == regressor)][metric].mean()
                row_data.append(mean_val)
            data.append(row_data)
            row_labels.append(f"{model_label} + {regressor}")
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Score', rotation=-90, va="bottom")
    
    # Set ticks
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=10)
    
    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(metric_labels)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha='center', va='center', color='black',
                          fontsize=10, fontweight='bold')
    
    ax.set_title('Regression Performance Summary\n(Higher is Better)', 
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / "regression_heatmap_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """Create summary table as CSV and formatted text."""
    
    models = ['esm2_t6_8M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t36_3B_UR50D']
    
    summary_data = []
    
    for model in models:
        for regressor in ['KNN', 'MLP']:
            subset = df[(df['model'] == model) & (df['regressor'] == regressor)]
            
            row = {
                'Model': MODEL_LABELS[model],
                'Regressor': regressor,
                'R² (mean)': subset['test_r2'].mean(),
                'R² (std)': subset['test_r2'].std(),
                'RMSE (mean)': subset['test_rmse'].mean(),
                'RMSE (std)': subset['test_rmse'].std(),
                'MAE (mean)': subset['test_mae'].mean(),
                'MAE (std)': subset['test_mae'].std(),
                'Pearson r (mean)': subset['test_pearson_r'].mean(),
                'Spearman r (mean)': subset['test_spearman_r'].mean(),
                'CCC (mean)': subset['test_ccc'].mean(),
            }
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    csv_path = output_dir / "regression_summary_table.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"✅ Saved: {csv_path}")
    
    # Print formatted table
    print()
    print("=" * 100)
    print("📊 REGRESSION RESULTS SUMMARY TABLE")
    print("=" * 100)
    print()
    print(f"{'Model':<15} {'Regressor':<10} {'R²':<18} {'RMSE':<18} {'MAE':<18} {'Pearson r':<12}")
    print("-" * 100)
    
    for _, row in summary_df.iterrows():
        r2_str = f"{row['R² (mean)']:.4f} ± {row['R² (std)']:.4f}"
        rmse_str = f"{row['RMSE (mean)']:.4f} ± {row['RMSE (std)']:.4f}"
        mae_str = f"{row['MAE (mean)']:.4f} ± {row['MAE (std)']:.4f}"
        
        print(f"{row['Model']:<15} {row['Regressor']:<10} {r2_str:<18} {rmse_str:<18} {mae_str:<18} {row['Pearson r (mean)']:.4f}")
    
    print("-" * 100)
    
    return summary_df


def main():
    """Generate all visualizations."""
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("📊 GENERATING REGRESSION VISUALIZATIONS")
    print("=" * 70)
    print()
    
    # Load results
    df = load_results()
    print(f"✅ Loaded {len(df)} experiment results")
    print()
    
    # Generate all plots
    print("📈 Generating plots...")
    plot_metrics_comparison(df, FIGURES_DIR)
    plot_r2_focus(df, FIGURES_DIR)
    plot_correlation_comparison(df, FIGURES_DIR)
    plot_error_metrics(df, FIGURES_DIR)
    plot_heatmap_summary(df, FIGURES_DIR)
    
    # Create summary table
    create_summary_table(df, FIGURES_DIR)
    
    print()
    print("=" * 70)
    print(f"✅ All visualizations saved to: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
