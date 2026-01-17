#!/usr/bin/env python3
"""
02_run_regression.py - Run regression experiments for ablation study.

This script runs regression experiments using ESM-2 embeddings (8M, 150M, 3B)
with KNN Regressor and MLP Regressor, using random 80/10/10 split.

Metrics:
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- Pearson Correlation
- Spearman Correlation

Author: DockTKinase Team
Date: January 2026
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Suppress warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"

# ESM-2 models to evaluate
ESM_MODELS = [
    'esm2_t6_8M_UR50D',      # 8M parameters, 320-dim
    'esm2_t30_150M_UR50D',   # 150M parameters, 640-dim
    'esm2_t36_3B_UR50D',     # 3B parameters, 2560-dim
]

ESM_DIMS = {
    'esm2_t6_8M_UR50D': 320,
    'esm2_t30_150M_UR50D': 640,
    'esm2_t36_3B_UR50D': 2560,
}

# Path to pre-computed embeddings
EMBEDDINGS_DIR = Path("/media/leon/ssd2tb/docktkinase/results/protein_model_benchmark_non_human_v2")

# Split configuration
RANDOM_SEED = 420
TEST_SIZE = 0.1   # 10% for test
VAL_SIZE = 0.1    # 10% for validation (from remaining 90%)

# Seeds for multiple runs
SEEDS = [42, 123, 456, 789, 1024]


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def create_knn_regressor(random_state: int = RANDOM_SEED) -> Pipeline:
    """Create KNN Regressor with StandardScaler."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsRegressor(
            n_neighbors=5,
            weights='distance',
            metric='cosine',
            n_jobs=-1
        ))
    ])


def create_mlp_regressor(random_state: int = RANDOM_SEED) -> Pipeline:
    """Create MLP Regressor with StandardScaler."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
            verbose=False
        ))
    ])


# =============================================================================
# METRICS
# =============================================================================

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all regression metrics including CCC (equivalent to MCC for regression)."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Correlation metrics
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    
    # Concordance Correlation Coefficient (CCC) - Lin's Concordance
    # This is the regression equivalent of MCC
    # CCC = 2 * covariance / (var(y_true) + var(y_pred) + (mean(y_true) - mean(y_pred))^2)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2)
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'ccc': float(ccc),  # Concordance Correlation Coefficient (MCC equivalent for regression)
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_embeddings(model_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load embeddings and interaction labels for a model."""
    model_dir = EMBEDDINGS_DIR / model_name / "build"
    
    if not model_dir.exists():
        model_dir = EMBEDDINGS_DIR / model_name
    
    embeddings_path = model_dir / "embedding_matrix.npy"
    labels_path = model_dir / "interaction_labels.npy"
    
    if not embeddings_path.exists():
        print(f"   ⚠️ Embeddings not found: {embeddings_path}")
        return None, None
    
    embeddings = np.load(embeddings_path)
    
    # Load interaction labels (pchembl values)
    if labels_path.exists():
        # interaction_labels.npy contains: [molregno, target_name, assay_type, standard_value, pchembl_value]
        # We need the last column (pchembl_value)
        labels_data = np.load(labels_path, allow_pickle=True)
        labels = labels_data[:, -1].astype(np.float64)  # pchembl_value is the last column
    else:
        print(f"   ⚠️ Labels not found: {labels_path}")
        return None, None
    
    return embeddings, labels


def load_interactions() -> pd.DataFrame:
    """Load interactions dataframe with pchembl values."""
    interactions_path = PROCESSED_DIR / 'interactions_regression.csv'
    
    if not interactions_path.exists():
        # Fallback to classification interactions
        interactions_path = BASE_DIR.parent / "classification" / "data" / "processed" / "interactions.csv"
    
    return pd.read_csv(interactions_path)


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_single_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int
) -> Dict[str, Any]:
    """Run a single experiment with both KNN and MLP regressors."""
    
    results = {}
    
    # KNN Regressor
    print(f"      🔧 Training KNN Regressor...")
    knn_start = time.time()
    knn_model = create_knn_regressor(seed)
    knn_model.fit(X_train, y_train)
    knn_time = time.time() - knn_start
    
    knn_val_pred = knn_model.predict(X_val)
    knn_test_pred = knn_model.predict(X_test)
    
    results['knn'] = {
        'train_time': knn_time,
        'val': calculate_regression_metrics(y_val, knn_val_pred),
        'test': calculate_regression_metrics(y_test, knn_test_pred),
        'val_predictions': knn_val_pred.tolist(),
        'test_predictions': knn_test_pred.tolist(),
    }
    print(f"         ✅ KNN done in {knn_time:.2f}s | Val R²: {results['knn']['val']['r2']:.4f} | Test R²: {results['knn']['test']['r2']:.4f}")
    sys.stdout.flush()
    
    # MLP Regressor
    print(f"      🔧 Training MLP Regressor...")
    sys.stdout.flush()
    mlp_start = time.time()
    mlp_model = create_mlp_regressor(seed)
    mlp_model.fit(X_train, y_train)
    mlp_time = time.time() - mlp_start
    
    mlp_val_pred = mlp_model.predict(X_val)
    mlp_test_pred = mlp_model.predict(X_test)
    
    # Get number of iterations
    n_iter = mlp_model.named_steps['model'].n_iter_
    
    results['mlp'] = {
        'train_time': mlp_time,
        'n_iterations': n_iter,
        'val': calculate_regression_metrics(y_val, mlp_val_pred),
        'test': calculate_regression_metrics(y_test, mlp_test_pred),
        'val_predictions': mlp_val_pred.tolist(),
        'test_predictions': mlp_test_pred.tolist(),
    }
    print(f"         ✅ MLP done in {mlp_time:.2f}s ({n_iter} epochs) | Val R²: {results['mlp']['val']['r2']:.4f} | Test R²: {results['mlp']['test']['r2']:.4f}")
    sys.stdout.flush()
    
    return results


def save_intermediate_results(all_results: Dict, summary_rows: List[Dict], checkpoint_name: str = 'checkpoint'):
    """Save intermediate results to avoid data loss."""
    # Save detailed results
    checkpoint_path = RESULTS_DIR / f'regression_results_{checkpoint_name}.json'
    with open(checkpoint_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        checkpoint_csv = RESULTS_DIR / f'regression_summary_{checkpoint_name}.csv'
        summary_df.to_csv(checkpoint_csv, index=False)
    
    sys.stdout.flush()  # Force flush output buffer


def run_all_experiments() -> Dict[str, Any]:
    """Run all regression experiments."""
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'experiment': 'regression_ablation',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'split': 'random',
            'split_ratio': '80/10/10',
            'seeds': SEEDS,
            'models': ESM_MODELS,
        },
        'results': {}
    }
    
    print("=" * 70)
    print("📊 REGRESSION ABLATION STUDY")
    print("=" * 70)
    print()
    print("📋 Configuration:")
    print(f"   Split: 80% train / 10% val / 10% test (RANDOM)")
    print(f"   ESM-2 Models: {', '.join(ESM_MODELS)}")
    print(f"   Regressors: KNN, MLP")
    print(f"   Seeds: {SEEDS}")
    print()
    sys.stdout.flush()
    
    summary_rows = []
    
    for model_name in ESM_MODELS:
        model_dim = ESM_DIMS.get(model_name, 0)
        
        print("=" * 70)
        print(f"🧬 Model: {model_name} ({model_dim}-dim)")
        print("=" * 70)
        
        # Load embeddings
        embeddings, labels = load_embeddings(model_name)
        
        if embeddings is None or labels is None:
            print(f"   ⚠️ Skipping {model_name} - data not found")
            continue
        
        print(f"   ✅ Loaded: {len(embeddings):,} samples, {embeddings.shape[1]} dimensions")
        print(f"   📊 Target stats: min={labels.min():.2f}, max={labels.max():.2f}, mean={labels.mean():.2f}")
        sys.stdout.flush()
        
        all_results['results'][model_name] = {
            'n_samples': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'target_stats': {
                'min': float(labels.min()),
                'max': float(labels.max()),
                'mean': float(labels.mean()),
                'std': float(labels.std()),
            },
            'seeds': {}
        }
        
        for seed in SEEDS:
            print(f"\n   🎲 Seed: {seed}")
            
            # Random split: 80/10/10
            # First split: 90% train+val, 10% test
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                embeddings, labels,
                test_size=TEST_SIZE,
                random_state=seed
            )
            
            # Second split: ~88.9% train, ~11.1% val (from 90%)
            # This gives us 80/10/10 overall
            val_ratio = VAL_SIZE / (1 - TEST_SIZE)  # 0.1 / 0.9 ≈ 0.111
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval,
                test_size=val_ratio,
                random_state=seed
            )
            
            print(f"      Split: Train={len(X_train):,} ({100*len(X_train)/len(embeddings):.1f}%) | "
                  f"Val={len(X_val):,} ({100*len(X_val)/len(embeddings):.1f}%) | "
                  f"Test={len(X_test):,} ({100*len(X_test)/len(embeddings):.1f}%)")
            
            # Run experiment
            seed_results = run_single_experiment(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                seed
            )
            
            # Store results (without predictions to save space)
            seed_results_clean = {
                'knn': {k: v for k, v in seed_results['knn'].items() if 'predictions' not in k},
                'mlp': {k: v for k, v in seed_results['mlp'].items() if 'predictions' not in k},
            }
            all_results['results'][model_name]['seeds'][str(seed)] = seed_results_clean
            
            # Add to summary
            for regressor in ['knn', 'mlp']:
                row = {
                    'model': model_name,
                    'seed': seed,
                    'regressor': regressor.upper(),
                    'n_samples': len(embeddings),
                    'n_features': embeddings.shape[1],
                }
                for metric in ['mse', 'rmse', 'mae', 'r2', 'pearson_r', 'spearman_r', 'ccc']:
                    row[f'val_{metric}'] = seed_results[regressor]['val'][metric]
                    row[f'test_{metric}'] = seed_results[regressor]['test'][metric]
                if regressor == 'mlp':
                    row['n_iterations'] = seed_results['mlp']['n_iterations']
                summary_rows.append(row)
            
            # 💾 Save intermediate checkpoint after each seed
            checkpoint_name = f"{model_name}_seed{seed}"
            save_intermediate_results(all_results, summary_rows, checkpoint_name)
            print(f"      💾 Checkpoint saved: {checkpoint_name}")
            sys.stdout.flush()
    
    # Save detailed results
    results_path = RESULTS_DIR / 'regression_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Detailed results saved to: {results_path}")
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / 'regression_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Summary CSV saved to: {summary_path}")
    
    # Print aggregate statistics
    print_aggregate_stats(summary_df)
    
    return all_results


def print_aggregate_stats(df: pd.DataFrame):
    """Print aggregate statistics across seeds."""
    
    print()
    print("=" * 70)
    print("📊 AGGREGATE RESULTS (Mean ± Std across 5 seeds)")
    print("=" * 70)
    
    for model in df['model'].unique():
        print(f"\n🧬 {model}")
        print("-" * 60)
        
        for regressor in ['KNN', 'MLP']:
            subset = df[(df['model'] == model) & (df['regressor'] == regressor)]
            
            print(f"\n   {regressor}:")
            for metric in ['r2', 'rmse', 'mae', 'pearson_r', 'spearman_r', 'ccc']:
                mean_val = subset[f'test_{metric}'].mean()
                std_val = subset[f'test_{metric}'].std()
                print(f"      {metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Best model summary
    print()
    print("=" * 70)
    print("🏆 BEST MODELS (by Test R²)")
    print("=" * 70)
    
    best_results = []
    for model in df['model'].unique():
        for regressor in ['KNN', 'MLP']:
            subset = df[(df['model'] == model) & (df['regressor'] == regressor)]
            mean_r2 = subset['test_r2'].mean()
            std_r2 = subset['test_r2'].std()
            best_results.append({
                'model': model,
                'regressor': regressor,
                'r2_mean': mean_r2,
                'r2_std': std_r2,
            })
    
    best_df = pd.DataFrame(best_results).sort_values('r2_mean', ascending=False)
    print()
    print(f"{'Model':<25} {'Regressor':<10} {'Test R² (mean ± std)':<20}")
    print("-" * 60)
    for _, row in best_df.iterrows():
        print(f"{row['model']:<25} {row['regressor']:<10} {row['r2_mean']:.4f} ± {row['r2_std']:.4f}")


if __name__ == '__main__':
    run_all_experiments()
