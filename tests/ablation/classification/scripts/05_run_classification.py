"""
Script 05: Classification Experiments
Trains KNN and MLP classifiers on all 10 representation combinations.
Runs 5 random seeds per combination and saves metrics for comparison.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run classification experiments')
    parser.add_argument('--tsv-path', type=str, help='TSV path (unused)')
    parser.add_argument('--results-suffix', type=str, default='results_non_human',
                       help='Results directory suffix')
    parser.add_argument('--embeddings-dir', type=str, help='Embeddings directory (unused)')
    return parser.parse_args()


# Configuration
args = parse_args()
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / args.results_suffix / 'combinations'
RESULTS_DIR = BASE_DIR / args.results_suffix
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset: {args.results_suffix}")
print(f"Data: {DATA_DIR}")
print(f"Results: {RESULTS_DIR}\n")

RANDOM_SEEDS = [42, 123, 456, 789, 1024]
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # 10% of training data for validation

# Classifier configurations
KNN_CONFIG = {
    'n_neighbors': 5,
    'metric': 'cosine',
    'weights': 'distance',
    'n_jobs': -1
}

MLP_CONFIG = {
    'hidden_layer_sizes': (512,),
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 500,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 20,
    'random_state': None  # Will be set per seed
}


def load_combination(combination_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load features and labels for a specific combination."""
    features_file = DATA_DIR / f'{combination_name}_features.npy'
    labels_file = DATA_DIR / f'{combination_name}_labels.npy'
    
    features = np.load(features_file)
    labels = np.load(labels_file)
    
    return features, labels


def split_data(features: np.ndarray, labels: np.ndarray, seed: int) -> Dict:
    """Split data into train, validation, and test sets."""
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        features, labels, 
        test_size=TEST_SIZE, 
        random_state=seed,
        stratify=labels
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_SIZE,
        random_state=seed,
        stratify=y_trainval
    )
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


def normalize_features(splits: Dict) -> Tuple[Dict, StandardScaler]:
    """Normalize features using StandardScaler fitted on training data."""
    scaler = StandardScaler()
    scaler.fit(splits['X_train'])
    
    normalized = {
        'X_train': scaler.transform(splits['X_train']),
        'X_val': scaler.transform(splits['X_val']),
        'X_test': scaler.transform(splits['X_test']),
        'y_train': splits['y_train'],
        'y_val': splits['y_val'],
        'y_test': splits['y_test']
    }
    
    return normalized, scaler


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
    """Calculate classification metrics."""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_proba)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }


def train_knn(data: Dict) -> Tuple[KNeighborsClassifier, Dict]:
    """Train KNN classifier and evaluate."""
    knn = KNeighborsClassifier(**KNN_CONFIG)
    knn.fit(data['X_train'], data['y_train'])
    
    # Predictions
    y_pred_val = knn.predict(data['X_val'])
    y_proba_val = knn.predict_proba(data['X_val'])[:, 1]
    
    y_pred_test = knn.predict(data['X_test'])
    y_proba_test = knn.predict_proba(data['X_test'])[:, 1]
    
    # Metrics
    metrics = {
        'val': calculate_metrics(data['y_val'], y_pred_val, y_proba_val),
        'test': calculate_metrics(data['y_test'], y_pred_test, y_proba_test)
    }
    
    return knn, metrics


def train_mlp(data: Dict, seed: int) -> Tuple[MLPClassifier, Dict]:
    """Train MLP classifier and evaluate."""
    mlp_config = MLP_CONFIG.copy()
    mlp_config['random_state'] = seed
    
    mlp = MLPClassifier(**mlp_config)
    mlp.fit(data['X_train'], data['y_train'])
    
    # Predictions
    y_pred_val = mlp.predict(data['X_val'])
    y_proba_val = mlp.predict_proba(data['X_val'])[:, 1]
    
    y_pred_test = mlp.predict(data['X_test'])
    y_proba_test = mlp.predict_proba(data['X_test'])[:, 1]
    
    # Metrics
    metrics = {
        'val': calculate_metrics(data['y_val'], y_pred_val, y_proba_val),
        'test': calculate_metrics(data['y_test'], y_pred_test, y_proba_test),
        'n_iterations': int(mlp.n_iter_)
    }
    
    return mlp, metrics


def run_single_experiment(combination_name: str, seed: int) -> Dict:
    """Run complete experiment for one combination and one seed."""
    print(f"  Seed {seed}...", end=' ', flush=True)
    
    # Load data
    features, labels = load_combination(combination_name)
    
    # Split data
    splits = split_data(features, labels, seed)
    
    # Normalize
    normalized, scaler = normalize_features(splits)
    
    # Train KNN
    knn, knn_metrics = train_knn(normalized)
    
    # Train MLP
    mlp, mlp_metrics = train_mlp(normalized, seed)
    
    print(f"KNN Test AUC: {knn_metrics['test']['auc']:.4f} | MLP Test AUC: {mlp_metrics['test']['auc']:.4f}")
    
    return {
        'seed': seed,
        'combination': combination_name,
        'n_samples': len(features),
        'n_features': features.shape[1],
        'n_train': len(splits['X_train']),
        'n_val': len(splits['X_val']),
        'n_test': len(splits['X_test']),
        'knn': knn_metrics,
        'mlp': mlp_metrics
    }


def run_all_experiments() -> List[Dict]:
    """Run experiments for all combinations and seeds."""
    # Get all combination files
    combination_files = sorted(DATA_DIR.glob('*_features.npy'))
    all_combinations = [f.stem.replace('_features', '') for f in combination_files]
    
    # Order by complexity: C4 (simplest) → C3 → C2 → C1 (most complex)
    # C4: AAC+DPC + Morgan (2468D) - simplest
    # C3: AAC+DPC + SMI-TED (1188D)
    # C2: ESM + Morgan (2368D, 2688D, 4608D)
    # C1: ESM + SMI-TED (1088D, 1408D, 3328D) - most complex
    
    ordered_combinations = []
    for prefix in ['C4_', 'C3_', 'C2_', 'C1_']:
        ordered_combinations.extend(sorted([c for c in all_combinations if c.startswith(prefix)]))
    
    print(f"Found {len(all_combinations)} total combinations")
    print(f"Order: C4 (simplest) → C3 → C2 → C1 (most complex)")
    print(f"Running {len(RANDOM_SEEDS)} seeds per combination")
    print(f"Total experiments: {len(ordered_combinations) * len(RANDOM_SEEDS) * 2} (KNN + MLP)\n")
    
    all_results = []
    
    for i, combination in enumerate(ordered_combinations, 1):
        print(f"[{i}/{len(ordered_combinations)}] {combination}")
        
        for seed in RANDOM_SEEDS:
            result = run_single_experiment(combination, seed)
            all_results.append(result)
        
        print()
    
    return all_results


def save_results(results: List[Dict]):
    """Save results to JSON and create summary CSV."""
    # Save complete results as JSON
    json_file = RESULTS_DIR / 'classification_results.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved detailed results to {json_file}")
    
    # Create summary DataFrame
    summary_rows = []
    for result in results:
        base_row = {
            'combination': result['combination'],
            'seed': result['seed'],
            'n_samples': result['n_samples'],
            'n_features': result['n_features']
        }
        
        # KNN metrics
        knn_row = base_row.copy()
        knn_row['classifier'] = 'KNN'
        knn_row.update({
            'val_accuracy': result['knn']['val']['accuracy'],
            'val_precision': result['knn']['val']['precision'],
            'val_recall': result['knn']['val']['recall'],
            'val_f1': result['knn']['val']['f1'],
            'val_auc': result['knn']['val']['auc'],
            'val_mcc': result['knn']['val']['mcc'],
            'test_accuracy': result['knn']['test']['accuracy'],
            'test_precision': result['knn']['test']['precision'],
            'test_recall': result['knn']['test']['recall'],
            'test_f1': result['knn']['test']['f1'],
            'test_auc': result['knn']['test']['auc'],
            'test_mcc': result['knn']['test']['mcc']
        })
        summary_rows.append(knn_row)
        
        # MLP metrics
        mlp_row = base_row.copy()
        mlp_row['classifier'] = 'MLP'
        mlp_row.update({
            'val_accuracy': result['mlp']['val']['accuracy'],
            'val_precision': result['mlp']['val']['precision'],
            'val_recall': result['mlp']['val']['recall'],
            'val_f1': result['mlp']['val']['f1'],
            'val_auc': result['mlp']['val']['auc'],
            'val_mcc': result['mlp']['val']['mcc'],
            'test_accuracy': result['mlp']['test']['accuracy'],
            'test_precision': result['mlp']['test']['precision'],
            'test_recall': result['mlp']['test']['recall'],
            'test_f1': result['mlp']['test']['f1'],
            'test_auc': result['mlp']['test']['auc'],
            'test_mcc': result['mlp']['test']['mcc'],
            'n_iterations': result['mlp'].get('n_iterations', None)
        })
        summary_rows.append(mlp_row)
    
    summary_df = pd.DataFrame(summary_rows)
    csv_file = RESULTS_DIR / 'classification_summary.csv'
    summary_df.to_csv(csv_file, index=False)
    print(f"✓ Saved summary to {csv_file}")
    
    return summary_df


def print_aggregate_stats(df: pd.DataFrame):
    """Print aggregate statistics across seeds."""
    print("\n" + "="*80)
    print("AGGREGATE RESULTS (Mean ± Std across 5 seeds)")
    print("="*80)
    
    # Group by combination and classifier
    grouped = df.groupby(['combination', 'classifier'])
    
    for (combination, classifier), group in grouped:
        print(f"\n{combination} | {classifier}")
        print("-" * 60)
        
        # Test metrics
        for metric in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc', 'test_mcc']:
            mean = group[metric].mean()
            std = group[metric].std()
            metric_name = metric.replace('test_', '').upper()
            print(f"  {metric_name:10s}: {mean:.4f} ± {std:.4f}")


if __name__ == '__main__':
    print("="*80)
    print("ABLATION STUDY: Classification Experiments")
    print("="*80)
    print(f"Base directory: {BASE_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Random seeds: {RANDOM_SEEDS}")
    print(f"Test size: {TEST_SIZE * 100:.0f}%")
    print(f"Validation size: {VAL_SIZE * 100:.0f}% of training data")
    print("="*80 + "\n")
    
    # Run all experiments
    results = run_all_experiments()
    
    # Save results
    summary_df = save_results(results)
    
    # Print aggregate statistics
    print_aggregate_stats(summary_df)
    
    print("\n" + "="*80)
    print("CLASSIFICATION EXPERIMENTS COMPLETED")
    print("="*80)
