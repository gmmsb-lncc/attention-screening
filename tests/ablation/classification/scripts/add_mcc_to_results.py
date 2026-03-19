"""
Quick fix: Add MCC to existing classification results
Recalculates MCC from confusion matrices already saved in results.
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import matthews_corrcoef
import numpy as np

BASE_DIR = Path('/media/leon/ssd2tb/docktkinase/ablation/classification')
RESULTS_DIR = BASE_DIR / 'results'

def calculate_mcc_from_confusion_matrix(cm):
    """Calculate MCC from confusion matrix."""
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    # MCC formula
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def add_mcc_to_results():
    """Add MCC to existing results."""
    json_file = RESULTS_DIR / 'classification_results.json'
    
    print("Loading existing results...")
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    print(f"Processing {len(results)} experiment results...")
    
    # Add MCC to each result
    for result in results:
        for classifier in ['knn', 'mlp']:
            for split in ['val', 'test']:
                cm = result[classifier][split]['confusion_matrix']
                mcc = calculate_mcc_from_confusion_matrix(cm)
                result[classifier][split]['mcc'] = float(mcc)
    
    # Save updated JSON
    print("Saving updated JSON...")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Regenerate CSV with MCC
    print("Regenerating summary CSV...")
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
    
    print(f"✓ Updated {json_file}")
    print(f"✓ Updated {csv_file}")
    print(f"✓ Added MCC to {len(results)} experiments")
    
    # Show sample MCC values
    print("\nSample MCC values:")
    print(summary_df[['combination', 'classifier', 'test_mcc']].head(10))


if __name__ == '__main__':
    add_mcc_to_results()
