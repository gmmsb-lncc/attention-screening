#!/usr/bin/env python3
"""
eval_pretrained_universal.py

Evaluate DeepDTAGen pretrained models on the universal kinase test sets.

DeepDTAGen is a regression model (predicts continuous affinity).
Our universal datasets have binary labels (0=non-binder, 1=binder).

IMPORTANT: Each pretrained model has its own tokenizer (vocab).
The universal data .pt files were built with a kinase-specific tokenizer.
If the kinase vocabulary exceeds the pretrained model's embedding table,
we skip that model (davis/kiba typically have smaller vocabs than the
diverse kinase SMILES require). BindingDB usually works because it has
the broadest vocabulary (107 tokens).

Usage:
    conda activate deepdtagen
    python eval_pretrained_universal.py

Output:
    results/deepdtagen_pretrained_universal.csv
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    accuracy_score, precision_score, recall_score, f1_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DeepDTAGen
from utils import TestbedDataset
from torch_geometric.loader import DataLoader


PRETRAINED_MODELS = {
    'bindingdb': 'models/deepdtagen_model_bindingdb.pth',
    'davis': 'models/deepdtagen_model_davis.pth',
    'kiba': 'models/deepdtagen_model_kiba.pth',
}

UNIVERSAL_DATASETS = ['non_human', 'human', 'all']
BATCH_SIZE = 64


def find_optimal_threshold(y_true, y_score, n_thresholds=200):
    """Find threshold that maximizes MCC."""
    thresholds = np.linspace(y_score.min(), y_score.max(), n_thresholds)
    best_mcc, best_thresh = -1, 0
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t
    return best_thresh, best_mcc


def check_vocab_compatibility(model, test_loader, device):
    """Test one batch to check for index-out-of-range errors."""
    model.eval()
    batch = next(iter(test_loader))
    try:
        with torch.no_grad():
            pred, _, _, _ = model(batch.to(device))
        return True
    except IndexError:
        return False
    except RuntimeError as e:
        if "index out of range" in str(e):
            return False
        raise


def evaluate_on_dataset(model, dataset_name, device):
    """Evaluate a pretrained model on one universal dataset's test set."""
    pt_path = f'data/processed/kinase_{dataset_name}_test.pt'
    if not os.path.exists(pt_path):
        print(f"  [SKIP] {pt_path} not found. Run convert_universal_data.py first.")
        return None

    test_data = TestbedDataset(root='data', dataset=f'kinase_{dataset_name}_test')
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Check vocab compatibility before full eval
    if not check_vocab_compatibility(model, test_loader, device):
        print(f"  [SKIP] {dataset_name}: vocab mismatch (SMILES tokens exceed embedding table)")
        return None

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f'  {dataset_name}', leave=False):
            pred, _, _, _ = model(data.to(device))
            all_preds.append(pred.cpu().numpy().flatten())
            all_labels.append(data.y.cpu().numpy().flatten())

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_preds)

    # Binary classification metrics
    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = 0.0
    try:
        auprc = average_precision_score(y_true, y_score)
    except ValueError:
        auprc = 0.0

    threshold, best_mcc = find_optimal_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pos_rate = y_true.mean()

    return {
        'n_samples': len(y_true),
        'pos_rate': round(pos_rate, 4),
        'auroc': round(auroc, 4),
        'auprc': round(auprc, 4),
        'mcc': round(best_mcc, 4),
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'threshold': round(threshold, 4),
        'pred_mean': round(y_score.mean(), 4),
        'pred_std': round(y_score.std(), 4),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs('results', exist_ok=True)
    results = []

    for model_name, model_path in PRETRAINED_MODELS.items():
        if not os.path.exists(model_path):
            print(f"[SKIP] Pretrained model not found: {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f" Pretrained model: {model_name}")
        print(f"{'='*60}")

        # Load model with its own tokenizer
        tok_path = f'data/{model_name}_tokenizer.pkl'
        if not os.path.exists(tok_path):
            print(f"  [SKIP] Tokenizer not found: {tok_path}")
            continue

        with open(tok_path, 'rb') as f:
            tokenizer = pickle.load(f)

        model = DeepDTAGen(tokenizer).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print(f"  Loaded: {sum(p.numel() for p in model.parameters()):,} params "
              f"(vocab={len(tokenizer)})")

        for ds_name in UNIVERSAL_DATASETS:
            result = evaluate_on_dataset(model, ds_name, device)
            if result:
                result['pretrained_model'] = model_name
                result['eval_dataset'] = ds_name
                results.append(result)
                print(f"  {ds_name}: AUROC={result['auroc']:.4f}, "
                      f"AUPRC={result['auprc']:.4f}, MCC={result['mcc']:.4f}")

    if results:
        df = pd.DataFrame(results)
        cols = ['pretrained_model', 'eval_dataset', 'n_samples', 'pos_rate',
                'auroc', 'auprc', 'mcc', 'accuracy', 'precision', 'recall',
                'f1', 'threshold', 'pred_mean', 'pred_std']
        df = df[[c for c in cols if c in df.columns]]
        out_path = 'results/deepdtagen_pretrained_universal.csv'
        df.to_csv(out_path, index=False)
        print(f"\nResults saved: {out_path}")
        print(df.to_string(index=False))
    else:
        print("\nNo results generated. Ensure data is converted first.")


if __name__ == '__main__':
    main()
