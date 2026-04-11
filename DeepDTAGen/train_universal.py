#!/usr/bin/env python3
"""
train_universal.py

Train DeepDTAGen from scratch on the universal kinase scaffold-split datasets.

Usage:
    conda activate deepdtagen
    python train_universal.py --dataset non_human
    python train_universal.py --dataset human
    python train_universal.py --dataset all

Outputs:
    saved_models/deepdtagen_kinase_{dataset}.pth
    results/deepdtagen_trained_universal.csv
"""
import os
import sys
import argparse
import pickle
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    accuracy_score, precision_score, recall_score, f1_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DeepDTAGen
from FetterGrad import FetterGrad
from utils import TestbedDataset, logging, get_cindex, mse
from torch_geometric.loader import DataLoader

# Reproducibility
SEED = 4221
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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


def train_epoch(model, device, train_loader, optimizer, mse_f, epoch):
    """Train one epoch."""
    model.train()
    epoch_mse = 0
    n_batches = 0

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}") as t:
        for data in t:
            optimizer.zero_grad()
            pred, new_drug, lm_loss, kl_loss = model(data.to(device))

            mse_loss = mse_f(pred, data.y.view(-1, 1).float().to(device))
            loss = kl_loss * 0.001 + mse_loss + lm_loss

            losses = [loss, mse_loss]
            optimizer.ft_backward(losses)
            optimizer.step()

            epoch_mse += mse_loss.item()
            n_batches += 1
            t.set_postfix(MSE=mse_loss.item(), LM=lm_loss.item(), KL=kl_loss.item())

    return epoch_mse / max(n_batches, 1)


def evaluate(model, device, test_loader):
    """Evaluate model and return binary classification metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="  Evaluating", leave=False):
            pred, _, _, _ = model(data.to(device))
            all_preds.append(pred.cpu().numpy().flatten())
            all_labels.append(data.y.cpu().numpy().flatten())

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_preds)

    # Regression metrics
    mse_val = float(((y_true - y_score) ** 2).mean())

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

    return {
        'mse': round(mse_val, 6),
        'auroc': round(auroc, 4),
        'auprc': round(auprc, 4),
        'mcc': round(best_mcc, 4),
        'accuracy': round(acc, 4),
        'threshold': round(threshold, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['non_human', 'human', 'all'],
                        help='Universal dataset to train on')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    dataset = args.dataset
    ds_prefix = f'kinase_{dataset}'

    # Paths
    tokenizer_path = f'data/{ds_prefix}_tokenizer.pkl'
    train_pt = f'data/processed/{ds_prefix}_train.pt'
    test_pt = f'data/processed/{ds_prefix}_test.pt'

    for p in [tokenizer_path, train_pt, test_pt]:
        if not os.path.exists(p):
            print(f"[ERROR] {p} not found. Run convert_universal_data.py first.")
            sys.exit(1)

    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load data
    train_data = TestbedDataset(root='data', dataset=f'{ds_prefix}_train')
    test_data = TestbedDataset(root='data', dataset=f'{ds_prefix}_test')
    print(f"Dataset: {dataset} | Train: {len(train_data)} | Test: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Model
    model = DeepDTAGen(tokenizer).to(device)
    optimizer = FetterGrad(optim.Adam(model.parameters(), lr=args.lr))
    mse_f = nn.MSELoss()

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Training: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")

    # Directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    FLAGS = lambda: None
    FLAGS.log_dir = 'logs'
    FLAGS.dataset_name = f'kinase_{dataset}_{int(time.time())}'

    # Training loop
    best_auroc = 0
    best_metrics = None

    for epoch in range(args.epochs):
        avg_mse = train_epoch(model, device, train_loader, optimizer, mse_f, epoch)

        if (epoch + 1) % args.eval_every == 0:
            metrics = evaluate(model, device, test_loader)
            msg = (f"Epoch {epoch+1}: MSE={metrics['mse']:.4f}, "
                   f"AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, "
                   f"MCC={metrics['mcc']:.4f}")
            print(f"  [EVAL] {msg}")
            logging(msg, FLAGS)

            # Save best model by AUROC
            if metrics['auroc'] > best_auroc:
                best_auroc = metrics['auroc']
                best_metrics = metrics.copy()
                best_metrics['epoch'] = epoch + 1
                save_path = f'saved_models/deepdtagen_kinase_{dataset}.pth'
                torch.save(model.state_dict(), save_path)
                print(f"  [BEST] Saved to {save_path}")

    # Final evaluation
    print(f"\n{'='*60}")
    print(f" Training complete: {dataset}")
    if best_metrics:
        print(f" Best epoch: {best_metrics['epoch']}")
        print(f" AUROC: {best_metrics['auroc']:.4f}")
        print(f" AUPRC: {best_metrics['auprc']:.4f}")
        print(f" MCC:   {best_metrics['mcc']:.4f}")

        # Save results
        result_row = {
            'model': 'DeepDTAGen_trained',
            'dataset': dataset,
            **best_metrics,
        }
        result_path = f'results/deepdtagen_trained_{dataset}.csv'
        pd.DataFrame([result_row]).to_csv(result_path, index=False)
        print(f" Results: {result_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
