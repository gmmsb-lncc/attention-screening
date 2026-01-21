"""Model evaluation utilities."""

from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    """
    Evaluate model on a dataset.

    Args:
        model: Neural network model
        loader: Data loader
        device: Device to evaluate on

    Returns:
        Dictionary with metrics: accuracy, f1, mcc, auc
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            protein = batch['protein_matrix'].to(device)
            ligand = batch['ligand_matrix'].to(device)
            protein_mask = batch['protein_mask'].to(device)
            ligand_mask = batch['ligand_mask'].to(device)
            labels = batch['labels'].to(device)

            output = model(protein, ligand, protein_mask, ligand_mask)
            probs = torch.sigmoid(output)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend((probs > 0.5).cpu().numpy().flatten().astype(int))
            all_labels.extend(labels.cpu().numpy().flatten().astype(int))

    # Calculate metrics (handle NaN values)
    import numpy as np
    all_probs = np.array(all_probs)

    # Check for NaN or Inf in probabilities
    if np.any(np.isnan(all_probs)) or np.any(np.isinf(all_probs)):
        print("WARNING: Model produced NaN or Inf values. Replacing with 0.5")
        all_probs = np.nan_to_num(all_probs, nan=0.5, posinf=0.5, neginf=0.5)
        all_preds = [1 if p > 0.5 else 0 for p in all_probs]

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'mcc': matthews_corrcoef(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    }
