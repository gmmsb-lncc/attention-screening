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

    # Check for NaN or Inf values (indicates training failure)
    import numpy as np
    all_probs = np.array(all_probs)

    if np.any(np.isnan(all_probs)) or np.any(np.isinf(all_probs)):
        raise ValueError(
            "Model produced NaN or Inf values during evaluation!\n"
            "This indicates training instability. Possible causes:\n"
            "  1. Exploding gradients - try gradient clipping\n"
            "  2. Learning rate too high - reduce learning rate\n"
            "  3. Numerical instability - check model architecture\n"
            "Training cannot continue with invalid outputs."
        )

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'mcc': matthews_corrcoef(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    }
