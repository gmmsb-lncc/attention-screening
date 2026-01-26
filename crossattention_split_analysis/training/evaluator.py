"""Model evaluation utilities with explicit failure handling."""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef


class EvaluationError(Exception):
    """Raised when model evaluation fails due to numerical issues."""
    pass


class EvaluationResult:
    """
    Structured evaluation result with explicit validity tracking.

    Attributes:
        metrics: Dictionary with computed metrics
        is_valid: Whether evaluation completed successfully
        failure_reason: Description of failure if is_valid=False
        nan_count: Number of NaN values detected
        inf_count: Number of Inf values detected
    """

    def __init__(
        self,
        metrics: Dict[str, float],
        is_valid: bool = True,
        failure_reason: Optional[str] = None,
        nan_count: int = 0,
        inf_count: int = 0
    ):
        self.metrics = metrics
        self.is_valid = is_valid
        self.failure_reason = failure_reason
        self.nan_count = nan_count
        self.inf_count = inf_count

    def __getitem__(self, key: str) -> float:
        """Allow dict-like access for backward compatibility."""
        return self.metrics[key]

    def get(self, key: str, default: float = None) -> float:
        return self.metrics.get(key, default)

    def to_dict(self) -> Dict:
        return {
            **self.metrics,
            '_is_valid': self.is_valid,
            '_failure_reason': self.failure_reason,
            '_nan_count': self.nan_count,
            '_inf_count': self.inf_count
        }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    raise_on_invalid: bool = True
) -> EvaluationResult:
    """
    Evaluate model on a dataset with explicit failure handling.

    Args:
        model: Neural network model (CrossAttentionAffinityModel)
        data_loader: Data loader
        device: Device to evaluate on
        raise_on_invalid: If True, raise EvaluationError on NaN/Inf

    Returns:
        EvaluationResult with metrics and validity status

    Raises:
        EvaluationError: If model produces NaN/Inf and raise_on_invalid=True
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            protein_matrix = batch['protein_matrix'].to(device)
            ligand_matrix = batch['ligand_matrix'].to(device)
            protein_mask = batch['protein_mask'].to(device)
            ligand_mask = batch['ligand_mask'].to(device)
            labels = batch['labels']

            output = model(protein_matrix, ligand_matrix, protein_mask, ligand_mask)
            probs = torch.sigmoid(output['classification']).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy().flatten())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Count invalid values
    nan_count = int(np.isnan(all_probs).sum())
    inf_count = int(np.isinf(all_probs).sum())

    if nan_count > 0 or inf_count > 0:
        failure_msg = (
            f"Model produced invalid values: {nan_count} NaN, {inf_count} Inf "
            f"out of {len(all_probs)} predictions."
        )

        if raise_on_invalid:
            raise EvaluationError(failure_msg)

        return EvaluationResult(
            metrics={
                'accuracy': np.nan,
                'f1': np.nan,
                'mcc': np.nan,
                'auc': np.nan
            },
            is_valid=False,
            failure_reason=failure_msg,
            nan_count=nan_count,
            inf_count=inf_count
        )

    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'f1': float(f1_score(all_labels, all_preds, zero_division=0)),
        'mcc': float(matthews_corrcoef(all_labels, all_preds)),
        'auc': float(
            roc_auc_score(all_labels, all_probs)
            if len(np.unique(all_labels)) > 1 else np.nan
        )
    }

    return EvaluationResult(metrics=metrics, is_valid=True)
