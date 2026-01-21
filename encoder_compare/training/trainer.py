"""Model training utilities."""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..config import TrainingConfig
from .evaluator import evaluate


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Train for one epoch.

    Args:
        model: Neural network model
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0

    for batch in loader:
        protein = batch['protein_matrix'].to(device)
        ligand = batch['ligand_matrix'].to(device)
        protein_mask = batch['protein_mask'].to(device)
        ligand_mask = batch['ligand_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        output = model(protein, ligand, protein_mask, ligand_mask)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device
) -> Tuple[nn.Module, Dict]:
    """
    Train model with early stopping based on validation MCC.

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on

    Returns:
        Tuple of (best_model, training_history)
    """
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_mcc = -1
    best_model_state = None
    best_epoch = 0
    history = {'train_loss': [], 'val_mcc': []}

    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_mcc'].append(val_metrics['mcc'])

        # Update best model
        is_best = val_metrics['mcc'] > best_val_mcc
        if is_best:
            best_val_mcc = val_metrics['mcc']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            best_marker = " 🏆 NEW BEST!" if is_best else ""
            print(f"      Epoch {epoch+1:3d}/{config.num_epochs}: "
                  f"loss={train_loss:.4f}, val_mcc={val_metrics['mcc']:.4f}, "
                  f"val_acc={val_metrics['accuracy']:.4f}, val_auc={val_metrics['auc']:.4f}{best_marker}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"      ✓ Best model from epoch {best_epoch} (val_mcc={best_val_mcc:.4f})")

    return model, history
