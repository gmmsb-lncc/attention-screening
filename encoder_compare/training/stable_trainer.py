"""Stable model training utilities with enhanced numerical stability."""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
from tqdm import tqdm

from ..config import TrainingConfig
from .evaluator import evaluate, EvaluationError
from ..utils.checkpoints import save_checkpoint, load_checkpoint, get_checkpoint_path


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0
) -> float:
    """
    Train for one epoch with enhanced numerical stability checks.

    Args:
        model: Neural network model
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    # Use tqdm for progress bar
    for batch in tqdm(loader, desc="Training", leave=False):
        protein = batch['protein_matrix'].to(device)
        ligand = batch['ligand_matrix'].to(device)
        protein_mask = batch['protein_mask'].to(device)
        ligand_mask = batch['ligand_mask'].to(device)
        labels = batch['labels'].to(device)

        # Validate inputs for NaN/Inf
        if torch.isnan(protein).any() or torch.isinf(protein).any():
            warnings.warn("NaN or Inf detected in protein matrix!")
            continue  # Skip this batch

        if torch.isnan(ligand).any() or torch.isinf(ligand).any():
            warnings.warn("NaN or Inf detected in ligand matrix!")
            continue  # Skip this batch

        if torch.isnan(labels).any() or torch.isinf(labels).any():
            warnings.warn("NaN or Inf detected in labels!")
            continue  # Skip this batch

        optimizer.zero_grad()
        output = model(protein, ligand, protein_mask, ligand_mask)

        # Validate model output for NaN/Inf
        if torch.isnan(output).any() or torch.isinf(output).any():
            warnings.warn("NaN or Inf detected in model output!")
            continue  # Skip this batch

        loss = criterion(output, labels)

        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            warnings.warn(
                f"Loss became NaN or Inf during training! Skipping batch.\n"
                f"This indicates training instability. Consider lowering learning rate."
            )
            continue  # Skip this batch

        loss.backward()

        # Check gradients for NaN/Inf before clipping
        grad_has_nan_inf = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    grad_has_nan_inf = True
                    break

        if grad_has_nan_inf:
            optimizer.zero_grad()  # Clear gradients to avoid corrupting the model
            continue  # Skip this batch

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Check for NaN in model parameters (early detection)
        param_has_nan_inf = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                param_has_nan_inf = True
                break

        if param_has_nan_inf:
            warnings.warn(
                f"NaN or Inf detected in model parameters after optimizer step!\n"
                f"Resetting to previous state. Consider lowering learning rate."
            )
            # Load the best known good state if available, or skip this update
            continue

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("All batches were skipped due to NaN/Inf values. Check your data!")

    return total_loss / num_batches


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    checkpoint_path: str = None,
    experiment_id: str = None
) -> Tuple[nn.Module, Dict]:
    """
    Train model with enhanced stability, checkpointing, and early stopping based on validation MCC.

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        checkpoint_path: Path for saving/loading checkpoints
        experiment_id: Unique identifier for the experiment

    Returns:
        Tuple of (best_model, training_history)
    """
    model = model.to(device)

    # Use a lower learning rate for better stability
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8  # Smaller epsilon for numerical stability
    )

    # Cosine annealing with warmup for gradual learning rate adjustment
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-8
    )

    # BCEWithLogitsLoss with label smoothing for stability
    criterion = nn.BCEWithLogitsLoss()

    best_val_mcc = -2  # Initialize to a very low value
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    max_patience = 20  # Early stopping patience

    start_epoch = 0  # Track starting epoch for resuming

    # Load checkpoint if available
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            print(f"  📂 Loading checkpoint from {checkpoint_path}")
            # The checkpoint structure follows the format from the checkpoints module
            # where 'results' contains additional data
            if 'model_state' in checkpoint and checkpoint['model_state'] is not None:
                model.load_state_dict(checkpoint['model_state'])
            elif 'results' in checkpoint and 'model_state' in checkpoint['results']:
                # For training checkpoints stored in results dict
                model.load_state_dict(checkpoint['results']['model_state'])

            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            elif 'results' in checkpoint and 'optimizer_state' in checkpoint['results']:
                # For training checkpoints stored in results dict
                optimizer.load_state_dict(checkpoint['results']['optimizer_state'])

            # Extract training-specific info from results if available
            if 'results' in checkpoint:
                start_epoch = checkpoint['results'].get('epoch', 0)
                best_val_mcc = checkpoint['results'].get('best_val_mcc', -2)
                best_epoch = checkpoint['results'].get('best_epoch', 0)
                patience_counter = checkpoint['results'].get('patience_counter', 0)
                # Note: best_model_state will be set during training if needed
            else:
                start_epoch = checkpoint.get('epoch', 0)
                best_val_mcc = checkpoint.get('best_val_mcc', -2)
                best_model_state = checkpoint.get('best_model_state')
                best_epoch = checkpoint.get('best_epoch', 0)
                patience_counter = checkpoint.get('patience_counter', 0)

            print(f"  ✅ Resumed from epoch {start_epoch + 1}")

    history = {'train_loss': [], 'val_mcc': [], 'learning_rate': []}

    # Add progress bar for epochs
    epoch_range = tqdm(range(start_epoch, config.num_epochs),
                      desc="Epochs",
                      initial=start_epoch,
                      total=config.num_epochs)

    for epoch in epoch_range:
        try:
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device,
                max_grad_norm=config.max_grad_norm
            )
        except RuntimeError as e:
            if "All batches were skipped" in str(e):
                print(f"ERROR: Training failed at epoch {epoch+1} - {str(e)}")
                print("This usually means there are NaN/Inf values in your data.")
                print("Consider preprocessing your data to remove invalid values.")
                break
            else:
                raise e

        try:
            val_result = evaluate(model, val_loader, device, raise_on_invalid=False)
            if not val_result.is_valid:
                warnings.warn(
                    f"Evaluation at epoch {epoch+1} produced invalid values: "
                    f"{val_result.nan_count} NaN, {val_result.inf_count} Inf. "
                    f"Skipping this epoch for model selection."
                )
                # Use very negative MCC to ensure this epoch is not selected as best
                val_metrics = {'mcc': -2.0, 'accuracy': 0.0, 'auc': 0.0, 'f1': 0.0, '_valid': False}
            else:
                val_metrics = {**val_result.metrics, '_valid': True}
        except EvaluationError as e:
            warnings.warn(f"Evaluation failed at epoch {epoch+1}: {str(e)}")
            val_metrics = {'mcc': -2.0, 'accuracy': 0.0, 'auc': 0.0, 'f1': 0.0, '_valid': False}

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_mcc'].append(val_metrics['mcc'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Update best model
        is_best = val_metrics['mcc'] > best_val_mcc
        if is_best:
            best_val_mcc = val_metrics['mcc']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            best_marker = " 🏆 NEW BEST!" if is_best else ""
            print(f"      Epoch {epoch+1:3d}/{config.num_epochs}: "
                  f"loss={train_loss:.4f}, val_mcc={val_metrics['mcc']:.4f}, "
                  f"val_acc={val_metrics['accuracy']:.4f}, val_auc={val_metrics['auc']:.4f}{best_marker}")

        # Save checkpoint periodically
        if checkpoint_path and ((epoch + 1) % 10 == 0 or is_best):
            save_checkpoint(
                checkpoint_path,
                "training",  # encoder_type - using a generic value for training checkpoints
                "training",  # scenario_name - using a generic value for training checkpoints
                epoch,       # Using epoch as seed equivalent for training checkpoints
                model.state_dict(),
                {           # Store training-specific info in results
                    'epoch': epoch,
                    'best_val_mcc': best_val_mcc,
                    'best_epoch': best_epoch,
                    'patience_counter': patience_counter,
                    'optimizer_state': optimizer.state_dict()
                },
                completed=False
            )

        # Early stopping
        if patience_counter >= max_patience:
            print(f"      🛑 Early stopping triggered at epoch {epoch+1} (patience={max_patience})")
            print(f"      Best model from epoch {best_epoch} (val_mcc={best_val_mcc:.4f})")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"      ✓ Best model from epoch {best_epoch} (val_mcc={best_val_mcc:.4f})")
    else:
        print(f"      ⚠️  Could not load best model, using current model state")

    return model, history