"""Model training utilities."""

from typing import Tuple, Dict, Optional
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import TrainingConfig
from .evaluator import evaluate, EvaluationError
from ..utils.checkpoints import save_checkpoint, load_checkpoint


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    max_grad_norm: float = 1.0,
    aux_loss_scale: float = 1.0,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: MultiTaskLoss function
        device: Device to train on
        epoch: Current epoch number
        num_epochs: Total number of epochs
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Dictionary with loss values
    """
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    total_aux_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

    for batch in pbar:
        protein_matrix = batch['protein_matrix'].to(device)
        ligand_matrix = batch['ligand_matrix'].to(device)
        protein_mask = batch['protein_mask'].to(device)
        ligand_mask = batch['ligand_mask'].to(device)
        labels = batch['labels'].to(device)
        reg_targets = batch['regression_targets'].to(device)
        reg_mask = batch['regression_mask'].to(device)

        optimizer.zero_grad()

        output = model(protein_matrix, ligand_matrix, protein_mask, ligand_mask)

        losses = loss_fn(
            output['classification'],
            output['regression'],
            labels,
            reg_targets,
            reg_mask
        )
        aux_loss = output.get('aux_loss')
        if aux_loss is not None:
            scaled_aux = aux_loss * float(aux_loss_scale)
            losses['total'] = losses['total'] + scaled_aux
            losses['aux'] = scaled_aux.detach().item()

        # Check for NaN loss
        if torch.isnan(losses['total']) or torch.isinf(losses['total']):
            warnings.warn("NaN/Inf loss detected, skipping batch")
            continue

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += losses['total'].item()
        total_cls_loss += losses['classification'].item()
        total_reg_loss += losses['regression'].item()
        if 'aux' in losses:
            total_aux_loss += losses['aux']
        num_batches += 1

        postfix = {
            'loss': f"{losses['total'].item():.4f}",
            'cls': f"{losses['classification'].item():.4f}"
        }
        if 'aux' in losses:
            postfix['aux'] = f"{losses['aux']:.4f}"
        pbar.set_postfix(postfix)

    if num_batches == 0:
        raise RuntimeError("All batches were skipped due to NaN/Inf values")

    metrics = {
        'total': total_loss / num_batches,
        'classification': total_cls_loss / num_batches,
        'regression': total_reg_loss / num_batches
    }
    if total_aux_loss > 0:
        metrics['aux'] = total_aux_loss / num_batches
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    loss_fn: nn.Module,
    checkpoint_path: Optional[str] = None,
    checkpoint_interval: int = 10
) -> Tuple[nn.Module, Dict]:
    """
    Train model with early stopping and checkpointing.

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        loss_fn: Loss function (MultiTaskLoss)
        checkpoint_path: Path for saving/loading checkpoints
        checkpoint_interval: Save checkpoint every N epochs

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

    best_val_mcc = -1
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_mcc': [], 'val_acc': []}
    start_epoch = 0

    # Load checkpoint if exists
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(checkpoint_path, device)
        if checkpoint is not None:
            if checkpoint.get('scenario_completed', False):
                print("  [CHECKPOINT] Scenario already completed, loading best model...")
                if checkpoint['best_model_state'] is not None:
                    model.load_state_dict(checkpoint['best_model_state'])
                return model, checkpoint['history']

            # Resume training
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_val_mcc = checkpoint['best_val_mcc']
            best_epoch = checkpoint.get('best_epoch', 0)
            best_model_state = checkpoint['best_model_state']
            patience_counter = checkpoint['patience_counter']
            history = checkpoint['history']
            start_epoch = checkpoint['epoch'] + 1
            print(f"  [CHECKPOINT] Resuming from epoch {start_epoch}")

    print(f"  Training for {config.num_epochs} epochs...")
    if start_epoch > 0:
        print(f"  Continuing from epoch {start_epoch + 1}...")

    for epoch in range(start_epoch, config.num_epochs):
        if config.diffusion_loss_anneal == "linear":
            denom = max(1, config.num_epochs - 1)
            aux_loss_scale = max(0.0, 1.0 - (epoch / denom))
        else:
            aux_loss_scale = 1.0

        # Train
        try:
            train_metrics = train_epoch(
                model, train_loader, optimizer, loss_fn, device,
                epoch, config.num_epochs, config.max_grad_norm, aux_loss_scale
            )
        except RuntimeError as e:
            if "skipped" in str(e):
                print(f"ERROR: Training failed at epoch {epoch+1} - {e}")
                break
            raise

        # Validate
        try:
            val_result = evaluate(model, val_loader, device, raise_on_invalid=False)
            if not val_result.is_valid:
                warnings.warn(f"Invalid evaluation at epoch {epoch+1}")
                val_metrics = {'mcc': -2.0, 'accuracy': 0.0, 'auc': 0.0}
            else:
                val_metrics = val_result.metrics
        except EvaluationError as e:
            warnings.warn(f"Evaluation failed at epoch {epoch+1}: {e}")
            val_metrics = {'mcc': -2.0, 'accuracy': 0.0, 'auc': 0.0}

        scheduler.step()

        history['train_loss'].append(train_metrics['total'])
        history['val_mcc'].append(val_metrics['mcc'])
        history['val_acc'].append(val_metrics['accuracy'])

        # Track best model
        improved = val_metrics['mcc'] > best_val_mcc
        if improved:
            best_val_mcc = val_metrics['mcc']
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}: loss={train_metrics['total']:.4f}, "
                  f"val_mcc={val_metrics['mcc']:.4f}, val_acc={val_metrics['accuracy']:.4f}")

        # Save checkpoint
        if checkpoint_path is not None:
            should_save = ((epoch + 1) % checkpoint_interval == 0) or improved
            if should_save:
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_mcc=best_val_mcc,
                    best_epoch=best_epoch,
                    best_model_state=best_model_state,
                    patience_counter=patience_counter,
                    history=history,
                    scenario_completed=False
                )

        # Early stopping
        if config.patience is not None and patience_counter >= config.patience:
            print(f"    Early stopping at epoch {epoch+1} (patience={config.patience})")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Best model from epoch {best_epoch} (val_mcc={best_val_mcc:.4f})")

    # Save final checkpoint
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_mcc=best_val_mcc,
            best_epoch=best_epoch,
            best_model_state=best_model_state,
            patience_counter=patience_counter,
            history=history,
            scenario_completed=True
        )
        print("  [CHECKPOINT] Saved final checkpoint (completed)")

    return model, history
