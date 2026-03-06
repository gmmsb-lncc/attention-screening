"""Model training utilities."""

from typing import Tuple, Dict, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import TrainingConfig
from .evaluator import evaluate, optimize_decision_threshold, EvaluationError
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

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

    for batch in progress_bar:
        protein_embeddings = batch['protein_matrix'].to(device)
        ligand_embeddings = batch['ligand_matrix'].to(device)
        protein_padding_mask = batch['protein_mask'].to(device)
        ligand_padding_mask = batch['ligand_mask'].to(device)
        classification_labels = batch['labels'].to(device)
        regression_targets = batch['regression_targets'].to(device)
        regression_mask = batch['regression_mask'].to(device)

        optimizer.zero_grad()

        model_output = model(protein_embeddings, ligand_embeddings, protein_padding_mask, ligand_padding_mask)

        # Handle classification-only models (Level 3)
        if model_output['regression'] is None:
            # Use simple BCE loss for classification only
            classification_logits = model_output['classification'].squeeze(-1)
            classification_labels_squeezed = classification_labels.squeeze(-1).float()
            classification_loss = F.binary_cross_entropy_with_logits(
                classification_logits, 
                classification_labels_squeezed, 
                pos_weight=loss_fn.pos_weight if hasattr(loss_fn, 'pos_weight') else None
            )
            losses = {
                'total': classification_loss, 
                'classification': classification_loss, 
                'regression': torch.tensor(0.0, device=device)
            }
        else:
            # Multi-task loss (for other levels)
            losses = loss_fn(
                model_output['classification'],
                model_output['regression'],
                classification_labels,
                regression_targets,
                regression_mask
            )
        auxiliary_loss = model_output.get('aux_loss')
        if auxiliary_loss is not None:
            scaled_auxiliary_loss = auxiliary_loss * float(aux_loss_scale)
            losses['total'] = losses['total'] + scaled_auxiliary_loss
            losses['aux'] = scaled_auxiliary_loss.detach().item()

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

        progress_info = {
            'loss': f"{losses['total'].item():.4f}",
            'cls': f"{losses['classification'].item():.4f}"
        }
        if 'aux' in losses:
            progress_info['aux'] = f"{losses['aux']:.4f}"
        progress_bar.set_postfix(progress_info)

    if num_batches == 0:
        raise RuntimeError("All batches were skipped due to NaN/Inf values")

    avg_metrics = {
        'total': total_loss / num_batches,
        'classification': total_cls_loss / num_batches,
        'regression': total_reg_loss / num_batches
    }
    if total_aux_loss > 0:
        avg_metrics['aux'] = total_aux_loss / num_batches
    return avg_metrics


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

    # 3-epoch warmup + cosine annealing
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, config.num_epochs - warmup_epochs)
        return 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
            if config.optimize_threshold:
                threshold_result = optimize_decision_threshold(
                    model,
                    val_loader,
                    device,
                    metric=config.threshold_metric,
                    raise_on_invalid=False,
                )
                if not threshold_result.is_valid:
                    warnings.warn(f"Invalid threshold optimization at epoch {epoch+1}")
                    val_metrics = {'mcc': -2.0, 'accuracy': 0.0, 'auc': 0.0}
                else:
                    decision_threshold = float(threshold_result.metrics["decision_threshold"])
                    val_result = evaluate(
                        model,
                        val_loader,
                        device,
                        raise_on_invalid=False,
                        decision_threshold=decision_threshold,
                    )
                    if not val_result.is_valid:
                        warnings.warn(f"Invalid evaluation at epoch {epoch+1}")
                        val_metrics = {'mcc': -2.0, 'accuracy': 0.0, 'auc': 0.0}
                    else:
                        val_metrics = val_result.metrics
            else:
                val_result = evaluate(
                    model,
                    val_loader,
                    device,
                    raise_on_invalid=False,
                    decision_threshold=config.fixed_threshold,
                )
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

        # Print progress every epoch (AUC instead of loss)
        print(
            f"    Epoch {epoch+1}: val_auc={val_metrics['auc']:.4f}, "
            f"val_mcc={val_metrics['mcc']:.4f}, val_acc={val_metrics['accuracy']:.4f}"
        )

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
