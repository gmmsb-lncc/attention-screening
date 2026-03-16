"""Checkpoint management for resuming experiments."""

import os
from typing import Optional, Dict
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim


def get_checkpoint_path(output_dir: str, prefix: str, scenario_name: str) -> str:
    """
    Get checkpoint file path for a scenario.

    Args:
        output_dir: Output directory
        prefix: File prefix (e.g., 'non_human_t30_150M_')
        scenario_name: Scenario name

    Returns:
        Path to checkpoint file
    """
    safe_scenario = (
        scenario_name
        .replace('\n', '_')
        .replace(' ', '_')
        .replace('+', 'plus')
    )
    return os.path.join(output_dir, f'{prefix}checkpoint_{safe_scenario}.pt')


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val_mcc: float,
    best_selection_score: float,
    model_selection_metric: str,
    best_epoch: int,
    best_model_state: Optional[dict],
    patience_counter: int,
    history: Dict,
    scenario_completed: bool = False
) -> str:
    """
    Save training checkpoint to disk.

    Uses atomic write (temp file + rename) to prevent corruption.

    Args:
        checkpoint_path: Path to save checkpoint
        model: Current model state
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        best_val_mcc: Best validation MCC so far
        best_epoch: Epoch of best model
        best_model_state: State dict of best model
        patience_counter: Current patience counter
        history: Training history
        scenario_completed: Whether scenario is finished

    Returns:
        Path to saved checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_mcc': best_val_mcc,
        'best_selection_score': best_selection_score,
        'model_selection_metric': model_selection_metric,
        'best_epoch': best_epoch,
        'best_model_state': best_model_state,
        'patience_counter': patience_counter,
        'history': history,
        'scenario_completed': scenario_completed,
        'timestamp': datetime.now().isoformat()
    }

    # Atomic write: save to temp file, then rename
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    temp_path = checkpoint_path + '.tmp'
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, checkpoint_path)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device
) -> Optional[Dict]:
    """
    Load checkpoint from disk if it exists.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to map tensors to

    Returns:
        Checkpoint dictionary or None if not found
    """
    if not os.path.exists(checkpoint_path):
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"  [CHECKPOINT] Loaded from epoch {checkpoint['epoch']+1}")
        print(f"               Best val_mcc: {checkpoint['best_val_mcc']:.4f}")
        if 'best_selection_score' in checkpoint and 'model_selection_metric' in checkpoint:
            print(
                "               Best "
                f"{checkpoint['model_selection_metric']}: "
                f"{checkpoint['best_selection_score']:.6f}"
            )
        print(f"               Saved at: {checkpoint['timestamp']}")
        return checkpoint
    except Exception as e:
        print(f"  [WARNING] Failed to load checkpoint: {e}")
        return None


def is_scenario_completed(checkpoint_path: str, device: torch.device) -> bool:
    """Check if a scenario has been completed."""
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint is None:
        return False
    return checkpoint.get('scenario_completed', False)
