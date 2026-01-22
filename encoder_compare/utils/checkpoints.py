"""Checkpoint management for resuming experiments."""

import os
from typing import Optional, Dict
from datetime import datetime
import torch


def save_checkpoint(
    checkpoint_path: str,
    encoder_type: Optional[str],
    scenario_name: Optional[str],
    seed: Optional[int],
    model_state: Optional[dict],
    results: dict,
    completed: bool = False
) -> None:
    """
    Save checkpoint for resuming later.

    Args:
        checkpoint_path: Path to save checkpoint
        encoder_type: Current encoder type
        scenario_name: Current scenario name
        seed: Current seed
        model_state: Model state dict (not used currently)
        results: Results dictionary
        completed: Whether all experiments are completed
    """
    checkpoint = {
        'encoder_type': encoder_type,
        'scenario_name': scenario_name,
        'seed': seed,
        'model_state': model_state,
        'results': results,
        'completed': completed,
        'timestamp': datetime.now().isoformat()
    }
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str) -> Optional[Dict]:
    """
    Load checkpoint if exists.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Checkpoint dictionary or None if doesn't exist
    """
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path, weights_only=False)
    return None


def get_checkpoint_path(output_dir: str, embedding_name: str, dataset_type: str) -> str:
    """
    Get checkpoint file path.

    Args:
        output_dir: Output directory
        embedding_name: Embedding model name
        dataset_type: Dataset type ('human' or 'non_human')

    Returns:
        Path to checkpoint file
    """
    short_name = embedding_name.replace('esm2_', '').replace('_UR50D', '')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    return os.path.join(checkpoint_dir, f'{dataset_type}_{short_name}_checkpoint.pt')
