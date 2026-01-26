"""Device detection and reproducibility utilities."""

import os
import random
from typing import Dict
import numpy as np
import torch


def get_device() -> torch.device:
    """
    Get the best available device.

    Returns:
        torch.device (cuda > mps > cpu)
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed: int, deterministic: bool = True) -> Dict:
    """
    Set all random seeds for reproducibility.

    Configures Python, NumPy, and PyTorch random number generators.
    When deterministic=True, also configures CUDA for deterministic operations.

    Args:
        seed: Random seed value
        deterministic: If True, enforce deterministic CUDA operations

    Returns:
        Dictionary with configuration details
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    config = {
        'seed': seed,
        'deterministic': deterministic,
        'cuda_available': torch.cuda.is_available(),
    }

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
                config['use_deterministic_algorithms'] = True
            except RuntimeError:
                config['use_deterministic_algorithms'] = False

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        config['cublas_workspace_config'] = ':4096:8'

    return config


def get_reproducibility_info() -> Dict:
    """
    Get current reproducibility configuration for logging.

    Returns:
        Dictionary with environment details
    """
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['gpu_name'] = torch.cuda.get_device_name(0)

    return info
