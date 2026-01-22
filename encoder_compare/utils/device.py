"""Device detection and reproducibility utilities."""

import os
import random
import numpy as np
import torch


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).

    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_seed(seed: int, deterministic: bool = True) -> dict:
    """
    Set all random seeds for reproducibility.

    Configures Python, NumPy, and PyTorch random number generators.
    When deterministic=True, also configures CUDA for deterministic operations.

    Args:
        seed: Random seed value
        deterministic: If True, enforce deterministic CUDA operations.
                      May reduce performance but ensures reproducibility.

    Returns:
        Dictionary with configuration details for logging
    """
    # Python built-in random
    random.seed(seed)

    # NumPy - use both legacy and new API for compatibility
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    config = {
        'seed': seed,
        'deterministic': deterministic,
        'cuda_available': torch.cuda.is_available(),
    }

    if deterministic:
        # Ensure deterministic algorithms in PyTorch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+ deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
                config['use_deterministic_algorithms'] = True
            except RuntimeError as e:
                # Some operations don't have deterministic implementations
                # Fall back to allowing non-deterministic with warning
                config['use_deterministic_algorithms'] = False
                config['deterministic_warning'] = str(e)

        # Environment variable for CUDA determinism
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        config['cublas_workspace_config'] = ':4096:8'
    else:
        # Allow non-deterministic for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        config['cudnn_benchmark'] = True

    return config


def get_reproducibility_info() -> dict:
    """
    Get current reproducibility configuration for logging.

    Returns:
        Dictionary with environment and configuration details
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
