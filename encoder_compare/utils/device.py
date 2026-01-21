"""Device detection utilities."""

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
