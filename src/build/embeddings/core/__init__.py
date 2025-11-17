"""
Core modules for modular embeddings generation.
"""

from .data_loader import DataManager
from .model_manager import ModelManager
from .generator import EmbeddingGenerator

__all__ = [
    'DataManager',
    'ModelManager',
    'EmbeddingGenerator'
]
