"""
Core modules for modular embeddings generation.
"""

from .embeddings.core.data_loader import DataManager
from .embeddings.core.model_manager import ModelManager
from .embeddings.core.generator import EmbeddingGenerator

__all__ = [
    'DataManager',
    'ModelManager',
    'EmbeddingGenerator'
]
