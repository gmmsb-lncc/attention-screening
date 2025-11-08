"""
Core modules for modular embeddings generation.
"""

from build.embeddings.core.data_loader import DataManager
from build.embeddings.core.model_manager import ModelManager
from build.embeddings.core.generator import EmbeddingGenerator

__all__ = [
    'DataManager',
    'ModelManager',
    'EmbeddingGenerator'
]
