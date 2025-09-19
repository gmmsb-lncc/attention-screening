"""
Matrix module for embedding matrix construction.

Provides base classes and implementations for building concatenated embedding matrices.
"""

from .base_matrix import BaseMatrix
from .embedding_matrix import EmbeddingMatrix
from .kinase_matrix import KinaseMatrix

# Backward compatibility alias
EmbeddingMatrixReconstructor = EmbeddingMatrix

__all__ = [
    'BaseMatrix',
    'EmbeddingMatrix',
    'KinaseMatrix',
    'EmbeddingMatrixReconstructor'
]
