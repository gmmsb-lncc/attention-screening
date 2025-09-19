"""
Matrix module for embedding matrix construction.

Provides base classes and implementations for building concatenated embedding matrices.
"""

from build.matrix.base_matrix import BaseMatrix
from build.matrix.embedding_matrix import EmbeddingMatrix
from build.matrix.kinase_matrix import KinaseMatrix

# Backward compatibility alias
EmbeddingMatrixReconstructor = EmbeddingMatrix

__all__ = [
    'BaseMatrix',
    'EmbeddingMatrix',
    'KinaseMatrix',
    'EmbeddingMatrixReconstructor'
]
