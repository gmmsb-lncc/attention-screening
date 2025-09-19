"""
Embeddings module for protein and ligand representation generation.

Provides base classes and implementations for generating embeddings using ESM and FM4M models.
"""

from .base_embedding import BaseEmbedding
from .protein_embedding import ProteinEmbedding
from .ligand_embedding import LigandEmbedding

__all__ = [
    'BaseEmbedding',
    'ProteinEmbedding', 
    'LigandEmbedding'
]
