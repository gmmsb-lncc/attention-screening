"""
Embeddings module for protein and ligand representation generation.

Provides base classes and implementations for generating embeddings using ESM and FM4M models.
"""

from build.embeddings.base_embedding import BaseEmbedding
from build.embeddings.protein_embedding import ProteinEmbedding
from build.embeddings.ligand_embedding import LigandEmbedding

__all__ = [
    'BaseEmbedding',
    'ProteinEmbedding', 
    'LigandEmbedding'
]
