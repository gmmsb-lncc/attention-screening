"""
MoLFormer - Molecular Transformer for Ligand Embeddings
========================================================

MoLFormer (DeepChem/MoLFormer-c3-1.1B) is a Transformer encoder trained
on SMILES representations of molecules.

Key Advantage over SMI-TED:
- MoLFormer produces MATRIX embeddings [seq_len, 768] (per-token)
- SMI-TED only produces VECTOR embeddings [1, 768] (global)

This makes MoLFormer ideal for cross-attention models that need to
attend over individual tokens in the ligand representation.

Usage:
    from llm.MoLFormer import MoLFormerEmbedder

    embedder = MoLFormerEmbedder()

    # Matrix embedding (per-token)
    matrix = embedder.extract_matrix_embedding("CCO")  # [seq_len, 768]

    # Vector embedding (pooled)
    vector = embedder.extract_vector_embedding("CCO")  # [768]
"""

from .extract_embeddings import MoLFormerEmbedder

__all__ = ["MoLFormerEmbedder"]
__version__ = "1.0.0"
