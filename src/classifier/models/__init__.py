"""Modelos neurais do DockTKinase Classifier."""

# Imports dos modelos modularizados
from .mlp_classifier import MLPEmbeddingClassifier, create_mlp_model
from .cross_attention_model import (
    CrossAttentionAffinityModel,
    MultiTaskLoss,
    create_cross_attention_model,
    CNNEncoder,
    CrossAttention,
    CrossAttentionBlock
)
from .matrix_embedding_extractor import (
    MatrixEmbeddingExtractor,
    create_synthetic_embeddings,
    extract_matrix_embeddings
)

__all__ = [
    # MLP models
    "MLPEmbeddingClassifier",
    "create_mlp_model",
    # Cross-Attention models
    "CrossAttentionAffinityModel",
    "MultiTaskLoss",
    "create_cross_attention_model",
    "CNNEncoder",
    "CrossAttention",
    "CrossAttentionBlock",
    # Matrix Embedding Extractor
    "MatrixEmbeddingExtractor",
    "create_synthetic_embeddings",
    "extract_matrix_embeddings"
]
