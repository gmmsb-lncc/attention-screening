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
    "CrossAttentionBlock"
]
