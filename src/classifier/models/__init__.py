"""Modelos neurais do DockTKinase Classifier."""

# Imports dos modelos modularizados
from .mlp_classifier import MLPEmbeddingClassifier, create_mlp_model

__all__ = [
    "MLPEmbeddingClassifier",
    "create_mlp_model"
]
