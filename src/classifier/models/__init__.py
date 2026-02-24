"""DockTKinase Classifier Neural Models."""

# Core model imports
from .mlp_classifier import MLPEmbeddingClassifier, create_mlp_model
from .cross_attention_model import (
    CrossAttentionAffinityModel,
    CrossAttentionLiteAffinityModel,
    MultiTaskLoss,
    create_cross_attention_model,
    create_cross_attention_lite_model,
    CNNEncoder,
    LinearEncoder,
    CrossAttention,
    CrossAttentionBlock
)
from .matrix_embedding_extractor import (
    MatrixEmbeddingExtractor,
    create_synthetic_embeddings,
    extract_matrix_embeddings
)

# Modular CNN components (optimized architecture)
from .cnn_blocks import (
    SqueezeExcitation,
    DepthwiseSeparableConv1d,
    OptimizedConv1DBlock,
    Conv1DBlock
)
from .cnn_encoder import (
    OptimizedCNNEncoder,
    create_encoder
)

# Positional Encoding (RoPE recommended for variable-length sequences)
from .positional_encoding import (
    RotaryPositionalEmbedding,
    RoPE,
    SinusoidalPositionalEncoding,
    RoPEMultiHeadAttention,
    create_positional_encoding
)

__all__ = [
    # MLP models
    "MLPEmbeddingClassifier",
    "create_mlp_model",
    # Cross-Attention models
    "CrossAttentionAffinityModel",
    "CrossAttentionLiteAffinityModel",
    "MultiTaskLoss",
    "create_cross_attention_model",
    "create_cross_attention_lite_model",
    "CNNEncoder",
    "LinearEncoder",
    "CrossAttention",
    "CrossAttentionBlock",
    # CNN building blocks (modular)
    "SqueezeExcitation",
    "DepthwiseSeparableConv1d",
    "OptimizedConv1DBlock",
    "Conv1DBlock",
    # Optimized CNN encoder
    "OptimizedCNNEncoder",
    "create_encoder",
    # Positional Encoding
    "RotaryPositionalEmbedding",
    "RoPE",
    "SinusoidalPositionalEncoding",
    "RoPEMultiHeadAttention",
    "create_positional_encoding",
    # Matrix Embedding Extractor
    "MatrixEmbeddingExtractor",
    "create_synthetic_embeddings",
    "extract_matrix_embeddings"
]
