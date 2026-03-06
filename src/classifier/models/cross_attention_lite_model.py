"""
Cross-Attention Lite model entry points.

This module keeps a dedicated import path for the lightweight variant while
reusing the core implementation in cross_attention_model.py.
"""

from .cross_attention_model import (
    CrossAttentionLiteAffinityModel,
    create_cross_attention_lite_model,
)

__all__ = [
    "CrossAttentionLiteAffinityModel",
    "create_cross_attention_lite_model",
]
