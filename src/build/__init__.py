"""
Main build module exports.

Provides convenient imports for the modular build system components.
"""

# Core components
from .core import BuildConfig, BaseBuilder, BuildException

# Pipeline orchestrator  
from .pipeline import BuildPipeline

# Matrix components - for backward compatibility
from .matrix import EmbeddingMatrix

# Backward compatibility aliases
EmbeddingMatrixReconstructor = EmbeddingMatrix

__all__ = [
    'BuildConfig',
    'BaseBuilder', 
    'BuildException',
    'BuildPipeline',
    'EmbeddingMatrix',
    'EmbeddingMatrixReconstructor'
]

__version__ = '1.0.0'
