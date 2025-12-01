"""
Main build module exports.

Provides convenient imports for the modular build system components.
"""

# ============================================================================
# CRITICAL: Fix OpenMP conflict on macOS
# FAISS and other libraries (PyTorch, sklearn, scipy) use different OpenMP
# runtimes. This must be set BEFORE any imports.
# ============================================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import FAISS first to avoid OpenMP conflicts (optional dependency)
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

# Core components
from .core import BuildConfig, BaseBuilder, BuildException

# Pipeline orchestrator  
from .pipeline import BuildPipeline

# Matrix components - for backward compatibility
from .matrix import EmbeddingMatrix

# Stratification components
from .stratification import Stratifier, SplitValidator, CosineSimilarityCalculator, ClusterAnalyzer

# Backward compatibility aliases
EmbeddingMatrixReconstructor = EmbeddingMatrix

__all__ = [
    'BuildConfig',
    'BaseBuilder', 
    'BuildException',
    'BuildPipeline',
    'EmbeddingMatrix',
    'Stratifier',
    'SplitValidator', 
    'CosineSimilarityCalculator',
    'ClusterAnalyzer',
    'EmbeddingMatrixReconstructor'
]

__version__ = '1.0.0'
