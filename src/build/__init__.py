"""
Main build module exports.

Provides convenient imports for the modular build system components.
"""

"""
Main build module exports.

Provides convenient imports for the modular build system components.
"""

# Core components
from build.core import BuildConfig, BaseBuilder, BuildException

# Pipeline orchestrator  
from build.pipeline import BuildPipeline

# Matrix components - for backward compatibility
from build.matrix import EmbeddingMatrix

# Stratification components
from build.stratification import Stratifier, SplitValidator, CosineSimilarityCalculator, ClusterAnalyzer

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
