"""
Pipeline module for orchestrating the complete build process.

Provides the main pipeline coordinator and utilities for
executing the full embedding matrix construction workflow.
"""

from build.pipeline.build_pipeline import BuildPipeline

__all__ = [
    'BuildPipeline'
]
