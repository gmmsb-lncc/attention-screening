"""
Database Core Module - Base classes and configuration for database operations.

This module provides the foundation for database analysis and processing operations,
including base classes, configuration management, and common utilities.
"""

from .config import DatabaseConfig
from .base_analyzer import BaseAnalyzer
from .exceptions import DatabaseError, AnalysisError, ProcessingError

__all__ = [
    'DatabaseConfig',
    'BaseAnalyzer', 
    'DatabaseError',
    'AnalysisError',
    'ProcessingError'
]
