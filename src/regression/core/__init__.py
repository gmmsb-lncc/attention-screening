"""
Core modules para regressão - DockTKinase
==========================================

Módulos centrais do pipeline de regressão modularizado.
"""

from .evaluator import RegressionEvaluator
from .data_loader import DataManager
from .trainer import RegressionTrainer
from .cross_validator import (
    RegressionCrossValidator,
    CrossValidationConfig,
    CrossValidationResults,
    FoldMetrics,
    quick_cross_validate
)

__all__ = [
    'RegressionEvaluator',
    'DataManager',
    'RegressionTrainer',
    'RegressionCrossValidator',
    'CrossValidationConfig',
    'CrossValidationResults',
    'FoldMetrics',
    'quick_cross_validate'
]
