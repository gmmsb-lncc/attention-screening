"""Training and evaluation modules."""

from .trainer import train_model, train_epoch
from .evaluator import (
    evaluate,
    optimize_decision_threshold,
    optimize_threshold_from_predictions,
    EvaluationResult,
    EvaluationError,
)

__all__ = [
    'train_model',
    'train_epoch',
    'evaluate',
    'optimize_decision_threshold',
    'optimize_threshold_from_predictions',
    'EvaluationResult',
    'EvaluationError'
]
