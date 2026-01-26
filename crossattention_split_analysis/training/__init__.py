"""Training and evaluation modules."""

from .trainer import train_model, train_epoch
from .evaluator import evaluate, EvaluationResult, EvaluationError

__all__ = [
    'train_model',
    'train_epoch',
    'evaluate',
    'EvaluationResult',
    'EvaluationError'
]
