"""Training and evaluation modules."""

from .trainer import train_model as original_train_model, train_epoch as original_train_epoch
from .stable_trainer import train_model as stable_train_model, train_epoch as stable_train_epoch
from .evaluator import evaluate, EvaluationResult, EvaluationError

# Default to stable trainer for better numerical stability
train_model = stable_train_model
train_epoch = stable_train_epoch

__all__ = [
    'train_model', 'train_epoch',
    'original_train_model', 'original_train_epoch',
    'stable_train_model', 'stable_train_epoch',
    'evaluate', 'EvaluationResult', 'EvaluationError'
]
