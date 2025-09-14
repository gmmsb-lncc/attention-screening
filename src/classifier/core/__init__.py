"""Lógica principal de treinamento e validação."""

from .trainer import ModelTrainer, TrainingConfig
from .cross_validator import CrossValidator, CrossValidationConfig
from .hyperopt import HyperparameterOptimizer, OptimizationConfig

# Módulos opcionais com dependências externas
try:
    from .data_manager import DataManager, ScalableDataset, DatasetInfo
    DATA_MANAGER_AVAILABLE = True
except ImportError:
    DATA_MANAGER_AVAILABLE = False

try:
    from .memory_manager import MemoryManager, MemoryTracker, MemorySnapshot
    MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    MEMORY_MANAGER_AVAILABLE = False

__all__ = [
    'ModelTrainer', 'TrainingConfig',
    'CrossValidator', 'CrossValidationConfig',
    'HyperparameterOptimizer', 'OptimizationConfig'
]

# Adicionar módulos opcionais se disponíveis
if DATA_MANAGER_AVAILABLE:
    __all__.extend(['DataManager', 'ScalableDataset', 'DatasetInfo'])

if MEMORY_MANAGER_AVAILABLE:
    __all__.extend(['MemoryManager', 'MemoryTracker', 'MemorySnapshot'])
