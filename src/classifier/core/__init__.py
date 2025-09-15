"""Lógica principal de treinamento e validação."""

from .trainer import ModelTrainer, TrainingConfig
from .cross_validator import CrossValidator, CrossValidationConfig

# Hyperopt movido para optional/
try:
    from ..optional.hyperopt import HyperparameterOptimizer, OptimizationConfig
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

# Módulos opcionais com dependências externas
try:
    from .data_manager import DataManager, SimpleDataManager, Dataset
    DATA_MANAGER_AVAILABLE = True
except ImportError:
    try:
        from .data_manager import DataManager, SimpleDataManager
        # Dataset pode não estar disponível em versões antigas
        from .data_manager import ScalableDataset as Dataset
        DATA_MANAGER_AVAILABLE = True
    except ImportError:
        DATA_MANAGER_AVAILABLE = False

# Memory manager removido
MEMORY_MANAGER_AVAILABLE = False

__all__ = [
    'ModelTrainer', 'TrainingConfig',
    'CrossValidator', 'CrossValidationConfig'
]

# Adicionar hyperopt se disponível
if HYPEROPT_AVAILABLE:
    __all__.extend(['HyperparameterOptimizer', 'OptimizationConfig'])
if DATA_MANAGER_AVAILABLE:
    __all__.extend(['DataManager', 'ScalableDataset', 'DatasetInfo'])

if MEMORY_MANAGER_AVAILABLE:
    __all__.extend(['MemoryManager', 'MemoryTracker', 'MemorySnapshot'])
