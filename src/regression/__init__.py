"""
Módulo de Regressão - DockTKinase
==================================

Módulo para predição de valores de atividade (Ki, Kd, IC50) usando embeddings ESM-2.

COMPATIBILIDADE:
- Mantém imports originais (trainer, evaluator, visualizer)
- Adiciona estrutura modular (core, models, utils)
"""

# Imports originais (compatibilidade) - REFATORADO para usar core/ e models/
from .models.models import RegressionModels  # MOVED: models/ directory
from .core.trainer import RegressionTrainer  # MOVED: core/ directory
from .core.evaluator import RegressionEvaluator  # MOVED: core/ directory
from .visualizer import RegressionVisualizer  # Original: visualizer.py

# Imports de utils originais
try:
    from .utils_original import (
        prepare_regression_targets,
        load_embeddings_cache,
        save_embeddings_cache,
        load_split_indices
    )
except ImportError:
    # Fallback para utils.py direto
    from .utils import (
        prepare_regression_targets,
        load_embeddings_cache,
        save_embeddings_cache,
        load_split_indices
    )

# Imports modulares (novos componentes)
try:
    from .core import DataManager
    from .utils import MetricsCalculator
    from .modular_pipeline import RegressionPipeline
    MODULAR_AVAILABLE = True
except ImportError:
    MODULAR_AVAILABLE = False

__all__ = [
    # Original
    'RegressionModels',
    'RegressionTrainer',
    'RegressionEvaluator',
    'RegressionVisualizer',
    'prepare_regression_targets',
    'load_embeddings_cache',
    'save_embeddings_cache',
    'load_split_indices',
    
    # Modular (se disponível)
    'DataManager',
    'MetricsCalculator',
    'RegressionPipeline',
    'MODULAR_AVAILABLE'
]
