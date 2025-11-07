"""
Utils para regressão - DockTKinase
===================================

Utilitários e funções auxiliares para regressão.
"""

from .metrics import MetricsCalculator, calculate_regression_metrics

# Importar funções do utils.py original (no diretório pai)
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from utils import (
        prepare_regression_targets,
        load_embeddings_cache,
        save_embeddings_cache,
        load_split_indices
    )
except ImportError:
    # Fallback se não existir
    prepare_regression_targets = None
    load_embeddings_cache = None
    save_embeddings_cache = None
    load_split_indices = None

__all__ = [
    'MetricsCalculator',
    'calculate_regression_metrics',
    'prepare_regression_targets',
    'load_embeddings_cache',
    'save_embeddings_cache',
    'load_split_indices'
]
