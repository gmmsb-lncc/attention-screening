"""
Utils para regressão - DockTKinase
===================================

Utilitários e funções auxiliares para regressão.
"""

from .metrics import MetricsCalculator, calculate_regression_metrics

# Importar funções do utils.py original (no diretório pai)
# utils.py está em src/regression/utils.py (mesmo nível que este diretório utils/)
import importlib.util
from pathlib import Path

# Carregar utils.py diretamente pelo caminho
_utils_path = Path(__file__).parent.parent / "utils.py"
if _utils_path.exists():
    _spec = importlib.util.spec_from_file_location("regression_utils", _utils_path)
    _utils_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_utils_module)
    
    prepare_regression_targets = _utils_module.prepare_regression_targets
    load_embeddings_cache = _utils_module.load_embeddings_cache
    save_embeddings_cache = _utils_module.save_embeddings_cache
    load_split_indices = _utils_module.load_split_indices
    save_split_indices = _utils_module.save_split_indices
else:
    # Fallback se não existir
    prepare_regression_targets = None
    load_embeddings_cache = None
    save_embeddings_cache = None
    load_split_indices = None
    save_split_indices = None

__all__ = [
    'MetricsCalculator',
    'calculate_regression_metrics',
    'prepare_regression_targets',
    'load_embeddings_cache',
    'save_embeddings_cache',
    'load_split_indices'
]
