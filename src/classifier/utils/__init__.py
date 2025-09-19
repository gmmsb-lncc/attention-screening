"""Utilitários, métricas e validação de dados."""

# Imports dos utilitários modularizados  
from .import_utils import (
    get_classifier_root,
    setup_classifier_imports,
    safe_import,
    safe_import_optional
)

__all__ = [
    "get_classifier_root", 
    "setup_classifier_imports",
    "safe_import",
    "safe_import_optional"
]
