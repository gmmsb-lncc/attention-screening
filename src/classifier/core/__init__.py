"""
Core - Módulos principais do sistema modularizado.
Avaliação de modelos e gerenciamento de dados.
"""

# Imports dos módulos modularizados
from .evaluator import ModelEvaluator, DataTypeConverter
from .data_loader import DataManager

__all__ = [
    "ModelEvaluator", 
    "DataTypeConverter",
    "DataManager"
]
