"""
DockTKinase Classifier - Predição de interações kinase-composto
==============================================================

Módulos:
- config: Configurações de experimentos e modelos
- models: Arquiteturas de modelos neurais
- core: Lógica principal de treinamento e validação  
- utils: Utilitários, métricas e validação de dados
- tests: Testes unitários e de integração

Uso:
    from classifier.config import MLPConfig
    from classifier.models import MLPEmbeddingClassifier
    from classifier.core import Trainer, CrossValidator
"""

__version__ = "2.0.0"
__author__ = "DockTKinase Team"

# Imports principais para compatibilidade
from .config.mlp_config import MLPConfig
from .models.mlp import MLPEmbeddingClassifier
from .core.trainer import ModelTrainer
from .core.cross_validator import CrossValidator

__all__ = [
    "MLPConfig",
    "MLPEmbeddingClassifier", 
    "ModelTrainer",
    "CrossValidator"
]
