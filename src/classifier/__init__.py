"""
DockTKinase Classifier - Predição de interações kinase-composto (Versão Modularizada)
=====================================================================================

Módulos Modularizados:
- models: Arquitetura MLP modularizada
- core: Avaliador e gerenciador de dados
- utils: Sistema de imports seguros
- modular_classifier.py: Interface CLI principal
- modular_pipeline.py: Pipeline de treinamento

Uso:
    from .modular_pipeline import MLPEmbeddingPipeline
    from .models.mlp_classifier import MLPEmbeddingClassifier
    from .core.evaluator import ModelEvaluator
    from .core.data_loader import DataManager
"""

__version__ = "2.0.0-modular"
__author__ = "DockTKinase Team"

# Imports principais do sistema modularizado
try:
    from .modular_pipeline import MLPEmbeddingPipeline
    from .models.mlp_classifier import MLPEmbeddingClassifier
    from .core.evaluator import ModelEvaluator
    from .core.data_loader import DataManager
    from .utils.import_utils import safe_import_optional
    
    __all__ = [
        "MLPEmbeddingPipeline",
        "MLPEmbeddingClassifier", 
        "ModelEvaluator",
        "DataManager",
        "safe_import_optional"
    ]
    
except ImportError as e:
    # Sistema modular pode não estar completamente disponível
    print(f"⚠️  Aviso: Imports modularizados não disponíveis: {e}")
    __all__ = []
