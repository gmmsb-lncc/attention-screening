"""
DockTKinase Classifier - Kinase-compound interaction prediction (Modularized Version)
=====================================================================================

Modularized Modules:
- models: Modularized MLP architecture
- core: Evaluator and data manager
- utils: Safe imports system
- modular_classifier.py: Main CLI interface
- modular_pipeline.py: Training pipeline

Usage:
    from .modular_pipeline import MLPEmbeddingPipeline
    from .models.mlp_classifier import MLPEmbeddingClassifier
    from .core.evaluator import ModelEvaluator
    from .core.data_loader import DataManager
"""

__version__ = "2.0.0-modular"
__author__ = "DockTKinase Team"

# Main imports from modularized system
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
    # Modular system may not be fully available
    print(f"⚠️  Warning: Modular imports not available: {e}")
    __all__ = []
