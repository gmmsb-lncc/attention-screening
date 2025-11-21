"""
Factories para criação de estratégias de modelos.
Implementa Factory Pattern para desacoplar criação de objetos.
"""

from src.build.embeddings.factories.protein_model_factory import ProteinModelFactory

__all__ = [
    'ProteinModelFactory',
]
