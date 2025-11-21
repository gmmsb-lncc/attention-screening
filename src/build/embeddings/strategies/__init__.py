"""
Estratégias de modelos de proteína para embeddings.
Implementa Strategy Pattern para suportar múltiplos modelos (ESM-2, ESM-3, etc).
"""

from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy
from src.build.embeddings.strategies.esm2_strategy import ESM2Strategy

__all__ = [
    'BaseProteinStrategy',
    'ESM2Strategy',
]
