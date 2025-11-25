"""
Estratégias de modelos de proteína para embeddings.
Implementa Strategy Pattern para suportar múltiplos modelos (ESM-2, ESM-C, etc).
"""

from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy
from src.build.embeddings.strategies.esm2_strategy import ESM2Strategy
from src.build.embeddings.strategies.esmc_strategy import ESMCStrategy
from src.build.embeddings.strategies.esmc_forge_strategy import ESMCForgeStrategy

__all__ = [
    'BaseProteinStrategy',
    'ESM2Strategy',
    'ESMCStrategy',
    'ESMCForgeStrategy',
]
