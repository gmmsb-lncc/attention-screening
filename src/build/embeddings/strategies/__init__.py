"""
Estratégias de modelos de proteína para embeddings.
Implementa Strategy Pattern para suportar múltiplos modelos (ESM-2, ESM-C, Boltz-2, OpenFold-3, etc).
"""

from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy
from src.build.embeddings.strategies.esm2_strategy import ESM2Strategy
from src.build.embeddings.strategies.esmc_strategy import ESMCStrategy
from src.build.embeddings.strategies.esmc_forge_strategy import ESMCForgeStrategy
from src.build.embeddings.strategies.boltz_strategy import BoltzStrategy
from src.build.embeddings.strategies.openfold_strategy import OpenFoldStrategy

__all__ = [
    'BaseProteinStrategy',
    'ESM2Strategy',
    'ESMCStrategy',
    'ESMCForgeStrategy',
    'BoltzStrategy',
    'OpenFoldStrategy',
]
