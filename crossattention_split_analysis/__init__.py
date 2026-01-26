"""
CrossAttention Split Analysis Package.

Evaluates CNN + Cross-Attention models for protein-ligand affinity prediction
using three evaluation scenarios with scientific rigor.
"""

from .config import TrainingConfig, SUPPORTED_EMBEDDINGS, PROTEIN_DIMS
from .experiment import run_crossattention_analysis

__all__ = [
    'TrainingConfig',
    'SUPPORTED_EMBEDDINGS',
    'PROTEIN_DIMS',
    'run_crossattention_analysis'
]
