"""
Labels module for generating interaction and binary labels.

Provides base classes and implementations for different types of 
label generation in the protein-ligand interaction pipeline.
"""

from build.labels.base_labels import BaseLabels
from build.labels.interaction_labels import InteractionLabels  
from build.labels.binary_labels import BinaryLabels

__all__ = [
    'BaseLabels',
    'InteractionLabels',
    'BinaryLabels'
]
