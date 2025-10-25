"""
Módulo de Utilitários - DockTKinase
===================================

Funções utilitárias compartilhadas para todo o projeto.
"""

from .data_utils import (
    safe_get,
    safe_get_numeric,
    safe_get_int,
    safe_get_str,
    get_safe  # alias
)

__all__ = [
    'safe_get',
    'safe_get_numeric',
    'safe_get_int',
    'safe_get_str',
    'get_safe',
]
