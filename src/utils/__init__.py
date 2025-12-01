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

from .json_serializer import (
    make_json_serializable,
    save_json,
    load_json
)

from .checkpoint_manager import CheckpointManager

__all__ = [
    # Data utils
    'safe_get',
    'safe_get_numeric',
    'safe_get_int',
    'safe_get_str',
    'get_safe',
    # JSON serialization
    'make_json_serializable',
    'save_json',
    'load_json',
    # Checkpoint management
    'CheckpointManager',
]
