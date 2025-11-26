"""
Configuração centralizada para o MLP Classifier.

Este módulo fornece a classe MLPConfig e funções de factory para
criar configurações padrão para o classificador MLP.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class MLPConfig:
    """Configuração centralizada para o MLP."""
    # Arquitetura
    hidden_dim: int = 1024
    dropout: float = 0.3
    activation: str = "relu"
    use_batch_norm: bool = True
    n_layers: int = 3
    
    # Treinamento
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    early_stopping_patience: int = 5
    early_metric: str = "loss"  # "loss" ou "auc"
    
    # Otimização
    dtype: str = "float32"  # "float32", "float16", "bfloat16"
    amp: bool = False
    compile_model: bool = False
    num_workers: int = 0
    
    # Validação
    cv_folds: int = 5
    test_size: float = 0.2
    
    # Arquivos
    model_path: str = "mlp_model.pth"
    metrics_path: str = "metrics.json"
    
    # Modo
    verbose: bool = True
    
    def to_dict(self):
        """Converter config para dicionário."""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict):
        """Criar config a partir de dicionário."""
        return cls(**config_dict)


def create_default_config() -> MLPConfig:
    """
    Cria configuração padrão (compatível com classifier.py).
    
    Returns:
        MLPConfig com valores padrão
    """
    return MLPConfig()


def create_light_config() -> MLPConfig:
    """Configuração leve para testes rápidos."""
    return MLPConfig(
        hidden_dim=256,
        dropout=0.2,
        epochs=20,
        batch_size=32
    )


def create_heavy_config() -> MLPConfig:
    """Configuração pesada para máximo desempenho."""
    return MLPConfig(
        hidden_dim=2048,
        dropout=0.4,
        epochs=100,
        batch_size=128,
        amp=True,
        compile_model=True
    )
