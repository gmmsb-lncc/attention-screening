"""
Configurações para o MLP do DockTKinase Classifier.
"""

from dataclasses import dataclass, field
from typing import Literal, List


@dataclass
class MLPConfig:
    """Configuração centralizada e validada para o MLP."""
    
    # === Arquitetura do Modelo ===
    input_size: int = 1024
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256])
    output_size: int = 1
    activation: Literal["ReLU", "GELU", "LeakyReLU", "ELU", "Tanh"] = "ReLU"
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    
    # === Treinamento ===
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # === Performance ===
    amp_enabled: bool = False
    
    def __post_init__(self):
        """Validações após inicialização."""
        self._validate_architecture()
        self._validate_training()
    
    def _validate_architecture(self):
        """Valida parâmetros da arquitetura."""
        if self.input_size <= 0:
            raise ValueError(f"input_size deve ser positivo, recebido: {self.input_size}")
        
        if self.output_size <= 0:
            raise ValueError(f"output_size deve ser positivo, recebido: {self.output_size}")
        
        if not self.hidden_layers:
            raise ValueError("hidden_layers não pode estar vazio")
        
        for layer_size in self.hidden_layers:
            if layer_size <= 0:
                raise ValueError(f"Todas as camadas devem ter tamanho positivo, encontrado: {layer_size}")
        
        if not 0.0 <= self.dropout_rate <= 0.9:
            raise ValueError(f"dropout_rate deve estar em [0.0, 0.9], recebido: {self.dropout_rate}")
    
    def _validate_training(self):
        """Valida parâmetros de treinamento."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate deve ser positivo, recebido: {self.learning_rate}")
        
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay deve ser não-negativo, recebido: {self.weight_decay}")
    
    def get_architecture_summary(self) -> str:
        """Retorna resumo da arquitetura."""
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        return " -> ".join(str(size) for size in layers)


def create_default_config(input_size: int = 1024) -> MLPConfig:
    """Cria configuração padrão otimizada."""
    return MLPConfig(
        input_size=input_size,
        hidden_layers=[512, 256, 128],
        output_size=1,
        activation="ReLU",
        dropout_rate=0.3,
        learning_rate=0.001,
        weight_decay=1e-4,
        use_batch_norm=True,
        amp_enabled=False
    )
