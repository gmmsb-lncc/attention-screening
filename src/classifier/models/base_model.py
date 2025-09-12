"""
Interface base para modelos do DockTKinase Classifier.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any


class BaseClassifier(ABC, nn.Module):
    """Interface base para todos os modelos classificadores."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass retornando logits."""
        pass
    
    @abstractmethod
    def get_architecture_info(self) -> Dict[str, Any]:
        """Retorna informações sobre a arquitetura."""
        pass
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predição de probabilidades (aplica sigmoid)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predição de classes (0/1)."""
        probas = self.predict_proba(x)
        return (probas >= threshold).float()
    
    def count_parameters(self) -> int:
        """Conta número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Retorna tamanho do modelo em MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024**2)
