"""
Implementação modular do MLP para classificação binária.
"""

import torch
import torch.nn as nn
from typing import List
import logging

import sys
sys.path.append('..')

from models.base_model import BaseClassifier
from config.mlp_config import MLPConfig

logger = logging.getLogger(__name__)


class MLPEmbeddingClassifier(BaseClassifier):
    """MLP modular para classificação de embeddings."""
    
    def __init__(self, config: MLPConfig):
        super().__init__(config.input_size)
        self.config = config
        
        # Mapeamento de ativações
        self.activation_map = {
            "ReLU": nn.ReLU(),
            "GELU": nn.GELU(), 
            "LeakyReLU": nn.LeakyReLU(),
            "ELU": nn.ELU(),
            "Tanh": nn.Tanh()
        }
        
        # Construir rede
        self.layers = self._build_network()
        
        # Inicializar pesos
        self.apply(self._init_weights)
    
    def _build_network(self) -> nn.Sequential:
        """Constrói a arquitetura da rede."""
        layers = []
        
        # Camadas ocultas
        layer_sizes = [self.config.input_size] + self.config.hidden_layers
        
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            
            # Linear layer
            layers.append(nn.Linear(in_size, out_size))
            
            # Batch normalization
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(out_size))
            
            # Activation
            if self.config.activation in self.activation_map:
                layers.append(self.activation_map[self.config.activation])
            else:
                layers.append(nn.ReLU())
            
            # Dropout
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Camada de saída
        layers.append(nn.Linear(self.config.hidden_layers[-1], self.config.output_size))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """Inicialização de pesos."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.layers(x)
    
    def count_parameters(self) -> int:
        """Conta número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_architecture_info(self) -> dict:
        """Retorna informações sobre a arquitetura."""
        return {
            "architecture": self.config.get_architecture_summary(),
            "total_parameters": self.count_parameters(),
            "activation": self.config.activation,
            "dropout_rate": self.config.dropout_rate,
            "use_batch_norm": self.config.use_batch_norm,
            "amp_enabled": self.config.amp_enabled
        }
