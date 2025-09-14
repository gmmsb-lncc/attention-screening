"""
Implementação modular do MLP para classificação binária.
"""

import torch
import torch.nn as nn
from typing import List, Optional
import logging

# Imports relativos com fallbacks para execução direta
try:
    from .base_model import BaseClassifier
    from ..config.mlp_config import MLPConfig
except ImportError:
    # Fallback para execução direta - ajustar sys.path se necessário
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    classifier_dir = os.path.dirname(current_dir)
    if classifier_dir not in sys.path:
        sys.path.insert(0, classifier_dir)
    
    from models.base_model import BaseClassifier
    from config.mlp_config import MLPConfig

logger = logging.getLogger(__name__)


class MLPEmbeddingClassifier(BaseClassifier):
    """MLP modular para classificação de embeddings com auto-detecção de input_size."""
    
    def __init__(self, config: MLPConfig, input_size: Optional[int] = None):
        """
        Args:
            config: Configuração do MLP
            input_size: Tamanho de entrada. Se None, será auto-detectado no primeiro forward.
        """
        # Determinar input_size final
        final_input_size = input_size or config.input_size
        
        # Atualizar config com input_size se fornecido
        if input_size is not None and config.input_size is None:
            config.input_size = input_size
        
        super().__init__(final_input_size or -1)  # -1 indica que será detectado
        self.config = config
        self._input_size_detected = final_input_size is not None
        self._layers_built = False
        
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
        
        # Se input_size não está definido, retorna Sequential vazio
        # A rede será construída no primeiro forward
        if self.config.input_size is None:
            return nn.Sequential()
        
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
    
    def _detect_and_build(self, x: torch.Tensor):
        """Detecta input_size e constrói a rede dinamicamente."""
        if not self._input_size_detected:
            # Detectar tamanho da entrada
            input_size = x.shape[1]
            
            # Atualizar configuração
            self.config.input_size = input_size
            
            # Reconstruir rede
            self.layers = self._build_network()
            
            # Re-inicializar pesos
            self.apply(self._init_weights)
            
            # Marcar como detectado
            self._input_size_detected = True
            
            print(f"[MLPEmbeddingClassifier] Auto-detectado input_size: {input_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Auto-detectar e construir rede se necessário
        if self.config.input_size is None:
            self._detect_and_build(x)
        
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
