"""
Interface abstrata para estratégias de modelos de proteína.
Implementa Strategy Pattern para permitir múltiplos modelos (ESM-2, ESM-3, etc.)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
import numpy as np
import torch


class BaseProteinStrategy(ABC):
    """
    Interface abstrata para estratégias de modelos de proteína.
    
    Cada modelo (ESM-2, ESM-3, etc.) implementa esta interface,
    permitindo adicionar novos modelos sem modificar código existente.
    
    Princípios SOLID aplicados:
    - Single Responsibility: Cada strategy cuida apenas de seu modelo
    - Open/Closed: Aberto para extensão (novos modelos), fechado para modificação
    - Liskov Substitution: Strategies são intercambiáveis
    - Interface Segregation: Interface focada e coesa
    - Dependency Inversion: ProteinEmbedding depende da abstração, não implementação
    """
    
    @abstractmethod
    def load(
        self, 
        model_name: str, 
        device: torch.device,
        offload_folder: Optional[str] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Carrega modelo e componentes necessários (alphabet, tokenizer, etc.)
        
        Args:
            model_name: Nome do modelo (ex: "esm2_t48_15B_UR50D")
            device: Dispositivo PyTorch (cuda/cpu/mps)
            offload_folder: Pasta para CPU offloading (opcional)
            **kwargs: Parâmetros específicos do modelo
            
        Returns:
            Tuple contendo (model, auxiliary_objects)
            - auxiliary_objects pode ser alphabet, tokenizer, etc.
            
        Raises:
            ValueError: Se modelo não for suportado
            ModelLoadError: Se falhar ao carregar modelo
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        model: Any,
        auxiliary_objects: Any,
        sequence: str,
        device: torch.device,
        **kwargs
    ) -> np.ndarray:
        """
        Gera embedding para uma sequência de proteína.
        
        Args:
            model: Modelo carregado
            auxiliary_objects: Objetos auxiliares (alphabet, tokenizer)
            sequence: Sequência de aminoácidos
            device: Dispositivo PyTorch
            **kwargs: Parâmetros específicos (layers, pooling, etc.)
            
        Returns:
            Embedding numpy array (shape: [embedding_dim])
            
        Raises:
            EmbeddingError: Se falhar ao gerar embedding
        """
        pass
    
    @abstractmethod
    def get_max_length(self, model_name: str) -> int:
        """
        Retorna comprimento máximo de sequência para o modelo.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Tamanho máximo em tokens/aminoácidos
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self, model_name: str) -> int:
        """
        Retorna dimensão do embedding gerado pelo modelo.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Dimensão do vetor de embedding
        """
        pass
    
    @abstractmethod
    def cleanup(self, model: Any, auxiliary_objects: Any) -> None:
        """
        Libera recursos (memória GPU, tensors, etc.)
        
        Chamado após generate() para garantir limpeza adequada.
        Importante para evitar memory leaks em pipelines longos.
        
        Args:
            model: Modelo a ser limpo
            auxiliary_objects: Objetos auxiliares a serem limpos
        """
        pass
