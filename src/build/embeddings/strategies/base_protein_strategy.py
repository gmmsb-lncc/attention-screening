"""
Abstract interface for protein model strategies.
Implements Strategy Pattern to support multiple models (ESM-2, ESM-3, OpenFold, etc.)

This module defines the core contract that ALL protein language model implementations
must follow. It enables seamless integration of new models without modifying existing code.

INTEGRATION GUIDE FOR NEW MODELS:
=================================

To add ESM-3, OpenFold, or any new protein model:

1. Create new strategy class inheriting from BaseProteinStrategy
2. Implement all 5 abstract methods (load, generate, get_max_length, get_embedding_dim, cleanup)
3. Register in ProteinModelFactory
4. Update constants.py with model specifications
5. Add tests in tests/test_<model>_strategy.py

Example: Adding ESM-3
---------------------

    from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy
    
    class ESM3Strategy(BaseProteinStrategy):
        def load(self, model_name, device, offload_folder=None, **kwargs):
            # Load ESM-3 model
            import esm3
            model, tokenizer = esm3.pretrained.load_model(model_name)
            return model.to(device).eval(), tokenizer
        
        def generate(self, model, auxiliary_objects, sequence, device, **kwargs):
            # Generate ESM-3 embedding
            tokenizer = auxiliary_objects
            tokens = tokenizer.encode(sequence).to(device)
            with torch.no_grad():
                outputs = model(tokens)
                embedding = outputs.sequence_embeddings.mean(dim=1).squeeze()
            return embedding.cpu().numpy()
        
        def get_max_length(self, model_name):
            return 4096  # ESM-3 supports longer sequences
        
        def get_embedding_dim(self, model_name):
            return 2560  # ESM-3 embedding dimension
        
        def cleanup(self, model, auxiliary_objects):
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

Then register in ProteinModelFactory:

    ESM3_MODELS = {'esm3_sm_open_v1', 'esm3_medium', 'esm3_large'}
    
    @staticmethod
    def create_strategy(model_name: str) -> BaseProteinStrategy:
        if model_name in ESM3_MODELS:
            from .esm3_strategy import ESM3Strategy
            return ESM3Strategy()
        # ... rest of code

For complete integration guide, see: docs/04-modules/PROTEIN_EMBEDDING_API.md

"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
import numpy as np
import torch


class BaseProteinStrategy(ABC):
    """
    Abstract base class for protein model strategies.
    
    This interface defines the contract that all protein language model implementations
    must follow, enabling support for ESM-2, ESM-3, OpenFold, and future models through
    a unified API.
    
    DESIGN PRINCIPLES (SOLID):
    ---------------------------
    - Single Responsibility: Each strategy handles only ONE model family
    - Open/Closed: Open for extension (new models), closed for modification
    - Liskov Substitution: All strategies are interchangeable
    - Interface Segregation: Minimal interface (5 methods only)
    - Dependency Inversion: High-level code depends on abstraction, not concrete implementations
    
    METHOD CONTRACT:
    ----------------
    All implementations MUST provide these 5 methods:
    
    1. load() - Load model and auxiliary components (tokenizer, alphabet)
    2. generate() - Generate embedding for a protein sequence
    3. get_max_length() - Return maximum sequence length supported
    4. get_embedding_dim() - Return embedding vector dimension
    5. cleanup() - Free resources (GPU memory, tensors)
    
    MEMORY MANAGEMENT:
    ------------------
    Implementations MUST properly manage memory to avoid leaks:
    - Use torch.no_grad() during inference
    - Delete large tensors explicitly after use
    - Call gc.collect() in cleanup
    - Call torch.cuda.empty_cache() if using CUDA
    - Consider torch.cuda.synchronize() for complete cleanup
    
    DEVICE SUPPORT:
    ---------------
    Implementations MUST support all PyTorch devices:
    - CUDA (NVIDIA GPUs)
    - MPS (Apple Silicon GPUs)
    - CPU (fallback)
    
    CPU OFFLOADING:
    ---------------
    For large models (>3B parameters), implement CPU offloading using accelerate:
    
        from accelerate import dispatch_model, infer_auto_device_map
        device_map = infer_auto_device_map(model, max_memory={0: "20GiB", "cpu": "30GiB"})
        model = dispatch_model(model, device_map=device_map, offload_folder=offload_folder)
    
    ERROR HANDLING:
    ---------------
    Use specific exceptions from src.build.core.exceptions:
    - ModelLoadError: For model loading failures
    - EmbeddingError: For embedding generation failures
    - Provide actionable error messages with solutions
    
    TESTING:
    --------
    Each strategy implementation should have:
    - Unit tests for each method
    - Integration tests with ProteinEmbedding
    - End-to-end test with real sequence
    - Memory leak tests (verify cleanup works)
    
    Example implementations:
    - ESM2Strategy: src/build/embeddings/strategies/esm2_strategy.py
    - Test suite: tests/test_solid_refactoring.py
    
    For complete API documentation and integration examples, see:
    docs/04-modules/PROTEIN_EMBEDDING_API.md
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
