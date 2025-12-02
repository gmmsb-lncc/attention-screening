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
        # NOTE: offload_folder renamed to offload_dir in accelerate >= 1.0
        model = dispatch_model(model, device_map=device_map, offload_dir=offload_folder)
    
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
        Load model and required components (alphabet, tokenizer, etc.)
        
        Args:
            model_name: Model name (e.g., "esm2_t48_15B_UR50D")
            device: PyTorch device (cuda/cpu/mps)
            offload_folder: Folder for CPU offloading (optional)
            **kwargs: Model-specific parameters
            
        Returns:
            Tuple containing (model, auxiliary_objects)
            - auxiliary_objects can be alphabet, tokenizer, etc.
            
        Raises:
            ValueError: If model is not supported
            ModelLoadError: If failed to load model
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
        Generate embedding for a protein sequence.
        
        Args:
            model: Loaded model
            auxiliary_objects: Auxiliary objects (alphabet, tokenizer)
            sequence: Amino acid sequence
            device: PyTorch device
            **kwargs: Specific parameters (layers, pooling, etc.)
            
        Returns:
            Embedding numpy array (shape: [embedding_dim])
            
        Raises:
            EmbeddingError: If failed to generate embedding
        """
        pass
    
    @abstractmethod
    def get_max_length(self, model_name: str) -> int:
        """
        Return maximum sequence length for the model.
        
        Args:
            model_name: Model name
            
        Returns:
            Maximum size in tokens/amino acids
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self, model_name: str) -> int:
        """
        Return embedding dimension generated by the model.
        
        Args:
            model_name: Model name
            
        Returns:
            Embedding vector dimension
        """
        pass
    
    @abstractmethod
    def cleanup(self, model: Any, auxiliary_objects: Any) -> None:
        """
        Free resources (GPU memory, tensors, etc.)
        
        Called after generate() to ensure proper cleanup.
        Important to avoid memory leaks in long pipelines.
        
        Args:
            model: Model to be cleaned
            auxiliary_objects: Auxiliary objects to be cleaned
        """
        pass
    
    # =========================================================================
    # OPTIONAL METHODS (with default implementations)
    # =========================================================================
    
    def generate_matrix(
        self,
        model: Any,
        auxiliary_objects: Any,
        sequence: str,
        device: torch.device,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Generate per-token embedding matrix (no pooling).
        
        This method is OPTIONAL - strategies that don't support it can return None.
        Implementations should return the complete matrix [seq_len, embed_dim]
        instead of the pooled vector [embed_dim].
        
        Args:
            model: Loaded model
            auxiliary_objects: Auxiliary objects (alphabet, tokenizer)
            sequence: Amino acid sequence
            device: PyTorch device
            **kwargs: Specific parameters (layers, etc.)
            
        Returns:
            Numpy array matrix (shape: [seq_len, embedding_dim]) or None if not supported
            
        Note:
            - Returns None by default (backward compatible)
            - Subclasses should override to support matrices
            - Matrix does NOT include special tokens (BOS/EOS)
        """
        return None
    
    def supports_matrix_output(self) -> bool:
        """
        Indicate if strategy supports matrix generation.
        
        Returns:
            True if generate_matrix() is implemented, False otherwise
        """
        return False
