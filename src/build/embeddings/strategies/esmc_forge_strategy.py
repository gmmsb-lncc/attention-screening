"""
ESM-C Forge API strategy implementation for ESM-C 6B model.

This strategy uses the EvolutionaryScale Forge API to access the ESM-C 6B model,
which is not available locally and requires API authentication.

Models (via Forge API):
- esmc-6b-2024-12: 6B params, 3072-dim (requires ESM_API_KEY)

Key Features:
- Access to ESM-C 6B (largest ESM-C model)
- Cloud-based inference (no local GPU required)
- Requires API key from EvolutionaryScale
"""

import os
import gc
from pathlib import Path
from typing import Tuple, Any, Optional
import numpy as np
import torch

from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy
from src.build.core.exceptions import ModelLoadError, EmbeddingError


class ESMCForgeStrategy(BaseProteinStrategy):
    """
    Strategy implementation for ESM-C 6B model via Forge API.
    
    This strategy connects to EvolutionaryScale's Forge API to use the
    ESM-C 6B model, which provides the highest quality embeddings but
    requires an API key.
    
    Supported models:
    - esmc-6b-2024-12 (6B parameters, 3072-dim)
    
    Requirements:
    - ESM_API_KEY environment variable or passed via kwargs
    """
    
    # Constants
    DEFAULT_POOLING = 'mean'
    VALID_POOLING_STRATEGIES = {'mean'}
    VALID_AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    FORGE_URL = "https://forge.evolutionaryscale.ai"
    
    # Model specifications
    MODEL_SPECS = {
        'esmc-6b-2024-12': {
            'dim': 3072, 
            'layers': 56, 
            'max_len': 2048,
            'forge_model_name': 'esmc-6b-2024-12',
        },
    }
    
    def __init__(self, logger=None):
        """Initialize ESM-C Forge strategy.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger
        self.client = None
        self.model_name = None
        self.api_key = None
        self._esm3_path = None
    
    def load(
        self, 
        model_name: str, 
        device: torch.device,
        offload_folder: Optional[str] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Initialize Forge API client for ESM-C 6B.
        
        Args:
            model_name: Model identifier ('esmc-6b-2024-12')
            device: PyTorch device (not used for API, but kept for interface)
            offload_folder: Not used for API
            **kwargs: Additional parameters (api_key, logger)
        
        Returns:
            Tuple (client, None) - client is the Forge API client
        """
        self.logger = kwargs.get('logger')
        self._validate_model(model_name)
        
        # Get API key
        self.api_key = kwargs.get('api_key') or os.environ.get('ESM_API_KEY')
        if not self.api_key:
            raise ModelLoadError(
                "ESM-C 6B requires an API key from EvolutionaryScale.\n"
                "Set ESM_API_KEY environment variable or pass api_key parameter.\n"
                "Get your key at: https://forge.evolutionaryscale.ai/"
            )
        
        if self.logger:
            self.logger.info(f"Loading ESM-C 6B via Forge API...")
        
        # Setup ESM-3 path for imports
        self._setup_esm3_path()
        
        # Import and create Forge client
        client = self._create_forge_client(model_name)
        
        self.client = client
        self.model_name = model_name
        self.device = device
        
        if self.logger:
            self.logger.info("✅ ESM-C 6B Forge client initialized")
            self.logger.info(f"   Model: {model_name}")
            self.logger.info(f"   Dimension: {self.MODEL_SPECS[model_name]['dim']}")
            self.logger.info(f"   API: {self.FORGE_URL}")
        
        return client, None
    
    def _validate_model(self, model_name: str) -> None:
        """Validate that model_name is supported."""
        if model_name not in self.MODEL_SPECS:
            raise ValueError(
                f"ESM-C Forge model '{model_name}' not supported.\n"
                f"Available models: {list(self.MODEL_SPECS.keys())}"
            )
    
    def _setup_esm3_path(self) -> None:
        """Setup ESM-3 path for imports."""
        import sys
        
        # ESM-3 source path
        self._esm3_path = Path(__file__).parent.parent.parent.parent.parent / "llm" / "ESM" / "esm-3" / "esm-main"
        
        if not self._esm3_path.exists():
            raise ModelLoadError(
                f"ESM-3 not found at: {self._esm3_path}\n"
                "Install ESM-3 first: git clone https://github.com/evolutionaryscale/esm.git"
            )
        
        # Clear existing esm modules
        esm_modules = [k for k in list(sys.modules.keys()) if k.startswith('esm')]
        for mod in esm_modules:
            del sys.modules[mod]
        
        # Add ESM-3 to path
        esm3_str = str(self._esm3_path)
        if esm3_str in sys.path:
            sys.path.remove(esm3_str)
        sys.path.insert(0, esm3_str)
    
    def _create_forge_client(self, model_name: str):
        """Create ESM-C Forge API client.
        
        Args:
            model_name: Model name for Forge API
            
        Returns:
            ESMCForgeInferenceClient instance
        """
        try:
            from esm.sdk.forge import ESMCForgeInferenceClient
            
            forge_model_name = self.MODEL_SPECS[model_name]['forge_model_name']
            
            client = ESMCForgeInferenceClient(
                model=forge_model_name,
                url=self.FORGE_URL,
                token=self.api_key,
                request_timeout=300,  # 5 minutes timeout
                max_retry_attempts=3
            )
            
            return client
            
        except ImportError as e:
            raise ModelLoadError(
                f"Failed to import Forge client. Make sure ESM-3 is installed.\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to create Forge client: {e}")
    
    def generate(
        self,
        model: Any,  # This is actually the Forge client
        auxiliary_objects: Any,
        sequence: str,
        device: torch.device,
        **kwargs
    ) -> np.ndarray:
        """
        Generate ESM-C 6B embedding via Forge API.
        
        Args:
            model: Forge API client
            auxiliary_objects: Not used (None)
            sequence: Amino acid sequence
            device: PyTorch device (used only for final tensor)
            **kwargs: Additional parameters (logger)
        
        Returns:
            Embedding numpy array (shape: [3072])
        """
        client = model
        self.logger = kwargs.get('logger', self.logger)
        
        # Validate and clean sequence
        clean_sequence = self._clean_sequence(sequence)
        
        # Truncate if needed
        max_len = self.MODEL_SPECS.get(self.model_name, {}).get('max_len', 2048)
        if len(clean_sequence) > max_len:
            if self.logger:
                self.logger.warning(
                    f"Sequence truncated: {len(clean_sequence)} → {max_len} aa"
                )
            clean_sequence = clean_sequence[:max_len]
        
        try:
            # Import API types
            from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig
            
            # Create protein object
            protein = ESMProtein(sequence=clean_sequence)
            
            # Encode to get tokens
            if self.logger:
                self.logger.debug(f"Encoding sequence ({len(clean_sequence)} aa) via Forge API...")
            
            protein_tensor = client.encode(protein)
            
            if isinstance(protein_tensor, ESMProteinError):
                raise EmbeddingError(
                    f"Forge API encode error: {protein_tensor.error_code} - {protein_tensor.error_msg}"
                )
            
            # Get embeddings using logits endpoint with return_mean_embedding=True
            config = LogitsConfig(
                return_mean_embedding=True,
                sequence=False  # Don't need sequence logits
            )
            
            if self.logger:
                self.logger.debug("Fetching embeddings from Forge API...")
            
            output = client.logits(protein_tensor, config)
            
            if isinstance(output, ESMProteinError):
                raise EmbeddingError(
                    f"Forge API logits error: {output.error_code} - {output.error_msg}"
                )
            
            # Extract mean embedding
            if output.mean_embedding is None:
                raise EmbeddingError("Forge API returned no embedding")
            
            # Convert to numpy - handle bfloat16 conversion
            if isinstance(output.mean_embedding, torch.Tensor):
                # Convert bfloat16 to float32 before numpy conversion
                embedding_tensor = output.mean_embedding
                if embedding_tensor.dtype == torch.bfloat16:
                    embedding_tensor = embedding_tensor.to(torch.float32)
                result = embedding_tensor.cpu().numpy()
            else:
                result = np.array(output.mean_embedding, dtype=np.float32)
            
            # Ensure correct shape
            if result.ndim > 1:
                result = result.squeeze()
            
            if self.logger:
                self.logger.debug(f"✅ Embedding generated: shape={result.shape}")
            
            return result
            
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"ESM-C 6B Forge API embedding failed: {e}")
    
    def generate_matrix(
        self,
        model: Any,
        auxiliary_objects: Any,
        sequence: str,
        device: torch.device,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Generate ESM-C 6B embedding matrix via Forge API.
        
        Note: The Forge API currently only provides mean-pooled embeddings,
        not per-token embeddings. This method returns None until the API
        supports per-token output.
        
        Args:
            model: Forge API client
            auxiliary_objects: Not used (None)
            sequence: Amino acid sequence
            device: PyTorch device
            **kwargs: Additional parameters (logger)
        
        Returns:
            None (per-token embeddings not available via Forge API)
        """
        self.logger = kwargs.get('logger', self.logger)
        
        if self.logger:
            self.logger.warning(
                "ESM-C 6B Forge API does not support per-token embeddings. "
                "Only mean-pooled embeddings are available. "
                "For per-token embeddings, use local ESM-C (300M or 600M) models."
            )
        
        return None
    
    def get_max_length(self, model_name: str) -> int:
        """Return max sequence length for model."""
        return self.MODEL_SPECS.get(model_name, {}).get('max_len', 2048)
    
    def get_embedding_dim(self, model_name: str) -> int:
        """Return embedding dimension for model."""
        return self.MODEL_SPECS.get(model_name, {}).get('dim', 3072)
    
    def cleanup(self, model: Any, auxiliary_objects: Any) -> None:
        """Clean up resources."""
        import sys
        
        # Clear ESM-3 modules from cache
        esm_modules = [k for k in list(sys.modules.keys()) if k.startswith('esm')]
        for mod in esm_modules:
            del sys.modules[mod]
        
        gc.collect()
    
    def _clean_sequence(self, sequence: str) -> str:
        """Clean and validate amino acid sequence."""
        sequence = sequence.strip()
        valid_aa = self.VALID_AMINO_ACIDS
        clean = ''.join(c for c in sequence.upper() if c in valid_aa)
        
        if not clean:
            raise EmbeddingError(
                f"Sequence contains no valid amino acids. Valid codes: {valid_aa}"
            )
        
        if len(clean) != len(sequence) and self.logger:
            removed = len(sequence) - len(clean)
            self.logger.warning(f"Removed {removed} invalid characters from sequence")
        
        return clean
