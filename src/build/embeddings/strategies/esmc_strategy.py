"""
ESM-C strategy implementation (EvolutionaryScale Cambrian).

ESM-C is a protein representation learning model optimized for embeddings.
It provides a simpler, more efficient alternative to ESM-3 for representation tasks.

Models:
- esmc-300m-2024-12: 300M params, 960-dim, 30 layers
- esmc-600m-2024-12: 600M params, 1152-dim, 36 layers
- esmc-6b-2024-12: 6B params, 3072-dim, 56 layers (requires GPU >= 48GB)

Key Features:
- Fast inference (optimized for embeddings)
- Flash Attention support (when available)
- Compatible with ESM-2 use cases
- Longer sequences (up to 2048 tokens)
"""

import os
import gc
from pathlib import Path
from typing import Tuple, Any, Optional
import numpy as np
import torch

from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy
from src.build.core.exceptions import ModelLoadError, EmbeddingError


class ESMCStrategy(BaseProteinStrategy):
    """
    Strategy implementation for ESM-C models (Cambrian).
    
    ESM-C is designed for fast, high-quality protein embeddings.
    Similar to ESM-2 but with improved architecture and performance.
    
    Supported models:
    - esmc-300m-2024-12 (300M parameters, 960-dim)
    - esmc-600m-2024-12 (600M parameters, 1152-dim)
    - esmc-6b-2024-12 (6B parameters, 3072-dim, requires GPU >= 48GB)
    """
    
    # Constants - Extract magic strings to avoid repetition
    DEFAULT_POOLING = 'mean'
    VALID_POOLING_STRATEGIES = {'mean', 'cls'}
    VALID_AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Model specifications
    MODEL_SPECS = {
        'esmc-300m-2024-12': {
            'dim': 960, 
            'layers': 30, 
            'max_len': 2048,
            'registry_name': 'esmc_300m',
        },
        'esmc-600m-2024-12': {
            'dim': 1152, 
            'layers': 36, 
            'max_len': 2048,
            'registry_name': 'esmc_600m',
        },
        'esmc-6b-2024-12': {
            'dim': 3072,
            'layers': 56,
            'max_len': 2048,
            'registry_name': 'esmc_6b',
        },
    }
    
    def __init__(self, logger=None):
        """Initialize ESM-C strategy.
        
        Args:
            logger: Optional logger instance (Dependency Injection)
        """
        self.logger = logger
        self._cache_dir = None
        self._esm3_path = None
        # Initialize state attributes
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_name = None
    
    def load(
        self, 
        model_name: str, 
        device: torch.device,
        offload_folder: Optional[str] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load ESM-C model and tokenizer.
        
        Args:
            model_name: Model identifier ('esmc-300m-2024-12' or 'esmc-600m-2024-12')
            device: PyTorch device (cuda/cpu/mps)
            offload_folder: CPU offloading directory (optional, not used for ESM-C)
            **kwargs: Additional parameters (logger, etc.)
        
        Returns:
            Tuple (model, tokenizer)
        """
        self.logger = kwargs.get('logger')
        self._validate_model(model_name)
        self._setup_cache_and_paths()
        
        if self.logger:
            self.logger.info(f"Loading ESM-C model: {model_name}")
        
        # Import ESM-C with namespace resolution
        ESMC = self._import_esmc()
        
        # Load model from registry
        model, tokenizer = self._load_model_from_registry(
            ESMC, model_name, device
        )
        
        return model, tokenizer
    
    def _validate_model(self, model_name: str) -> None:
        """Validate that model_name is supported."""
        if model_name not in self.MODEL_SPECS:
            raise ValueError(
                f"ESM-C model '{model_name}' not supported.\n"
                f"Available models: {list(self.MODEL_SPECS.keys())}"
            )
    
    def _import_esmc(self):
        """Import ESMC class with namespace conflict resolution.
        
        Resolves conflict between fair-esm (site-packages) and ESM-3 (local).
        Both use 'esm' namespace, so we prioritize ESM-3 in sys.path.
        
        Returns:
            ESMC class from ESM-3
        """
        try:
            import sys
            
            # Save original sys.path to restore later
            self._original_sys_path = sys.path.copy()
            
            # Clear all esm modules from cache
            esm_modules = [
                key for key in list(sys.modules.keys()) 
                if key.startswith('esm')
            ]
            for mod_key in esm_modules:
                del sys.modules[mod_key]
            
            if self.logger:
                self.logger.debug(
                    f"Cleared {len(esm_modules)} esm modules from sys.modules"
                )
            
            # Prioritize ESM-3: insert at beginning of sys.path
            esm3_path_str = str(self._esm3_path)
            if esm3_path_str in sys.path:
                sys.path.remove(esm3_path_str)
            sys.path.insert(0, esm3_path_str)
            
            if self.logger:
                self.logger.debug(f"Prioritized ESM-3 in sys.path: {esm3_path_str}")
            
            # Import ESMC from ESM-3 (now first in path)
            from esm.models.esmc import ESMC
            
            if self.logger:
                self.logger.debug("✅ ESMC imported successfully from ESM-3")
            
            return ESMC
            
        except ImportError as e:
            raise ModelLoadError(
                f"ESM-C not available. Make sure ESM-3 is installed.\n"
                f"Expected path: {self._esm3_path}\n"
                f"Install with: cd {self._esm3_path} && pip install -e .\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to import ESM-C: {e}")
    
    def _load_model_from_registry(
        self, 
        ESMC, 
        model_name: str, 
        device: torch.device
    ) -> Tuple[Any, Any]:
        """Load model from ESM-3 registry.
        
        Args:
            ESMC: ESMC class from ESM-3
            model_name: User-friendly model name
            device: Target device
            
        Returns:
            Tuple (model, tokenizer)
        """
        try:
            # Map user-friendly name to registry name
            registry_name = self.MODEL_SPECS[model_name].get(
                'registry_name', model_name
            )
            
            if self.logger:
                self.logger.debug(f"Loading with registry name: {registry_name}")
            
            # Check Flash Attention availability
            use_flash_attn = self._check_flash_attention()
            
            # Load model (downloads if not cached)
            model = ESMC.from_pretrained(
                model_name=registry_name,
                device=device
            )
            model = model.eval()
            
            # Get tokenizer from model
            tokenizer = model.tokenizer
            
            # Store as instance attributes
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.model_name = model_name
            
            # Log success
            if self.logger:
                self.logger.info("✅ ESM-C model loaded successfully")
                if use_flash_attn:
                    self.logger.info("   Flash Attention: Enabled")
                self.logger.info(f"   Model: {model_name}")
                self.logger.info(
                    f"   Dimension: {self.MODEL_SPECS[model_name]['dim']}"
                )
                self.logger.info(f"   Cache: {self._cache_dir}")
            
            return model, tokenizer
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load ESM-C model '{model_name}': {e}"
            )
    
    def generate(
        self,
        model: Any,
        auxiliary_objects: Any,
        sequence: str,
        device: torch.device,
        **kwargs
    ) -> np.ndarray:
        """
        Generate ESM-C embedding using mean pooling.
        
        Args:
            model: Loaded ESM-C model
            auxiliary_objects: Tokenizer
            sequence: Amino acid sequence
            device: PyTorch device
            **kwargs: Additional parameters (logger, pooling_strategy)
        
        Returns:
            Embedding numpy array (shape: [embedding_dim])
        """
        tokenizer = auxiliary_objects
        self.logger = kwargs.get('logger', self.logger)
        pooling_strategy = kwargs.get('pooling_strategy', self.DEFAULT_POOLING)
        
        # Validate pooling strategy
        if pooling_strategy not in self.VALID_POOLING_STRATEGIES:
            raise ValueError(
                f"Invalid pooling strategy '{pooling_strategy}'. "
                f"Valid: {self.VALID_POOLING_STRATEGIES}"
            )
        
        # Validate and clean sequence
        clean_sequence = self._clean_sequence(sequence)
        
        # Truncate if needed
        max_len = self.get_max_length(model.__class__.__name__)
        if len(clean_sequence) > max_len:
            if self.logger:
                self.logger.warning(
                    f"Sequence truncated: {len(clean_sequence)} → {max_len} aa"
                )
            clean_sequence = clean_sequence[:max_len]
        
        try:
            # Tokenize using model's built-in method
            # This handles padding and special tokens automatically
            tokens = model._tokenize([clean_sequence])
            
            # Generate embedding
            with torch.no_grad():
                output = model.forward(sequence_tokens=tokens)
                
                # Extract embeddings
                # ESM-C returns embeddings in output.embeddings
                embeddings = output.embeddings  # Shape: [batch, length, dim]
                
                # Apply pooling strategy
                if pooling_strategy == 'cls':
                    # Use CLS token (first token)
                    sequence_embedding = embeddings[:, 0, :]
                else:
                    # Mean pooling (default)
                    # Exclude padding tokens from mean
                    pad_token_id = tokenizer.pad_token_id
                    mask = (tokens != pad_token_id).unsqueeze(-1).float()
                    masked_embeddings = embeddings * mask
                    sequence_embedding = masked_embeddings.sum(dim=1) / mask.sum(dim=1)
                
                # Convert to numpy
                result = sequence_embedding.squeeze().cpu().numpy()
            
            # Critical cleanup
            del tokens, output, embeddings, sequence_embedding
            if pooling_strategy == 'mean':
                del mask, masked_embeddings
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return result
            
        except Exception as e:
            # Cleanup on error
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            raise EmbeddingError(f"ESM-C embedding generation failed: {e}")
    
    def get_max_length(self, model_name: str) -> int:
        """Return max sequence length for model."""
        return self.MODEL_SPECS.get(model_name, {}).get('max_len', 2048)
    
    def get_embedding_dim(self, model_name: str) -> int:
        """Return embedding dimension for model."""
        return self.MODEL_SPECS.get(model_name, {}).get('dim', 960)
    
    def cleanup(self, model: Any, auxiliary_objects: Any) -> None:
        """Clean up resources and restore sys.path for ESM-2 compatibility."""
        import sys
        
        # Clear ESM-3 modules from cache to avoid conflicts
        esm_modules = [
            key for key in list(sys.modules.keys()) 
            if key.startswith('esm')
        ]
        for mod_key in esm_modules:
            del sys.modules[mod_key]
        
        # Restore original sys.path if saved
        if hasattr(self, '_original_sys_path'):
            sys.path = self._original_sys_path.copy()
            if self.logger:
                self.logger.debug("Restored original sys.path for ESM-2 compatibility")
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ===== Helper Methods =====
    
    def _setup_cache_and_paths(self) -> None:
        """Setup cache directories and ESM-3 path."""
        # Cache directory for ESM-C models
        self._cache_dir = Path(__file__).parent.parent.parent.parent.parent / "llm" / "models_cache" / "ESM3"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variable for ESM data root
        os.environ['ESM_DATA_ROOT'] = str(self._cache_dir)
        
        # ESM-3 source path
        self._esm3_path = Path(__file__).parent.parent.parent.parent.parent / "ESM" / "esm-3" / "esm-main"
        
        # Add ESM-3 to Python path if not already there
        import sys
        esm3_str = str(self._esm3_path)
        if esm3_str not in sys.path:
            sys.path.insert(0, esm3_str)
            if self.logger:
                self.logger.debug(f"Added ESM-3 to path: {esm3_str}")
    
    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention is available."""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _clean_sequence(self, sequence: str) -> str:
        """
        Clean and validate amino acid sequence.
        
        Args:
            sequence: Raw sequence string
            
        Returns:
            Cleaned sequence with only valid amino acids
            
        Raises:
            EmbeddingError: If sequence is invalid
        """
        # Remove whitespace
        sequence = sequence.strip()
        
        # Use class constant for valid amino acids
        valid_aa = self.VALID_AMINO_ACIDS
        
        # Clean sequence (keep only valid amino acids)
        clean = ''.join(c for c in sequence.upper() if c in valid_aa)
        
        if not clean:
            raise EmbeddingError(
                "Sequence contains no valid amino acids. "
                f"Valid codes: {valid_aa}"
            )
        
        # Log if sequence was modified
        if len(clean) != len(sequence) and self.logger:
            removed = len(sequence) - len(clean)
            self.logger.warning(
                f"Removed {removed} invalid characters from sequence"
            )
        
        return clean
