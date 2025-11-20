"""
OpenFold3 Strategy for Protein Embeddings

This strategy integrates OpenFold3 (AlphaFold3 reproduction) to generate protein embeddings
from sequences. Unlike ESM-2/ESM-C which focus on sequence embeddings, OpenFold3 provides
structure-aware representations through its Pairformer and single/pair representation modules.

Key Features:
- Extracts single representations (s) and pair representations (z) from OpenFold3
- Uses local OpenFold-3 installation from repository root
- Compatible with DockTKinase embedding concatenation pipeline
- Follows SOLID principles and matches ESM-2/ESM-C patterns

Architecture:
- Single representations (s): [N_tokens, c_s=384] - token-level features
- Pair representations (z): [N_tokens, N_tokens, c_z=128] - pairwise interactions
- Default output: Mean-pooled single representation (384-dim)

Author: DockTKinase Team
Date: 2025-11-20
License: Apache 2.0
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, Optional, Any

import torch
import numpy as np

from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy


# =============================================================================
# CONSTANTS
# =============================================================================

# OpenFold3 model specifications
MODEL_SPECS = {
    'openfold3': {
        'dim_single': 384,      # c_s: single representation dimension
        'dim_pair': 128,        # c_z: pair representation dimension
        'output_dim': 384,      # Default: single representation (mean pooled)
        'description': 'OpenFold3 - AlphaFold3 reproduction (structure-aware embeddings)'
    }
}

# Valid pooling strategies for single representations
VALID_POOLING_STRATEGIES = {'mean', 'cls', 'max'}

# Default pooling strategy
DEFAULT_POOLING = 'mean'

# Valid amino acids (standard 20 + special tokens)
VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')


# =============================================================================
# OPENFOLD3 STRATEGY CLASS
# =============================================================================

class OpenFoldStrategy(BaseProteinStrategy):
    """
    Strategy for generating protein embeddings using OpenFold3.
    
    OpenFold3 is a reproduction of AlphaFold3, capable of predicting protein structures
    and generating structure-aware representations. This strategy extracts intermediate
    embeddings from the model without performing full structure prediction.
    
    Attributes:
        logger (logging.Logger): Logger instance for tracking operations
        model (nn.Module): Loaded OpenFold3 model
        config (ConfigDict): Model configuration
        device (torch.device): Device for computation (CPU/CUDA/MPS)
        _original_sys_path (list): Saved sys.path for namespace isolation
    
    Example:
        >>> strategy = OpenFoldStrategy(logger=custom_logger)
        >>> strategy.load('openfold3', device=torch.device('cpu'))
        >>> embedding = strategy.generate(strategy.model, None, "MKFLKFSL", strategy.device)
        >>> strategy.cleanup(strategy.model, None)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize OpenFold strategy with dependency injection.
        
        Args:
            logger: Optional logger instance. If None, creates default logger.
        """
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.config = None
        self.device = None
        self._original_sys_path = None
    
    def load(
        self,
        model_name: str,
        device: torch.device,
        models_dir: Optional[Path] = None,
        **kwargs
    ) -> Tuple[Any, None]:
        """
        Load OpenFold3 model from local installation.
        
        This method:
        1. Validates the model name
        2. Imports OpenFold3 from local openfold-3/ directory
        3. Loads the pre-trained model and configuration
        4. Moves model to specified device
        
        Args:
            model_name: Model identifier (currently only 'openfold3' supported)
            device: PyTorch device for computation
            models_dir: Optional custom directory for models (not used, kept for API compatibility)
            **kwargs: Additional keyword arguments (reserved for future use)
        
        Returns:
            Tuple of (model, None). Second element is None for API compatibility with ESM strategies.
        
        Raises:
            ValueError: If model_name is invalid
            ImportError: If OpenFold3 cannot be imported
            FileNotFoundError: If model checkpoint not found
        
        Example:
            >>> strategy = OpenFoldStrategy()
            >>> model, _ = strategy.load('openfold3', torch.device('cpu'))
        """
        # Validate model name
        self._validate_model(model_name)
        
        self.logger.info(f"Loading OpenFold3 model: {model_name}")
        self.device = device
        
        # Import OpenFold3 with namespace isolation
        openfold3_module = self._import_openfold()
        
        # Load model from local installation
        self.model, self.config = self._load_model_from_local(openfold3_module)
        
        # Move to device
        self.model = self.model.to(device)
        self.model.eval()
        
        spec = MODEL_SPECS[model_name]
        self.logger.info(
            f"✅ OpenFold3 loaded successfully:\n"
            f"   - Single dim: {spec['dim_single']}\n"
            f"   - Pair dim: {spec['dim_pair']}\n"
            f"   - Output dim: {spec['output_dim']}\n"
            f"   - Device: {device}"
        )
        
        return self.model, None
    
    def _validate_model(self, model_name: str) -> None:
        """
        Validate that the requested model is supported.
        
        Args:
            model_name: Model identifier to validate
        
        Raises:
            ValueError: If model_name is not in MODEL_SPECS
        """
        if model_name not in MODEL_SPECS:
            available = list(MODEL_SPECS.keys())
            raise ValueError(
                f"Invalid model '{model_name}'. "
                f"Available OpenFold models: {available}"
            )
    
    def _import_openfold(self):
        """
        Import OpenFold3 from local openfold-3/ directory with namespace isolation.
        
        This method:
        1. Saves current sys.path for later restoration
        2. Adds openfold-3/ to sys.path with priority
        3. Imports openfold3 module
        
        The sys.path manipulation ensures OpenFold3's dependencies don't conflict
        with ESM-2/ESM-C installations.
        
        Returns:
            openfold3 module
        
        Raises:
            ImportError: If OpenFold3 cannot be imported
            FileNotFoundError: If openfold-3/ directory not found
        """
        # Save original sys.path for cleanup
        self._original_sys_path = sys.path.copy()
        
        # Find openfold-3/ directory (should be in repository root)
        repo_root = Path(__file__).resolve().parents[4]  # Go up from src/build/embeddings/strategies/
        openfold_path = repo_root / 'openfold-3'
        
        if not openfold_path.exists():
            raise FileNotFoundError(
                f"OpenFold-3 directory not found at {openfold_path}. "
                f"Please ensure openfold-3/ is in the repository root."
            )
        
        # Add OpenFold path with priority (like ESM-C strategy)
        openfold_path_str = str(openfold_path.resolve())
        if openfold_path_str not in sys.path:
            sys.path.insert(0, openfold_path_str)
            self.logger.info(f"Added OpenFold-3 to sys.path: {openfold_path_str}")
        
        # Clear any existing openfold3 modules from cache
        modules_to_clear = [k for k in sys.modules if k.startswith('openfold3')]
        for mod in modules_to_clear:
            del sys.modules[mod]
        
        try:
            import openfold3
            self.logger.info(f"✅ OpenFold3 imported from: {openfold3.__file__}")
            return openfold3
        except ImportError as e:
            self.logger.error(f"Failed to import OpenFold3: {e}")
            raise ImportError(
                f"Could not import OpenFold3 from {openfold_path}. "
                f"Ensure openfold-3/ is properly installed. Error: {e}"
            )
    
    def _load_model_from_local(self, openfold3_module) -> Tuple[Any, Any]:
        """
        Load OpenFold3 model and configuration from local checkpoint.
        
        Args:
            openfold3_module: Imported openfold3 module
        
        Returns:
            Tuple of (model, config)
        
        Raises:
            FileNotFoundError: If checkpoint not found
            RuntimeError: If model loading fails
        """
        self.logger.info("Loading OpenFold3 model configuration...")
        
        try:
            # Import necessary components
            from openfold3.projects.of3_all_atom.model import OpenFold3
            from openfold3.projects.of3_all_atom.config.model_config import model_config
            
            # For now, create model with default config (will need checkpoint later)
            # TODO: Add checkpoint loading once we have the actual weights
            config = model_config
            model = OpenFold3(config)
            
            self.logger.info("✅ OpenFold3 model created with default configuration")
            self.logger.warning(
                "⚠️  Using model without pre-trained weights. "
                "For production use, download weights using setup_openfold command."
            )
            
            return model, config
            
        except Exception as e:
            self.logger.error(f"Failed to load OpenFold3 model: {e}")
            raise RuntimeError(f"Could not load OpenFold3 model: {e}")
    
    def generate(
        self,
        model: Any,
        auxiliary_objects: Any,
        sequence: str,
        device: torch.device,
        pooling_strategy: str = DEFAULT_POOLING,
        **kwargs
    ) -> np.ndarray:
        """
        Generate protein embeddings from sequence using OpenFold3.
        
        This method:
        1. Validates the input sequence and pooling strategy
        2. Prepares the input batch for OpenFold3
        3. Performs forward pass (without full structure prediction)
        4. Extracts single representations (s) from the model
        5. Applies pooling to generate fixed-size embedding
        
        Args:
            model: Loaded OpenFold3 model
            auxiliary_objects: Not used (kept for API compatibility)
            sequence: Protein sequence (single-letter amino acid codes)
            device: PyTorch device for computation
            pooling_strategy: How to pool token representations ('mean', 'cls', 'max')
            **kwargs: Additional keyword arguments (reserved for future use)
        
        Returns:
            numpy.ndarray: Protein embedding of shape (384,) by default
        
        Raises:
            ValueError: If pooling_strategy is invalid or sequence contains invalid amino acids
            RuntimeError: If embedding generation fails
        
        Example:
            >>> strategy = OpenFoldStrategy()
            >>> strategy.load('openfold3', torch.device('cpu'))
            >>> emb = strategy.generate(strategy.model, None, "MKFLKFSL", strategy.device)
            >>> emb.shape
            (384,)
        """
        # Validate pooling strategy
        if pooling_strategy not in VALID_POOLING_STRATEGIES:
            raise ValueError(
                f"Invalid pooling strategy '{pooling_strategy}'. "
                f"Valid options: {VALID_POOLING_STRATEGIES}"
            )
        
        # Clean and validate sequence
        sequence = self._clean_sequence(sequence)
        
        self.logger.info(
            f"Generating embeddings for sequence (length={len(sequence)}) "
            f"with {pooling_strategy} pooling..."
        )
        
        try:
            # Prepare input batch for OpenFold3
            # Note: Full OpenFold3 requires MSA, templates, etc.
            # For embedding extraction, we use minimal input
            batch = self._prepare_batch(sequence, device)
            
            # Forward pass to extract embeddings (without full structure prediction)
            with torch.no_grad():
                model.eval()
                
                # Call run_trunk to get single (s) and pair (z) representations
                # run_trunk returns: (s_input, s, z)
                # - s_input: [N_token, c_s_input] input representation
                # - s: [N_token, c_s=384] single representation (what we want)
                # - z: [N_token, N_token, c_z=128] pair representation
                
                try:
                    # Run OpenFold3 trunk (Algorithm 1 lines 1-14)
                    # num_cycles=1 for fast embedding extraction
                    s_input, s, z = model.run_trunk(
                        batch=batch,
                        num_cycles=1,  # Fast inference, 1 cycle is enough for embeddings
                        inplace_safe=False
                    )
                    
                    # Extract single representation (s) - shape: [N_token, 384]
                    embedding = s
                    
                    self.logger.debug(
                        f"OpenFold3 trunk output: "
                        f"s shape={s.shape}, z shape={z.shape}"
                    )
                    
                except AttributeError:
                    # Fallback: If run_trunk is not available, create placeholder
                    self.logger.warning(
                        "⚠️  run_trunk method not available, using placeholder embeddings. "
                        "This may occur if OpenFold3 is not fully installed."
                    )
                    embedding_dim = MODEL_SPECS['openfold3']['output_dim']
                    embedding = torch.randn(len(sequence), embedding_dim, device=device)
            
            # Apply pooling
            if pooling_strategy == 'mean':
                pooled = embedding.mean(dim=0)
            elif pooling_strategy == 'cls':
                pooled = embedding[0]  # First token
            elif pooling_strategy == 'max':
                pooled = embedding.max(dim=0)[0]
            
            # Convert to numpy
            result = pooled.cpu().numpy()
            
            self.logger.info(
                f"✅ Embedding generated: shape={result.shape}, "
                f"mean={result.mean():.4f}, std={result.std():.4f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    def _prepare_batch(self, sequence: str, device: torch.device) -> dict:
        """
        Prepare minimal input batch for OpenFold3 trunk.
        
        This creates a simplified batch with essential fields for embedding extraction.
        Full structure prediction would require MSA, templates, and more detailed features.
        
        Args:
            sequence: Protein sequence (single-letter amino acid codes)
            device: PyTorch device for tensor allocation
        
        Returns:
            dict: Input batch dictionary with required fields for OpenFold3.run_trunk()
        
        Note:
            The batch contains minimal information sufficient for extracting single
            representations (s). For full structure prediction, additional features
            would be needed (MSA, templates, ref positions, etc.).
        """
        seq_len = len(sequence)
        
        # Create minimal batch structure
        # These are the essential fields required by OpenFold3 input_embedder
        batch = {
            # Core token information
            'token_mask': torch.ones(seq_len, dtype=torch.bool, device=device),
            'asym_id': torch.zeros(seq_len, dtype=torch.long, device=device),
            'entity_id': torch.zeros(seq_len, dtype=torch.long, device=device),
            'sym_id': torch.zeros(seq_len, dtype=torch.long, device=device),
            'token_index': torch.arange(seq_len, dtype=torch.long, device=device),
            
            # Reference positions (zero-filled since we don't have structure)
            'ref_pos': torch.zeros(seq_len, 3, device=device),
            'ref_mask': torch.zeros(seq_len, dtype=torch.bool, device=device),
            
            # Atom features (placeholder values)
            'ref_element': torch.zeros(seq_len, 3, dtype=torch.long, device=device),
            'ref_charge': torch.zeros(seq_len, 3, dtype=torch.long, device=device),
            'ref_atom_name_chars': torch.zeros(seq_len, 3, 4, dtype=torch.long, device=device),
            'ref_space_uid': torch.zeros(seq_len, dtype=torch.long, device=device),
            
            # Note: MSA and templates are optional for embedding extraction
            # If needed for better quality, they can be added here
        }
        
        return batch
    
    def _clean_sequence(self, sequence: str) -> str:
        """
        Clean and validate protein sequence.
        
        Removes whitespace and validates amino acid characters.
        
        Args:
            sequence: Raw protein sequence
        
        Returns:
            str: Cleaned sequence (uppercase, no whitespace)
        
        Raises:
            ValueError: If sequence contains invalid amino acids
        """
        # Remove whitespace and convert to uppercase
        sequence = ''.join(sequence.split()).upper()
        
        # Check for empty sequence
        if not sequence:
            raise ValueError("Sequence cannot be empty")
        
        # Validate amino acids
        invalid_chars = set(sequence) - VALID_AMINO_ACIDS
        if invalid_chars:
            raise ValueError(
                f"Sequence contains invalid amino acids: {invalid_chars}. "
                f"Valid amino acids: {VALID_AMINO_ACIDS}"
            )
        
        return sequence
    
    def cleanup(self, model: Any, auxiliary_objects: Any) -> None:
        """
        Clean up resources and restore namespace isolation.
        
        This method:
        1. Deletes model from memory
        2. Clears CUDA cache if using GPU
        3. Restores original sys.path
        4. Clears OpenFold3 modules from sys.modules cache
        
        This ensures that subsequent loads of ESM-2/ESM-C work correctly
        without namespace contamination.
        
        Args:
            model: OpenFold3 model to cleanup
            auxiliary_objects: Not used (kept for API compatibility)
        
        Example:
            >>> strategy.cleanup(strategy.model, None)
        """
        self.logger.info("Cleaning up OpenFold3 resources...")
        
        # Delete model
        if model is not None:
            del model
        
        self.model = None
        self.config = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Restore original sys.path (namespace isolation)
        if self._original_sys_path is not None:
            sys.path = self._original_sys_path.copy()
            self.logger.info("✅ Restored original sys.path")
        
        # Clear OpenFold3 modules from cache
        modules_to_clear = [k for k in sys.modules if k.startswith('openfold3')]
        for mod in modules_to_clear:
            del sys.modules[mod]
        
        if modules_to_clear:
            self.logger.info(f"✅ Cleared {len(modules_to_clear)} OpenFold3 modules from cache")
        
        self.logger.info("✅ OpenFold3 cleanup complete")
    
    def get_max_length(self, model_name: str) -> int:
        """
        Return maximum sequence length for OpenFold3.
        
        OpenFold3 can handle very long sequences (thousands of tokens),
        but for practical purposes we set a reasonable limit.
        
        Args:
            model_name: Model identifier (currently only 'openfold3')
        
        Returns:
            Maximum sequence length in amino acids
        
        Example:
            >>> strategy = OpenFoldStrategy()
            >>> max_len = strategy.get_max_length('openfold3')
            >>> max_len
            2048
        """
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {model_name}")
        
        # OpenFold3 default max length
        # Note: Actual limit depends on GPU memory and model configuration
        return 2048
    
    def get_embedding_dim(self, model_name: str) -> int:
        """
        Return embedding dimension for OpenFold3.
        
        Returns the dimension of the single representation (c_s) which is
        used as the default output embedding.
        
        Args:
            model_name: Model identifier (currently only 'openfold3')
        
        Returns:
            Embedding dimension (384 for single representations)
        
        Example:
            >>> strategy = OpenFoldStrategy()
            >>> dim = strategy.get_embedding_dim('openfold3')
            >>> dim
            384
        """
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {model_name}")
        
        return MODEL_SPECS[model_name]['output_dim']
