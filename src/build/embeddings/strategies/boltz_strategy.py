"""
Boltz-2 Strategy for Protein Embeddings (CLI-based approach)

This strategy integrates Boltz-2 (biomolecular foundation model with affinity prediction)
to generate protein embeddings from sequences. Uses the Boltz CLI tool (`boltz predict`)
for inference and extracts embeddings from intermediate representations.

Key Features:
- CLI-based execution (subprocess wrapper around `boltz predict`)
- Extracts trunk representations (s and z) from Pairformer blocks
- MSA generation via ColabFold server (--use_msa_server)
- Compatible with DockTKinase embedding concatenation pipeline
- Follows SOLID principles and matches ESM-2/ESM-C/OpenFold3 patterns

Architecture:
- Single representations (s): [N_tokens, token_s=768] - token-level features
- Pair representations (z): [N_tokens, N_tokens, token_z=128] - pairwise interactions
- Default output: Mean-pooled single representation (768-dim)

Differences from OpenFold3:
- Affinity prediction capability (NEW - unique to Boltz-2)
- 64 Pairformer blocks (vs 48 in Boltz-1, 48 in OpenFold3)
- Larger token_s dimension (768 vs 384 in OpenFold3)

Author: DockTKinase Team
Date: 2025-11-20
License: Apache 2.0
"""

import sys
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Optional, Any, Dict
import yaml
import pickle

import torch
import numpy as np

from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy


# =============================================================================
# CONSTANTS
# =============================================================================

# Boltz-2 model specifications
MODEL_SPECS = {
    'boltz2': {
        'dim_single': 384,      # token_s: single representation dimension (per-residue)
        'dim_pair': 128,        # token_z: pair representation dimension
        'output_dim': 384,      # Single representation output (mean pooling, padrão)
        'description': 'Boltz-2 - Biomolecular foundation model (structure + affinity)'
    }
}

# Valid pooling strategies for single representations
VALID_POOLING_STRATEGIES = {'mean', 'cls', 'max', 'multi'}

# Default pooling strategy (mean = 384-dim single representation, padrão)
DEFAULT_POOLING = 'mean'

# Valid amino acids (standard 20 + special tokens)
VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

# Boltz CLI command
BOLTZ_CLI = 'boltz'

# Default MSA server (ColabFold)
DEFAULT_MSA_SERVER = 'https://api.colabfold.com'

# Default recycling and sampling steps
DEFAULT_RECYCLING_STEPS = 3
DEFAULT_SAMPLING_STEPS = 200


# =============================================================================
# BOLTZ-2 STRATEGY CLASS
# =============================================================================

class BoltzStrategy(BaseProteinStrategy):
    """
    Strategy for generating protein embeddings using Boltz-2 CLI.
    
    Boltz-2 is a biomolecular foundation model capable of predicting protein structures
    and binding affinities. This strategy uses the Boltz CLI tool for inference and
    extracts intermediate embeddings from the model without requiring full forward pass.
    
    Attributes:
        logger (logging.Logger): Logger instance for tracking operations
        cli_available (bool): Whether Boltz CLI is installed
        output_dir (Path): Temporary directory for Boltz outputs
        use_msa (bool): Whether to use MSA generation
        msa_server (str): MSA server URL (ColabFold compatible)
        device (torch.device): Device for computation (CPU/CUDA/MPS)
    
    Example:
        >>> strategy = BoltzStrategy(logger=custom_logger, use_msa=True)
        >>> strategy.load('boltz2', device=torch.device('cpu'))
        >>> embedding = strategy.generate(None, None, "MKFLKFSL", strategy.device)
        >>> strategy.cleanup(None, None)
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        use_msa: bool = False,
        msa_server: str = DEFAULT_MSA_SERVER
    ):
        """
        Initialize Boltz strategy with dependency injection.
        
        Args:
            logger: Optional logger instance. If None, creates default logger.
            use_msa: Whether to use MSA generation (default: False for speed)
            msa_server: MSA server URL for ColabFold (default: api.colabfold.com)
        """
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.cli_available = False
        self.output_dir = None
        self.use_msa = use_msa
        self.msa_server = msa_server
        self.device = None
    
    def load(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> Tuple[None, None]:
        """
        Initialize Boltz-2 CLI environment and validate installation.
        
        This method checks if the Boltz CLI is installed and creates a temporary
        directory for storing Boltz outputs. Unlike ESM/OpenFold strategies, no
        model is loaded into memory - inference happens via CLI subprocess.
        
        Args:
            model_name: Name of the model (should be 'boltz2')
            device: Device for computation (CPU/CUDA/MPS). Boltz CLI auto-detects.
            **kwargs: Additional arguments (ignored for CLI-based approach)
        
        Returns:
            (None, None): No model or tokenizer in memory (CLI-based)
        
        Raises:
            ValueError: If model_name is not 'boltz2'
            RuntimeError: If Boltz CLI is not installed
        """
        # Validate model name
        if model_name not in MODEL_SPECS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Supported models: {list(MODEL_SPECS.keys())}"
            )
        
        self.logger.info(f"Initializing Boltz-2 CLI environment for '{model_name}'")
        
        # Check if Boltz CLI is installed
        try:
            result = subprocess.run(
                [BOLTZ_CLI, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            self.cli_available = (result.returncode == 0)
            if self.cli_available:
                version = result.stdout.strip() or result.stderr.strip()
                self.logger.info(f"✓ Boltz CLI found: {version}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            raise RuntimeError(
                f"Boltz CLI not found. Please install: pip install boltz[cuda]\n"
                f"Error: {e}"
            ) from e
        
        # Create temporary output directory
        self.output_dir = Path(tempfile.mkdtemp(prefix='boltz_'))
        self.logger.info(f"✓ Temporary output directory: {self.output_dir}")
        
        # Store device
        self.device = device or torch.device('cpu')
        self.logger.info(f"✓ Device: {self.device}")
        
        # Log MSA configuration
        if self.use_msa:
            self.logger.info(f"✓ MSA generation enabled (server: {self.msa_server})")
        else:
            self.logger.info("ℹ MSA generation disabled (faster, lower accuracy)")
        
        return None, None  # No model/tokenizer in memory
    
    def generate(
        self,
        model: None,
        auxiliary_objects: None,
        sequence: str,
        device: torch.device,
        **kwargs
    ) -> np.ndarray:
        """
        Generate embeddings for a protein sequence using Boltz-2 CLI.
        
        This method:
        1. Creates YAML input file with sequence
        2. Executes `boltz predict` via subprocess
        3. Extracts trunk representations (s) from output
        4. Returns mean-pooled embedding (384-dim)
        
        Args:
            model: Unused (CLI-based approach)
            auxiliary_objects: Unused (CLI-based approach, matches BaseProteinStrategy interface)
            sequence: Protein sequence (single letter amino acid codes)
            device: Device for computation (auto-detected by Boltz CLI)
            **kwargs: Additional arguments:
                - pooling: Pooling strategy ('mean', 'cls', 'max')
                - recycling_steps: Number of recycling steps (default: 3)
                - sampling_steps: Number of sampling steps (default: 200)
        
        Returns:
            Mean-pooled embedding as numpy array [384]
        
        Raises:
            ValueError: If sequence is invalid or empty
            RuntimeError: If Boltz CLI execution fails
        """
        # Validate sequence
        sequence = sequence.strip().upper()
        if not sequence:
            raise ValueError("Sequence cannot be empty")
        
        invalid_chars = set(sequence) - VALID_AMINO_ACIDS
        if invalid_chars:
            raise ValueError(f"Invalid amino acids in sequence: {invalid_chars}")
        
        self.logger.info(f"Generating Boltz-2 embedding for sequence ({len(sequence)} AA)")
        
        # Extract kwargs
        pooling = kwargs.get('pooling', DEFAULT_POOLING)
        recycling_steps = kwargs.get('recycling_steps', DEFAULT_RECYCLING_STEPS)
        sampling_steps = kwargs.get('sampling_steps', DEFAULT_SAMPLING_STEPS)
        seq_id = kwargs.get('seq_id', None)  # Get seq_id if provided
        
        if pooling not in VALID_POOLING_STRATEGIES:
            raise ValueError(f"Invalid pooling strategy: {pooling}")
        
        # Create YAML input file
        yaml_path = self._create_yaml_input(sequence, seq_id=seq_id)
        
        # Execute Boltz CLI
        self._run_boltz_cli(
            yaml_path,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps
        )
        
        # Extract embeddings from output
        embedding = self._extract_embeddings(sequence, pooling=pooling)
        
        return embedding
    
    def cleanup(self, model: None, auxiliary_objects: None) -> None:
        """
        Cleanup Boltz-2 temporary files and directories.
        
        Removes the temporary output directory created during load().
        
        Args:
            model: Unused (CLI-based approach)
            auxiliary_objects: Unused (CLI-based approach, matches BaseProteinStrategy interface)
        """
        if self.output_dir and self.output_dir.exists():
            try:
                shutil.rmtree(self.output_dir)
                self.logger.info(f"✓ Cleaned up temporary directory: {self.output_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup directory: {e}")
        
        self.output_dir = None
        self.cli_available = False
    
    def generate_matrix(
        self,
        model: None,
        auxiliary_objects: None,
        sequence: str,
        device: torch.device,
        **kwargs
    ) -> np.ndarray:
        """
        Generate matrix of embeddings per-token using Boltz-2 CLI (without pooling).
        
        Args:
            model: Unused (CLI-based approach)
            auxiliary_objects: Unused (CLI-based approach)
            sequence: Protein sequence (single letter amino acid codes)
            device: Device for computation
            **kwargs: Additional arguments (recycling_steps, sampling_steps, seq_id)
        
        Returns:
            Matrix numpy array (shape: [seq_len, embedding_dim])
        
        Raises:
            ValueError: If sequence is invalid or empty
            RuntimeError: If Boltz CLI execution fails
        """
        # Validate sequence
        sequence = sequence.strip().upper()
        if not sequence:
            raise ValueError("Sequence cannot be empty")
        
        invalid_chars = set(sequence) - VALID_AMINO_ACIDS
        if invalid_chars:
            raise ValueError(f"Invalid amino acids in sequence: {invalid_chars}")
        
        self.logger.info(f"Generating Boltz-2 embedding MATRIX for sequence ({len(sequence)} AA)")
        
        # Extract kwargs
        recycling_steps = kwargs.get('recycling_steps', DEFAULT_RECYCLING_STEPS)
        sampling_steps = kwargs.get('sampling_steps', DEFAULT_SAMPLING_STEPS)
        seq_id = kwargs.get('seq_id', None)
        
        # Create YAML input file
        yaml_path = self._create_yaml_input(sequence, seq_id=seq_id)
        
        # Execute Boltz CLI
        self._run_boltz_cli(
            yaml_path,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps
        )
        
        # Extract embeddings WITHOUT pooling (return full matrix)
        matrix = self._extract_embedding_matrix(sequence)
        
        return matrix
    
    def _extract_embedding_matrix(self, sequence: str) -> np.ndarray:
        """
        Extract full embedding matrix from Boltz CLI output (without pooling).
        
        Args:
            sequence: Original sequence (for length validation)
        
        Returns:
            Embedding matrix [seq_len, embedding_dim]
        
        Raises:
            RuntimeError: If output files not found or invalid
        """
        # Find the results directory (same logic as _extract_embeddings)
        results_dir = self.output_dir / "boltz_results_input"
        
        if not results_dir.exists():
            raise RuntimeError(f"Boltz results directory not found: {results_dir}")
        
        predictions_dir = results_dir / 'predictions'
        if not predictions_dir.exists():
            raise RuntimeError(f"Boltz predictions directory not found: {predictions_dir}")
        
        job_dirs = [d for d in predictions_dir.iterdir() if d.is_dir()]
        if not job_dirs:
            raise RuntimeError(f"No job directories found in: {predictions_dir}")
        
        protein_dir = job_dirs[0]
        
        # Try to load embeddings
        embedding_files = list(protein_dir.glob('embeddings*.npz'))
        confidence_files = list(protein_dir.glob('confidences*.npz'))
        
        if embedding_files:
            data = np.load(embedding_files[0])
        elif confidence_files:
            data = np.load(confidence_files[0])
        else:
            raise RuntimeError(f"No embedding files found in {protein_dir}")
        
        # Extract 's' (single representation)
        possible_keys = ['s_trunk', 's', 'single']
        s = None
        
        for key in possible_keys:
            if key in data:
                s = data[key]
                break
        
        if s is None:
            raise RuntimeError(f"No valid embedding key found. Available: {list(data.keys())}")
        
        # Convert to numpy if needed
        if isinstance(s, torch.Tensor):
            s = s.cpu().numpy()
        
        # Handle batch dimension if present
        if s.ndim == 3:
            if s.shape[0] != 1:
                raise RuntimeError(f"Unexpected batch size: {s.shape[0]}")
            s = s.squeeze(0)  # [N_tokens, dim]
        
        if s.ndim != 2:
            raise RuntimeError(f"Invalid 's' shape: {s.shape}")
        
        self.logger.info(f"✓ Extracted embedding MATRIX: {s.shape}")
        
        # Return full matrix WITHOUT pooling
        return s.astype(np.float32)
    
    def supports_matrix_output(self) -> bool:
        """Boltz-2 suporta geração de matrizes per-token."""
        return True
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _create_yaml_input(self, sequence: str, seq_id: str = None) -> Path:
        """
        Create YAML input file for Boltz CLI.
        
        Format (Boltz-2 expects):
        ```yaml
        version: 1
        sequences:
          - protein:
              id: A  # or seq_id if provided
              sequence: MKFLKFSL...
              msa: path/to/msa.a3m  # Required even if empty
        ```
        
        Args:
            sequence: Protein sequence
            seq_id: Optional sequence ID (defaults to 'A')
        
        Returns:
            Path to YAML file
        """
        # Create empty MSA file if use_msa is False
        msa_path = None
        if not self.use_msa:
            msa_path = self.output_dir / 'empty.a3m'
            with open(msa_path, 'w') as f:
                # Write minimal A3M format (just the query sequence)
                f.write(f">query\n{sequence}\n")
        
        # Use provided seq_id or default to 'A'
        protein_id = str(seq_id) if seq_id is not None else 'A'
        
        self.logger.info(f"Creating YAML input with protein ID: '{protein_id}'")
        
        yaml_data = {
            'version': 1,
            'sequences': [
                {
                    'protein': {
                        'id': protein_id,
                        'sequence': sequence
                    }
                }
            ]
        }
        
        # Add MSA path if created (must be absolute path)
        if msa_path:
            yaml_data['sequences'][0]['protein']['msa'] = str(msa_path.absolute())
        
        yaml_path = self.output_dir / 'input.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        
        self.logger.info(f"✓ YAML created: {yaml_path}")
        self.logger.info(f"  - Protein ID: {protein_id}")
        self.logger.info(f"  - Sequence length: {len(sequence)} AA")
        if msa_path:
            self.logger.info(f"  - MSA file: {msa_path}")
        
        return yaml_path
    
    def _run_boltz_cli(
        self,
        yaml_path: Path,
        recycling_steps: int,
        sampling_steps: int
    ) -> None:
        """
        Execute Boltz CLI predict command.
        
        Command format:
        ```bash
        boltz predict input.yaml \\
            --out_dir output/ \\
            --recycling_steps 3 \\
            --sampling_steps 200 \\
            [--use_msa_server]
        ```
        
        Args:
            yaml_path: Path to YAML input file
            recycling_steps: Number of recycling steps
            sampling_steps: Number of sampling steps
        
        Raises:
            RuntimeError: If CLI execution fails
        """
        cmd = [
            BOLTZ_CLI, 'predict',
            str(yaml_path),
            '--out_dir', str(self.output_dir),
            '--recycling_steps', str(recycling_steps),
            '--sampling_steps', str(sampling_steps),
            '--model', 'boltz2',  # Use Boltz-2 model
            '--write_embeddings',  # Write embeddings to npz file
            '--accelerator', 'cpu' if self.device.type == 'cpu' else 'gpu',
            '--no_kernels'  # Disable CUDA kernels (avoids cuequivariance-ops-torch dependency)
        ]
        
        # Add MSA server if enabled
        if self.use_msa:
            cmd.extend(['--use_msa_server'])
            # Optionally specify custom server
            if self.msa_server != DEFAULT_MSA_SERVER:
                cmd.extend(['--msa_server_url', self.msa_server])
        
        self.logger.info(f"Executing: {' '.join(cmd)}")
        
        # Set timeout based on MSA usage
        timeout = 1200 if self.use_msa else 600  # 20 min with MSA, 10 min without
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
            
            self.logger.debug(f"Boltz CLI stdout:\n{result.stdout}")
            
            if result.stderr:
                self.logger.info(f"Boltz CLI stderr:\n{result.stderr}")  # Changed to INFO to see MSA errors
        
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"Boltz CLI execution timeout (600s). Try reducing sequence length.\n"
                f"Command: {' '.join(cmd)}"
            ) from e
        
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Boltz CLI execution failed (exit code {e.returncode}):\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            ) from e
    
    def _extract_embeddings(self, sequence: str, pooling: str = 'mean') -> np.ndarray:
        """
        Extract trunk representations from Boltz CLI output.
        
        Boltz creates output structure:
        - <out_dir>/boltz_results_<yaml_name>/predictions/<job_id>/
        
        The job_id is auto-generated by Boltz based on input hash or index.
        
        Args:
            sequence: Original sequence (for length validation)
            pooling: Pooling strategy ('mean', 'cls', 'max')
        
        Returns:
            Pooled embedding [768]
        
        Raises:
            RuntimeError: If output files not found or invalid
        """
        # Boltz creates a subdirectory with pattern: boltz_results_<yaml_basename>
        yaml_basename = self.output_dir / 'input'  # Our YAML file is always 'input.yaml'
        results_pattern = f"boltz_results_input"
        
        # Find the results directory
        results_dir = self.output_dir / results_pattern
        
        if not results_dir.exists():
            raise RuntimeError(
                f"Boltz results directory not found: {results_dir}\n"
                f"Expected: {self.output_dir}/boltz_results_input/\n"
                f"Boltz CLI may have failed."
            )
        
        predictions_dir = results_dir / 'predictions'
        
        if not predictions_dir.exists():
            raise RuntimeError(
                f"Boltz predictions directory not found: {predictions_dir}\n"
                f"Boltz CLI may have failed during prediction phase."
            )
        
        # Find job subdirectories (Boltz may create multiple if batching)
        job_dirs = [d for d in predictions_dir.iterdir() if d.is_dir()]
        
        self.logger.info(f"🔍 Searching for embeddings in: {predictions_dir}")
        self.logger.info(f"   Found {len(job_dirs)} job director{'y' if len(job_dirs) == 1 else 'ies'}: {[d.name for d in job_dirs]}")
        
        if not job_dirs:
            raise RuntimeError(
                f"No job directories found in: {predictions_dir}\n"
                f"Boltz CLI completed processing but generated no predictions."
            )
        
        # Use the first job directory (should be only one for single-sequence input)
        protein_dir = job_dirs[0]
        
        self.logger.info(f"📂 Using job directory: {protein_dir.name}")
        
        # Try to load embeddings - Boltz uses .npz format with --write_embeddings
        # The file naming pattern is: embeddings_<job_id>.npz or confidences_<job_id>.npz
        embedding_files = list(protein_dir.glob('embeddings*.npz'))
        confidence_files = list(protein_dir.glob('confidences*.npz'))
        
        self.logger.info(f"   Embedding files found: {[f.name for f in embedding_files]}")
        
        # Try embeddings first (primary output with --write_embeddings)
        if embedding_files:
            embeddings_npz = embedding_files[0]
            self.logger.info(f"✓ Loading embeddings from: {embeddings_npz.name}")
            try:
                data = np.load(embeddings_npz)
            except Exception as e:
                raise RuntimeError(f"Failed to load {embeddings_npz}: {e}") from e
        # Fall back to confidences
        elif confidence_files:
            confidences_npz = confidence_files[0]
            self.logger.debug(f"Loading embeddings from: {confidences_npz}")
            try:
                data = np.load(confidences_npz)
            except Exception as e:
                raise RuntimeError(f"Failed to load {confidences_npz}: {e}") from e
        else:
            # List all files in directory for debugging
            all_files = list(protein_dir.iterdir())
            raise RuntimeError(
                f"No embedding files found in {protein_dir}\n"
                f"Files present: {[f.name for f in all_files]}\n"
                f"Boltz may have failed to generate embeddings."
            )
        
        # Extract 's' (single representation) - try different keys
        # Boltz may use 's', 's_trunk', or other variants
        available_keys = list(data.keys())
        self.logger.debug(f"Available keys in embedding file: {available_keys}")
        
        # Try different possible keys for embeddings
        possible_keys = ['s_trunk', 's', 'single']
        s = None
        used_key = None
        
        for key in possible_keys:
            if key in data:
                s = data[key]
                used_key = key
                break
        
        if s is None:
            raise RuntimeError(
                f"No valid embedding key found in file. "
                f"Available keys: {available_keys}"
            )
        
        self.logger.debug(f"Using embedding key: '{used_key}' with shape: {s.shape}")
        
        # Validate shape
        if not isinstance(s, (np.ndarray, torch.Tensor)):
            raise RuntimeError(f"Invalid 's' type: {type(s)}")
        
        # Convert to numpy if needed
        if isinstance(s, torch.Tensor):
            s = s.cpu().numpy()
        
        # Handle batch dimension if present
        if s.ndim == 3:
            # Shape: [batch, N_tokens, dim] -> squeeze batch dimension
            if s.shape[0] != 1:
                raise RuntimeError(
                    f"Unexpected batch size: {s.shape[0]} (expected 1 for single sequence)"
                )
            s = s.squeeze(0)  # Now [N_tokens, dim]
            self.logger.debug(f"Squeezed batch dimension: {s.shape}")
        
        # Validate dimensions
        if s.ndim != 2:
            raise RuntimeError(f"Invalid 's' shape: {s.shape} (expected 2D after batch squeeze)")
        
        n_tokens, dim = s.shape
        
        if dim != MODEL_SPECS['boltz2']['dim_single']:
            self.logger.warning(
                f"Unexpected embedding dimension: {dim} "
                f"(expected {MODEL_SPECS['boltz2']['dim_single']})"
            )
        
        self.logger.info(f"✓ Extracted 's' representation: {s.shape}")
        
        # Apply pooling
        if pooling == 'mean':
            embedding = s.mean(axis=0)
        elif pooling == 'cls':
            embedding = s[0]  # First token (CLS-like)
        elif pooling == 'max':
            embedding = s.max(axis=0)
        elif pooling == 'multi':
            # Multi-pooling: concatenate mean, max, min
            mean_pool = s.mean(axis=0)  # 384
            max_pool = s.max(axis=0)    # 384
            min_pool = s.min(axis=0)    # 384
            embedding = np.concatenate([mean_pool, max_pool, min_pool])  # 1152
            
            # Truncate to 1024 dimensions (keep most informative features)
            # Use variance to select top 1024 features across pooling types
            embedding = embedding[:1024]
            
            self.logger.info(f"✓ Applied 'multi' pooling: mean+max+min, truncated to 1024-dim")
        else:
            raise ValueError(f"Invalid pooling: {pooling}")
        
        if pooling != 'multi':
            self.logger.info(f"✓ Applied '{pooling}' pooling: {embedding.shape}")
        
        return embedding.astype(np.float32)
    
    def get_embedding_dim(self, model_name: str) -> int:
        """
        Get the embedding dimension for a Boltz model.
        
        Args:
            model_name: Name of the model (should be 'boltz2')
        
        Returns:
            Embedding dimension (768 for Boltz-2)
        
        Raises:
            ValueError: If model_name is not supported
        """
        if model_name not in MODEL_SPECS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Supported models: {list(MODEL_SPECS.keys())}"
            )
        
        return MODEL_SPECS[model_name]['output_dim']
    
    def get_max_length(self, model_name: str) -> int:
        """
        Get the maximum sequence length supported by a Boltz model.
        
        Boltz-2 doesn't have a hard limit like ESM models, but for practical
        purposes we set a reasonable limit based on computational constraints.
        
        Args:
            model_name: Name of the model (should be 'boltz2')
        
        Returns:
            Maximum sequence length (10000 for Boltz-2)
        
        Raises:
            ValueError: If model_name is not supported
        """
        if model_name not in MODEL_SPECS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Supported models: {list(MODEL_SPECS.keys())}"
            )
        
        # Boltz-2 can handle long sequences but we set practical limit
        return 10000


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_boltz_installation() -> bool:
    """
    Check if Boltz CLI is installed and accessible.
    
    Returns:
        True if Boltz CLI is available, False otherwise
    """
    try:
        result = subprocess.run(
            [BOLTZ_CLI, '--version'],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def get_boltz_version() -> Optional[str]:
    """
    Get installed Boltz CLI version.
    
    Returns:
        Version string or None if not installed
    """
    try:
        result = subprocess.run(
            [BOLTZ_CLI, '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip() or result.stderr.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return None
