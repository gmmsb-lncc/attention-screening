"""
Generation of ligand embeddings using FM4M models (IBM).
"""

import os
import sys
import time
import random
from typing import Dict, Any, List, Tuple, Optional, Union, TYPE_CHECKING
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    from src.build.core import BuildConfig

from src.build.embeddings.base_embedding import BaseEmbedding
from src.build.core.constants import FM4M_MODELS
from src.build.core.exceptions import DependencyError, EmbeddingError
from src.build.utils import ProgressLogger, ensure_directory, optimize_batch_size, memory_monitor

class LigandEmbedding(BaseEmbedding):
    """Ligand embedding generator using FM4M."""
    
    def __init__(self, 
                 config: Optional['BuildConfig'] = None,
                 model_name: str = "SMI-TED",
                 use_parallel: bool = True,
                 checkpoint_enabled: bool = True,
                 **kwargs):
        """
        Initialize ligand embedding generator.
        
        Args:
            config: Build system configuration
            model_name: FM4M model name
            use_parallel: Whether to use parallel processing
            checkpoint_enabled: Whether to use checkpoint system
            **kwargs: Additional arguments
        """
        # Definir atributos antes da inicialização do pai
        self.use_parallel = use_parallel
        self.checkpoint_enabled = checkpoint_enabled
        self.processed_files = set()
        self.checkpoint_file = None

        # Cache do modelo SMI-TED
        self._smited_model = None
        self._smited_model_loaded = False

        # Cache do modelo MoLFormer
        self._molformer_model = None
        self._molformer_tokenizer = None
        self._molformer_model_loaded = False

        super().__init__(model_name=model_name, config=config, **kwargs)
        
        # Verificar dependências
        self._check_dependencies()
        self._setup_fm4m_path()
    
    def _check_dependencies(self) -> None:
        """Check if dependencies are available."""
        try:
            import pandas as pd
            self.pd = pd
        except ImportError:
            raise DependencyError("Pandas not available. Install with: pip install pandas")
        
        # FM4M will be checked during initialization
        self.fm4m = None
        self.fm4m_available = False
    
    def _validate_config(self) -> None:
        """Validate ligand embedding specific configuration."""
        super()._validate_config()
        
        # Validate FM4M model
        if self.model_name.upper() not in [model.upper() for model in FM4M_MODELS.keys()] and self.model_name.upper() != "MOLFORMER":
            raise EmbeddingError(f"Invalid FM4M model: {self.model_name}. Available models: {list(FM4M_MODELS.keys()) + ['MOLFORMER']}")
        
        # Check parallel processing configuration
        if self.use_parallel:
            try:
                import multiprocessing
                cpu_count = multiprocessing.cpu_count()
                if cpu_count < 2:
                    self.logger.warning("Parallel processing requested but only 1 CPU available")
                    self.use_parallel = False
            except Exception:
                self.logger.warning("Could not check available CPUs. Disabling parallel processing.")
                self.use_parallel = False
    
    def _setup_fm4m_path(self) -> None:
        """Configure path for FM4M."""
        # Adicionar FM4M ao path se necessário - usar insert(0) para prioridade
        current_dir = Path(__file__).parent.parent.parent.parent  # Volta para raiz
        fm4m_path = current_dir / "llm" / "FM4M"
        
        if fm4m_path.exists():
            fm4m_str = str(fm4m_path)
            models_path = str(fm4m_path / "models")
            
            # Usar insert(0) para garantir prioridade sobre outros módulos 'models'
            if models_path not in sys.path:
                sys.path.insert(0, models_path)
            if fm4m_str not in sys.path:
                sys.path.insert(0, fm4m_str)
    
    def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Return supported FM4M models."""
        return FM4M_MODELS.copy()
    
    def _load_model(self) -> Any:
        """Load SMI-TED model directly (avoids fm4m.py which has bug)."""
        try:
            # NOTE: Don't import models.fm4m directly as it causes crash
            # Import only the specific model needed
            self.fm4m = None  # Don't use generic fm4m.py
            self.fm4m_available = True
            
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Load SMI-TED model directly
            if self.model_name == "SMI-TED":
                import os
                import torch
                from smi_ted.smi_ted_light.load import load_smi_ted

                # Locate model files
                materials_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                model_files_path = os.path.join(materials_path, "llm", "FM4M", "model_files")

                self.logger.info(f"Pre-loading SMI-TED model from: {model_files_path}")

                # Check if files exist
                vocab_file = os.path.join(model_files_path, "bert_vocab_curated.txt")
                ckpt_file = os.path.join(model_files_path, "smi-ted-Light_40.pt")

                if not os.path.exists(vocab_file) or not os.path.exists(ckpt_file):
                    raise DependencyError(
                        f"SMI-TED model files not found at {model_files_path}. "
                        f"Please run: cd llm/FM4M && python download_model_files.py"
                    )

                # Load model ONCE and cache
                self._smited_model = load_smi_ted(folder=model_files_path, ckpt_filename='smi-ted-Light_40.pt')
                self._smited_model_loaded = True
                self.logger.info("✅ SMI-TED model pre-loaded and cached!")

                return self._smited_model

            # Load MoLFormer model
            elif self.model_name.upper() == "MOLFORMER":
                import os
                import torch
                from transformers import AutoTokenizer, AutoModelForMaskedLM

                # Locate cached model
                materials_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                model_cache_path = os.path.join(materials_path, "llm", "models_cache", "molformer")

                self.logger.info(f"Pre-loading MoLFormer model from: {model_cache_path}")

                # Check if cached model exists
                model_path = os.path.join(model_cache_path, "model")
                tokenizer_path = os.path.join(model_cache_path, "tokenizer")

                if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
                    raise DependencyError(
                        f"MoLFormer model files not found at {model_cache_path}. "
                        f"Please run: python llm/MoLFormer/download_model.py"
                    )

                # Load model and tokenizer ONCE and cache
                self._molformer_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                self._molformer_model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
                self._molformer_model_loaded = True
                self.logger.info("✅ MoLFormer model pre-loaded and cached!")

                return self._molformer_model
            else:
                # For other models, try importing fm4m (may fail)
                try:
                    import models.fm4m as fm4m
                    self.fm4m = fm4m
                except ImportError:
                    raise DependencyError(f"Model {self.model_name} requires fm4m which is not available")

                self._smited_model = None
                self._smited_model_loaded = False
                return self.fm4m
            
        except ImportError as e:
            raise DependencyError(
                f"FM4M is not available: {e}. "
                "Make sure the FM4M directory is present and dependencies are installed."
            )
    
    def _do_initialize(self) -> None:
        """Ligand-specific initialization."""
        # Call parent initialization first to load the model
        super()._do_initialize()

        # Update fm4m_available flag based on model type
        if self.model_name.upper() == "MOLFORMER":
            # For MoLFormer, we don't use the generic fm4m, but the model is available
            self.fm4m_available = True
        # For SMI-TED and other FM4M models, fm4m_available is already set in _load_model

        # Configure checkpoint if enabled
        if self.checkpoint_enabled:
            self.checkpoint_file = Path(self.get_config('base_dir', '.')) / 'processed_ligands.log'
            self._load_checkpoint()
    
    def _load_checkpoint(self) -> None:
        """Load already processed files from checkpoint."""
        if self.checkpoint_file and self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self.processed_files = {line.strip() for line in f}
                self.logger.info(f"Checkpoint loaded: {len(self.processed_files)} files already processed")
            except Exception as e:
                self.logger.warning(f"Error loading checkpoint: {e}")
                self.processed_files = set()
    
    def _update_checkpoint(self, file_name: str) -> None:
        """Update checkpoint with processed file."""
        if self.checkpoint_enabled and self.checkpoint_file:
            try:
                with open(self.checkpoint_file, 'a') as f:
                    f.write(f"{file_name}\\n")
                self.processed_files.add(file_name)
            except Exception as e:
                self.logger.warning(f"Error updating checkpoint: {e}")
    
    def _generate_single_embedding(self, smiles: str) -> np.ndarray:
        """
        Generate embedding for a single SMILES string.
        
        Args:
            smiles: Ligand SMILES string
            
        Returns:
            NumPy array with embedding
        """
        return self._generate_batch_embeddings([smiles])[0]
    
    def generate_embedding_matrix(self, smiles: str) -> Optional[np.ndarray]:
        """
        Generate embedding matrix per token/atom (no pooling).
        
        Returns representations for each SMILES token,
        preserving positional information for use with architectures
        like CNN + Cross-Attention.
        
        Args:
            smiles: Ligand SMILES string
            
        Returns:
            NumPy array with shape [n_tokens, embed_dim] or None if not supported
            
        Note:
            Most ligand models (including SMI-TED) produce
            only global representations (vectors). This method returns None
            when per-token representations are not available.
            
            To use cross-attention with ligands, consider:
            1. Using SMILES tokenization and learned embeddings
            2. Using 3D fingerprints if structure available
            3. Using GraphTransformer for per-atom representations
        """
        # Ensure model is loaded
        if not self._model_loaded:
            self._do_initialize()

        # SMI-TED does not support per-token representations
        # Returns None to indicate matrix is not available
        if self.model_name == "SMI-TED":
            self.logger.debug(
                f"Model {self.model_name} does not support per-token representations. "
                "Returning None. Use generate_embedding() for global vector."
            )
            return None

        # MoLFormer supports per-token representations
        elif self.model_name.upper() == "MOLFORMER":
            import torch
            import numpy as np

            # Tokenize the SMILES string
            inputs = self._molformer_tokenizer(
                smiles,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=512
            )

            # Get model outputs
            with torch.no_grad():
                outputs = self._molformer_model(**inputs, output_hidden_states=True)

            # Get the last hidden state (this gives us per-token representations)
            last_hidden_state = outputs.hidden_states[-1][0]  # Shape: [seq_len, hidden_size]

            # Remove [CLS] and [SEP] tokens if not needed (optional)
            # last_hidden_state = last_hidden_state[1:-1]  # Skip first and last token

            # Convert to numpy array
            embedding_matrix = last_hidden_state.cpu().numpy()

            self.logger.debug(
                f"Generated per-token embedding matrix for {self.model_name}: "
                f"shape {embedding_matrix.shape}"
            )
            return embedding_matrix

        # For other models, check if they support per-token representations
        # For now, return None for all other FM4M models
        else:
            self.logger.warning(
                f"Model {self.model_name} does not support per-token/atom representations. "
                "Use generate_embedding() to get global ligand representation."
            )
            return None
    
    def _generate_batch_embeddings(self, smiles_list: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for batch of SMILES with retry logic and individual processing.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of NumPy arrays with embeddings
        """
        if not self.fm4m_available:
            raise EmbeddingError("FM4M is not loaded")
        
        # Filter empty SMILES
        valid_smiles = [s for s in smiles_list if s and s.strip()]
        if not valid_smiles:
            raise EmbeddingError("No valid SMILES provided")
        
        try:
            # If model is cached (SMI-TED), use directly
            if self._smited_model_loaded and self.model_name == "SMI-TED":
                import torch

                # For SMI-TED, process individually to avoid C++ crashes
                embeddings = []
                for smiles in valid_smiles:
                    try:
                        with torch.no_grad():
                            # Process SMILES individually
                            representation = self._smited_model.encode([smiles], return_torch=False)

                            # Convert to numpy array
                            if hasattr(representation, 'values'):
                                embedding = representation.values[0]
                            else:
                                embedding = np.array(representation[0])

                            embeddings.append(embedding)

                    except Exception as e:
                        # Log error but don't fail entire batch
                        if self.logger:
                            self.logger.warning(f"Failed to process SMILES '{smiles[:50]}...': {e}")
                        # Re-raise for individual SMILES (will be caught at upper level)
                        raise EmbeddingError(f"Error processing SMILES: {e}")

                return embeddings
            # If model is cached (MoLFormer), use directly
            elif self._molformer_model_loaded and self.model_name.upper() == "MOLFORMER":
                import torch

                # For MoLFormer, process individually to get pooled embeddings
                embeddings = []
                for smiles in valid_smiles:
                    try:
                        # Tokenize the SMILES string
                        inputs = self._molformer_tokenizer(
                            smiles,
                            return_tensors="pt",
                            padding=False,
                            truncation=True,
                            max_length=512
                        )

                        # Get model outputs
                        with torch.no_grad():
                            outputs = self._molformer_model(**inputs, output_hidden_states=True)

                        # Get the last hidden state and pool it (mean pooling)
                        last_hidden_state = outputs.hidden_states[-1][0]  # Shape: [seq_len, hidden_size]

                        # Apply mean pooling to get a single vector representation
                        pooled_embedding = torch.mean(last_hidden_state, dim=0)  # Shape: [hidden_size]

                        # Convert to numpy array
                        embedding = pooled_embedding.cpu().numpy()
                        embeddings.append(embedding)

                    except Exception as e:
                        # Log error but don't fail entire batch
                        if self.logger:
                            self.logger.warning(f"Failed to process SMILES '{smiles[:50]}...': {e}")
                        # Re-raise for individual SMILES (will be caught at upper level)
                        raise EmbeddingError(f"Error processing SMILES: {e}")

                return embeddings
            else:
                # Fallback: use original method (for other models)
                representations = self._get_representation_with_retry(valid_smiles)

                # Convert to list of numpy arrays
                if hasattr(representations, 'values'):
                    embeddings = [representations.values[i] for i in range(len(representations.values))]
                else:
                    embeddings = [np.array(row) for row in representations]

                return embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Error generating FM4M embeddings: {e}")
    
    def _get_representation_with_retry(self, smiles_list: List[str], max_retries: int = 3):
        """Generate representations with retry logic for rate limiting."""
        
        def _get_representation():
            return self.fm4m.get_representation(
                train_data=smiles_list,
                test_data=smiles_list,  # Reutilizar para evitar erro de test_data vazio
                model_type=self.model_name,
                return_tensor=False
            )
        
        for attempt in range(max_retries):
            try:
                return _get_representation()
                
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        self.logger.warning(f"Rate limited. Waiting {wait_time:.2f}s before retrying...")
                        time.sleep(wait_time)
                    else:
                        raise EmbeddingError(f"Rate limit exceeded after {max_retries} attempts")
                else:
                    raise e
    
    @memory_monitor(threshold_percent=85.0)
    def generate_batch_embeddings(self,
                                 input_list: List[str],
                                 batch_size: Optional[int] = None,
                                 show_progress: bool = True) -> List[np.ndarray]:
        """
        Override base class method to use FM4M implementation.

        Args:
            input_list: List of SMILES
            batch_size: Batch size
            show_progress: Whether to show progress

        Returns:
            List of embeddings
        """
        # Ensure model is loaded
        if not self._model_loaded:
            self._do_initialize()

        # Now check if the model is available
        if not self.fm4m_available and not self._molformer_model_loaded:
            raise EmbeddingError("FM4M or MoLFormer is not loaded")

        if not input_list:
            return []

        # Optimize batch size
        if batch_size is None:
            batch_size = self.get_config('batch_size', 32)
        batch_size = optimize_batch_size(batch_size)

        embeddings = []
        total_items = len(input_list)

        # Progress logger
        if show_progress:
            progress_logger = ProgressLogger(
                self.logger,
                total_items,
                f"Generating FM4M embeddings ({self.model_name})"
            )

        # Process in batches
        for i in range(0, total_items, batch_size):
            batch = input_list[i:i + batch_size]

            try:
                batch_embeddings = self._generate_batch_embeddings(batch)
                embeddings.extend(batch_embeddings)

                if show_progress:
                    progress_logger.update(len(batch))

            except Exception as e:
                self.logger.error(f"Error in batch {i//batch_size + 1}: {e}")
                # Add zero embeddings to maintain consistency
                zero_embeddings = [np.zeros(self.embedding_dim) for _ in batch]
                embeddings.extend(zero_embeddings)

                if show_progress:
                    progress_logger.update(len(batch))

        if show_progress:
            progress_logger.finish()

        return embeddings
    
    def process_smi_file(self, 
                        smi_file: Path,
                        output_dir: Path,
                        batch_size: Optional[int] = None) -> Tuple[int, int]:
        """
        Process .smi file with SMILES.
        
        Args:
            smi_file: Input .smi file
            output_dir: Output directory
            batch_size: Batch size
            
        Returns:
            Tuple (successes, failures)
        """
        file_name = smi_file.name
        
        # Check checkpoint
        if self.checkpoint_enabled and file_name in self.processed_files:
            self.logger.info(f"File already processed (checkpoint): {file_name}")
            return 1, 0  # Assume success
        
        try:
            # Read SMILES
            with open(smi_file, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
            
            if not smiles_list:
                self.logger.warning(f"Empty file: {file_name}")
                return 0, 1
            
            self.logger.info(f"Processing {len(smiles_list)} SMILES from {file_name}")
            
            # Generate embeddings
            embeddings = self.generate_batch_embeddings(smiles_list, batch_size)
            
            # Prepare output
            output_path = ensure_directory(output_dir)
            output_file = output_path / f"{smi_file.stem}.npy"
            
            # Convert to numpy array and save
            embeddings_array = np.array(embeddings)
            np.save(output_file, embeddings_array)
            
            # Update checkpoint
            self._update_checkpoint(file_name)
            
            self.logger.info(f"Embeddings saved: {output_file}")
            return 1, 0
            
        except Exception as e:
            self.logger.error(f"Error processing {file_name}: {e}")
            return 0, 1
    
    def process_smiles_directory(self,
                               input_dir: Path,
                               output_dir: Path,
                               use_parallel: Optional[bool] = None,
                               max_workers: Optional[int] = None) -> Tuple[int, int]:
        """
        Process directory with .smi files.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            use_parallel: Whether to use parallel processing
            max_workers: Maximum number of workers
            
        Returns:
            Tuple (successes, failures)
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise EmbeddingError(f"Directory not found: {input_dir}")
        
        # Find .smi files
        smi_files = list(input_path.glob("*.smi"))
        
        if not smi_files:
            raise EmbeddingError(f"No .smi files found in {input_dir}")
        
        self.logger.info(f"Found {len(smi_files)} .smi files")
        
        # Use class configuration if not specified
        if use_parallel is None:
            use_parallel = self.use_parallel
        
        total_successes = 0
        total_failures = 0
        
        if use_parallel and len(smi_files) > 1:
            # Parallel processing
            if max_workers is None:
                max_workers = min(os.cpu_count(), len(smi_files))
            
            self.logger.info(f"Parallel processing with {max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs
                futures = []
                for smi_file in smi_files:
                    future = executor.submit(self.process_smi_file, smi_file, output_dir)
                    futures.append(future)
                
                # Collect results
                for future in futures:
                    try:
                        successes, failures = future.result()
                        total_successes += successes
                        total_failures += failures
                    except Exception as e:
                        self.logger.error(f"Error in parallel worker: {e}")
                        total_failures += 1
        else:
            # Sequential processing
            self.logger.info("Sequential processing")
            
            for smi_file in smi_files:
                successes, failures = self.process_smi_file(smi_file, output_dir)
                total_successes += successes
                total_failures += failures
        
        self.logger.info(f"Directory processing completed: {total_successes} successes, {total_failures} failures")
        return total_successes, total_failures
    
    def generate_embeddings(self, 
                          tsv_path: Path, 
                          output_dir: Optional[Path] = None,
                          save_matrix: bool = False,
                          matrix_output_dir: Optional[Path] = None) -> bool:
        """
        Generate embeddings from TSV file (interface for pipeline).
        
        Args:
            tsv_path: TSV file with data
            output_dir: Output directory (uses config if None)
            save_matrix: If True, also save embedding matrices [n_tokens, dim]
                        (Note: SMI-TED does not support per-token matrices)
            matrix_output_dir: Directory for matrices (uses 'ligand_matrix_embeddings' if None)
            
        Returns:
            True if successful
        """
        import pandas as pd
        from src.build.utils import ensure_directory
        from src.build.core.constants import BuildConstants
        
        try:
            # Determinar diretório de saída ANTES de carregar modelo
            if output_dir is None:
                output_dir = Path(self.get_config('ligand_output_dir', 'ligand_embeddings'))
            
            output_dir = Path(output_dir)
            output_dir = ensure_directory(output_dir)
            
            # Determine output directory for matrices (if enabled)
            # Note: SMI-TED does not support matrices, but we keep interface consistent
            if save_matrix:
                if matrix_output_dir is None:
                    matrix_output_dir = Path(
                        self.get_config(
                            'ligand_matrix_output_dir', 
                            BuildConstants.DEFAULT_LIGAND_MATRIX_OUTPUT_DIR
                        )
                    )
                matrix_output_dir = Path(matrix_output_dir)
                matrix_output_dir = ensure_directory(matrix_output_dir)
                self.logger.info(f"📊 Ligand matrix directory configured: {matrix_output_dir}")
                # Only warn if the model doesn't support matrices
                if self.model_name.upper() != "MOLFORMER":
                    self.logger.warning(
                        f"⚠️ Note: {self.model_name} model does not support per-token/atom representations. "
                        "Ligand matrices will not be generated."
                    )
                else:
                    self.logger.info(
                        f"✅ {self.model_name} model supports per-token representations. "
                        f"Ligand matrices will be generated in: {matrix_output_dir}"
                    )
            
            # Load TSV
            self.logger.info(f"Loading data from {tsv_path}")
            df = pd.read_csv(tsv_path, sep='\t')
            
            # Check required columns
            if 'chembl_id' not in df.columns or 'canonical_smiles' not in df.columns:
                raise EmbeddingError("TSV must contain columns 'chembl_id' and 'canonical_smiles'")
            
            # Obter ligantes únicos
            unique_smiles = df.groupby('chembl_id')['canonical_smiles'].first()
            
            # Verificar quantos embeddings já existem (procura também em ligand_embeddings global)
            import shutil
            existing_embeddings = []
            missing_embeddings = []

            # Diretório global de embeddings (workaround para testes)
            global_embeddings_dir = Path(__file__).resolve().parents[3] / 'ligand_embeddings'

            for chembl_id in unique_smiles.index:
                filename = f"{chembl_id}_embedding.npy"
                output_file = output_dir / filename

                # Se já existe no output do run, conta como existente
                if output_file.exists():
                    existing_embeddings.append(chembl_id)
                    continue

                # Se existe no diretório global de embeddings, copie para o output e conte como existente
                global_file = global_embeddings_dir / filename
                if global_file.exists():
                    try:
                        shutil.copy(str(global_file), str(output_file))
                        existing_embeddings.append(chembl_id)
                        if self.logger:
                            self.logger.info(f"Copied global embedding to output: {filename}")
                        continue
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Failed to copy global embedding {filename}: {e}")

                # Caso contrário, está faltando
                missing_embeddings.append(chembl_id)
            
            # If ALL embeddings already exist, skip model initialization
            if len(missing_embeddings) == 0:
                self.logger.info(f"✅ All {len(existing_embeddings)} embeddings already exist - skipping generation")
                self._output_path = output_dir
                self.logger.info(f"Ligand embeddings: {len(existing_embeddings)} successes, 0 failures")
                return True
            
            # If there are missing embeddings, initialize model
            self.logger.info(f"Existing embeddings: {len(existing_embeddings)}/{len(unique_smiles)}")
            self.logger.info(f"Missing embeddings: {len(missing_embeddings)}")
            
            if not self._model_loaded:
                self.logger.info("Initializing FM4M model...")
                self._do_initialize()
            
            # Get unique SMILES
            unique_smiles = df.groupby('chembl_id')['canonical_smiles'].first()
            self.logger.info(f"Processing {len(unique_smiles)} unique ligands")
            
            # Process each ligand
            successes = 0
            failures = 0
            
            progress_logger = ProgressLogger(
                self.logger,
                len(unique_smiles),
                "Generating ligand embeddings"
            )
            
            for chembl_id, smiles in unique_smiles.items():
                try:
                    # Check if already exists
                    output_file = output_dir / f"{chembl_id}_embedding.npy"
                    if output_file.exists():
                        self.logger.debug(f"Embedding already exists: {chembl_id}")

                        # Even if embedding exists, check if matrix needs to be generated
                        if save_matrix and matrix_output_dir:
                            matrix_output_file = matrix_output_dir / f"{chembl_id}_matrix.npy"
                            if not matrix_output_file.exists():
                                matrix = self.generate_embedding_matrix(smiles)
                                if matrix is not None:
                                    matrix_output_file = matrix_output_dir / f"{chembl_id}_matrix.npy"
                                    np.save(matrix_output_file, matrix)
                                    self.logger.debug(f"Saved matrix for {chembl_id}: {matrix.shape}")
                                else:
                                    self.logger.debug(f"No matrix available for {chembl_id}")

                        successes += 1
                        progress_logger.update()
                        continue

                    # Generate embedding
                    embedding = self.generate_embedding(smiles)

                    # Save
                    np.save(output_file, embedding)

                    # Also generate and save matrix if requested and model supports it
                    if save_matrix and matrix_output_dir:
                        matrix = self.generate_embedding_matrix(smiles)
                        if matrix is not None:
                            matrix_output_file = matrix_output_dir / f"{chembl_id}_matrix.npy"
                            np.save(matrix_output_file, matrix)
                            self.logger.debug(f"Saved matrix for {chembl_id}: {matrix.shape}")
                        else:
                            self.logger.debug(f"No matrix available for {chembl_id}")

                    successes += 1

                except Exception as e:
                    self.logger.error(f"Error processing {chembl_id}: {e}")
                    failures += 1

                progress_logger.update()
            
            progress_logger.finish()
            
            # Save output path
            self._output_path = output_dir
            
            self.logger.info(f"Ligand embeddings: {successes} successes, {failures} failures")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def get_output_path(self) -> Optional[Path]:
        """Return vector embeddings output path."""
        return getattr(self, '_output_path', None)
    
    def get_matrix_output_path(self) -> Optional[Path]:
        """
        Return embedding matrices output path.
        
        Note: SMI-TED does not support per-token matrices, so this
        directory will typically be empty for ligands.
        """
        return getattr(self, '_matrix_output_path', None)
    
    def get_embeddings_info(self) -> Dict[str, Any]:
        """Return information about generated embeddings."""
        output_path = self.get_output_path()
        matrix_output_path = self.get_matrix_output_path()
        
        info = {
            'output_path': None,
            'count': 0,
            'model': self.model_name,
            'dimension': self.embedding_dim,
            'matrix_output_path': None,
            'matrix_count': 0,
            'matrix_supported': self.model_name.upper() == "MOLFORMER"  # MoLFormer supports matrices
        }
        
        if output_path and output_path.exists():
            embedding_files = list(output_path.glob("*_embedding.npy"))
            info['output_path'] = str(output_path)
            info['count'] = len(embedding_files)
        
        if matrix_output_path and matrix_output_path.exists():
            matrix_files = list(matrix_output_path.glob("*_matrix.npy"))
            info['matrix_output_path'] = str(matrix_output_path)
            info['matrix_count'] = len(matrix_files)
        
        return info
    
    def build(self) -> Dict[str, Any]:
        """Build processing summary."""
        result = super().build()
        
        result.update({
            'fm4m_available': self.fm4m_available,
            'use_parallel': self.use_parallel,
            'checkpoint_enabled': self.checkpoint_enabled,
            'processed_files_count': len(self.processed_files),
        })
        
        return result
