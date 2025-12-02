"""
Geração de embeddings de proteínas usando modelos ESM (Meta AI).
Utiliza código fonte local do ESM incluído no repositório.

REFATORADO: Agora usa Strategy Pattern para suportar múltiplos modelos.
"""

import os
import sys
import gc
from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING, Union
import numpy as np
from pathlib import Path

# Add local ESM to path
ESM_LOCAL_PATH = Path(__file__).parent.parent.parent.parent / "llm" / "ESM"
if str(ESM_LOCAL_PATH) not in sys.path:
    sys.path.insert(0, str(ESM_LOCAL_PATH))

# Pre-import ESM from local to avoid segfault issues with strategy loading
try:
    import esm as _esm_module
except ImportError:
    _esm_module = None

if TYPE_CHECKING:
    from src.build.core import BuildConfig

from src.build.embeddings.base_embedding import BaseEmbedding
from src.build.core.constants import ESM_MODELS, DEFAULT_ESM_MODEL
from src.build.core.exceptions import DependencyError, EmbeddingError, ModelLoadError
from src.build.utils import ProgressLogger, ensure_directory

# Importar factory e estratégias
from src.build.embeddings.factories.protein_model_factory import ProteinModelFactory
from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy

class ProteinEmbedding(BaseEmbedding):
    """
    Gerador de embeddings de proteínas usando ESM.
    
    REFATORADO: Atua como orchestrator que delega para estratégias específicas.
    - Usa ProteinModelFactory para criar estratégias (ESM2Strategy, ESM3Strategy)
    - Delega carregamento e inferência para a estratégia
    - Mantém API pública retrocompatível
    """
    
    def __init__(self, 
                 config: Optional['BuildConfig'] = None,
                 model_name: str = DEFAULT_ESM_MODEL,
                 use_gpu: bool = False,
                 **kwargs):
        """
        Inicializa gerador de embeddings de proteínas.
        
        Args:
            config: Build system configuration
            model_name: ESM model name
            use_gpu: Whether to use GPU when available
            **kwargs: Additional arguments
        """
        # Define attributes before parent initialization
        self.use_gpu = use_gpu
        self.device = None
        self.alphabet = None  # Will be configured by strategy
        self.batch_converter = None
        self.strategy: Optional[BaseProteinStrategy] = None  # Model strategy
            
        super().__init__(model_name=model_name, config=config, **kwargs)
        
        # Verificar dependências
        self._check_dependencies()
        
        # Criar estratégia apropriada usando factory
        self._create_strategy()
    
    def _check_dependencies(self) -> None:
        """Check if dependencies are available."""
        try:
            import torch
            self.torch = torch
            self.torch_available = True
        except ImportError:
            raise DependencyError(
                "PyTorch is not available. Install with: pip install torch"
            )
        
        try:
            # Import ESM from local source code
            import esm
            self.esm = esm
            self.esm_available = True
            self.logger.info(f"ESM loaded from local source code: {ESM_LOCAL_PATH}")
        except ImportError as e:
            raise DependencyError(
                f"ESM is not available in local repository ({ESM_LOCAL_PATH}). "
                f"Check if llm/ESM/ folder exists and contains source code. Error: {e}"
            )
    
    def _create_strategy(self) -> None:
        """
        Create appropriate strategy using factory.
        
        Raises:
            ValueError: If model is not supported
        """
        try:
            factory = ProteinModelFactory()
            self.strategy = factory.create_strategy(self.model_name)
            self.logger.info(f"✅ Strategy created: {self.strategy.__class__.__name__}")
        except ValueError as e:
            raise EmbeddingError(f"Failed to create strategy: {e}")
    
    def _validate_config(self) -> None:
        """Validate configuration specific to protein embeddings."""
        super()._validate_config()
        
        # Validate model
        if self.model_name not in ESM_MODELS:
            raise EmbeddingError(f"Invalid ESM model: {self.model_name}. Available models: {list(ESM_MODELS.keys())}")
        
        # Check GPU configuration (CUDA or MPS)
        if self.use_gpu:
            import torch
            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
            if not (has_cuda or has_mps):
                self.logger.warning("GPU requested but not available. Using CPU.")
                self.use_gpu = False
            elif has_mps and not has_cuda:
                self.logger.info("✨ Using MPS (Metal Performance Shaders) for GPU acceleration")
    
    def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Return supported ESM models."""
        return ESM_MODELS.copy()
    
    def _load_model(self) -> Any:
        """
        Load model using appropriate strategy.
        
        REFACTORED: Delegates to strategy.load() instead of direct code.
        """
        self.logger.info(f"Configuring device...")
        
        # Configure local cache for ESM models
        import os
        cache_dir = Path(__file__).parent.parent.parent.parent / "llm" / "models_cache" / "ESM"
        cache_dir.mkdir(parents=True, exist_ok=True)
        offload_folder = cache_dir / "offload"
        
        # Configure device (priority: CUDA > MPS > CPU)
        if self.use_gpu:
            if self.torch.cuda.is_available():
                self.device = self.torch.device("cuda")
                self.logger.info(f"Using CUDA GPU: {self.torch.cuda.get_device_name()}")
            elif hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
                self.device = self.torch.device("mps")
                self.logger.info("✨ Using MPS GPU (Metal Performance Shaders - Apple Silicon)")
            else:
                self.device = self.torch.device("cpu")
                self.logger.info("Using CPU")
        else:
            self.device = self.torch.device("cpu")
            self.logger.info("Using CPU")
        
        # Delegar carregamento para estratégia
        model, self.alphabet = self.strategy.load(
            model_name=self.model_name,
            device=self.device,
            offload_folder=str(offload_folder),
            logger=self.logger
        )
        
        # Configurar conversor de batch (compatibilidade retroativa)
        # NOTE: ESM-C and CLI-based strategies don't use batch_converter
        # - ESM-2: alphabet has get_batch_converter() method
        # - ESM-C: tokenizer is HuggingFace EsmSequenceTokenizer (no batch_converter)
        # - CLI strategies (Boltz, OpenFold): return (None, None)
        if self.alphabet is not None and hasattr(self.alphabet, 'get_batch_converter'):
            self.batch_converter = self.alphabet.get_batch_converter()
        else:
            self.batch_converter = None
            if self.alphabet is not None:
                self.logger.info("Estratégia ESM-C detectada (tokenizer HuggingFace, sem batch_converter)")
            else:
                self.logger.info("Estratégia CLI-based detectada (sem model/tokenizer em memória)")
        
        return model
    
    def _generate_single_embedding(self, sequence: str) -> np.ndarray:
        """
        Gera embedding para uma única sequência de proteína.
        
        REFATORADO: Delega para strategy.generate() ao invés de código direto.
        
        Args:
            sequence: Sequência de aminoácidos
            
        Returns:
            Array NumPy com embedding
        """
        # Delegar geração para estratégia
        # NOTE: Strategy signature is (model, auxiliary_objects, sequence, device, **kwargs)
        # - ESM strategies: auxiliary_objects = alphabet (batch converter)
        # - CLI strategies (Boltz): auxiliary_objects = None (CLI-based, no tokenizer needed)
        return self.strategy.generate(
            model=self.model,
            auxiliary_objects=self.alphabet,
            sequence=sequence,
            device=self.device,
            logger=self.logger
        )
    
    def generate_embedding_matrix(self, sequence: str) -> Optional[np.ndarray]:
        """
        Gera matriz de embeddings por token (sem pooling).
        
        Retorna representações para cada resíduo/token da sequência,
        preservando informação posicional para uso com arquiteturas
        como CNN + Cross-Attention.
        
        Args:
            sequence: Sequência de aminoácidos
            
        Returns:
            Array NumPy com shape [seq_len, embed_dim] ou None se não suportado
            
        Note:
            Nem todas as estratégias suportam extração de matriz.
            Verifique se o retorno não é None antes de usar.
        """
        # Garantir que modelo está carregado
        if not self._model_loaded:
            self._do_initialize()
        
        # Verificar se estratégia suporta generate_matrix
        if self.strategy is None:
            self.logger.warning("Nenhuma estratégia configurada")
            return None
        
        if not hasattr(self.strategy, 'generate_matrix'):
            self.logger.warning(
                f"Estratégia {self.strategy.__class__.__name__} não suporta generate_matrix()"
            )
            return None
        
        # Delegar para estratégia
        try:
            matrix = self.strategy.generate_matrix(
                model=self.model,
                auxiliary_objects=self.alphabet,
                sequence=sequence,
                device=self.device,
                logger=self.logger
            )
            
            if matrix is not None:
                self.logger.debug(
                    f"Matriz gerada: shape={matrix.shape}, dtype={matrix.dtype}"
                )
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar matriz de embedding: {e}")
            return None
    
    def process_fasta_file(self,
                          fasta_file: Path,
                          output_dir: Path,
                          batch_size: Optional[int] = None) -> Tuple[int, int]:
        """
        Processa arquivo FASTA com sequências de proteínas.
        
        Args:
            fasta_file: Input FASTA file
            output_dir: Output directory
            batch_size: Batch size
            
        Returns:
            Tuple (successes, failures)
        """
        sequences = self._read_fasta_file(fasta_file)
        
        if not sequences:
            raise EmbeddingError(f"No sequences found in {fasta_file}")
        
        self.logger.info(f"Processing {len(sequences)} sequences from {fasta_file}")
        
        # Prepare output
        output_path = ensure_directory(output_dir)
        
        successes = 0
        failures = 0
        
        # Progress logger
        progress_logger = ProgressLogger(
            self.logger,
            len(sequences),
            "Processing FASTA sequences"
        )
        
        for seq_id, sequence in sequences:
            try:
                # Check if already exists
                output_file = output_path / f"{seq_id}_embedding.npy"
                if output_file.exists():
                    self.logger.debug(f"Embedding already exists, skipping: {seq_id}")
                    successes += 1
                    progress_logger.update()
                    continue
                
                # Generate embedding
                embedding = self.generate_embedding(sequence)
                
                # Save
                np.save(output_file, embedding)
                successes += 1
                
            except Exception as e:
                self.logger.error(f"Error processing sequence {seq_id}: {e}")
                failures += 1
            
            progress_logger.update()
        
        progress_logger.finish()
        self.logger.info(f"FASTA processing completed: {successes} successes, {failures} failures")
        
        return successes, failures
    
    def _read_fasta_file(self, fasta_file: Path) -> List[Tuple[str, str]]:
        """
        Read FASTA file and return list of (ID, sequence).
        
        Args:
            fasta_file: Arquivo FASTA
            
        Returns:
            Lista de tuplas (ID, sequência)
        """
        sequences = []
        current_id = None
        current_sequence = []
        
        try:
            with open(fasta_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    if line.startswith('>'):
                        # Salvar sequência anterior se existir
                        if current_id and current_sequence:
                            sequences.append((current_id, ''.join(current_sequence)))
                        
                        # Nova sequência
                        current_id = line[1:].split()[0]  # Primeira palavra após >
                        current_sequence = []
                        
                    elif line and current_id:
                        current_sequence.append(line)
                
                # Salvar última sequência
                if current_id and current_sequence:
                    sequences.append((current_id, ''.join(current_sequence)))
                    
        except Exception as e:
            raise EmbeddingError(f"Erro ao ler arquivo FASTA {fasta_file}: {e}")
        
        return sequences
    
    def process_sequence_directory(self,
                                  seq_input_dir: Path,
                                  output_dir: Path,
                                  batch_size: Optional[int] = None) -> Tuple[int, int]:
        """
        Process directory with sequence files.
        
        Args:
            seq_input_dir: Directory with sequence files
            output_dir: Output directory
            batch_size: Batch size
            
        Returns:
            Tuple (successes, failures)
        """
        seq_dir = Path(seq_input_dir)
        
        if not seq_dir.exists():
            raise EmbeddingError(f"Directory not found: {seq_input_dir}")
        
        # Find sequence files
        sequence_files = list(seq_dir.glob("*.fasta")) + list(seq_dir.glob("*.txt"))
        
        if not sequence_files:
            raise EmbeddingError(f"No sequence files found in {seq_input_dir}")
        
        self.logger.info(f"Found {len(sequence_files)} sequence files")
        
        total_successes = 0
        total_failures = 0
        
        for seq_file in sequence_files:
            try:
                if seq_file.suffix.lower() == '.fasta':
                    successes, failures = self.process_fasta_file(seq_file, output_dir, batch_size)
                else:
                    # Process as plain text file
                    successes, failures = self._process_text_file(seq_file, output_dir)
                
                total_successes += successes
                total_failures += failures
                
            except Exception as e:
                self.logger.error(f"Error processing file {seq_file}: {e}")
                total_failures += 1
        
        return total_successes, total_failures
    
    def _process_text_file(self, text_file: Path, output_dir: Path) -> Tuple[int, int]:
        """Process plain text file with sequence."""
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Join lines that don't start with >
            sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
            
            if not sequence:
                self.logger.warning(f"Empty file or no sequence: {text_file}")
                return 0, 1
            
            # File name without extension
            seq_id = text_file.stem
            
            # Generate embedding
            embedding = self.generate_embedding(sequence)
            
            # Salvar
            output_path = ensure_directory(output_dir)
            output_file = output_path / f"{seq_id}_embedding.npy"
            np.save(output_file, embedding)
            
            return 1, 0
            
        except Exception as e:
            self.logger.error(f"Erro ao processar arquivo de texto {text_file}: {e}")
            return 0, 1
    
    def get_memory_usage_estimate(self, sequence_length: int) -> float:
        """
        Estima uso de memória para sequência.
        
        Args:
            sequence_length: Comprimento da sequência
            
        Returns:
            Estimativa de uso em GB
        """
        # Estimativa baseada no modelo e comprimento da sequência
        base_model_memory = 3.0  # GB para modelo base
        sequence_memory = sequence_length * 0.001  # Aproximação
        
        return base_model_memory + sequence_memory
    
    def generate_embeddings(self, 
                          tsv_path: Path, 
                          output_dir: Optional[Path] = None,
                          save_matrix: bool = False,
                          matrix_output_dir: Optional[Path] = None) -> bool:
        """
        Gera embeddings a partir de arquivo TSV (interface para pipeline).
        
        Args:
            tsv_path: Arquivo TSV com dados
            output_dir: Diretório de saída para vetores (usa config se None)
            save_matrix: Se True, também salva matrizes de embedding [seq_len, dim]
            matrix_output_dir: Diretório para matrizes (usa 'protein_matrix_embeddings' se None)
            
        Returns:
            True if successful
        """
        import pandas as pd
        from src.build.utils import ensure_directory
        from src.build.core.constants import BuildConstants
        
        try:
            # Ensure model is initialized
            if not self._model_loaded:
                self.logger.info("Initializing ESM model...")
                self._do_initialize()
            
            # Determine output directory for vectors
            if output_dir is None:
                output_dir = Path(self.get_config('protein_output_dir', 'protein_embeddings'))
            
            output_dir = Path(output_dir)
            output_dir = ensure_directory(output_dir)
            
            # Determine output directory for matrices (if enabled)
            if save_matrix:
                if matrix_output_dir is None:
                    matrix_output_dir = Path(
                        self.get_config(
                            'protein_matrix_output_dir', 
                            BuildConstants.DEFAULT_PROTEIN_MATRIX_OUTPUT_DIR
                        )
                    )
                matrix_output_dir = Path(matrix_output_dir)
                matrix_output_dir = ensure_directory(matrix_output_dir)
                self.logger.info(f"📊 Saving embedding matrices to: {matrix_output_dir}")
            
            # Load TSV
            self.logger.info(f"Loading data from {tsv_path}")
            df = pd.read_csv(tsv_path, sep='\t')
            
            # Check required columns
            if 'seq_id' not in df.columns or 'seq' not in df.columns:
                raise EmbeddingError("TSV must contain columns 'seq_id' and 'seq'")
            
            # Get unique sequences
            unique_seqs = df.groupby('seq_id')['seq'].first()
            self.logger.info(f"Processing {len(unique_seqs)} unique sequences")
            
            # Process each sequence
            successes = 0
            failures = 0
            matrix_successes = 0
            
            progress_logger = ProgressLogger(
                self.logger,
                len(unique_seqs),
                "Generating protein embeddings"
            )
            
            for seq_id, sequence in unique_seqs.items():
                try:
                    # Check if vector already exists
                    output_file = output_dir / f"{seq_id}_embedding.npy"
                    vector_exists = output_file.exists()
                    
                    # Check if matrix already exists (if save_matrix enabled)
                    matrix_exists = False
                    if save_matrix:
                        matrix_file = matrix_output_dir / f"{seq_id}_matrix.npy"
                        matrix_exists = matrix_file.exists()
                    
                    # Skip if everything already exists
                    if vector_exists and (not save_matrix or matrix_exists):
                        self.logger.debug(f"Embedding(s) already exist(s): {seq_id}")
                        successes += 1
                        if save_matrix and matrix_exists:
                            matrix_successes += 1
                        progress_logger.update()
                        continue
                    
                    # Generate vector embedding (if doesn't exist)
                    if not vector_exists:
                        embedding = self.generate_embedding(sequence)
                        np.save(output_file, embedding)
                    
                    # Generate embedding matrix (if enabled and doesn't exist)
                    if save_matrix and not matrix_exists:
                        matrix = self.generate_embedding_matrix(sequence)
                        if matrix is not None:
                            np.save(matrix_file, matrix)
                            matrix_successes += 1
                            self.logger.debug(
                                f"Matrix saved: {seq_id} shape={matrix.shape}"
                            )
                        else:
                            self.logger.warning(
                                f"Strategy does not support matrix for: {seq_id}"
                            )
                    
                    successes += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing {seq_id}: {e}")
                    failures += 1
                
                progress_logger.update()
            
            progress_logger.finish()
            
            # Save output paths
            self._output_path = output_dir
            if save_matrix:
                self._matrix_output_path = matrix_output_dir
            
            # Log summary
            self.logger.info(f"Protein embeddings: {successes} successes, {failures} failures")
            if save_matrix:
                self.logger.info(f"📊 Embedding matrices: {matrix_successes} generated")
            
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
        """Return embedding matrices output path."""
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
            'matrix_count': 0
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
            'device': str(self.device) if self.device else None,
            'use_gpu': self.use_gpu,
            'torch_available': self.torch_available,
            'esm_available': self.esm_available,
        })
        
        return result
    
    def __del__(self):
        """
        Cleanup when object is destroyed.
        
        REFACTORED: Delegates to strategy.cleanup() instead of direct code.
        """
        if hasattr(self, 'strategy') and self.strategy:
            try:
                self.strategy.cleanup(self.model, self.alphabet)
            except:
                pass  # Ignore errors in destructor
