"""
Concrete implementation of strategy for ESM-2 models.
Contains all ESM-2 specific logic for loading, inference and cleanup.
"""

import os
import gc
from pathlib import Path
from typing import Tuple, Any, Optional
import numpy as np
import torch

from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy
from src.build.core.constants import ESM_MODELS
from src.build.core.exceptions import ModelLoadError, EmbeddingError


class ESM2Strategy(BaseProteinStrategy):
    """
    Implementation strategy for ESM-2 models (Meta AI).
    
    Features:
    - Supports models from 8M to 15B parameters
    - Automatic CPU offloading for large models (3B, 15B)
    - Mean pooling over sequence (does not use CLS token)
    - Optimized memory management (gc + empty_cache)
    - Supports CUDA, MPS (Apple Silicon) and CPU
    
    Responsibilities:
    - Load ESM-2 model with appropriate configurations
    - Generate embeddings with automatic truncation
    - Manage GPU/CPU memory efficiently
    """
    
    def __init__(self):
        """Initialize ESM-2 strategy."""
        self.logger = None  # Will be configured when needed
        self._cache_dir = None
        self._offload_folder = None
    
    def load(
        self, 
        model_name: str, 
        device: torch.device,
        offload_folder: Optional[str] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load ESM-2 model with optional CPU offloading.
        
        Args:
            model_name: Model name (e.g., "esm2_t48_15B_UR50D")
            device: PyTorch device (cuda/cpu/mps)
            offload_folder: Custom folder for offloading (optional)
            **kwargs: Additional parameters (logger, etc.)
            
        Returns:
            Tuple (model, alphabet) where alphabet is the ESM-2 tokenizer
            
        Raises:
            ValueError: If model is not a valid ESM-2
            ModelLoadError: If failed to load model
        """
        # Configure logger if provided
        self.logger = kwargs.get('logger')
        
        # Validate model
        if model_name not in ESM_MODELS:
            raise ValueError(
                f"ESM-2 model '{model_name}' not found. "
                f"Available models: {list(ESM_MODELS.keys())}"
            )
        
        # Configure local cache
        self._setup_cache_dirs(offload_folder)
        
        # Configure CUDA memory (new variable since PyTorch 2.0)
        os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Determine if CPU offloading is needed
        large_models = ['esm2_t48_15B_UR50D', 'esm2_t36_3B_UR50D']
        needs_offload = model_name in large_models and str(device) == 'cuda'
        
        try:
            if self.logger:
                self.logger.info(f"Loading ESM-2 model: {model_name}")
                self.logger.info(f"Model cache: {self._cache_dir}")
            
            # Import ESM - try local first, then installed
            esm = self._import_esm()
            
            if needs_offload:
                model, alphabet = self._load_with_offloading(esm, model_name, device)
            else:
                model, alphabet = self._load_standard(esm, model_name, device)
            
            model = model.eval()
            
            if self.logger:
                self.logger.info("✅ ESM-2 model loaded successfully")
            
            return model, alphabet
            
        except ImportError as e:
            raise ModelLoadError(
                f"ESM is not available. Error: {e}\n"
                f"Check if the ESM/ folder exists in the repository."
            )
        except Exception as e:
            self._handle_load_error(e, model_name)
    
    def generate(
        self,
        model: Any,
        auxiliary_objects: Any,
        sequence: str,
        device: torch.device,
        **kwargs
    ) -> np.ndarray:
        """
        Generate ESM-2 embedding with mean pooling.
        
        Args:
            model: Loaded ESM-2 model
            auxiliary_objects: Alphabet (tokenizer) from ESM-2
            sequence: Amino acid sequence
            device: PyTorch device
            **kwargs: Optional parameters (logger, etc.)
            
        Returns:
            Embedding numpy array (shape: [embedding_dim])
            
        Raises:
            EmbeddingError: If failed to generate embedding
        """
        alphabet = auxiliary_objects
        self.logger = kwargs.get('logger', self.logger)
        
        # Validate sequence
        if not sequence or not sequence.strip():
            raise EmbeddingError("Empty sequence")
        
        # Clean sequence (only valid amino acids)
        clean_sequence = ''.join(
            c for c in sequence.upper() 
            if c in 'ACDEFGHIKLMNPQRSTVWY'
        )
        
        if not clean_sequence:
            raise EmbeddingError("Sequence contains no valid amino acids")
        
        # Truncate if necessary
        max_len = self.get_max_length(model.args.arch if hasattr(model, 'args') else model.__class__.__name__)
        
        if len(clean_sequence) > max_len:
            if self.logger:
                self.logger.warning(
                    f"Sequence truncated: {len(clean_sequence)} → {max_len} aa"
                )
            clean_sequence = clean_sequence[:max_len]
        
        try:
            # Prepare tokens
            batch_converter = alphabet.get_batch_converter()
            batch_labels, batch_strs, batch_tokens = batch_converter(
                [("sequence", clean_sequence)]
            )
            batch_tokens = batch_tokens.to(device)
            
            # Inference
            with torch.no_grad():
                results = model(
                    batch_tokens,
                    repr_layers=[model.num_layers],
                    return_contacts=False
                )
                
                # Extract embeddings (remove special BOS/EOS tokens)
                token_representations = results["representations"][model.num_layers]
                embedding = token_representations[0, 1:-1]  # [seq_len, embed_dim]
                
                # Mean pooling over the sequence
                sequence_embedding = embedding.mean(dim=0)  # [embed_dim]
                
                # Move to CPU
                result = sequence_embedding.cpu().numpy()
            
            # CRITICAL memory cleanup
            del batch_tokens, results, token_representations, embedding, sequence_embedding
            gc.collect()
            
            if str(device) == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return result
            
        except Exception as e:
            # Clean memory even on error
            if str(device) == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            raise EmbeddingError(f"Error generating ESM-2 embedding: {e}")
    
    def get_max_length(self, model_name: str) -> int:
        """Return maximum sequence length for the model."""
        return ESM_MODELS.get(model_name, {}).get('max_len', 1024)
    
    def get_embedding_dim(self, model_name: str) -> int:
        """Return embedding dimension for the model."""
        return ESM_MODELS.get(model_name, {}).get('dim', 1280)
    
    def cleanup(self, model: Any, auxiliary_objects: Any) -> None:
        """
        Libera recursos de memória.
        
        Args:
            model: Modelo a ser limpo (não usado atualmente)
            auxiliary_objects: Alphabet (não usado atualmente)
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate_matrix(
        self,
        model: Any,
        auxiliary_objects: Any,
        sequence: str,
        device: torch.device,
        **kwargs
    ) -> np.ndarray:
        """
        Generate ESM-2 embedding matrix per-token (no pooling).
        
        Args:
            model: Loaded ESM-2 model
            auxiliary_objects: Alphabet (tokenizer) from ESM-2
            sequence: Amino acid sequence
            device: PyTorch device
            **kwargs: Optional parameters (logger, etc.)
            
        Returns:
            Numpy array matrix (shape: [seq_len, embedding_dim])
            
        Raises:
            EmbeddingError: If failed to generate embedding
        """
        alphabet = auxiliary_objects
        self.logger = kwargs.get('logger', self.logger)
        
        # Validate sequence
        if not sequence or not sequence.strip():
            raise EmbeddingError("Empty sequence")
        
        # Clean sequence (only valid amino acids)
        clean_sequence = ''.join(
            c for c in sequence.upper() 
            if c in 'ACDEFGHIKLMNPQRSTVWY'
        )
        
        if not clean_sequence:
            raise EmbeddingError("Sequence contains no valid amino acids")
        
        # Truncate if necessary
        max_len = self.get_max_length(model.args.arch if hasattr(model, 'args') else model.__class__.__name__)
        
        if len(clean_sequence) > max_len:
            if self.logger:
                self.logger.warning(
                    f"Sequence truncated: {len(clean_sequence)} → {max_len} aa"
                )
            clean_sequence = clean_sequence[:max_len]
        
        try:
            # Prepare tokens
            batch_converter = alphabet.get_batch_converter()
            batch_labels, batch_strs, batch_tokens = batch_converter(
                [("sequence", clean_sequence)]
            )
            batch_tokens = batch_tokens.to(device)
            
            # Inference
            with torch.no_grad():
                results = model(
                    batch_tokens,
                    repr_layers=[model.num_layers],
                    return_contacts=False
                )
                
                # Extract embeddings (remove special BOS/EOS tokens)
                token_representations = results["representations"][model.num_layers]
                embedding_matrix = token_representations[0, 1:-1]  # [seq_len, embed_dim]
                
                # Move to CPU - NO POOLING (keep complete matrix)
                result = embedding_matrix.cpu().numpy()
            
            # CRITICAL memory cleanup
            del batch_tokens, results, token_representations, embedding_matrix
            gc.collect()
            
            if str(device) == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return result
            
        except Exception as e:
            # Clean memory even on error
            if str(device) == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            raise EmbeddingError(f"Error generating ESM-2 embedding matrix: {e}")
    
    def supports_matrix_output(self) -> bool:
        """ESM-2 supports per-token matrix generation."""
        return True
    
    # ===== Private Methods =====
    
    def _setup_cache_dirs(self, offload_folder: Optional[str] = None) -> None:
        """Configure cache and offload directories."""
        # Cache principal
        self._cache_dir = Path(__file__).parent.parent.parent.parent.parent / "llm" / "models_cache" / "ESM"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['TORCH_HOME'] = str(self._cache_dir)
        
        # Pasta de offload
        if offload_folder:
            self._offload_folder = Path(offload_folder)
        else:
            self._offload_folder = self._cache_dir / "offload"
        self._offload_folder.mkdir(parents=True, exist_ok=True)
    
    def _import_esm(self) -> Any:
        """
        Import ESM module with automatic fallback.
        
        Order of attempts:
        1. ESM local in llm/ESM/ folder (preferred to avoid pip segfaults on some systems)
        2. ESM installed via pip (fair-esm) as fallback
        
        Returns:
            Imported ESM module
            
        Raises:
            ModelLoadError: If ESM is not available
        """
        import sys
        
        # Attempt 1: Check if ESM is already imported and valid
        if 'esm' in sys.modules:
            esm = sys.modules['esm']
            if hasattr(esm, 'pretrained'):
                if self.logger:
                    self.logger.info(f"✅ ESM already loaded from: {getattr(esm, '__file__', 'unknown')}")
                return esm
        
        # Attempt 2: ESM local in llm/ESM/ folder (preferred)
        # Calculate path relative to project (works on any machine)
        project_root = Path(__file__).parent.parent.parent.parent.parent
        esm_local_path = project_root / "llm" / "ESM"
        
        if esm_local_path.exists():
            esm_str = str(esm_local_path)
            if esm_str not in sys.path:
                sys.path.insert(0, esm_str)
            
            try:
                import esm
                if hasattr(esm, 'pretrained'):
                    if self.logger:
                        self.logger.info(f"✅ ESM loaded from local: {esm_local_path}")
                    return esm
            except ImportError as e:
                if self.logger:
                    self.logger.warning(f"⚠️ Failed to load local ESM: {e}")
        
        # Attempt 3: ESM installed via pip (fallback)
        try:
            import esm
            if hasattr(esm, 'pretrained'):
                if self.logger:
                    self.logger.info("✅ ESM loaded from pip package (fair-esm)")
                return esm
        except ImportError:
            pass
        
        # No ESM available - clear instructions for user
        raise ModelLoadError(
            "ESM module not available.\n\n"
            "SOLUTIONS:\n"
            "1. Clone the ESM repository (RECOMMENDED):\n"
            "   git clone https://github.com/facebookresearch/esm.git ESM\n\n"
            "2. Or install via pip:\n"
            "   pip install fair-esm\n\n"
            "For more info: https://github.com/facebookresearch/esm"
        )
    
    def _load_with_offloading(
        self, 
        esm_module: Any, 
        model_name: str, 
        device: torch.device
    ) -> Tuple[Any, Any]:
        """
        Carrega modelo com CPU offloading para modelos grandes.
        
        Args:
            esm_module: Módulo ESM importado
            model_name: Nome do modelo
            device: Dispositivo PyTorch
            
        Returns:
            Tuple (model, alphabet)
        """
        try:
            from accelerate import dispatch_model, infer_auto_device_map
            from accelerate.utils import get_balanced_memory
            
            if self.logger:
                self.logger.info(f"🔄 Modelo grande detectado: {model_name}")
                self.logger.info("🔄 Ativando CPU offloading...")
            
            # Carregar modelo sem mover para device
            model, alphabet = esm_module.pretrained.load_model_and_alphabet(model_name)
            
            # Calcular memória balanceada (20GB GPU + 30GB CPU)
            max_memory = get_balanced_memory(
                model,
                max_memory={0: "20GiB", "cpu": "30GiB"},
                no_split_module_classes=["TransformerLayer"]
            )
            
            # Criar device map automático
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=["TransformerLayer"]
            )
            
            # Dispatch para múltiplos devices
            # NOTE: offload_folder foi renomeado para offload_dir no accelerate >= 1.0
            try:
                # Tentar com offload_dir (novo)
                model = dispatch_model(
                    model,
                    device_map=device_map,
                    offload_dir=str(self._offload_folder)
                )
            except TypeError:
                try:
                    # Fallback para offload_folder (antigo)
                    model = dispatch_model(
                        model,
                        device_map=device_map,
                        offload_folder=str(self._offload_folder)
                    )
                except TypeError:
                    # Sem offload (último recurso)
                    model = dispatch_model(
                        model,
                        device_map=device_map
                    )
            
            if self.logger:
                self.logger.info("✅ CPU offloading ativado")
                self.logger.info(f"   Offload dir: {self._offload_folder}")
            
            return model, alphabet
            
        except ImportError:
            if self.logger:
                self.logger.warning("⚠️  accelerate não encontrado. Carregando sem offloading...")
            return self._load_standard(esm_module, model_name, device)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️  Falha no offloading: {e}")
                self.logger.warning("   Tentando carregamento padrão...")
            return self._load_standard(esm_module, model_name, device)
    
    def _load_standard(
        self, 
        esm_module: Any, 
        model_name: str, 
        device: torch.device
    ) -> Tuple[Any, Any]:
        """
        Carregamento padrão sem offloading.
        
        Args:
            esm_module: Módulo ESM importado
            model_name: Nome do modelo
            device: Dispositivo PyTorch
            
        Returns:
            Tuple (model, alphabet)
        """
        model, alphabet = esm_module.pretrained.load_model_and_alphabet(model_name)
        model = model.to(device)
        return model, alphabet
    
    def _handle_load_error(self, error: Exception, model_name: str) -> None:
        """
        Trata erros de carregamento de forma específica.
        
        Args:
            error: Exceção capturada
            model_name: Nome do modelo
            
        Raises:
            ModelLoadError: Com mensagem apropriada
        """
        error_msg = str(error).lower()
        
        # Erro 404 / arquivo não encontrado
        if "404" in error_msg or "not found" in error_msg or "could not load" in error_msg:
            raise ModelLoadError(
                f"Modelo ESM-2 '{model_name}' não está disponível.\n"
                f"URL esperada: https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt\n"
                f"Erro: {error}\n\n"
                f"SOLUÇÕES:\n"
                f"1. Verifique o nome do modelo\n"
                f"2. Modelo 15B pode não estar público (baixe manualmente)\n"
                f"3. Use modelo menor: esm2_t36_3B_UR50D ou esm2_t33_650M_UR50D"
            )
        
        # Erro de memória
        elif "out of memory" in error_msg or "oom" in error_msg:
            raise ModelLoadError(
                f"Memória insuficiente para '{model_name}'.\n"
                f"Erro: {error}\n\n"
                f"SOLUÇÕES:\n"
                f"1. Use modelo menor\n"
                f"2. Aumente memória disponível\n"
                f"3. Instale accelerate para CPU offloading"
            )
        
        # Erro genérico
        else:
            raise ModelLoadError(
                f"Falha ao carregar modelo ESM-2 '{model_name}'.\n"
                f"Cache: {self._cache_dir}\n"
                f"Erro: {type(error).__name__}: {error}"
            )
