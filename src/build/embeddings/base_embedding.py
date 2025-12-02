"""
Base interface for embedding generation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
from pathlib import Path

from src.build.core.base_builder import BaseBuilder
from src.build.core.exceptions import EmbeddingError, ModelLoadError, DependencyError
from src.build.utils import ProgressLogger, memory_monitor

class BaseEmbedding(BaseBuilder):
    """Abstract base class for embedding generation."""
    
    def __init__(self, model_name: str = None, config=None, **kwargs):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the model to use
            config: Build configuration
            **kwargs: Configuration arguments
        """
        # Ensure we have a model_name BEFORE calling super
        if model_name is None:
            model_name = "default"
        
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self._model_loaded = False
        
        # Now call super with everything defined
        super().__init__(config=config, **kwargs)
    
    def _validate_config(self) -> None:
        """Validate embedding-specific configuration."""
        if not self.model_name:
            raise EmbeddingError("Model name is required")
    
    def build(self) -> Any:
        """
        Build embeddings.
        
        Returns:
            Generated embeddings or model
        """
        try:
            return self._load_model()
        except Exception as e:
            self.logger.error(f"Embedding build failed: {e}")
            raise
    
    @abstractmethod
    def _load_model(self) -> Any:
        """
        Load specific model.
        Must be implemented by each subclass.
        
        Returns:
            Loaded model
        """
        pass
    
    @abstractmethod
    def _generate_single_embedding(self, input_data: str) -> np.ndarray:
        """
        Generate embedding for a single input.
        
        Args:
            input_data: Input data (sequence, SMILES, etc.)
            
        Returns:
            NumPy array with embedding
        """
        pass
    
    @abstractmethod
    def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Return supported models.
        
        Returns:
            Dictionary with models and their properties
        """
        pass
    
    def _do_initialize(self) -> None:
        """Embedding-specific initialization."""
        super()._do_initialize()
        
        # Check if model is supported
        supported_models = self.get_supported_models()
        if self.model_name not in supported_models:
            raise EmbeddingError(
                f"Unsupported model: {self.model_name}. "
                f"Available: {list(supported_models.keys())}"
            )
        
        # Load model
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.model = self._load_model()
            
            # Determine embedding dimension
            # Priority: 1) custom dimension via config, 2) model default dimension
            custom_dim = None
            if self.config:
                # Try to get custom dimension from config
                if hasattr(self.config, 'get'):
                    custom_dim = self.config.get('protein_dim') if 'protein' in self.__class__.__name__.lower() else self.config.get('ligand_dim')
                elif hasattr(self.config, 'protein_dim'):
                    custom_dim = self.config.protein_dim if 'protein' in self.__class__.__name__.lower() else self.config.ligand_dim
            
            if custom_dim is not None:
                self.embedding_dim = custom_dim
                self.logger.info(f"Model loaded - Custom dimension: {self.embedding_dim}")
            else:
                self.embedding_dim = supported_models[self.model_name]['dim']
                self.logger.info(f"Model loaded - Default dimension: {self.embedding_dim}")
            
            self._model_loaded = True
            
        except Exception as e:
            raise ModelLoadError(f"Error loading model {self.model_name}: {e}")
    
    def _do_cleanup(self) -> None:
        """Embedding-specific cleanup."""
        if self.model is not None:
            try:
                # Try to clean model from memory
                del self.model
                self.model = None
                self._model_loaded = False
                
                # Force garbage collection
                import gc
                gc.collect()
                
                self.logger.info("Model removed from memory")
            except Exception as e:
                self.logger.warning(f"Error during model cleanup: {e}")
        
        super()._do_cleanup()
    
    def generate_embedding(self, input_data: str) -> np.ndarray:
        """
        Generate embedding for individual input.
        
        Args:
            input_data: Input data
            
        Returns:
            NumPy array with embedding
        """
        if not self._model_loaded:
            raise EmbeddingError("Model not loaded. Run initialize() first.")
        
        if not input_data or not input_data.strip():
            raise EmbeddingError("Empty input data")
        
        try:
            return self._generate_single_embedding(input_data.strip())
        except Exception as e:
            raise EmbeddingError(f"Error generating embedding: {e}")
    
    @memory_monitor(threshold_percent=85.0)
    def generate_batch_embeddings(self, 
                                 input_list: List[str],
                                 batch_size: Optional[int] = None,
                                 show_progress: bool = True) -> List[np.ndarray]:
        """
        Generate embeddings for multiple inputs.
        
        Args:
            input_list: List of input data
            batch_size: Batch size (uses config if None)
            show_progress: Whether to show progress
            
        Returns:
            List of NumPy arrays with embeddings
        """
        if not self._model_loaded:
            raise EmbeddingError("Model not loaded. Run initialize() first.")
        
        if not input_list:
            return []
        
        # Use batch size from config if not specified
        if batch_size is None:
            batch_size = self.get_config('batch_size', 32)
        
        # Optimize batch size based on available memory
        from src.build.utils import optimize_batch_size
        batch_size = optimize_batch_size(batch_size)
        
        embeddings = []
        total_items = len(input_list)
        
        # Progress logger
        if show_progress:
            progress_logger = ProgressLogger(
                self.logger, 
                total_items, 
                f"Generating embeddings ({self.model_name})"
            )
        
        try:
            # Process in batches
            for i in range(0, total_items, batch_size):
                batch = input_list[i:i + batch_size]
                
                # Generate batch embeddings
                batch_embeddings = []
                for item in batch:
                    try:
                        embedding = self.generate_embedding(item)
                        batch_embeddings.append(embedding)
                    except Exception as e:
                        self.logger.warning(f"Error generating embedding for item {i}: {e}")
                        # Use zero embedding in case of error
                        zero_embedding = np.zeros(self.embedding_dim)
                        batch_embeddings.append(zero_embedding)
                
                embeddings.extend(batch_embeddings)
                
                # Update progress
                if show_progress:
                    progress_logger.update(len(batch))
                
                # Memory cleanup between batches
                if i % (batch_size * 10) == 0:  # Every 10 batches
                    import gc
                    gc.collect()
            
            if show_progress:
                progress_logger.finish()
            
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Error in batch processing: {e}")
    
    def process_file(self, 
                    input_file: Union[str, Path],
                    output_dir: Union[str, Path],
                    id_column: str = 'id',
                    data_column: str = 'sequence',
                    batch_size: Optional[int] = None) -> Tuple[int, int]:
        """
        Process file with data for embeddings.
        
        Args:
            input_file: Input file (TSV)
            output_dir: Output directory
            id_column: Name of ID column
            data_column: Name of data column
            batch_size: Batch size
            
        Returns:
            Tuple (successes, failures)
        """
        from src.build.utils import load_tsv, ensure_directory, save_numpy
        
        # Load data
        try:
            df = load_tsv(input_file)
            self.logger.info(f"Loaded {len(df)} records from {input_file}")
        except Exception as e:
            raise EmbeddingError(f"Error loading file {input_file}: {e}")
        
        # Check required columns
        if id_column not in df.columns:
            raise EmbeddingError(f"Column '{id_column}' not found")
        if data_column not in df.columns:
            raise EmbeddingError(f"Column '{data_column}' not found")
        
        # Prepare output
        output_path = ensure_directory(output_dir)
        
        # Generate embeddings
        data_list = df[data_column].tolist()
        id_list = df[id_column].tolist()
        
        embeddings = self.generate_batch_embeddings(
            data_list, 
            batch_size=batch_size
        )
        
        # Save individual embeddings
        successes = 0
        failures = 0
        
        for embedding_id, embedding in zip(id_list, embeddings):
            try:
                output_file = output_path / f"{embedding_id}.npy"
                save_numpy(embedding, output_file)
                successes += 1
            except Exception as e:
                self.logger.error(f"Error saving embedding {embedding_id}: {e}")
                failures += 1
        
        self.logger.info(f"Processing completed: {successes} successes, {failures} failures")
        return successes, failures
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Get embedding dimension."""
        return self.embedding_dim
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""
        supported_models = self.get_supported_models()
        if self.model_name in supported_models:
            return supported_models[self.model_name].copy()
        return {}
    
    def build(self) -> Dict[str, Any]:
        """
        Implementa método build da classe base.
        Para embeddings, retorna informações do modelo.
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'model_loaded': self._model_loaded,
            'model_info': self.get_model_info()
        }
