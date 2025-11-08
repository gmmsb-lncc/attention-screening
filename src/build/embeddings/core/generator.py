"""
Embedding Generator

Handles the actual generation of embeddings using loaded models.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm


class EmbeddingGenerator:
    """
    Generates embeddings from sequences/SMILES using loaded models.
    
    Features:
    - Batch processing
    - Progress tracking
    - Memory-efficient generation
    - Support for both ESM and FM4M models
    """
    
    def __init__(
        self,
        model_manager,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """
        Initialize EmbeddingGenerator.
        
        Args:
            model_manager: ModelManager instance with loaded models
            batch_size: Batch size for processing
            verbose: Whether to print progress information
        """
        self.model_manager = model_manager
        self.batch_size = batch_size
        self.verbose = verbose
    
    def generate_esm_embeddings(
        self,
        sequences: List[str],
        model_name: str = 'esm2_t33_650M_UR50D',
        repr_layer: int = 33,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate ESM embeddings for protein sequences.
        
        Args:
            sequences: List of protein sequences
            model_name: ESM model name
            repr_layer: Representation layer to use
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings (n_sequences, embedding_dim)
        """
        if self.verbose:
            print(f"\n🧬 Generating ESM embeddings for {len(sequences)} sequences")
        
        # Load model
        model, alphabet = self.model_manager.load_esm_model(model_name, repr_layer)
        batch_converter = alphabet.get_batch_converter()
        
        # Prepare batches
        all_embeddings = []
        
        # Create iterator
        n_batches = (len(sequences) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(sequences), self.batch_size)
        
        if show_progress and self.verbose:
            iterator = tqdm(iterator, total=n_batches, desc="   Processing batches")
        
        with torch.no_grad():
            for i in iterator:
                batch_sequences = sequences[i:i + self.batch_size]
                
                # Prepare batch data
                batch_labels = [f"seq_{j}" for j in range(len(batch_sequences))]
                batch_data = list(zip(batch_labels, batch_sequences))
                
                # Convert batch
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
                batch_tokens = batch_tokens.to(self.model_manager.device)
                
                # Generate embeddings
                results = model(batch_tokens, repr_layers=[repr_layer])
                
                # Extract representations (mean pooling over sequence length)
                token_representations = results["representations"][repr_layer]
                
                # Mean pooling (excluding padding tokens)
                for j, (label, seq) in enumerate(zip(batch_labels, batch_strs)):
                    # Get sequence length (excluding BOS and EOS tokens)
                    seq_len = len(seq)
                    # Mean pool over sequence (tokens 1 to seq_len+1)
                    embedding = token_representations[j, 1:seq_len+1].mean(0)
                    all_embeddings.append(embedding.cpu().numpy())
        
        embeddings_array = np.vstack(all_embeddings)
        
        if self.verbose:
            print(f"   ✅ Generated embeddings shape: {embeddings_array.shape}")
        
        return embeddings_array
    
    def generate_fm4m_embeddings(
        self,
        smiles_list: List[str],
        model_name: str = 'default',
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate FM4M embeddings for SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            model_name: FM4M model identifier
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings (n_smiles, embedding_dim)
        """
        if self.verbose:
            print(f"\n💊 Generating FM4M embeddings for {len(smiles_list)} molecules")
        
        # Load model
        model = self.model_manager.load_fm4m_model(model_name)
        
        # FM4M models use batch encoding for efficiency
        # The encode method handles batching internally
        try:
            if self.verbose:
                print(f"   Encoding {len(smiles_list)} molecules with SMI-TED...")
            
            # Encode all SMILES at once (model handles batching)
            embeddings = model.encode(
                smiles_list,
                useCuda=torch.cuda.is_available(),
                batch_size=100,
                return_torch=False  # Return numpy arrays
            )
            
            # Convert to numpy if needed
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            elif not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # Ensure 2D array (n_samples, embedding_dim)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            if self.verbose:
                print(f"   ✅ Generated embeddings shape: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Error encoding SMILES: {e}")
            raise RuntimeError(f"Failed to generate FM4M embeddings: {e}")
    
    def _generate_fm4m_embeddings_individual(
        self,
        smiles_list: List[str],
        model_name: str,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate FM4M embeddings one-by-one (fallback method).
        
        Args:
            smiles_list: List of SMILES strings
            model_name: FM4M model name
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings (n_samples, embedding_dim)
        """
        # Load model
        model = self.model_manager.load_fm4m_model(model_name)
        
        # Generate embeddings one by one
        all_embeddings = []
        
        # Create iterator
        iterator = smiles_list
        if show_progress and self.verbose:
            iterator = tqdm(smiles_list, desc="   Processing molecules")
        
        for smiles in iterator:
            try:
                # Generate embedding for single SMILES using encode
                embedding = model.encode(
                    [smiles],  # encode expects a list
                    useCuda=torch.cuda.is_available(),
                    return_torch=False
                )
                
                # Handle different return types
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                elif not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                
                # Ensure 1D array
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
                
                all_embeddings.append(embedding)
                
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  Failed to generate embedding for SMILES: {smiles[:50]}...")
                    print(f"      Error: {e}")
                # Add zero vector as placeholder
                all_embeddings.append(np.zeros(768))  # FM4M default dimension
        
        embeddings_array = np.vstack(all_embeddings)
        
        if self.verbose:
            print(f"   ✅ Generated embeddings shape: {embeddings_array.shape}")
        
        return embeddings_array
    
    def generate_with_ids(
        self,
        sequences: List[str],
        ids: List[str],
        embedding_type: str = 'protein',
        **kwargs
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings and keep track of IDs.
        
        Args:
            sequences: List of sequences/SMILES
            ids: List of identifiers
            embedding_type: 'protein' or 'ligand'
            **kwargs: Additional arguments for generation
            
        Returns:
            Tuple of (embeddings, ids)
        """
        if len(sequences) != len(ids):
            raise ValueError("Number of sequences must match number of IDs")
        
        if embedding_type == 'protein':
            embeddings = self.generate_esm_embeddings(sequences, **kwargs)
        elif embedding_type == 'ligand':
            embeddings = self.generate_fm4m_embeddings(sequences, **kwargs)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        return embeddings, ids
    
    def get_embedding_dim(self, model_type: str, model_name: str) -> int:
        """
        Get embedding dimension for a model.
        
        Args:
            model_type: 'esm' or 'fm4m'
            model_name: Model name/identifier
            
        Returns:
            Embedding dimension
        """
        if model_type == 'esm':
            model, _ = self.model_manager.load_esm_model(model_name)
            return model.embed_dim
        elif model_type == 'fm4m':
            return 768  # FM4M fixed dimension
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EmbeddingGenerator(batch_size={self.batch_size}, "
            f"device={self.model_manager.device})"
        )
