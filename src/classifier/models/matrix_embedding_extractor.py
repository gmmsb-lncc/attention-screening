"""
Matrix Embedding Extractor for CNN + Cross-Attention Pipeline.

This module provides high-level interface to extract matrix embeddings
[seq_len, dim] from protein sequences and ligand SMILES for use with
the CrossAttentionAffinityModel.

Author: DockTKinase Team
Date: 2025
"""

import gc
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import torch

from src.build.embeddings.protein_embedding import ProteinEmbedding
from src.build.embeddings.ligand_embedding import LigandEmbedding


class MatrixEmbeddingExtractor:
    """
    Extractor for matrix embeddings suitable for CNN + Cross-Attention model.
    
    This class provides a unified interface to extract:
    - Protein embeddings: [seq_len, protein_dim] from ESM-2
    - Ligand embeddings: [n_tokens, ligand_dim] from SMILES tokenization
    
    Attributes:
        protein_model: Name of ESM model (e.g., 'esm2_t36_3B_UR50D')
        ligand_model: Name of ligand model (e.g., 'SMI-TED')
        protein_dim: Dimension of protein embeddings
        ligand_dim: Dimension of ligand embeddings
        device: Device for computation (cuda/mps/cpu)
    """
    
    # ESM-2 model dimensions
    ESM2_DIMS = {
        'esm2_t6_8M_UR50D': 320,
        'esm2_t12_35M_UR50D': 480,
        'esm2_t30_150M_UR50D': 640,
        'esm2_t33_650M_UR50D': 1280,
        'esm2_t36_3B_UR50D': 2560,
        'esm2_t48_15B_UR50D': 5120,
    }
    
    def __init__(
        self,
        protein_model: str = 'esm2_t6_8M_UR50D',  # Use smaller model for testing
        ligand_model: str = 'SMI-TED',
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the matrix embedding extractor.
        
        Args:
            protein_model: ESM-2 model name
            ligand_model: Ligand embedding model name
            device: Device string ('cuda', 'mps', 'cpu') or None for auto-detect
            logger: Optional logger instance
        """
        self.protein_model_name = protein_model
        self.ligand_model_name = ligand_model
        self.logger = logger or logging.getLogger(__name__)
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Set dimensions
        self.protein_dim = self.ESM2_DIMS.get(protein_model, 1280)
        self.ligand_dim = 768  # SMI-TED embedding dimension
        
        # Lazy-load models
        self._protein_embedder: Optional[ProteinEmbedding] = None
        self._ligand_embedder: Optional[LigandEmbedding] = None
        
        self.logger.info(
            f"MatrixEmbeddingExtractor initialized:\n"
            f"  Protein model: {protein_model} (dim={self.protein_dim})\n"
            f"  Ligand model: {ligand_model} (dim={self.ligand_dim})\n"
            f"  Device: {self.device}"
        )
    
    @property
    def protein_embedder(self) -> ProteinEmbedding:
        """Lazy-load protein embedding model."""
        if self._protein_embedder is None:
            self.logger.info(f"Loading protein model: {self.protein_model_name}")
            self._protein_embedder = ProteinEmbedding(
                model_name=self.protein_model_name,
                use_gpu=(self.device in ['cuda', 'mps'])
            )
            self._protein_embedder.initialize()
        return self._protein_embedder
    
    @property
    def ligand_embedder(self) -> LigandEmbedding:
        """Lazy-load ligand embedding model."""
        if self._ligand_embedder is None:
            self.logger.info(f"Loading ligand model: {self.ligand_model_name}")
            self._ligand_embedder = LigandEmbedding(
                model_name=self.ligand_model_name
            )
            self._ligand_embedder.initialize()
        return self._ligand_embedder
    
    def extract_protein_matrix(self, sequence: str) -> np.ndarray:
        """
        Extract protein embedding matrix [seq_len, dim].
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Numpy array with shape [seq_len, protein_dim]
        """
        try:
            matrix = self.protein_embedder.generate_embedding_matrix(sequence)
            
            if matrix is None:
                # Fallback: expand vector embedding
                self.logger.warning(
                    "Matrix extraction not supported, using vector expansion"
                )
                vector = self.protein_embedder.generate_embedding(sequence)
                # Expand to [seq_len, dim] - repeat vector for each residue
                matrix = np.tile(vector, (len(sequence), 1))
            
            self.logger.debug(f"Protein matrix shape: {matrix.shape}")
            return matrix.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error extracting protein matrix: {e}")
            raise
    
    def extract_ligand_matrix(self, smiles: str) -> np.ndarray:
        """
        Extract ligand embedding matrix [n_tokens, dim].
        
        Since SMI-TED doesn't support per-token embeddings, we create
        a pseudo-matrix by tokenizing SMILES and creating embeddings
        for each token.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Numpy array with shape [n_tokens, ligand_dim]
        """
        try:
            # SMI-TED doesn't support per-token embeddings
            # Option 1: Try matrix extraction (may return None)
            matrix = self.ligand_embedder.generate_embedding_matrix(smiles)
            
            if matrix is None:
                # Option 2: Create pseudo-matrix from global embedding
                self.logger.debug(
                    "Ligand matrix not supported, creating pseudo-matrix"
                )
                
                # Get global embedding
                vector = self.ligand_embedder.generate_embedding(smiles)
                
                # Tokenize SMILES for length estimation
                # Simple tokenization: each character is a token
                n_tokens = len(smiles)
                
                # Create pseudo-matrix: [n_tokens, dim]
                # Use learned positional encoding by varying the embedding slightly
                matrix = self._create_positional_matrix(vector, n_tokens)
            
            self.logger.debug(f"Ligand matrix shape: {matrix.shape}")
            return matrix.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error extracting ligand matrix: {e}")
            raise
    
    def _create_positional_matrix(
        self, 
        base_vector: np.ndarray, 
        n_positions: int
    ) -> np.ndarray:
        """
        Create a pseudo-matrix with positional information.
        
        Uses sinusoidal position encoding to add positional
        information to the base embedding.
        
        Args:
            base_vector: Base embedding vector [dim]
            n_positions: Number of positions (tokens)
            
        Returns:
            Matrix with shape [n_positions, dim]
        """
        dim = len(base_vector)
        
        # Create sinusoidal position encodings
        position = np.arange(n_positions)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
        
        pe = np.zeros((n_positions, dim))
        pe[:, 0::2] = np.sin(position * div_term)
        if dim % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        
        # Combine base vector with position encoding
        # Scale position encoding to not dominate
        alpha = 0.1  # Position encoding weight
        matrix = np.tile(base_vector, (n_positions, 1)) + alpha * pe
        
        return matrix
    
    def extract_pair(
        self,
        sequence: str,
        smiles: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract both protein and ligand matrices.
        
        Args:
            sequence: Amino acid sequence
            smiles: SMILES string
            
        Returns:
            Tuple (protein_matrix, ligand_matrix)
        """
        protein_matrix = self.extract_protein_matrix(sequence)
        ligand_matrix = self.extract_ligand_matrix(smiles)
        return protein_matrix, ligand_matrix
    
    def batch_extract_and_save(
        self,
        data: List[Dict[str, str]],
        protein_output_dir: Path,
        ligand_output_dir: Path,
        id_column: str = 'id',
        sequence_column: str = 'sequence',
        smiles_column: str = 'smiles'
    ) -> Tuple[int, int]:
        """
        Extract and save embeddings for a batch of protein-ligand pairs.
        
        Args:
            data: List of dicts with id, sequence, and smiles
            protein_output_dir: Directory for protein matrices
            ligand_output_dir: Directory for ligand matrices
            id_column: Column name for sample ID
            sequence_column: Column name for sequence
            smiles_column: Column name for SMILES
            
        Returns:
            Tuple (successful, failed) counts
        """
        protein_output_dir = Path(protein_output_dir)
        ligand_output_dir = Path(ligand_output_dir)
        protein_output_dir.mkdir(parents=True, exist_ok=True)
        ligand_output_dir.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        failed = 0
        
        for i, item in enumerate(data):
            sample_id = item.get(id_column, f"sample_{i}")
            sequence = item.get(sequence_column, "")
            smiles = item.get(smiles_column, "")
            
            if not sequence or not smiles:
                self.logger.warning(f"Skipping {sample_id}: missing sequence or SMILES")
                failed += 1
                continue
            
            try:
                # Extract matrices
                protein_matrix, ligand_matrix = self.extract_pair(sequence, smiles)
                
                # Save to files
                protein_path = protein_output_dir / f"{sample_id}.npy"
                ligand_path = ligand_output_dir / f"{sample_id}.npy"
                
                np.save(protein_path, protein_matrix)
                np.save(ligand_path, ligand_matrix)
                
                successful += 1
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(data)} samples")
                    
            except Exception as e:
                self.logger.error(f"Error processing {sample_id}: {e}")
                failed += 1
        
        self.logger.info(
            f"Batch extraction complete: {successful} successful, {failed} failed"
        )
        return successful, failed
    
    def cleanup(self):
        """Release model resources."""
        if self._protein_embedder is not None:
            try:
                self._protein_embedder.cleanup()
            except Exception:
                pass
            self._protein_embedder = None
        
        if self._ligand_embedder is not None:
            try:
                self._ligand_embedder.cleanup()
            except Exception:
                pass
            self._ligand_embedder = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("MatrixEmbeddingExtractor resources released")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()


def create_synthetic_embeddings(
    n_samples: int,
    protein_seq_len: int = 100,
    ligand_seq_len: int = 50,
    protein_dim: int = 2560,
    ligand_dim: int = 768,
    output_protein_dir: Optional[Path] = None,
    output_ligand_dir: Optional[Path] = None,
    seed: int = 42
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Create synthetic matrix embeddings for testing.
    
    Args:
        n_samples: Number of samples to generate
        protein_seq_len: Length of protein sequences
        ligand_seq_len: Length of ligand sequences  
        protein_dim: Dimension of protein embeddings
        ligand_dim: Dimension of ligand embeddings
        output_protein_dir: Optional dir to save protein matrices
        output_ligand_dir: Optional dir to save ligand matrices
        seed: Random seed
        
    Returns:
        Tuple (protein_matrices, ligand_matrices, sample_ids)
    """
    np.random.seed(seed)
    
    protein_matrices = []
    ligand_matrices = []
    sample_ids = []
    
    # Create output directories if specified
    if output_protein_dir:
        output_protein_dir = Path(output_protein_dir)
        output_protein_dir.mkdir(parents=True, exist_ok=True)
    if output_ligand_dir:
        output_ligand_dir = Path(output_ligand_dir)
        output_ligand_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(n_samples):
        sample_id = f"synthetic_{i:05d}"
        
        # Vary sequence lengths slightly
        prot_len = protein_seq_len + np.random.randint(-20, 21)
        lig_len = ligand_seq_len + np.random.randint(-10, 11)
        prot_len = max(10, prot_len)
        lig_len = max(5, lig_len)
        
        # Generate random matrices with some structure
        protein_matrix = np.random.randn(prot_len, protein_dim).astype(np.float32) * 0.1
        ligand_matrix = np.random.randn(lig_len, ligand_dim).astype(np.float32) * 0.1
        
        protein_matrices.append(protein_matrix)
        ligand_matrices.append(ligand_matrix)
        sample_ids.append(sample_id)
        
        # Save if directories specified
        if output_protein_dir:
            np.save(output_protein_dir / f"{sample_id}.npy", protein_matrix)
        if output_ligand_dir:
            np.save(output_ligand_dir / f"{sample_id}.npy", ligand_matrix)
    
    return protein_matrices, ligand_matrices, sample_ids


# Convenience function for quick extraction
def extract_matrix_embeddings(
    sequences: List[str],
    smiles_list: List[str],
    protein_model: str = 'esm2_t6_8M_UR50D',
    ligand_model: str = 'SMI-TED',
    device: Optional[str] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Quick function to extract matrix embeddings for multiple pairs.
    
    Args:
        sequences: List of amino acid sequences
        smiles_list: List of SMILES strings
        protein_model: ESM-2 model name
        ligand_model: Ligand model name
        device: Device string or None for auto-detect
        
    Returns:
        Tuple (protein_matrices, ligand_matrices)
    """
    extractor = MatrixEmbeddingExtractor(
        protein_model=protein_model,
        ligand_model=ligand_model,
        device=device
    )
    
    try:
        protein_matrices = []
        ligand_matrices = []
        
        for seq, smi in zip(sequences, smiles_list):
            prot_mat, lig_mat = extractor.extract_pair(seq, smi)
            protein_matrices.append(prot_mat)
            ligand_matrices.append(lig_mat)
        
        return protein_matrices, ligand_matrices
        
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Testing MatrixEmbeddingExtractor...")
    
    # Test synthetic embeddings
    print("\n1. Testing synthetic embedding generation...")
    prot_mats, lig_mats, ids = create_synthetic_embeddings(
        n_samples=5,
        protein_dim=320,  # Small model dim
        ligand_dim=768
    )
    print(f"   Generated {len(prot_mats)} protein matrices")
    print(f"   First protein shape: {prot_mats[0].shape}")
    print(f"   First ligand shape: {lig_mats[0].shape}")
    
    print("\n✓ MatrixEmbeddingExtractor module ready!")
