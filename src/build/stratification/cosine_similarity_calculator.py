"""
Cosine similarity calculator for high-dimensional embeddings.

This module provides efficient cosine similarity computation for
molecular embeddings, with support for batch processing and
memory-efficient operations on large datasets.
"""

import numpy as np
from typing import Union, Optional, Tuple, List
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity
from src.build.core.base_builder import BaseBuilder
from src.build.core.config import BuildConfig
from src.build.core.exceptions import BuildException


class CosineSimilarityCalculator(BaseBuilder):
    """
    Efficient cosine similarity calculator for molecular embeddings.
    
    Supports both single and batch processing of high-dimensional
    protein-ligand embedding vectors.
    """
    
    def __init__(self, 
                 config: Optional[BuildConfig] = None,
                 normalize: bool = True,
                 batch_size: Optional[int] = None,
                 **kwargs):
        """
        Initialize cosine similarity calculator.

        Args:
            config: Build configuration
            normalize: Whether to normalize embeddings before similarity computation
            batch_size: Batch size for memory-efficient processing
            **kwargs: Additional configuration options
        """
        # Set attributes before calling parent constructor
        self.normalize = normalize
        self.batch_size = batch_size or 1000  # Default batch size
        self.similarity_matrix = None
        
        super().__init__(config, **kwargs)
        
    def _validate_config(self) -> None:
        """Validate configuration."""
        # Ensure batch size is positive
        if self.batch_size <= 0:
            raise BuildException("Batch size must be positive")
        
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit vectors.

        Args:
            embeddings: Input embedding matrix of shape (n_samples, n_features)

        Returns:
            Normalized embedding matrix
        """
        if not self.normalize:
            return embeddings
        
        # L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def calculate_single_pair(self, 
                             embedding1: np.ndarray, 
                             embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity value between -1 and 1
        """
        # Normalize embeddings if required
        emb1 = self._normalize_embeddings(embedding1.reshape(1, -1))
        emb2 = self._normalize_embeddings(embedding2.reshape(1, -1))
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2.T)[0, 0]
        return float(similarity)
    
    def calculate_batch(self, 
                       embeddings: np.ndarray, 
                       batch_size: Optional[int] = None) -> np.ndarray:
        """
        Calculate all-pairs cosine similarity for embeddings in batch mode.

        Args:
            embeddings: Embedding matrix of shape (n_samples, n_features)
            batch_size: Size of processing batches (default: self.batch_size)

        Returns:
            Similarity matrix of shape (n_samples, n_samples)
        """
        batch_size = batch_size or self.batch_size
        
        n_samples = embeddings.shape[0]
        if n_samples == 0:
            return np.array([])
        
        # Normalize embeddings
        normalized_embeddings = self._normalize_embeddings(embeddings)
        
        # For small datasets, compute all similarities at once
        if n_samples <= batch_size:
            similarity_matrix = cosine_similarity(normalized_embeddings)
            # Ensure diagonal is exactly 1.0 for self-similarity (due to floating point precision)
            np.fill_diagonal(similarity_matrix, 1.0)
            self.similarity_matrix = similarity_matrix
            return similarity_matrix
        
        # For larger datasets, use batch processing to manage memory
        similarity_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            batch_embeddings = normalized_embeddings[i:end_i]
            
            # Compute similarities with all embeddings
            batch_similarities = cosine_similarity(batch_embeddings, normalized_embeddings)
            similarity_matrix[i:end_i, :] = batch_similarities
            
            # Ensure diagonal is exactly 1.0 for self-similarity within the batch
            for diag_idx in range(i, min(end_i, similarity_matrix.shape[0])):
                similarity_matrix[diag_idx, diag_idx] = 1.0
            
            self.logger.debug(f"Processed batch {i//batch_size + 1}/{(n_samples-1)//batch_size + 1}")
        
        self.similarity_matrix = similarity_matrix
        return similarity_matrix
    
    def calculate_similarity_with_indices(self, 
                                        embeddings: np.ndarray, 
                                        indices1: np.ndarray, 
                                        indices2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate cosine similarity between specific pairs of embeddings.

        Args:
            embeddings: Embedding matrix of shape (n_samples, n_features)
            indices1: First set of indices
            indices2: Second set of indices (if None, uses indices1 for square matrix)

        Returns:
            Similarity matrix of shape (len(indices1), len(indices2))
        """
        # Normalize embeddings
        normalized_embeddings = self._normalize_embeddings(embeddings)
        
        # Extract embeddings for specified indices
        emb1 = normalized_embeddings[indices1]
        
        if indices2 is None:
            # Square matrix with same set of embeddings
            similarity = cosine_similarity(emb1)
        else:
            emb2 = normalized_embeddings[indices2]
            similarity = cosine_similarity(emb1, emb2)
        
        return similarity
    
    def calculate_similarity_threshold(self, 
                                    embeddings: np.ndarray, 
                                    threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find embedding pairs with similarity above threshold.

        Args:
            embeddings: Embedding matrix of shape (n_samples, n_features)
            threshold: Similarity threshold

        Returns:
            Tuple of (indices1, indices2) for pairs above threshold
        """
        if embeddings.shape[0] <= self.batch_size:
            # Compute all similarities at once for smaller datasets
            sim_matrix = self.calculate_batch(embeddings)
            indices1, indices2 = np.where(sim_matrix >= threshold)
        else:
            # For larger datasets, use batch processing
            indices1, indices2 = [], []
            
            normalized_embeddings = self._normalize_embeddings(embeddings)
            n_samples = embeddings.shape[0]
            
            for i in range(0, n_samples, self.batch_size):
                end_i = min(i + self.batch_size, n_samples)
                batch_emb1 = normalized_embeddings[i:end_i]
                
                # Compute similarities with all embeddings
                batch_similarities = cosine_similarity(batch_emb1, normalized_embeddings)
                
                # Find pairs above threshold
                batch_indices1, batch_indices2 = np.where(batch_similarities >= threshold)
                batch_indices1 += i  # Adjust for batch offset
                
                indices1.extend(batch_indices1)
                indices2.extend(batch_indices2)
        
        return np.array(indices1), np.array(indices2)
    
    def find_similar_embeddings(self, 
                              query_embedding: np.ndarray, 
                              reference_embeddings: np.ndarray, 
                              threshold: float = 0.8,
                              top_k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find embeddings similar to a query embedding.

        Args:
            query_embedding: Query embedding vector
            reference_embeddings: Reference embedding matrix
            threshold: Minimum similarity threshold
            top_k: Return top-k most similar embeddings (if None, returns all above threshold)

        Returns:
            Tuple of (indices, similarities) for similar embeddings
        """
        # Normalize embeddings
        query_norm = self._normalize_embeddings(query_embedding.reshape(1, -1))
        ref_norm = self._normalize_embeddings(reference_embeddings)
        
        # Calculate similarities
        similarities = cosine_similarity(query_norm, ref_norm)[0]
        
        # Find embeddings above threshold
        above_threshold = similarities >= threshold
        indices = np.where(above_threshold)[0]
        filtered_similarities = similarities[above_threshold]
        
        # Sort by similarity (descending)
        sort_idx = np.argsort(filtered_similarities)[::-1]
        indices = indices[sort_idx]
        filtered_similarities = filtered_similarities[sort_idx]
        
        # Limit to top_k if specified
        if top_k is not None:
            indices = indices[:top_k]
            filtered_similarities = filtered_similarities[:top_k]
        
        return indices, filtered_similarities
    
    def calculate_multi_view_similarity(self,
                                      protein_embeddings: np.ndarray,
                                      ligand_embeddings: np.ndarray,
                                      protein_weight: float = 0.6,
                                      ligand_weight: float = 0.4,
                                      normalize_weights: bool = True) -> np.ndarray:
        """
        Calculate combined similarity considering both protein and ligand spaces.

        Args:
            protein_embeddings: Protein embedding matrix
            ligand_embeddings: Ligand embedding matrix
            protein_weight: Weight for protein similarity
            ligand_weight: Weight for ligand similarity
            normalize_weights: Whether to normalize weights to sum to 1

        Returns:
            Combined similarity matrix
        """
        if normalize_weights:
            total_weight = protein_weight + ligand_weight
            protein_weight /= total_weight
            ligand_weight /= total_weight
        
        # Calculate individual similarities
        protein_sim = self.calculate_batch(protein_embeddings)
        ligand_sim = self.calculate_batch(ligand_embeddings)
        
        # Combine with weights
        combined_sim = protein_weight * protein_sim + ligand_weight * ligand_sim
        
        return combined_sim
    
    def get_memory_usage_estimate(self, n_samples: int, n_features: int) -> float:
        """
        Estimate memory usage for similarity matrix computation.

        Args:
            n_samples: Number of samples
            n_features: Number of features per sample

        Returns:
            Estimated memory usage in GB
        """
        # Similarity matrix requires n_samples^2 * 4 bytes (float32)
        matrix_memory_gb = (n_samples * n_samples * 4) / (1024**3)
        
        # Original embeddings require n_samples * n_features * 4 bytes
        embeddings_memory_gb = (n_samples * n_features * 4) / (1024**3)
        
        # Total estimated memory usage
        total_memory_gb = matrix_memory_gb + embeddings_memory_gb
        
        return total_memory_gb

    def build(self) -> dict:
        """
        Build method for BaseBuilder compatibility.

        Returns:
            Dictionary with calculator information
        """
        result = {
            'normalize': self.normalize,
            'batch_size': self.batch_size,
            'similarity_matrix_available': self.similarity_matrix is not None
        }
        
        if self.similarity_matrix is not None:
            result['similarity_matrix_shape'] = self.similarity_matrix.shape
        
        return result