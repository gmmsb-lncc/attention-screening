"""
Modular Embedding Pipeline

Main pipeline orchestrating all modular components for embedding generation.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple
import pandas as pd

from src.build.embeddings.core import DataManager, ModelManager, EmbeddingGenerator
from src.build.embeddings.models import ModelRegistry
from src.build.embeddings.utils import CacheManager, validate_protein_batch, validate_smiles_batch


class EmbeddingPipeline:
    """
    Orchestrates the complete embedding generation pipeline.
    
    This is the main interface for generating embeddings in a modular way.
    
    Features:
    - End-to-end pipeline: load → validate → generate → cache → save
    - Support for both proteins (ESM) and ligands (FM4M)
    - Automatic caching and validation
    - Progress tracking
    - Memory management
    
    Example:
        >>> pipeline = EmbeddingPipeline(use_gpu=True)
        >>> embeddings = pipeline.generate_protein_embeddings(
        ...     sequences=['MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL'],
        ...     model_name='esm2_t33_650M_UR50D'
        ... )
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        cache_dir: Optional[Path] = None,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """
        Initialize EmbeddingPipeline.
        
        Args:
            use_gpu: Whether to use GPU if available
            cache_dir: Directory for caching embeddings
            batch_size: Batch size for processing
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.batch_size = batch_size
        
        if self.verbose:
            print("\n" + "="*80)
            print("🚀 Initializing Modular Embedding Pipeline")
            print("="*80)
        
        # Initialize components
        self.data_manager = DataManager(verbose=verbose)
        self.model_manager = ModelManager(use_gpu=use_gpu, verbose=verbose)
        self.cache_manager = CacheManager(cache_dir=cache_dir, verbose=verbose)
        self.generator = EmbeddingGenerator(
            model_manager=self.model_manager,
            batch_size=batch_size,
            verbose=verbose
        )
        
        if self.verbose:
            print(f"\n✅ Pipeline initialized successfully")
            print(f"   Device: {self.model_manager.device}")
            print(f"   Batch size: {batch_size}")
            if cache_dir:
                print(f"   Cache: {cache_dir}")
    
    def generate_protein_embeddings(
        self,
        source: Union[str, Path, List[str], pd.DataFrame],
        model_name: str = 'esm2_t33_650M_UR50D',
        repr_layer: Optional[int] = None,
        sequence_column: str = 'sequence',
        id_column: Optional[str] = None,
        validate: bool = True,
        use_cache: bool = True,
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Generate protein embeddings using ESM models.
        
        Args:
            source: Source of sequences (file, list, or DataFrame)
            model_name: ESM model name
            repr_layer: Representation layer (None = use default)
            sequence_column: Column name for sequences (if DataFrame/CSV)
            id_column: Column name for IDs (if DataFrame/CSV)
            validate: Whether to validate sequences
            use_cache: Whether to use caching
            output_path: Path to save embeddings (optional)
            
        Returns:
            NumPy array of embeddings
        """
        if self.verbose:
            print("\n" + "="*80)
            print("🧬 PROTEIN EMBEDDING GENERATION")
            print("="*80)
            print(f"Model: {model_name}")
            if repr_layer:
                print(f"Representation Layer: {repr_layer}")
        
        # Step 1: Load sequences
        if self.verbose:
            print("\n📂 Step 1: Loading sequences...")
        
        sequences, ids = self.data_manager.load_sequences(
            source,
            sequence_column=sequence_column,
            id_column=id_column
        )
        
        # Step 2: Validate sequences
        if validate:
            if self.verbose:
                print("\n✓ Step 2: Validating sequences...")
            
            valid_seqs, valid_indices = validate_protein_batch(
                sequences,
                verbose=self.verbose
            )
            
            if len(valid_seqs) < len(sequences) and self.verbose:
                print(f"   ⚠️  Removed {len(sequences) - len(valid_seqs)} invalid sequences")
            
            sequences = valid_seqs
            ids = [ids[i] for i in valid_indices]
        
        if not sequences:
            raise ValueError("No valid sequences to process")
        
        # Step 3: Check cache
        if use_cache:
            if self.verbose:
                print("\n💾 Step 3: Checking cache...")
            
            repr_layer_use = repr_layer or ModelRegistry.get_repr_layer(model_name)
            
            cached = self.cache_manager.load_embeddings(
                sequences=sequences,
                model_name=model_name,
                model_type='esm',
                repr_layer=repr_layer_use
            )
            
            if cached is not None:
                if self.verbose:
                    print("   ✅ Loaded from cache!")
                return cached
        
        # Step 4: Generate embeddings
        if self.verbose:
            print("\n🔮 Step 4: Generating embeddings...")
        
        repr_layer_use = repr_layer or ModelRegistry.get_repr_layer(model_name)
        
        embeddings = self.generator.generate_esm_embeddings(
            sequences=sequences,
            model_name=model_name,
            repr_layer=repr_layer_use,
            show_progress=True
        )
        
        # Validate embeddings for NaN/Inf
        nan_count = np.isnan(embeddings).sum()
        inf_count = np.isinf(embeddings).sum()
        if nan_count > 0 or inf_count > 0:
            if self.verbose:
                print(f"   ⚠️  Warning: Embeddings contain {nan_count} NaN and {inf_count} Inf values")
                print(f"      This may indicate issues with input sequences or model")
        
        # Step 5: Cache embeddings
        if use_cache:
            if self.verbose:
                print("\n💾 Step 5: Caching embeddings...")
            
            self.cache_manager.save_embeddings(
                embeddings=embeddings,
                sequences=sequences,
                model_name=model_name,
                model_type='esm',
                metadata={'ids': ids},
                repr_layer=repr_layer_use
            )
        
        # Step 6: Save to file
        if output_path:
            if self.verbose:
                print(f"\n💾 Step 6: Saving to {output_path}...")
            
            self._save_embeddings(embeddings, ids, output_path)
        
        if self.verbose:
            print("\n" + "="*80)
            print("✅ PROTEIN EMBEDDING GENERATION COMPLETE")
            print("="*80)
            print(f"Generated {len(embeddings)} embeddings")
            print(f"Embedding dimension: {embeddings.shape[1]}")
        
        return embeddings
    
    def generate_ligand_embeddings(
        self,
        source: Union[str, List[str]],
        model_name: str = 'smi_ted_light',  # Only Light model available
        batch_size: int = 100,
        output_file: Optional[str] = None,
        return_dict: bool = False,
        use_cache: bool = True,
        smiles_column: str = 'SMILES',
        id_column: Optional[str] = None,
        validate: bool = True
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate ligand embeddings using FM4M models.
        
        Args:
            source: Source of SMILES (file, list, or DataFrame)
            model_name: FM4M model name (default: 'smi_ted_light' - only model available)
            smiles_column: Column name for SMILES (if DataFrame/CSV)
            id_column: Column name for IDs (if DataFrame/CSV)
            validate: Whether to validate SMILES
            use_cache: Whether to use caching
            output_file: Path to save embeddings (optional)
            
        Returns:
            NumPy array of embeddings
        """
        # Convert output_file to Path if provided
        output_path = Path(output_file) if output_file else None
        
        if self.verbose:
            print("\n" + "="*80)
            print("💊 LIGAND EMBEDDING GENERATION")
            print("="*80)
            print(f"Model: {model_name}")
        
        # Step 1: Load SMILES
        if self.verbose:
            print("\n📂 Step 1: Loading SMILES...")
        
        smiles_list, ids = self.data_manager.load_smiles(
            source,
            smiles_column=smiles_column,
            id_column=id_column
        )
        
        # Step 2: Validate SMILES
        if validate:
            if self.verbose:
                print("\n✓ Step 2: Validating SMILES...")
            
            valid_smiles, valid_indices = validate_smiles_batch(
                smiles_list,
                verbose=self.verbose
            )
            
            if len(valid_smiles) < len(smiles_list) and self.verbose:
                print(f"   ⚠️  Removed {len(smiles_list) - len(valid_smiles)} invalid SMILES")
            
            smiles_list = valid_smiles
            ids = [ids[i] for i in valid_indices]
        
        if not smiles_list:
            raise ValueError("No valid SMILES to process")
        
        # Step 3: Check cache
        if use_cache:
            if self.verbose:
                print("\n💾 Step 3: Checking cache...")
            
            cached = self.cache_manager.load_embeddings(
                sequences=smiles_list,
                model_name=model_name,
                model_type='fm4m'
            )
            
            if cached is not None:
                if self.verbose:
                    print("   ✅ Loaded from cache!")
                return cached
        
        # Step 4: Generate embeddings
        if self.verbose:
            print("\n🔮 Step 4: Generating embeddings...")
        
        embeddings = self.generator.generate_fm4m_embeddings(
            smiles_list=smiles_list,
            model_name=model_name,
            show_progress=True
        )
        
        # Validate embeddings for NaN (FM4M limitation with complex SMILES)
        nan_count = np.isnan(embeddings).sum()
        if nan_count > 0:
            nan_rows = np.isnan(embeddings).any(axis=1).sum()
            if self.verbose:
                print(f"   ⚠️  Warning: {nan_rows} embeddings contain NaN values")
                print(f"      This is a known FM4M limitation with complex SMILES")
                print(f"      Consider using simpler SMILES or filtering these molecules")
        
        # Step 5: Cache embeddings
        if use_cache:
            if self.verbose:
                print("\n💾 Step 5: Caching embeddings...")
            
            self.cache_manager.save_embeddings(
                embeddings=embeddings,
                sequences=smiles_list,
                model_name=model_name,
                model_type='fm4m',
                metadata={'ids': ids}
            )
        
        # Step 6: Save to file
        if output_path:
            if self.verbose:
                print(f"\n💾 Step 6: Saving to {output_path}...")
            
            self._save_embeddings(embeddings, ids, output_path)
        
        if self.verbose:
            print("\n" + "="*80)
            print("✅ LIGAND EMBEDDING GENERATION COMPLETE")
            print("="*80)
            print(f"Generated {len(embeddings)} embeddings")
            print(f"Embedding dimension: {embeddings.shape[1]}")
        
        return embeddings
    
    def _save_embeddings(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        output_path: Path
    ):
        """Save embeddings to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.npy':
            np.save(output_path, embeddings)
        elif output_path.suffix == '.npz':
            np.savez(output_path, embeddings=embeddings, ids=ids)
        elif output_path.suffix == '.csv':
            df = pd.DataFrame(embeddings)
            df.insert(0, 'id', ids)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
        
        if self.verbose:
            print(f"   ✅ Saved embeddings to {output_path}")
    
    def list_available_models(self, model_type: Optional[str] = None):
        """
        List available models.
        
        Args:
            model_type: 'esm' or 'fm4m' (None = all)
        """
        ModelRegistry.print_models(model_type)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information and statistics.
        
        Returns:
            Dictionary with cache statistics including size and entry count
        """
        return self.cache_manager.get_cache_info()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get detailed cache statistics.
        
        Returns:
            Dictionary with:
            - total_entries: Number of cached embeddings
            - memory_cache_size: Size of in-memory cache
            - disk_cache_size: Size of disk cache (if available)
            - cache_hit_rate: Percentage of cache hits (if tracked)
        """
        stats = self.get_cache_info()
        
        # Add additional statistics
        if hasattr(self.cache_manager, 'cache_hits') and hasattr(self.cache_manager, 'cache_misses'):
            total = self.cache_manager.cache_hits + self.cache_manager.cache_misses
            hit_rate = (self.cache_manager.cache_hits / total * 100) if total > 0 else 0
            stats['cache_hit_rate'] = f"{hit_rate:.1f}%"
            stats['cache_hits'] = self.cache_manager.cache_hits
            stats['cache_misses'] = self.cache_manager.cache_misses
        
        return stats
    
    def clear_cache(self):
        """Clear all caches."""
        self.cache_manager.clear_all()
        self.model_manager.clear_cache()
        if self.verbose:
            print("   ✅ All caches cleared")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EmbeddingPipeline(\n"
            f"  device={self.model_manager.device},\n"
            f"  batch_size={self.batch_size},\n"
            f"  cache={self.cache_manager.cache_dir}\n"
            f")"
        )


# Convenience function
def generate_embeddings(
    sequences: Union[List[str], str, Path],
    embedding_type: str = 'protein',
    **kwargs
) -> np.ndarray:
    """
    Convenience function for quick embedding generation.
    
    Args:
        sequences: Sequences/SMILES or path to file
        embedding_type: 'protein' or 'ligand'
        **kwargs: Additional arguments for the pipeline
        
    Returns:
        NumPy array of embeddings
    
    Example:
        >>> embeddings = generate_embeddings(['MKTAYIAK...'], embedding_type='protein')
    """
    pipeline = EmbeddingPipeline(**kwargs)
    
    if embedding_type == 'protein':
        return pipeline.generate_protein_embeddings(sequences)
    elif embedding_type == 'ligand':
        return pipeline.generate_ligand_embeddings(sequences)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
