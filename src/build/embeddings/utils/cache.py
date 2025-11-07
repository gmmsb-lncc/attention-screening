"""
Cache Manager for Embeddings

Handles caching of generated embeddings to disk and memory.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List
import hashlib


class CacheManager:
    """
    Manages caching of embeddings to improve performance.
    
    Features:
    - Save embeddings to disk
    - Load cached embeddings
    - Cache validation
    - Memory cache
    - Automatic cache invalidation
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_memory_cache: bool = True,
        verbose: bool = True
    ):
        """
        Initialize CacheManager.
        
        Args:
            cache_dir: Directory for disk cache (None = no disk cache)
            use_memory_cache: Whether to use in-memory cache
            verbose: Whether to print progress information
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_memory_cache = use_memory_cache
        self.verbose = verbose
        
        # Memory cache
        self._memory_cache: Dict[str, np.ndarray] = {}
        
        # Create cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print(f"   💾 Cache directory: {self.cache_dir}")
    
    def _generate_cache_key(
        self,
        sequences: List[str],
        model_name: str,
        model_type: str,
        **kwargs
    ) -> str:
        """
        Generate unique cache key for a set of embeddings.
        
        Args:
            sequences: List of sequences/SMILES
            model_name: Model name
            model_type: Model type ('esm' or 'fm4m')
            **kwargs: Additional parameters
            
        Returns:
            Cache key string
        """
        # Create string representation
        content = f"{model_type}_{model_name}_"
        content += "_".join(sorted(sequences))
        
        # Add kwargs
        for k, v in sorted(kwargs.items()):
            content += f"_{k}={v}"
        
        # Hash it
        cache_key = hashlib.md5(content.encode()).hexdigest()
        
        return cache_key
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        sequences: List[str],
        model_name: str,
        model_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Save embeddings to cache.
        
        Args:
            embeddings: NumPy array of embeddings
            sequences: List of sequences/SMILES
            model_name: Model name
            model_type: Model type
            metadata: Additional metadata to save
            **kwargs: Additional parameters for cache key
            
        Returns:
            Cache key if saved, None otherwise
        """
        cache_key = self._generate_cache_key(sequences, model_name, model_type, **kwargs)
        
        # Memory cache
        if self.use_memory_cache:
            self._memory_cache[cache_key] = embeddings
            if self.verbose:
                print(f"   💾 Saved to memory cache: {cache_key[:16]}...")
        
        # Disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            # Prepare data
            cache_data = {
                'embeddings': embeddings,
                'sequences': sequences,
                'model_name': model_name,
                'model_type': model_type,
                'metadata': metadata or {},
                'kwargs': kwargs
            }
            
            # Save
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                if self.verbose:
                    print(f"   💾 Saved to disk cache: {cache_file.name}")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  Failed to save to disk cache: {e}")
        
        return cache_key
    
    def load_embeddings(
        self,
        sequences: List[str],
        model_name: str,
        model_type: str,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Load embeddings from cache.
        
        Args:
            sequences: List of sequences/SMILES
            model_name: Model name
            model_type: Model type
            **kwargs: Additional parameters for cache key
            
        Returns:
            NumPy array of embeddings or None if not cached
        """
        cache_key = self._generate_cache_key(sequences, model_name, model_type, **kwargs)
        
        # Try memory cache first
        if self.use_memory_cache and cache_key in self._memory_cache:
            if self.verbose:
                print(f"   ♻️  Loaded from memory cache: {cache_key[:16]}...")
            return self._memory_cache[cache_key]
        
        # Try disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    embeddings = cache_data['embeddings']
                    
                    # Also load into memory cache
                    if self.use_memory_cache:
                        self._memory_cache[cache_key] = embeddings
                    
                    if self.verbose:
                        print(f"   ♻️  Loaded from disk cache: {cache_file.name}")
                    
                    return embeddings
                    
                except Exception as e:
                    if self.verbose:
                        print(f"   ⚠️  Failed to load from disk cache: {e}")
        
        return None
    
    def clear_memory_cache(self):
        """Clear memory cache."""
        self._memory_cache.clear()
        if self.verbose:
            print("   🗑️  Cleared memory cache")
    
    def clear_disk_cache(self):
        """Clear disk cache."""
        if not self.cache_dir:
            return
        
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1
        
        if self.verbose:
            print(f"   🗑️  Cleared {count} files from disk cache")
    
    def clear_all(self):
        """Clear both memory and disk cache."""
        self.clear_memory_cache()
        self.clear_disk_cache()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cache status.
        
        Returns:
            Dictionary with cache statistics
        """
        info = {
            'cache_dir': str(self.cache_dir) if self.cache_dir else None,
            'use_memory_cache': self.use_memory_cache,
            'memory_cache_size': len(self._memory_cache)
        }
        
        if self.cache_dir and self.cache_dir.exists():
            disk_files = list(self.cache_dir.glob("*.pkl"))
            info['disk_cache_size'] = len(disk_files)
        else:
            info['disk_cache_size'] = 0
        
        return info
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CacheManager(cache_dir={self.cache_dir}, "
            f"memory_cache_size={len(self._memory_cache)})"
        )

