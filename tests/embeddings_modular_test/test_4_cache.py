"""
Test 4: CacheManager - Memory and disk caching
===============================================
Tests the two-level caching system.
"""

import sys
from pathlib import Path
import tempfile
import shutil
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from build.embeddings.utils.cache import CacheManager


def test_cache_manager_init():
    """Test 4.1: Initialize cache manager"""
    print("\n" + "="*70)
    print("TEST 4.1: Cache Manager Initialization")
    print("="*70)
    
    # Create temporary cache directory
    cache_dir = tempfile.mkdtemp()
    
    try:
        cache = CacheManager(
            cache_dir=cache_dir,
            use_memory_cache=True,
            # disk cache via cache_dir,
            verbose=True
        )
        
        print(f"\n📊 Cache initialized:")
        print(f"   - Cache dir: {cache_dir}")
        print(f"   - Memory cache: enabled")
        print(f"   - Disk cache: enabled")
        
        info = cache.get_cache_info()
        print(f"\n📊 Cache info:")
        print(f"   - Memory cache size: {info['memory_cache_size']}")
        print(f"   - Disk cache files: {info['disk_cache_files']}")
        
        assert info['memory_cache_size'] == 0
        assert info['disk_cache_files'] == 0
        
        print("\n✅ TEST 4.1 PASSED!")
        
    finally:
        shutil.rmtree(cache_dir)


def test_memory_cache():
    """Test 4.2: Memory caching"""
    print("\n" + "="*70)
    print("TEST 4.2: Memory Cache")
    print("="*70)
    
    cache_dir = tempfile.mkdtemp()
    
    try:
    try:
        cache = CacheManager(
            cache_dir=None,  # No disk cache
            use_memory_cache=True,
            verbose=True
        )
        
        # Create fake embeddings
        sequences = ["MKTAYIAK", "ACDEFGH"]
        embeddings = np.random.rand(2, 1280).astype(np.float32)
        
        print(f"\n📊 Saving to memory cache...")
        print(f"   - Sequences: {len(sequences)}")
        print(f"   - Shape: {embeddings.shape}")
        
        cache.save_embeddings(
            sequences=sequences,
            embeddings=embeddings,
            model_name="esm2_t33_650M_UR50D",
            model_type="esm"
        )
        
        # Load from cache
        print(f"\n📊 Loading from memory cache...")
        loaded = cache.load_embeddings(
            sequences=sequences,
            model_name="esm2_t33_650M_UR50D",
            model_type="esm"
        )
        
        assert loaded is not None, "Should load from memory cache"
        assert np.allclose(loaded, embeddings), "Embeddings should match"
        
        print(f"   ✅ Loaded successfully")
        print(f"   - Shape: {loaded.shape}")
        
        # Check cache info
        info = cache.get_cache_info()
        assert info['memory_cache_size'] == 1, "Should have 1 item in memory cache"
        
        print("\n✅ TEST 4.2 PASSED!")
        
    finally:
        shutil.rmtree(cache_dir)


def test_disk_cache():
    """Test 4.3: Disk caching"""
    print("\n" + "="*70)
    print("TEST 4.3: Disk Cache")
    print("="*70)
    
    cache_dir = tempfile.mkdtemp()
    
    try:
        cache = CacheManager(
            cache_dir=cache_dir,  # Disk cache enabled
            use_memory_cache=False,
            verbose=True
        )
        
        # Create fake embeddings
        sequences = ["MKTAYIAK", "ACDEFGH", "MKWVTFIS"]
        embeddings = np.random.rand(3, 1280).astype(np.float32)
        
        print(f"\n📊 Saving to disk cache...")
        print(f"   - Sequences: {len(sequences)}")
        print(f"   - Shape: {embeddings.shape}")
        
        cache.save_embeddings(
            sequences=sequences,
            embeddings=embeddings,
            model_name="esm2_t33_650M_UR50D",
            model_type="esm"
        )
        
        # Clear memory cache (to force disk load)
        cache.clear_memory_cache()
        
        # Load from disk
        print(f"\n📊 Loading from disk cache...")
        loaded = cache.load_embeddings(
            sequences=sequences,
            model_name="esm2_t33_650M_UR50D",
            model_type="esm"
        )
        
        assert loaded is not None, "Should load from disk cache"
        assert np.allclose(loaded, embeddings), "Embeddings should match"
        
        print(f"   ✅ Loaded successfully from disk")
        print(f"   - Shape: {loaded.shape}")
        
        # Check cache files
        info = cache.get_cache_info()
        assert info['disk_cache_files'] >= 1, "Should have cache files on disk"
        
        print("\n✅ TEST 4.3 PASSED!")
        
    finally:
        shutil.rmtree(cache_dir)


def test_cache_miss():
    """Test 4.4: Cache miss (not found)"""
    print("\n" + "="*70)
    print("TEST 4.4: Cache Miss")
    print("="*70)
    
    cache_dir = tempfile.mkdtemp()
    
    try:
        cache = CacheManager(
            cache_dir=cache_dir,
            use_memory_cache=True,
            # disk cache via cache_dir,
            verbose=True
        )
        
        # Try to load non-existent cache
        sequences = ["NONEXISTENT"]
        
        print(f"\n📊 Attempting to load non-existent cache...")
        
        loaded = cache.load_embeddings(
            sequences=sequences,
            model_name="esm2_t33_650M_UR50D",
            model_type="esm"
        )
        
        assert loaded is None, "Should return None for cache miss"
        
        print(f"   ✅ Correctly returned None for cache miss")
        
        print("\n✅ TEST 4.4 PASSED!")
        
    finally:
        shutil.rmtree(cache_dir)


def test_clear_cache():
    """Test 4.5: Clear all caches"""
    print("\n" + "="*70)
    print("TEST 4.5: Clear Cache")
    print("="*70)
    
    cache_dir = tempfile.mkdtemp()
    
    try:
        cache = CacheManager(
            cache_dir=cache_dir,
            use_memory_cache=True,
            # disk cache via cache_dir,
            verbose=True
        )
        
        # Save some data
        sequences = ["MKTAYIAK", "ACDEFGH"]
        embeddings = np.random.rand(2, 1280).astype(np.float32)
        
        cache.save_embeddings(
            sequences=sequences,
            embeddings=embeddings,
            model_name="esm2_t33_650M_UR50D",
            model_type="esm"
        )
        
        # Check it's cached
        info = cache.get_cache_info()
        print(f"\n📊 Before clear:")
        print(f"   - Memory cache: {info['memory_cache_size']}")
        print(f"   - Disk cache: {info['disk_cache_files']}")
        
        # Clear all
        print(f"\n📊 Clearing all caches...")
        cache.clear_all()
        
        # Check it's empty
        info = cache.get_cache_info()
        print(f"\n📊 After clear:")
        print(f"   - Memory cache: {info['memory_cache_size']}")
        print(f"   - Disk cache: {info['disk_cache_files']}")
        
        assert info['memory_cache_size'] == 0
        assert info['disk_cache_files'] == 0
        
        print("\n✅ TEST 4.5 PASSED!")
        
    finally:
        shutil.rmtree(cache_dir)


if __name__ == "__main__":
    print("\n" + "🧪 RUNNING CACHE MANAGER TESTS ".center(70, "="))
    
    try:
        test_cache_manager_init()
        test_memory_cache()
        test_disk_cache()
        test_cache_miss()
        test_clear_cache()
        
        print("\n" + "="*70)
        print("✅ ALL CACHE MANAGER TESTS PASSED!".center(70))
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
