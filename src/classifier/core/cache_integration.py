"""
Cache integration layer for DockTkinase pipeline.

Provides decorators and utilities to add caching to existing pipeline functions
without requiring code changes.

Usage:
    # Option 1: Decorator approach
    @cached_pipeline_stage("embedding", ttl=3600)
    def get_embeddings(sequence: str, model: str) -> np.ndarray:
        return esm_model.embed(sequence)
    
    # Option 2: Context manager approach
    cache = get_pipeline_cache()
    key = CacheKey.from_sequence(sequence, stage="msa")
    
    embeddings, found = cache.get(key)
    if not found:
        embeddings = compute_msa(sequence)
        cache.set(key, embeddings, stage="msa")
    
    # Option 3: Manual integration
    cache_manager = CachedPipelineStage("embedding")
    embeddings = cache_manager.compute_or_cache(
        input_sequence=sequence,
        compute_fn=lambda seq: esm_model.embed(seq),
        save_to_disk=True
    )
"""

import functools
import logging
from typing import Any, Callable, Dict, Optional, TypeVar

from src.classifier.core.pipeline_cache import CacheKey, PipelineCache, get_pipeline_cache

logger = logging.getLogger(__name__)

T = TypeVar("T")


def cached_pipeline_stage(
    stage_name: str,
    ttl: Optional[float] = None,
    save_to_disk: bool = False,
    cache: Optional[PipelineCache] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to cache pipeline stage outputs.

    Args:
        stage_name: Name of the pipeline stage (e.g., 'embedding', 'msa')
        ttl: Time-to-live for cache entries in seconds
        save_to_disk: Whether to persist to disk cache
        cache: PipelineCache instance (uses global if None)

    Returns:
        Decorator function

    Example:
        @cached_pipeline_stage("embedding", ttl=3600)
        def get_embeddings(sequence: str, model: str) -> np.ndarray:
            return model.embed(sequence)
    """
    if cache is None:
        cache = get_pipeline_cache()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key from arguments
            # Typically: sequence is first arg, model_name is in kwargs
            try:
                if args:
                    sequence = args[0]
                    model_name = kwargs.get("model", "default")
                    key = CacheKey.from_sequence(
                        sequence, stage=stage_name, model=model_name
                    )
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Could not generate cache key: {e}")
                return func(*args, **kwargs)

            # Try to get from cache
            result, found = cache.get(key)
            if found:
                logger.debug(f"Cache hit: {stage_name}")
                return result

            # Compute result
            logger.debug(f"Cache miss: {stage_name}")
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(
                key, result, ttl=ttl, stage=stage_name, save_to_disk=save_to_disk
            )

            return result

        return wrapper

    return decorator


class CachedPipelineStage:
    """
    Context manager for caching pipeline stages.

    Simplifies adding caching to existing functions without decorators.

    Usage:
        stage = CachedPipelineStage("msa_computation")
        result = stage.compute_or_cache(
            input_sequence=seq,
            compute_fn=compute_msa,
            save_to_disk=True
        )
    """

    def __init__(
        self,
        stage_name: str,
        ttl: Optional[float] = None,
        cache: Optional[PipelineCache] = None,
    ):
        """
        Initialize cached pipeline stage.

        Args:
            stage_name: Name of the pipeline stage
            ttl: Time-to-live for cache entries
            cache: PipelineCache instance (uses global if None)
        """
        self.stage_name = stage_name
        self.ttl = ttl
        self.cache = cache or get_pipeline_cache()
        self.last_found = False
        self.last_key = None

    def compute_or_cache(
        self,
        input_sequence: Optional[str] = None,
        compute_fn: Optional[Callable[[], T]] = None,
        save_to_disk: bool = False,
        **context_kwargs: Any,
    ) -> T:
        """
        Compute result or retrieve from cache.

        Args:
            input_sequence: Input sequence for cache key generation
            compute_fn: Function to call if cache miss
            save_to_disk: Whether to persist result
            **context_kwargs: Additional context for cache key

        Returns:
            Computed or cached result
        """
        if compute_fn is None:
            raise ValueError("compute_fn is required")

        # Generate cache key
        if input_sequence:
            key = CacheKey.from_sequence(
                input_sequence, stage=self.stage_name, **context_kwargs
            )
        else:
            key = CacheKey.from_dict(
                context_kwargs, stage=self.stage_name
            )

        self.last_key = key

        # Try cache
        result, found = self.cache.get(key)
        self.last_found = found

        if found:
            logger.debug(f"Cache hit: {self.stage_name}")
            return result

        # Compute
        logger.debug(f"Cache miss: {self.stage_name}")
        result = compute_fn()

        # Store in cache
        self.cache.set(
            key,
            result,
            ttl=self.ttl,
            stage=self.stage_name,
            save_to_disk=save_to_disk,
        )

        return result

    def was_cached(self) -> bool:
        """Return whether last result was from cache."""
        return self.last_found


class CacheAwarePipeline:
    """
    Wrapper for existing pipeline with transparent caching.

    Adds caching to any pipeline without modifying its code.

    Usage:
        original_pipeline = load_pipeline()
        cached_pipeline = CacheAwarePipeline(original_pipeline)
        
        # Now predictions are cached automatically
        predictions = cached_pipeline.predict(sequence)
    """

    def __init__(
        self,
        pipeline: Any,
        cache: Optional[PipelineCache] = None,
        cache_stages: Optional[Dict[str, bool]] = None,
    ):
        """
        Initialize cache-aware pipeline wrapper.

        Args:
            pipeline: Original pipeline object
            cache: PipelineCache instance
            cache_stages: Dict of {stage_name: should_cache}
        """
        self.pipeline = pipeline
        self.cache = cache or get_pipeline_cache()

        # Default cache stages
        self.cache_stages = cache_stages or {
            "embedding": True,
            "msa": True,
            "plddt": True,
            "structure": False,  # Too large
        }

        self._cache_stats: Dict[str, Dict[str, Any]] = {}

    def predict(self, sequence: str, **kwargs: Any) -> Any:
        """
        Run pipeline prediction with caching.

        Args:
            sequence: Input protein sequence
            **kwargs: Additional arguments for pipeline

        Returns:
            Pipeline predictions
        """
        # Check cache for full prediction
        full_key = CacheKey.from_sequence(
            sequence, stage="full_prediction", **kwargs
        )
        result, found = self.cache.get(full_key)

        if found:
            logger.info(f"Full prediction cache hit: {sequence[:20]}...")
            return result

        # Run pipeline
        result = self.pipeline.predict(sequence, **kwargs)

        # Cache result
        self.cache.set(
            full_key, result, stage="full_prediction", save_to_disk=False
        )

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.cache.clear()


# Utility functions for common pipeline operations

def cache_embeddings(
    sequence: str,
    embedding_fn: Callable[[str], Any],
    model_name: str = "default",
    ttl: Optional[float] = None,
) -> Any:
    """
    Compute or retrieve cached embeddings.

    Args:
        sequence: Input sequence
        embedding_fn: Function to compute embeddings
        model_name: Name of embedding model
        ttl: Cache TTL in seconds

    Returns:
        Embeddings (cached or computed)
    """
    cache = get_pipeline_cache()
    key = CacheKey.from_sequence(
        sequence, stage="embedding", model=model_name
    )

    result, found = cache.get(key)
    if found:
        return result

    result = embedding_fn(sequence)
    cache.set(key, result, ttl=ttl, stage="embedding", save_to_disk=True)

    return result


def cache_msa(
    sequence: str,
    msa_fn: Callable[[str], Any],
    database: str = "default",
    ttl: Optional[float] = None,
) -> Any:
    """
    Compute or retrieve cached MSA.

    Args:
        sequence: Input sequence
        msa_fn: Function to compute MSA
        database: MSA database identifier
        ttl: Cache TTL in seconds

    Returns:
        MSA (cached or computed)
    """
    cache = get_pipeline_cache()
    key = CacheKey.from_sequence(
        sequence, stage="msa", database=database
    )

    result, found = cache.get(key)
    if found:
        return result

    result = msa_fn(sequence)
    cache.set(key, result, ttl=ttl, stage="msa", save_to_disk=True)

    return result


def cache_structure_prediction(
    sequence: str,
    prediction_fn: Callable[[str], Any],
    model: str = "default",
    ttl: Optional[float] = None,
) -> Any:
    """
    Compute or retrieve cached structure prediction.

    Args:
        sequence: Input sequence
        prediction_fn: Function to predict structure
        model: Model name/version
        ttl: Cache TTL in seconds

    Returns:
        Structure prediction (cached or computed)
    """
    cache = get_pipeline_cache()
    key = CacheKey.from_sequence(
        sequence, stage="structure", model=model
    )

    result, found = cache.get(key)
    if found:
        return result

    result = prediction_fn(sequence)
    cache.set(
        key, result, ttl=ttl, stage="structure", save_to_disk=False
    )

    return result
