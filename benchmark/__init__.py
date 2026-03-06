"""Benchmark package for semantic-screening model comparison.

Provides a modular, extensible framework for running multi-level benchmarks
comparing fingerprint, embedding, matrix-pooling, and cross-attention models.

Usage:
    from benchmark.orchestrator import BenchmarkOrchestrator
    from benchmark.config import BenchmarkConfig

    config = BenchmarkConfig(dataset="human", embedding="8M", levels=[1, 2, 3, 4])
    orchestrator = BenchmarkOrchestrator(config)
    results = orchestrator.run()
"""

from benchmark.config import (
    LEVEL_COLORS,
    LEVEL_LABELS,
    METRICS_ORDER,
    SUPPORTED_EMBEDDINGS,
    BenchmarkConfig,
)
from benchmark.orchestrator import BenchmarkOrchestrator

__all__ = [
    "BenchmarkConfig",
    "BenchmarkOrchestrator",
    "SUPPORTED_EMBEDDINGS",
    "LEVEL_LABELS",
    "LEVEL_COLORS",
    "METRICS_ORDER",
]
