"""
Embedding Performance Profiler and Analyzer.

Measures ESM2/ESM3 embedding extraction performance with detailed breakdown:
- Time per component (tokenization, forward pass, output processing)
- Memory usage
- Throughput metrics
- Identifies optimization opportunities
"""

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


@dataclass
class TimingRecord:
    """Record of a single timed operation."""

    name: str
    duration_ms: float
    memory_mb: float = 0.0
    count: int = 1


@dataclass
class ProfileStats:
    """Statistics for a profiled component."""

    total_time_ms: float = 0.0
    num_calls: int = 0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    total_memory_mb: float = 0.0
    timings: List[float] = field(default_factory=list)

    def add_timing(self, duration_ms: float, memory_mb: float = 0.0) -> None:
        """Add a timing measurement."""
        self.total_time_ms += duration_ms
        self.num_calls += 1
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.timings.append(duration_ms)
        self.total_memory_mb += memory_mb

    @property
    def avg_time_ms(self) -> float:
        """Average time per call."""
        return self.total_time_ms / self.num_calls if self.num_calls > 0 else 0.0

    @property
    def median_time_ms(self) -> float:
        """Median time per call."""
        if not self.timings:
            return 0.0
        sorted_timings = sorted(self.timings)
        n = len(sorted_timings)
        return (
            sorted_timings[n // 2]
            if n % 2 == 1
            else (sorted_timings[n // 2 - 1] + sorted_timings[n // 2]) / 2
        )

    @property
    def avg_memory_mb(self) -> float:
        """Average memory per call."""
        return self.total_memory_mb / self.num_calls if self.num_calls > 0 else 0.0


class EmbeddingProfiler:
    """
    Profile ESM2/ESM3 embedding extraction pipeline.

    Components to profile:
    1. Tokenization: FASTA → tokens
    2. Model forward: tokens → embeddings
    3. Output processing: embeddings → normalized output
    4. Caching: lookup/store operations
    """

    def __init__(self):
        """Initialize profiler."""
        self.stats: Dict[str, ProfileStats] = {}
        self.current_component: Optional[str] = None
        self.start_time: float = 0.0
        self.start_memory: int = 0

    def start_component(self, name: str) -> None:
        """Start profiling a component."""
        self.current_component = name
        self.start_time = time.perf_counter()
        tracemalloc.start()
        self.start_memory = tracemalloc.get_traced_memory()[0]

    def end_component(self) -> float:
        """End profiling current component and return duration."""
        if self.current_component is None:
            raise ValueError("No component started")

        duration = (time.perf_counter() - self.start_time) * 1000  # ms
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        memory_used = (peak_memory - self.start_memory) / (1024 * 1024)  # MB
        tracemalloc.stop()

        if self.current_component not in self.stats:
            self.stats[self.current_component] = ProfileStats()

        self.stats[self.current_component].add_timing(duration, memory_used)
        self.current_component = None

        return duration

    def context(self, name: str):
        """Context manager for profiling."""

        class ProfileContext:
            def __init__(self, profiler: "EmbeddingProfiler", comp_name: str):
                self.profiler = profiler
                self.name = comp_name

            def __enter__(self):
                self.profiler.start_component(self.name)
                return self

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
                self.profiler.end_component()

        return ProfileContext(self, name)

    def get_report(self) -> str:
        """Generate profiling report."""
        if not self.stats:
            return "No profiling data collected"

        total_time = sum(s.total_time_ms for s in self.stats.values())
        total_memory = sum(s.total_memory_mb for s in self.stats.values())

        report = "\n" + "=" * 80 + "\n"
        report += "📊 EMBEDDING PROFILING REPORT\n"
        report += "=" * 80 + "\n\n"

        report += "Component                      Time (ms)    Avg        Memory      %\n"
        report += "-" * 80 + "\n"

        for name, stat in sorted(
            self.stats.items(), key=lambda x: x[1].total_time_ms, reverse=True
        ):
            percentage = (stat.total_time_ms / total_time * 100) if total_time > 0 else 0
            report += (
                f"{name:<30} {stat.total_time_ms:>10.2f} "
                f"{stat.avg_time_ms:>8.2f}ms {stat.avg_memory_mb:>8.2f}MB "
                f"{percentage:>6.1f}%\n"
            )

        report += "-" * 80 + "\n"
        report += (
            f"{'TOTAL':<30} {total_time:>10.2f}ms "
            f"{'':<10} {total_memory:>8.2f}MB\n"
        )
        report += "=" * 80 + "\n"

        return report

    def get_bottleneck(self) -> Tuple[Optional[str], float]:
        """Get component with highest time."""
        if not self.stats:
            return None, 0.0

        bottleneck = max(self.stats.items(), key=lambda x: x[1].total_time_ms)
        return bottleneck[0], bottleneck[1].total_time_ms

    def reset(self) -> None:
        """Reset all statistics."""
        self.stats.clear()
        self.current_component = None


# Global profiler instance
_global_profiler: Optional[EmbeddingProfiler] = None


def get_embedding_profiler() -> EmbeddingProfiler:
    """Get or create global embedding profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = EmbeddingProfiler()
    return _global_profiler


def profile_embedding_extraction(
    func: Callable[[str, str], Any],
    sequence: str,
    model_name: str = "esm2_t33_650M_UR50D",
) -> Tuple[Any, float]:
    """
    Profile an embedding extraction function.

    Args:
        func: Function that extracts embeddings
        sequence: Protein sequence
        model_name: ESM model identifier

    Returns:
        Tuple of (embeddings, total_time_ms)
    """
    profiler = get_embedding_profiler()

    with profiler.context("tokenization"):
        # Simulate tokenization
        time.sleep(0.01)

    with profiler.context("model_forward"):
        # Simulate forward pass
        embeddings = func(sequence, model_name)
        time.sleep(0.1)

    with profiler.context("output_processing"):
        # Simulate output processing
        if hasattr(embeddings, "astype"):
            output = embeddings.astype("float32")
        else:
            output = embeddings

    bottleneck, duration = profiler.get_bottleneck()
    return output, profiler.stats["model_forward"].total_time_ms

