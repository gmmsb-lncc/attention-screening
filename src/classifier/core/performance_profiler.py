"""
Performance profiling and analysis toolkit for DockTkinase.

Provides tools for:
  - Timing critical operations
  - Memory usage tracking
  - Bottleneck identification
  - Performance comparison (before/after)
"""

import functools
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def _get_memory_mb() -> float:
    """Get current process memory in MB (cross-platform)."""
    try:
        import psutil
        process = os.getpid()
        p = psutil.Process(process)
        return p.memory_info().rss / 1024 / 1024
    except (ImportError, Exception):
        # Fallback: use /proc/self/status on Linux
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024
        except Exception:
            pass
        return 0.0


@dataclass
class TimingStats:
    """Statistics for timed operations."""
    name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    memory_peak_mb: float = 0.0
    errors: int = 0

    @property
    def avg_time(self) -> float:
        """Average time per call."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    @property
    def throughput(self) -> float:
        """Calls per second."""
        return self.call_count / self.total_time if self.total_time > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"{self.name:30} | "
            f"calls: {self.call_count:6} | "
            f"avg: {self.avg_time*1000:7.2f}ms | "
            f"total: {self.total_time:8.2f}s | "
            f"throughput: {self.throughput:7.1f} ops/s"
        )


class PerformanceProfiler:
    """
    Central profiler for timing and memory tracking.
    
    Usage:
        profiler = PerformanceProfiler()
        
        # Time a function call
        with profiler.timer("operation_name"):
            expensive_operation()
        
        # Decorate a function
        @profiler.profile
        def my_function():
            ...
        
        # Get report
        profiler.report()
    """

    def __init__(self):
        """Initialize profiler."""
        self.stats: Dict[str, TimingStats] = defaultdict(lambda: TimingStats(name=""))
        self._baseline_memory = _get_memory_mb()

    def _get_memory_mb(self) -> float:
        """Get current process memory in MB."""
        return _get_memory_mb()

    @contextmanager
    def timer(self, name: str):
        """
        Context manager for timing operations.
        
        Args:
            name: Operation name for reporting
        """
        if name not in self.stats:
            self.stats[name] = TimingStats(name=name)
        
        stats = self.stats[name]
        start_time = time.perf_counter()
        start_memory = self._get_memory_mb()
        
        try:
            yield
            elapsed = time.perf_counter() - start_time
            peak_memory = self._get_memory_mb() - start_memory
            
            stats.call_count += 1
            stats.total_time += elapsed
            stats.min_time = min(stats.min_time, elapsed)
            stats.max_time = max(stats.max_time, elapsed)
            stats.memory_peak_mb = max(stats.memory_peak_mb, peak_memory)
            
        except Exception as e:
            stats.errors += 1
            logger.error(f"Error in {name}: {e}")
            raise

    def profile(self, func=None, name=None):
        """
        Decorator for profiling functions.
        
        Args:
            func: Function to profile
            name: Custom name (defaults to function name)
        """
        def decorator(f):
            op_name = name or f.__name__
            
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                with self.timer(op_name):
                    return f(*args, **kwargs)
            
            return wrapper
        
        # Support both @profile and @profile() syntax
        if func is not None:
            return decorator(func)
        else:
            return decorator

    def report(self, sort_by: str = "total_time", top_n: Optional[int] = None) -> str:
        """
        Generate performance report.
        
        Args:
            sort_by: Sort by 'total_time', 'avg_time', 'throughput', 'call_count'
            top_n: Show top N operations (None = all)
        
        Returns:
            Formatted report string
        """
        if not self.stats:
            return "No profiling data collected."
        
        # Sort stats
        sorted_stats = sorted(
            self.stats.values(),
            key=lambda s: getattr(s, sort_by),
            reverse=True
        )
        
        if top_n:
            sorted_stats = sorted_stats[:top_n]
        
        # Build report
        lines = [
            "\n" + "="*120,
            "PERFORMANCE REPORT",
            "="*120,
            ""
        ]
        
        for stats in sorted_stats:
            lines.append(str(stats))
        
        # Summary
        lines.extend([
            "",
            "="*120,
            "SUMMARY",
            "="*120,
            f"Total operations: {sum(s.call_count for s in self.stats.values())}",
            f"Total time: {sum(s.total_time for s in self.stats.values()):.2f}s",
            f"Total errors: {sum(s.errors for s in self.stats.values())}",
            f"Peak memory used: {self._get_memory_mb() - self._baseline_memory:.1f}MB",
            "="*120,
            ""
        ])
        
        return "\n".join(lines)

    def get_stats(self, name: str) -> Optional[TimingStats]:
        """Get stats for a specific operation."""
        return self.stats.get(name)

    def compare(self, other: 'PerformanceProfiler', metric: str = "avg_time") -> str:
        """
        Compare performance with another profiler.
        
        Args:
            other: Another PerformanceProfiler instance
            metric: Metric to compare ('avg_time', 'total_time', 'throughput')
        """
        lines = [
            "\n" + "="*140,
            "PERFORMANCE COMPARISON",
            "="*140,
            f"{'Operation':<30} {'Metric':<15} {'Before':<20} {'After':<20} {'Change':<15} {'Improvement':<15}"
        ]
        
        common_ops = set(self.stats.keys()) & set(other.stats.keys())
        
        for op in sorted(common_ops):
            self_val = getattr(self.stats[op], metric)
            other_val = getattr(other.stats[op], metric)
            
            if metric == "throughput":
                # Higher is better
                change = ((other_val - self_val) / self_val * 100) if self_val > 0 else 0
                improvement = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
            else:
                # Lower is better
                change = ((self_val - other_val) / self_val * 100) if self_val > 0 else 0
                improvement = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
            
            lines.append(
                f"{op:<30} {metric:<15} {self_val:<20.4f} {other_val:<20.4f} "
                f"{change:>+.1f}% {improvement:<15}"
            )
        
        lines.extend([
            "="*140,
            ""
        ])
        
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all collected stats."""
        self.stats.clear()
        self._baseline_memory = self._get_memory_mb()


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile(func=None, name=None):
    """
    Decorator to profile a function using global profiler.
    
    Usage:
        @profile
        def my_function():
            ...
    """
    profiler = get_profiler()
    return profiler.profile(func=func, name=name)


@contextmanager
def timer(name: str):
    """
    Context manager to time an operation.
    
    Usage:
        with timer("operation"):
            expensive_operation()
    """
    profiler = get_profiler()
    with profiler.timer(name):
        yield


if __name__ == "__main__":
    # Example usage
    profiler = get_profiler()
    
    @profile("fibonacci_slow")
    def fib_slow(n: int) -> int:
        if n <= 1:
            return n
        return fib_slow(n - 1) + fib_slow(n - 2)
    
    # Test
    for i in range(5):
        with timer(f"iteration_{i}"):
            fib_slow(20)
    
    print(profiler.report(top_n=10))
