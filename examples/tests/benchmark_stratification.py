"""
Performance benchmarking for stratification system.

This script measures:
1. Time overhead: stratification vs random split
2. Memory usage comparison
3. Scalability with different dataset sizes
4. Impact of different clustering algorithms
"""

import time
import numpy as np
import psutil
import os
from pathlib import Path
import sys
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.build.core.config import BuildConfig
from src.build.pipeline.stratification_manager import StratificationManager
from sklearn.model_selection import train_test_split


class MemoryMonitor:
    """Monitor memory usage during operations."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def get_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_delta(self) -> float:
        """Get memory increase from baseline in MB."""
        return self.get_usage() - self.baseline


class BenchmarkResults:
    """Store and analyze benchmark results."""
    
    def __init__(self):
        self.results = {
            'stratification': {
                'times': [],
                'memory': [],
                'dataset_sizes': []
            },
            'random_split': {
                'times': [],
                'memory': [],
                'dataset_sizes': []
            }
        }
    
    def add_result(self, method: str, time_ms: float, memory_mb: float, size: int):
        """Add a benchmark result."""
        self.results[method]['times'].append(time_ms)
        self.results[method]['memory'].append(memory_mb)
        self.results[method]['dataset_sizes'].append(size)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {}
        for method in ['stratification', 'random_split']:
            times = np.array(self.results[method]['times'])
            memory = np.array(self.results[method]['memory'])
            
            summary[method] = {
                'time_mean_ms': float(np.mean(times)),
                'time_std_ms': float(np.std(times)),
                'time_min_ms': float(np.min(times)),
                'time_max_ms': float(np.max(times)),
                'memory_mean_mb': float(np.mean(memory)),
                'memory_std_mb': float(np.std(memory)),
                'memory_min_mb': float(np.min(memory)),
                'memory_max_mb': float(np.max(memory))
            }
        
        # Calculate overhead
        strat_time = summary['stratification']['time_mean_ms']
        random_time = summary['random_split']['time_mean_ms']
        time_overhead_pct = ((strat_time - random_time) / random_time) * 100
        
        strat_mem = summary['stratification']['memory_mean_mb']
        random_mem = summary['random_split']['memory_mean_mb']
        memory_overhead_pct = ((strat_mem - random_mem) / random_mem) * 100
        
        summary['overhead'] = {
            'time_overhead_pct': float(time_overhead_pct),
            'memory_overhead_pct': float(memory_overhead_pct),
            'time_overhead_ms': float(strat_time - random_time),
            'memory_overhead_mb': float(strat_mem - random_mem)
        }
        
        return summary
    
    def save_to_json(self, filepath: str):
        """Save results to JSON file."""
        summary = self.get_summary()
        summary['raw_data'] = self.results
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {filepath}")


def benchmark_random_split(
    protein_embeddings: np.ndarray,
    ligand_embeddings: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Benchmark random stratified split (sklearn baseline).
    
    Returns:
        (time_ms, memory_mb)
    """
    monitor = MemoryMonitor()
    
    start_time = time.perf_counter()
    
    # Split train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        np.arange(len(labels)),
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    # Split train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=random_state
    )
    
    elapsed_time = (time.perf_counter() - start_time) * 1000  # ms
    memory_used = monitor.get_delta()
    
    return elapsed_time, memory_used


def benchmark_stratification(
    protein_embeddings: np.ndarray,
    ligand_embeddings: np.ndarray,
    labels: np.ndarray,
    config: BuildConfig,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    clustering_algorithm: str = 'kmeans'
) -> Tuple[float, float]:
    """
    Benchmark StratificationManager with embedding-based clustering.
    
    Returns:
        (time_ms, memory_mb)
    """
    monitor = MemoryMonitor()
    
    manager = StratificationManager(
        config,
        clustering_algorithm=clustering_algorithm,
        random_state=random_state
    )
    
    start_time = time.perf_counter()
    
    splits = manager.stratify(
        protein_embeddings=protein_embeddings,
        ligand_embeddings=ligand_embeddings,
        labels=labels,
        test_size=test_size,
        val_size=val_size
    )
    
    elapsed_time = (time.perf_counter() - start_time) * 1000  # ms
    memory_used = monitor.get_delta()
    
    return elapsed_time, memory_used


def run_benchmark_suite(
    dataset_sizes: List[int] = [100, 500, 1000, 2000, 5000],
    n_iterations: int = 5,
    output_dir: str = "results/benchmarks"
) -> BenchmarkResults:
    """
    Run complete benchmark suite across different dataset sizes.
    
    Args:
        dataset_sizes: List of dataset sizes to test
        n_iterations: Number of iterations per size
        output_dir: Directory to save results
    """
    config = BuildConfig()
    results = BenchmarkResults()
    
    print("=" * 70)
    print("STRATIFICATION PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Iterations per size: {n_iterations}")
    print()
    
    for size in dataset_sizes:
        print(f"\n{'='*70}")
        print(f"Testing with {size} samples...")
        print(f"{'='*70}")
        
        for iteration in range(n_iterations):
            print(f"\n  Iteration {iteration + 1}/{n_iterations}")
            
            # Generate synthetic data
            protein_embeddings = np.random.randn(size, 320).astype(np.float32)
            ligand_embeddings = np.random.randn(size, 768).astype(np.float32)
            labels = np.random.randint(0, 2, size).astype(np.int32)
            
            # Benchmark random split
            print("    - Random split...", end=" ", flush=True)
            time_random, mem_random = benchmark_random_split(
                protein_embeddings, ligand_embeddings, labels
            )
            print(f"{time_random:.2f}ms, {mem_random:.2f}MB")
            results.add_result('random_split', time_random, mem_random, size)
            
            # Benchmark stratification
            print("    - Stratification...", end=" ", flush=True)
            time_strat, mem_strat = benchmark_stratification(
                protein_embeddings, ligand_embeddings, labels, config
            )
            print(f"{time_strat:.2f}ms, {mem_strat:.2f}MB")
            results.add_result('stratification', time_strat, mem_strat, size)
            
            # Show overhead for this iteration
            time_overhead = ((time_strat - time_random) / time_random) * 100
            mem_overhead = ((mem_strat - mem_random) / mem_random) * 100 if mem_random > 0 else 0
            print(f"    → Overhead: {time_overhead:+.1f}% time, {mem_overhead:+.1f}% memory")
    
    return results


def plot_benchmark_results(results: BenchmarkResults, output_dir: str):
    """Create visualization plots for benchmark results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    sizes_strat = np.array(results.results['stratification']['dataset_sizes'])
    times_strat = np.array(results.results['stratification']['times'])
    memory_strat = np.array(results.results['stratification']['memory'])
    
    sizes_random = np.array(results.results['random_split']['dataset_sizes'])
    times_random = np.array(results.results['random_split']['times'])
    memory_random = np.array(results.results['random_split']['memory'])
    
    # Plot 1: Execution Time vs Dataset Size
    ax1 = axes[0, 0]
    ax1.scatter(sizes_strat, times_strat, alpha=0.6, label='Stratification', s=50)
    ax1.scatter(sizes_random, times_random, alpha=0.6, label='Random Split', s=50)
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time vs Dataset Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory Usage vs Dataset Size
    ax2 = axes[0, 1]
    ax2.scatter(sizes_strat, memory_strat, alpha=0.6, label='Stratification', s=50)
    ax2.scatter(sizes_random, memory_random, alpha=0.6, label='Random Split', s=50)
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title('Memory Usage vs Dataset Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time Overhead Distribution
    ax3 = axes[1, 0]
    unique_sizes = np.unique(sizes_strat)
    overhead_times = []
    for size in unique_sizes:
        mask_strat = sizes_strat == size
        mask_random = sizes_random == size
        t_strat = times_strat[mask_strat].mean()
        t_random = times_random[mask_random].mean()
        overhead_pct = ((t_strat - t_random) / t_random) * 100
        overhead_times.append(overhead_pct)
    
    ax3.bar(unique_sizes, overhead_times, alpha=0.7, color='coral')
    ax3.set_xlabel('Dataset Size')
    ax3.set_ylabel('Time Overhead (%)')
    ax3.set_title('Stratification Time Overhead vs Dataset Size')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary = results.get_summary()
    summary_text = f"""
PERFORMANCE SUMMARY

Stratification:
  Time:   {summary['stratification']['time_mean_ms']:.2f} ± {summary['stratification']['time_std_ms']:.2f} ms
  Memory: {summary['stratification']['memory_mean_mb']:.2f} ± {summary['stratification']['memory_std_mb']:.2f} MB

Random Split:
  Time:   {summary['random_split']['time_mean_ms']:.2f} ± {summary['random_split']['time_std_ms']:.2f} ms
  Memory: {summary['random_split']['memory_mean_mb']:.2f} ± {summary['random_split']['memory_std_mb']:.2f} MB

Overhead:
  Time:   {summary['overhead']['time_overhead_pct']:+.1f}% ({summary['overhead']['time_overhead_ms']:+.2f} ms)
  Memory: {summary['overhead']['memory_overhead_pct']:+.1f}% ({summary['overhead']['memory_overhead_mb']:+.2f} MB)

Conclusion:
  {'✓ Acceptable overhead' if summary['overhead']['time_overhead_pct'] < 50 else '⚠ High overhead'}
  {'✓ Low memory impact' if summary['overhead']['memory_overhead_mb'] < 100 else '⚠ High memory usage'}
    """
    
    ax4.text(0.1, 0.5, summary_text, 
             verticalalignment='center',
             fontfamily='monospace',
             fontsize=10)
    
    plt.tight_layout()
    plot_path = output_path / "benchmark_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()


def main():
    """Run benchmark suite and generate report."""
    output_dir = "results/benchmarks"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    results = run_benchmark_suite(
        dataset_sizes=[100, 500, 1000, 2000],
        n_iterations=3,
        output_dir=output_dir
    )
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    summary = results.get_summary()
    
    print("\nStratification Performance:")
    print(f"  Average time:   {summary['stratification']['time_mean_ms']:.2f} ± "
          f"{summary['stratification']['time_std_ms']:.2f} ms")
    print(f"  Average memory: {summary['stratification']['memory_mean_mb']:.2f} ± "
          f"{summary['stratification']['memory_std_mb']:.2f} MB")
    
    print("\nRandom Split Performance:")
    print(f"  Average time:   {summary['random_split']['time_mean_ms']:.2f} ± "
          f"{summary['random_split']['time_std_ms']:.2f} ms")
    print(f"  Average memory: {summary['random_split']['memory_mean_mb']:.2f} ± "
          f"{summary['random_split']['memory_std_mb']:.2f} MB")
    
    print("\nOverhead Analysis:")
    print(f"  Time overhead:   {summary['overhead']['time_overhead_pct']:+.1f}% "
          f"({summary['overhead']['time_overhead_ms']:+.2f} ms)")
    print(f"  Memory overhead: {summary['overhead']['memory_overhead_pct']:+.1f}% "
          f"({summary['overhead']['memory_overhead_mb']:+.2f} MB)")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    time_overhead = summary['overhead']['time_overhead_pct']
    if time_overhead < 20:
        print("✓ EXCELLENT: Time overhead is negligible (<20%)")
    elif time_overhead < 50:
        print("✓ ACCEPTABLE: Time overhead is reasonable (20-50%)")
    elif time_overhead < 100:
        print("⚠ MODERATE: Time overhead is noticeable (50-100%)")
    else:
        print("⚠ HIGH: Time overhead is significant (>100%)")
    
    mem_overhead = summary['overhead']['memory_overhead_mb']
    if mem_overhead < 50:
        print("✓ EXCELLENT: Memory overhead is negligible (<50 MB)")
    elif mem_overhead < 200:
        print("✓ ACCEPTABLE: Memory overhead is reasonable (50-200 MB)")
    else:
        print("⚠ HIGH: Memory overhead is significant (>200 MB)")
    
    print("\nRecommendation:")
    if time_overhead < 50 and mem_overhead < 200:
        print("✓ Stratification overhead is ACCEPTABLE for production use.")
        print("  Benefits (improved model performance, reproducibility) outweigh costs.")
    else:
        print("⚠ Consider profiling for optimization opportunities.")
    
    # Save results
    json_path = Path(output_dir) / "benchmark_results.json"
    results.save_to_json(str(json_path))
    
    # Generate plots
    try:
        plot_benchmark_results(results, output_dir)
    except Exception as e:
        print(f"\nNote: Could not generate plots: {e}")
        print("Install matplotlib and seaborn to enable visualization.")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
