"""
Tier 3.1 Embedding Optimization - Practical Examples

This module contains ready-to-use examples for integrating embedding profiling
and quantization into protein embedding workflows.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path

# Import optimization components
from src.classifier.core.embedding_integration import (
    OptimizedEmbeddingExtractor,
    EmbeddingOptimizationContext,
    create_optimized_extractor,
    optimize_extraction_pipeline
)


# ==============================================================================
# Example 1: Basic Single-Sequence Extraction
# ==============================================================================

def example_basic_extraction():
    """
    Extract embeddings from a single protein sequence with profiling.
    
    This is the simplest use case - extract one sequence with all optimizations.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Single-Sequence Extraction")
    print("="*70)
    
    # Initialize extractor
    extractor = OptimizedEmbeddingExtractor(
        enable_profiling=True,
        enable_quantization=True,
        quantization_method="fp16"
    )
    
    # Sample protein sequence (E. coli maltose binding protein)
    sequence = "MKIKLIVVTALLTSVVFAFSSCGDDDDTGNEDDDDTGNEDDDDTGNEDDDDTGNENND"
    
    # Extract embeddings (you would use your actual model/alphabet)
    # embeddings = extractor.extract(
    #     sequence=sequence,
    #     model=your_esm_model,
    #     alphabet=your_alphabet,
    #     device=your_device
    # )
    
    print(f"Sequence: {sequence}")
    print(f"Length: {len(sequence)} amino acids")
    print("\n[Extraction would occur here with actual model]")
    
    # Get report
    # report = extractor.get_report()
    # print(f"\n✓ Extraction complete")
    # print(f"  Time: {report['average_time']:.4f}s")


# ==============================================================================
# Example 2: Batch Processing Multiple Sequences
# ==============================================================================

def example_batch_processing():
    """
    Process multiple protein sequences with performance tracking.
    
    Shows how to extract embeddings for a batch of sequences and track
    cumulative performance metrics.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Processing Multiple Sequences")
    print("="*70)
    
    # Initialize extractor once for batch
    extractor = OptimizedEmbeddingExtractor(
        enable_profiling=True,
        enable_quantization=True,
        quantization_method="fp16"
    )
    
    # Multiple sequences
    sequences = [
        "MKIKLIVVTALLTSVVFAFSSCGDDDD",
        "MKVLWALLLTAVTFLAGCAKAKPQDLLF",
        "MFITLGKHSDDLGTGSEPKLGSTQQ",
    ]
    
    embeddings_list = []
    
    print(f"Processing {len(sequences)} sequences...")
    
    for i, seq in enumerate(sequences, 1):
        print(f"  [{i}/{len(sequences)}] Length: {len(seq)}")
        # embeddings = extractor.extract(seq, model, alphabet, device)
        # embeddings_list.append(embeddings)
    
    # Get batch report
    # report = extractor.get_report()
    print(f"\n✓ Batch complete")
    # print(f"  Total sequences: {report['extraction_count']}")
    # print(f"  Total time: {report['total_time']:.2f}s")
    # print(f"  Average per sequence: {report['average_time']:.4f}s")
    # print(f"  Average speedup: {report['average_speedup']:.2f}x")


# ==============================================================================
# Example 3: Comparing Quantization Methods
# ==============================================================================

def example_compare_quantization():
    """
    Compare FP16 vs INT8 quantization for the same sequence.
    
    Shows how to measure performance and accuracy impact of different
    quantization strategies.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Comparing Quantization Methods")
    print("="*70)
    
    # Test sequence
    sequence = "MKIKLIVVTALLTSVVFAFSSCGDDDDTGNEDDDDTGNEDDDDTGNEDDDDTGNENND"
    
    methods = ["fp16", "int8"]
    results = {}
    
    for method in methods:
        print(f"\nTesting {method.upper()} quantization...")
        
        # Create extractor with specific method
        extractor = OptimizedEmbeddingExtractor(
            enable_profiling=True,
            enable_quantization=True,
            quantization_method=method,
            calibration_samples=100 if method == "int8" else None
        )
        
        # Would extract here with actual model
        # embeddings = extractor.extract(sequence, model, alphabet, device)
        
        # Collect metrics
        # report = extractor.get_report()
        # results[method] = report
        
        print(f"  ✓ {method.upper()} profiling complete")
    
    # Compare results
    print(f"\nComparison Summary:")
    print(f"  {'Method':<10} {'Avg Time':<12} {'Speedup':<10} {'Compression':<12}")
    print(f"  {'-'*50}")
    
    for method in methods:
        if method in results:
            report = results[method]
            # print(f"  {method:<10} {report['average_time']:.4f}s      {report['average_speedup']:.2f}x      {method}")


# ==============================================================================
# Example 4: Using Context Manager
# ==============================================================================

def example_context_manager():
    """
    Use context manager for automatic report generation.
    
    Demonstrates clean syntax and automatic profiling report at context exit.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Context Manager Usage")
    print("="*70)
    
    sequences = [
        "MKIKLIVVTALLTSVVFAFSSCGDDDD",
        "MKVLWALLLTAVTFLAGCAKAKPQ",
    ]
    
    print("Processing with context manager...")
    
    # Use context manager - report auto-prints on exit
    try:
        with EmbeddingOptimizationContext(quantization_method="fp16") as optimizer:
            for seq in sequences:
                print(f"  Processing: {seq[:20]}...")
                # embeddings = optimizer.extract(seq, model, alphabet, device)
        
        print("\n✓ Context manager exited - report printed above")
    except Exception as e:
        print(f"Note: {e}")


# ==============================================================================
# Example 5: Profiling Only (No Quantization)
# ==============================================================================

def example_profiling_only():
    """
    Profile extraction without quantization.
    
    Useful for understanding baseline performance before optimization.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Profiling Only (No Quantization)")
    print("="*70)
    
    # Disable quantization, keep profiling
    extractor = OptimizedEmbeddingExtractor(
        enable_profiling=True,
        enable_quantization=False  # Profiling only
    )
    
    sequences = [
        "MKIKLIVVTALLTSVVFAFSSCGDDDD",
        "MKVLWALLLTAVTFLAGCAKAKPQ",
        "MFITLGKHSDDLGTGSEPKL",
    ]
    
    print("Profiling baseline performance...")
    
    for seq in sequences:
        print(f"  Sequence: {seq[:20]}...")
        # embeddings = extractor.extract(seq, model, alphabet, device)
    
    # Get profiling-only report
    # report = extractor.get_report()
    print(f"\n✓ Profiling complete")
    # print(f"  Extractions: {report['extraction_count']}")
    # print(f"  Total time: {report['total_time']:.2f}s")
    # print(f"  Note: No quantization applied")


# ==============================================================================
# Example 6: Quantization Only (No Profiling)
# ==============================================================================

def example_quantization_only():
    """
    Apply quantization without profiling overhead.
    
    Useful for production deployments where you want maximum speed without
    profiling overhead.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Quantization Only (No Profiling)")
    print("="*70)
    
    # Disable profiling, keep quantization
    extractor = OptimizedEmbeddingExtractor(
        enable_profiling=False,  # No profiling overhead
        enable_quantization=True,
        quantization_method="fp16"
    )
    
    sequence = "MKIKLIVVTALLTSVVFAFSSCGDDDDTGNEDDDDTGNEDDDDTGNEDDDDTGNENND"
    
    print("Extracting with quantization (no profiling)...")
    print(f"Sequence: {sequence[:30]}...")
    
    # embeddings = extractor.extract(sequence, model, alphabet, device)
    # print(f"\n✓ Extraction complete")
    # print(f"  Embedding shape: {embeddings.shape}")
    # print(f"  Embedding dtype: {embeddings.dtype}")


# ==============================================================================
# Example 7: Identifying Bottlenecks
# ==============================================================================

def example_identify_bottleneck():
    """
    Identify performance bottlenecks in extraction pipeline.
    
    Shows how to find which component consumes most time.
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Identifying Bottlenecks")
    print("="*70)
    
    extractor = OptimizedEmbeddingExtractor(
        enable_profiling=True,
        enable_quantization=True
    )
    
    # Process multiple sequences to get meaningful statistics
    sequences = [
        "MKIKLIVVTALLTSVVFAFSSCGDDDD",
        "MKVLWALLLTAVTFLAGCAKAKPQ",
        "MFITLGKHSDDLGTGSEPKL",
        "MKIKLIVVTALLTSVVFAFSSCGDDDDTG",
    ]
    
    print(f"Processing {len(sequences)} sequences to identify bottleneck...")
    
    for seq in sequences:
        # embeddings = extractor.extract(seq, model, alphabet, device)
        pass
    
    # Get bottleneck info
    # bottleneck_component, bottleneck_time = extractor.get_bottleneck()
    # report = extractor.get_report()
    
    print(f"\n✓ Analysis complete")
    # print(f"  Main bottleneck: {bottleneck_component}")
    # print(f"  Time: {bottleneck_time:.4f}s")
    # print(f"  % of total: {(bottleneck_time / report['average_time'] * 100):.1f}%")
    # print(f"\n  Recommendations:")
    # if "model_forward" in bottleneck_component:
    #     print(f"    - Consider GPU acceleration")
    #     print(f"    - Try model quantization")
    # elif "quantization" in bottleneck_component:
    #     print(f"    - Switch to faster quantization method (FP16)")


# ==============================================================================
# Example 8: Saving Reports for Analysis
# ==============================================================================

def example_save_reports():
    """
    Save profiling reports to JSON for later analysis.
    
    Shows how to persist metrics for tracking performance over time.
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Saving Reports for Analysis")
    print("="*70)
    
    extractor = OptimizedEmbeddingExtractor(
        enable_profiling=True,
        enable_quantization=True,
        quantization_method="fp16"
    )
    
    # Simulate processing
    sequences = ["MKIKLIVVTALLTSVVFAFSSCGDDDD" * 2 for _ in range(3)]
    
    print(f"Processing {len(sequences)} sequences...")
    for seq in sequences:
        # embeddings = extractor.extract(seq, model, alphabet, device)
        pass
    
    # Save report
    report_path = Path("./optimization_reports/embedding_report.json")
    print(f"\nSaving report to: {report_path}")
    
    # extractor.save_report(report_path)
    print(f"✓ Report saved")
    
    # Show what would be in report
    # report = extractor.get_report()
    # print(f"\nReport contents:")
    # print(f"  Extractions: {report['extraction_count']}")
    # print(f"  Total time: {report['total_time']:.2f}s")
    # print(f"  Average speedup: {report['average_speedup']:.2f}x")


# ==============================================================================
# Example 9: Integration with Protein Classifier
# ==============================================================================

class ProteinClassifier:
    """
    Example protein classifier using optimized embeddings.
    
    Demonstrates how to integrate the optimization components into
    a real-world classification pipeline.
    """
    
    def __init__(self, model, alphabet, device):
        """Initialize classifier with model and embedding extractor."""
        self.model = model
        self.alphabet = alphabet
        self.device = device
        self.extractor = OptimizedEmbeddingExtractor(
            enable_profiling=True,
            enable_quantization=True,
            quantization_method="fp16"
        )
    
    def extract_embeddings(self, sequence: str) -> np.ndarray:
        """Extract optimized embeddings for sequence."""
        return self.extractor.extract(
            sequence,
            self.model,
            self.alphabet,
            self.device
        )
    
    def predict(self, sequence: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict protein class.
        
        Returns:
            (predictions, profiling_info)
        """
        # Extract embeddings with optimization
        embeddings = self.extract_embeddings(sequence)
        
        # Use embeddings for classification
        # predictions = self.classifier.predict(embeddings)
        
        # Get profiling info
        report = self.extractor.get_report()
        
        return embeddings, report
    
    def batch_predict(self, sequences: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict for batch of sequences."""
        embeddings_list = []
        
        for seq in sequences:
            emb = self.extract_embeddings(seq)
            embeddings_list.append(emb)
        
        embeddings = np.array(embeddings_list)
        report = self.extractor.get_report()
        
        return embeddings, report
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get current performance report."""
        return self.extractor.get_report()


def example_classifier_integration():
    """
    Show how to use optimized embeddings in a protein classifier.
    """
    print("\n" + "="*70)
    print("EXAMPLE 9: Integration with Protein Classifier")
    print("="*70)
    
    print("Creating classifier with optimized embeddings...")
    
    # classifier = ProteinClassifier(model, alphabet, device)
    
    print("Single prediction:")
    # predictions, report = classifier.predict("MKIKLIVVTALLTSVVFAFSSCGDDDD")
    # print(f"  Time: {report['average_time']:.4f}s")
    
    print("\nBatch prediction:")
    # sequences = ["MKIKLIVVTALLTSVVFAFSSCGDDDD", "MKVLWALLLTAVTFLAGCAKAKPQ"]
    # embeddings, report = classifier.batch_predict(sequences)
    # print(f"  Total time: {report['total_time']:.2f}s")
    # print(f"  Average: {report['average_time']:.4f}s")


# ==============================================================================
# Example 10: Custom Metrics and Advanced Usage
# ==============================================================================

def example_advanced_usage():
    """
    Advanced example showing custom metrics collection and analysis.
    """
    print("\n" + "="*70)
    print("EXAMPLE 10: Advanced Usage - Custom Analysis")
    print("="*70)
    
    extractor = OptimizedEmbeddingExtractor(
        enable_profiling=True,
        enable_quantization=True
    )
    
    # Different sequence lengths
    sequence_lengths = [20, 50, 100, 200]
    timing_by_length = {}
    
    print("Analyzing performance across sequence lengths...")
    
    for length in sequence_lengths:
        seq = "M" + "K" * (length - 1)  # Synthetic sequence
        
        # Would extract here
        # embeddings = extractor.extract(seq, model, alphabet, device)
        
        # Report = extractor.get_report()
        # timing_by_length[length] = report['average_time']
        
        print(f"  Length {length}: [Would measure here]")
    
    print(f"\n✓ Analysis complete")
    # print(f"\nTiming by sequence length:")
    # for length, time in timing_by_length.items():
    #     print(f"  {length:4d} aa: {time:.4f}s")


# ==============================================================================
# Main: Run All Examples
# ==============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("TIER 3.1 EMBEDDING OPTIMIZATION - PRACTICAL EXAMPLES")
    print("="*70)
    
    # Run examples
    example_basic_extraction()
    example_batch_processing()
    example_compare_quantization()
    example_context_manager()
    example_profiling_only()
    example_quantization_only()
    example_identify_bottleneck()
    example_save_reports()
    example_classifier_integration()
    example_advanced_usage()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
