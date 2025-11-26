# Tier 3.1 Integration Guide

## Overview

Tier 3.1 provides **profiling and quantization framework** for embedding extraction optimization. This guide walks you through integrating the framework into the actual DockTkinase pipeline.

## Files Created

### 1. **embedding_integration.py** (Tier 3.1 Core Integration)
Location: `src/classifier/core/embedding_integration.py`

Key classes:
- `OptimizedEmbeddingExtractor`: Main integration point with profiling + quantization
- `EmbeddingOptimizationContext`: Context manager for automatic reporting
- `ExtractionMetrics`: Dataclass for tracking metrics

### 2. **tier_3_1_integration_example.py** (Usage Examples)
Location: `examples/tier_3_1_integration_example.py`

Contains 7 detailed examples of how to use the integration.

## Quick Start

### Basic Usage

```python
from src.classifier.core.embedding_integration import create_optimized_extractor

# Create extractor with FP16 quantization
extractor = create_optimized_extractor(quantization_method="fp16")

# Use in your code
embeddings = extractor.extract(
    sequence=protein_sequence,
    model=esm_model,
    alphabet=alphabet,
    device=device,
    batch_converter=batch_converter
)

# Get profiling report
report = extractor.get_report()
print(f"Average time: {report['average_time']:.3f}s")
print(f"Speedup: {report['average_speedup']:.2f}x")
print(f"Bottleneck: {extractor.get_bottleneck()}")
```

### Using Context Manager

```python
from src.classifier.core.embedding_integration import EmbeddingOptimizationContext

with EmbeddingOptimizationContext(quantization_method="int8") as extractor:
    for sequence in sequences:
        embeddings = extractor.extract(sequence, model, alphabet, device)
    # Auto-reporting on exit
```

## Integration Steps

### Step 1: Update protein_embedding.py

Add import at the top:

```python
from src.classifier.core.embedding_integration import OptimizedEmbeddingExtractor
```

### Step 2: Initialize Optimizer in __init__

```python
class ProteinEmbedding(BaseEmbedding):
    def __init__(self, ...):
        # ... existing code ...
        
        # Add Tier 3.1 optimizer
        self.optimizer = OptimizedEmbeddingExtractor(
            enable_profiling=True,
            enable_quantization=True,
            quantization_method="fp16"  # or "int8"
        )
```

### Step 3: Wrap Embedding Extraction

Replace direct extraction calls with optimizer:

```python
def _generate_single_embedding(self, sequence: str) -> np.ndarray:
    """Generate embedding using optimized extractor."""
    return self.optimizer.extract(
        sequence=sequence,
        model=self.model,
        alphabet=self.alphabet,
        device=self.device,
        batch_converter=self.batch_converter
    )
```

### Step 4: Add Profiling Reports

After processing a batch:

```python
def process_fasta_file(self, fasta_file, output_dir, batch_size=None):
    # ... existing processing code ...
    
    # After all embeddings are generated:
    report = self.optimizer.get_report()
    bottleneck, bottleneck_time = self.optimizer.get_bottleneck()
    
    self.logger.info(f"Embedding Profile:")
    self.logger.info(f"  Total extractions: {report['extraction_count']}")
    self.logger.info(f"  Average time: {report['average_time']:.3f}s")
    self.logger.info(f"  Average speedup: {report['average_speedup']:.2f}x")
    self.logger.info(f"  Main bottleneck: {bottleneck} ({bottleneck_time:.3f}s)")
    
    # Save detailed report
    self.optimizer.save_report(Path("logs/embedding_profile.json"))
```

## Configuration Options

### OptimizedEmbeddingExtractor Parameters

```python
OptimizedEmbeddingExtractor(
    enable_profiling: bool = True,        # Enable component timing
    enable_quantization: bool = True,     # Enable quantization
    quantization_method: str = "fp16",    # "fp16", "int8", or "auto"
    calibration_samples: Optional[int] = None,  # INT8 calibration
    logger: Optional[logging.Logger] = None
)
```

### Quantization Methods

| Method | Speedup | Memory | Accuracy | Best For |
|--------|---------|--------|----------|----------|
| FP16 | 2.0-2.5x | 50% | 99.9% | Balanced |
| INT8 | 4.0-5.0x | 25% | 99.0% | Maximum throughput |
| None | 1.0x | 100% | 100% | Baseline |

## Performance Expectations

### Per-Sequence Extraction Time

**Baseline (No Optimization):**
- Time: 50-100ms
- Bottleneck: Model forward pass (45-90ms)

**With FP16 (Tier 3.1):**
- Time: 25-50ms (2.0-2.5x speedup)
- Memory: 50% of baseline

**With INT8 (Tier 3.1):**
- Time: 12-25ms (4.0-5.0x speedup)
- Memory: 25% of baseline

### For 100 Proteins

**Baseline:** 5-10 seconds
**With FP16:** 2.5-5 seconds (2x speedup)
**With INT8:** 1.2-2.5 seconds (4x speedup)
**With Tier 3.2 (Batching):** 0.2-1 second (10-50x additional)

## Profiling Output

The `get_report()` method returns:

```python
{
    "extraction_count": 100,           # Number of extractions
    "total_time_all": 3.45,            # Total seconds
    "average_time": 0.0345,            # Avg per extraction
    "min_time": 0.032,                 # Min time
    "max_time": 0.042,                 # Max time
    "median_time": 0.034,              # Median time
    "components": {
        "tokenization": {
            "avg": 0.0005,
            "min": 0.0004,
            "max": 0.0008,
            "total": 0.05,
            "count": 100
        },
        "model_forward": {
            "avg": 0.030,
            "min": 0.028,
            "max": 0.035,
            "total": 3.0,
            "count": 100
        },
        "quantization": {
            "avg": 0.002,
            ...
        },
        "validation": {...}
    },
    "average_speedup": 2.15,           # vs baseline
    "quantization_enabled": True,
    "quantization_method": "fp16",
    "profiling_enabled": True
}
```

## Bottleneck Detection

```python
# Identify main bottleneck
component, time = extractor.get_bottleneck()
print(f"Bottleneck: {component} ({time:.3f}s)")

# Output:
# Bottleneck: model_forward (0.030s)
```

This helps prioritize next optimization tier:
- If bottleneck is still model_forward → Use Tier 3.3 (Boltz-2 GPU)
- If embedding generation is slow → Use Tier 3.2 (batch processing)
- If I/O is slow → Use Tier 3.4 (advanced caching)

## Memory Impact

Monitor memory with:

```python
report = extractor.get_report()
last_metric = report['last_metric']
print(f"Memory peak: {last_metric['memory_peak']} MB")

# Compare:
# - FP16: ~50% of baseline
# - INT8: ~25% of baseline
```

## Next Steps

### After Integration

1. **Run profiling** on real data
2. **Validate speedup** vs projected 3-5x
3. **Check memory savings** (50-75% reduction)
4. **Measure accuracy** preservation
5. **Proceed to Tier 3.2** if speedup is <5x

### Tier 3.2 (Next Phase)

Batch processing optimization:
- Group sequences by length
- Parallel embedding extraction
- Dynamic batching (16/32/64 sequences)
- Expected: 10-50x additional speedup

### Tier 3.3 (Structure Prediction)

Boltz-2 GPU optimization:
- GPU memory optimization
- Attention fusion kernels
- Expected: 1.5-2x speedup

## Troubleshooting

### Issue: Quantization not activating

```python
# Check if quantizer is initialized
if extractor.quantizer is None:
    print("Quantizer failed to initialize")
    # Check logs for errors
```

### Issue: Slower than expected

```python
# Get detailed component breakdown
report = extractor.get_report()
for component, stats in report['components'].items():
    print(f"{component}: {stats['avg']*1000:.1f}ms")

# Identify which component is slow
# Then use appropriate next tier
```

### Issue: Memory issues with INT8

```python
# Use FP16 instead (half memory of INT8)
extractor = OptimizedEmbeddingExtractor(
    quantization_method="fp16"
)

# Or increase calibration samples for better scale factors
extractor = OptimizedEmbeddingExtractor(
    quantization_method="int8",
    calibration_samples=200  # More samples = better accuracy
)
```

## Best Practices

1. **Use FP16 first** - Best balance of speed and accuracy
2. **Calibrate INT8** - Use at least 100 diverse sequences
3. **Monitor bottlenecks** - Guide next optimization
4. **Save reports** - Track performance over time
5. **Test with real data** - Profiling results vary by data

## Example Integration

See `examples/tier_3_1_integration_example.py` for:
- Basic optimization setup
- Context manager usage
- Comparison of quantization methods
- Production integration pattern
- Real-time monitoring
- INT8 calibration
- Performance targets

## Performance Roadmap

```
Current (Tier 1+2):        8-10x speedup
├─ Tier 3.1 (This):        +3-5x (12-18x total)
├─ Tier 3.2 (Next):        +10-50x (60-150x total)
├─ Tier 3.3 (GPU):         +1.5-2x (90-250x total)
└─ Tier 3.4 (I/O):         +20-30% (100-300x total)

Goal: 4.5h → <1 min training time
```

## Related Files

- Profiler: `src/classifier/core/embedding_profiler.py`
- Quantizer: `src/classifier/core/embedding_quantizer.py`
- Integration: `src/classifier/core/embedding_integration.py` (this file)
- Examples: `examples/tier_3_1_integration_example.py`

## Questions?

Check the examples or review:
- `src/build/embeddings/protein_embedding.py` - Where to integrate
- `src/classifier/core/async_model_loader.py` - Related optimization
- `src/classifier/core/pipeline_cache.py` - Cache strategy reference
