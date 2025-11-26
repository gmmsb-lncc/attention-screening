# Tier 3.1 Embedding Optimization Integration Guide

## Overview

The Tier 3.1 Embedding Optimization system integrates profiling and quantization into the protein embedding extraction pipeline. This guide provides comprehensive instructions for using these components.

## Components Overview

### 1. **EmbeddingProfiler** (`embedding_profiler.py`)
Measures timing and memory for embedding extraction components.

- **Component tracking**: Profile individual phases (tokenization, forward pass, quantization)
- **Memory monitoring**: Track memory usage before, after, and peak
- **Statistics aggregation**: Generate comprehensive profiling reports

**Features:**
- Per-component timing
- Memory peak tracking
- Statistics (min, max, mean, median)
- Timestamp recording

### 2. **EmbeddingQuantizer** (`embedding_quantizer.py`)
Reduces embedding size using FP16 or INT8 quantization.

- **FP16 quantization**: 50% reduction with minimal accuracy loss
- **INT8 quantization**: 75% reduction with calibration
- **Dynamic quantization**: Range-based scaling for INT8

**Supported Methods:**
- `fp16`: Convert float32 to float16
- `int8`: Symmetric quantization with calibration
- `auto`: Automatic selection based on input

### 3. **OptimizedEmbeddingExtractor** (`embedding_integration.py`)
Main integration point that combines profiling and quantization.

## Installation & Setup

### Prerequisites

```bash
pip install torch numpy PyYAML
```

### Import in Your Project

```python
from src.classifier.core.embedding_integration import OptimizedEmbeddingExtractor
from src.classifier.core.embedding_profiler import EmbeddingProfiler
from src.classifier.core.embedding_quantizer import EmbeddingQuantizer
```

## Usage Examples

### Basic Usage: Extract with Profiling and Quantization

```python
from src.classifier.core.embedding_integration import OptimizedEmbeddingExtractor

# Initialize extractor
extractor = OptimizedEmbeddingExtractor(
    enable_profiling=True,
    enable_quantization=True,
    quantization_method="fp16"  # or "int8"
)

# Extract embeddings
sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGNEDDDDTGNEDDDDTGNEDDDDTGNENND"
embeddings = extractor.extract(
    sequence=sequence,
    model=your_model,
    alphabet=your_alphabet,
    device=your_device,
    batch_converter=your_batch_converter  # Optional
)

# Get performance report
report = extractor.get_report()
print(report)
```

### Using Context Manager

```python
from src.classifier.core.embedding_integration import EmbeddingOptimizationContext

with EmbeddingOptimizationContext(quantization_method="fp16") as optimizer:
    embeddings = optimizer.extract(sequence, model, alphabet, device)

# Report is automatically printed on context exit
```

### Advanced Configuration

```python
# Custom configuration for INT8 quantization
extractor = OptimizedEmbeddingExtractor(
    enable_profiling=True,
    enable_quantization=True,
    quantization_method="int8",
    calibration_samples=200,  # For INT8 calibration
    logger=your_logger
)

# Extract multiple sequences
sequences = ["SEQUENCE1", "SEQUENCE2", "SEQUENCE3"]
embeddings_list = []

for seq in sequences:
    emb = extractor.extract(seq, model, alphabet, device)
    embeddings_list.append(emb)

# Get accumulated report
report = extractor.get_report()
print(f"Processed {report['extraction_count']} sequences")
print(f"Average time: {report['average_time']:.4f}s")
print(f"Total speedup: {report['average_speedup']:.2f}x")

# Save report to file
from pathlib import Path
extractor.save_report(Path("optimization_report.json"))
```

### Using Just Profiling (No Quantization)

```python
extractor = OptimizedEmbeddingExtractor(
    enable_profiling=True,
    enable_quantization=False  # Disable quantization
)

embeddings = extractor.extract(sequence, model, alphabet, device)
report = extractor.get_report()
```

### Using Just Quantization (No Profiling)

```python
extractor = OptimizedEmbeddingExtractor(
    enable_profiling=False,  # Disable profiling
    enable_quantization=True,
    quantization_method="fp16"
)

embeddings = extractor.extract(sequence, model, alphabet, device)
```

### Identifying Bottlenecks

```python
extractor = OptimizedEmbeddingExtractor()

# Extract from multiple sequences
for seq in sequences:
    extractor.extract(seq, model, alphabet, device)

# Find bottleneck
bottleneck_component, bottleneck_time = extractor.get_bottleneck()
print(f"Main bottleneck: {bottleneck_component}")
print(f"Time: {bottleneck_time:.4f}s")
```

## API Reference

### OptimizedEmbeddingExtractor

#### Constructor

```python
OptimizedEmbeddingExtractor(
    enable_profiling: bool = True,
    enable_quantization: bool = True,
    quantization_method: str = "fp16",
    calibration_samples: Optional[int] = None,
    logger: Optional[logging.Logger] = None
)
```

**Parameters:**
- `enable_profiling`: Enable component timing and memory tracking
- `enable_quantization`: Enable FP16/INT8 quantization
- `quantization_method`: "fp16", "int8", or "auto"
- `calibration_samples`: Number of samples for INT8 calibration (default: 100)
- `logger`: Custom logger instance (optional)

#### Methods

**extract(sequence, model, alphabet, device, batch_converter=None) → np.ndarray**
```python
embeddings = extractor.extract(
    sequence="MKVLWALLLTAVTFLAGCAKAKPQ...",
    model=esm_model,
    alphabet=esm_alphabet,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_converter=batch_converter  # Optional for ESM-2
)
```

**get_report() → Dict[str, Any]**
```python
report = extractor.get_report()
# Returns:
# {
#     "extraction_count": 10,
#     "total_time": 5.234,
#     "average_time": 0.523,
#     "min_time": 0.450,
#     "max_time": 0.680,
#     "average_speedup": 1.25,
#     "quantization_method": "fp16",
#     "last_metric": {...}
# }
```

**get_bottleneck() → Tuple[str, float]**
```python
component, time = extractor.get_bottleneck()
print(f"Bottleneck: {component} takes {time:.4f}s")
# Example output: Bottleneck: model_forward takes 0.315s
```

**save_report(output_file: Path) → None**
```python
from pathlib import Path
extractor.save_report(Path("reports/embedding_optimization.json"))
```

**reset_metrics() → None**
```python
extractor.reset_metrics()  # Clear history, start fresh
```

### EmbeddingProfiler

#### Component Tracking

```python
profiler = EmbeddingProfiler()

# Start profiling a component
profiler.start_component("tokenization")
# ... do work ...
profiler.end_component("tokenization")

# Get stats
stats = profiler.get_stats("tokenization")
print(f"Count: {stats['count']}, Avg: {stats['avg']:.4f}s")
```

#### Memory Monitoring

```python
# Get current memory
memory = profiler.get_memory()
print(f"Current memory: {memory / 1024**2:.2f} MB")

# Memory peak
peak = profiler.get_memory_peak()
print(f"Peak memory: {peak / 1024**2:.2f} MB")
```

### EmbeddingQuantizer

#### Direct Usage

```python
from src.classifier.core.embedding_quantizer import EmbeddingQuantizer, QuantizationConfig

# Create quantizer
config = QuantizationConfig(method="fp16", preserve_accuracy=True)
quantizer = EmbeddingQuantizer(config)

# Quantize embeddings
embeddings = np.random.randn(1024).astype(np.float32)
quantized = quantizer.quantize_fp16(embeddings)

print(f"Original: {embeddings.dtype}, shape {embeddings.shape}")
print(f"Quantized: {quantized.dtype}, shape {quantized.shape}")
print(f"Reduction: {embeddings.nbytes / quantized.nbytes:.2f}x")
```

#### INT8 Quantization with Calibration

```python
# Create quantizer with calibration
config = QuantizationConfig(
    method="int8",
    preserve_accuracy=True,
    calibration_samples=100,
    dynamic=True
)
quantizer = EmbeddingQuantizer(config)

# Calibrate with sample data
calibration_data = [np.random.randn(1024).astype(np.float32) for _ in range(100)]
quantizer.calibrate(calibration_data)

# Quantize
quantized, scale = quantizer.quantize_int8(embeddings)
print(f"Scale factor: {scale}")
```

## Performance Metrics

### Expected Performance Characteristics

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| Tokenization | 1-5ms | 1-10MB | Highly sequence-dependent |
| Model Forward | 400-600ms | 2-4GB | Main bottleneck |
| FP16 Quantization | 1-2ms | <1MB | Minimal overhead |
| INT8 Quantization | 2-5ms | <1MB | Includes calibration |

### Optimization Impact

**FP16 Quantization:**
- Memory reduction: ~50%
- Accuracy preservation: >99%
- Speed impact: Negligible (<1%)

**INT8 Quantization:**
- Memory reduction: ~75%
- Accuracy preservation: >95%
- Speed impact: Negligible (<1%)

## Integration with Existing Pipelines

### Example: Integration with Protein Classification

```python
from src.classifier.core.embedding_integration import OptimizedEmbeddingExtractor

class ProteinClassifier:
    def __init__(self, model, alphabet, device):
        self.model = model
        self.alphabet = alphabet
        self.device = device
        self.extractor = OptimizedEmbeddingExtractor(
            enable_profiling=True,
            enable_quantization=True,
            quantization_method="fp16"
        )
    
    def predict(self, sequence):
        # Extract optimized embeddings
        embeddings = self.extractor.extract(
            sequence, self.model, self.alphabet, self.device
        )
        
        # Use embeddings for classification
        predictions = self.classifier.predict(embeddings)
        return predictions
    
    def get_performance_report(self):
        return self.extractor.get_report()
```

### Example: Batch Processing

```python
extractor = OptimizedEmbeddingExtractor()

def process_protein_batch(sequences, model, alphabet, device):
    embeddings_list = []
    
    for seq in sequences:
        emb = extractor.extract(seq, model, alphabet, device)
        embeddings_list.append(emb)
    
    # Get report after batch
    report = extractor.get_report()
    print(f"Batch processing complete:")
    print(f"  Sequences: {report['extraction_count']}")
    print(f"  Total time: {report['total_time']:.2f}s")
    print(f"  Avg/seq: {report['average_time']:.4f}s")
    
    return embeddings_list, report
```

## Troubleshooting

### Issue: Quantization Failed, Using Unquantized

**Cause:** Incompatible tensor shape or configuration
**Solution:** Check tensor dimensions match expected size

```python
# Verify embedding dimensions
embeddings = extractor.extract(sequence, model, alphabet, device)
print(f"Embedding shape: {embeddings.shape}")  # Should be 1D or 2D
```

### Issue: Memory Usage Not Decreasing with Quantization

**Cause:** Embeddings may be copied, not replaced
**Solution:** Ensure quantized embeddings are used

```python
# Verify quantization
original_size = embeddings.nbytes
quantized = quantizer.quantize_fp16(embeddings)
print(f"Size reduction: {original_size} → {quantized.nbytes} bytes")
```

### Issue: Profile Report Shows High Model_Forward Time

**Cause:** Normal - model inference is the main bottleneck
**Solution:** Consider:
- GPU acceleration
- Model compression
- Batch processing

```python
# Check if GPU is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embeddings = extractor.extract(sequence, model, alphabet, device)
print(f"Device: {device}")
```

## Best Practices

1. **Always enable profiling in development** - Understand performance characteristics
2. **Start with FP16 quantization** - Good balance of compression and accuracy
3. **Use INT8 for memory-critical scenarios** - Higher compression with calibration
4. **Batch process when possible** - Amortize overhead
5. **Monitor metrics regularly** - Track optimization effectiveness
6. **Save reports for analysis** - Identify trends and bottlenecks

## Advanced Topics

### Custom Metrics Collection

```python
# Extend ExtractionMetrics for custom tracking
from dataclasses import dataclass

@dataclass
class CustomMetrics(ExtractionMetrics):
    custom_field: str = ""

# Use in custom extractor subclass
class CustomExtractor(OptimizedEmbeddingExtractor):
    def _create_metrics(self, sequence, embeddings, start_time):
        base_metrics = super()._create_metrics(sequence, embeddings, start_time)
        # Add custom logic
        return base_metrics
```

### Profiler Integration with Logging

```python
import logging

# Configure logging for profiler
logger = logging.getLogger(__name__)
handler = logging.FileHandler("profiler.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Use with extractor
extractor = OptimizedEmbeddingExtractor(logger=logger)
```

## See Also

- [Tier 3.0 Embedding Profiling](./embedding_tier3_profiling.md)
- [Tier 3.0 Embedding Quantization](./embedding_tier3_quantization.md)
- [Performance Benchmarks](../06-validation-reports/)
- [Architecture Overview](../03-architecture/)

---

**Last Updated:** 2024
**Module Location:** `src/classifier/core/embedding_*.py`
