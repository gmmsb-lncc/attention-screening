# Tier 3.1 Embedding Optimization - Quick Reference

## 📋 Quick Syntax Reference

### Import
```python
from src.classifier.core.embedding_integration import OptimizedEmbeddingExtractor
```

### Initialization
```python
# Standard setup
extractor = OptimizedEmbeddingExtractor()

# With options
extractor = OptimizedEmbeddingExtractor(
    enable_profiling=True,
    enable_quantization=True,
    quantization_method="fp16",  # or "int8"
)
```

### Extract Embeddings
```python
embeddings = extractor.extract(
    sequence="MKVLWALLLTAVTFLAGCAKAKPQ",
    model=model,
    alphabet=alphabet,
    device=device
)
```

### Get Metrics
```python
report = extractor.get_report()
print(f"Average time: {report['average_time']:.4f}s")
print(f"Speedup: {report['average_speedup']:.2f}x")
```

### Save Report
```python
from pathlib import Path
extractor.save_report(Path("report.json"))
```

## ⚡ Common Patterns

### Batch Processing
```python
extractor = OptimizedEmbeddingExtractor()

for sequence in sequences:
    embeddings = extractor.extract(sequence, model, alphabet, device)
    # Use embeddings...

report = extractor.get_report()  # Get cumulative stats
```

### Find Bottleneck
```python
component, time = extractor.get_bottleneck()
print(f"Main bottleneck: {component} ({time:.4f}s)")
```

### Context Manager
```python
with EmbeddingOptimizationContext(quantization_method="fp16") as opt:
    embeddings = opt.extract(sequence, model, alphabet, device)
```

### Profiling Only
```python
extractor = OptimizedEmbeddingExtractor(
    enable_profiling=True,
    enable_quantization=False
)
```

### Quantization Only
```python
extractor = OptimizedEmbeddingExtractor(
    enable_profiling=False,
    enable_quantization=True
)
```

## 📊 Output Interpretation

### Report Dictionary
| Key | Meaning |
|-----|---------|
| `extraction_count` | Number of extractions performed |
| `total_time` | Sum of all extraction times |
| `average_time` | Mean extraction time per sequence |
| `min_time` | Fastest extraction |
| `max_time` | Slowest extraction |
| `average_speedup` | Performance improvement factor |
| `quantization_method` | Method used (fp16/int8 or None) |

### Example Report
```json
{
  "extraction_count": 10,
  "total_time": 5.234,
  "average_time": 0.523,
  "min_time": 0.450,
  "max_time": 0.680,
  "average_speedup": 1.25,
  "quantization_method": "fp16"
}
```

## 🎯 Configuration Guide

| Scenario | Config |
|----------|--------|
| **Development** | Profiling + FP16 |
| **Production** | FP16 only (no profiling) |
| **Accuracy Critical** | Profiling only (no quantization) |
| **Memory Critical** | INT8 with calibration |
| **Speed Priority** | Quantization only (no profiling) |
| **Baseline Measurement** | Profiling only (no quantization) |

## 🐛 Troubleshooting Checklist

- [ ] Torch installed: `pip install torch`
- [ ] GPU available (optional): `torch.cuda.is_available()`
- [ ] Sequences are strings: `isinstance(sequence, str)`
- [ ] Model is on correct device: `model.to(device)`
- [ ] Report shows reasonable times: >0s, <10s per sequence
- [ ] Quantization not failing: Check logs for warnings

## 📈 Performance Targets

| Metric | Expected |
|--------|----------|
| Avg time | 0.3-1.0s per sequence |
| Speedup | 1.0-2.0x |
| Memory reduction (FP16) | ~50% |
| Memory reduction (INT8) | ~75% |
| Accuracy (FP16) | >99% |
| Accuracy (INT8) | >95% |

## 🔗 Related Files

- Implementation: `src/classifier/core/embedding_*.py`
- Guide: `docs/04-modules/embedding_tier3_integration_guide.md`
- Examples: `examples/tier3_embedding_optimization_examples.py`
- Overview: `docs/04-modules/embedding_tier3_complete_implementation.md`

## ⚙️ Method Comparison

| Method | Speed | Memory | Accuracy | Use |
|--------|-------|--------|----------|-----|
| FP16 | Fast | 50% | >99% | Default |
| INT8 | Fast | 75% | >95% | Strict limits |
| None | Baseline | 100% | 100% | Comparison |

## 💡 Tips & Tricks

1. **Start with FP16**: Good balance of compression and accuracy
2. **Profile first**: Identify bottlenecks before optimizing
3. **Batch process**: More efficient than single sequences
4. **Save reports**: Track performance over time
5. **Disable profiling in production**: Reduces overhead
6. **Check GPU**: Verify CUDA availability for speed

## 🚀 Integration Checklist

- [ ] Install torch: `pip install torch`
- [ ] Import extractor class
- [ ] Create extractor instance
- [ ] Call extract() method
- [ ] Check report metrics
- [ ] Save report if needed
- [ ] Reset metrics for new batch (optional)

## 📝 Template Code

```python
# Setup
from src.classifier.core.embedding_integration import OptimizedEmbeddingExtractor

extractor = OptimizedEmbeddingExtractor(
    enable_profiling=True,
    enable_quantization=True,
    quantization_method="fp16"
)

# Process
sequences = ["SEQ1", "SEQ2", "SEQ3"]
results = []

for seq in sequences:
    emb = extractor.extract(seq, model, alphabet, device)
    results.append(emb)

# Analyze
report = extractor.get_report()
print(f"✓ Processed {report['extraction_count']} sequences")
print(f"  Average time: {report['average_time']:.4f}s")
print(f"  Speedup: {report['average_speedup']:.2f}x")

# Save
from pathlib import Path
extractor.save_report(Path("results/report.json"))
```

## 📞 Support Resources

| Issue | Resource |
|-------|----------|
| API details | `embedding_tier3_integration_guide.md` |
| Code examples | `tier3_embedding_optimization_examples.py` |
| How it works | `embedding_tier3_complete_implementation.md` |
| Source code | `embedding_*.py` files |

---

**Use this quick reference to get started immediately!**
