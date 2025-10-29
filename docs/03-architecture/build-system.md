# Build System

**Last Updated**: October 28, 2025  
**Section**: Chapter 03 - Architecture  
**Audience**: Developers

---

## Overview

The build system orchestrates the complete pipeline from raw data to trained models, managing dependencies, caching, and automated workflows.

## Architecture

### Components

1. **Build Pipeline** (`src/build/pipeline.py`)
   - Orchestrates complete workflow
   - Manages pipeline stages
   - Handles errors and retries

2. **Embeddings Generator** (`src/build/embeddings.py`)
   - Protein embeddings (ESM-2)
   - Ligand embeddings (FM4M/SMI-TED)
   - Embedding cache management

3. **Data Stratification** (`src/build/stratification.py`)
   - Train/test splitting
   - Stratified sampling
   - Class balance validation

4. **Matrix Builder** (`src/build/matrix.py`)
   - Feature matrix construction
   - Sparse matrix optimization
   - NPZ format output

---

## Build Process

### Stage 1: Data Loading
```python
from src.build import BuildPipeline

pipeline = BuildPipeline(
    input_tsv='data/kinase_data.tsv',
    output_dir='results/experiment'
)
```

### Stage 2: Embedding Generation
- Protein sequences → ESM-2 embeddings (1280-dim)
- Ligand SMILES → FM4M/SMI-TED embeddings (varies)
- Cached in `results/<experiment>/embeddings/`

### Stage 3: Stratification
- Train/test split (default 80/20)
- Stratified by class labels
- Configurable via `stratification_config.json`

### Stage 4: Matrix Construction
- Combine protein + ligand embeddings
- Build feature matrix
- Save as `.npz` (compressed NumPy)

### Stage 5: Model Training
- Classification models (6 models)
- Regression models (11 models)
- Parallel training support

---

## Configuration

### Build Configuration (`src/build/core.py`)

```python
from src.build import BuildConfig

config = BuildConfig(
    batch_size=32,
    use_cache=True,
    n_jobs=-1,
    device='cuda'  # or 'cpu'
)
```

### Stratification Configuration (`src/stratification_config.json`)

```json
{
  "test_size": 0.2,
  "random_state": 42,
  "stratify": true,
  "shuffle": true
}
```

---

## Caching Strategy

### Embedding Cache
- Location: `results/<experiment>/embeddings/`
- Format: `.pt` (PyTorch tensors)
- Reused across runs (significant speedup)

### Model Cache
- Location: `models_cache/`
- ESM-2 models cached locally
- Avoids repeated downloads

---

## Automated Testing

### Test Suite (19 tests)
```bash
pytest tests/build/ -v
```

Tests cover:
- Pipeline execution
- Embedding generation
- Stratification validation
- Matrix construction
- Error handling

---

## Performance

### Optimizations
- **Batch Processing**: Process multiple sequences simultaneously
- **GPU Acceleration**: ESM-2 inference on CUDA
- **Caching**: Avoid redundant computations
- **Parallel Jobs**: Multi-core model training

### Benchmarks
| Dataset Size | Build Time | With Cache |
|--------------|------------|------------|
| 1,000 samples | ~5 min | ~30 sec |
| 10,000 samples | ~30 min | ~3 min |
| 100,000 samples | ~4 hours | ~20 min |

---

## Related Documentation

- **Pipeline Architecture**: [Dual Pipeline](dual-pipeline.md)
- **Build Module**: [Chapter 04: Build Module](../04-modules/build-module.md)
- **Testing**: [Chapter 05: Testing Guide](../05-development/testing-guide.md)

---

**Previous**: [← Modularization](modularization.md) | **Next**: [Dual Pipeline →](dual-pipeline.md)
