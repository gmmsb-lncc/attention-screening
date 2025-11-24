# Modular Embeddings Architecture

## 📋 Overview

This document describes the **modular architecture** for the embeddings generation module. The modularization follows the same proven pattern used in the regression module, providing better maintainability, testability, and extensibility.

## 🏗️ Architecture

### Directory Structure

```
src/build/embeddings/
├── core/                        # Core embedding generation logic
│   ├── __init__.py
│   ├── data_loader.py          # DataManager: Load sequences/SMILES
│   ├── model_manager.py        # ModelManager: Manage ESM/FM4M models
│   └── generator.py            # EmbeddingGenerator: Generate embeddings
│
├── models/                      # Model definitions and registry
│   ├── __init__.py
│   └── model_registry.py       # ModelRegistry: ESM/FM4M model catalog
│
├── utils/                       # Supporting utilities
│   ├── __init__.py
│   ├── cache.py                # CacheManager: Cache embeddings
│   └── validators.py           # Validation functions
│
├── modular_pipeline.py         # EmbeddingPipeline: Main orchestrator
├── README_MODULAR.md           # This file
│
└── (original files maintained for 100% backward compatibility)
    ├── __init__.py
    ├── base_embedding.py
    ├── protein_embedding.py
    └── ligand_embedding.py
```

## 🎯 Design Principles

### 1. Separation of Concerns
- **DataManager**: Handles all data loading (FASTA, CSV, lists)
- **ModelManager**: Manages model lifecycle (loading, caching, device management)
- **EmbeddingGenerator**: Generates embeddings using loaded models
- **CacheManager**: Handles embedding caching (memory + disk)
- **Validators**: Input validation logic

### 2. Single Responsibility
Each class has one clear purpose and does it well.

### 3. Dependency Injection
Components receive dependencies through constructors, making testing easier.

### 4. Backward Compatibility
Original classes (`BaseEmbedding`, `ProteinEmbedding`, `LigandEmbedding`) remain unchanged.

## 🚀 Quick Start

### Basic Usage - Protein Embeddings

```python
from build.embeddings.modular_pipeline import EmbeddingPipeline

# Initialize pipeline
pipeline = EmbeddingPipeline(use_gpu=True)

# Generate embeddings from list
sequences = ['MKTAYIAKQRQISFVK', 'ARNDCEQGHILKMFPSTWYV']
embeddings = pipeline.generate_protein_embeddings(
    source=sequences,
    model_name='esm2_t33_650M_UR50D'
)

print(f"Generated embeddings shape: {embeddings.shape}")
```

### Basic Usage - Ligand Embeddings

```python
# Generate ligand embeddings
smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O']
embeddings = pipeline.generate_ligand_embeddings(
    source=smiles_list,
    model_name='smi_ted_light'
)

print(f"Generated embeddings shape: {embeddings.shape}")
```

### Load from File

```python
# From FASTA file
embeddings = pipeline.generate_protein_embeddings(
    source='protein_sequences.fasta',
    output_path='protein_embeddings.npy'
)

# From CSV file
embeddings = pipeline.generate_protein_embeddings(
    source='sequences.csv',
    sequence_column='sequence',
    id_column='protein_id'
)
```

### Advanced Usage

```python
from pathlib import Path

# Custom configuration
pipeline = EmbeddingPipeline(
    use_gpu=True,
    cache_dir=Path('embeddings_cache'),
    batch_size=64,
    verbose=True
)

# Generate with all options
embeddings = pipeline.generate_protein_embeddings(
    source='sequences.fasta',
    model_name='esm2_t36_3B_UR50D',
    repr_layer=36,
    validate=True,        # Validate sequences
    use_cache=True,       # Use caching
    output_path='embeddings.npz'
)
```

## 📦 Component Details

### 1. DataManager (`core/data_loader.py`)

Handles loading and preprocessing of input data.

**Features:**
- Load from multiple formats (FASTA, CSV, TSV, lists, DataFrames)
- Automatic format detection
- Sequence validation
- Batch creation
- Statistics generation

**Example:**
```python
from build.embeddings.core import DataManager

dm = DataManager()

# Load from FASTA
sequences, ids = dm.load_sequences('proteins.fasta')

# Load from DataFrame
import pandas as pd
df = pd.read_csv('data.csv')
sequences, ids = dm.load_from_dataframe(
    df, 
    sequence_column='seq',
    id_column='id'
)

# Validate
valid_seqs, valid_idx = dm.validate_sequences(sequences)

# Get statistics
stats = dm.get_stats(sequences)
print(f"Mean length: {stats['mean_length']}")
```

### 2. ModelManager (`core/model_manager.py`)

Manages loading and caching of ESM and FM4M models.

**Features:**
- Lazy model loading
- Model caching (avoid reloading)
- Device management (CPU/GPU)
- Memory optimization

**Example:**
```python
from build.embeddings.core import ModelManager

mm = ModelManager(use_gpu=True)

# Load ESM model
model, alphabet = mm.load_esm_model('esm2_t33_650M_UR50D')

# Load FM4M model
fm4m_model = mm.load_fm4m_model()

# Get device info
info = mm.get_device_info()
print(f"Using device: {info['device']}")

# Clear cache
mm.clear_cache('esm')
```

### 3. EmbeddingGenerator (`core/generator.py`)

Generates embeddings using loaded models.

**Features:**
- Batch processing
- Progress tracking (tqdm)
- Memory-efficient generation
- Both ESM and FM4M support

**Example:**
```python
from build.embeddings.core import ModelManager, EmbeddingGenerator

mm = ModelManager(use_gpu=True)
gen = EmbeddingGenerator(mm, batch_size=32)

# Generate protein embeddings
sequences = ['MKTAYIAK', 'ARNDCEQG']
embeddings = gen.generate_esm_embeddings(
    sequences,
    model_name='esm2_t33_650M_UR50D',
    repr_layer=33,
    show_progress=True
)

# Generate ligand embeddings
smiles = ['CCO', 'c1ccccc1']
embeddings = gen.generate_fm4m_embeddings(smiles)
```

### 4. ModelRegistry (`models/model_registry.py`)

Centralized catalog of all supported models.

**Features:**
- List all ESM models (ESM-2, ESM-1b)
- List FM4M models
- Get model info (embedding dim, layers, GPU requirements)
- Validation

**Example:**
```python
from build.embeddings.models import ModelRegistry

# List all ESM models
ModelRegistry.print_models('esm')

# Get model info
info = ModelRegistry.get_model_info('esm2_t33_650M_UR50D')
print(f"Embedding dim: {info.embedding_dim}")
print(f"Default layer: {info.default_layer}")

# Check if model exists
is_valid = ModelRegistry.is_valid_model('esm2_t33_650M_UR50D', 'esm')

# Get default model
default = ModelRegistry.get_default_model('esm')
```

### 5. CacheManager (`utils/cache.py`)

Handles caching of generated embeddings.

**Features:**
- Memory cache (fast)
- Disk cache (persistent)
- Automatic cache key generation
- Cache validation

**Example:**
```python
from build.embeddings.utils import CacheManager
from pathlib import Path

cache = CacheManager(
    cache_dir=Path('embeddings_cache'),
    use_memory_cache=True
)

# Save embeddings
cache.save_embeddings(
    embeddings=embeddings,
    sequences=sequences,
    model_name='esm2_t33_650M_UR50D',
    model_type='esm'
)

# Load embeddings
cached = cache.load_embeddings(
    sequences=sequences,
    model_name='esm2_t33_650M_UR50D',
    model_type='esm'
)

# Get cache info
info = cache.get_cache_info()
cache.print_cache_info()

# Clear cache
cache.clear_all()
```

### 6. Validators (`utils/validators.py`)

Input validation functions.

**Features:**
- Protein sequence validation
- SMILES validation
- Batch validation
- Duplicate detection
- Statistics

**Example:**
```python
from build.embeddings.utils import (
    validate_protein_sequence,
    validate_smiles,
    validate_protein_batch
)

# Validate single sequence
is_valid, error = validate_protein_sequence('ARNDCEQG')

# Validate batch
sequences = ['ARNDCEQG', 'INVALID123', 'MKTAYIAK']
valid_seqs, valid_idx, errors = validate_protein_batch(sequences)

# Validate SMILES
is_valid, error = validate_smiles('CCO')

# Check duplicates
from build.embeddings.utils import check_duplicates
unique, dup_idx = check_duplicates(sequences)
```

## 🎨 Usage Patterns

### Pattern 1: Quick Generation (Convenience Function)

```python
from build.embeddings.modular_pipeline import generate_embeddings

# Protein
embeddings = generate_embeddings(
    ['ARNDCEQG', 'MKTAYIAK'],
    embedding_type='protein',
    use_gpu=True
)

# Ligand
embeddings = generate_embeddings(
    ['CCO', 'c1ccccc1'],
    embedding_type='ligand'
)
```

### Pattern 2: Full Pipeline with Caching

```python
from build.embeddings.modular_pipeline import EmbeddingPipeline
from pathlib import Path

pipeline = EmbeddingPipeline(
    use_gpu=True,
    cache_dir=Path('cache'),
    batch_size=64
)

# First run: generates embeddings
embeddings1 = pipeline.generate_protein_embeddings(
    source='sequences.fasta',
    use_cache=True
)

# Second run: loads from cache (instant!)
embeddings2 = pipeline.generate_protein_embeddings(
    source='sequences.fasta',
    use_cache=True
)
```

### Pattern 3: Custom Component Assembly

```python
from build.embeddings.core import DataManager, ModelManager, EmbeddingGenerator

# Create custom pipeline
dm = DataManager()
mm = ModelManager(use_gpu=True)
gen = EmbeddingGenerator(mm, batch_size=32)

# Load data
sequences, ids = dm.load_sequences('sequences.fasta')

# Validate
valid_seqs, valid_idx = dm.validate_sequences(sequences)

# Generate
embeddings = gen.generate_esm_embeddings(valid_seqs)

# Save
import numpy as np
np.save('embeddings.npy', embeddings)
```

## 📊 Model Catalog

### ESM Models (Protein Language Models)

| Model | Parameters | Embedding Dim | Layers | GPU Required |
|-------|-----------|---------------|--------|--------------|
| `esm2_t48_15B_UR50D` | 15B | 5120 | 48 | ✅ Yes |
| `esm2_t36_3B_UR50D` | 3B | 2560 | 36 | ✅ Yes |
| `esm2_t33_650M_UR50D` | 650M | 1280 | 33 | ❌ No (default) |
| `esm2_t30_150M_UR50D` | 150M | 640 | 30 | ❌ No |
| `esm2_t12_35M_UR50D` | 35M | 480 | 12 | ❌ No |
| `esm2_t6_8M_UR50D` | 8M | 320 | 6 | ❌ No |
| `esm1b_t33_650M_UR50S` | 650M | 1280 | 33 | ❌ No (legacy) |

### 🆕 Boltz-2 Models (Multi-Pooling Strategy) **← RECOMMENDED**

| Model | Strategy | Embedding Dim | Speed | Performance |
|-------|----------|---------------|-------|-------------|
| `boltz2_multi` | CLS + mean + max | 384 | ⚡ Fast (18 min/299) | 🏆 Best |
| `boltz2_cls` | CLS token only | 128 | ⚡⚡ Very fast | ✅ Good |
| `boltz2_mean` | Mean pooling | 128 | ⚡⚡ Very fast | ✅ Good |
| `boltz2_max` | Max pooling | 128 | ⚡⚡ Very fast | ✅ Good |

**Benchmarks**: Boltz-2 multi-pooling achieves comparable performance to ESM-2 3B (2560-dim) with 85% smaller embeddings and 10x faster inference.

See **[BOLTZ_STRATEGY_GUIDE.md](../../docs/04-modules/BOLTZ_STRATEGY_GUIDE.md)** for complete integration guide.

### FM4M Models (Small Molecule Embeddings)

| Model | Embedding Dim | Description |
|-------|---------------|-------------|
| `smi_ted_light` | 768 | SMI-TED Light (default) |

## 🔄 Backward Compatibility

The modular architecture is **100% backward compatible**. All original classes work unchanged:

```python
# Original API still works!
from build.embeddings import ProteinEmbedding, LigandEmbedding

# Original protein embeddings
protein_emb = ProteinEmbedding(config={'use_gpu': True})
embeddings = protein_emb.generate_embeddings(['ARNDCEQG'])

# Original ligand embeddings
ligand_emb = LigandEmbedding(config={})
embeddings = ligand_emb.generate_embeddings(['CCO'])
```

## ✅ Testing

```python
# Test protein embeddings
from build.embeddings.modular_pipeline import EmbeddingPipeline

pipeline = EmbeddingPipeline()

# Small test
sequences = ['ARNDCEQGHILKMFPSTWYV', 'MKTAYIAKQRQISFVK']
embeddings = pipeline.generate_protein_embeddings(sequences)

assert embeddings.shape[0] == 2
assert embeddings.shape[1] == 1280  # esm2_t33_650M_UR50D dimension
print("✅ Test passed!")
```

## 🎯 Benefits of Modular Architecture

### 1. **Maintainability**
- Clear separation of concerns
- Easy to locate and fix bugs
- Self-documenting code structure

### 2. **Testability**
- Each component can be tested independently
- Mock dependencies for unit tests
- Integration tests are straightforward

### 3. **Extensibility**
- Easy to add new models
- Easy to add new data formats
- Easy to add new caching strategies

### 4. **Reusability**
- Components can be used independently
- Mix and match as needed
- Share components across projects

### 5. **Performance**
- Efficient caching (memory + disk)
- Batch processing
- Lazy model loading
- Device management

## � Automatic Quality Checks (New!)

### 1. NaN/Inf Detection in Protein Embeddings

The pipeline now **automatically detects** invalid values in protein embeddings:

```python
pipeline = EmbeddingPipeline(use_gpu=True)
embeddings = pipeline.generate_protein_embeddings(sequences)

# If NaN/Inf detected, you'll see:
# ⚠️  Warning: Embeddings contain 5 NaN and 0 Inf values
#    This may indicate issues with input sequences or model
```

**What it means:**
- NaN/Inf values indicate problematic sequences or model issues
- The embeddings are still returned, but flagged for review
- Consider removing problematic sequences and regenerating

### 2. FM4M Limitation Detection

The pipeline warns about **known FM4M limitations** with complex SMILES:

```python
embeddings = pipeline.generate_ligand_embeddings(complex_smiles)

# If complex stereochemistry detected:
# ⚠️  Warning: 3 embeddings contain NaN values
#    This is a known FM4M limitation with complex SMILES
#    Consider using simpler SMILES or filtering these molecules
```

**Known issue:**
- FM4M can generate NaN for SMILES with complex stereochemistry
- Example: Ibuprofen derivatives with multiple chiral centers
- Solution: Filter or simplify problematic molecules

### 3. Enhanced Cache Statistics

New method to monitor cache performance:

```python
pipeline = EmbeddingPipeline(cache_dir=Path('cache'))

# Generate some embeddings with cache enabled
embeddings = pipeline.generate_protein_embeddings(
    sequences, 
    use_cache=True
)

# Get detailed cache statistics
stats = pipeline.get_cache_stats()
print(stats)
# Output:
# {
#   'cache_dir': PosixPath('cache'),
#   'total_entries': 150,
#   'memory_cache_size': 45,
#   'disk_cache_size': 105,
#   'cache_hit_rate': '87.3%'  # If tracking enabled
# }
```

**Benefits:**
- Monitor cache efficiency
- Track hit/miss rates
- Optimize caching strategy
- Debug cache issues

## �📈 Performance Tips

### 1. Use GPU When Available
```python
pipeline = EmbeddingPipeline(use_gpu=True)
```

### 2. Enable Caching
```python
pipeline = EmbeddingPipeline(cache_dir=Path('cache'))
embeddings = pipeline.generate_protein_embeddings(
    sequences,
    use_cache=True
)
```

### 3. Optimize Batch Size
```python
# Larger batch = faster, but more memory
pipeline = EmbeddingPipeline(batch_size=64)  # Default is 32
```

### 4. Choose Appropriate Model
```python
# Fast: esm2_t6_8M_UR50D (320 dim)
# Balanced: esm2_t33_650M_UR50D (1280 dim) ⭐ DEFAULT
# Best: esm2_t48_15B_UR50D (5120 dim, requires GPU)
```

### 5. Monitor Cache Performance (New!)
```python
# Check cache statistics periodically
stats = pipeline.get_cache_stats()
print(f"Cache hit rate: {stats.get('cache_hit_rate', 'N/A')}")

# Clear cache if hit rate is low
if stats['memory_cache_size'] > 1000:
    pipeline.clear_cache()
```

## 🔧 Troubleshooting

### Issue: Out of Memory (OOM)

**Solution 1**: Reduce batch size
```python
pipeline = EmbeddingPipeline(batch_size=16)  # or even smaller
```

**Solution 2**: Use smaller model
```python
embeddings = pipeline.generate_protein_embeddings(
    sequences,
    model_name='esm2_t12_35M_UR50D'  # Much smaller
)
```

### Issue: Slow Generation

**Solution 1**: Enable GPU
```python
pipeline = EmbeddingPipeline(use_gpu=True)
```

**Solution 2**: Enable caching
```python
pipeline = EmbeddingPipeline(cache_dir=Path('cache'))
```

**Solution 3**: Increase batch size
```python
pipeline = EmbeddingPipeline(batch_size=64)
```

### Issue: NaN Values in Embeddings (New!)

**For Proteins (ESM):**
```python
# Check which sequences caused NaN
embeddings = pipeline.generate_protein_embeddings(sequences)

# If warning appears, inspect sequences:
for i, seq in enumerate(sequences):
    if np.isnan(embeddings[i]).any():
        print(f"Problematic sequence: {seq}")
        
# Solution: Remove or fix problematic sequences
valid_idx = ~np.isnan(embeddings).any(axis=1)
clean_embeddings = embeddings[valid_idx]
clean_sequences = [seq for i, seq in enumerate(sequences) if valid_idx[i]]
```

**For Ligands (FM4M):**
```python
# Known limitation with complex stereochemistry
embeddings = pipeline.generate_ligand_embeddings(smiles_list)

# If warning appears, filter NaN embeddings:
valid_idx = ~np.isnan(embeddings).any(axis=1)
clean_embeddings = embeddings[valid_idx]
clean_smiles = [s for i, s in enumerate(smiles_list) if valid_idx[i]]

# Or use simpler SMILES representations
from rdkit import Chem
simplified = [Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) 
              for s in smiles_list]
```

### Issue: Low Cache Hit Rate (New!)

**Check cache performance:**
```python
stats = pipeline.get_cache_stats()
print(f"Hit rate: {stats.get('cache_hit_rate', 'N/A')}")
```

**Solutions:**
- Ensure `use_cache=True` in generation calls
- Verify sequences haven't changed (cache uses sequence as key)
- Check cache directory is writable
- Increase memory cache size if using many unique sequences

## 📚 Additional Resources

- **ESM**: https://github.com/facebookresearch/esm
- **FM4M**: https://github.com/IBM/molformer
- **Original Documentation**: See `base_embedding.py`, `protein_embedding.py`, `ligand_embedding.py`

## 🎓 Migration Guide

### From Original API to Modular API

**Before (Original):**
```python
from build.embeddings import ProteinEmbedding

emb = ProteinEmbedding(
    config={'use_gpu': True},
    model_name='esm2_t33_650M_UR50D'
)
embeddings = emb.generate_embeddings(['ARNDCEQG'])
```

**After (Modular):**
```python
from build.embeddings.modular_pipeline import EmbeddingPipeline

pipeline = EmbeddingPipeline(use_gpu=True)
embeddings = pipeline.generate_protein_embeddings(
    ['ARNDCEQG'],
    model_name='esm2_t33_650M_UR50D'
)
```

### Benefits of Migration
- Automatic validation
- Built-in caching
- Progress bars
- Better error handling
- More flexible input formats

---

**Created**: 2024  
**Author**: DockTKinase Team  
**Version**: 1.0.0  
**License**: See LICENSE file
