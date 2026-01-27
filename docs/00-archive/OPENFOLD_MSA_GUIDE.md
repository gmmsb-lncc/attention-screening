# OpenFold3 MSA Configuration Guide

## 📋 Overview

OpenFold3 integrates with ColabFold MSA server to enhance protein embedding quality through Multiple Sequence Alignments. This guide covers optimal MSA configuration for DockTKinase's 700+ protein sequences.

## 🎯 Quick Start (Recommended)

```python
from src.build.embeddings.strategies.openfold_strategy import OpenFoldStrategy
from src.build.embeddings.config.msa_config import MsaConfig

# Production mode (recommended for 700+ sequences)
strategy = OpenFoldStrategy(
    msa_config=MsaConfig.for_production()
)

model, _ = strategy.load('openfold3', device)
embedding = strategy.generate(model, None, sequence, device)
```

**Time estimate**: 3-5 minutes for 700 unique sequences

---

## 🚀 MSA Modes

### 1. **Production Mode** (⭐ RECOMMENDED)

```python
config = MsaConfig.for_production()
```

**Best for**: DockTKinase with 700+ sequences

- ✅ Main MSA with environmental databases
- ✅ Diversity filter enabled
- ✅ NPZ format (faster loading)
- ⏱️ **Time**: 3-5 minutes for 700 sequences
- 📊 **Quality**: High

---

### 2. **Development Mode** (Fast)

```python
config = MsaConfig.for_development()
```

**Best for**: Quick testing and prototyping

- ⚡ Fast mode (UniRef90 only)
- ✅ Diversity filter enabled
- ⏱️ **Time**: 1-2 minutes for 700 sequences
- 📊 **Quality**: Medium

---

### 3. **Research Mode** (Maximum Quality)

```python
config = MsaConfig.for_research()
```

**Best for**: Detailed evolutionary analysis

- 🔬 No diversity filter (larger MSAs)
- ✅ All environmental databases
- 📄 A3M format (human-readable)
- ⏱️ **Time**: 5-10 minutes for 700 sequences
- 📊 **Quality**: Very High

---

### 4. **No MSA Mode** (Fastest)

```python
config = MsaConfig.no_msa()
```

**Best for**: Instant testing, sequence-only embeddings

- ⚡ No MSA computation
- ⏱️ **Time**: Instant
- 📊 **Quality**: Lower (sequence-only)

---

## 🔧 Custom Configuration

```python
config = MsaConfig(
    mode=MsaMode.MAIN_STANDARD,
    file_format=MsaFileFormat.NPZ,
    use_env=True,              # Environmental databases
    use_filter=True,           # Diversity filter
    use_templates=False,       # Not needed for embeddings
    output_directory=Path("./my_msa_cache"),
    enable_caching=True,       # Reuse MSAs
    cleanup_after_use=True,
    user_agent="myproject/1.0 contact@email.com"
)
```

---

## 📊 Mode Comparison

| Mode | Time (700 seqs) | Quality | Databases | Filter | Recommendation |
|------|-----------------|---------|-----------|--------|----------------|
| **Production** | 3-5 min | High | All | Yes | ⭐ **Default** |
| Development | 1-2 min | Medium | UniRef90 | Yes | Testing |
| Research | 5-10 min | Very High | All | No | Analysis |
| No MSA | Instant | Lower | None | N/A | Quick tests |

---

## ⚠️ Important Notes

### Main MSA vs Paired MSA

**For DockTKinase (700+ sequences):**

✅ **Use Main MSA** (default)
- 1 query for all sequences
- Fast and scalable
- Perfect for embedding extraction

❌ **Avoid Paired MSA**
- 1 query per complex
- Very slow for 700+ sequences
- Only useful for protein-protein complexes

### Deduplication

OpenFold automatically deduplicates identical sequences:

```python
# If you have 700 sequences but only 500 are unique
# OpenFold processes only 500 sequences
# Estimated time: ~2-4 minutes instead of 3-5
```

### Caching

Enable caching to reuse MSAs across runs:

```python
config = MsaConfig.for_production()
config.enable_caching = True  # Reuse MSAs
config.cleanup_after_use = False  # Keep raw files
```

---

## 🔍 Configuration Parameters

### Core Settings

```python
mode: MsaMode                    # MSA computation mode
file_format: MsaFileFormat       # NPZ (fast) or A3M (readable)
use_env: bool                    # Use environmental databases
use_filter: bool                 # Apply diversity filter
use_templates: bool              # Fetch template structures
```

### Server Settings

```python
server_url: str                  # ColabFold API endpoint
user_agent: str                  # API user identification
timeout_seconds: int             # Request timeout
max_retries: int                 # Retry attempts
```

### Storage Settings

```python
output_directory: Path           # MSA cache location
cleanup_after_use: bool          # Remove raw files
enable_caching: bool             # Reuse computed MSAs
```

---

## 💡 Best Practices

### 1. Set User Agent

**Required** for ColabFold API compliance:

```python
config.user_agent = "docktkinase/1.0 contact@institution.edu"
```

### 2. Enable Caching

For multiple runs with same sequences:

```python
config.enable_caching = True
config.output_directory = Path("./msa_cache_persistent")
```

### 3. Monitor Progress

```python
import logging
logging.basicConfig(level=logging.INFO)

# Will show progress:
# "Submitting 700 sequences to ColabFold MSA server..."
# "Processing: 150/700 completed..."
```

### 4. Batch Processing

For very large datasets:

```python
# Process in chunks if needed
config.chunk_size = 100  # Process 100 sequences at a time
```

---

## 📈 Performance Optimization

### For 700+ Sequences

**Recommended settings**:

```python
config = MsaConfig(
    mode=MsaMode.MAIN_STANDARD,
    file_format=MsaFileFormat.NPZ,  # Faster parsing
    use_env=True,
    use_filter=True,
    enable_caching=True,              # Reuse MSAs
    output_directory=Path("./msa_cache"),
)
```

**Expected performance**:
- **First run**: 3-5 minutes (computes MSAs)
- **Subsequent runs**: < 1 minute (uses cache)
- **Unique sequences**: Only unique sequences are processed
- **Memory**: ~1-2 GB for 700 sequences

---

## 🧪 Testing

### Quick Test (No MSA)

```python
# Fastest way to test the pipeline
config = MsaConfig.no_msa()
strategy = OpenFoldStrategy(msa_config=config)
# Instant embedding generation
```

### Development Test (Fast MSA)

```python
# Test with real MSAs but faster
config = MsaConfig.for_development()
strategy = OpenFoldStrategy(msa_config=config)
# ~1-2 minutes for 700 sequences
```

---

## 📝 Examples

See complete examples in:
```
examples/openfold_msa_embedding_extraction.py
```

Includes:
- Production mode usage
- Development mode usage
- Research mode usage
- Custom configurations
- Batch processing (700+ sequences)

---

## 🔗 Related Documentation

- [OpenFold Embedding Extraction](./OPENFOLD_EMBEDDING_EXTRACTION.md)
- [OpenFold Installation Guide](./OPENFOLD_INSTALLATION.md)
- [ColabFold MSA Server Documentation](https://github.com/sokrypton/ColabFold)

---

## 🎓 Summary

**For DockTKinase (700+ sequences)**:

1. ✅ Use **Production mode** (Main MSA)
2. ✅ Enable **caching** for multiple runs
3. ✅ Use **NPZ format** for faster loading
4. ✅ Set **user_agent** for API compliance
5. ❌ Avoid **Paired MSA** (too slow)

**Expected workflow**:
```
Production mode → 3-5 minutes → High-quality embeddings
```

**First run**: Computes MSAs (3-5 min)  
**Subsequent runs**: Uses cache (< 1 min)
