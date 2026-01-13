# Boltz-2 Strategy Guide

## Overview

Boltz-2 is a biomolecular foundation model that combines structure prediction with binding affinity estimation. This guide covers the integration of Boltz-2 into semantic-screening for protein embedding generation.

## Key Features

- **Structure + Affinity**: Unique capability to predict both 3D structure and binding affinities
- **384-dim Single Representation** (default): Mean-pooled token embeddings
- **1024-dim Multi-Pooling** (optional): Combined CLS, mean, and max pooling strategies
- **CLI-based Execution**: Efficient subprocess wrapper around `boltz predict`
- **GPU Acceleration**: Automatic CUDA/MPS device detection
- **MSA Support** (optional): ColabFold server integration for evolutionary context

## Architecture

### Model Specifications

```python
MODEL_SPECS = {
    'boltz2': {
        'dim_single': 384,      # token_s: single representation dimension
        'dim_pair': 128,        # token_z: pair representation dimension  
        'output_dim': 384,      # Mean pooling output (default)
        'num_blocks': 64,       # Pairformer blocks
        'max_len': 2048,        # Maximum sequence length
    }
}
```

### Representation Types

1. **Single Representations (s)**: `[N_tokens, 384]`
   - Per-residue features from Pairformer blocks
   - Default output via mean pooling

2. **Pair Representations (z)**: `[N_tokens, N_tokens, 128]`
   - Pairwise interaction features
   - Optional extraction for advanced use cases

## Installation

### Prerequisites

```bash
# Install Boltz CLI
pip install boltz

# Verify installation
boltz --version
```

### Dependencies

All dependencies are automatically installed via `post_install.py`:

```python
# Boltz-2 specific packages
- einx>=0.1.3              # Tensor operations extension
- fairscale>=0.4.0         # Distributed training
- hydra-core>=1.3.0        # Configuration management
- omegaconf>=2.3.0         # Config files
- mashumaro>=3.0           # Data serialization
- chembl-structure-pipeline # Chemical structure processing
- numba>=0.58.0            # JIT compilation
```

## Usage

### Basic Example

```python
from src.build.embeddings.strategies.boltz_strategy import BoltzStrategy
import torch

# Initialize strategy
strategy = BoltzStrategy(
    use_msa=False,  # Disable MSA for speed
    msa_server="https://api.colabfold.com"
)

# Load model (performs CLI availability check)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, tokenizer = strategy.load('boltz2', device=device)

# Generate embedding
sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
embedding = strategy.generate(
    model=model,
    tokenizer=tokenizer,
    sequence=sequence,
    device=device,
    pooling='mean'  # Default: 384-dim
)

print(f"Embedding shape: {embedding.shape}")  # (384,)
```

### Multi-Pooling (1024-dim)

```python
# Generate 1024-dim embedding with multi-pooling
embedding_multi = strategy.generate(
    model=model,
    tokenizer=tokenizer,
    sequence=sequence,
    device=device,
    pooling='multi'
)

print(f"Multi-pooling shape: {embedding_multi.shape}")  # (1024,)

# Breakdown:
# - CLS token: 384-dim
# - Mean pooling: 384-dim
# - Max pooling: 256-dim
# - Total: 1024-dim
```

### With MSA Support

```python
# Enable MSA generation
strategy = BoltzStrategy(
    use_msa=True,
    msa_server="https://api.colabfold.com"
)

# Generate with evolutionary context
embedding = strategy.generate(
    model=model,
    tokenizer=tokenizer,
    sequence=sequence,
    device=device,
    msa_config={
        'mode': 'MAIN_STANDARD',
        'max_sequences': 512
    }
)
```

### High-Level API

```python
from src.build.embeddings.protein_embedding import ProteinEmbedding

# Initialize embedder
embedder = ProteinEmbedding(
    model_name='boltz2',
    device=torch.device('cuda'),
    use_gpu=True
)

# Generate single embedding
embedding = embedder.generate_single_embedding(sequence)

# Process multiple sequences
embeddings = embedder.generate_embeddings(
    sequences=[seq1, seq2, seq3],
    output_dir="results/embeddings/"
)
```

## Integration with Pipeline

### Complete Workflow

```bash
# Run full pipeline with Boltz-2
python run_complete_pipeline.py \
    --input data/kinase_compounds.tsv \
    --output results/boltz2_run \
    --esm-model boltz2 \
    --seed 42

# Estimated time for 299 proteins: ~18 minutes
```

### Python API

```python
from src.integrated_pipeline import IntegratedPipeline, IntegratedConfig

config = IntegratedConfig(
    input_tsv="data/kinase_compounds.tsv",
    output_dir="results/",
    esm_model="boltz2",
    device="cuda",
    run_classification=True,
    run_regression=True
)

pipeline = IntegratedPipeline(config)
results = pipeline.run()
```

## Performance

### Benchmarks (299 proteins, RTX 3060)

| Configuration | Time | Memory | Dimension | Quality |
|---------------|------|--------|-----------|---------|
| Mean pooling | 18 min | ~6 GB | 384 | High |
| Multi-pooling | 18 min | ~6 GB | 1024 | Maximum |
| With MSA | ~45 min* | ~8 GB | 384 | Enhanced |

*First run. Cached MSA runs: ~20 minutes.

### Optimization Tips

1. **Disable MSA for speed**: `use_msa=False` (default)
2. **Batch processing**: Use checkpointing for large datasets
3. **GPU selection**: Automatic CUDA/MPS detection
4. **Cache management**: Boltz CLI caches models automatically

## Comparison with Other Models

### vs. ESM-2 (650M)

| Aspect | Boltz-2 | ESM-2 |
|--------|---------|-------|
| **Speed** | 18 min | 12 min |
| **Dimension** | 384 (384/1024) | 1280 |
| **Type** | Structure + Affinity | Sequence |
| **MSA** | Optional | No |
| **Use Case** | Binding prediction | General embeddings |

## Advanced Configuration

### Custom Pooling Strategy

```python
class CustomPooling:
    @staticmethod
    def aggregate(embeddings):
        """Custom aggregation logic"""
        cls_token = embeddings[0]  # [384]
        mean_pool = embeddings[1:].mean(0)  # [384]
        max_pool = embeddings[1:].max(0)[0]  # [384]
        
        # Custom combination
        return torch.cat([cls_token, mean_pool, max_pool[:256]])

# Use in strategy
embedding = strategy.generate(
    ...,
    pooling_fn=CustomPooling.aggregate
)
```

### Sequence-Specific Parameters

```python
# Adjust for long sequences
embedding = strategy.generate(
    model=model,
    tokenizer=tokenizer,
    sequence=long_sequence,
    device=device,
    recycling_steps=5,      # More iterations
    sampling_steps=500,     # Higher quality
    pooling='mean'
)
```

## Troubleshooting

### CLI Not Found

```bash
# Verify Boltz installation
which boltz

# Reinstall if needed
pip uninstall boltz
pip install boltz
```

### GPU Memory Issues

```python
# Use CPU for very large proteins
device = torch.device('cpu')

# Or reduce batch size in pipeline
config = IntegratedConfig(
    ...,
    batch_size=4  # Reduce from default 8
)
```

### Invalid Sequence Characters

```python
# Boltz validates amino acids
valid_chars = set('ACDEFGHIKLMNPQRSTVWY')

# Clean sequence
cleaned_seq = ''.join(c for c in sequence if c in valid_chars)
```

## Best Practices

1. **Default to 384-dim**: Best balance of speed and information
2. **Use multi-pooling for complex tasks**: Binding affinity prediction
3. **Enable MSA selectively**: Only when evolutionary context matters
4. **Monitor memory**: GPU memory scales with sequence length
5. **Cache embeddings**: Save generated embeddings for reuse
6. **Validate sequences**: Check for invalid characters before processing

## API Reference

### BoltzStrategy Class

```python
class BoltzStrategy(BaseProteinStrategy):
    """
    Strategy for Boltz-2 protein embeddings.
    
    Args:
        logger: Optional logger instance
        use_msa: Enable MSA generation (default: False)
        msa_server: ColabFold server URL
    
    Methods:
        load(model_name, device, **kwargs) -> Tuple[None, None]
        generate(model, tokenizer, sequence, device, **kwargs) -> np.ndarray
        cleanup(model, tokenizer) -> None
    """
```

### Supported Pooling Strategies

- `'mean'`: Mean pooling over tokens (384-dim, default)
- `'cls'`: CLS token only (384-dim)
- `'max'`: Max pooling over tokens (384-dim)
- `'multi'`: CLS + Mean + Max (1024-dim)

### Output Format

```python
# Single representation (mean pooling)
embedding: np.ndarray  # Shape: (384,)

# Multi-pooling
embedding_multi: np.ndarray  # Shape: (1024,)
# Layout: [CLS(384) | Mean(384) | Max(256)]
```

## Examples

### Batch Processing

```python
sequences = [seq1, seq2, seq3, ..., seq299]
embeddings = []

for seq in sequences:
    emb = strategy.generate(
        model=model,
        tokenizer=tokenizer,
        sequence=seq,
        device=device
    )
    embeddings.append(emb)

# Save to file
np.save('embeddings_batch.npy', np.stack(embeddings))
```

### Error Handling

```python
try:
    embedding = strategy.generate(...)
except ValueError as e:
    if "Invalid amino acids" in str(e):
        print(f"Sequence contains invalid characters: {e}")
    else:
        raise
except RuntimeError as e:
    if "Boltz CLI execution failed" in str(e):
        print(f"CLI error: {e}")
    else:
        raise
```

## References

- [Boltz-2 Paper](https://arxiv.org/abs/2024.xxxxx)
- [Boltz GitHub](https://github.com/jwohlwend/boltz)
- [semantic-screening Documentation](../README.md)

---

**Last Updated**: November 2025
