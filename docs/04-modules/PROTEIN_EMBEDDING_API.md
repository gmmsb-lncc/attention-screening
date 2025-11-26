# Protein Embedding API Documentation

Complete API documentation for the protein embedding system, designed to facilitate integration with ESM-2, ESM-C, Boltz-2, and other future protein language models.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Interfaces](#core-interfaces)
3. [API Reference](#api-reference)
4. [Integration Guide](#integration-guide)
5. [Examples](#examples)
6. [Migration Guide](#migration-guide)

---

## Architecture Overview

The protein embedding system follows **SOLID principles** using the **Strategy Pattern** + **Factory Pattern**:

```
┌─────────────────────────────────────────────────────────────┐
│                   ProteinEmbedding                          │
│                   (Orchestrator)                            │
│  • Manages lifecycle                                        │
│  • Delegates to strategies                                  │
│  • Public API for users                                     │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ uses
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              ProteinModelFactory                            │
│              (Factory Pattern)                              │
│  • Detects model type from name                            │
│  • Creates appropriate strategy                             │
│  • Validates supported models                               │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ creates
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           BaseProteinStrategy (ABC)                         │
│           (Interface Contract)                              │
│  • load()           - Load model                           │
│  • generate()       - Generate embeddings                   │
│  • get_max_length() - Get sequence limit                    │
│  • get_embedding_dim() - Get vector dimension               │
│  • cleanup()        - Free resources                        │
└─────────────────────────────────────────────────────────────┘
                          △
                          │ implements
          ┌───────────────┼───────────────┐
          │               │               │
┌─────────────────┐ ┌─────────────┐ ┌─────────────┐
│  ESM2Strategy   │ │ESMCStrategy │ │BoltzStrategy│
│  (Implemented)  │ │(Implemented)│ │(Implemented)│
└─────────────────┘ └─────────────┘ └─────────────┘
```

### Key Design Principles

1. **Single Responsibility**: Each strategy handles only one model family
2. **Open/Closed**: Add new models without modifying existing code
3. **Liskov Substitution**: All strategies are interchangeable
4. **Interface Segregation**: Minimal 5-method interface
5. **Dependency Inversion**: Orchestrator depends on abstractions

---

## Core Interfaces

### BaseProteinStrategy (Abstract Base Class)

The core interface that ALL protein model strategies must implement.

#### Method Contracts

##### 1. load()

```python
@abstractmethod
def load(
    self, 
    model_name: str, 
    device: torch.device,
    offload_folder: Optional[str] = None,
    **kwargs
) -> Tuple[Any, Any]:
    """
    Load model and auxiliary components (alphabet, tokenizer, etc.)
    
    Args:
        model_name: Model identifier (e.g., "esm2_t48_15B_UR50D")
        device: PyTorch device (cuda/cpu/mps)
        offload_folder: Directory for CPU offloading (optional)
        **kwargs: Model-specific parameters
        
    Returns:
        Tuple containing (model, auxiliary_objects)
        - model: Loaded PyTorch model
        - auxiliary_objects: Tokenizer, alphabet, or other necessary objects
        
    Raises:
        ValueError: If model is not supported
        ModelLoadError: If loading fails
        
    Implementation Requirements:
        - Must validate model_name against supported models
        - Must handle device placement (cuda/cpu/mps)
        - Should implement CPU offloading for large models
        - Must set model to eval() mode
        - Should cache models when possible
        - Must provide clear error messages
    """
    pass
```

**Critical Implementation Notes:**
- Always validate `model_name` first
- Handle all device types: CUDA, MPS (Apple Silicon), CPU
- Implement CPU offloading for models > 3B parameters
- Set `model.eval()` before returning
- Cache models in `models_cache/` directory
- Provide actionable error messages with solutions

##### 2. generate()

```python
@abstractmethod
def generate(
    self,
    model: Any,
    auxiliary_objects: Any,
    sequence: str,
    device: torch.device,
    **kwargs
) -> np.ndarray:
    """
    Generate embedding for a protein sequence.
    
    Args:
        model: Loaded model from load()
        auxiliary_objects: Auxiliary objects from load()
        sequence: Amino acid sequence
        device: PyTorch device
        **kwargs: Generation parameters (layers, pooling, etc.)
        
    Returns:
        Embedding numpy array (shape: [embedding_dim])
        
    Raises:
        EmbeddingError: If generation fails
        
    Implementation Requirements:
        - Must validate sequence (non-empty, valid amino acids)
        - Must truncate sequences exceeding max_length
        - Should use torch.no_grad() context
        - Must perform pooling (mean/max/cls) to get fixed-size vector
        - MUST clean up tensors after generation (gc.collect(), empty_cache())
        - Must return numpy array (not torch tensor)
        - Should handle special tokens (BOS, EOS, PAD)
    """
    pass
```

**Critical Implementation Notes:**
- Validate sequence: `''.join(c for c in seq.upper() if c in 'ACDEFGHIKLMNPQRSTVWY')`
- Truncate to max_length with warning
- Always use `torch.no_grad()` for inference
- Remove special tokens before pooling (BOS/EOS)
- Use mean pooling over sequence (not CLS token for ESM-2)
- **CRITICAL**: Clean up memory after each generation:
  ```python
  del batch_tokens, results, embeddings
  gc.collect()
  if device.type == 'cuda':
      torch.cuda.empty_cache()
      torch.cuda.synchronize()
  ```
- Return `.cpu().numpy()` array

##### 3. get_max_length()

```python
@abstractmethod
def get_max_length(self, model_name: str) -> int:
    """
    Return maximum sequence length for the model.
    
    Args:
        model_name: Model identifier
        
    Returns:
        Maximum length in tokens/amino acids
        
    Implementation Requirements:
        - Must return actual model limit (e.g., 1024 for ESM-2, 4096 for ESM-3)
        - Should be consistent with model architecture
        - Used for automatic truncation
    """
    pass
```

**Typical Values:**
- ESM-2: 1024, 4096, or 5120 depending on model
- ESM-C: 4096 or 8192
- Boltz-2: No limit (CLI-based)

##### 4. get_embedding_dim()

```python
@abstractmethod
def get_embedding_dim(self, model_name: str) -> int:
    """
    Return embedding dimension for the model.
    
    Args:
        model_name: Model identifier
        
    Returns:
        Embedding vector dimension
        
    Implementation Requirements:
        - Must return correct dimension for output vector
        - Should match model's hidden dimension
        - Used for validation and array initialization
    """
    pass
```

**Typical Values:**
- ESM-2 8M: 320
- ESM-2 35M: 480
- ESM-2 650M: 1280
- ESM-2 3B: 2560
- ESM-2 15B: 5120
- ESM-C: 960, 1152, or 3072
- Boltz-2: 384

##### 5. cleanup()

```python
@abstractmethod
def cleanup(self, model: Any, auxiliary_objects: Any) -> None:
    """
    Free resources (GPU memory, tensors, etc.)
    
    Args:
        model: Model to clean up
        auxiliary_objects: Auxiliary objects to clean up
        
    Implementation Requirements:
        - Must call gc.collect()
        - Must call torch.cuda.empty_cache() if CUDA available
        - Should delete large tensors explicitly
        - Optional: torch.cuda.synchronize() for complete cleanup
    """
    pass
```

**Standard Implementation:**
```python
def cleanup(self, model: Any, auxiliary_objects: Any) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

---

## API Reference

### ProteinModelFactory

Factory class for creating strategy instances.

#### Class Attributes

```python
ESM2_MODELS: Set[str]  # Supported ESM-2 models
ESM3_MODELS: Set[str]  # Supported ESM-3 models (future)
```

#### Methods

##### create_strategy()

```python
@staticmethod
def create_strategy(model_name: str) -> BaseProteinStrategy:
    """
    Create appropriate strategy based on model name.
    
    Args:
        model_name: Model identifier
        
    Returns:
        Strategy instance (ESM2Strategy, ESM3Strategy, etc.)
        
    Raises:
        ValueError: If model is not supported
        
    Example:
        >>> factory = ProteinModelFactory()
        >>> strategy = factory.create_strategy("esm2_t33_650M_UR50D")
        >>> isinstance(strategy, ESM2Strategy)
        True
    """
```

##### is_esm2_model()

```python
@staticmethod
def is_esm2_model(model_name: str) -> bool:
    """
    Check if model is ESM-2.
    
    Args:
        model_name: Model identifier
        
    Returns:
        True if ESM-2, False otherwise
        
    Example:
        >>> ProteinModelFactory.is_esm2_model("esm2_t48_15B_UR50D")
        True
    """
```

##### is_esm3_model()

```python
@staticmethod
def is_esm3_model(model_name: str) -> bool:
    """
    Check if model is ESM-3.
    
    Args:
        model_name: Model identifier
        
    Returns:
        True if ESM-3, False otherwise
    """
```

##### list_supported_models()

```python
@staticmethod
def list_supported_models() -> Dict[str, List[str]]:
    """
    List all supported models by type.
    
    Returns:
        Dictionary {type: [models]}
        
    Example:
        >>> factory = ProteinModelFactory()
        >>> models = factory.list_supported_models()
        >>> models['esm2']
        ['esm1b_t33_650M_UR50S', 'esm2_t12_35M_UR50D', ...]
    """
```

### ProteinEmbedding (Orchestrator)

High-level API for generating protein embeddings.

#### Constructor

```python
def __init__(
    self, 
    config: Optional['BuildConfig'] = None,
    model_name: str = DEFAULT_ESM_MODEL,
    use_gpu: bool = False,
    **kwargs
):
    """
    Initialize protein embedding generator.
    
    Args:
        config: Build system configuration
        model_name: ESM model name
        use_gpu: Whether to use GPU when available
        **kwargs: Additional arguments
    """
```

#### Public Methods

##### initialize()

```python
def initialize(self) -> None:
    """
    Initialize model and load weights.
    
    Must be called before generate_embedding().
    Delegates to strategy.load() internally.
    """
```

##### generate_embedding()

```python
def generate_embedding(self, sequence: str) -> np.ndarray:
    """
    Generate embedding for a single protein sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Embedding numpy array
        
    Example:
        >>> gen = ProteinEmbedding(model_name="esm2_t6_8M_UR50D")
        >>> gen.initialize()
        >>> emb = gen.generate_embedding("MKFLILLFNILCLFPVLAADNH")
        >>> emb.shape
        (320,)
    """
```

##### generate_embeddings()

```python
def generate_embeddings(
    self, 
    tsv_path: Path, 
    output_dir: Optional[Path] = None
) -> bool:
    """
    Generate embeddings from TSV file (pipeline interface).
    
    Args:
        tsv_path: TSV file with columns 'seq_id' and 'seq'
        output_dir: Output directory (uses config if None)
        
    Returns:
        True if successful
        
    Example:
        >>> gen = ProteinEmbedding(model_name="esm2_t33_650M_UR50D")
        >>> gen.initialize()
        >>> success = gen.generate_embeddings("data.tsv", "output/")
        >>> success
        True
    """
```

##### get_supported_models()

```python
def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
    """
    Get supported ESM models with specifications.
    
    Returns:
        Dictionary of model specifications
        
    Example:
        >>> gen = ProteinEmbedding()
        >>> models = gen.get_supported_models()
        >>> models['esm2_t33_650M_UR50D']
        {'dim': 1280, 'max_len': 1024}
    """
```

---

## Integration Guide

### Adding a New Model (e.g., ESM-3)

Follow these steps to integrate a new protein language model:

#### Step 1: Create Strategy Implementation

Create `/src/build/embeddings/strategies/esm3_strategy.py`:

```python
"""
ESM-3 strategy implementation.
"""

from typing import Tuple, Any, Optional
import numpy as np
import torch

from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy
from src.build.core.exceptions import ModelLoadError, EmbeddingError


class ESM3Strategy(BaseProteinStrategy):
    """
    Strategy for ESM-3 models (EvolutionaryScale).
    
    Key Differences from ESM-2:
    - Uses structure-aware tokens
    - Supports longer sequences (4096+)
    - Different tokenization scheme
    - Optional structure prediction outputs
    """
    
    def __init__(self):
        self.logger = None
        self._cache_dir = None
    
    def load(
        self, 
        model_name: str, 
        device: torch.device,
        offload_folder: Optional[str] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load ESM-3 model.
        
        Implementation notes:
        - Use esm3.pretrained.load_model() or similar
        - Handle structure-aware tokenizer
        - Configure for sequence-only mode if no structure needed
        """
        self.logger = kwargs.get('logger')
        
        # Validate model
        supported_models = {
            'esm3_sm_open_v1',
            'esm3_medium',
            'esm3_large',
        }
        
        if model_name not in supported_models:
            raise ValueError(f"ESM-3 model '{model_name}' not supported")
        
        try:
            # Import ESM-3 (adjust based on actual API)
            from esm3 import pretrained
            
            # Load model
            model, tokenizer = pretrained.load_model(model_name)
            model = model.to(device).eval()
            
            if self.logger:
                self.logger.info(f"✅ ESM-3 model loaded: {model_name}")
            
            return model, tokenizer
            
        except ImportError as e:
            raise ModelLoadError(f"ESM-3 not available: {e}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load ESM-3: {e}")
    
    def generate(
        self,
        model: Any,
        auxiliary_objects: Any,
        sequence: str,
        device: torch.device,
        **kwargs
    ) -> np.ndarray:
        """
        Generate ESM-3 embedding.
        
        Implementation notes:
        - Use structure-aware tokenizer if available
        - Extract sequence representation (not structure tokens)
        - Apply appropriate pooling strategy
        """
        tokenizer = auxiliary_objects
        
        # Validate and clean sequence
        clean_sequence = ''.join(
            c for c in sequence.upper() 
            if c in 'ACDEFGHIKLMNPQRSTVWY'
        )
        
        if not clean_sequence:
            raise EmbeddingError("Empty sequence")
        
        # Truncate if needed
        max_len = self.get_max_length(model_name)
        if len(clean_sequence) > max_len:
            clean_sequence = clean_sequence[:max_len]
        
        try:
            # Tokenize (adjust based on ESM-3 API)
            tokens = tokenizer.encode(clean_sequence)
            tokens = tokens.to(device)
            
            # Generate embedding
            with torch.no_grad():
                outputs = model(tokens)
                embeddings = outputs.sequence_embeddings  # Adjust key
                
                # Mean pooling
                embedding = embeddings.mean(dim=1).squeeze()
                result = embedding.cpu().numpy()
            
            # Cleanup
            del tokens, outputs, embeddings, embedding
            import gc
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            raise EmbeddingError(f"ESM-3 generation failed: {e}")
    
    def get_max_length(self, model_name: str) -> int:
        """ESM-3 max lengths."""
        max_lengths = {
            'esm3_sm_open_v1': 4096,
            'esm3_medium': 4096,
            'esm3_large': 8192,
        }
        return max_lengths.get(model_name, 4096)
    
    def get_embedding_dim(self, model_name: str) -> int:
        """ESM-3 embedding dimensions."""
        dimensions = {
            'esm3_sm_open_v1': 1536,
            'esm3_medium': 2560,
            'esm3_large': 5120,
        }
        return dimensions.get(model_name, 2560)
    
    def cleanup(self, model: Any, auxiliary_objects: Any) -> None:
        """Standard cleanup."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

#### Step 2: Register in Factory

Update `/src/build/embeddings/factories/protein_model_factory.py`:

```python
class ProteinModelFactory:
    # ... existing code ...
    
    # Add ESM-3 models
    ESM3_MODELS = {
        'esm3_sm_open_v1',
        'esm3_medium',
        'esm3_large',
    }
    
    @staticmethod
    def create_strategy(model_name: str) -> BaseProteinStrategy:
        # ... existing ESM-2 check ...
        
        # Add ESM-3 detection
        if model_name in ProteinModelFactory.ESM3_MODELS:
            from src.build.embeddings.strategies.esm3_strategy import ESM3Strategy
            return ESM3Strategy()
        
        # ... rest of code ...
```

#### Step 3: Update Constants

Update `/src/build/core/constants.py`:

```python
# Add ESM-3 models
ESM_MODELS.update({
    'esm3_sm_open_v1': {'dim': 1536, 'max_len': 4096},
    'esm3_medium': {'dim': 2560, 'max_len': 4096},
    'esm3_large': {'dim': 5120, 'max_len': 8192},
})
```

#### Step 4: Add Tests

Create `/tests/test_esm3_strategy.py`:

```python
"""Tests for ESM-3 strategy."""

import pytest
import numpy as np
from src.build.embeddings.strategies.esm3_strategy import ESM3Strategy
from src.build.embeddings.factories.protein_model_factory import ProteinModelFactory


def test_esm3_strategy_creation():
    """Test ESM-3 strategy creation via factory."""
    factory = ProteinModelFactory()
    strategy = factory.create_strategy("esm3_sm_open_v1")
    assert isinstance(strategy, ESM3Strategy)


def test_esm3_max_length():
    """Test ESM-3 max length."""
    strategy = ESM3Strategy()
    assert strategy.get_max_length("esm3_sm_open_v1") == 4096


def test_esm3_embedding_dim():
    """Test ESM-3 embedding dimension."""
    strategy = ESM3Strategy()
    assert strategy.get_embedding_dim("esm3_medium") == 2560


@pytest.mark.slow
def test_esm3_end_to_end():
    """Test ESM-3 end-to-end generation."""
    from src.build.embeddings.protein_embedding import ProteinEmbedding
    
    gen = ProteinEmbedding(model_name="esm3_sm_open_v1")
    gen.initialize()
    
    sequence = "MKFLILLFNILCLFPVLAADNH"
    embedding = gen.generate_embedding(sequence)
    
    assert embedding.shape == (1536,)
    assert not np.isnan(embedding).any()
    assert not np.isinf(embedding).any()
```

#### Step 5: Update Documentation

Update this file and README with new model support.

---

## Examples

### Example 1: Basic Usage

```python
from src.build.embeddings.protein_embedding import ProteinEmbedding

# Initialize generator
gen = ProteinEmbedding(
    model_name="esm2_t33_650M_UR50D",
    use_gpu=True
)

# Load model
gen.initialize()

# Generate embedding
sequence = "MKFLILLFNILCLFPVLAADNH"
embedding = gen.generate_embedding(sequence)

print(f"Shape: {embedding.shape}")  # (1280,)
print(f"Range: [{embedding.min():.3f}, {embedding.max():.3f}]")
```

### Example 2: Batch Processing

```python
from pathlib import Path
from src.build.embeddings.protein_embedding import ProteinEmbedding

# Initialize
gen = ProteinEmbedding(model_name="esm2_t6_8M_UR50D")
gen.initialize()

# Process TSV file
tsv_path = Path("data/proteins.tsv")
output_dir = Path("output/embeddings/")

success = gen.generate_embeddings(tsv_path, output_dir)

# Check results
info = gen.get_embeddings_info()
print(f"Generated {info['count']} embeddings")
print(f"Model: {info['model']}")
print(f"Dimension: {info['dimension']}")
```

### Example 3: Multiple Models

```python
from src.build.embeddings.protein_embedding import ProteinEmbedding

models = [
    "esm2_t6_8M_UR50D",      # Fast, 320-dim
    "esm2_t33_650M_UR50D",   # Balanced, 1280-dim
    "esm2_t36_3B_UR50D",     # High quality, 2560-dim
]

sequence = "MKFLILLFNILCLFPVLAADNH"

for model_name in models:
    gen = ProteinEmbedding(model_name=model_name, use_gpu=True)
    gen.initialize()
    
    embedding = gen.generate_embedding(sequence)
    print(f"{model_name}: {embedding.shape}")
```

### Example 4: Custom Strategy

```python
from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy
from src.build.embeddings.factories.protein_model_factory import ProteinModelFactory

# Create custom strategy
class MyCustomStrategy(BaseProteinStrategy):
    def load(self, model_name, device, **kwargs):
        # Your implementation
        pass
    
    def generate(self, model, aux, sequence, device, **kwargs):
        # Your implementation
        pass
    
    def get_max_length(self, model_name):
        return 2048
    
    def get_embedding_dim(self, model_name):
        return 768
    
    def cleanup(self, model, aux):
        import gc
        gc.collect()

# Register in factory (modify protein_model_factory.py)
# Then use via ProteinEmbedding
```

### Example 5: Error Handling

```python
from src.build.embeddings.protein_embedding import ProteinEmbedding
from src.build.core.exceptions import ModelLoadError, EmbeddingError

try:
    gen = ProteinEmbedding(model_name="invalid_model")
    gen.initialize()
except ModelLoadError as e:
    print(f"Model loading failed: {e}")

try:
    gen = ProteinEmbedding(model_name="esm2_t6_8M_UR50D")
    gen.initialize()
    
    # Invalid sequence
    embedding = gen.generate_embedding("")
except EmbeddingError as e:
    print(f"Embedding generation failed: {e}")
```

---

## Migration Guide

### For External Users

If you were using the old API directly:

**Old Code (Before Refactoring):**
```python
from src.build.embeddings.protein_embedding import ProteinEmbedding

gen = ProteinEmbedding(model_name="esm2_t33_650M_UR50D")
gen.initialize()
embedding = gen.generate_embedding("MKFLILLFN")
```

**New Code (After Refactoring):**
```python
# SAME CODE - Backward compatible!
from src.build.embeddings.protein_embedding import ProteinEmbedding

gen = ProteinEmbedding(model_name="esm2_t33_650M_UR50D")
gen.initialize()
embedding = gen.generate_embedding("MKFLILLFN")
```

**No changes needed!** The public API is 100% backward compatible.

### For Developers Adding New Models

**Old Way (Modifying ProteinEmbedding):**
1. Edit `protein_embedding.py` (650+ lines)
2. Add model-specific logic in multiple methods
3. Risk breaking existing models
4. Difficult to test in isolation

**New Way (Strategy Pattern):**
1. Create new strategy file (e.g., `esm3_strategy.py`)
2. Implement 5 abstract methods
3. Register in factory
4. Test independently
5. Zero changes to existing code

**Benefits:**
- ✅ Isolated code (easier to maintain)
- ✅ No risk to existing models
- ✅ Easy to test
- ✅ Follows SOLID principles
- ✅ Clear structure

---

## Best Practices

### Memory Management

```python
# ✅ DO: Clean up after each generation
def generate(self, model, aux, sequence, device, **kwargs):
    # ... generation code ...
    
    # CRITICAL: Clean up tensors
    del batch_tokens, results, embeddings
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Optional but recommended
    
    return result

# ❌ DON'T: Let tensors accumulate
def generate(self, model, aux, sequence, device, **kwargs):
    # ... generation code ...
    return result  # Memory leak!
```

### Error Handling

```python
# ✅ DO: Provide actionable error messages
if model_name not in SUPPORTED_MODELS:
    raise ValueError(
        f"Model '{model_name}' not supported.\n\n"
        f"Supported models:\n" + 
        "\n".join(f"  • {m}" for m in SUPPORTED_MODELS) +
        "\n\nTo add support, see docs/04-modules/PROTEIN_EMBEDDING_API.md"
    )

# ❌ DON'T: Generic errors
if model_name not in SUPPORTED_MODELS:
    raise ValueError("Invalid model")
```

### Device Handling

```python
# ✅ DO: Support all device types
def _setup_device(self, use_gpu: bool) -> torch.device:
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")

# ❌ DON'T: Assume CUDA only
def _setup_device(self, use_gpu: bool) -> torch.device:
    return torch.device("cuda" if use_gpu else "cpu")
```

### Sequence Validation

```python
# ✅ DO: Clean and validate sequences
def _clean_sequence(self, sequence: str) -> str:
    clean = ''.join(
        c for c in sequence.upper() 
        if c in 'ACDEFGHIKLMNPQRSTVWY'
    )
    
    if not clean:
        raise EmbeddingError("No valid amino acids in sequence")
    
    return clean

# ❌ DON'T: Assume input is clean
def _clean_sequence(self, sequence: str) -> str:
    return sequence.upper()
```

---

## Testing Checklist

When adding a new model, ensure:

- [ ] Strategy implements all 5 abstract methods
- [ ] Factory creates strategy correctly
- [ ] Constants updated with model specs
- [ ] Unit tests for strategy in isolation
- [ ] Integration tests with ProteinEmbedding
- [ ] End-to-end test with real sequence
- [ ] Memory cleanup verified (no leaks)
- [ ] Error handling tested (invalid inputs)
- [ ] Documentation updated
- [ ] Examples added

---

## Troubleshooting

### Common Issues

**Issue**: `TypeError: Can't instantiate abstract class`

**Cause**: Strategy doesn't implement all abstract methods

**Solution**: Implement all 5 methods: load, generate, get_max_length, get_embedding_dim, cleanup

---

**Issue**: `CUDA out of memory`

**Cause**: Model too large for GPU

**Solutions**:
1. Use smaller model
2. Implement CPU offloading (see ESM2Strategy)
3. Use CPU device

---

**Issue**: `ValueError: Model 'xyz' not supported`

**Cause**: Model not registered in factory

**Solution**: Add model to factory's model set and create_strategy method

---

**Issue**: Memory leak (GPU memory grows)

**Cause**: Tensors not cleaned up after generation

**Solution**: Call `gc.collect()` and `torch.cuda.empty_cache()` in cleanup

---

## Version History

- **v2.0** (2025-11-18): Refactored to Strategy Pattern + Factory
- **v1.0** (2024): Initial monolithic implementation

---

## Support

For questions or issues:
1. Check this documentation
2. Review test files in `/tests/test_solid_refactoring.py`
3. Check implementation examples in `/src/build/embeddings/strategies/`
4. Open GitHub issue

---

**Last Updated**: 2025-11-18  
**Maintainer**: Docktkinase Team  
**License**: See LICENSE file
