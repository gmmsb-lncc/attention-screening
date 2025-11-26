# Integration Examples for Protein Embedding System

This document provides complete, copy-paste ready examples for integrating new protein models (ESM-3, ProtTrans, etc.) into the docktkinase protein embedding system.

## Table of Contents

1. [ESM-3 Integration](#esm-3-integration)
2. [ProtTrans Integration](#prottrans-integration)
3. [Custom Model Integration](#custom-model-integration)
4. [Testing Your Integration](#testing-your-integration)

---

## ESM-3 Integration

### Step 1: Create ESM-3 Strategy

Create file: `/src/build/embeddings/strategies/esm3_strategy.py`

```python
"""
ESM-3 strategy implementation (EvolutionaryScale).

ESM-3 is a multimodal protein language model that supports:
- Sequence-only embeddings
- Structure-aware embeddings
- Function prediction
- Longer sequences (4096+ tokens)
"""

from typing import Tuple, Any, Optional
import numpy as np
import torch
import gc
from pathlib import Path

from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy
from src.build.core.exceptions import ModelLoadError, EmbeddingError


class ESM3Strategy(BaseProteinStrategy):
    """
    Strategy implementation for ESM-3 models.
    
    Supported models:
    - esm3_sm_open_v1 (small, open source)
    - esm3_medium (medium size)
    - esm3_large (large size)
    """
    
    # Model specifications
    MODEL_SPECS = {
        'esm3_sm_open_v1': {'dim': 1536, 'max_len': 4096},
        'esm3_medium': {'dim': 2560, 'max_len': 4096},
        'esm3_large': {'dim': 5120, 'max_len': 8192},
    }
    
    def __init__(self):
        """Initialize ESM-3 strategy."""
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
        Load ESM-3 model and tokenizer.
        
        Args:
            model_name: Model identifier ('esm3_sm_open_v1', etc.)
            device: PyTorch device
            offload_folder: CPU offloading directory (optional)
            **kwargs: Additional parameters (logger, etc.)
        
        Returns:
            Tuple (model, tokenizer)
        """
        self.logger = kwargs.get('logger')
        
        # Validate model
        if model_name not in self.MODEL_SPECS:
            raise ValueError(
                f"ESM-3 model '{model_name}' not supported.\n"
                f"Available models: {list(self.MODEL_SPECS.keys())}"
            )
        
        # Setup cache
        self._setup_cache(offload_folder)
        
        try:
            if self.logger:
                self.logger.info(f"Loading ESM-3 model: {model_name}")
            
            # Import ESM-3 (adjust based on actual API)
            # NOTE: This is example code - adjust to actual ESM-3 API
            try:
                from esm import pretrained as esm3_pretrained
            except ImportError:
                raise ModelLoadError(
                    "ESM-3 not available. Install with: pip install esm3"
                )
            
            # Load model
            model, tokenizer = esm3_pretrained.load_esm3_model(model_name)
            model = model.to(device).eval()
            
            if self.logger:
                self.logger.info(f"✅ ESM-3 model loaded successfully")
            
            return model, tokenizer
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load ESM-3 model '{model_name}': {e}")
    
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
        
        Args:
            model: Loaded ESM-3 model
            auxiliary_objects: Tokenizer
            sequence: Amino acid sequence
            device: PyTorch device
            **kwargs: Additional parameters
        
        Returns:
            Embedding numpy array
        """
        tokenizer = auxiliary_objects
        self.logger = kwargs.get('logger', self.logger)
        
        # Validate sequence
        clean_sequence = self._clean_sequence(sequence)
        
        # Truncate if needed
        max_len = self.get_max_length(model.__class__.__name__)
        if len(clean_sequence) > max_len:
            if self.logger:
                self.logger.warning(
                    f"Sequence truncated: {len(clean_sequence)} → {max_len} aa"
                )
            clean_sequence = clean_sequence[:max_len]
        
        try:
            # Tokenize
            tokens = tokenizer.encode(clean_sequence)
            tokens = tokens.to(device)
            
            # Generate embedding
            with torch.no_grad():
                outputs = model(tokens, output_hidden_states=True)
                
                # Extract sequence embeddings (adjust key based on actual API)
                hidden_states = outputs.hidden_states[-1]  # Last layer
                
                # Mean pooling (exclude special tokens if present)
                # Adjust indexing based on ESM-3 tokenizer
                sequence_embedding = hidden_states[:, 1:-1, :].mean(dim=1).squeeze()
                
                # Convert to numpy
                result = sequence_embedding.cpu().numpy()
            
            # Critical cleanup
            del tokens, outputs, hidden_states, sequence_embedding
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return result
            
        except Exception as e:
            # Cleanup on error
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            raise EmbeddingError(f"ESM-3 embedding generation failed: {e}")
    
    def get_max_length(self, model_name: str) -> int:
        """Return max sequence length for model."""
        return self.MODEL_SPECS.get(model_name, {}).get('max_len', 4096)
    
    def get_embedding_dim(self, model_name: str) -> int:
        """Return embedding dimension for model."""
        return self.MODEL_SPECS.get(model_name, {}).get('dim', 2560)
    
    def cleanup(self, model: Any, auxiliary_objects: Any) -> None:
        """Clean up resources."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Helper methods
    
    def _setup_cache(self, offload_folder: Optional[str] = None) -> None:
        """Setup cache directories."""
        self._cache_dir = Path(__file__).parent.parent.parent.parent.parent / "models_cache" / "ESM3"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        if offload_folder:
            self._offload_folder = Path(offload_folder)
        else:
            self._offload_folder = self._cache_dir / "offload"
        self._offload_folder.mkdir(parents=True, exist_ok=True)
    
    def _clean_sequence(self, sequence: str) -> str:
        """Clean and validate sequence."""
        # Remove invalid amino acids
        valid_aa = 'ACDEFGHIKLMNPQRSTVWY'
        clean = ''.join(c for c in sequence.upper() if c in valid_aa)
        
        if not clean:
            raise EmbeddingError("Sequence contains no valid amino acids")
        
        return clean
```

### Step 2: Register in Factory

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
        # Existing ESM-2 check
        if model_name in ProteinModelFactory.ESM2_MODELS:
            return ESM2Strategy()
        
        # Add ESM-3 check
        if model_name in ProteinModelFactory.ESM3_MODELS:
            from src.build.embeddings.strategies.esm3_strategy import ESM3Strategy
            return ESM3Strategy()
        
        # ... rest of error handling ...
    
    @staticmethod
    def is_esm3_model(model_name: str) -> bool:
        """Check if model is ESM-3."""
        return model_name in ProteinModelFactory.ESM3_MODELS
```

### Step 3: Update Constants

Update `/src/build/core/constants.py`:

```python
# Add ESM-3 models
ESM_MODELS.update({
    'esm3_sm_open_v1': {'dim': 1536, 'max_len': 4096},
    'esm3_medium': {'dim': 2560, 'max_len': 4096},
    'esm3_large': {'dim': 5120, 'max_len': 8192},
})
```

### Step 4: Add Tests

Create `/tests/test_esm3_strategy.py`:

```python
"""Tests for ESM-3 strategy."""

import pytest
import numpy as np
import torch

from src.build.embeddings.strategies.esm3_strategy import ESM3Strategy
from src.build.embeddings.factories.protein_model_factory import ProteinModelFactory
from src.build.embeddings.protein_embedding import ProteinEmbedding


def test_esm3_strategy_creation():
    """Test ESM-3 strategy creation via factory."""
    factory = ProteinModelFactory()
    strategy = factory.create_strategy("esm3_sm_open_v1")
    assert isinstance(strategy, ESM3Strategy)


def test_esm3_max_length():
    """Test ESM-3 max length specifications."""
    strategy = ESM3Strategy()
    assert strategy.get_max_length("esm3_sm_open_v1") == 4096
    assert strategy.get_max_length("esm3_large") == 8192


def test_esm3_embedding_dim():
    """Test ESM-3 embedding dimensions."""
    strategy = ESM3Strategy()
    assert strategy.get_embedding_dim("esm3_sm_open_v1") == 1536
    assert strategy.get_embedding_dim("esm3_medium") == 2560
    assert strategy.get_embedding_dim("esm3_large") == 5120


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_esm3_end_to_end():
    """Test ESM-3 end-to-end generation."""
    gen = ProteinEmbedding(model_name="esm3_sm_open_v1", use_gpu=True)
    gen.initialize()
    
    sequence = "MKFLILLFNILCLFPVLAADNH"
    embedding = gen.generate_embedding(sequence)
    
    assert embedding.shape == (1536,)
    assert not np.isnan(embedding).any()
    assert not np.isinf(embedding).any()
    assert -50 < embedding.min() < 50  # Reasonable range
    assert -50 < embedding.max() < 50


def test_esm3_cleanup():
    """Test ESM-3 resource cleanup."""
    strategy = ESM3Strategy()
    strategy.cleanup(None, None)  # Should not raise
```

### Step 5: Usage Example

```python
from src.build.embeddings.protein_embedding import ProteinEmbedding

# Use ESM-3
gen = ProteinEmbedding(model_name="esm3_sm_open_v1", use_gpu=True)
gen.initialize()

sequence = "MKFLILLFNILCLFPVLAADNH"
embedding = gen.generate_embedding(sequence)

print(f"ESM-3 Embedding shape: {embedding.shape}")  # (1536,)
```

---

## Testing Your Integration

### Complete Test Suite Template

Create `/tests/test_<model>_integration.py`:

```python
"""Complete integration tests for new model."""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.build.embeddings.strategies.<model>_strategy import <Model>Strategy
from src.build.embeddings.factories.protein_model_factory import ProteinModelFactory
from src.build.embeddings.protein_embedding import ProteinEmbedding


class TestStrategyCreation:
    """Test strategy creation and registration."""
    
    def test_factory_creates_strategy(self):
        """Factory creates correct strategy instance."""
        factory = ProteinModelFactory()
        strategy = factory.create_strategy("<model_name>")
        assert isinstance(strategy, <Model>Strategy)
    
    def test_factory_recognizes_model(self):
        """Factory recognizes model type."""
        assert ProteinModelFactory.is_<model>_model("<model_name>")
    
    def test_strategy_listed_in_factory(self):
        """Strategy models listed in factory."""
        models = ProteinModelFactory.list_supported_models()
        assert '<model_family>' in models
        assert '<model_name>' in models['<model_family>']


class TestStrategySpecifications:
    """Test model specifications."""
    
    def test_max_length(self):
        """Max length correct for all models."""
        strategy = <Model>Strategy()
        assert strategy.get_max_length("<model_small>") == 2048
        assert strategy.get_max_length("<model_large>") == 4096
    
    def test_embedding_dim(self):
        """Embedding dimensions correct."""
        strategy = <Model>Strategy()
        assert strategy.get_embedding_dim("<model_small>") == 384
        assert strategy.get_embedding_dim("<model_large>") == 768
    
    def test_cleanup_no_error(self):
        """Cleanup executes without error."""
        strategy = <Model>Strategy()
        strategy.cleanup(None, None)  # Should not raise


class TestEmbeddingGeneration:
    """Test embedding generation."""
    
    @pytest.mark.slow
    def test_simple_sequence(self):
        """Generate embedding for simple sequence."""
        gen = ProteinEmbedding(model_name="<model_name>")
        gen.initialize()
        
        sequence = "MKFLILLFNILCLFPVLA"
        embedding = gen.generate_embedding(sequence)
        
        assert embedding.shape == (384,)  # Adjust dimension
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()
    
    @pytest.mark.slow
    def test_long_sequence(self):
        """Handle long sequence with truncation."""
        gen = ProteinEmbedding(model_name="<model_name>")
        gen.initialize()
        
        # Create sequence longer than max
        long_sequence = "M" * 5000
        embedding = gen.generate_embedding(long_sequence)
        
        assert embedding.shape == (384,)
        assert not np.isnan(embedding).any()
    
    def test_invalid_sequence_raises(self):
        """Invalid sequence raises error."""
        gen = ProteinEmbedding(model_name="<model_name>")
        gen.initialize()
        
        with pytest.raises(Exception):  # EmbeddingError
            gen.generate_embedding("")
        
        with pytest.raises(Exception):
            gen.generate_embedding("123456789")


class TestMemoryManagement:
    """Test memory cleanup."""
    
    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_multiple_generations_no_leak(self):
        """Multiple generations don't leak memory."""
        gen = ProteinEmbedding(model_name="<model_name>", use_gpu=True)
        gen.initialize()
        
        # Get initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Generate 100 embeddings
        sequence = "MKFLILLFNILCLFPVLA"
        for _ in range(100):
            embedding = gen.generate_embedding(sequence)
        
        # Check memory didn't grow significantly
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Allow 50MB growth (should be much less)
        assert final_memory - initial_memory < 50 * 1024 * 1024


class TestBatchProcessing:
    """Test batch/pipeline processing."""
    
    @pytest.mark.slow
    def test_tsv_processing(self, tmp_path):
        """Process TSV file successfully."""
        # Create test TSV
        tsv_path = tmp_path / "test.tsv"
        with open(tsv_path, 'w') as f:
            f.write("seq_id\tseq\n")
            f.write("seq1\tMKFLILLFNIL\n")
            f.write("seq2\tADNHKPRSTVW\n")
        
        # Process
        gen = ProteinEmbedding(model_name="<model_name>")
        gen.initialize()
        
        output_dir = tmp_path / "output"
        success = gen.generate_embeddings(tsv_path, output_dir)
        
        assert success
        assert (output_dir / "seq1_embedding.npy").exists()
        assert (output_dir / "seq2_embedding.npy").exists()
        
        # Verify embeddings
        emb1 = np.load(output_dir / "seq1_embedding.npy")
        assert emb1.shape == (384,)


# Run with: pytest tests/test_<model>_integration.py -v --tb=short
```

---

## Quick Reference Checklist

When integrating a new model, complete these steps:

- [ ] 1. Create strategy file (`strategies/<model>_strategy.py`)
- [ ] 2. Implement all 5 abstract methods
- [ ] 3. Add model specs (dimensions, max lengths)
- [ ] 4. Register in ProteinModelFactory
- [ ] 5. Update constants.py
- [ ] 6. Create test file
- [ ] 7. Run all tests (pytest)
- [ ] 8. Update documentation
- [ ] 9. Add usage examples
- [ ] 10. Test memory cleanup

---

For complete API reference, see: `docs/04-modules/PROTEIN_EMBEDDING_API.md`
