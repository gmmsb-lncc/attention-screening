# ESM-3 Integration Plan for DockTKinase

## 📋 Table of Contents
1. [Executive Summary](#executive-summary)
2. [ESM-3 Architecture Analysis](#esm-3-architecture-analysis)
3. [Integration Strategy](#integration-strategy)
4. [Implementation Plan](#implementation-plan)
5. [API Comparison](#api-comparison)
6. [Migration Path](#migration-path)

---

## 🎯 Executive Summary

### Current State
- **ESM-2**: Fully integrated using Strategy Pattern
- **ESM-3**: Available in `/ESM/esm-3/esm-main/` but not integrated
- **Architecture**: Strategy + Factory Pattern ready for extension

### ESM-3 Key Differences from ESM-2

| Feature | ESM-2 | ESM-3 |
|---------|-------|-------|
| **Type** | Encoder-only (BERT-like) | Encoder-Decoder (multimodal) |
| **Modalities** | Sequence only | Sequence + Structure + Function |
| **Output** | Embeddings (representation) | Generative (prediction) |
| **Use Case** | Embedding generation | Generation + Prediction |
| **API** | `pretrained.load_model_and_alphabet()` | `ESM3.from_pretrained()` + SDK |
| **Models** | 6 models (8M-15B) | 3 models (1.4B, 7B, 98B) |
| **Embeddings** | Direct from representations | Via `output.embeddings` |

---

## 🏗️ ESM-3 Architecture Analysis

### Available Models

```python
# From ESM/esm-3/esm-main/esm/pretrained.py

LOCAL_MODEL_REGISTRY = {
    'esm3_sm_open_v1': ESM3_sm_open_v0,  # 1.4B params, 1536-dim
    # Future: 'esm3_medium': 7B params
    # Future: 'esm3_large': 98B params (via API)
}

# ESM C (Companion models for representation learning)
ESMC_MODELS = {
    'esmc-300m-2024-12': 300M params, 960-dim,
    'esmc-600m-2024-12': 600M params, 1152-dim,
    'esmc-6b-2024-12': 6B params (not local)
}
```

### Model Specifications

| Model | Parameters | Dimension | Layers | Heads | Use Case |
|-------|-----------|-----------|--------|-------|----------|
| **ESM3-small (open)** | 1.4B | 1536 | 48 | 24 | Generation + Embeddings |
| **ESM3-medium** | 7B | ~2560 | ~60 | ~40 | Better generation |
| **ESM3-large** | 98B | ~5120 | ~96 | ~80 | Production (API only) |
| **ESMC-300m** | 300M | 960 | 30 | 15 | Fast embeddings |
| **ESMC-600m** | 600M | 1152 | 36 | 18 | Balanced embeddings |

### API Structure

```python
# ESM-3 follows a different paradigm than ESM-2

# ESM-2 (Current):
model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
tokens = alphabet.get_batch_converter()([("id", sequence)])
with torch.no_grad():
    results = model(tokens, repr_layers=[model.num_layers])
    embedding = results["representations"][model.num_layers].mean(dim=0)

# ESM-3 (New):
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
protein = ESMProtein(sequence=sequence)
output = model.forward(sequence_tokens=tokens)
embedding = output.embeddings  # Direct access
```

### Key Components

```
ESM/esm-3/esm-main/
├── esm/
│   ├── models/
│   │   ├── esm3.py           # Main ESM3 model
│   │   ├── esmc.py           # ESM C (representation model)
│   │   ├── vqvae.py          # Structure encoder/decoder
│   │   └── function_decoder.py
│   ├── pretrained.py         # Model loading
│   ├── tokenization/         # Tokenizers
│   │   ├── sequence_tokenizer.py
│   │   ├── structure_tokenizer.py
│   │   └── function_tokenizer.py
│   ├── sdk/
│   │   └── api.py            # ESMProtein, ESM3InferenceClient
│   └── utils/
│       ├── encoding.py
│       └── generation.py
```

---

## 🎯 Integration Strategy

### Option 1: Full ESM-3 Integration (Recommended)

**Approach**: Create `ESM3Strategy` as a new strategy alongside `ESM2Strategy`

**Pros**:
- ✅ Maintains backward compatibility
- ✅ Allows users to choose ESM-2 or ESM-3
- ✅ Leverages existing Strategy Pattern
- ✅ Clean separation of concerns

**Cons**:
- ⚠️ Different API requires adaptation layer
- ⚠️ Larger model size (1.4B+ vs 8M-650M for ESM-2)

### Option 2: ESM-C Only Integration

**Approach**: Add ESM-C models (300M, 600M) as lighter alternatives

**Pros**:
- ✅ More similar to ESM-2 API
- ✅ Smaller models (300M, 600M)
- ✅ Optimized for embeddings (not generation)

**Cons**:
- ❌ Loses ESM-3's generative capabilities
- ⚠️ Still requires new strategy

### Option 3: Hybrid Approach (Best)

**Approach**: Integrate both ESM-3 and ESM-C models

**Implementation**:
```python
# For embeddings (equivalent to ESM-2)
ESM3Strategy → uses ESM-C models (300M, 600M)

# For generation (new capability)
ESM3GenerativeStrategy → uses ESM3 models (1.4B, 7B)
```

---

## 📐 Implementation Plan

### Phase 1: ESM-C Integration (Quick Win)

ESM-C is designed as a **drop-in replacement for ESM-2** for embedding tasks.

#### Step 1.1: Create ESMCStrategy

```python
# src/build/embeddings/strategies/esmc_strategy.py

from esm.models.esmc import ESMC
from esm.tokenization import EsmSequenceTokenizer

class ESMCStrategy(BaseProteinStrategy):
    """
    Strategy for ESM-C models (representation learning).
    
    ESM-C is optimized for embeddings, similar to ESM-2.
    Models: esmc-300m, esmc-600m
    """
    
    MODEL_SPECS = {
        'esmc-300m-2024-12': {'dim': 960, 'max_len': 2048},
        'esmc-600m-2024-12': {'dim': 1152, 'max_len': 2048},
    }
    
    def load(self, model_name, device, **kwargs):
        """Load ESM-C model."""
        model = ESMC.from_pretrained(model_name, device=device)
        tokenizer = model.tokenizer
        return model, tokenizer
    
    def generate(self, model, tokenizer, sequence, device, **kwargs):
        """Generate embedding using ESM-C."""
        # Tokenize
        tokens = model._tokenize([sequence])
        
        # Forward pass
        with torch.no_grad():
            output = model.forward(sequence_tokens=tokens)
            # Extract embeddings (CLS token or mean pooling)
            embedding = output.embeddings[:, 0, :]  # CLS token
            # OR: embedding = output.embeddings.mean(dim=1)  # Mean pooling
        
        return embedding.cpu().numpy().squeeze()
    
    def get_max_length(self, model_name):
        return self.MODEL_SPECS.get(model_name, {}).get('max_len', 2048)
    
    def get_embedding_dim(self, model_name):
        return self.MODEL_SPECS.get(model_name, {}).get('dim', 1152)
    
    def cleanup(self, model, tokenizer):
        gc.collect()
        torch.cuda.empty_cache()
```

#### Step 1.2: Register in Factory

```python
# src/build/embeddings/factories/protein_model_factory.py

class ProteinModelFactory:
    ESM2_MODELS = {...}  # Existing
    
    # NEW: ESM-C models
    ESMC_MODELS = {
        'esmc-300m-2024-12',
        'esmc-600m-2024-12',
    }
    
    @staticmethod
    def create_strategy(model_name: str):
        if model_name in ProteinModelFactory.ESM2_MODELS:
            return ESM2Strategy()
        
        # NEW: ESM-C support
        if model_name in ProteinModelFactory.ESMC_MODELS:
            from src.build.embeddings.strategies.esmc_strategy import ESMCStrategy
            return ESMCStrategy()
        
        raise ValueError(f"Model '{model_name}' not supported")
```

#### Step 1.3: Update Constants

```python
# src/build/core/constants.py

ESM_MODELS = {
    # Existing ESM-2 models...
    'esm2_t6_8M_UR50D': {'dim': 320, 'max_len': 1024},
    
    # NEW: ESM-C models
    'esmc-300m-2024-12': {'dim': 960, 'layers': 30, 'max_len': 2048},
    'esmc-600m-2024-12': {'dim': 1152, 'layers': 36, 'max_len': 2048},
}
```

---

### Phase 2: Full ESM-3 Integration (Advanced)

ESM-3 is a **generative model**, different from ESM-2/ESM-C.

#### Step 2.1: Create ESM3Strategy

```python
# src/build/embeddings/strategies/esm3_strategy.py

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein
from esm.tokenization import get_esm3_model_tokenizers

class ESM3Strategy(BaseProteinStrategy):
    """
    Strategy for ESM-3 models (generative + embeddings).
    
    ESM-3 is a multimodal generative model.
    Can be used for embeddings by extracting from forward pass.
    """
    
    MODEL_SPECS = {
        'esm3_sm_open_v1': {'dim': 1536, 'max_len': 4096},
        # Future: 'esm3-medium': {'dim': 2560, 'max_len': 8192},
    }
    
    def load(self, model_name, device, **kwargs):
        """Load ESM-3 model."""
        model = ESM3.from_pretrained(model_name, device=device)
        tokenizers = get_esm3_model_tokenizers(model_name)
        return model, tokenizers
    
    def generate(self, model, tokenizers, sequence, device, **kwargs):
        """Generate embedding using ESM-3."""
        # Tokenize sequence
        sequence_tokens = tokenizers.sequence.encode(sequence)
        sequence_tokens = torch.tensor(sequence_tokens, dtype=torch.int64).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model.forward(sequence_tokens=sequence_tokens)
            
            # Extract embeddings
            # ESM-3 provides embeddings in output.embeddings
            embedding = output.embeddings.mean(dim=1).squeeze()  # Mean pool over sequence
        
        return embedding.cpu().numpy()
    
    def get_max_length(self, model_name):
        return self.MODEL_SPECS.get(model_name, {}).get('max_len', 4096)
    
    def get_embedding_dim(self, model_name):
        return self.MODEL_SPECS.get(model_name, {}).get('dim', 1536)
    
    def cleanup(self, model, tokenizers):
        gc.collect()
        torch.cuda.empty_cache()
```

---

## 🔄 API Comparison: ESM-2 vs ESM-C vs ESM-3

### ESM-2 (Current Implementation)

```python
# src/build/embeddings/strategies/esm2_strategy.py

import esm

# Load
model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
model = model.to(device).eval()

# Tokenize
batch_converter = alphabet.get_batch_converter()
batch_labels, batch_strs, batch_tokens = batch_converter([("sequence", sequence)])
batch_tokens = batch_tokens.to(device)

# Generate embedding
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[model.num_layers])
    embedding = results["representations"][model.num_layers][0, 1:-1].mean(dim=0)

# Cleanup
del batch_tokens, results
gc.collect()
torch.cuda.empty_cache()
```

### ESM-C (New - Similar to ESM-2)

```python
# Proposed implementation

from esm.models.esmc import ESMC

# Load (very similar!)
model = ESMC.from_pretrained("esmc-600m-2024-12", device=device)
tokenizer = model.tokenizer

# Tokenize (different but simpler)
tokens = model._tokenize([sequence])  # Built-in method!

# Generate embedding
with torch.no_grad():
    output = model.forward(sequence_tokens=tokens)
    embedding = output.embeddings[:, 0, :]  # CLS token

# Cleanup (same)
del tokens, output
gc.collect()
torch.cuda.empty_cache()
```

### ESM-3 (New - More Complex)

```python
# Proposed implementation

from esm.models.esm3 import ESM3
from esm.tokenization import get_esm3_model_tokenizers

# Load
model = ESM3.from_pretrained("esm3_sm_open_v1", device=device)
tokenizers = get_esm3_model_tokenizers("esm3_sm_open_v1")

# Tokenize
sequence_tokens = tokenizers.sequence.encode(sequence)
sequence_tokens = torch.tensor(sequence_tokens).unsqueeze(0).to(device)

# Generate embedding
with torch.no_grad():
    output = model.forward(sequence_tokens=sequence_tokens)
    embedding = output.embeddings.mean(dim=1).squeeze()

# Cleanup
del sequence_tokens, output
gc.collect()
torch.cuda.empty_cache()
```

---

## 🛤️ Migration Path

### For Users

**No breaking changes!** Users can continue using ESM-2:

```python
# Existing code continues to work
gen = ProteinEmbedding(model_name="esm2_t6_8M_UR50D")
gen.initialize()
embedding = gen.generate_embedding(sequence)
```

**New models available via same API**:

```python
# ESM-C (lightweight, fast)
gen = ProteinEmbedding(model_name="esmc-300m-2024-12")

# ESM-3 (powerful, generative)
gen = ProteinEmbedding(model_name="esm3_sm_open_v1")
```

### Recommended Model Selection

| Use Case | Model | Params | Dim | Speed | Quality |
|----------|-------|--------|-----|-------|---------|
| **Fast prototyping** | esm2_t6_8M_UR50D | 8M | 320 | ⚡⚡⚡ | ⭐⭐ |
| **Balanced** | esmc-300m-2024-12 | 300M | 960 | ⚡⚡ | ⭐⭐⭐ |
| **High quality** | esmc-600m-2024-12 | 600M | 1152 | ⚡ | ⭐⭐⭐⭐ |
| **Best quality** | esm3_sm_open_v1 | 1.4B | 1536 | 🐢 | ⭐⭐⭐⭐⭐ |
| **Production** | esm2_t36_3B_UR50D | 3B | 2560 | 🐢🐢 | ⭐⭐⭐⭐⭐ |

---

## ✅ Implementation Checklist

### Phase 1: ESM-C (Priority 1)

- [ ] **1.1** Create `esmc_strategy.py`
- [ ] **1.2** Register in `protein_model_factory.py`
- [ ] **1.3** Update `constants.py`
- [ ] **1.4** Add tests (`test_esmc_strategy.py`)
- [ ] **1.5** Update documentation
- [ ] **1.6** Test end-to-end with pipeline

### Phase 2: ESM-3 (Priority 2)

- [ ] **2.1** Create `esm3_strategy.py`
- [ ] **2.2** Register in factory
- [ ] **2.3** Update constants
- [ ] **2.4** Add tests
- [ ] **2.5** Update documentation
- [ ] **2.6** Test end-to-end

### Phase 3: Advanced Features (Future)

- [ ] **3.1** ESM-3 structure prediction support
- [ ] **3.2** ESM-3 function annotation support
- [ ] **3.3** Multi-modal embedding fusion
- [ ] **3.4** API access for ESM-3 large models

---

## 🧪 Testing Strategy

### Unit Tests

```python
# tests/test_esmc_strategy.py

def test_esmc_strategy_creation():
    strategy = ESMCStrategy()
    assert isinstance(strategy, BaseProteinStrategy)

def test_esmc_embedding_generation():
    strategy = ESMCStrategy()
    model, tokenizer = strategy.load("esmc-300m-2024-12", torch.device("cpu"))
    
    sequence = "MKFLILLFNILCLFPVLA"
    embedding = strategy.generate(model, tokenizer, sequence, torch.device("cpu"))
    
    assert embedding.shape == (960,)
    assert not np.isnan(embedding).any()
```

### Integration Tests

```python
# tests/test_esm3_integration.py

def test_esmc_end_to_end():
    gen = ProteinEmbedding(model_name="esmc-300m-2024-12")
    gen.initialize()
    
    embedding = gen.generate_embedding("MKFLILLFNILCLFPVLA")
    assert embedding.shape == (960,)

def test_esm3_end_to_end():
    gen = ProteinEmbedding(model_name="esm3_sm_open_v1")
    gen.initialize()
    
    embedding = gen.generate_embedding("MKFLILLFNILCLFPVLA")
    assert embedding.shape == (1536,)
```

---

## 📊 Performance Comparison

### Expected Performance

| Model | Load Time | Embedding Time (1 seq) | Memory |
|-------|-----------|------------------------|--------|
| esm2_t6_8M | ~2s | ~50ms | ~500MB |
| esmc-300m | ~5s | ~100ms | ~1.5GB |
| esmc-600m | ~8s | ~150ms | ~3GB |
| esm3_sm_open | ~15s | ~300ms | ~6GB |

---

## 🎯 Conclusion

### Recommended Approach: **Hybrid Integration**

1. **Start with ESM-C** (Phase 1) - Quick win, similar API
2. **Add ESM-3** (Phase 2) - Full generative capabilities
3. **Maintain ESM-2** - Backward compatibility

### Key Benefits

✅ **Backward Compatible**: Existing ESM-2 code continues to work  
✅ **Strategy Pattern**: Clean, extensible architecture  
✅ **User Choice**: Users can select best model for their needs  
✅ **Future-Proof**: Easy to add more models (ESM-3 medium/large)  
✅ **Progressive Enhancement**: Start small (ESM-C), scale up (ESM-3)

### Timeline Estimate

- **Phase 1 (ESM-C)**: 2-3 days
- **Phase 2 (ESM-3)**: 3-4 days
- **Testing & Documentation**: 2 days
- **Total**: ~1-2 weeks

---

## 📚 References

- ESM-3 Paper: https://www.science.org/doi/10.1126/science.ads0018
- ESM-3 GitHub: /ESM/esm-3/esm-main/
- ESM-C Blog: https://www.evolutionaryscale.ai/blog/esm-cambrian
- Current Implementation: src/build/embeddings/strategies/esm2_strategy.py
