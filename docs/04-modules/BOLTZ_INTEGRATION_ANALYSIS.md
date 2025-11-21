# Boltz-2 Integration Analysis

**Date**: 2025-11-20  
**Branch**: `boltz`  
**Status**: Planning Phase  

## 📋 Executive Summary

This document analyzes the architecture and integration requirements for Boltz-2 into the DockTKinase pipeline, following the same pattern used for OpenFold3 integration.

---

## 🎯 Integration Goals

1. **Extract embeddings** from Boltz-2 model (not full structure prediction)
2. **Maintain compatibility** with existing DockTKinase pipeline
3. **Follow SOLID principles** and match existing strategy patterns
4. **Provide multiple extraction modes** (trunk representations, confidence features, etc.)

---

## 🏗️ Boltz-2 Architecture Analysis

### Model Overview

**Boltz-2** is a biomolecular foundation model that:
- Predicts complex structures (protein, DNA, RNA, ligands)
- Computes binding affinities
- Goes beyond AlphaFold3 and Boltz-1 in accuracy
- Uses MSA (via ColabFold) similar to OpenFold3
- Supports inference-time potentials for physical plausibility

### Key Technical Specifications

#### Model Components (from `boltz2.py`):

```python
class Boltz2(LightningModule):
    def __init__(
        self,
        atom_s: int,              # Atom single representation dim
        atom_z: int,              # Atom pair representation dim
        token_s: int,             # Token single representation dim
        token_z: int,             # Token pair representation dim
        num_bins: int,
        embedder_args: dict,      # Input embedder configuration
        msa_args: dict,           # MSA module configuration
        pairformer_args: dict,    # Pairformer module configuration
        confidence_model_args: dict,  # Confidence prediction
        affinity_model_args: dict,    # Affinity prediction (NEW!)
        ...
    )
```

#### Key Modules:

1. **InputEmbedder**: Converts features to initial embeddings
2. **MSAModule**: Multiple Sequence Alignment processing
3. **PairformerModule**: Pairformer stack (64 blocks in Boltz-2 vs 48 in Boltz-1)
4. **DiffusionConditioning**: Prepares features for structure generation
5. **AtomDiffusion**: Structure prediction via diffusion
6. **ConfidenceModule**: Confidence predictions (pLDDT, PAE, etc.)
7. **AffinityModule**: Binding affinity prediction (NEW in Boltz-2!)

### Forward Pass Architecture

```python
def forward(self, feats, recycling_steps=0, ...):
    # 1. Input embeddings
    s_inputs = self.input_embedder(feats)
    
    # 2. Initialize representations
    s_init = self.s_init(s_inputs)          # Token single [B, N, token_s]
    z_init = self.z_init_1(s_inputs)[...] + # Token pair [B, N, N, token_z]
             self.z_init_2(s_inputs)[...]
    
    # 3. Recycling loop (structure)
    for i in range(recycling_steps + 1):
        s = s_init + self.s_recycle(self.s_norm(s))
        z = z_init + self.z_recycle(self.z_norm(z))
        
        # 3a. Template module (if enabled)
        if self.use_templates:
            z = z + self.template_module(z, feats, pair_mask)
        
        # 3b. MSA module
        z = z + self.msa_module(z, s_inputs, feats)
        
        # 3c. Pairformer stack
        s, z = self.pairformer_module(s, z, mask, pair_mask)
    
    # 4. Output representations
    dict_out = {
        's': s,          # Token single representations
        'z': z,          # Token pair representations
        'pdistogram': self.distogram_module(z),  # Distance predictions
    }
    
    # 5. Structure prediction (diffusion)
    if run_structure:
        q, c, ... = self.diffusion_conditioning(s, z, ...)
        struct_out = self.structure_module.sample(...)
        dict_out.update(struct_out)
    
    # 6. Confidence prediction
    if self.confidence_prediction:
        conf_out = self.confidence_module(s, z, ...)
        dict_out.update(conf_out)
    
    # 7. Affinity prediction (NEW!)
    if self.affinity_prediction:
        aff_out = self.affinity_module(s, z, ...)
        dict_out.update(aff_out)
    
    return dict_out
```

---

## 🔍 Comparison: Boltz-2 vs OpenFold3

| Feature | OpenFold3 | Boltz-2 |
|---------|-----------|---------|
| **Primary Goal** | Structure prediction | Structure + Affinity |
| **Pairformer Blocks** | 48 | 64 |
| **MSA Support** | Yes (via ColabFold) | Yes (via ColabFold) |
| **Single Rep Dim** | 384 (c_s) | Configurable (token_s) |
| **Pair Rep Dim** | 128 (c_z) | Configurable (token_z) |
| **Affinity Module** | ❌ | ✅ (NEW!) |
| **Template Support** | Yes | Yes (improved v2) |
| **Confidence Outputs** | pLDDT, PAE | pLDDT, PAE, token-level |
| **Inference API** | `run_trunk()` | `forward()` |
| **Model Access** | Local (OPENFOLD-3/) | Local (BOLTZ-2/) |

---

## 📐 Proposed BoltzStrategy Design

### Class Structure

```python
class BoltzStrategy(BaseProteinStrategy):
    """
    Strategy for generating protein embeddings using Boltz-2.
    
    Boltz-2 is a biomolecular foundation model that predicts structures
    and binding affinities. This strategy extracts intermediate embeddings
    without performing full structure prediction.
    
    Extraction Modes:
    - 'trunk': Extract s and z from trunk (after pairformer)
    - 'confidence': Extract confidence representations
    - 'affinity': Extract affinity-aware representations
    - 'combined': Combine multiple representations
    """
    
    def __init__(
        self,
        extraction_mode: str = 'trunk',
        pooling_strategy: str = 'mean',
        use_msa: bool = True,
        recycling_steps: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Boltz strategy.
        
        Args:
            extraction_mode: How to extract embeddings
                - 'trunk': s and z after pairformer (default)
                - 'confidence': Include confidence features
                - 'affinity': Include affinity features
                - 'combined': All features
            pooling_strategy: How to pool token representations
                - 'mean': Mean pooling over sequence (default)
                - 'cls': Use first token
                - 'max': Max pooling
            use_msa: Whether to use MSA (requires ColabFold server)
            recycling_steps: Number of recycling iterations
        """
        pass
    
    def load(self, model_name, device, **kwargs):
        """
        Load Boltz-2 model from local installation.
        
        Supports:
        - 'boltz1': Boltz-1 (structure only)
        - 'boltz2': Boltz-2 (structure + affinity)
        - 'boltz2-conf': Boltz-2 confidence model
        - 'boltz2-aff': Boltz-2 affinity model
        """
        pass
    
    def generate(self, model, tokenizer, sequence, device):
        """
        Generate embeddings from Boltz-2.
        
        Process:
        1. Prepare input features (sequence, MSA if enabled)
        2. Run forward pass (trunk only, no structure prediction)
        3. Extract representations based on extraction_mode
        4. Pool to fixed-dimensional embedding
        
        Returns:
            numpy array: Fixed-dimensional embedding
        """
        pass
    
    def cleanup(self, model, tokenizer):
        """Clean up model and restore namespace."""
        pass
```

### Dimension Specifications

#### Boltz-1 (default config):
- `token_s`: 768
- `token_z`: 128
- **Output embedding**: 768-dim (mean-pooled `s`)

#### Boltz-2 (default config):
- `token_s`: 768 (likely, check checkpoint)
- `token_z`: 128
- **Output embedding**: 768-dim (or combined with confidence/affinity)

### Extraction Strategies

#### 1. **Trunk Mode** (Default - fastest)
```python
# Extract only trunk representations
s = output['s']  # [B, N, token_s]
z = output['z']  # [B, N, N, token_z]

# Pool to fixed dimension
embedding = mean_pool(s, mask)  # [B, token_s]
```

#### 2. **Confidence Mode**
```python
# Extract trunk + confidence
s = output['s']
confidence_feats = extract_confidence_features(output)

# Combine
embedding = concat([mean_pool(s), confidence_feats])
```

#### 3. **Affinity Mode** (Boltz-2 only)
```python
# Extract trunk + affinity
s = output['s']
affinity_feats = extract_affinity_features(output)

# Combine
embedding = concat([mean_pool(s), affinity_feats])
```

#### 4. **Combined Mode**
```python
# All features
embedding = concat([
    mean_pool(s),
    confidence_feats,
    affinity_feats
])
```

---

## 🔧 Implementation Requirements

### Dependencies (similar to OpenFold3)

```python
# Already installed (from OpenFold3):
- gemmi>=0.7.3
- ml-collections>=1.1.0
- einops>=0.8.0
- biopython>=1.86
- pydantic>=2.0
- lmdb>=1.7.0
- biotite>=1.0
- memory-profiler>=0.61.0
- lightning>=2.0

# Potentially NEW for Boltz-2:
- rdkit>=2023.9.1 (for ligand handling)
# Check BOLTZ-2/boltz-main/pyproject.toml
```

### File Structure

```
src/build/embeddings/
├── strategies/
│   ├── base_protein_strategy.py     (existing)
│   ├── openfold_strategy.py         (existing)
│   └── boltz_strategy.py            (NEW)
├── factories/
│   └── protein_model_factory.py     (update)
└── config/
    ├── msa_config.py                (existing, reuse?)
    └── boltz_config.py              (NEW, if needed)
```

### Configuration Management

Option 1: **Reuse MSA Config** (if compatible)
```python
from src.build.embeddings.config.msa_config import MsaConfig

# Same as OpenFold3
msa_config = MsaConfig.for_production()
```

Option 2: **Create Boltz-specific Config**
```python
@dataclass
class BoltzConfig:
    """Boltz-2 specific configuration."""
    extraction_mode: str = 'trunk'
    pooling_strategy: str = 'mean'
    use_msa: bool = True
    msa_config: Optional[MsaConfig] = None
    recycling_steps: int = 3
    run_structure: bool = False  # Don't run diffusion
    run_confidence: bool = False
    run_affinity: bool = False
    
    @classmethod
    def for_embeddings_only(cls):
        """Fast mode: trunk only, no MSA."""
        return cls(
            extraction_mode='trunk',
            use_msa=False,
            run_structure=False
        )
    
    @classmethod
    def for_affinity_embeddings(cls):
        """Affinity-aware embeddings."""
        return cls(
            extraction_mode='affinity',
            use_msa=True,
            run_structure=False,
            run_affinity=True
        )
```

---

## ⚠️ Key Challenges & Differences

### 1. **Model Checkpoint Size**
- Boltz-2 checkpoints are **larger** than OpenFold3
- Need to check available disk space
- Consider auto-download vs manual download

### 2. **MSA Handling**
- Boltz uses same ColabFold API as OpenFold3 ✅
- Can reuse MSA caching infrastructure ✅
- MSA format compatible ✅

### 3. **Input Format Differences**
- OpenFold3: Simple sequence string
- Boltz-2: **YAML format** with complex specifications
  ```yaml
  sequences:
    - protein:
        id: A
        sequence: MKFLKFSL...
        msa: path/to/msa.a3m  # or auto-generate
  ```
- **Adaptation needed**: Create minimal YAML from sequence

### 4. **Inference API**
- OpenFold3: `model.run_trunk(batch)` → returns s, z directly
- Boltz-2: `model.forward(feats, recycling_steps=N)` → dict output
- **Need wrapper** to extract only trunk representations

### 5. **Namespace Isolation**
- Boltz-2 in `BOLTZ-2/boltz-main/src/boltz/`
- Must isolate from ESM, OpenFold3, FM4M
- Use same pattern as OpenFold3 ✅

### 6. **Device Compatibility**
- Boltz-2 supports CUDA, CPU
- Check MPS (Apple Silicon) support
- May need CPU fallback for some operations

---

## 📊 Embedding Dimension Comparison

| Model | Default Output Dim | Representation Source |
|-------|-------------------|----------------------|
| ESM-2 (650M) | 1280 | Last layer, mean pool |
| ESM-C (300M) | 960 | Last layer, mean pool |
| ESM-3 (large) | 1536 | Last layer, mean pool |
| OpenFold3 | 384 | Single rep (s), mean pool |
| **Boltz-1** | **768** | Token single (s), mean pool |
| **Boltz-2** | **768+** | s + confidence/affinity |

**Recommendation**: 
- Default: 768-dim (trunk only)
- Extended: 768 + confidence (e.g., 768+32 = 800-dim)
- Full: 768 + confidence + affinity (e.g., 768+32+16 = 816-dim)

---

## 🎯 Recommended Implementation Plan

### Phase 1: Basic Integration (Similar to OpenFold3)
1. ✅ Create `docs/04-modules/BOLTZ_INTEGRATION_ANALYSIS.md` (this file)
2. ⏳ Create `BoltzStrategy` class in `src/build/embeddings/strategies/boltz_strategy.py`
3. ⏳ Implement `load()`: Import Boltz-2, load model
4. ⏳ Implement `generate()`: Extract trunk representations (s), mean pool
5. ⏳ Update `ProteinModelFactory` to recognize 'boltz1', 'boltz2'
6. ⏳ Add tests: `test_boltz_basic.py`, `test_boltz_embedding.py`

### Phase 2: MSA Integration
1. ⏳ Integrate ColabFold MSA (reuse OpenFold3 infrastructure)
2. ⏳ Create YAML input from sequence
3. ⏳ Test with/without MSA

### Phase 3: Advanced Features
1. ⏳ Implement confidence extraction mode
2. ⏳ Implement affinity extraction mode (Boltz-2 only)
3. ⏳ Implement combined mode
4. ⏳ Create `BoltzConfig` for configuration management

### Phase 4: Pipeline Integration
1. ⏳ Update `run_complete_pipeline.py`
2. ⏳ Add dimension mapping: `'boltz1': 768, 'boltz2': 768`
3. ⏳ Test end-to-end: sequence → embedding → classification
4. ⏳ Benchmark performance vs OpenFold3

### Phase 5: Documentation & Testing
1. ⏳ Create `BOLTZ_USAGE_GUIDE.md`
2. ⏳ Update README.md
3. ⏳ Add examples: `examples/boltz_embedding_extraction.py`
4. ⏳ Performance benchmarks

---

## 📝 Code Snippets for Reference

### Minimal Boltz-2 Forward Pass (Trunk Only)

```python
import torch
from boltz.model.models.boltz2 import Boltz2

# Load model
model = Boltz2.load_from_checkpoint('path/to/boltz2_conf.ckpt')
model.eval()

# Prepare features (simplified)
feats = prepare_features(sequence, msa_path=None)

# Forward pass (trunk only, no structure)
with torch.no_grad():
    output = model.forward(
        feats,
        recycling_steps=3,
        run_confidence_sequentially=False
    )

# Extract embeddings
s = output['s']  # [B, N_tokens, token_s]
z = output['z']  # [B, N_tokens, N_tokens, token_z]

# Mean pool
mask = feats['token_pad_mask']
embedding = mean_pool(s, mask)  # [B, token_s]
```

### Feature Preparation Helper

```python
def prepare_boltz_features(sequence: str, use_msa: bool = False):
    """
    Prepare Boltz-2 input features from sequence.
    
    Minimal implementation for embedding extraction.
    """
    from boltz.data.parse.yaml import parse_yaml
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    
    # Create minimal YAML
    yaml_content = f"""
    version: 1
    sequences:
      - protein:
          id: A
          sequence: {sequence}
          {'msa: empty' if not use_msa else ''}
    """
    
    # Parse and prepare
    manifest = parse_yaml(yaml_content)
    # ... (continue with data preparation)
    
    return feats
```

---

## ✅ Success Criteria

1. **Functional Integration**:
   - BoltzStrategy loads Boltz-2 model successfully
   - Generates fixed-dimensional embeddings (768-dim default)
   - Compatible with existing pipeline

2. **Performance**:
   - Trunk extraction: < 5 seconds per sequence (no MSA)
   - With MSA: 3-5 minutes first run, < 1 min cached (similar to OpenFold3)

3. **Code Quality**:
   - Follows SOLID principles
   - Matches existing strategy patterns
   - Comprehensive tests (>80% coverage)
   - Full documentation

4. **Pipeline Compatibility**:
   - Works with `run_complete_pipeline.py`
   - Embeddings concatenate correctly with ESM/OpenFold3
   - No namespace conflicts

---

## 🔗 Related Documentation

- [OpenFold3 Integration](./OPENFOLD_INTEGRATION_GUIDE.md)
- [MSA Configuration](./OPENFOLD_MSA_GUIDE.md)
- [Base Strategy Pattern](../../src/build/embeddings/strategies/base_protein_strategy.py)
- [Boltz-2 Paper](https://doi.org/10.1101/2025.06.14.659707)
- [Boltz-2 GitHub](https://github.com/jwohlwend/boltz)

---

## 📅 Next Steps

**IMMEDIATE**:
1. Check Boltz-2 checkpoint availability and size
2. Test basic Boltz-2 import and forward pass
3. Determine exact `token_s` and `token_z` dimensions from checkpoint
4. Create minimal feature preparation from sequence

**SHORT-TERM**:
1. Implement `BoltzStrategy.load()`
2. Implement `BoltzStrategy.generate()` (trunk mode)
3. Create basic tests
4. Update ProteinModelFactory

**MEDIUM-TERM**:
1. MSA integration
2. Advanced extraction modes
3. Full pipeline integration
4. Performance optimization

---

**Document Status**: ✅ Complete  
**Ready for Implementation**: ✅ Yes  
**Reviewed by**: DockTKinase Team  
**Last Updated**: 2025-11-20
