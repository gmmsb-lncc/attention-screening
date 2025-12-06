# Modules

**Last Updated**: November 25, 2025  
**Section**: Chapter 04  
**Audience**: Developers

---

Detailed documentation for each module in the DockTKinase project.

## 📚 Contents

### Core Modules

1. **[Build Module](build-module.md)** 🔨
   - `src/build/` documentation
   - Automated builds
   - Testing infrastructure

2. **[Classifier Module](classifier-module.md)** 🎯
   - `src/classifier/` documentation
   - 10 classification models
   - Training and prediction

3. **[Regression Module](regression-module.md)** 📊
   - `src/regression/` documentation
   - 10 regression models
   - Validation strategies

4. **[Utils Module](utils-module.md)** 🛠️
   - `src/utils/` documentation
   - Configuration management
   - Device management
   - Shared utilities

### Integration Modules

5. **[ESM Integration](esm-integration.md)** 🧬
   - ESM-2 protein embeddings
   - Integration with classifier/regression
   - Cache management

6. **[FM4M Integration](fm4m-integration.md)** 🔬
   - SMI-TED ligand embeddings
   - Feature extraction
   - Model integration

7. **[Boltz-2 Strategy](BOLTZ_STRATEGY_GUIDE.md)** ⚡
   - Structure + Affinity prediction
   - 384-dim embeddings

8. **[OpenFold3 Strategy](OPENFOLD_EMBEDDING_EXTRACTION.md)** 🧬
   - Structure-aware embeddings
   - AlphaFold3 reproduction

### 🆕 Stratification System (NEW - November 2025!)

9. **[ADAPTIVE_CLUSTERING_GUIDE.md](ADAPTIVE_CLUSTERING_GUIDE.md)** 🎯 **← LATEST**
   - **Automatic threshold detection** for homogeneous embeddings
   - 5 optimization methods: silhouette, elbow, target, percentile, manual
   - JSON metrics export (clustering_metrics.json, split_info.json)
   - CLI options: `--stratifier-threshold`, `--stratifier-method`
   - Complete API reference & examples

8. **[MULTI_VIEW_STRATIFICATION.md](MULTI_VIEW_STRATIFICATION.md)** ⚖️
   - **Multi-view similarity calculation**
   - Protein/ligand weight balancing (α=0.6, β=0.4)
   - Biologically coherent clustering
   - Data leakage prevention

### 🆕 Protein Embedding System

9. **[BOLTZ_STRATEGY_GUIDE.md](BOLTZ_STRATEGY_GUIDE.md)** ⚡
   - **Boltz-2 Integration Guide** (350+ lines)
   - Multi-pooling strategy (CLS + mean + max)
   - 384-dim single representation
   - Performance benchmarks vs ESM-2/ESMFold
   - Memory optimization strategies
   - Complete API reference & examples
   - **Recommended for production workflows**

10. **[PROTEIN_EMBEDDING_API.md](PROTEIN_EMBEDDING_API.md)** 🎯
    - **Complete API Reference** (11,500+ lines)
    - Architecture overview with diagrams
    - Integration guide for ESM-3, OpenFold, custom models
    - Memory management & best practices
    - Testing & troubleshooting guide
    - **Essential for model integration**

11. **[INTEGRATION_EXAMPLES.md](INTEGRATION_EXAMPLES.md)** 🚀
    - **Copy-paste ready code** (3,500+ lines)
    - Complete ESM-3 integration example
    - Complete OpenFold integration template
    - ProtTrans integration pattern
    - Test suite templates
    - **Essential for implementation**

### 🆕 Deep Learning Architecture (NEW - CNN Optimization!)

12. **[CNN_CROSS_ATTENTION_ARCHITECTURE.md](CNN_CROSS_ATTENTION_ARCHITECTURE.md)** 🧠 **← NEW**
    - **CNN + Cross-Attention Model Architecture** (500+ lines)
    - Depthwise Separable Convolutions (Chollet, 2017)
    - Squeeze-and-Excitation blocks (Hu et al., 2018)
    - Pre-LayerNorm Transformers (Xiong et al., 2020)
    - 56% parameter reduction with 16% larger receptive field
    - Scientific references and design rationale
    - **Essential for understanding the affinity prediction model**

---

## 🎯 Module Overview

### Classification Pipeline Modules
```
src/classifier/
├── data_loader.py          # Data loading and preprocessing
├── model_*.py              # 6 classifier implementations
├── train.py                # Training scripts
└── predict.py              # Prediction interface
```

### Regression Pipeline Modules
```
src/regression/
├── data_loader.py          # Data loading and preprocessing
├── models/                 # 11 regression model implementations
├── train.py                # Training scripts
└── validation.py           # Validation strategies
```

### Utility Modules
```
src/utils/
├── config_manager.py       # Configuration management
├── device_manager.py       # GPU/CPU device handling
├── logger.py               # Logging utilities
└── metrics.py              # Evaluation metrics
```

---

## 🔗 Related Documentation

- **Architecture Overview?** → [Chapter 03: Architecture](../03-architecture/README.md)
- **Development Guide?** → [Chapter 05: Development](../05-development/README.md)
- **API Reference?** → [Chapter 10: Reference](../10-reference/README.md)

---

**Previous**: [← Architecture](../03-architecture/README.md) | **Next**: [Development →](../05-development/README.md)
