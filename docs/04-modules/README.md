# Modules

**Last Updated**: October 28, 2025  
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
   - 6 classification models
   - Training and prediction

3. **[Regression Module](regression-module.md)** 📊
   - `src/regression/` documentation
   - 11 regression models
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
   - FM4M protein features
   - Feature extraction
   - Model integration

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
