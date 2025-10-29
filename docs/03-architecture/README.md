# Architecture

**Last Updated**: October 28, 2025  
**Section**: Chapter 03  
**Audience**: Developers & Advanced Users

---

Comprehensive documentation of DockTKinase's system architecture, modularization, and design principles.

## 📚 Contents

### System Design

1. **[Project Structure](project-structure.md)** 🏗️
   - Complete directory layout
   - 9 main modules
   - Component organization

2. **[Modularization](modularization.md)** 🧩
   - Modular architecture design
   - Module responsibilities
   - Integration patterns

3. **[Build System](build-system.md)** ⚙️
   - Build pipeline architecture
   - Dependency management
   - Compilation process

### Pipeline Architecture

4. **[Dual Pipeline](dual-pipeline.md)** 🔀
   - Classification pipeline (6 models)
   - Regression pipeline (11 models)
   - Pipeline orchestration

5. **[Data Flow](data-flow.md)** 🌊
   - Data processing pipeline
   - Feature engineering
   - Model inference flow

---

## 🎯 Key Concepts

### Modular Design
- **9 Core Modules**: build, classifier, database, regression, utils, etc.
- **Separation of Concerns**: Clear module boundaries
- **Reusability**: Shared utilities and components

### Dual Pipeline System
- **Classification**: 6 models for kinase family prediction
- **Regression**: 11 models for binding affinity prediction
- **Integration**: ESM-2 embeddings + FM4M features

### Build Strategy
- **Incremental builds**: Optimized compilation
- **Dependency tracking**: Automatic rebuilds
- **Testing integration**: 19 automated tests

---

## 🔗 Related Documentation

- **Module Details?** → [Chapter 04: Modules](../04-modules/README.md)
- **Development Guide?** → [Chapter 05: Development](../05-development/README.md)
- **Validation Reports?** → [Chapter 06: Validation Reports](../06-validation-reports/README.md)

---

**Previous**: [← User Guide](../02-user-guide/README.md) | **Next**: [Modules →](../04-modules/README.md)
