# 📚 DockTKinase Documentation

**Last Updated**: October 28, 2025  
**Version**: 2.0  
**Branch**: regression

---

Welcome to the complete documentation for **DockTKinase**, a dual-pipeline machine learning system for protein-ligand interaction prediction.

## 🎯 What is DockTKinase?

DockTKinase provides:
- **Classification Pipeline**: 6 models for binary kinase activity prediction
- **Regression Pipeline**: 11 models for quantitative binding affinity prediction
- **Protein Embeddings**: ESM-2 transformer-based representations
- **Ligand Embeddings**: FM4M and SMI-TED molecular features
- **Automated Workflows**: End-to-end pipelines from data to predictions

---

## 🚀 Quick Navigation

### For New Users
**Start here!** Get up and running in 5 minutes:
- 📖 [Chapter 01: Getting Started](01-getting-started/README.md)
  - [Quick Start Guide](01-getting-started/quick-start.md) ⚡
  - [Installation Instructions](01-getting-started/installation.md)
  - [Prerequisites](01-getting-started/prerequisites.md)
  - [Basic Troubleshooting](01-getting-started/troubleshooting.md)

### For End Users
Learn how to use the pipelines effectively:
- 📘 [Chapter 02: User Guide](02-user-guide/README.md)
  - [User Manual](02-user-guide/user-manual.md)
  - [Execution Guide](02-user-guide/execution-guide.md)
  - [Classification Pipeline](02-user-guide/classification-pipeline.md)
  - [Regression Pipeline](02-user-guide/regression-pipeline.md)
  - [Model Comparison](02-user-guide/model-comparison.md)
  - [Visualization](02-user-guide/visualization.md)

### For Developers
Understand the architecture and contribute:
- 🏗️ [Chapter 03: Architecture](03-architecture/README.md)
  - [Project Structure](03-architecture/project-structure.md)
  - [Modularization](03-architecture/modularization.md)
  - [Build System](03-architecture/build-system.md)
  - [Dual Pipeline](03-architecture/dual-pipeline.md)

- 🧩 [Chapter 04: Modules](04-modules/README.md)
  - [Build Module](04-modules/build-module.md)
  - [Classifier Module](04-modules/classifier-module.md)
  - [Regression Module](04-modules/regression-module.md)
  - [Utils Module](04-modules/utils-module.md)
  - [ESM Integration](04-modules/esm-integration.md)

- 👨‍💻 [Chapter 05: Development](05-development/README.md)
  - [Modularization Strategy](05-development/modularization-strategy.md)
  - [Modularization Status](05-development/modularization-status.md)
  - [Dependency Management](05-development/dependency-management.md)

### Quality Assurance
- ✅ [Chapter 06: Validation Reports](06-validation-reports/README.md)
  - [Build Validation](06-validation-reports/build-validation.md)
  - [ESM Validation](06-validation-reports/esm-validation.md)
  - [Optimization Validation](06-validation-reports/optimization-validation.md)
  - [Pipeline Success](06-validation-reports/pipeline-success.md)

### Need Help?
- 🔧 [Chapter 07: Troubleshooting](07-troubleshooting/README.md)
  - [HuggingFace Rate Limit](07-troubleshooting/huggingface-rate-limit.md)
  - [Warnings Resolution](07-troubleshooting/warnings-resolution.md)
  - [Memory Management](07-troubleshooting/memory-management.md)
  - [Setup Issues](07-troubleshooting/setup-issues.md)

### Maintenance & History
- 🧹 [Chapter 08: Maintenance](08-maintenance/README.md)
- 📝 [Chapter 09: Changelogs](09-changelogs/README.md)
- 📚 [Chapter 10: Reference](10-reference/README.md)

---

## 📊 System Overview

### Dual Pipeline System

```
┌─────────────────────────────────────────────┐
│         INPUT: TSV Dataset                  │
│  • Protein Sequences                        │
│  • Ligand SMILES                            │
│  • Labels (Binary + Affinity Values)        │
└────────────────┬────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────┐   ┌──────────────┐
│ PROTEIN      │   │  LIGAND      │
│ EMBEDDINGS   │   │  EMBEDDINGS  │
│ (ESM-2)      │   │  (FM4M/SMI)  │
└──────┬───────┘   └──────┬───────┘
       │                  │
       └────────┬─────────┘
                │
                ▼
        ┌──────────────┐
        │   FEATURE    │
        │   MATRIX     │
        └──────┬───────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│CLASSIFICATION│  │ REGRESSION  │
│  (6 models)  │  │ (11 models) │
└──────┬───────┘  └──────┬──────┘
       │                 │
       ▼                 ▼
┌─────────────┐  ┌─────────────┐
│   BINARY    │  │QUANTITATIVE │
│ PREDICTIONS │  │ PREDICTIONS │
└─────────────┘  └─────────────┘
```

### Key Features

#### 🎯 Classification Pipeline
- **6 Machine Learning Models**
  - Random Forest
  - XGBoost  
  - Gradient Boosting
  - SVM
  - KNN
  - MLP Neural Network

- **Binary Prediction**: Active/Inactive
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC

#### 📊 Regression Pipeline
- **11 Regression Models**
  - Linear: LinearRegression, Ridge, Lasso, ElasticNet
  - Tree-based: DecisionTree, RandomForest, GradientBoosting, XGBoost
  - Other: SVR, KNN, MLP

- **Quantitative Prediction**: Ki, Kd, IC50 (nM)
- **Metrics**: R², MAE, RMSE, Pearson Correlation

#### 🧬 Protein Embeddings
- **ESM-2**: Facebook AI's protein language model
- **1280-dimensional** contextualized embeddings
- **GPU-accelerated** inference

#### 🔬 Ligand Embeddings
- **FM4M**: Molecular fingerprint features
- **SMI-TED**: SMILES-based embeddings
- **Flexible architecture** for custom features

---

## 📋 Quick Start Commands

```bash
# 1. Clone and install
git clone https://github.com/gmmsb-lncc/docktkinase.git
cd docktkinase
python setup.py

# 2. Activate environment
source env/bin/activate

# 3. Run classification pipeline
python run_complete_pipeline.py

# 4. Run regression pipeline
python run_regression_pipeline.py
```

**Results**: Generated in `results/<dataset>/` directory

---

## 🎓 Learning Paths

### Beginner Path
1. Read [Quick Start](01-getting-started/quick-start.md)
2. Follow [Installation Guide](01-getting-started/installation.md)
3. Try [User Manual](02-user-guide/user-manual.md)
4. Run first predictions

### Advanced User Path
1. Study [Architecture](03-architecture/README.md)
2. Understand [Dual Pipeline](03-architecture/dual-pipeline.md)
3. Read [Module Documentation](04-modules/README.md)
4. Customize pipelines

### Developer Path
1. Review [Project Structure](03-architecture/project-structure.md)
2. Study [Modularization](05-development/modularization-strategy.md)
3. Read [Testing Guide](05-development/testing-guide.md)
4. Contribute code

---

## 📈 Project Statistics

- **Total Modules**: 9
- **Classification Models**: 6
- **Regression Models**: 11
- **Automated Tests**: 19
- **Documentation Files**: 40+
- **Supported Platforms**: macOS, Linux (CUDA), Linux (CPU)

---

## 🔗 Important Links

### External Resources
- **ESM-2**: https://github.com/facebookresearch/esm
- **PyTorch**: https://pytorch.org/
- **scikit-learn**: https://scikit-learn.org/
- **RDKit**: https://www.rdkit.org/

### Project Repository
- **GitHub**: https://github.com/gmmsb-lncc/docktkinase
- **Main Branch**: `esm`
- **Development Branch**: `regression`

---

## 🆘 Getting Help

### Documentation Search
Use the chapter-based structure to find what you need:
- **Installation issues?** → [Chapter 01](01-getting-started/README.md) or [Chapter 07](07-troubleshooting/README.md)
- **How to use?** → [Chapter 02](02-user-guide/README.md)
- **Understanding code?** → [Chapter 03](03-architecture/README.md) or [Chapter 04](04-modules/README.md)
- **Contributing?** → [Chapter 05](05-development/README.md)

### Common Questions
- **"How do I install?"** → [Installation Guide](01-getting-started/installation.md)
- **"How do I run predictions?"** → [Quick Start](01-getting-started/quick-start.md)
- **"What models are available?"** → [Classification](02-user-guide/classification-pipeline.md) + [Regression](02-user-guide/regression-pipeline.md)
- **"How does it work?"** → [Architecture](03-architecture/README.md)
- **"I found a bug"** → [GitHub Issues](https://github.com/gmmsb-lncc/docktkinase/issues)

---

## ✅ Documentation Status

- ✅ **Getting Started**: Complete
- ✅ **User Guide**: Complete
- ✅ **Architecture**: Complete
- ✅ **Modules**: Complete
- ✅ **Development**: Complete
- ✅ **Validation**: Complete
- ✅ **Troubleshooting**: Complete
- ✅ **Maintenance**: Complete
- ✅ **Changelogs**: Complete
- 🔄 **Reference**: In Progress

---

## 📝 Documentation Conventions

### Status Indicators
- ✅ **Complete**: Fully documented and validated
- 🔄 **In Progress**: Currently being documented
- 🆕 **New**: Recently added
- ⚠️ **Needs Update**: Requires revision

### Document Headers
All documents include:
- **Last Updated**: Date of last modification
- **Section**: Chapter number and name
- **Audience**: Target reader (Users/Developers/All)
- **Status**: Current documentation status

---

## 🌟 Latest Updates (October 2025)

- ✨ **NEW**: Complete documentation reorganization
- ✨ **NEW**: English-only documentation
- ✨ **NEW**: Chapter-based structure (01-10)
- ✨ **NEW**: Improved navigation
- ✨ **ENHANCED**: Regression pipeline documentation
- ✨ **ENHANCED**: Module-specific guides

---

**Ready to start?** → [Go to Quick Start →](01-getting-started/quick-start.md)

**Need installation help?** → [Go to Installation Guide →](01-getting-started/installation.md)

**Want to understand the system?** → [Go to Architecture →](03-architecture/README.md)

---

*Last documentation update: October 28, 2025*  
*Documentation version: 2.0*  
*Project branch: regression*
