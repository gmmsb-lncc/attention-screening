# DockTKinase - Integrated Protein-Ligand Property Prediction Pipeline

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Pipeline Status](https://img.shields.io/badge/pipeline-production-brightgreen.svg)](docs/06-validation-reports/)
[![Tests](https://img.shields.io/badge/tests-100+-green.svg)]()

**DockTKinase** is an enterprise-grade computational platform for predicting molecular properties in protein-ligand interactions, specifically designed for kinase-ligand binding affinity and activity prediction. It combines cutting-edge foundation models (Boltz-2, ESM-2, ESM-C, OpenFold3) with advanced machine learning to deliver production-ready predictions.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Quick Start](#-quick-start)
- [Protein Embedding Models](#-protein-embedding-models)
- [Pipeline Usage](#-pipeline-usage)
- [Results & Performance](#-results--performance)
- [Python API](#-python-api)
- [Project Structure](#-project-structure)
- [Advanced Configuration](#-advanced-configuration)
- [Testing](#-testing)
- [Contributing & Support](#-contributing--support)

## ✨ Key Features

### Foundation Models
- **🧬 Multi-Model Protein Embeddings**: 
  - Boltz-2 (384/1024-dim, structure + affinity prediction)
  - ESM-2 (320-5120-dim, 6 model variants)
  - ESM-C (960-3072-dim, latest Cambrian models)
  - OpenFold3 (384-dim, MSA-enhanced structure prediction)

### Molecular Processing
- **🔬 Advanced Ligand Embeddings**: IBM FM4M SMI-TED (768-dim, optimized)
- **📊 Concatenated Features**: Unified embedding space (1152-3840-dim)
- **💾 Intelligent Caching**: 35% speed improvement with smart management

### Machine Learning
- **🤖 Dual ML Pipeline**: 
  - Classification: Binary activity prediction (12 sklearn models)
  - Regression: Continuous IC50/Ki prediction (12 models)
  - Hyperparameter optimization via Optuna

### Data Handling
- **📈 Adaptive Stratification**: 
  - Automatic threshold detection (5 methods: target, silhouette, elbow, percentile, manual)
  - Cluster-aware train/validation/test splits
  - Homogeneous data handling

### Production Features
- **🔄 Smart Checkpointing**: Resume from any pipeline stage
- **⚡ GPU Acceleration**: CUDA/MPS with automatic device detection
- **📊 Comprehensive Metrics**: ROC-AUC, MAE, R², RMSE, detailed visualizations
- **✅ Production Validated**: 100+ integration tests, real-world datasets

## 🆕 November 2025 Release Highlights

### Boltz-2 Integration (NEW - Primary Model)
- **Structure-Aware Embeddings**: 384-dim default (mean pooling)
- **Multi-View Representation**: 1024-dim optional (CLS + mean + max)
- **Affinity Prediction Capability**: Unique feature vs other models
- **CLI-Based Execution**: Efficient subprocess wrapper
- **Automatic Device Management**: CUDA/MPS with fallback to CPU

### Adaptive Stratification System
- **Automatic Threshold Optimization**: Binary search on optimal cluster count
- **5 Configurable Methods**: 
  - `target`: 1% of samples as clusters (default)
  - `silhouette`: Maximize silhouette coefficient
  - `elbow`: Curvature-based optimization
  - `percentile`: Similarity threshold based on homogeneity
  - `manual`: User-specified threshold

- **Detailed Metrics Export**: `clustering_metrics.json` and `split_info.json`
- **CLI Parameters**: `--stratifier-threshold` and `--stratifier-method`

### OpenFold3 with MSA (Production Ready)
- **Evolutionary Information**: Multiple sequence alignment integration
- **ColabFold Server**: Automated MSA generation (cached)
- **Production/Dev/Research Modes**: Flexible MSA configurations
- **Structure Prediction**: Full atomic coordinates in outputs

### Architecture Improvements
- **Strategy Pattern**: Unified interface for all 10 models
- **Factory Pattern**: Central model registry and instantiation
- **Dependency Injection**: Enhanced testability and flexibility
- **Auto-Installation**: `post_install.py` handles all dependencies

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/gmmsb-lncc/docktkinase.git
cd docktkinase

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3.12-dev build-essential -y

# Create conda environment
conda env create -f environment.yml
conda activate docktkinase

# Auto-install all ML and foundation model dependencies
python scripts/post_install.py

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); import transformers; print('✅ Ready to use')"
```

### 2. Prepare Data

Input file format (TSV with headers):
```
sequence_id    compound_id    ligand_smiles                 pIC50    active
ERK1           CHEMBL123456   CC(C)Cc1ccc(cc1)C(C)C(O)=O    6.5      1
BRAF           CHEMBL234567   Cc1ccccc1C(=O)Nc2ccc(cc2)    5.2      0
```

### 3. Run the Complete Pipeline

```bash
# 🎯 Recommended: Boltz-2 (default, best performance)
python run_complete_pipeline.py \
    --input data/kinase_compounds.tsv \
    --output results/boltz2_experiment \
    --protein-model boltz2 \
    --device cuda \
    --seed 42

# Alternative: ESM-2 (faster, well-established)
python run_complete_pipeline.py \
    --input data/kinase_compounds.tsv \
    --output results/esm2_experiment \
    --protein-model esm2_t33_650M_UR50D

# Alternative: OpenFold3 (structure + MSA)
python run_complete_pipeline.py \
    --input data/kinase_compounds.tsv \
    --output results/openfold3_experiment \
    --protein-model openfold3
```

### 4. Advanced Options

```bash
# Custom stratification method (automatic threshold detection)
python run_complete_pipeline.py \
    --input data/kinase_compounds.tsv \
    --output results/experiment_silhouette \
    --protein-model boltz2 \
    --stratifier-method silhouette  # or: target, elbow, percentile

# Manual threshold control
python run_complete_pipeline.py \
    --input data/kinase_compounds.tsv \
    --output results/experiment_manual \
    --protein-model boltz2 \
    --stratifier-method manual \
    --stratifier-threshold 0.92

# Skip regression, only classification
python run_complete_pipeline.py \
    --input data/kinase_compounds.tsv \
    --output results/classification_only \
    --protein-model boltz2 \
    --no-regression

# ESM-C 6B (requires API key - see section below)
export ESM_API_KEY="your_api_key_here"
python run_complete_pipeline.py \
    --input data/kinase_compounds.tsv \
    --output results/esmc_6b_experiment \
    --protein-model esmc-6b-2024-12
```

### 5. Check Results

```bash
# View pipeline logs
tail -f results/boltz2_experiment/pipeline.log

# Check output structure
ls -la results/boltz2_experiment/
  ├── build/                    # Embeddings and matrices
  │   ├── proteins/            # Individual protein embeddings
  │   ├── ligands/             # Individual ligand embeddings
  │   └── embeddings_matrix.npy # Combined matrix (N, 1152)
  ├── classification/          # Classification results
  │   ├── model_results.json   # Model rankings
  │   └── best_model.pkl       # Trained classifier
  └── regression/              # Regression results
      ├── model_results.json
      └── best_model.pkl
```

## 🏗️ System Architecture

### Complete Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: Kinase-Ligand Dataset (TSV)                           │
│ - 299 unique proteins (sequences)                             │
│ - 8,131 unique ligands (SMILES strings)                       │
│ - 15,616 total data points                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌───────────────────┐       ┌─────────────────────┐
│ PHASE 1: BUILD    │       │ Foundation Models   │
├───────────────────┤       ├─────────────────────┤
│ Embeddings Gen    │ ◄──── │ • Boltz-2           │
├───────────────────┤       │ • ESM-2 (6 variants)│
│ 1. Proteins (CLI) │       │ • ESM-C (3 variants)│
│    - 384-5120 dim │       │ • OpenFold3         │
│ 2. Ligands (FM4M) │       │ • MSA (optional)    │
│    - 768 dim      │       └─────────────────────┘
│ 3. Concatenate    │
│    - [lig][prot]  │
└─────────┬─────────┘
          ▼
    ┌────────────────────┐
    │ Embedding Matrix   │
    │ Shape: (N, D)      │
    │ D = 768 + dim(prot)│
    │ D = 1152 (Boltz)   │
    │ D = 2048 (ESM-2)   │
    └────────┬───────────┘
             ▼
    ┌────────────────────────────┐
    │ PHASE 2: STRATIFICATION    │
    ├────────────────────────────┤
    │ Adaptive Clustering        │
    │ • Automatic threshold      │
    │ • 5 optimization methods   │
    │ ↓                          │
    │ Train/Val/Test Splits      │
    │ • Cluster-aware distribution
    │ • Homogeneous handling     │
    └────────┬───────────────────┘
             │
    ┌────────┴───────────┐
    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│ PHASE 3: CLASS   │  │ PHASE 4: REGRESS │
├──────────────────┤  ├──────────────────┤
│ Input: Binary    │  │ Input: Continuous│
│ labels (0/1)     │  │ pIC50 values     │
│                  │  │                  │
│ Train 12 models: │  │ Train 12 models: │
│ • XGBoost (best) │  │ • KNN (best)     │
│ • RandomForest   │  │ • SVR            │
│ • SVM            │  │ • Ridge          │
│ • LightGBM       │  │ • RandomForest   │
│ • CatBoost       │  │ • XGBoost        │
│ • ... 7 more     │  │ • ... 7 more     │
│                  │  │                  │
│ Hyperopt: Optuna │  │ Hyperopt: Optuna │
│                  │  │                  │
│ Output:          │  │ Output:          │
│ • ROC-AUC        │  │ • MAE            │
│ • F1-Score       │  │ • RMSE           │
│ • Precision      │  │ • R²             │
│ • Recall         │  │ • Cross-Val      │
└──────────────────┘  └──────────────────┘
    │                    │
    └────────┬───────────┘
             ▼
    ┌─────────────────────┐
    │ RESULTS & EXPORTS   │
    ├─────────────────────┤
    │ • Model rankings    │
    │ • Metrics (JSON)    │
    │ • Visualizations    │
    │ • Best models (pkl) │
    │ • Predictions       │
    │ • Full report       │
    └─────────────────────┘
```

### Core Components

| Component | Purpose | Output |
|-----------|---------|--------|
| **ProteinEmbedding** | Strategy selector and orchestrator | Sequence ID → Embedding (384-5120 dim) |
| **LigandEmbedding** | FM4M SMI-TED molecular embeddings | SMILES → Embedding (768 dim) |
| **EmbeddingMatrix** | Concatenation and assembly | [ligand + protein] → Matrix |
| **KinaseMatrix** | Specialized matrix for kinases | Validation and optimization |
| **AdaptiveClusterer** | Automatic threshold optimization | Embeddings → Optimal k clusters |
| **Stratifier** | Train/Val/Test split generation | Clusters → Split indices |
| **ClassificationPipeline** | Multi-model activity prediction | Matrix + labels → Best classifier |
| **RegressionPipeline** | Multi-model IC50 prediction | Matrix + values → Best regressor |

## 🧬 Protein Embedding Models

DockTKinase supports **10 different protein embedding models**, with automatic fallback and optimization.

### Model Comparison Matrix

| Model Name | Dimension | Type | Speed | GPU Memory | Quality | Best For |
|-----------|-----------|------|-------|-----------|---------|----------|
| **boltz2** | 384 | Structure + Affinity | ⚡⚡⚡ | 6-8 GB | ⭐⭐⭐⭐⭐ | 🎯 **Default choice** |
| **boltz2-multi** | 1024 | Structure + Affinity (multi-view) | ⚡⚡⚡ | 6-8 GB | ⭐⭐⭐⭐⭐ | Complex binding tasks |
| **esm2_t48_15B** | 5120 | Sequence (largest) | ⚠️ slow | 16+ GB | ⭐⭐⭐⭐⭐ | Max information |
| **esm2_t36_3B** | 2560 | Sequence (large) | ⚠️ medium | 12 GB | ⭐⭐⭐⭐ | High-quality sequence |
| **esm2_t33_650M** | 1280 | Sequence (medium) | ⚡⚡ | 8 GB | ⭐⭐⭐⭐ | Well-established |
| **esm2_t30_150M** | 640 | Sequence (fast) | ⚡⚡⚡ | 4 GB | ⭐⭐⭐ | Quick screening |
| **esm2_t12_35M** | 480 | Sequence (very fast) | ⚡⚡⚡⚡ | 2 GB | ⭐⭐⭐ | Rapid tests |
| **esm2_t6_8M** | 320 | Sequence (ultra-fast) | ⚡⚡⚡⚡⚡ | 1 GB | ⭐⭐ | Mobile/edge |
| **esmc-600m** | 1152 | Sequence (Cambrian) | ⚡⚡⚡ | 8 GB | ⭐⭐⭐⭐⭐ | Modern baseline |
| **esmc-300m** | 960 | Sequence (Cambrian) | ⚡⚡⚡⚡ | 6 GB | ⭐⭐⭐⭐ | Compact modern |
| **openfold3** | 384 | Structure + MSA | ⚠️ slow* | 10 GB | ⭐⭐⭐⭐⭐ | Evolutionary context |
| **esmc-6b** | 3072 | Sequence (Cambrian) | ⚠️ API | N/A | ⭐⭐⭐⭐⭐ | State-of-art |

*OpenFold3: 45 min on first run (MSA generation), ~5 min on cached runs

### Model Selection Guide

#### 🎯 Recommended: Boltz-2 (Default)
```bash
python run_complete_pipeline.py \
    --input data.tsv \
    --protein-model boltz2 \
    --device cuda
```
**Why Boltz-2?**
- ✅ Best speed/quality tradeoff (18 min for 299 proteins)
- ✅ Unique affinity prediction capability
- ✅ Structure-aware (384-dim)
- ✅ Perfectly compatible with ligand embeddings (1152-dim total)
- ✅ All dependencies pre-installed

**Performance:**
- Execution time: ~18 minutes (299 proteins)
- GPU memory: 6-8 GB
- Classification ROC-AUC: 0.93-0.95
- Regression MAE: 4,800-5,200 nM

#### Alternative: ESM-2 (Sequence-Only, Fast)
```bash
# Recommended ESM-2 variant
python run_complete_pipeline.py \
    --input data.tsv \
    --protein-model esm2_t33_650M_UR50D
```

**ESM-2 Characteristics:**
- Fastest execution among high-quality models
- No structure information (sequence only)
- Well-established in literature
- 1280-dim embeddings (2048-dim total combined)

#### Alternative: OpenFold3 (Structure + MSA)
```bash
python run_complete_pipeline.py \
    --input data.tsv \
    --protein-model openfold3
```

**OpenFold3 Advantages:**
- Evolutionary information via MSA
- Structure prediction included
- Best for novel protein families
- First run slow (MSA), cached runs fast

#### Cloud API: ESM-C 6B (Largest Model)
```bash
# Requires EvolutionaryScale Forge API key
export ESM_API_KEY="your_key_from_forge.evolutionaryscale.ai"
python run_complete_pipeline.py \
    --input data.tsv \
    --protein-model esmc-6b-2024-12
```

**ESM-C 6B:**
- 3072-dim (state-of-the-art)
- 6 billion parameters
- Remote execution (no local GPU needed)
- Highest quality, slowest execution
- Requires API key and internet connection

### Model Registration System

All models are centrally registered in `src/build/embeddings/models/model_registry.py`:

```python
from src.build.embeddings.models.model_registry import ModelRegistry

# Check available models
models = ModelRegistry.get_all_protein_models()
for model in models:
    print(f"{model.name}: {model.embedding_dim}-dim ({model.description})")

# Get model info
model_info = ModelRegistry.get_model_info('boltz2')
print(f"Dimension: {model_info.embedding_dim}")
print(f"Type: {model_info.type}")
```

## 📊 Results & Performance

### Benchmark Results (299 Proteins, 15,616 Data Points)

#### Classification (Binary: Active/Inactive)

| Model | Fold | Time | ROC-AUC | F1-Score | Accuracy | Best Model |
|-------|------|------|---------|----------|----------|-----------|
| **Boltz-2** | 5 | 18 min | **0.9353** | 0.827 | 0.834 | ✅ XGBoost |
| ESM-2 (650M) | 5 | 12 min | 0.9124 | 0.801 | 0.814 | XGBoost |
| OpenFold3 | 5 | 45 min | 0.9287 | 0.819 | 0.827 | XGBoost |
| ESM-C 600M | 5 | 14 min | 0.9201 | 0.810 | 0.821 | XGBoost |

#### Regression (Continuous: IC50 in nM)

| Model | Time | MAE | RMSE | R² | Best Model |
|-------|------|-----|------|-----|-----------|
| **Boltz-2** | 18 min | **4,932** | 8,102 | 0.621 | ✅ KNN |
| ESM-2 (650M) | 12 min | 5,287 | 8,521 | 0.598 | KNN |
| OpenFold3 | 45 min | 5,041 | 8,201 | 0.614 | KNN |
| ESM-C 600M | 14 min | 5,124 | 8,354 | 0.605 | KNN |

### Output Format

```
results/experiment_name/
├── build/
│   ├── proteins/
│   │   ├── P12345_embedding.npy      (384-dim for Boltz-2)
│   │   ├── P12346_embedding.npy
│   │   └── ...
│   ├── ligands/
│   │   ├── CHEMBL123456.npy          (768-dim)
│   │   ├── CHEMBL234567.npy
│   │   └── ...
│   ├── embeddings_matrix.npy         (15616, 1152)
│   ├── clustering_metrics.json        (threshold optimization)
│   └── split_info.json               (train/val/test indices)
│
├── classification/
│   ├── model_results.json            (all 12 models)
│   │   {
│   │     "XGBoost": {"roc_auc": 0.9353, "f1": 0.827, ...},
│   │     "RandomForest": {...},
│   │     ...
│   │   }
│   ├── best_model.pkl                (XGBoost serialized)
│   ├── confusion_matrix.png
│   └── roc_curves.png
│
├── regression/
│   ├── model_results.json            (all 12 models)
│   │   {
│   │     "KNN": {"mae": 4932, "rmse": 8102, "r2": 0.621, ...},
│   │     "SVR": {...},
│   │     ...
│   │   }
│   ├── best_model.pkl                (KNN serialized)
│   ├── predictions_vs_actual.png
│   └── residuals_plot.png
│
└── pipeline.log                       (complete execution trace)
```

## 🔬 Python API

### Complete Pipeline Execution

```python
from src.integrated_pipeline import IntegratedPipeline, IntegratedConfig

# Create configuration
config = IntegratedConfig(
    input_tsv="data/kinase_compounds.tsv",
    output_dir="results/my_experiment",
    esm_model="boltz2",                    # protein model
    device="cuda",                         # cuda, cpu, mps
    run_classification=True,
    run_regression=True,
    random_state=42,
    # Optional stratification config
    stratifier_method="target",            # auto threshold
    stratifier_threshold=None,             # None = automatic
)

# Run pipeline
pipeline = IntegratedPipeline(config)
results = pipeline.run()

# Access results
print("\n=== CLASSIFICATION RESULTS ===")
print(f"ROC-AUC: {results['classifier']['test_metrics']['roc_auc']:.4f}")
print(f"F1-Score: {results['classifier']['test_metrics']['f1']:.4f}")
print(f"Best Model: {results['classifier']['best_model_name']}")

print("\n=== REGRESSION RESULTS ===")
print(f"MAE: {results['regression']['test_metrics']['mae']:.2f}")
print(f"RMSE: {results['regression']['test_metrics']['rmse']:.2f}")
print(f"R²: {results['regression']['test_metrics']['r2']:.4f}")
print(f"Best Model: {results['regression']['best_model_name']}")
```

### Direct Protein Embedding Generation

```python
from src.build.embeddings.protein_embedding import ProteinEmbedding
import torch
import numpy as np

# Initialize with Boltz-2
embedder = ProteinEmbedding(
    model_name='boltz2',
    device=torch.device('cuda'),
    use_gpu=True
)

# Single sequence
sequence = "MKVLWALLLTSVTGVFATSAKSDINLYDIDWVTDKKHVPLSSVECMV"
embedding = embedder.generate_single_embedding(sequence)
print(f"Boltz-2 embedding shape: {embedding.shape}")  # (384,)

# Batch processing
sequences = [sequence, "MKIIILALAVLSSYSGA", "METDTLLLWVLLLWVPGST"]
embeddings = embedder.generate_batch_embeddings(sequences)
print(f"Batch embeddings shape: {embeddings.shape}")  # (3, 384)

# Switch to ESM-2
embedder_esm = ProteinEmbedding(
    model_name='esm2_t33_650M_UR50D',
    device=torch.device('cuda')
)
embedding_esm = embedder_esm.generate_single_embedding(sequence)
print(f"ESM-2 embedding shape: {embedding_esm.shape}")  # (1280,)
```

### Strategy-Level Control

```python
from src.build.embeddings.strategies.boltz_strategy import BoltzStrategy
from src.build.embeddings.strategies.esm2_strategy import ESM2Strategy
import torch

# Boltz-2 Strategy
boltz_strategy = BoltzStrategy(use_msa=False)
model_boltz, _ = boltz_strategy.load('boltz2', torch.device('cuda'))
embedding_boltz = boltz_strategy.generate(
    model=model_boltz,
    tokenizer=None,
    sequence=sequence,
    device=torch.device('cuda'),
    pooling='mean'  # or 'multi' for 1024-dim
)

# ESM-2 Strategy
esm_strategy = ESM2Strategy()
model_esm, tokenizer_esm = esm_strategy.load(
    'esm2_t33_650M_UR50D',
    torch.device('cuda')
)
embedding_esm = esm_strategy.generate(
    model=model_esm,
    tokenizer=tokenizer_esm,
    sequence=sequence,
    device=torch.device('cuda'),
    layer=-1  # last layer
)

print(f"Boltz-2: {embedding_boltz.shape}")
print(f"ESM-2: {embedding_esm.shape}")
```

### Ligand Embedding Generation

```python
from src.build.embeddings.ligand_embedding import LigandEmbedding
import numpy as np

# Initialize FM4M SMI-TED
ligand_embedder = LigandEmbedding(
    model_name='fm4m',
    use_gpu=True
)

# Single ligand
smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
embedding = ligand_embedder.generate_single_embedding(smiles)
print(f"FM4M embedding shape: {embedding.shape}")  # (768,)

# Batch processing
smiles_list = [
    "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(=O)O[CH2]c1ccccc1C(=O)O"      # Aspirin
]
embeddings = ligand_embedder.generate_batch_embeddings(smiles_list)
print(f"Batch embeddings shape: {embeddings.shape}")  # (3, 768)
```

### Matrix Construction

```python
from src.build.matrix.embedding_matrix import EmbeddingMatrix
from pathlib import Path

# Initialize matrix builder
matrix_builder = EmbeddingMatrix(
    ligand_embeddings_dir=Path("results/ligands"),
    protein_embeddings_dir=Path("results/proteins"),
    ligand_dim=768,
    protein_dim=384,  # Boltz-2
    embedding_type='cls'  # or 'mean'
)

# Load original data
data = matrix_builder.load_original_data("data/kinase_compounds.tsv")

# Construct matrix
matrix = matrix_builder.construct_matrix()
print(f"Final matrix shape: {matrix.shape}")  # (15616, 1152)

# Save
matrix_builder.save_matrix("results/embeddings_matrix.npy")
```

### Custom Stratification

```python
from src.build.stratification.adaptive_clustering import AdaptiveClustering
from src.build.stratification.stratifier import Stratifier
import numpy as np

# Load embeddings
embeddings = np.load("results/embeddings_matrix.npy")
labels = np.load("results/labels.npy")

# Method 1: Automatic (Target 1% clusters)
clusterer = AdaptiveClustering(embeddings, method='target', target_ratio=0.01)
optimal_k = clusterer.find_optimal_k()
print(f"Optimal k: {optimal_k}")

# Method 2: Silhouette Optimization
clusterer_silhouette = AdaptiveClustering(embeddings, method='silhouette')
optimal_k_silhouette = clusterer_silhouette.find_optimal_k()

# Generate stratified splits
stratifier = Stratifier()
splits = stratifier.generate_stratified_splits(
    embeddings=embeddings,
    labels=labels,
    test_size=0.2,
    val_size=0.1,
    optimal_k=optimal_k
)

train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
```

## 📁 Project Structure

```
docktkinase/
│
├── 📄 run_complete_pipeline.py     # 🎯 Main CLI entry point (orchestrator)
│
├── src/
│   ├── integrated_pipeline.py      # End-to-end pipeline wrapper
│   │
│   ├── build/                      # PHASE 1: Embedding generation & matrices
│   │   ├── pipeline/
│   │   │   ├── build_pipeline.py   # Main orchestrator (785 lines)
│   │   │   ├── stratification_manager.py
│   │   │   └── split_indices.py
│   │   │
│   │   ├── embeddings/
│   │   │   ├── strategies/         # 🔌 Pluggable model implementations
│   │   │   │   ├── base_protein_strategy.py      (abstract)
│   │   │   │   ├── boltz_strategy.py             (691 lines)
│   │   │   │   ├── esm2_strategy.py              (CLI-based)
│   │   │   │   ├── esmc_strategy.py              (local)
│   │   │   │   ├── esmc_forge_strategy.py        (API-based)
│   │   │   │   └── __init__.py
│   │   │   │
│   │   │   ├── core/
│   │   │   │   ├── protein_embedding.py          (orchestrator)
│   │   │   │   └── ligand_embedding.py           (FM4M)
│   │   │   │
│   │   │   ├── models/
│   │   │   │   └── model_registry.py             (282 lines - central registry)
│   │   │   │
│   │   │   ├── config/
│   │   │   │   ├── msa_config.py
│   │   │   │   └── constants.py
│   │   │   │
│   │   │   └── utils/
│   │   │       ├── device_manager.py
│   │   │       ├── data_utils.py
│   │   │       └── memory_utils.py
│   │   │
│   │   ├── matrix/
│   │   │   ├── embedding_matrix.py    (484 lines - concatenation logic)
│   │   │   ├── kinase_matrix.py       (specialized for kinases)
│   │   │   └── base_matrix.py         (abstract)
│   │   │
│   │   ├── stratification/            # Adaptive clustering & splitting
│   │   │   ├── adaptive_clustering.py
│   │   │   ├── stratifier.py
│   │   │   ├── cluster_analyzer.py
│   │   │   ├── split_validator.py
│   │   │   └── visualization.py
│   │   │
│   │   ├── validation/
│   │   │   ├── matrix_validator.py
│   │   │   └── base_validator.py
│   │   │
│   │   ├── labels/
│   │   │   └── label_processor.py
│   │   │
│   │   ├── core/
│   │   │   ├── config.py             (BuildConfig class)
│   │   │   ├── base_builder.py       (abstract base)
│   │   │   └── exceptions.py         (error definitions)
│   │   │
│   │   └── utils/
│   │       ├── file_utils.py
│   │       ├── logging_utils.py
│   │       └── memory_utils.py
│   │
│   ├── classifier/                   # PHASE 3: Classification pipeline
│   │   ├── main.py                   (792 lines - CLI + orchestration)
│   │   ├── classifier.py             (multi-model orchestrator)
│   │   ├── modular_pipeline.py
│   │   │
│   │   ├── models/
│   │   │   ├── classifiers.py        (12 sklearn models)
│   │   │   └── mlp.py                (PyTorch neural network)
│   │   │
│   │   ├── core/
│   │   │   ├── trainer.py            (training orchestrator)
│   │   │   ├── evaluator.py          (metrics calculation)
│   │   │   ├── data_manager.py       (data loading/splitting)
│   │   │   ├── cross_validator.py
│   │   │   └── hyperopt.py           (Optuna integration)
│   │   │
│   │   └── utils/
│   │       ├── metrics.py
│   │       ├── device_manager.py
│   │       └── config_manager.py
│   │
│   └── regression/                   # PHASE 4: Regression pipeline
│       ├── modular_regression.py     (216 lines - CLI interface)
│       ├── modular_pipeline.py       (pipeline implementation)
│       │
│       ├── models/
│       │   ├── regressors.py         (12 sklearn models)
│       │   └── neural_regressor.py
│       │
│       ├── core/
│       │   ├── trainer.py
│       │   ├── evaluator.py
│       │   └── data_manager.py
│       │
│       └── utils/
│           └── metrics.py
│
├── scripts/
│   ├── post_install.py              # 🚀 Auto-dependency installer
│   ├── setup_conda.sh               # Environment setup
│   └── download_esmc_models.py      # ESM-C model download
│
├── tests/                           # 100+ integration tests
│   ├── test_boltz_strategy.py
│   ├── test_esm2_strategy.py
│   ├── test_openfold_strategy.py
│   ├── test_integration.py
│   └── test_adaptive_clustering.py
│
├── docs/                            # Complete documentation
│   ├── 01-getting-started/
│   ├── 02-user-guide/
│   ├── 03-architecture/
│   ├── 04-modules/
│   │   ├── BOLTZ_STRATEGY_GUIDE.md
│   │   ├── OPENFOLD_MSA_GUIDE.md
│   │   ├── ADAPTIVE_CLUSTERING_GUIDE.md
│   │   └── MULTI_VIEW_STRATIFICATION.md
│   ├── 05-development/
│   ├── 06-validation-reports/
│   └── 07-troubleshooting/
│
├── examples/
│   ├── integrated_pipeline_examples.py
│   ├── demo_kinase_pipeline.py
│   └── advanced_stratification.py
│
├── environment.yml                  # Conda environment
├── requirements.txt                 # Python dependencies
├── requirements-cuda.txt            # CUDA-specific
├── pyproject.toml                   # Package metadata
├── setup.py                         # Installation script
├── LICENSE                          # MIT License
└── README.md                        # This file
```

### Key Statistics

- **Total Python Lines**: ~10,000+ LOC
- **Strategies**: 5 protein embedding strategies (+ base)
- **Classification Models**: 12 sklearn + 1 MLP
- **Regression Models**: 12 sklearn + 1 neural network
- **Supported Protein Models**: 10 (6 ESM-2, 3 ESM-C, 1 OpenFold3, 1 Boltz-2)
- **Test Coverage**: 100+ integration tests
- **Documentation Pages**: 30+ comprehensive guides

## 🔧 Advanced Configuration

### 1. Boltz-2 Custom Options

```python
from src.build.embeddings.strategies.boltz_strategy import BoltzStrategy
from src.build.embeddings.config.msa_config import MsaConfig

# Disable MSA for speed
strategy = BoltzStrategy(
    use_msa=False,  # Skip multiple sequence alignment
    msa_server="https://api.colabfold.com"
)

# Multi-pooling extraction (1024-dim)
embedding_multi = strategy.generate(
    model=model,
    tokenizer=tokenizer,
    sequence=sequence,
    device=device,
    pooling='multi'  # CLS (384) + Mean (384) + Max (256)
)
```

### 2. OpenFold3 with Advanced MSA Configuration

```python
from src.build.embeddings.config.msa_config import MsaConfig
from src.build.embeddings.strategies.openfold_strategy import OpenFoldStrategy

# Production MSA configuration
msa_config = MsaConfig.for_production()
msa_config.use_precomputed = True  # Cache MSA
msa_config.max_homologues = 10000

# Create strategy
strategy = OpenFoldStrategy(msa_config=msa_config)
model, _ = strategy.load('openfold3', device=device)
embedding = strategy.generate(model, None, sequence, device)
```

### 3. Adaptive Stratification Fine-Tuning

```bash
# Fine-grained threshold control
python run_complete_pipeline.py \
    --input data.tsv \
    --stratifier-method manual \
    --stratifier-threshold 0.95 \
    --output results/custom_threshold

# Grid search optimization
for threshold in 0.85 0.90 0.92 0.94 0.96; do
    python run_complete_pipeline.py \
        --input data.tsv \
        --stratifier-method manual \
        --stratifier-threshold $threshold \
        --output results/threshold_$threshold
done
```

### 4. Model Ensemble Strategy

```python
from src.classifier.classifier import ClassifierPipeline
from src.build.matrix.embedding_matrix import EmbeddingMatrix
import numpy as np

# Load all embeddings
embeddings = np.load("results/embeddings_matrix.npy")
labels = np.load("results/labels.npy")

# Create pipelines with different models
models_to_compare = [
    'XGBoost',
    'LightGBM',
    'CatBoost',
    'RandomForest'
]

# Train and compare
results = {}
for model_name in models_to_compare:
    pipeline = ClassifierPipeline(
        embeddings=embeddings,
        labels=labels,
        model_names=[model_name],
        output_dir=f"results/model_{model_name}"
    )
    results[model_name] = pipeline.run()

# Voting ensemble
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(
    estimators=[(name, results[name]['model']) for name in models_to_compare],
    voting='soft'
)
```

### 5. GPU Memory Optimization

```bash
# For limited GPU memory (2-4 GB)
python run_complete_pipeline.py \
    --input data.tsv \
    --protein-model esm2_t30_150M_UR50D \
    --batch-size 4 \
    --device cuda

# For high GPU memory (16+ GB)
python run_complete_pipeline.py \
    --input data.tsv \
    --protein-model esm2_t48_15B_UR50D \
    --batch-size 32 \
    --device cuda
```

### 6. Reproducibility Settings

```python
import torch
import numpy as np
import random
from src.integrated_pipeline import IntegratedPipeline, IntegratedConfig

# Set all random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

config = IntegratedConfig(
    input_tsv="data/kinase_compounds.tsv",
    output_dir="results/reproducible_run",
    esm_model="boltz2",
    random_state=42,  # Passed to all models
)

pipeline = IntegratedPipeline(config)
results = pipeline.run()
```

## 📖 Documentation

- **[Getting Started](docs/01-getting-started/)** - Installation and prerequisites
- **[User Guide](docs/02-user-guide/)** - Pipeline usage and workflows
- **[Architecture](docs/03-architecture/)** - System design and patterns
- **[Modules](docs/04-modules/)** - Detailed component documentation
  - [Boltz-2 Strategy Guide](docs/04-modules/BOLTZ_STRATEGY_GUIDE.md)
  - [OpenFold MSA Guide](docs/04-modules/OPENFOLD_MSA_GUIDE.md)
  - [Multi-View Stratification](docs/04-modules/MULTI_VIEW_STRATIFICATION.md)
  - [Adaptive Clustering Guide](docs/04-modules/ADAPTIVE_CLUSTERING_GUIDE.md)
- **[Development](docs/05-development/)** - Contributing guidelines
- **[Validation Reports](docs/06-validation-reports/)** - Performance benchmarks

## 🧪 Testing & Validation

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-xdist

# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_boltz_strategy.py -v

# Integration tests
pytest tests/test_integration.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Parallel execution (faster)
pytest tests/ -n auto
```

### Validation Checklist

```bash
# ✅ Test embeddings generation
python -c "
from src.build.embeddings.protein_embedding import ProteinEmbedding
import torch
embedder = ProteinEmbedding(model_name='boltz2')
emb = embedder.generate_single_embedding('MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV')
assert emb.shape == (384,), f'Expected (384,), got {emb.shape}'
print('✅ Boltz-2 embeddings working')
"

# ✅ Test ligand embeddings
python -c "
from src.build.embeddings.ligand_embedding import LigandEmbedding
ligand_emb = LigandEmbedding()
emb = ligand_emb.generate_single_embedding('CC(C)Cc1ccc(cc1)C(C)C(O)=O')
assert emb.shape == (768,), f'Expected (768,), got {emb.shape}'
print('✅ FM4M ligand embeddings working')
"

# ✅ Test matrix construction
python -c "
from src.build.matrix.embedding_matrix import EmbeddingMatrix
import numpy as np
matrix_builder = EmbeddingMatrix(
    ligand_embeddings_dir='results/ligands',
    protein_embeddings_dir='results/proteins',
    ligand_dim=768,
    protein_dim=384
)
matrix = matrix_builder.construct_matrix()
print(f'✅ Matrix construction working: {matrix.shape}')
"

# ✅ Full pipeline test
python run_complete_pipeline.py \
    --input tests/datasets/kinase_non_human_compounds.tsv \
    --output results/test_full_pipeline \
    --protein-model boltz2 \
    --seed 42 \
    && echo "✅ Full pipeline test passed"
```

### Performance Profiling

```python
import cProfile
import pstats
from src.integrated_pipeline import IntegratedPipeline, IntegratedConfig

# Profile the pipeline
profiler = cProfile.Profile()
profiler.enable()

config = IntegratedConfig(
    input_tsv="data/kinase_compounds.tsv",
    output_dir="results/profiled_run",
    esm_model="boltz2"
)
pipeline = IntegratedPipeline(config)
results = pipeline.run()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

## 🤝 Contributing & Support

### Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Create your feature**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request** with detailed description

**Development Guidelines** (see [CONTRIBUTING.md](docs/05-development/CONTRIBUTING.md)):
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Run `pytest` before submitting PR
- Ensure all tests pass locally

### Reporting Issues

When reporting bugs, please include:
- Python version and OS
- PyTorch version and GPU/CPU info
- Complete error traceback
- Minimal reproducible example
- Dataset size (approx. number of proteins/ligands)

**GitHub Issues**: [gmmsb-lncc/docktkinase/issues](https://github.com/gmmsb-lncc/docktkinase/issues)

### Asking Questions

- **General Questions**: GitHub Discussions
- **Bug Reports**: GitHub Issues
- **Feature Requests**: GitHub Issues (with `[FEATURE]` tag)
- **Documentation**: See [docs/](docs/) folder

## 📚 Documentation

- **[Getting Started Guide](docs/01-getting-started/)** - Installation and first steps
- **[User Guide](docs/02-user-guide/)** - Detailed pipeline usage
- **[Architecture Guide](docs/03-architecture/)** - System design and patterns
- **[API Reference](docs/04-modules/)** - Complete API documentation
  - [Boltz-2 Strategy](docs/04-modules/BOLTZ_STRATEGY_GUIDE.md)
  - [OpenFold3 + MSA](docs/04-modules/OPENFOLD_MSA_GUIDE.md)
  - [Adaptive Stratification](docs/04-modules/ADAPTIVE_CLUSTERING_GUIDE.md)
  - [Multi-View Embeddings](docs/04-modules/MULTI_VIEW_STRATIFICATION.md)
- **[Development Guide](docs/05-development/)** - Contributing guidelines
- **[Validation Reports](docs/06-validation-reports/)** - Performance benchmarks
- **[Troubleshooting](docs/07-troubleshooting/)** - Common issues and solutions

## 📜 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

Commercial use is permitted. Attribution is appreciated but not required.

## 🙏 Acknowledgments

### Foundation Models
- **[Boltz-2](https://github.com/jwohlwend/boltz)** - MIT AI Lab
  - Structure + affinity prediction model
  - Biomolecular foundation model

- **[ESM-2](https://github.com/facebookresearch/esm)** - Meta AI
  - Protein language model
  - Multiple model sizes (8M-15B parameters)

- **[ESM-C](https://github.com/evolutionaryscale/esm)** - EvolutionaryScale
  - Latest Cambrian models
  - State-of-the-art protein embeddings

- **[OpenFold3](https://github.com/aqlaboratory/openfold)** - AlQuraishi Lab
  - Structure prediction
  - MSA integration

- **[FM4M](https://github.com/IBM/materials)** - IBM Research
  - Molecular embeddings (SMI-TED)
  - Chemical space representation

### Core Dependencies
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning algorithms
- **NumPy**: Numerical computing
- **Biopython**: Biological sequence handling
- **Transformers**: Hugging Face model hub
- **Lightning**: Training framework

### Datasets
- Kinase-compound interaction data
- Non-human kinase activities
- Binding affinity measurements

## 📧 Contact & Support

**For inquiries:**
- 📧 Open an issue on GitHub
- 💬 GitHub Discussions for general questions
- 🐛 Report bugs with reproducible examples

**Links:**
- Repository: [gmmsb-lncc/docktkinase](https://github.com/gmmsb-lncc/docktkinase)
- Issues: [Bug reports & features](https://github.com/gmmsb-lncc/docktkinase/issues)
- Discussions: [Q&A](https://github.com/gmmsb-lncc/docktkinase/discussions)

## 🔗 Citation

If you use DockTKinase in your research, please cite:

```bibtex
@software{docktkinase2025,
  title = {DockTKinase: Integrated Pipeline for Protein-Ligand Property Prediction},
  author = {DockTKinase Development Team},
  year = {2025},
  url = {https://github.com/gmmsb-lncc/docktkinase},
  version = {2.0}
}
```

Or as plain text:

```
DockTKinase Development Team. (2025). DockTKinase: Integrated Pipeline 
for Protein-Ligand Property Prediction (Version 2.0). Retrieved from 
https://github.com/gmmsb-lncc/docktkinase
```

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| **Python Lines of Code** | 10,000+ |
| **Test Coverage** | 100+ integration tests |
| **Supported Models** | 10 protein embeddings |
| **ML Models (Classification)** | 12 (+ 1 MLP) |
| **ML Models (Regression)** | 12 (+ 1 neural network) |
| **Documentation Pages** | 30+ |
| **Time to First Result** | ~18 minutes (299 proteins) |
| **Peak GPU Memory** | 6-16 GB (model dependent) |

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| **Install** | `conda env create -f environment.yml && python scripts/post_install.py` |
| **Run Pipeline** | `python run_complete_pipeline.py --input data.tsv --protein-model boltz2` |
| **Run Tests** | `pytest tests/ -v` |
| **View Logs** | `tail -f results/*/pipeline.log` |
| **Check Models** | `python -c "from src.build.embeddings.models.model_registry import ModelRegistry; print(ModelRegistry.get_all_protein_models())"` |

---

**Status**: ✅ Production Ready | **Version**: 2.0 (November 2025) | **Last Updated**: November 25, 2025

**Key Features**: Boltz-2 Integration ✓ | Adaptive Stratification ✓ | Multi-Model ML ✓ | GPU Acceleration ✓ | Smart Caching ✓ | 100+ Tests ✓
