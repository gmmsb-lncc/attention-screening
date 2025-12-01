# 🧬 DockTKinase Source Code

**Modular architecture for comprehensive kinase inhibitor analysis and machine learning classification.**

This directory contains the core source code of DockTKinase, organized into three main modules that work together to provide a complete drug discovery pipeline for non-human kinases.

---

## 📁 **Module Overview**

### 🏗️ [`build/`](build/) - Pipeline Architecture & Embedding Generation
**Complete modular system for molecular embedding generation and data pipeline orchestration.**

- **Purpose**: Generate high-quality molecular embeddings for kinase inhibitors and proteins
- **Key Features**:
  - Ligand embeddings using IBM's FM4M SMI-TED model
  - Protein embeddings using Meta's ESM model  
  - Scalable Apache Spark processing
  - Checkpoint system for resumable operations
  - Matrix concatenation and validation
- **Architecture**: Fully modularized with core/, embeddings/, pipeline/, utils/, validation/
- **Documentation**: [Build Module README](build/README.md)

### 🤖 [`classifier/`](classifier/) - Machine Learning Classification
**Advanced ML pipeline for kinase inhibitor activity prediction with automated optimization.**

- **Purpose**: Train and deploy ML models for activity classification
- **Key Features**:
  - MLP neural networks with automated hyperparameter optimization
  - Cross-validation with statistical significance testing
  - ROC-AUC: 0.85 ± 0.01 performance
  - Optuna-based hyperparameter tuning
  - Model persistence and deployment utilities
- **Architecture**: Modular design with core/, models/, config/, utils/
- **Documentation**: [Classifier Module README](classifier/README.md)

### 📈 [`regression/`](regression/) - Machine Learning Regression **NEW**
**Professional regression pipeline for quantitative activity prediction with production-ready infrastructure and modular architecture.**

- **Purpose**: Predict continuous activity values (Ki, Kd, IC50 in nM)
- **🏗️ Architecture** ⭐ **NOVO** (Nov 2025):
  - **Modular structure**: `core/` (DataManager, Trainer, Evaluator) | `models/` (ML models) | `utils/` (MetricsCalculator)
  - **Same pattern** as classifier for consistency
  - **Two interfaces**: Traditional pipeline OR standalone modular API/CLI
- **Key Features**:
  - **11 Regression Models**: RandomForest, XGBoost, LightGBM, CatBoost, Ridge, Lasso, ElasticNet, SVR, KNN, MLP, GradientBoosting
  - **Target Prioritization**: Ki > Kd > IC50 (configurable)
  - **Stratified Split**: Quantile-based stratification for regression ⭐ NOVO
  - **15+ Metrics**: MAE, RMSE, R², MAPE, percentiles, CV-RMSE ⭐ NOVO
  - **Robust Validation**: 10+ automatic checks (NaN, Inf, outliers, variance)
  - **Professional Logging**: Colored console output with file logging
  - **Centralized Configuration**: JSON-serializable configs with profiles
  - Model comparison and automatic best model selection
- **Quality**: 100% tested with realistic tests, 45 bugs fixed, production-ready
- **Documentation**: 
  - [Modular Architecture Guide](regression/README_MODULAR.md) ⭐ **NOVO**
  - [Regression Improvements README](regression/README_IMPROVEMENTS.md)

### 🔧 [`utils/`](utils/) - Shared Utilities **NEW**
**Centralized utilities following DRY (Don't Repeat Yourself) principle.**

- **Purpose**: Provide shared functions used across multiple modules
- **Key Features**:
  - `safe_get()`: Safe dictionary access with NaN/None handling
  - `safe_get_numeric()`: Safe numeric extraction with type conversion
  - `safe_get_int()`: Safe integer extraction
  - `safe_get_str()`: Safe string extraction
  - Complete type hints and comprehensive docstrings
- **Architecture**: Single source of truth for common operations
- **Benefits**: Eliminates code duplication, easier maintenance, consistent behavior

### 🧬 [`database/`](database/) - Molecular Data Analysis
**Professional modular system for molecular database analysis, processing, and visualization.**

- **Purpose**: Analyze and process molecular databases for kinase research
- **Key Features**:
  - FAISS-based molecular clustering for scalable stratification ⭐ **UPDATED**
  - Statistical analysis of human vs non-human kinases
  - Class balance analysis and FAISS K-means stratification
  - SMILES standardization and cleaning
  - Comprehensive molecular descriptors calculation
- **Architecture**: Organized into core/, processing/, analysis/, sql/
- **Documentation**: [Database Module README](database/README.md)

---

## 🔄 **Module Integration Workflow**

```mermaid
graph TB
    A[Raw Data] --> B[database/]
    B --> C[build/]
    C --> D1[classifier/]
    C --> D2[regression/]
    D1 --> E1[Classification Model]
    D2 --> E2[Regression Model]
    
    U[utils/] -.-> |Shared Functions| B
    U -.-> |Shared Functions| D1
    U -.-> |Shared Functions| D2
    
    B -.-> |Analysis & Processing| B1[Molecular Clustering]
    B -.-> |Quality Control| B2[Data Validation]
    
    C -.-> |Embeddings| C1[Ligand Embeddings]
    C -.-> |Embeddings| C2[Protein Embeddings] 
    C -.-> |Pipeline| C3[Matrix Generation]
    
    D1 -.-> |ML Pipeline| D1A[Binary Classification]
    D1 -.-> |Optimization| D1B[Hyperparameter Tuning]
    D1 -.-> |Validation| D1C[Cross-Validation]
    
    D2 -.-> |ML Pipeline| D2A[Quantitative Prediction]
    D2 -.-> |Infrastructure| D2B[Validation/Logging/Config]
    D2 -.-> |Evaluation| D2C[Model Comparison]
```

### **Processing Flow:**

1. **Data Preparation** (`database/`)
   - Load and validate molecular data
   - Perform quality control and standardization
   - Analyze class balance and molecular diversity

2. **Feature Generation** (`build/`)
   - Generate ligand embeddings from SMILES
   - Generate protein embeddings from sequences
   - Create concatenated feature matrices

3. **Model Training - Classification** (`classifier/`)
   - Train MLP classifiers with optimized hyperparameters
   - Perform cross-validation and statistical testing
   - Deploy trained models for binary prediction (active/inactive)

4. **Model Training - Regression** (`regression/`) **NEW**
   - Train 11 regression models for quantitative prediction
   - Robust validation with 10+ automatic checks
   - Professional logging and centralized configuration
   - Predict continuous activity values (Ki, Kd, IC50 in nM)

5. **Shared Utilities** (`utils/`) **NEW**
   - Common functions used across all modules
   - Safe data access and type conversion
   - Eliminates code duplication (DRY principle)

---

## 🚀 **Quick Start**

### **End-to-End Pipeline - Classification**
```python
# Complete classification pipeline
from src.build import BuildPipeline
from src.classifier.modular_pipeline import MLPEmbeddingPipeline
from src.database import ComparativeAnalyzer
from src.utils import safe_get  # NEW: Shared utilities

# 1. Analyze and prepare data
analyzer = ComparativeAnalyzer()
data_stats = analyzer.compare_datasets("data/kinase_data.tsv")

# 2. Generate embeddings
pipeline = BuildPipeline()
embeddings = pipeline.run_complete_pipeline(
    input_file="data/kinase_data.tsv",
    output_dir="embeddings/"
)

# 3. Train classifier
classifier = MLPEmbeddingPipeline()
model = classifier.train_with_optimization(
    features_path="embeddings/concatenated_matrix.npy",
    labels_path="embeddings/labels.npy"
)
```

### **End-to-End Pipeline - Regression** **NEW**
```python
# Complete regression pipeline with professional infrastructure
from src.regression import RegressionTrainer, RegressionEvaluator, prepare_regression_targets
from src.regression.config import get_production_config
from src.regression.logger import create_logger
from src.regression.validation import validate_regression_data
from src.utils import safe_get_numeric  # NEW: Shared utilities

# 1. Setup configuration
config = get_production_config()
config.update(dataset_name='human', output_dir=Path('results/exp1'))
config.save('config/exp1.json')

# 2. Professional logging
logger = create_logger(log_dir=config.output_dir / 'logs', verbose=True)
logger.section('REGRESSION PIPELINE')

# 3. Prepare regression targets (Ki > Kd > IC50)
y, df_filtered, measure_types, kept_indices = prepare_regression_targets(
    df, priority=['Ki', 'Kd', 'IC50'], verbose=True
)

# 4. Robust validation
X, y = validate_regression_data(X, y, feature_names=features)

# 5. Train all models
trainer = RegressionTrainer(config=config)
trainer.train_all(X_train, y_train, X_val, y_val)

# 6. Evaluate and select best
evaluator = RegressionEvaluator(verbose=config.verbose)
test_results = evaluator.evaluate_all(trainer.trained_models, X_test, y_test)
best_model = evaluator.get_best_model(metric='RMSE')
logger.success(f'Best model: {best_model}')
```

### **Individual Module Usage**
```python
# Database analysis only
from src.database import MolecularClusterer, BalanceChecker

clusterer = MolecularClusterer()
clusters = clusterer.cluster_by_similarity("data.tsv", threshold=0.7)

balance_checker = BalanceChecker()
balance_report = balance_checker.analyze_balance("data.tsv")

# Embedding generation only  
from src.build.embeddings import LigandEmbedding, ProteinEmbedding

ligand_emb = LigandEmbedding(model_name="smi-ted")
protein_emb = ProteinEmbedding(model_name="esm2")

# Classification only
from src.classifier.modular_classifier import ModularClassifier

classifier = ModularClassifier()
results = classifier.train_and_evaluate("features.npy", "labels.npy")

# Regression only (NEW)
from src.regression import RegressionModels

models = RegressionModels(random_state=42)
models.train_random_forest(X_train, y_train)
predictions = models.predict('RandomForest', X_test)

# Shared utilities (NEW)
from src.utils import safe_get, safe_get_numeric, safe_get_int, safe_get_str

# Safe dictionary access
value = safe_get(row, 'activity', default=np.nan)
numeric_value = safe_get_numeric(row, 'pchembl_value', default=0.0)
```

---

## 📊 **Performance & Capabilities**

### **System Performance**
- **Processing Scale**: Handles datasets with 100K+ molecular compounds
- **Memory Efficiency**: Optimized for large-scale molecular data processing
- **Parallel Processing**: Apache Spark integration for distributed computing
- **GPU Acceleration**: CUDA support for embedding generation and ML training

### **Classification Performance**
- **ROC-AUC**: 0.85 ± 0.01 (cross-validated)
- **Precision**: 0.83 ± 0.02
- **Recall**: 0.81 ± 0.02
- **F1-Score**: 0.82 ± 0.02

### **Supported Data Types**
- **Ligands**: SMILES strings, SDF files, molecular fingerprints
- **Proteins**: FASTA sequences, UniProt IDs, PDB structures
- **Labels**: Binary classification, multi-class, regression targets

---

## 🛠️ **Development & Customization**

### **Adding New Embedding Models**
```python
# Extend base embedding classes
from src.build.embeddings import BaseEmbedding

class CustomLigandEmbedding(BaseEmbedding):
    def generate_embeddings(self, smiles_list):
        # Implement custom embedding logic
        pass
```

### **Adding New Classifiers**
```python
# Extend base classifier
from src.classifier.core.base_classifier import BaseClassifier

class CustomClassifier(BaseClassifier):
    def train(self, X, y):
        # Implement custom ML algorithm
        pass
```

### **Custom Database Analyzers**
```python
# Extend analysis capabilities
from src.database.core import BaseAnalyzer

class CustomAnalyzer(BaseAnalyzer):
    def analyze(self, data):
        # Implement custom analysis
        pass
```

---

## 📚 **Documentation**

- **[Build Module](build/README.md)**: Complete pipeline architecture and embedding generation
- **[Classifier Module](classifier/README.md)**: Machine learning classification system
- **[Database Module](database/README.md)**: Molecular data analysis and processing
- **[Migration Guides](build/MODULAR_MIGRATION_GUIDE.md)**: Legacy to modular system transition

---

## 🧪 **Testing**

```bash
# Run comprehensive tests
python -m pytest tests/ -v

# Module-specific tests
python -m pytest tests/test_build.py -v
python -m pytest tests/test_classifier.py -v  
python -m pytest tests/test_database.py -v

# Performance benchmarks
python tests/benchmark_pipeline.py
```

---

## 🤝 **Contributing**

1. **Follow modular architecture**: Each module should be self-contained
2. **Maintain backward compatibility**: Legacy scripts should continue working
3. **Add comprehensive tests**: Include unit tests and integration tests
4. **Update documentation**: Keep README files current with changes
5. **Performance considerations**: Profile code and optimize for large datasets

---

## 📄 **License & Citation**

This project is licensed under the MIT License. If you use DockTKinase in your research, please cite:

```bibtex
@software{docktkinase2024,
  title={DockTKinase: Comprehensive Pipeline for Non-Human Kinase Analysis},
  author={GMMSB-LNCC Team},
  year={2024},
  url={https://github.com/gmmsb-lncc/docktkinase}
}
```

---

**🎯 For specific module documentation, please refer to the individual README files in each subdirectory.**
