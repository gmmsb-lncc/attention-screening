# Build System - Modular Architecture

**A comprehensive modular system for building embedding matrices and processing molecular data for DockTKinase.**

## 🏗️ Overview

The `src/build/` directory contains a completely modular architecture that replaces legacy build scripts with a clean, maintainable, and highly configurable system. This system handles:

- **Protein embeddings** generation using ESM models
- **Ligand embeddings** generation using FM4M models  
- **Matrix concatenation** of protein and ligand embeddings
- **Label generation** for interaction and binary classification
- **Data validation** and integrity checks
- **Pipeline orchestration** for automated workflows

## 📁 Directory Structure

```
src/build/
├── core/                   # Foundation classes and configuration
│   ├── __init__.py
│   ├── base_builder.py     # Abstract base class for all builders
│   ├── config.py           # Centralized configuration system
│   ├── constants.py        # System constants and model definitions
│   └── exceptions.py       # Custom exception hierarchy
│
├── embeddings/             # Embedding generation modules
│   ├── __init__.py
│   ├── base_embedding.py   # Base class for embedding generators
│   ├── protein_embedding.py # ESM-based protein embeddings
│   └── ligand_embedding.py  # FM4M-based ligand embeddings
│
├── matrix/                 # Matrix construction and manipulation
│   ├── __init__.py
│   ├── base_matrix.py      # Base matrix operations
│   ├── embedding_matrix.py # Concatenated embedding matrices
│   └── kinase_matrix.py    # Kinase-specific matrix operations
│
├── labels/                 # Label generation for ML tasks
│   ├── __init__.py
│   ├── base_labels.py      # Base label generator
│   ├── interaction_labels.py # Protein-ligand interaction labels
│   └── binary_labels.py    # Binary classification labels
│
├── utils/                  # Shared utilities
│   ├── __init__.py
│   ├── spark_utils.py      # Apache Spark integration
│   ├── memory_utils.py     # Memory management utilities
│   └── progress_utils.py   # Progress tracking and logging
│
├── validation/             # Data validation and quality checks
│   ├── __init__.py
│   ├── base_validator.py   # Base validation framework
│   └── matrix_validator.py # Matrix-specific validation
│
├── pipeline/               # Pipeline orchestration
│   ├── __init__.py
│   └── build_pipeline.py   # Main pipeline coordinator
│
├── build.py                # Complete workflow demonstration
└── example_usage.py        # Usage examples and tutorials
```

## 🚀 Quick Start

### Run Complete Workflow Demonstration

```bash
# See the complete system in action
python src/build/build.py
```

This will demonstrate all features and show you how to use each component.

### Basic Usage

```python
from build.core import BuildConfig
from build.pipeline import BuildPipeline

# 1. Create configuration
config = BuildConfig(
    ligand_dim=768,
    protein_dim=2560,
    batch_size=32,
    use_gpu=True
)

# 2. Initialize pipeline
pipeline = BuildPipeline(config)

# 3. Run complete build process
results = pipeline.run()
print(f"✅ Build completed: {results}")
```

### Advanced Configuration

```python
from build.core import BuildConfig

# Custom configuration
config = BuildConfig({
    # Directories
    'ligand_dir': 'custom_ligand_embeddings',
    'protein_dir': 'custom_protein_embeddings',
    'output_dir': 'custom_output',
    
    # Models
    'esm_model': 'esm2_t36_3B_UR50D',    # ESM-2 modelo mais recente (3B params)
    'esm_model_path': '../ESM',          # Caminho do código ESM local
    'fm4m_model': 'SELFIES-TED',         # Alternative ligand model
    'fm4m_model_path': '../FM4M',        # Caminho do código FM4M local
    
    # Performance
    'batch_size': 16,
    'use_parallel': True,
    'checkpoint_enabled': True
})
```

### Component-by-Component Usage

```python
# Individual component usage
from build.embeddings import ProteinEmbedding, LigandEmbedding
from build.matrix import EmbeddingMatrix
from build.core import BuildConfig

config = BuildConfig()

# Generate protein embeddings
protein_emb = ProteinEmbedding(config)
protein_results = protein_emb.build()

# Generate ligand embeddings  
ligand_emb = LigandEmbedding(config)
ligand_results = ligand_emb.build()

# Create concatenated matrix
matrix = EmbeddingMatrix(config)
final_matrix = matrix.build()
```

## 🔧 Configuration Options

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ligand_dim` | int | 768 | Dimension of ligand embeddings |
| `protein_dim` | int | 2560 | Dimension of protein embeddings |
| `embedding_type` | str | 'cls' | Embedding type ('cls' or 'mean') |
| `batch_size` | int | 32 | Batch size for processing |
| `use_gpu` | bool | True | Enable GPU acceleration |
| `use_parallel` | bool | True | Enable parallel processing |

### Model Selection

**ESM Models (Proteins):**
- `esm2_t48_15B_UR50D` - Largest, highest quality (15B parameters)
- `esm2_t36_3B_UR50D` - **Default** - Good balance (3B parameters)
- `esm2_t33_650M_UR50D` - Faster processing (650M parameters)
- `esm2_t30_150M_UR50D` - Lightweight (150M parameters)

**FM4M Models (Ligands):**
- `SMI-TED` - **Default** - SMILES-based transformer
- `SELFIES-TED` - SELFIES-based transformer  
- `SMI-SSED` - SMILES encoder
- `MHG` - Molecular hypergraph
- `MOL-MOE` - Mixture of experts

### Directory Structure

```python
config = BuildConfig({
    'base_dir': '.',                        # Project root
    'ligand_dir': 'ligand_embeddings',      # Input ligand embeddings
    'protein_dir': 'protein_embeddings',    # Input protein embeddings
    'matrix_output_dir': 'output_matrices', # Output matrices
    'concatenated_output_dir': 'final_output' # Final concatenated results
})
```

## 🔄 Backward Compatibility

The system maintains **100% backward compatibility** with legacy scripts:

```python
# Legacy interface (still works)
from build.matrix import EmbeddingMatrixReconstructor

matrix = EmbeddingMatrixReconstructor('/path/to/data.tsv')
result = matrix.reconstruct_matrix()
```

## 🎯 Enhanced Train/Validation/Test Splitting

### Cosine Similarity-Based Stratification

The system now includes advanced stratification methods using cosine similarity to create more balanced and representative train/validation/test splits:

```python
from build.core import BuildConfig
from build.pipeline import BuildPipeline

# Enable stratification with custom parameters
config = BuildConfig({
    'stratification_enabled': True,
    'stratification_params': {
        'clustering_algorithm': 'dbscan',  # 'dbscan', 'hierarchical', 'kmeans', 'random'
        'similarity_threshold': 0.8,
        'cluster_min_size': 5,
        'stratify_by': 'both',  # 'ligand', 'protein', 'both', 'combined'
        'protein_weight': 0.6,
        'ligand_weight': 0.4
    }
})

# Run pipeline with stratified splits
pipeline = BuildPipeline(config)
success = pipeline.run_complete_pipeline(
    input_tsv_path='input.tsv',
    output_dir='output/',
    stratify_splits=True,  # Enable stratified splitting
    test_size=0.2,
    val_size=0.1
)
```

### Multi-View Stratification

The system supports stratification based on:
- **Ligand similarity**: Groups compounds by structural/chemical similarity
- **Protein similarity**: Groups proteins by sequence/structural similarity  
- **Combined similarity**: Uses weighted combination of both views

### Split Validation

Comprehensive validation ensures high-quality splits:
- Label distribution balance across splits
- Similarity preservation assessment
- Novelty validation for test set
- Diversity metrics for each split

## 📊 Performance Features

### Intelligent Caching
- **Embedding cache**: Avoids reloading frequently used embeddings
- **Memory management**: Automatic cleanup and garbage collection
- **Progress tracking**: Detailed progress bars and ETA

### Parallel Processing
- **Multi-threaded**: Parallel embedding generation
- **Spark integration**: Distributed processing for large datasets
- **GPU acceleration**: CUDA support for compatible models

### Checkpointing
```python
config = BuildConfig({
    'checkpoint_enabled': True,
    'checkpoint_file': 'build_checkpoint.json'
})
# Automatically resumes from last checkpoint on interruption
```

## 🛡️ Error Handling and Validation

### Robust Error Handling
```python
from build.core.exceptions import BuildError, ConfigurationError, MatrixError

try:
    pipeline = BuildPipeline(config)
    results = pipeline.run()
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
except MatrixError as e:
    print(f"Matrix processing error: {e}")
except BuildError as e:
    print(f"General build error: {e}")
```

### Automatic Validation
- **Input validation**: Checks file formats and dimensions
- **Matrix validation**: Verifies concatenation integrity
- **Missing data detection**: Logs and handles missing embeddings
- **Dimension consistency**: Ensures compatible embedding dimensions

## 📝 Logging and Monitoring

### Structured Logging
```python
import logging
logging.basicConfig(level=logging.INFO)

# Detailed logs from all components
pipeline = BuildPipeline(config)
results = pipeline.run()  # Automatically logged
```

### Log Files
- `missing_embeddings.log` - Records missing embedding files
- `build_progress.log` - Tracks build progress and timing
- `validation_report.log` - Validation results and warnings

## 🧪 Testing and Validation

### Unit Tests
```bash
# Run all build system tests
python -m pytest tests/test_build_system.py -v

# Test specific components
python -m pytest tests/test_embeddings.py -v
python -m pytest tests/test_matrix.py -v
```

### Integration Tests
```python
# Full pipeline test
from build.tests import test_full_pipeline
test_full_pipeline()  # ✅ Comprehensive validation
```

## 🚀 Production Deployment

### Recommended Configuration for Production

```python
production_config = BuildConfig({
    # Performance optimization
    'batch_size': 64,
    'use_gpu': True,
    'use_parallel': True,
    'checkpoint_enabled': True,
    
    # Quality settings
    'esm_model': 'esm2_t36_3B_UR50D',  # High quality protein embeddings
    'fm4m_model': 'SMI-TED',           # Proven ligand model
    'embedding_type': 'cls',           # Best for classification
    
    # Resource management
    'memory_config': {
        'low_memory_threshold': 8,     # GB
        'high_memory_threshold': 32    # GB
    },
    
    # Validation
    'validation_enabled': True,
    'min_embedding_size': 100,
    'max_embedding_size': 10000
})
```

### Docker Deployment
```dockerfile
# Example Dockerfile snippet
FROM pytorch/pytorch:latest

COPY src/build/ /app/src/build/
WORKDIR /app

# Install dependencies
RUN pip install -r requirements.txt

# Run build system
CMD ["python", "-c", "from build.pipeline import BuildPipeline; from build.core import BuildConfig; BuildPipeline(BuildConfig()).run()"]
```

## 🔍 Troubleshooting

### Common Issues

**1. CUDA/GPU Issues**
```python
# Disable GPU if issues
config = BuildConfig({'use_gpu': False})
```

**2. Memory Issues**
```python
# Reduce batch size and enable checkpointing
config = BuildConfig({
    'batch_size': 8,
    'checkpoint_enabled': True
})
```

**3. Missing Dependencies**
```bash
# Install optional dependencies
pip install torch transformers  # For ESM
pip install pyspark             # For distributed processing
```

**4. Model Download Issues**
```python
# Pre-download models
from build.embeddings import ProteinEmbedding
protein_emb = ProteinEmbedding()
protein_emb.download_model('esm2_t36_3B_UR50D')  # Manual download
```

## 📚 Examples and Tutorials

### Example 1: Basic Build
```python
from build.core import BuildConfig
from build.pipeline import BuildPipeline

config = BuildConfig()
pipeline = BuildPipeline(config)
results = pipeline.run()
print(f"✅ Processed {results['total_pairs']} protein-ligand pairs")
```

### Example 2: Custom Model Selection
```python
config = BuildConfig({
    'esm_model': 'esm2_t36_3B_UR50D',    # ESM-2 modelo padrão (3B params)
    'esm_model_path': '../ESM',          # Código ESM local
    'fm4m_model': 'SELFIES-TED',         # Alternative encoding
    'fm4m_model_path': '../FM4M',        # Código FM4M local
    'batch_size': 16
})

pipeline = BuildPipeline(config)
results = pipeline.run()
```

### Example 3: Large-Scale Processing
```python
config = BuildConfig({
    'use_parallel': True,
    'batch_size': 128,
    'checkpoint_enabled': True,
    'spark_config': {
        'app_name': 'DockTKinase-Production',
        'memory_fraction': 0.8
    }
})

pipeline = BuildPipeline(config)
results = pipeline.run()
```

## 🤝 Contributing

To contribute to the build system:

1. **Follow the module pattern**: Inherit from base classes
2. **Add comprehensive tests**: Test both functionality and edge cases
3. **Update documentation**: Keep README and docstrings current
4. **Maintain compatibility**: Ensure backward compatibility with legacy scripts

### Adding New Components

```python
# Example: Adding a new embedding type
from build.embeddings.base_embedding import BaseEmbedding

class CustomEmbedding(BaseEmbedding):
    def _validate_config(self):
        # Add custom validation
        pass
        
    def build(self):
        # Implement custom embedding logic
        pass
```

## 📄 License and Credits

- **License**: Same as parent project
- **Original Scripts**: Available in `backup_legacy_scripts/`
- **Architecture**: Modular design maintaining 100% output compatibility
- **Testing**: Comprehensive test suite ensuring reliability

---

## ✅ System Status

- **🎯 Compatibility**: 100% with legacy scripts
- **🚀 Performance**: Optimized with caching and parallel processing  
- **🛡️ Reliability**: Comprehensive error handling and validation
- **📚 Documentation**: Complete with examples and troubleshooting
- **🧪 Testing**: Full test coverage with integration tests
- **🔧 Maintenance**: Clean, modular, and extensible architecture

**Ready for production use! 🚀**
