# Modular Build System - Migration Guide

## Overview

The original `src/build/` directory has been completely refactored into a modular architecture with clear separation of concerns, better error handling, and improved maintainability.

## Architecture Overview

```
src/build/
├── core/                   # Base classes and configuration
│   ├── base_builder.py     # Abstract base for all builders  
│   ├── config.py          # Configuration management
│   ├── constants.py       # System constants
│   ├── exceptions.py      # Custom exceptions
│   └── __init__.py
├── utils/                  # Shared utilities
│   ├── file_utils.py      # File operations
│   ├── memory_utils.py    # Memory management
│   ├── spark_utils.py     # Spark optimization
│   ├── logging_utils.py   # Advanced logging
│   └── __init__.py
├── embeddings/            # Embedding generation
│   ├── base_embedding.py  # Abstract embedding interface
│   ├── protein_embedding.py  # ESM protein embeddings
│   ├── ligand_embedding.py   # FM4M ligand embeddings  
│   └── __init__.py
├── matrix/                # Matrix construction
│   ├── base_matrix.py     # Abstract matrix interface
│   ├── embedding_matrix.py # Standard embedding matrices
│   ├── kinase_matrix.py   # Kinase-specific matrices
│   └── __init__.py
├── labels/                # Label generation
│   ├── base_labels.py     # Abstract labels interface
│   ├── interaction_labels.py # Interaction labels from TSV
│   ├── binary_labels.py   # Binary classification labels
│   └── __init__.py
├── validation/            # Data validation
│   ├── base_validator.py  # Abstract validator interface
│   ├── matrix_validator.py # Matrix validation (replaces checkConcatenate.py)
│   └── __init__.py
├── pipeline/              # Pipeline orchestration
│   ├── build_pipeline.py  # Main pipeline coordinator
│   └── __init__.py
├── __init__.py           # Main module exports
└── example_usage.py      # Usage examples and migration guide
```

## Migration Map

### Original Scripts → Modular Components

| Original File | New Module | New Class | Purpose |
|---------------|------------|-----------|---------|
| `build.py` | `pipeline.BuildPipeline` | `BuildPipeline` | Main orchestrator |
| `embeddingMeta.py` | `embeddings.ProteinEmbedding` | `ProteinEmbedding` | ESM protein embeddings |
| `embeddingIBM.py` | `embeddings.LigandEmbedding` | `LigandEmbedding` | FM4M ligand embeddings |
| `buildEmbeddingMatrix.py` | `matrix.EmbeddingMatrix` | `EmbeddingMatrix` | Matrix construction |
| `buildInteractionLabels.py` | `labels.InteractionLabels` | `InteractionLabels` | Interaction labels |
| `buildbinaryLabels.py` | `labels.BinaryLabels` | `BinaryLabels` | Binary classification |
| `checkConcatenate.py` | `validation.MatrixValidator` | `MatrixValidator` | Matrix validation |

## Key Improvements

### 1. Configuration Management
- **Before**: Hardcoded values scattered across files
- **After**: Centralized JSON configuration with validation

```python
# Old way
batch_size = 16  # hardcoded

# New way  
config = BuildConfig.from_json('config.json')
batch_size = config.get('esm_config.batch_size', 16)
```

### 2. Error Handling
- **Before**: Basic try/catch with print statements
- **After**: Comprehensive exception hierarchy with proper logging

```python
# Old way
try:
    result = some_operation()
except:
    print("Error occurred")

# New way
try:
    result = some_operation() 
except BuildException as e:
    self.logger.error(f"Build error: {e}")
    raise
```

### 3. Memory Management
- **Before**: Manual memory cleanup
- **After**: Context managers for automatic resource management

```python
# New way
with MemoryContext(self.config) as mem:
    large_data = load_large_dataset()
    processed = process_data(large_data)
    # Automatic cleanup on exit
```

### 4. Dependency Management
- **Before**: Hard dependencies causing import errors
- **After**: Graceful fallbacks for optional dependencies

```python
# New approach
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    
if not HAS_TORCH:
    raise BuildException("PyTorch is required for embedding generation")
```

## Usage Examples

### Quick Start - Replace build.py

```python
from src.build import BuildConfig, BuildPipeline

# Create configuration
config = BuildConfig.from_json('config.json')

# Run complete pipeline (replaces original build.py)
pipeline = BuildPipeline(config)
success = pipeline.run_complete_pipeline(
    input_tsv_path='data.tsv',
    output_dir='output/',
    matrix_type='embedding',
    binary_threshold=1000.0
)
```

### Individual Components

```python
# Generate only protein embeddings
from src.build.embeddings import ProteinEmbedding

protein_emb = ProteinEmbedding(config)
protein_emb.generate_embeddings(tsv_path='data.tsv')

# Validate matrices
from src.build.validation import MatrixValidator

validator = MatrixValidator(config)
is_valid = validator.validate(
    concatenated_path='embeddings.npy',
    labels_path='labels.npy'
)
```

### Custom Pipeline Steps

```python
pipeline = BuildPipeline(config)

# Run individual steps
pipeline.run_embedding_generation(input_tsv, output_dir)
pipeline.run_matrix_construction(output_dir, matrix_type='kinase')
pipeline.run_label_generation(input_tsv, output_dir, threshold=500.0)
pipeline.run_validation()
```

## Configuration File Example

```json
{
    "project_name": "docktkinase_embeddings",
    "log_level": "INFO",
    "output_directory": "concatenated_embeddings",
    "spark_config": {
        "app_name": "DocktKinaseEmbeddings", 
        "master": "local[*]",
        "memory": "8g"
    },
    "esm_config": {
        "model_name": "esm2_t36_3B_UR50D",
        "model_path": "../ESM",
        "batch_size": 16,
        "device": "auto"
    },
    "fm4m_config": {
        "model_path": "../FM4M",
        "batch_size": 32
    },
    "binary_threshold": 1000.0
}
```

## Testing the Migration

1. **Run example script:**
   ```bash
   cd src/build/
   python example_usage.py
   ```

2. **Validate with existing data:**
   ```python
   from src.build.validation import MatrixValidator
   
   validator = MatrixValidator(config)
   validator.validate_concatenated_embeddings(
       'old_embeddings.npy',
       'old_labels.npy', 
       'original.tsv'
   )
   ```

3. **Compare outputs:**
   - Original build.py outputs in `concatenated_embeddings/`
   - Modular outputs in `concatenated_embeddings_modular/`
   - Use validation tools to ensure consistency

## Benefits

1. **Modularity**: Each component can be used independently
2. **Testability**: Individual components can be unit tested
3. **Maintainability**: Clear separation of concerns
4. **Extensibility**: Easy to add new embedding types or matrix builders
5. **Configuration**: Centralized, validated configuration management
6. **Error Handling**: Comprehensive error reporting and recovery
7. **Memory Efficiency**: Automatic memory management and optimization
8. **Logging**: Detailed logging for debugging and monitoring

## Backward Compatibility

The original scripts are preserved and can still be used, but the new modular system is recommended for:
- Better error handling and debugging
- Easier testing and validation
- More flexible configuration
- Better resource management
- Future extensibility

To migrate existing workflows, simply replace calls to the original `build.py` with the new `BuildPipeline` class using the provided examples.
