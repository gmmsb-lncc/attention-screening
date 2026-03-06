# Attention Matrix Module

Cross-Attention based analysis of protein-ligand interactions using pre-computed embeddings.

## Overview

This module implements a Cross-Attention neural network architecture for:
- **Classification**: Predicting whether a compound is active (pChEMBL ≥ 7.0)
- **Regression**: Predicting the pChEMBL value (binding affinity)

The model uses matrix embeddings from:
- **Proteins**: ESM2 embeddings (per-residue representations)
- **Ligands**: SMI-TED embeddings (per-token representations)

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│  Protein Matrix │    │  Ligand Matrix  │
│  (seq, 320)     │    │  (tok, 768)     │
└────────┬────────┘    └────────┬────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│  Projection     │    │  Projection     │
│  → (seq, 256)   │    │  → (tok, 256)   │
└────────┬────────┘    └────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌─────────────────────┐
         │  Cross-Attention    │
         │  (Protein → Ligand) │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Mean Pooling       │
         │  + Concatenation    │
         └──────────┬──────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
┌─────────────────┐   ┌─────────────────┐
│  Regression     │   │  Classification │
│  (pChEMBL)      │   │  (active/not)   │
└─────────────────┘   └─────────────────┘
```

## Installation

The module is part of the DockTKinase project. No additional installation required.

## Usage

### From Command Line (via run_complete_pipeline.py)

```bash
# Run attention matrix analysis with default settings
python scripts/run_complete_pipeline.py \
    --input tests/datasets/kinase_non_human_compounds.tsv \
    --output results/my_experiment \
    --attention-matrix

# With custom settings
python scripts/run_complete_pipeline.py \
    --input tests/datasets/kinase_non_human_compounds.tsv \
    --output results/my_experiment \
    --attention-matrix \
    --attention-model-type improved \
    --attention-epochs 100 \
    --attention-split leakage_aware
```

### From Python Code

```python
from attention_matrix import AttentionMatrixPipeline, AttentionMatrixConfig

# Configure
config = AttentionMatrixConfig(
    protein_dim=320,           # ESM2 esm2_t6_8M_UR50D
    ligand_dim=768,            # SMI-TED
    hidden_dim=256,
    num_heads=8,
    num_layers=2,
    dropout=0.2,
    learning_rate=1e-4,
    batch_size=64,
    epochs=50,
    early_stopping_patience=10,
    activity_threshold=7.0     # pChEMBL = 7.0 → IC50 = 100 nM
)

# Create pipeline
pipeline = AttentionMatrixPipeline(
    config=config,
    output_dir="results/attention_matrix"
)

# Run complete pipeline
results = pipeline.run(
    data_path="data/compounds.tsv",
    protein_dir="embeddings/proteins",    # Directory with .npy files
    ligand_dir="embeddings/ligands",      # Directory with .npy files
    split_type="leakage_aware",           # or "simple"
    model_type="improved"                 # or "basic"
)

print(f"Accuracy: {results['test_metrics']['classification']['accuracy']:.4f}")
print(f"MAE: {results['test_metrics']['regression']['mae']:.4f}")
```

### Step-by-Step Usage

```python
from attention_matrix import (
    AttentionMatrixPipeline,
    AttentionMatrixConfig,
    AttentionAnalyzer
)

# 1. Configure
config = AttentionMatrixConfig(hidden_dim=256, num_heads=8)

# 2. Create pipeline
pipeline = AttentionMatrixPipeline(config, output_dir="results/attention")

# 3. Load data
pipeline.load_data(
    data_path="data/compounds.tsv",
    protein_dir="embeddings/proteins",
    ligand_dir="embeddings/ligands"
)

# 4. Split data (with leakage awareness)
pipeline.split_data(split_type="leakage_aware")

# 5. Build model
pipeline.build_model(model_type="improved")

# 6. Train
history = pipeline.train(epochs=50, patience=10)

# 7. Evaluate
metrics = pipeline.evaluate()

# 8. Analyze attention weights
analyzer = AttentionAnalyzer(pipeline.model, device=pipeline.device)
analysis = analyzer.analyze_batch(pipeline.test_loader, max_samples=100)
```

## Data Splitting

### Leakage-Aware Split (Recommended)

Prevents data leakage by ensuring similar proteins are not split across train/val/test sets:

1. Cluster proteins using hierarchical clustering on ESM2 embeddings
2. Allocate entire clusters to train/val/test sets
3. Guarantees zero protein overlap between splits

```python
from attention_matrix import LeakageAwareSplitter

splitter = LeakageAwareSplitter(
    n_clusters=None,      # Auto-detect
    test_size=0.1,
    val_size=0.1,
    random_state=42
)

train_idx, val_idx, test_idx = splitter.split(df, protein_dir)
```

### Simple Split (Baseline)

Standard stratified split. May have data leakage if similar proteins appear in different sets.

## Model Types

### Basic Model (`CrossAttentionModel`)
- Single cross-attention layer
- Simpler architecture
- Faster training
- Lower memory usage

### Improved Model (`ImprovedCrossAttentionModel`)
- Multiple cross-attention layers (default: 2)
- Deeper projection networks
- Feed-forward layers after attention
- Better performance on larger datasets

## Attention Analysis

Extract and interpret attention weights:

```python
from attention_matrix import AttentionAnalyzer, create_attention_heatmap_data

# Create analyzer
analyzer = AttentionAnalyzer(model, device)

# Extract attention for single sample
attention = analyzer.extract_attention(protein_emb, ligand_emb)

# Get residue importance scores
residue_scores = analyzer.get_residue_importance(attention['layer_0'])

# Get top interactions
top_interactions = analyzer.get_top_interactions(attention['layer_0'], top_k=10)
# Returns: [(residue_idx, ligand_idx, attention_weight), ...]

# Prepare data for visualization
heatmap_data = create_attention_heatmap_data(
    attention_map,
    residue_labels=['ALA1', 'GLY2', ...],
    ligand_labels=['C', 'C', 'O', ...]
)
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `protein_dim` | 320 | Protein embedding dimension (ESM2 model-dependent) |
| `ligand_dim` | 768 | Ligand embedding dimension (SMI-TED) |
| `hidden_dim` | 256 | Hidden dimension for projections |
| `num_heads` | 8 | Number of attention heads |
| `num_layers` | 2 | Number of cross-attention layers |
| `dropout` | 0.2 | Dropout rate |
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 1e-4 | Initial learning rate |
| `epochs` | 50 | Maximum training epochs |
| `early_stopping_patience` | 10 | Early stopping patience |
| `activity_threshold` | 7.0 | pChEMBL threshold for classification |
| `max_protein_len` | 256 | Maximum protein sequence length |
| `max_ligand_len` | 64 | Maximum ligand token length |

## Output Structure

```
results/attention_matrix_YYYYMMDD_HHMMSS/
├── config.json                  # Configuration used
├── pipeline.log                 # Training log
├── training_history.json        # Loss and metrics per epoch
├── test_metrics.json            # Final test metrics
├── results.json                 # Complete results summary
├── models/
│   └── best_model.pt            # Best model checkpoint
└── splits/
    ├── train_idx.npy            # Training indices
    ├── val_idx.npy              # Validation indices
    ├── test_idx.npy             # Test indices
    └── split_metadata.json      # Split statistics
```

## Module Structure

```
src/attention_matrix/
├── __init__.py              # Module exports
├── config.py                # AttentionMatrixConfig
├── model.py                 # CrossAttentionModel, ImprovedCrossAttentionModel
├── dataset.py               # ProteinLigandDataset, create_dataloaders
├── trainer.py               # AttentionTrainer
├── evaluator.py             # AttentionEvaluator
├── pipeline.py              # AttentionMatrixPipeline (orchestrator)
├── splitter.py              # LeakageAwareSplitter, SimpleSplitter
├── attention_analyzer.py    # AttentionAnalyzer
└── README.md                # This file
```

## Design Principles

- **SOLID**: Each class has a single responsibility
- **KISS**: Simple interfaces, minimal dependencies
- **Clean Code**: Well-documented, type-annotated, tested

## License

Same as DockTKinase project.
