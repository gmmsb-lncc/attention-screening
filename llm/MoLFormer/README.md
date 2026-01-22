# MoLFormer - Molecular Transformer for Ligand Embeddings

## Overview

**MoLFormer** (`DeepChem/MoLFormer-c3-1.1B`) is a Transformer encoder model trained with Masked Language Modeling (MLM) on SMILES representations of molecules. Unlike SMI-TED which only produces vector embeddings, **MoLFormer can extract matrix embeddings** (per-token representations) for ligands.

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | `DeepChem/MoLFormer-c3-1.1B` |
| **Architecture** | Transformer encoder (bidirectional, BERT-like) |
| **Parameters** | ~46.8 million |
| **Training Data** | ZINC20 (~1B SMILES) + PubChem (~100M SMILES) |
| **Hidden Dimension** | 768 |
| **Max Sequence Length** | 512 tokens |

## Advantages over SMI-TED

| Feature | SMI-TED | MoLFormer |
|---------|---------|-----------|
| **Output Type** | Vector only `[1, 768]` | Matrix `[seq_len, 768]` |
| **Per-token embeddings** | No | Yes |
| **Cross-attention compatible** | Limited | Full support |
| **Pooling flexibility** | None (fixed) | Mean, CLS, Max, etc. |

## Installation

```bash
pip install transformers torch
```

## Quick Start

### Extract Matrix Embeddings (per-token)

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("DeepChem/MoLFormer-c3-1.1B")
model = AutoModelForMaskedLM.from_pretrained("DeepChem/MoLFormer-c3-1.1B")
model.eval()

# SMILES input
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

# Tokenize
inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)

# Extract embeddings
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# Matrix embedding: [seq_len, hidden_dim]
matrix_embedding = outputs.hidden_states[-1][0]  # Last layer, first sample
print(f"Matrix embedding shape: {matrix_embedding.shape}")

# Vector embedding (mean pooling): [hidden_dim]
vector_embedding = matrix_embedding.mean(dim=0)
print(f"Vector embedding shape: {vector_embedding.shape}")
```

### Batch Processing

```python
smiles_list = ["CCO", "C1=CC=CC=C1", "CC(=O)O"]

inputs = tokenizer(
    smiles_list,
    padding=True,
    truncation=True,
    return_tensors="pt",
    max_length=512
)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# Matrix embeddings: [batch, seq_len, hidden_dim]
matrix_embeddings = outputs.hidden_states[-1]

# Vector embeddings (mean pooling): [batch, hidden_dim]
attention_mask = inputs['attention_mask'].unsqueeze(-1)
vector_embeddings = (matrix_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
```

## Output Shapes

- **Matrix embedding**: `[seq_len, 768]` - One vector per SMILES token
- **Vector embedding**: `[768]` - Single vector per molecule (after pooling)

## Use Cases

1. **Protein-Ligand Interaction Prediction**
   - Use matrix embeddings with cross-attention models
   - Token-level interaction analysis

2. **Molecular Similarity**
   - Compare molecules using vector embeddings
   - Clustering and visualization

3. **Property Prediction**
   - Transfer learning for ADMET properties
   - Fine-tuning for specific endpoints

## Integration with CrossAttention Model

The matrix embeddings from MoLFormer are compatible with the CrossAttention architecture used in this project:

```python
# Protein: [seq_len_protein, protein_dim] from ESM-2
# Ligand:  [seq_len_ligand, 768] from MoLFormer

# CrossAttention can now attend over ligand tokens!
```

## References

- [Hugging Face Model Card](https://huggingface.co/DeepChem/MoLFormer-c3-1.1B)
- [ChemBERTa-3 GitHub](https://github.com/deepforestsci/chemberta3)
- Part of the DeepChem ecosystem for chemical machine learning
