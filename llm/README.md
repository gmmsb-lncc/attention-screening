# LLM - Language Models Directory

This directory contains all language models and related caches for protein and ligand embeddings.

## Directory Structure

```
llm/
├── README.md                    # This file
├── ESM/                         # ESM-2 protein language models
│   ├── esm/                     # ESM repository (Facebook Research)
│   └── esm-3/                   # ESM-3/ESM-C models (EvolutionaryScale)
├── FM4M/                        # Foundation Models for Molecules
│   └── model_files/             # SMI-TED model weights
├── MoLFormer/                   # MoLFormer ligand embeddings (DeepChem)
│   ├── README.md                # MoLFormer documentation
│   ├── download_model.py        # Download and cache model
│   └── extract_embeddings.py    # Extract embeddings from SMILES
└── models_cache/                # Downloaded model cache
    ├── ESM/                     # ESM-2 checkpoints
    ├── ESM3/                    # ESM-C checkpoints
    ├── molformer/               # MoLFormer model cache
    └── embeddings/              # Cached embeddings
```

## Supported Models

### Protein Embeddings

#### ESM-2 (Meta AI / Facebook Research)
Bidirectional transformer trained with Masked Language Modeling (MLM) objective on UniRef50.

Local path: `llm/ESM/esm/`

| Model | Parameters | Size | Embedding Dim | Layers |
|-------|------------|------|---------------|--------|
| `esm2_t6_8M_UR50D` | 8M | ~31 MB | 320 | 6 |
| `esm2_t12_35M_UR50D` | 35M | ~138 MB | 480 | 12 |
| `esm2_t30_150M_UR50D` | 150M | ~573 MB | 640 | 30 |
| `esm2_t33_650M_UR50D` | 650M | ~2.5 GB | 1280 | 33 |
| `esm2_t36_3B_UR50D` | 3B | ~11 GB | 2560 | 36 |
| `esm2_t48_15B_UR50D` | 15B | ~55 GB | 5120 | 48 |

#### ESM-3 / ESM-C (EvolutionaryScale)
Causal transformer trained with Next Token Prediction (NTP). Better for capturing generative protein grammar and long-range dependencies.

Local path: `llm/ESM/esm-3/esm-main/`

| Model | Parameters | Embedding Dim | Layers | Notes |
|-------|------------|---------------|--------|-------|
| `esmc-300m-2024-12` | 300M | 960 | 30 | Local |
| `esmc-600m-2024-12` | 600M | 1152 | 36 | Local |
| `esmc-6b-2024-12` | 6B | 4096 | 56 | API only (requires ESM_API_KEY) |

### Ligand Embeddings

#### MoLFormer (DeepChem) - RECOMMENDED
Transformer-based molecular representation model trained on ZINC20 + PubChem.

Local path: `llm/MoLFormer/`

| Model | Parameters | Embedding Dim | Output Type |
|-------|------------|---------------|-------------|
| `MoLFormer-c3-1.1B` | 46.8M | 768 | **Matrix** `[seq_len, 768]` |

**Advantages**:
- Produces **per-token embeddings** (matrix) for cross-attention compatibility
- Larger training corpus (ZINC20 + PubChem)
- Full token-level representations for interaction analysis
- Recommended for DT-Kinase architecture

#### SMI-TED (IBM Foundation Models for Molecules)
SMILES-based Transformer Encoder-Decoder from IBM Research.

Local path: `llm/FM4M/model_files/`

| Model | Parameters | Embedding Dim | Output Type |
|-------|------------|---------------|-------------|
| `smi-ted-Light` | 40M | 768 | **Vector** `[1, 768]` |

**Use case**: Best for classical ML models that require fixed-size input vectors.

### Model Comparison

| Feature | MoLFormer | SMI-TED |
|---------|-----------|---------|
| Output shape | `[seq_len, 768]` | `[1, 768]` |
| Per-token embeddings | Yes | No |
| Cross-attention support | Full | Limited |
| Training data | ZINC20 + PubChem | ChEMBL |
| Best for | DT-Kinase, attention models | Classical ML |

## Installation

Models are loaded from local directories. No global installation required.

### Clone ESM-2 (if not present)
```bash
cd llm
git clone https://github.com/facebookresearch/esm.git ESM
```

### Clone ESM-3 (if not present)
```bash
cd llm/ESM
mkdir -p esm-3
git clone https://github.com/evolutionaryscale/esm.git esm-3/esm-main
```

### Clone FM4M (if not present)
```bash
cd llm
git clone https://github.com/IBM/foundation-models-for-materials.git FM4M
```

### Download MoLFormer
```bash
python llm/MoLFormer/download_model.py
```

## Usage

### ESM-2 Embeddings
```python
# ESM-2 is loaded via sys.path (NOT pip install)
import sys
sys.path.insert(0, 'llm/ESM')

import esm
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
```

### ESM-3/ESM-C Embeddings
```python
from src.build.embeddings.strategies.esmc_strategy import ESMCStrategy

strategy = ESMCStrategy()
strategy.load('esmc-600m-2024-12', device=torch.device('cuda'))
embedding = strategy.get_embedding(sequence)
```

### SMI-TED Embeddings
```python
# SMI-TED is loaded via FM4M
sys.path.insert(0, 'llm/FM4M')
from models.smi_ted.smi_ted_light.load import load_smi_ted
```

### MoLFormer Embeddings (RECOMMENDED for cross-attention)
```python
from llm.MoLFormer import MoLFormerEmbedder

embedder = MoLFormerEmbedder()

# Matrix embedding (per-token) - for cross-attention models
matrix = embedder.extract_matrix_embedding("CCO")  # [seq_len, 768]

# Vector embedding (pooled) - for similarity/classification
vector = embedder.extract_vector_embedding("CCO")  # [768]

# Batch processing
smiles_list = ["CCO", "C1=CC=CC=C1", "CC(=O)O"]
matrices = embedder.extract_batch(smiles_list, return_matrix=True)
vectors = embedder.extract_batch(smiles_list, return_matrix=False)
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ESM_API_KEY` | API key for ESM-C 6B (EvolutionaryScale Forge) | `your_api_key` |
| `ESM_CACHE` | Override ESM model cache | `/path/to/cache` |

## Notes

- This directory is ignored by Git (`.gitignore`)
- Models can occupy several GB of disk space
- Large models (15B+) require `accelerate` package for CPU offloading
- Use `--protein-model` to select which protein model to use in the pipeline

## Troubleshooting

### ESM import error
```
ModuleNotFoundError: No module named 'esm'
```
**Solution**: Ensure ESM is cloned to `llm/ESM/` and added to `sys.path` before import.

### Large model memory error
```
OutOfMemoryError: CUDA out of memory
```
**Solution**: Install `accelerate` package and use CPU offloading:
```bash
pip install accelerate>=0.20.0
```

### ESM-C 6B API error
```
Error: ESM_API_KEY not found
```
**Solution**: Set the API key:
```bash
export ESM_API_KEY="your_api_key"
```
Or pass via command line: `--api your_api_key`

### MoLFormer not found
```
FileNotFoundError: Model not found in cache
```
**Solution**: Download the model first:
```bash
python llm/MoLFormer/download_model.py
```
