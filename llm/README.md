# LLM - Language Models Directory

This directory contains all language models and related caches for protein and ligand embeddings.

## Directory Structure

```
llm/
├── README.md                    # This file
├── BOLTZ-2/                     # Boltz-2 biomolecular foundation model
│   └── boltz-main/              # Boltz repository clone
│       └── src/boltz/           # Boltz Python module
├── ESM/                         # ESM-2 protein language models
│   ├── esm/                     # ESM repository (Facebook Research)
│   └── esm-3/                   # ESM-3/ESM-C models (EvolutionaryScale)
├── FM4M/                        # Foundation Models for Molecules
│   └── model_files/             # SMI-TED model weights
├── MoLFormer/                   # MoLFormer ligand embeddings (DeepChem)
│   ├── README.md                # MoLFormer documentation
│   ├── download_model.py        # Download and cache model
│   └── extract_embeddings.py    # Extract embeddings from SMILES
├── OPENFOLD-3/                  # OpenFold3 structure prediction
└── models_cache/                # Downloaded model cache
    ├── ESM/                     # ESM-2 checkpoints
    ├── ESM3/                    # ESM-C checkpoints
    ├── molformer/               # MoLFormer model cache
    └── embeddings/              # Cached embeddings
```

## Supported Models

### Protein Embeddings

#### ESM-2 (Facebook Research)
Local path: `llm/ESM/esm/`

| Model | Parameters | Size | Embedding Dim |
|-------|------------|------|---------------|
| `esm2_t6_8M_UR50D` | 8M | ~31 MB | 320 |
| `esm2_t12_35M_UR50D` | 35M | ~138 MB | 480 |
| `esm2_t30_150M_UR50D` | 150M | ~573 MB | 640 |
| `esm2_t33_650M_UR50D` | 650M | ~2.5 GB | 1280 |
| `esm2_t36_3B_UR50D` | 3B | ~11 GB | 2560 |
| `esm2_t48_15B_UR50D` | 15B | ~55 GB | 5120 |

#### ESM-C/ESM-3 (EvolutionaryScale)
Local path: `llm/ESM/esm-3/esm-main/`

| Model | Parameters | Embedding Dim |
|-------|------------|---------------|
| `esmc-300m-2024-12` | 300M | 960 |
| `esmc-600m-2024-12` | 600M | 1152 |
| `esmc-6b-2024-12` | 6B | 2560 |

#### Boltz-2 (Structure + Affinity Prediction)
Local path: `llm/BOLTZ-2/boltz-main/src/`

| Model | Parameters | Embedding Dim | Features |
|-------|------------|---------------|----------|
| `boltz2` | ~400M | 384 | Structure prediction + binding affinity |

### Ligand Embeddings

#### MoLFormer (DeepChem) - RECOMMENDED
Local path: `llm/MoLFormer/`

| Model | Parameters | Embedding Dim | Output Type |
|-------|------------|---------------|-------------|
| `MoLFormer-c3-1.1B` | 46.8M | 768 | **Matrix** `[seq_len, 768]` |

**Advantages over SMI-TED:**
- Produces **per-token embeddings** (matrix) for cross-attention compatibility
- Larger training corpus (ZINC20 + PubChem)
- Full token-level representations for interaction analysis

#### SMI-TED (IBM Foundation Models for Molecules)
Local path: `llm/FM4M/model_files/`

| Model | Parameters | Embedding Dim | Output Type |
|-------|------------|---------------|-------------|
| `smi-ted-Light` | 40M | 768 | **Vector** `[1, 768]` |

**Comparison:**
| Feature | MoLFormer | SMI-TED |
|---------|-----------|---------|
| Output shape | `[seq_len, 768]` | `[1, 768]` |
| Per-token embeddings | Yes | No |
| Cross-attention support | Full | Limited |
| Training data | ZINC20 + PubChem | ChEMBL |

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

### Clone Boltz-2 (if not present)
```bash
cd llm
mkdir -p BOLTZ-2
git clone https://github.com/jwohlwend/boltz.git BOLTZ-2/boltz-main
cd BOLTZ-2/boltz-main
pip install -e .  # Optional: install as editable package
```

### Clone FM4M (if not present)
```bash
cd llm
git clone https://github.com/IBM/foundation-models-for-materials.git FM4M
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

### Boltz-2 Embeddings
```python
from src.build.embeddings.strategies.boltz_strategy import BoltzStrategy

strategy = BoltzStrategy()
strategy.load('boltz2', device=torch.device('cpu'))
# Automatically finds local installation in llm/BOLTZ-2/boltz-main/src/
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

**First-time download:**
```bash
python llm/MoLFormer/download_model.py
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `BOLTZ_HOME` | Override Boltz installation path | `/path/to/boltz` |
| `ESM_CACHE` | Override ESM model cache | `/path/to/cache` |

## Notes

- This directory is ignored by Git (`.gitignore`)
- Models can occupy several GB of disk space
- Large models (15B+) require `accelerate` package for CPU offloading
- Use `--esm-model` to select which protein model to use in the pipeline

## Troubleshooting

### Boltz not found
```
RuntimeError: Boltz CLI not found
```
**Solution**: Clone Boltz to `llm/BOLTZ-2/boltz-main/` or set `BOLTZ_HOME` environment variable.

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

