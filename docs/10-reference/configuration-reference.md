# Configuration Reference

**Last Updated**: December 6, 2025  
**Version**: 2.1

This document details the configuration constants and options available in DockTKinase.

---

## 🧬 Protein Models (ESM/Boltz)

Defined in `src.build.core.constants.py`.

### ESM-2 (Meta AI)
Standard protein language models.

| Model Name | Parameters | Dim | Layers | Max Len |
|------------|------------|-----|--------|---------|
| `esm2_t6_8M_UR50D` | 8M | 320 | 6 | 1024 |
| `esm2_t12_35M_UR50D` | 35M | 480 | 12 | 1024 |
| `esm2_t30_150M_UR50D` | 150M | 640 | 30 | 1024 |
| `esm2_t33_650M_UR50D` | 650M | 1280 | 33 | 1024 |
| `esm2_t36_3B_UR50D` | 3B | 2560 | 36 | 4096 |
| `esm2_t48_15B_UR50D` | 15B | 5120 | 48 | 5120 |

### ESM-C (EvolutionaryScale)
Next-generation generative models.

| Model Name | Parameters | Dim | Notes |
|------------|------------|-----|-------|
| `esmc-300m-2024-12` | 300M | 960 | Local |
| `esmc-600m-2024-12` | 600M | 1152 | Local |
| `esmc-6b-2024-12` | 6B | 3072 | Requires API Key |

### Structure Models
| Model Name | Dim | Type |
|------------|-----|------|
| `boltz2` | 384 | Structure + Affinity |
| `openfold3` | 384 | Structure Prediction |

---

## 🧪 Ligand Models (FM4M)

Foundation Models for Molecules.

| Model Name | Dim | Type | Default |
|------------|-----|------|---------|
| `SMI-TED` | 768 | Transformer | ✅ |
| `SELFIES-TED` | 768 | Transformer | |
| `SMI-SSED` | 768 | Encoder | |

---

## 📁 Directory Structure

Default paths used by the system.

| Constant | Default Value | Description |
|----------|---------------|-------------|
| `DEFAULT_LIGAND_DIR` | `ligand` | Raw ligand data |
| `DEFAULT_PROTEIN_DIR` | `protein` | Raw protein data |
| `DEFAULT_LIGAND_OUTPUT_DIR` | `ligand_embeddings` | Generated ligand embeddings |
| `DEFAULT_PROTEIN_OUTPUT_DIR` | `protein_embeddings` | Generated protein embeddings |
| `DEFAULT_MATRIX_OUTPUT_DIR` | `matrix_embedding` | Combined training matrices |

---

## ⚙️ System Configuration

### Spark
```python
SPARK_CONFIG = {
    'app_name': 'DockTKinase-Build',
    'memory_fraction': 0.8,
    'offheap_fraction': 0.2,
    'gc_type': 'G1GC'
}
```

### Memory Management
```python
MEMORY_CONFIG = {
    'low_memory_threshold': 4,  # GB
    'high_memory_threshold': 16, # GB
    'batch_size_factor': 2
}
```

### Stratification
```python
STRATIFICATION_DEFAULT_CLUSTERING_ALGORITHM = 'dbscan'
STRATIFICATION_DEFAULT_SIMILARITY_THRESHOLD = 0.8
STRATIFICATION_DEFAULT_CLUSTER_MIN_SIZE = 5
```
