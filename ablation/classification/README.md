# Classification Ablation Study

**Date**: January 16, 2026  
**Project**: DockTKinase - Representation Ablation Analysis

---

## 🎯 Objective

Perform a systematic ablation study to quantify the contribution of each representation (protein and ligand) to classification quality.

### Combinations to Test

| ID | Protein Representation | Ligand Representation | Dimensions | Classifiers |
|----|------------------------|----------------------|-------------|-------------|
| **C1** | ESM-2 embedding | SMI-TED embedding | 320/640/2560 + 768 | KNN, MLP |
| **C2** | ESM-2 embedding | Morgan FP 2048 bits | 320/640/2560 + 2048 | KNN, MLP |
| **C3** | One-Hot Encoding (AAC+DPC) | SMI-TED embedding | 420 + 768 | KNN, MLP |
| **C4** | One-Hot Encoding (AAC+DPC) | Morgan FP 2048 bits | 420 + 2048 | KNN, MLP |

- **C1** = Current baseline (existing embeddings)
- **C2, C3, C4** = New combinations to generate

---

## 📂 Directory Structure

```
/home/leon/ablation/classification/
├── README.md                           # This file
├── data/
│   ├── processed/
│   │   ├── proteins.csv                # seq_id, sequence (unique proteins)
│   │   ├── ligands.csv                 # chembl_id, smiles (unique ligands)
│   │   ├── interactions.csv            # seq_id, chembl_id, label
│   │   └── index_mapping.json          # Index mapping for reconstruction
│   ├── embeddings/
│   │   ├── protein_aac_dpc.npy         # AAC+DPC encoding [N_proteins, 420]
│   │   ├── ligand_morgan_2048.npy      # Morgan FP [N_ligands, 2048]
│   │   └── (symlinks to existing ESM-2 and SMI-TED embeddings)
│   └── combinations/
│       ├── C1_esm_smited/              # ESM-2 + SMI-TED (per model)
│       ├── C2_esm_morgan/              # ESM-2 + Morgan FP
│       ├── C3_aac_smited/              # AAC+DPC + SMI-TED
│       ├── C4_aac_morgan/              # AAC+DPC + Morgan FP
│       └── labels.npy                  # Binary labels
├── scripts/
│   ├── 01_extract_data.py              # Extract proteins/ligands from TSV
│   ├── 02_generate_morgan_fp.py        # Generate Morgan Fingerprints
│   ├── 03_generate_aac_dpc.py          # Generate AAC+DPC encoding
│   ├── 04_combine_representations.py   # Combine representations
│   ├── 05_run_ablation.py              # Run ablation study
│   └── 06_visualize_results.py         # Generate visualizations
└── results/
    ├── ablation_results.json           # All metrics
    └── ablation_comparison.png         # Comparative plot
```

---

## 📁 Existing Embeddings (Reuse)

### Non-Human Dataset
- **Location**: `/media/leon/ssd2tb/docktkinase/results/protein_model_benchmark_non_human_v2/`
- **Models**: `esm2_t6_8M_UR50D/`, `esm2_t30_150M_UR50D/`, `esm2_t36_3B_UR50D/`
- **Files**: 
  - `build/embedding_matrix.npy` - Combined [protein + ligand]
  - `build/binary_labels.npy` - Labels
  - `build/proteins/{seq_id}_embedding.npy` - Individual protein embeddings
  - `build/ligand_matrices/{chembl_id}_matrix.npy` - Individual ligand embeddings

### Human Dataset
- **Location**: `/data/docktkinase/results/protein_model_benchmark_human_v2/`
- **Same structure as non-human**

### Raw Data
- **Non-Human**: `/media/leon/ssd2tb/docktkinase/tests/datasets/kinase_non_human_compounds.tsv`
- **Human**: `/media/leon/ssd2tb/docktkinase/tests/datasets/kinase_human_compounds.tsv`

---

## 🔧 Implementation Steps

### Step 1: Extract and Index Data
**Script**: `01_extract_data.py`

Extract unique proteins and ligands from TSV, create index mappings.

**TSV Columns**:
- `chembl_id`: Ligand ID
- `canonical_smiles`: Ligand SMILES
- `seq`: Protein sequence
- `seq_id`: Protein ID
- `pchembl_value`: Activity value (label = 1 if >= 6.0)

---

### Step 2: Generate Morgan Fingerprints
**Script**: `02_generate_morgan_fp.py`

**Parameters**:
- Radius: 2 (ECFP4 equivalent)
- Bits: 2048
- useChirality: True

**Dependencies**: `rdkit`

---

### Step 3: Generate AAC+DPC Encoding
**Script**: `03_generate_aac_dpc.py`

**Amino Acid Composition (AAC)**: 20 dimensions  
**Dipeptide Composition (DPC)**: 400 dimensions  
**Total**: 420 dimensions

More compact than full one-hot encoding while preserving sequence information.

---

### Step 4: Combine Representations
**Script**: `04_combine_representations.py`

Create combination matrices for C1-C4 by concatenating:
- Protein representation (per seq_id)
- Ligand representation (per chembl_id)

---

### Step 5: Run Ablation Study
**Script**: `05_run_ablation.py`

**Configuration**:
- Seeds: [42, 123, 420, 777, 2024]
- Split: 80% train, 10% val, 10% test (stratified)
- Classifiers: 
  - KNN (k=5, weights='distance', metric='cosine')
  - MLP (hidden_layers=(256, 128), early_stopping=True)
- Metrics: AUC-ROC, MCC, Accuracy, F1

---

### Step 6: Visualization
**Script**: `06_visualize_results.py`

Generate comparative plots showing performance across all combinations.

---

## 📊 Expected Hypotheses

| Combination | Expected AUC | Reasoning |
|-------------|--------------|-----------|
| C1 (ESM+SMITED) | ~0.95 | Current baseline (inflated by data leakage) |
| C2 (ESM+Morgan) | ~0.90-0.93 | Morgan is simpler than SMI-TED |
| C3 (AAC+SMITED) | ~0.85-0.90 | AAC loses structural information |
| C4 (AAC+Morgan) | ~0.80-0.85 | Simplest representations |

**Key Questions**:
1. If C4 ≈ C1: Sophisticated embeddings add no value
2. If C1 >> C4: Embeddings capture important information
3. If C2 > C3: Ligand representation matters more
4. If C3 > C2: Protein representation matters more

---

## 🚀 Quick Start

```bash
cd /home/leon/ablation/classification

# Run all steps
python scripts/01_extract_data.py
python scripts/02_generate_morgan_fp.py
python scripts/03_generate_aac_dpc.py
python scripts/04_combine_representations.py
python scripts/05_run_ablation.py
python scripts/06_visualize_results.py
```

---

## 📝 Notes

- Use existing ESM-2 and SMI-TED embeddings from benchmark directories
- Only need to generate: Morgan FP (ligands) and AAC+DPC (proteins)
- All scripts should support both human and non-human datasets via CLI arguments

---

## 📅 Timeline

| Step | Time Estimate |
|------|---------------|
| Step 1: Data extraction | ~10 min |
| Step 2: Morgan FP | ~30 min |
| Step 3: AAC+DPC | ~10 min |
| Step 4: Combinations | ~30 min |
| Step 5: Ablation | ~2-3 hours |
| Step 6: Visualization | ~15 min |
| **Total** | **~4-5 hours** |

---

**Status**: 📋 Planning Complete  
**Next Step**: Implement `01_extract_data.py`
