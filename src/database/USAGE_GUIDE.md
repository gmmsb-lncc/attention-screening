# 🚀 **USAGE GUIDE - Database Scripts**

**How to use each script in the `src/database/` folder in a practical way**

---

## 📋 **DIRECTLY EXECUTABLE SCRIPTS**

### 1️⃣ **Comparative Analysis** - `comparative_analysis.py`

```bash
# Execute directly:
cd src/database/
python3 comparative_analysis.py
```

**Prerequisites:**
- `kinase_human_compounds.tsv` (human kinase data)
- `kinase_non_human_compounds.tsv` (non-human kinase data)

**What it does:**
- Compares data between human and non-human kinases
- Generates statistics and visualizations
- Creates graphs in the `analysis_output/` directory

---

### 2️⃣ **Redundancy Removal** - `remove_redundance.py`

```bash
# Execute directly:
cd src/database/
python3 remove_redundance.py
```

**Prerequisites:**
- `../0_database/kinase_all_compounds_formatted.tsv`

**What it does:**
- Removes salts and canonizes SMILES
- Removes duplicates
- Calculates molecular descriptors
- Generates clean files: `nr_kinase_all_compounds_salt_free_ver3.tsv`

---

## 🔧 **SCRIPTS FOR CODE IMPORT**

### 3️⃣ **Molecular Clustering** - `cluster.py`

```python
# Import in your code:
from src.database.cluster import MoleculeClusterer

# Basic usage:
clusterer = MoleculeClusterer("path/to/file.tsv")
clusterer.load_data("canonical_smiles")
clusterer.parallel_generate_fingerprints("canonical_smiles", batch_size=1000)
clusters = clusterer.cluster_by_similarity(threshold=0.8)

# Visualizations:
tsne_results = clusterer.calculate_tsne()
clusterer.plot_tsne(tsne_results, threshold=0.8)
```

**Main methods:**
- `load_data(smile_column)` - Loads data
- `parallel_generate_fingerprints()` - Generates fingerprints
- `cluster_by_similarity(threshold)` - Performs clustering
- `calculate_tsne()` - Dimensional reduction
- `plot_tsne()` - t-SNE visualization

---

### 4️⃣ **Molecular Descriptors** - `descriptors.py`

```python
# Import in your code:
from src.database.descriptors import MolecularDescriptors

# Basic usage:
descriptors = MolecularDescriptors("path/to/file.tsv")
descriptors.compute_descriptors()
descriptors.save_descriptors("output_descriptors.tsv")

# Visualizations:
descriptors.plot_histograms(output_path="histograms.png")
descriptors.violin_plot()
```

**Main methods:**
- `calculate_descriptors(smiles_list)` - Calculates descriptors
- `compute_descriptors()` - Processes entire dataset
- `save_descriptors(path)` - Saves results
- `plot_histograms()` - Generates histograms
- `violin_plot()` - Violin plots

---

## 🗃️ **SQL QUERIES** - `sql/` Folder

```bash
# Use the SQL queries directly on ChEMBL:
cat sql/kinase_humans.sql | mysql -h chembl_host -u user -p chembl_35

# Or copy the queries to your favorite SQL client
```

**Available files:**
- `kinase_humans.sql` - Extracts human kinase data
- `kinase_non_humans.sql` - Extracts non-human kinase data
- `kinase_compounds.sql` - General compound queries
- `kinase_compounds_and_seq.sql` - Includes sequences

---

## 🔍 **ADVANCED MODULAR STRUCTURE**

For advanced users who want to use internal modules:

```python
# Core configuration:
from src.database.core.config import DatabaseConfig

# Processing modules:
from src.database.processing.molecular_clustering import MolecularClusterer
from src.database.processing.molecular_descriptors import MolecularDescriptors
from src.database.processing.data_cleaner import DataCleaner

# Analysis modules:
from src.database.analysis.comparative_analyzer import ComparativeAnalyzer
from src.database.analysis.balance_checker import BalanceChecker
```

---

## 📁 **EXPECTED FILE STRUCTURE**

```
your_working_folder/
├── kinase_human_compounds.tsv          # For comparative_analysis.py
├── kinase_non_human_compounds.tsv      # For comparative_analysis.py
├── 0_database/
│   └── kinase_all_compounds_formatted.tsv  # For remove_redundance.py
└── output/                             # Output directory (created automatically)
```

---

## ⚡ **QUICK EXAMPLES**

### Quick Comparative Analysis:
```bash
cd src/database/
python3 comparative_analysis.py
```

### Quick Data Cleaning:
```bash
cd src/database/
python3 remove_redundance.py
```

### Clustering in Python:
```python
from src.database.cluster import MoleculeClusterer
clusterer = MoleculeClusterer("my_data.tsv")
clusterer.load_data("canonical_smiles")
# ... continue processing
```

### Descriptors in Python:
```python
from src.database.descriptors import MolecularDescriptors
desc = MolecularDescriptors("my_data.tsv")
desc.compute_descriptors()
desc.plot_histograms()
```

---

## 🆘 **TROUBLESHOOTING**

### Error: "File not found"
- Verify that TSV files are in the correct location
- Use absolute paths if necessary

### Error: "Import not found"
- Run from the correct directory: `cd src/database/`
- Verify that the Python environment has the required dependencies (rdkit, pandas, etc.)

### Error: "RDKit not found"
- Install: `conda install -c conda-forge rdkit`
- Or: `pip install rdkit-pypi`

---

## 📞 **SUMMARY**

| Script | How to Use | Purpose |
|--------|-----------|-----------|
| `comparative_analysis.py` | `python3 comparative_analysis.py` | Statistical analysis |
| `remove_redundance.py` | `python3 remove_redundance.py` | Data cleaning |
| `cluster.py` | `from cluster import MoleculeClusterer` | Molecular clustering |
| `descriptors.py` | `from descriptors import MolecularDescriptors` | Descriptor calculation |

**🎯 TIP**: Start with `comparative_analysis.py` or `remove_redundance.py` to see the scripts in action!
