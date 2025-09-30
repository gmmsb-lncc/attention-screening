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
# Execute diretamente:
cd src/database/
python3 remove_redundance.py
```

**Pré-requisitos:**
- `../0_database/kinase_all_compounds_formatted.tsv`

**O que faz:**
- Remove sais e canoniza SMILES
- Remove duplicatas
- Calcula descritores moleculares
- Gera arquivos limpos: `nr_kinase_all_compounds_salt_free_ver3.tsv`

---

## 🔧 **SCRIPTS PARA IMPORTAÇÃO EM CÓDIGO**

### 3️⃣ **Clustering Molecular** - `cluster.py`

```python
# Importe em seu código:
from src.database.cluster import MoleculeClusterer

# Uso básico:
clusterer = MoleculeClusterer("caminho/para/arquivo.tsv")
clusterer.load_data("canonical_smiles")
clusterer.parallel_generate_fingerprints("canonical_smiles", batch_size=1000)
clusters = clusterer.cluster_by_similarity(threshold=0.8)

# Visualizações:
tsne_results = clusterer.calculate_tsne()
clusterer.plot_tsne(tsne_results, threshold=0.8)
```

**Métodos principais:**
- `load_data(smile_column)` - Carrega dados
- `parallel_generate_fingerprints()` - Gera fingerprints
- `cluster_by_similarity(threshold)` - Faz clustering
- `calculate_tsne()` - Redução dimensional
- `plot_tsne()` - Visualização t-SNE

---

### 4️⃣ **Descritores Moleculares** - `descriptors.py`

```python
# Importe em seu código:
from src.database.descriptors import MolecularDescriptors

# Uso básico:
descriptors = MolecularDescriptors("caminho/para/arquivo.tsv")
descriptors.compute_descriptors()
descriptors.save_descriptors("output_descriptors.tsv")

# Visualizações:
descriptors.plot_histograms(output_path="histograms.png")
descriptors.violin_plot()
```

**Métodos principais:**
- `calculate_descriptors(smiles_list)` - Calcula descritores
- `compute_descriptors()` - Processa todo o dataset
- `save_descriptors(path)` - Salva resultados
- `plot_histograms()` - Gera histogramas
- `violin_plot()` - Gráficos de violino

---

## 🗃️ **CONSULTAS SQL** - Pasta `sql/`

```bash
# Use as consultas SQL diretamente no ChEMBL:
cat sql/kinase_humans.sql | mysql -h chembl_host -u user -p chembl_35

# Ou copie as queries para seu cliente SQL favorito
```

**Arquivos disponíveis:**
- `kinase_humans.sql` - Extrai dados de quinases humanas
- `kinase_non_humans.sql` - Extrai dados de quinases não humanas
- `kinase_compounds.sql` - Consultas gerais de compostos
- `kinase_compounds_and_seq.sql` - Inclui sequências

---

## 🔍 **ESTRUTURA MODULAR AVANÇADA**

Para usuários avançados que querem usar os módulos internos:

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

## 📁 **ESTRUTURA DE ARQUIVOS ESPERADA**

```
sua_pasta_trabalho/
├── kinase_human_compounds.tsv          # Para comparative_analysis.py
├── kinase_non_human_compounds.tsv      # Para comparative_analysis.py
├── 0_database/
│   └── kinase_all_compounds_formatted.tsv  # Para remove_redundance.py
└── output/                             # Diretório de saída (criado automaticamente)
```

---

## ⚡ **EXEMPLOS RÁPIDOS**

### Análise Comparativa Rápida:
```bash
cd src/database/
python3 comparative_analysis.py
```

### Limpeza de Dados Rápida:
```bash
cd src/database/
python3 remove_redundance.py
```

### Clustering em Python:
```python
from src.database.cluster import MoleculeClusterer
clusterer = MoleculeClusterer("meus_dados.tsv")
clusterer.load_data("canonical_smiles")
# ... continue processamento
```

### Descritores em Python:
```python
from src.database.descriptors import MolecularDescriptors
desc = MolecularDescriptors("meus_dados.tsv")
desc.compute_descriptors()
desc.plot_histograms()
```

---

## 🆘 **TROUBLESHOOTING**

### Erro: "Arquivo não encontrado"
- Verifique se os arquivos TSV estão no local correto
- Use caminhos absolutos se necessário

### Erro: "Import não encontrado"
- Execute a partir do diretório correto: `cd src/database/`
- Verifique se o ambiente Python tem as dependências (rdkit, pandas, etc.)

### Erro: "RDKit não encontrado"
- Instale: `conda install -c conda-forge rdkit`
- Ou: `pip install rdkit-pypi`

---

## 📞 **RESUMO**

| Script | Como Usar | Propósito |
|--------|-----------|-----------|
| `comparative_analysis.py` | `python3 comparative_analysis.py` | Análise estatística |
| `remove_redundance.py` | `python3 remove_redundance.py` | Limpeza de dados |
| `cluster.py` | `from cluster import MoleculeClusterer` | Clustering molecular |
| `descriptors.py` | `from descriptors import MolecularDescriptors` | Cálculo descritores |

**🎯 DICA**: Comece com `comparative_analysis.py` ou `remove_redundance.py` para ver os scripts em ação!
