# 🚀 Guia de Instalação - DockTKinase Build Pipeline

## 📋 **Sumário**
Este guia cobre a instalação de todas as dependências necessárias para rodar o pipeline completo de build, incluindo geração de embeddings, estratificação e preparação de dados.

---

## 💻 **Instalação no Mac M1/M2/M3 (Desenvolvimento)**

### **Opção 1: Instalação Rápida**
```bash
# 1. Criar e ativar ambiente virtual
python3 -m venv env
source env/bin/activate

# 2. Instalar dependências
pip install -r requirements-mac.txt

# 3. Testar instalação
python test_pipeline_setup.py
```

### **Opção 2: Instalação Completa (com conda para RDKit)**
```bash
# 1. Criar ambiente conda
conda create -n docktkinase python=3.11
conda activate docktkinase

# 2. Instalar RDKit via conda (recomendado para Mac)
conda install -c conda-forge rdkit

# 3. Instalar demais dependências
pip install -r requirements-mac.txt

# 4. Testar instalação
python test_pipeline_setup.py
```

### **Limitações no Mac M1:**
- ⚠️ **Sem GPU CUDA** - Embeddings serão processados na CPU (mais lento)
- ⚠️ **PyTorch Geometric** pode ter problemas de compatibilidade
- ✅ **RDKit** funciona melhor via conda
- ✅ **PySpark** funciona, mas pode ser lento para datasets grandes

---

## 🚀 **Instalação no Linux com CUDA (Produção - RTX 4090)**

### **Pré-requisitos:**
1. NVIDIA Driver instalado
2. CUDA Toolkit 12.1+ instalado
3. Python 3.8+

### **Verificar CUDA:**
```bash
nvidia-smi  # Deve mostrar sua GPU
nvcc --version  # Versão do CUDA
```

### **Instalação Passo a Passo:**

```bash
# 1. Criar e ativar ambiente virtual
python3 -m venv env
source env/bin/activate

# 2. Atualizar pip
pip install --upgrade pip

# 3. Instalar PyTorch com CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Testar CUDA
python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 5. Instalar PyTorch Geometric com CUDA
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 6. Instalar demais dependências
pip install -r requirements-cuda.txt

# 7. Testar instalação completa
python test_pipeline_setup.py
```

---

## 📦 **Dependências por Módulo**

### **Core (Obrigatórias):**
- `torch` - Deep learning framework
- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities
- `matplotlib`, `seaborn` - Visualizações

### **Embeddings:**
- `fair-esm` - Protein embeddings (ESM models)
- `transformers` - Hugging Face transformers (para ESM)
- `rdkit` - Molecular chemistry
- `umap-learn` - Dimensionality reduction

### **Stratification (Novo):**
- `scikit-learn` - Clustering algorithms
- `scipy` - Scientific computing

### **Big Data Processing:**
- `pyspark` - Distributed processing
- `pyarrow` - Apache Arrow

### **Optimization:**
- `optuna` - Hyperparameter tuning

---

## 🧪 **Testando a Instalação**

### **Teste Básico:**
```bash
python test_pipeline_setup.py
```

Este script verifica:
- ✅ PyTorch e CUDA (se disponível)
- ✅ Transformers e ESM
- ✅ scikit-learn e scipy
- ✅ Módulos do build (core, pipeline, embeddings, stratification)
- ✅ Arquivos de dados (kinase_all, kinase_humans, kinase_non_humans)

### **Teste Individual de Módulos:**
```python
# Testar imports
python -c "
from build.core import BuildConfig
from build.pipeline import BuildPipeline
from build.embeddings import ProteinEmbedding, LigandEmbedding
from build.stratification import Stratifier, SplitValidator
print('✅ Todos os módulos importados com sucesso!')
"
```

### **Teste de GPU (Linux apenas):**
```python
import torch
print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Versão CUDA: {torch.version.cuda}")
```

---

## 🔧 **Problemas Comuns e Soluções**

### **Mac M1: "No module named 'rdkit'"**
```bash
# Solução: Instalar via conda
conda install -c conda-forge rdkit
```

### **Linux: "CUDA out of memory"**
```python
# Solução: Reduzir batch_size no config
config = BuildConfig({
    'batch_size': 16,  # Reduzir de 32 para 16
})
```

### **"No module named 'torch_geometric'"**
```bash
# Mac: Pode não ter suporte completo, não é crítico
# Linux: Instalar com CUDA support
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### **"ImportError: No module named 'pyspark'"**
```bash
pip install pyspark>=3.0.0
```

### **ESM downloads são lentos**
```python
# Os modelos ESM são grandes (~3GB)
# Primeiro download pode demorar
# Modelos são salvos em: ~/.cache/torch/hub/checkpoints/
```

---

## 📊 **Verificar Dependências Instaladas**

```bash
# Listar todas as dependências
pip list

# Verificar versões específicas
pip show torch transformers fair-esm scikit-learn

# Verificar imports Python
python -c "
import torch
import esm
import transformers
import sklearn
import numpy as np
import pandas as pd
print('✅ Principais bibliotecas instaladas!')
"
```

---

## 🎯 **Próximos Passos Após Instalação**

1. ✅ **Testar pipeline com dataset pequeno**
   ```bash
   python scripts/run_build_pipeline.py --input src/kinase_humans/kinase_human_compounds.tsv
   ```

2. ✅ **Verificar estratificação**
   ```bash
   python scripts/test_stratification.py
   ```

3. ✅ **Executar pipeline completo**
   ```bash
   python scripts/run_complete_build.py
   ```

---

## 📚 **Recursos Adicionais**

- **ESM Documentation**: https://github.com/facebookresearch/esm
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **RDKit Documentation**: https://www.rdkit.org/docs/
- **DockTKinase Build README**: `src/build/README.md`

---

## ✅ **Checklist de Instalação**

- [ ] Ambiente virtual criado e ativado
- [ ] PyTorch instalado (com CUDA se Linux)
- [ ] Transformers e ESM instalados
- [ ] scikit-learn e scipy instalados
- [ ] RDKit instalado (via conda no Mac)
- [ ] Teste `test_pipeline_setup.py` passou
- [ ] GPU reconhecida (se Linux com CUDA)
- [ ] Módulos do build importam sem erros

---

**🎉 Se todos os testes passaram, você está pronto para usar o pipeline!**
