# 🚀 Guia de Instalação - DockTKinase

## 📋 **Sumário**
Este guia cobre a instalação de todas as dependências para os pipelines de **classificação** e **regressão**, incluindo geração de embeddings, estratificação, preparação de dados e modelos de machine learning.

---

## ⚡ **Instalação Rápida (Recomendada)**

```bash
# 1. Clonar repositório
git clone https://github.com/gmmsb-lncc/docktkinase.git
cd docktkinase

# 2. Criar ambiente virtual
python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# 3. Instalação automática
python setup.py

# 4. Testar instalação
python -m pytest tests/
```

**O script `setup.py` instala automaticamente**:
- ✅ PyTorch (com CUDA se disponível)
- ✅ ESM-2 (fair-esm, transformers)
- ✅ Dependências de classificação (scikit-learn, XGBoost)
- ✅ Dependências de regressão (joblib, seaborn)
- ✅ Molecular tools (RDKit, PyTorch Geometric)
- ✅ Big data (PySpark, PyArrow)

---

## 💻 **Instalação Manual no Mac M1/M2/M3**

### **Opção 1: Virtual Environment (venv) - Recomendada**
```bash
# 1. Criar e ativar ambiente virtual
python3 -m venv env
source env/bin/activate

# 2. Instalar dependências específicas Mac
pip install -r requirements-mac.txt

# 3. Testar instalação
python -m pytest tests/
```

### **Opção 2: Conda (se preferir)**
```bash
# 1. Criar ambiente conda
conda create -n docktkinase python=3.11
conda activate docktkinase

# 2. Instalar RDKit via conda (melhor compatibilidade no Mac)
conda install -c conda-forge rdkit

# 3. Instalar demais dependências
pip install -r requirements-mac.txt

# 4. Testar instalação
python -m pytest tests/
```

### **⚠️ Limitações no Mac M1/M2/M3:**
- ❌ **Sem GPU CUDA** - Processamento apenas CPU (mais lento)
- ⚠️ **PyTorch Geometric** - Compatibilidade limitada
- ✅ **RDKit** - Funciona melhor via conda
- ✅ **Todos os modelos funcionam** - Classificação e regressão

---

## 🚀 **Instalação no Linux com CUDA (GPU)**

### **Pré-requisitos de Sistema:**

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    build-essential \
    git \
    nvidia-driver-535  # ou versão mais recente
```

#### CentOS/RHEL:
```bash
sudo yum install -y \
    python3-devel \
    gcc \
    gcc-c++ \
    git
```

### **Verificar CUDA:**
```bash
nvidia-smi          # Deve mostrar sua GPU
nvcc --version      # Versão do CUDA (12.1+ recomendado)
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
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 5. Instalar PyTorch Geometric com CUDA
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 6. Instalar demais dependências
pip install -r requirements-cuda.txt

# 7. Testar instalação completa
python -m pytest tests/
```

---

## 📦 **Dependências por Módulo**

### **Core (Obrigatórias):**
- `torch>=2.0.0` - Deep learning framework
- `numpy>=1.26.0`, `pandas>=2.0.0` - Manipulação de dados
- `scikit-learn>=1.3.0` - Machine learning
- `matplotlib>=3.7.0`, `seaborn>=0.12.0` - Visualizações

### **Embeddings de Proteínas:**
- `fair-esm` - ESM-2 models (Facebook AI)
- `transformers>=4.38` - Hugging Face transformers
- `sentencepiece` - Tokenização

### **Embeddings de Ligantes:**
- `rdkit>=2024.3.5` - Química molecular
- Custom: FM4M/SMI-TED (incluído no repositório)

### **Classificação:**
- `scikit-learn>=1.3.0` - Algoritmos clássicos
- `xgboost>=1.6.2` - Gradient boosting
- `imbalanced-learn` - Balanceamento de classes

### **Regressão (Novo!):**
- `scikit-learn>=1.3.0` - 11 modelos de regressão
- `joblib>=1.3.0` - Persistência de modelos
- `seaborn>=0.12.0` - Visualizações avançadas

### **Utilities:**
- Módulo `src/utils/` - Funções DRY (safe_get, etc.)

### **Big Data Processing:**
- `pyspark>=3.0.0` - Processamento distribuído
- `pyarrow>=14.0.0` - Apache Arrow

### **Optimization:**
- `optuna` - Hyperparameter tuning (opcional)

---

## 🧪 **Testando a Instalação**

### **Teste Automático Completo:**
```bash
# Rodar todos os testes (19 testes)
python -m pytest tests/ -v

# Teste rápido (apenas smoke tests)
python -m pytest tests/ -k "test_imports"
```

### **Teste Manual de Imports:**
```python
# Testar módulos principais
python -c "
from src.build.core import BuildConfig
from src.build.pipeline import BuildPipeline
from src.build.embeddings import ProteinEmbedding, LigandEmbedding
from src.build.stratification import Stratifier, SplitValidator
from src.regression import RegressionTrainer, RegressionEvaluator
from src.utils import safe_get, safe_get_numeric
print('✅ Todos os módulos importados!')
"
```

### **Teste de GPU (Linux CUDA apenas):**
```python
import torch
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

## 🔧 **Problemas Comuns e Soluções**

### **1. Mac M1: "No module named 'rdkit'"**
```bash
# Solução: Instalar via conda
conda install -c conda-forge rdkit
```

### **2. Linux: "fatal error: Python.h: No such file or directory"**
```bash
# Solução: Instalar Python dev headers
sudo apt-get install python3.11-dev  # Ubuntu/Debian
sudo yum install python3-devel        # CentOS/RHEL
```

### **3. Linux CUDA: "CUDA out of memory"**
```python
# Solução: Reduzir batch_size
from src.regression import RegressionConfig
config = RegressionConfig(batch_size=16)  # Padrão: 32
```

### **4. "No module named 'torch_geometric'"**
```bash
# Mac: Não é crítico, pode pular
# Linux CUDA: Instalar com suporte CUDA
pip install torch-geometric torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### **5. "ImportError: No module named 'pyspark'"**
```bash
pip install pyspark>=3.0.0 pyarrow>=14.0.0
```

### **6. ESM downloads lentos**
```bash
# Modelos ESM são grandes (~3GB)
# Salvos em: ~/.cache/torch/hub/checkpoints/
# Primeiro download demora, depois é rápido (cache)
```

### **7. Mac: "Library not loaded: libomp.dylib"**
```bash
# Solução: Instalar OpenMP via Homebrew
brew install libomp
```

---

## 📊 **Verificar Dependências Instaladas**

```bash
# Listar todas
pip list

# Verificar versões específicas
pip show torch transformers fair-esm scikit-learn xgboost seaborn

# Verificar imports Python
python -c "
import torch, esm, transformers, sklearn
import numpy, pandas, matplotlib, seaborn
from src.regression import RegressionTrainer
from src.utils import safe_get
print('✅ Principais bibliotecas OK!')
"
```

---

## 🎯 **Próximos Passos Após Instalação**

1. ✅ **Ler documentação**
   - [QUICK_START.md](QUICK_START.md) - Início rápido
   - [USER_GUIDE.md](USER_GUIDE.md) - Manual completo
   - [../src/regression/README.md](../src/regression/README.md) - Módulo de regressão

2. ✅ **Testar com dataset pequeno**
   ```bash
   python scripts/run_complete_pipeline.py --test-mode
   ```

3. ✅ **Executar pipeline completo**
   ```bash
   python scripts/run_complete_pipeline.py      # Classificação
   python run_regression_pipeline.py    # Regressão
   ```

---

## 📚 **Recursos Adicionais**

- **ESM-2**: https://github.com/facebookresearch/esm
- **PyTorch**: https://pytorch.org/get-started/locally/
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **RDKit**: https://www.rdkit.org/docs/
- **scikit-learn**: https://scikit-learn.org/stable/
- **Documentação Build**: `src/build/README.md`
- **Documentação Regression**: `src/regression/README.md`

---

## ✅ **Checklist de Instalação**

- [ ] Repositório clonado
- [ ] Ambiente virtual criado (`venv` ou `conda`)
- [ ] Ambiente ativado
- [ ] PyTorch instalado (com CUDA se Linux)
- [ ] `python setup.py` executado OU `requirements*.txt` instalado
- [ ] ESM e transformers instalados
- [ ] scikit-learn e dependências ML instaladas
- [ ] RDKit instalado (conda no Mac)
- [ ] Testes passando: `python -m pytest tests/`
- [ ] GPU reconhecida (se Linux CUDA)
- [ ] Módulos importam sem erros

---

**🎉 Se todos os testes passaram, você está pronto para usar os pipelines de classificação E regressão!**

**Última atualização**: 28 de outubro de 2025 | **Branch**: regression
