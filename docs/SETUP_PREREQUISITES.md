# Pré-requisitos para Setup do DockTKinase

**Data**: 28 de Outubro de 2025  
**Branch**: regression  
**Sistema**: Dual Pipeline (Classification + Regression)

## ⚠️ REQUISITOS OBRIGATÓRIOS

Antes de executar `python setup.py`, você **DEVE** instalar os seguintes pacotes do sistema:

### 1. Python Development Headers (CRÍTICO)

**Por que é necessário?**
- Extensões PyG (torch-scatter, torch-sparse, torch-cluster) precisam compilar código C++
- Estas extensões incluem `<Python.h>` no código fonte
- Sem os headers, a compilação falhará com: `fatal error: Python.h: No such file or directory`

**Como instalar:**

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install python3.11-dev -y
# OU para Python 3.12
sudo apt-get install python3.12-dev -y
```

#### CentOS/RHEL:
```bash
sudo yum install python3-devel -y
```

#### Fedora:
```bash
sudo dnf install python3-devel -y
```

#### macOS (Homebrew):
```bash
# Python headers incluídos com Python do Homebrew
brew install python@3.11
```

#### Verificar instalação:
```bash
# Verificar se Python.h existe
ls /usr/include/python3.11/Python.h  # Linux
ls /opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Headers/Python.h  # macOS

# Ou usar python3-config
python3-config --includes
```

### 2. Compilador C++ (Opcional mas Recomendado)

```bash
# Ubuntu/Debian
sudo apt-get install build-essential -y

# CentOS/RHEL
sudo yum groupinstall "Development Tools" -y

# Fedora
sudo dnf groupinstall "Development Tools" -y

# macOS (Homebrew)
xcode-select --install
```

### 3. Git (para clonar o repositório)

```bash
# Ubuntu/Debian
sudo apt-get install git -y

# CentOS/RHEL
sudo yum install git -y

# Fedora
sudo dnf install git -y

# macOS (Homebrew)
brew install git
```

---

## 🚀 Instalação Completa (Recomendado)

Execute **ANTES** de rodar `python setup.py`:

### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    build-essential \
    git
```

### CentOS/RHEL:
```bash
sudo yum update -y
sudo yum install -y \
    python3 \
    python3-devel \
    gcc \
    gcc-c++ \
    make \
    git
```

### Fedora:
```bash
sudo dnf update -y
sudo dnf install -y \
    python3 \
    python3-devel \
    gcc \
    gcc-c++ \
    make \
    git
```

### macOS (Apple Silicon):
```bash
# Instalar Homebrew (se não tiver)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalar dependências
brew install python@3.11 git

# Opcional: conda para RDKit
brew install --cask miniconda
conda create -n docktkinase python=3.11
conda activate docktkinase
conda install -c conda-forge rdkit
```

---

## 📋 Checklist de Pré-requisitos

Antes de executar `python setup.py`, verifique:

- [ ] Python 3.8+ instalado
- [ ] **Python development headers instalados** (`python3.11-dev` ou `python3.12-dev`)
- [ ] Compilador C++ instalado (`build-essential` ou `gcc`)
- [ ] Git instalado
- [ ] Permissões para criar ambiente virtual
- [ ] Espaço em disco suficiente (~5-8GB para dependências + models)
- [ ] **Dual pipeline support** (Classification + Regression) ⭐
- [ ] Conexão de internet (para download de modelos)

---

## ⚙️ Executando o Setup

Após instalar os pré-requisitos:

```bash
# Clonar repositório
git clone https://github.com/gmmsb-lncc/docktkinase.git
cd docktkinase

# Executar setup automatizado
python setup.py

# Ativar ambiente
source env/bin/activate  # Linux/macOS
# OU
env\Scripts\activate  # Windows
```

O script verificará automaticamente se todos os pré-requisitos estão instalados.

---

## 🧪 Testar Instalação

Após setup completo:

```bash
# Ativar ambiente
source env/bin/activate

# Testar imports
python -c "
from src.build.core import BuildConfig
from src.classifier.core import DataManager
from src.regression import RegressionConfig  # NOVO!
from src.utils.data_utils import load_data  # NOVO!
print('✅ Dual pipeline system ready!')
"

# Executar testes automatizados (19 testes)
pytest tests/ -v

# Testar classification pipeline
python run_complete_pipeline.py --help

# Testar regression pipeline (NOVO!)
python run_regression_pipeline.py --help
```

---

## ❓ Troubleshooting

### Erro: "Python.h: No such file or directory"

**Causa:** Python development headers não instalados

**Solução:**
```bash
# Ubuntu/Debian
sudo apt-get install python3.11-dev -y

# macOS (Homebrew)
brew install python@3.11

# Reexecutar setup
python setup.py
```

### Erro: "ModuleNotFoundError: No module named 'torch'"

**Causa:** Usando `--no-build-isolation` sem torch instalado

**Solução:** O script já instala torch antes. Se o erro persistir:
```bash
source env/bin/activate
pip install torch
pip install --no-build-isolation torch-scatter torch-sparse torch-cluster
```

### Extensões PyG falharam mas setup continuou

**Isso é normal!** As extensões PyG são **opcionais**. O sistema funcionará normalmente sem elas.

Para instalar manualmente depois:
```bash
# Instalar headers primeiro
sudo apt-get install python3.11-dev -y  # Linux
brew install python@3.11  # macOS

# Ativar ambiente
source env/bin/activate

# Instalar extensões
pip install --no-build-isolation torch-scatter
pip install --no-build-isolation torch-sparse
pip install --no-build-isolation torch-cluster
```

### macOS Apple Silicon: RDKit Installation

**Recomendação:** Use conda para RDKit no macOS:

```bash
# Criar ambiente conda
conda create -n docktkinase python=3.11
conda activate docktkinase

# Instalar RDKit via conda
conda install -c conda-forge rdkit

# Instalar restante via pip
pip install -r requirements-mac.txt
```

### Erro: "xgboost installation failed" (Regression)

**Causa:** xgboost requer compilação em alguns sistemas

**Solução:**
```bash
# Ubuntu/Debian
sudo apt-get install cmake -y
pip install xgboost

# macOS
brew install cmake
pip install xgboost

# Alternativa: usar conda
conda install -c conda-forge xgboost
```

---

## 📚 Mais Informações

- [INSTALLATION_GUIDE.md](./INSTALLATION_GUIDE.md) - Guia detalhado de instalação
- [QUICK_START.md](./QUICK_START.md) - Início rápido
- [USER_GUIDE.md](./USER_GUIDE.md) - Guia do usuário completo
- [SETUP_SUMMARY.md](./SETUP_SUMMARY.md) - Resumo das dependências

---

## ✅ Resumo - Comandos Únicos

**Ubuntu/Debian:**
```bash
sudo apt-get update && \
sudo apt-get install -y python3.11 python3.11-dev python3.11-venv build-essential git && \
git clone https://github.com/gmmsb-lncc/docktkinase.git && \
cd docktkinase && \
python3 setup.py
```

**CentOS/RHEL:**
```bash
sudo yum update -y && \
sudo yum install -y python3 python3-devel gcc gcc-c++ make git && \
git clone https://github.com/gmmsb-lncc/docktkinase.git && \
cd docktkinase && \
python3 setup.py
```

**macOS (Apple Silicon):**
```bash
brew install python@3.11 git && \
git clone https://github.com/gmmsb-lncc/docktkinase.git && \
cd docktkinase && \
python3 setup.py
```

---

## 🎯 Sistema Dual Pipeline

Após instalação completa, você terá acesso a:

### Classification Pipeline (6 modelos):
```bash
python run_complete_pipeline.py \
    --dataset data/test_dataset_1000.tsv \
    --output-dir results/classification
```

### Regression Pipeline (11 modelos): ⭐ **NOVO!**
```bash
python run_regression_pipeline.py \
    --dataset data/test_dataset_1000.tsv \
    --activity-type ki \
    --models linear_regression ridge xgboost \
    --output-dir results/regression
```

---

**Última atualização**: 28 de Outubro de 2025  
**Branch**: regression  
**Sistema**: Dual Pipeline (Classification + Regression)  
**Modelos ML**: 17 total (6 classifiers + 11 regressors)  
**Testes**: 19 automatizados (100% passing)
