# Pré-requisitos para Setup do DockTKinase

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

#### Verificar instalação:
```bash
# Verificar se Python.h existe
ls /usr/include/python3.12/Python.h

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
```

### 3. Git (para clonar o repositório)

```bash
# Ubuntu/Debian
sudo apt-get install git -y

# CentOS/RHEL
sudo yum install git -y

# Fedora
sudo dnf install git -y
```

---

## 🚀 Instalação Completa (Recomendado)

Execute **ANTES** de rodar `python setup.py`:

### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
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

---

## 📋 Checklist de Pré-requisitos

Antes de executar `python setup.py`, verifique:

- [ ] Python 3.8+ instalado
- [ ] **Python development headers instalados** (`python3.12-dev`)
- [ ] Compilador C++ instalado (`build-essential` ou `gcc`)
- [ ] Git instalado
- [ ] Permissões para criar ambiente virtual
- [ ] Espaço em disco suficiente (~5GB para dependências)

---

## ⚙️ Executando o Setup

Após instalar os pré-requisitos:

```bash
# Clonar repositório
git clone https://github.com/gmmsb-lncc/docktkinase.git
cd docktkinase

# Executar setup
python3 setup.py
```

O script verificará automaticamente se todos os pré-requisitos estão instalados.

---

## ❓ Troubleshooting

### Erro: "Python.h: No such file or directory"

**Causa:** Python development headers não instalados

**Solução:**
```bash
sudo apt-get install python3.12-dev -y
python3 setup.py  # Execute novamente
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
sudo apt-get install python3.12-dev -y

# Ativar ambiente
source env/bin/activate

# Instalar extensões
pip install --no-build-isolation torch-scatter
pip install --no-build-isolation torch-sparse
pip install --no-build-isolation torch-cluster
```

---

## 📚 Mais Informações

- [INSTALLATION_GUIDE.md](./INSTALLATION_GUIDE.md) - Guia detalhado de instalação
- [QUICK_START.md](./QUICK_START.md) - Início rápido
- [USER_GUIDE.md](./USER_GUIDE.md) - Guia do usuário

---

## ✅ Resumo

**Comando único para Ubuntu/Debian:**
```bash
sudo apt-get update && \
sudo apt-get install -y python3.12 python3.12-dev python3.12-venv build-essential git && \
python3 setup.py
```

**Comando único para CentOS/RHEL:**
```bash
sudo yum update -y && \
sudo yum install -y python3 python3-devel gcc gcc-c++ make git && \
python3 setup.py
```

**Comando único para Fedora:**
```bash
sudo dnf update -y && \
sudo dnf install -y python3 python3-devel gcc gcc-c++ make git && \
python3 setup.py
```
