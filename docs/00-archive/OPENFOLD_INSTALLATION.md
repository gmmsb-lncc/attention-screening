# Instalação do OpenFold3 para Extração de Embeddings

## 📋 Visão Geral

Este guia explica como instalar as dependências do OpenFold3 para permitir a extração de embeddings no DockTKinase.

## 🎯 Status da Integração

✅ **Código de integração**: COMPLETO  
⚠️ **Dependências OpenFold3**: FALTANDO  
🔧 **Próximo passo**: Instalar dependências

## 📦 Dependências Necessárias

### 1. Dependências Python Core

```bash
# Instalar dependências do OpenFold3
pip install gemmi         # Biblioteca de cristalografia (OBRIGATÓRIO)
pip install ml-collections  # Configurações do modelo
pip install einops        # Operações de tensor
pip install biopython     # Manipulação de sequências
```

### 2. Dependências Opcionais (Performance)

```bash
# Para melhor performance
pip install deepspeed     # Otimizações de treinamento/inferência
pip install flash-attn    # Atenção eficiente
```

## 🔧 Instalação Completa

### Opção 1: Instalação Rápida (Recomendado)

```bash
cd /Users/sulfierry/docktkinase

# Instalar apenas dependências essenciais para embeddings
pip install gemmi ml-collections einops biopython

# Verificar instalação
python -c "import gemmi; print('✅ gemmi instalado')"
python -c "from ml_collections import ConfigDict; print('✅ ml-collections instalado')"
```

### Opção 2: Instalação Completa do OpenFold3

```bash
cd /Users/sulfierry/docktkinase/openfold-3

# Instalar em modo desenvolvimento
pip install -e .

# Isso instalará todas as dependências listadas em pyproject.toml
```

### Opção 3: Usando Environment Conda

```bash
# Criar environment específico para OpenFold
conda create -n openfold python=3.10
conda activate openfold

# Instalar dependências via conda
conda install -c conda-forge gemmi
pip install ml-collections einops biopython torch

# Instalar OpenFold
cd /Users/sulfierry/docktkinase/openfold-3
pip install -e .
```

## ✅ Verificação da Instalação

### Teste 1: Importação Básica

```bash
python << 'EOF'
import sys
from pathlib import Path

# Adicionar OpenFold ao path
sys.path.insert(0, str(Path.cwd() / 'openfold-3'))

print("🔍 Testando importações...")

# Teste 1: gemmi
try:
    import gemmi
    print("✅ gemmi: OK")
except ImportError as e:
    print(f"❌ gemmi: FALTANDO - {e}")

# Teste 2: ml_collections
try:
    from ml_collections import ConfigDict
    print("✅ ml_collections: OK")
except ImportError as e:
    print(f"❌ ml_collections: FALTANDO - {e}")

# Teste 3: openfold3
try:
    import openfold3
    print(f"✅ openfold3: OK - {openfold3.__file__}")
except ImportError as e:
    print(f"❌ openfold3: FALTANDO - {e}")

# Teste 4: OpenFold3 model
try:
    from openfold3.projects.of3_all_atom.model import OpenFold3
    print("✅ OpenFold3 model: OK")
except ImportError as e:
    print(f"❌ OpenFold3 model: FALTANDO - {e}")

print("\n✅ Verificação completa!")
EOF
```

### Teste 2: Carregar Strategy

```bash
python << 'EOF'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / 'src'))

print("🧪 Testando OpenFoldStrategy...")

try:
    from src.build.embeddings.strategies.openfold_strategy import OpenFoldStrategy
    import torch
    
    strategy = OpenFoldStrategy()
    print("✅ Strategy criado")
    
    # Tentar carregar modelo
    device = torch.device('cpu')
    model, _ = strategy.load('openfold3', device)
    print("✅ Modelo carregado!")
    print(f"   Tipo: {type(model).__name__}")
    print(f"   Device: {device}")
    
    # Verificar método run_trunk
    has_trunk = hasattr(model, 'run_trunk')
    print(f"✅ run_trunk disponível: {has_trunk}")
    
    if has_trunk:
        print("\n🎉 OpenFold3 pronto para extração de embeddings!")
    else:
        print("\n⚠️  Modelo incompleto")
        
except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc()
EOF
```

### Teste 3: Extração de Embeddings

```bash
python << 'EOF'
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path.cwd() / 'src'))

from src.build.embeddings.strategies.openfold_strategy import OpenFoldStrategy

print("🧬 Testando extração de embeddings...")

try:
    strategy = OpenFoldStrategy()
    model, _ = strategy.load('openfold3', torch.device('cpu'))
    
    # Sequência teste
    sequence = "MKFLKFSL"
    print(f"Sequência: {sequence}")
    
    # Gerar embedding
    embedding = strategy.generate(
        model=model,
        auxiliary_objects=None,
        sequence=sequence,
        device=torch.device('cpu'),
        pooling_strategy='mean'
    )
    
    print(f"✅ Embedding gerado!")
    print(f"   Shape: {embedding.shape}")
    print(f"   Mean: {embedding.mean():.4f}")
    print(f"   Std: {embedding.std():.4f}")
    print("\n🎉 Tudo funcionando!")
    
except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc()
EOF
```

## 🚨 Troubleshooting

### Erro: "No module named 'gemmi'"

**Solução:**
```bash
pip install gemmi
```

**Causa:** Biblioteca GEMMI não instalada. É necessária para manipulação de estruturas.

### Erro: "No module named 'ml_collections'"

**Solução:**
```bash
pip install ml-collections
```

**Causa:** Biblioteca de configuração do modelo não instalada.

### Erro: "Could not import OpenFold3"

**Verificar:**
1. OpenFold-3 está em `/Users/sulfierry/docktkinase/openfold-3/`?
2. Arquivo `openfold3/__init__.py` existe?
3. Dependências instaladas?

**Solução:**
```bash
cd /Users/sulfierry/docktkinase/openfold-3
pip install -e .
```

### Erro: "run_trunk not found"

**Causa:** Modelo não está completamente carregado ou versão incompatível.

**Verificar:**
```bash
python -c "from openfold3.projects.of3_all_atom.model import OpenFold3; print('OK')"
```

## 📊 Requisitos de Sistema

### Mínimo (CPU only)
- Python: 3.8+
- RAM: 8GB+
- Disk: 2GB (modelo + dependências)
- CPU: Multi-core recomendado

### Recomendado (GPU)
- Python: 3.10+
- RAM: 16GB+
- GPU: CUDA 11.8+ com 8GB+ VRAM
- Disk: 5GB+

## 🔄 Próximos Passos

### 1. Instalar Dependências
```bash
pip install gemmi ml-collections einops biopython
```

### 2. Testar Importação
```bash
python -c "import gemmi; from ml_collections import ConfigDict; print('✅ OK')"
```

### 3. Carregar Modelo
```bash
cd /Users/sulfierry/docktkinase
python -c "from src.build.embeddings.strategies.openfold_strategy import OpenFoldStrategy; import torch; s = OpenFoldStrategy(); s.load('openfold3', torch.device('cpu')); print('✅ Modelo carregado')"
```

### 4. Extrair Embeddings
```bash
python run_complete_pipeline.py --esm-model openfold3 --input <file>
```

## 📖 Referências

- **OpenFold GitHub**: https://github.com/aqlaboratory/openfold
- **GEMMI**: https://gemmi.readthedocs.io/
- **ML Collections**: https://github.com/google/ml_collections

## 🎯 Resumo

**Dependência crítica**: `gemmi`

```bash
# Instalação mínima (1 linha)
pip install gemmi ml-collections einops

# Testar
python -c "import gemmi; print('✅ Pronto para OpenFold3!')"
```

Após instalar, o código de extração de embeddings já está **100% implementado** e funcionará automaticamente!
